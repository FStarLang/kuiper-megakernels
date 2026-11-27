#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>
#include <cstdint>
#include <vector>

__device__ float contiguous_sum(const float* values, int size, int width) {
    float partial[256];
    int count = size / width;
    for (int i = 0; i < count; ++i) {
        float value = 0.0f;
        for (int j = 0; j < width; ++j)
            value = __fadd_rn(value, values[i * width + j]);
        partial[i] = value;
    }
    for (int offset = count / 2; offset; offset /= 2)
        for (int i = 0; i < offset; ++i)
            partial[i] = __fadd_rn(partial[i], partial[i + offset]);
    return partial[0];
}

__device__ float recurrent_sum(const float* values) {
    float partial[4];
    for (int y = 0; y < 4; ++y) {
        float a[4] = {};
        for (int i = y; i < 128; i += 16)
            for (int j = 0; j < 4; ++j)
                a[j] = __fadd_rn(a[j], values[i + 4 * j]);
        partial[y] = __fadd_rn(__fadd_rn(__fadd_rn(a[0], a[1]), a[2]), a[3]);
    }
    return __fadd_rn(__fadd_rn(partial[0], partial[2]), __fadd_rn(partial[1], partial[3]));
}

__device__ float sigmoid_value(float x) {
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x)));
}

__device__ float silu_value(float x) {
    return __fdiv_rn(x, __fadd_rn(1.0f, expf(-x)));
}


void check_tensor(const torch::Tensor& value, at::ScalarType dtype, int64_t count,
                  const torch::Tensor& device) {
    TORCH_CHECK(value.is_cuda() && value.is_contiguous() && value.scalar_type() == dtype
                && value.device() == device.device() && value.numel() == count,
                "Expected a contiguous CUDA tensor with the declared dtype, size and device");
}

__global__ void final_norm_projection(const at::BFloat16* weights,
                                      const at::BFloat16* norm_weights,
                                      const at::BFloat16* old_residual,
                                      at::BFloat16* hidden, at::BFloat16* logits) {
    int lane = threadIdx.x;
    __shared__ at::BFloat16 input[3584], projected[6144];
    __shared__ __nv_bfloat16 ma[256], mb[256];
    __shared__ float mc[256];
    at::BFloat16 residual[1024], normalized[1024], qkv[6144], z[2048], beta[16], gate[3584];
    at::BFloat16 core[2048], cosine[32], sine[32];
    float values[1024], products[128], updated[128], combined[16], scores[12];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> af;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> bf;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cf;

    if (lane == 0)
        for (int i = 0; i < 1024; ++i) residual[i] = old_residual[i];
    if (lane == 0) {
        for (int i = 0; i < 1024; ++i) {
            values[i] = __fmul_rn(float(residual[i]), float(residual[i]));
            hidden[i] = residual[i];
        }
        float inverse = rsqrtf(__fadd_rn(__fmul_rn(contiguous_sum(values, 1024, 4), 1.0f / 1024), 1e-6f));
        for (int i = 0; i < 1024; ++i) {
            normalized[i] = at::BFloat16(__fmul_rn(__fmul_rn(float(residual[i]), inverse), __fadd_rn(1.0f, float(norm_weights[i]))));
            input[i] = normalized[i];
        }
    }
    __syncwarp();
    int n = 248320, k = 1024;
    int64_t offset = 0;
    int chunk = k == 3584 ? 1792 : k;
    for (int row = 0; row < n; row += 16) {
        for (int start = 0; start < k; start += chunk) {
            nvcuda::wmma::fill_fragment(cf, 0.0f);
            for (int p = start; p < start + chunk; p += 16) {
                for (int i = lane; i < 256; i += 32) {
                    ma[i] = __ushort_as_bfloat16(weights[offset + int64_t(row + i / 16) * k + p + i % 16].x);
                    mb[i] = __ushort_as_bfloat16(input[p + i % 16].x);
                }
                __syncwarp();
                nvcuda::wmma::load_matrix_sync(af, ma, 16);
                nvcuda::wmma::load_matrix_sync(bf, mb, 16);
                nvcuda::wmma::mma_sync(cf, af, bf, cf);
            }
            nvcuda::wmma::store_matrix_sync(mc, cf, 16, nvcuda::wmma::mem_row_major);
            __syncwarp();
            if (lane == 0)
                for (int i = 0; i < 16; ++i)
                    combined[i] = start == 0 ? mc[i * 16] : __fadd_rn(combined[i], mc[i * 16]);
        }
        if (lane == 0)
            for (int i = 0; i < 16; ++i) {
                logits[row + i] = at::BFloat16(combined[i]);
            }
    }
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                  torch::Tensor input2) {
    check_tensor(input0, at::kBFloat16, 254279680, input0);
    check_tensor(input1, at::kBFloat16, 1024, input0);
    check_tensor(input2, at::kBFloat16, 1024, input0);
    auto hidden = torch::empty({1024}, input0.options());
    auto logits = torch::empty({248320}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    final_norm_projection<<<1, 32, 0, stream>>>(
        input0.data_ptr<at::BFloat16>(), input1.data_ptr<at::BFloat16>(),
        input2.data_ptr<at::BFloat16>(), hidden.data_ptr<at::BFloat16>(),
        logits.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {hidden, logits};
}
