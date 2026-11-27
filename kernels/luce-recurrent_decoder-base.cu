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

__global__ void recurrent_decoder(const at::BFloat16* weights, const at::BFloat16* old_residual,
                                  const at::BFloat16* old_cache, const float* old_recurrent,
                                  at::BFloat16* next_residual, at::BFloat16* new_cache,
                                  float* new_recurrent) {
    int lane = threadIdx.x;
    int64_t base = 0;
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
    int output_projection = 4;
    int projections = output_projection + 4;
    int64_t attention = base + 11012096;
    if (lane == 0) {
        int64_t norm = base;
        for (int i = 0; i < 1024; ++i) {
            values[i] = __fmul_rn(float(residual[i]), float(residual[i]));
        }
        float inverse = rsqrtf(__fadd_rn(__fmul_rn(contiguous_sum(values, 1024, 4), 1.0f / 1024), 1e-6f));
        for (int i = 0; i < 1024; ++i) {
            normalized[i] = at::BFloat16(__fmul_rn(__fmul_rn(float(residual[i]), inverse), __fadd_rn(1.0f, float(weights[norm + i]))));
            input[i] = normalized[i];
        }
    }
    __syncwarp();
    for (int projection = 0; projection < projections; ++projection) {
        int n = 0, k = 1024;
        int64_t offset = 0;
        if (projection == output_projection) {
            n = 1024; k = 2048;
            offset = attention + 8446112;
        } else if (projection > output_projection) {
            int p = projection - output_projection;
            n = p == 3 ? 1024 : 3584;
            k = p == 3 ? 3584 : 1024;
            offset = base + 2048 + (p - 1) * 3670016;
        } else {
            n = projection == 0 ? 6144 : (projection == 1 ? 2048 : 16);
            offset = attention + (projection == 0 ? 0 : (projection == 1 ? 6291456 : (projection == 2 ? 8388608 : 8404992)));
        }
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
                    projected[row + i] = at::BFloat16(combined[i]);
                }
        }

        if (lane == 0) {
            if (projection < output_projection) {
                {
                    if (projection == 0) for (int i = 0; i < 6144; ++i) qkv[i] = projected[i];
                    if (projection == 1) for (int i = 0; i < 2048; ++i) z[i] = projected[i];
                    if (projection == 2) for (int i = 0; i < 16; ++i) beta[i] = at::BFloat16(sigmoid_value(float(projected[i])));
                    if (projection == 3) {
                        int index = 0;
                        int conv = index * 24576;
                        for (int i = 0; i < 6144; ++i) {
                            float value = 0.0f;
                            for (int j = 0; j < 4; ++j) {
                                at::BFloat16 x = j == 3 ? qkv[i] : old_cache[conv + i * 4 + j + 1];
                                new_cache[conv + i * 4 + j] = x;
                                value = __fmaf_rn(float(x), float(weights[attention + 8421376 + i * 4 + j]), value);
                            }
                            qkv[i] = at::BFloat16(silu_value(float(at::BFloat16(value))));
                        }
                        for (int head = 0; head < 16; ++head) {
                            for (int part = 0; part < 2; ++part) {
                                int start = part * 2048 + head * 128;
                                for (int i = 0; i < 128; ++i) values[i] = float(at::BFloat16(__fmul_rn(float(qkv[start + i]), float(qkv[start + i]))));
                                float sum = float(at::BFloat16(contiguous_sum(values, 128, 8)));
                                float inverse = float(at::BFloat16(rsqrtf(float(at::BFloat16(__fadd_rn(sum, 1e-6f))))));
                                for (int i = 0; i < 128; ++i) qkv[start + i] = at::BFloat16(__fmul_rn(float(qkv[start + i]), inverse));
                            }
                            float a = __fadd_rn(float(projected[head]), float(weights[attention + 8445968 + head]));
                            float softplus = a > 20.0f ? a : log1pf(expf(a));
                            float decay = expf(__fmul_rn(-expf(float(weights[attention + 8445952 + head])), softplus));
                            int state = (index * 16 + head) * 16384;
                            for (int j = 0; j < 128; ++j) {
                                for (int i = 0; i < 128; ++i) {
                                    updated[i] = __fmul_rn(old_recurrent[state + i * 128 + j], decay);
                                    products[i] = __fmul_rn(updated[i], float(qkv[2048 + head * 128 + i]));
                                }
                                float delta = __fmul_rn(__fsub_rn(float(qkv[4096 + head * 128 + j]), recurrent_sum(products)), float(beta[head]));
                                for (int i = 0; i < 128; ++i) {
                                    float next = __fadd_rn(updated[i], __fmul_rn(float(qkv[2048 + head * 128 + i]), delta));
                                    new_recurrent[state + i * 128 + j] = next;
                                    float query = __fmul_rn(float(qkv[head * 128 + i]), 0.08838834764831845f);
                                    products[i] = __fmul_rn(next, query);
                                }
                                core[head * 128 + j] = at::BFloat16(recurrent_sum(products));
                            }
                            for (int i = 0; i < 128; ++i) values[i] = __fmul_rn(float(core[head * 128 + i]), float(core[head * 128 + i]));
                            float inverse = rsqrtf(__fadd_rn(__fmul_rn(contiguous_sum(values, 128, 4), 1.0f / 128), 1e-6f));
                            for (int i = 0; i < 128; ++i) {
                                at::BFloat16 unit = at::BFloat16(__fmul_rn(float(core[head * 128 + i]), inverse));
                                at::BFloat16 scaled = at::BFloat16(__fmul_rn(float(weights[attention + 8445984 + i]), float(unit)));
                                input[head * 128 + i] = at::BFloat16(__fmul_rn(float(scaled), silu_value(float(z[head * 128 + i]))));
                            }
                        }
                    }
                }
            } else if (projection == output_projection) {
                for (int i = 0; i < 1024; ++i) {
                    residual[i] = at::BFloat16(__fadd_rn(float(residual[i]), float(projected[i])));
                    values[i] = __fmul_rn(float(residual[i]), float(residual[i]));
                }
                float inverse = rsqrtf(__fadd_rn(__fmul_rn(contiguous_sum(values, 1024, 4), 1.0f / 1024), 1e-6f));
                for (int i = 0; i < 1024; ++i) {
                    normalized[i] = at::BFloat16(__fmul_rn(__fmul_rn(float(residual[i]), inverse), __fadd_rn(1.0f, float(weights[base + 1024 + i]))));
                    input[i] = normalized[i];
                }
            } else if (projection == output_projection + 1) {
                for (int i = 0; i < 3584; ++i) gate[i] = projected[i];
            } else if (projection == output_projection + 2) {
                for (int i = 0; i < 3584; ++i) input[i] = at::BFloat16(__fmul_rn(float(at::BFloat16(silu_value(float(gate[i])))), float(projected[i])));
            } else {
                for (int i = 0; i < 1024; ++i) residual[i] = at::BFloat16(__fadd_rn(float(residual[i]), float(projected[i])));
            }
            if (projection < output_projection - 1)
                for (int i = 0; i < 1024; ++i) input[i] = normalized[i];
        }
        __syncwarp();

    }

    if (lane == 0)
        for (int i = 0; i < 1024; ++i) next_residual[i] = residual[i];
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                  torch::Tensor input2, torch::Tensor input3) {
    check_tensor(input0, at::kBFloat16, 21555360, input0);
    check_tensor(input1, at::kBFloat16, 1024, input0);
    check_tensor(input2, at::kBFloat16, 24576, input0);
    check_tensor(input3, at::kFloat, 262144, input0);
    auto residual = torch::empty({1024}, input0.options());
    auto cache = torch::empty({24576}, input0.options());
    auto recurrent = torch::empty({262144}, input0.options().dtype(at::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream();
    recurrent_decoder<<<1, 32, 0, stream>>>(
        input0.data_ptr<at::BFloat16>(), input1.data_ptr<at::BFloat16>(),
        input2.data_ptr<at::BFloat16>(), input3.data_ptr<float>(),
        residual.data_ptr<at::BFloat16>(), cache.data_ptr<at::BFloat16>(),
        recurrent.data_ptr<float>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {residual, cache, recurrent};
}
