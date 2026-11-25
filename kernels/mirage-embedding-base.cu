#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <climits>

__device__ float rms_inverse(const at::BFloat16* input, int size) {
    // Replay the original four-value partial sums and descending reduction tree.
    float partial[128];
    int width = size == 1024 ? 128 : 32;
    for (int lane = 0; lane < width; ++lane) {
        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int start = lane * 4; start < size; start += width * 4) {
            for (int j = 0; j < 4; ++j) {
                float v = float(input[start + j]);
                sums[j] = __fadd_rn(sums[j], __fmul_rn(v, v));
            }
        }
        partial[lane] = __fadd_rn(__fadd_rn(__fadd_rn(sums[0], sums[1]), sums[2]), sums[3]);
    }
    for (int offset = width / 2; offset > 0; offset /= 2) {
        for (int lane = 0; lane < offset; ++lane) {
            partial[lane] = __fadd_rn(partial[lane], partial[lane + offset]);
        }
    }
    float mean = __fmul_rn(partial[0], 1.0f / float(size));
    return rsqrtf(__fadd_rn(mean, 1.0e-6f));
}

__global__ void embedding_norm(const at::BFloat16* weights, const int64_t* tokens,
                               at::BFloat16* residual, at::BFloat16* normalized,
                               at::BFloat16* rotary, int embedding_offset,
                               int norm_offset, int position_start) {
    for (int row = 0; row < 7; ++row) {
        for (int col = 0; col < 128; ++col) {
            float inverse = 1.0f / powf(1000000.0f, float(col % 64) / 64.0f);
            float angle = __fmul_rn(float(row + position_start), inverse);
            rotary[row * 128 + col] = at::BFloat16(cosf(angle));
            rotary[896 + row * 128 + col] = at::BFloat16(sinf(angle));
        }
    }
    for (int row = 0; row < 7; ++row) {
        int64_t start = embedding_offset + tokens[row] * 1024;
        float inverse = rms_inverse(weights + start, 1024);
        for (int col = 0; col < 1024; ++col) {
            at::BFloat16 value = weights[start + col];
            residual[row * 1024 + col] = value;
            at::BFloat16 scaled = at::BFloat16(__fmul_rn(float(value), inverse));
            normalized[row * 1024 + col] =
                at::BFloat16(__fmul_rn(float(scaled), float(weights[norm_offset + col])));
        }
    }
}

static void check_tensor(const torch::Tensor& tensor, at::ScalarType dtype,
                         c10::Device device, const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                tensor.scalar_type() == dtype && tensor.device() == device,
                name, " must be contiguous, have the declared dtype, and be on the selected CUDA device");
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                    int64_t input2, int64_t input3, int64_t input4) {
    check_tensor(input0, at::kBFloat16, input0.device(), "input0 weights");
    check_tensor(input1, at::kLong, input0.device(), "input1 prompt");
    TORCH_CHECK(input0.dim() == 1, "weights must be a flat packed tensor");
    TORCH_CHECK(input1.dim() == 1 && input1.numel() == 7, "prompt must have shape [7]");
    TORCH_CHECK(input2 >= 0 && input2 <= INT_MAX - 155582464 &&
                input2 <= input0.numel() - 155582464, "embedding offset is out of bounds");
    TORCH_CHECK(input3 >= 0 && input3 <= INT_MAX - 1024 &&
                input3 <= input0.numel() - 1024, "RMS weight offset is out of bounds");
    TORCH_CHECK(input4 >= 0 && input4 <= INT_MAX - 6, "position range is out of bounds");
    c10::cuda::CUDAGuard device_guard(input0.device());
    auto residual = torch::empty({7, 1024}, input0.options());
    auto normalized = torch::empty({7, 1024}, input0.options());
    auto rotary = torch::empty({2, 7, 128}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    embedding_norm<<<1, 1, 0, stream>>>(
        input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(),
        residual.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(),
        rotary.data_ptr<at::BFloat16>(), int(input2), int(input3), int(input4));
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {residual, normalized, rotary};
}
