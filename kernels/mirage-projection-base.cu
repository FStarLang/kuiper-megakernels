#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <climits>

__global__ void project(const at::BFloat16* input, const at::BFloat16* weights,
                        at::BFloat16* output, int columns, int inner, int weight_offset) {
    int group = int(threadIdx.x) / 4;
    int pair = int(threadIdx.x) % 4 * 2;
    // The original down projection uses two contiguous 1536-term split-K partials.
    int chunk = inner == 3072 ? 1536 : inner;
    for (int row = 0; row < columns; row += 16) {
        float total0 = 0.0f, total1 = 0.0f, total2 = 0.0f, total3 = 0.0f;
        for (int start = 0; start < inner; start += chunk) {
            float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
            for (int k = start; k < start + chunk; k += 16) {
                int a = weight_offset + (row + group) * inner + k + pair;
                uint32_t a0 = uint32_t(weights[a].x) | (uint32_t(weights[a + 1].x) << 16);
                uint32_t a1 = uint32_t(weights[a + 8 * inner].x) |
                              (uint32_t(weights[a + 8 * inner + 1].x) << 16);
                uint32_t a2 = uint32_t(weights[a + 8].x) | (uint32_t(weights[a + 9].x) << 16);
                uint32_t a3 = uint32_t(weights[a + 8 * inner + 8].x) |
                              (uint32_t(weights[a + 8 * inner + 9].x) << 16);
                uint32_t b0 = 0, b1 = 0;
                if (group < 7) {
                    int b = group * inner + k + pair;
                    b0 = uint32_t(input[b].x) | (uint32_t(input[b + 1].x) << 16);
                    b1 = uint32_t(input[b + 8].x) | (uint32_t(input[b + 9].x) << 16);
                }
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
            total0 = start == 0 ? c0 : __fadd_rn(total0, c0);
            total1 = start == 0 ? c1 : __fadd_rn(total1, c1);
            total2 = start == 0 ? c2 : __fadd_rn(total2, c2);
            total3 = start == 0 ? c3 : __fadd_rn(total3, c3);
        }
        output[pair * columns + row + group] = at::BFloat16(total0);
        output[pair * columns + row + group + 8] = at::BFloat16(total2);
        if (pair + 1 < 7) {
            output[(pair + 1) * columns + row + group] = at::BFloat16(total1);
            output[(pair + 1) * columns + row + group + 8] = at::BFloat16(total3);
        }
    }
}

static void check_tensor(const torch::Tensor& tensor, at::ScalarType dtype,
                         c10::Device device, const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                tensor.scalar_type() == dtype && tensor.device() == device,
                name, " must be contiguous, have the declared dtype, and be on the selected CUDA device");
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1, int64_t input2) {
    check_tensor(input0, at::kBFloat16, input0.device(), "input0 weights");
    check_tensor(input1, at::kBFloat16, input0.device(), "input1 final normalized hidden");
    TORCH_CHECK(input0.dim() == 1, "weights must be a flat packed tensor");
    TORCH_CHECK(input1.dim() == 2 && input1.size(0) == 7 && input1.size(1) == 1024,
                "final normalized hidden must have shape [7, 1024]");
    TORCH_CHECK(input2 >= 0 && input2 <= INT_MAX - 155582464 &&
                input2 <= input0.numel() - 155582464, "LM-head weight offset is out of bounds");
    c10::cuda::CUDAGuard device_guard(input0.device());
    auto logits = torch::empty({7, 151936}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    project<<<1, 32, 0, stream>>>(input1.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), 151936, 1024, int(input2));
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {logits};
}
