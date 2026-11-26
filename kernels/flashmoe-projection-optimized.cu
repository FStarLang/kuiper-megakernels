#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

__global__ void router_mma(
    const at::BFloat16* tokens,
    const at::BFloat16* weights,
    float* scores) {
    const int group = threadIdx.x / 4;
    const int pair = (threadIdx.x % 4) * 2;
    for (int token = 0; token < 256; token += 16) {
        float c0 = 0.0f;
        float c1 = 0.0f;
        float c2 = 0.0f;
        float c3 = 0.0f;
        for (int k = 0; k < 512; k += 16) {
            const int a = (token + group) * 512 + k + pair;
            const int b = (k + pair) * 8 + group;
            uint32_t a0 = uint32_t(tokens[a].x) |
                          (uint32_t(tokens[a + 1].x) << 16);
            uint32_t a1 = uint32_t(tokens[a + 8 * 512].x) |
                          (uint32_t(tokens[a + 8 * 512 + 1].x) << 16);
            uint32_t a2 = uint32_t(tokens[a + 8].x) |
                          (uint32_t(tokens[a + 9].x) << 16);
            uint32_t a3 = uint32_t(tokens[a + 8 * 512 + 8].x) |
                          (uint32_t(tokens[a + 8 * 512 + 9].x) << 16);
            uint32_t b0 = uint32_t(weights[b].x) |
                          (uint32_t(weights[b + 8].x) << 16);
            uint32_t b1 = uint32_t(weights[b + 8 * 8].x) |
                          (uint32_t(weights[b + 9 * 8].x) << 16);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                "{%0, %1, %2, %3};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
        scores[(token + group) * 8 + pair] = c0;
        scores[(token + group) * 8 + pair + 1] = c1;
        scores[(token + group + 8) * 8 + pair] = c2;
        scores[(token + group + 8) * 8 + pair + 1] = c3;
    }
}

__global__ void initialize_capacity_state(
    int32_t* output_counts,
    int64_t* token_indices,
    float* probabilities,
    int32_t* survival) {
    int counts[8] = {};
    for (int slot = 0; slot < 256; ++slot) {
        token_indices[slot] = -1;
        probabilities[slot] = 0.0f;
    }
    for (int i = 0; i < 256 * 8; ++i) {
        survival[i] = 0;
    }
    for (int expert = 0; expert < 8; ++expert) {
        output_counts[expert] = counts[expert];
    }
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1) {
    TORCH_CHECK(
        input0.is_cuda() && input0.is_contiguous() && input0.scalar_type() == at::kBFloat16 &&
        input0.dim() == 2 && input0.size(0) == 256 && input0.size(1) == 512,
        "projection requires contiguous CUDA bfloat16 tokens [256, 512]");
    TORCH_CHECK(
        input1.device() == input0.device() && input1.is_contiguous() &&
        input1.scalar_type() == at::kBFloat16 &&
        input1.dim() == 2 && input1.size(0) == 512 && input1.size(1) == 8,
        "projection requires contiguous bfloat16 weights [512, 8] on the token device");
    const c10::cuda::CUDAGuard guard(input0.device());
    auto scores = torch::empty({256, 8}, input0.options().dtype(at::kFloat));
    auto counts = torch::empty({8}, input0.options().dtype(at::kInt));
    auto token_indices = torch::empty({8, 32}, input0.options().dtype(at::kLong));
    auto probabilities = torch::empty({8, 32}, input0.options().dtype(at::kFloat));
    auto survival = torch::empty({256, 8}, input0.options().dtype(at::kInt));
    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    router_mma<<<1, 32, 0, stream>>>(
        input0.data_ptr<at::BFloat16>(), input1.data_ptr<at::BFloat16>(), scores.data_ptr<float>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    initialize_capacity_state<<<1, 1, 0, stream>>>(
        counts.data_ptr<int32_t>(), token_indices.data_ptr<int64_t>(),
        probabilities.data_ptr<float>(), survival.data_ptr<int32_t>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {scores, counts, token_indices, probabilities, survival};
}
