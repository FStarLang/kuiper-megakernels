#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

__global__ void initialize(
    const int64_t* prompt, float* states, float* conv, at::BFloat16* keys, at::BFloat16* values,
    int64_t* tokens, at::BFloat16* history, int64_t steps) {
    for (int i = 0; i < 4718592; ++i) states[i] = 0.0f;
    for (int i = 0; i < 442368; ++i) conv[i] = 0.0f;
    for (int i = 0; i < 6291456; ++i) {
        keys[i] = at::BFloat16(0.0f);
        values[i] = at::BFloat16(0.0f);
    }
    for (int i = 0; i < steps; ++i) tokens[i] = 0;
    tokens[0] = prompt[0];
    for (int i = 0; i < steps * 1024; ++i) history[i] = at::BFloat16(0.0f);
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, int64_t input1) {
    TORCH_CHECK(input0.is_cuda() && input0.is_contiguous() && input0.scalar_type() == at::kLong,
                "prompt must be contiguous CUDA int64");
    TORCH_CHECK(input0.dim() == 1 && input0.numel() > 0 && input1 > 0 && input1 <= 12,
                "expected a nonempty prompt and 1..12 generation steps");
    c10::cuda::CUDAGuard guard(input0.device());
    auto states = torch::empty({18, 16, 128, 128}, input0.options().dtype(at::kFloat));
    auto conv = torch::empty({18, 6144, 4}, input0.options().dtype(at::kFloat));
    auto keys = torch::empty({6, 2, 2048, 256}, input0.options().dtype(at::kBFloat16));
    auto values = torch::empty({6, 2, 2048, 256}, input0.options().dtype(at::kBFloat16));
    auto tokens = torch::empty({input1}, input0.options().dtype(at::kLong));
    auto history = torch::empty({input1, 1024}, input0.options().dtype(at::kBFloat16));
    auto stream = at::cuda::getCurrentCUDAStream();
    initialize<<<1, 1, 0, stream>>>(
        input0.data_ptr<int64_t>(), states.data_ptr<float>(), conv.data_ptr<float>(),
        keys.data_ptr<at::BFloat16>(), values.data_ptr<at::BFloat16>(),
        tokens.data_ptr<int64_t>(), history.data_ptr<at::BFloat16>(), input1);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {states, conv, keys, values, tokens, history};
}
