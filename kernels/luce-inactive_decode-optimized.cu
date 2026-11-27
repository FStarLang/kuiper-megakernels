#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

__global__ void inactive_decode(
    const float* state_in, const float* conv_in,
    const at::BFloat16* key_in, const at::BFloat16* value_in,
    const int64_t* tokens_in, const at::BFloat16* history_in,
    float* state_out, float* conv_out, at::BFloat16* key_out, at::BFloat16* value_out,
    int64_t* tokens_out, at::BFloat16* history_out, int64_t steps) {
    for (int i = 0; i < 6291456; ++i) {
        key_out[i] = key_in[i];
        value_out[i] = value_in[i];
    }
    for (int i = 0; i < steps; ++i) tokens_out[i] = tokens_in[i];
    for (int i = 0; i < steps * 1024; ++i) history_out[i] = history_in[i];
    for (int i = 0; i < 4718592; ++i) state_out[i] = state_in[i];
    for (int i = 0; i < 442368; ++i) conv_out[i] = conv_in[i];
}

std::vector<torch::Tensor> kplay_run(
    torch::Tensor input0, torch::Tensor input1, torch::Tensor input2,
    torch::Tensor input3, torch::Tensor input4, torch::Tensor input5,
    int64_t input6, int64_t input7) {
    TORCH_CHECK(input0.is_cuda(), "state must be on CUDA");
    const std::vector<torch::Tensor> inputs = {input0, input1, input2, input3, input4, input5};
    const at::ScalarType types[] = {
        at::kFloat, at::kFloat, at::kBFloat16, at::kBFloat16, at::kLong, at::kBFloat16};
    for (int i = 0; i < 6; ++i)
        TORCH_CHECK(inputs[i].device() == input0.device() && inputs[i].is_contiguous() &&
                    inputs[i].scalar_type() == types[i], "invalid inactive-decode state tensor ", i);
    TORCH_CHECK(input0.sizes() == at::IntArrayRef({18, 16, 128, 128}) &&
                input1.sizes() == at::IntArrayRef({18, 6144, 4}) &&
                input2.sizes() == at::IntArrayRef({6, 2, 2048, 256}) &&
                input3.sizes() == input2.sizes() &&
                input4.dim() == 1 && input4.numel() > 0 &&
                input5.sizes() == at::IntArrayRef({input4.numel(), 1024}),
                "invalid inactive-decode state shapes");
    TORCH_CHECK(input7 > 0 && input6 >= input7 + input4.numel() - 1 && input6 < 12,
                "inactive-decode requires the accepted early-return branch");
    c10::cuda::CUDAGuard guard(input0.device());
    auto state = torch::empty({18, 16, 128, 128}, input0.options().dtype(at::kFloat));
    auto conv = torch::empty({18, 6144, 4}, input0.options().dtype(at::kFloat));
    auto keys = torch::empty({6, 2, 2048, 256}, input0.options().dtype(at::kBFloat16));
    auto values = torch::empty({6, 2, 2048, 256}, input0.options().dtype(at::kBFloat16));
    auto tokens = torch::empty({input4.numel()}, input0.options().dtype(at::kLong));
    auto history = torch::empty({input4.numel(), 1024}, input0.options().dtype(at::kBFloat16));
    auto stream = at::cuda::getCurrentCUDAStream();
    inactive_decode<<<1, 1, 0, stream>>>(
        input0.data_ptr<float>(), input1.data_ptr<float>(), input2.data_ptr<at::BFloat16>(),
        input3.data_ptr<at::BFloat16>(), input4.data_ptr<int64_t>(), input5.data_ptr<at::BFloat16>(),
        state.data_ptr<float>(), conv.data_ptr<float>(), keys.data_ptr<at::BFloat16>(),
        values.data_ptr<at::BFloat16>(), tokens.data_ptr<int64_t>(), history.data_ptr<at::BFloat16>(),
        input4.numel());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {state, conv, keys, values, tokens, history};
}
