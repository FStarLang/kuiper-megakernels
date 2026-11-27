#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>
#include <cstdint>
#include <vector>


void check_tensor(const torch::Tensor& value, at::ScalarType dtype, int64_t count,
                  const torch::Tensor& device) {
    TORCH_CHECK(value.is_cuda() && value.is_contiguous() && value.scalar_type() == dtype
                && value.device() == device.device() && value.numel() == count,
                "Expected a contiguous CUDA tensor with the declared dtype, size and device");
}
__global__ void initialize_state(const at::BFloat16* weights, const int64_t* prompt,
                                 at::BFloat16* embeddings, at::BFloat16* cache, float* recurrent,
                                 int64_t* tokens, at::BFloat16* history,
                                 int prompt_size, int capacity, int steps) {
    for (int p = 0; p < prompt_size; ++p)
        for (int i = 0; i < 1024; ++i)
            embeddings[p * 1024 + i] = weights[prompt[p] * 1024 + i];
    for (int i = 0; i < 442368 + capacity * 6144; ++i) cache[i] = at::BFloat16(0.0f);
    for (int i = 0; i < 4718592; ++i) recurrent[i] = 0.0f;
    for (int i = 0; i < steps; ++i) tokens[i] = 0;
    for (int i = 0; i < steps * 1024; ++i) history[i] = at::BFloat16(0.0f);
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1, int64_t input2) {
    check_tensor(input0, at::kBFloat16, 254279680, input0);
    check_tensor(input1, at::kLong, input1.numel(), input0);
    TORCH_CHECK(input1.dim() == 1 && input1.numel() >= 1 && input2 >= 1
                && input1.numel() + input2 - 1 <= 12, "Invalid prompt or generation length");
    auto prompt_size = input1.numel();
    auto capacity = prompt_size + input2 - 1;
    auto embeddings = torch::empty({prompt_size, 1024}, input0.options());
    auto cache = torch::empty({442368 + capacity * 6144}, input0.options());
    auto recurrent = torch::empty({4718592}, input0.options().dtype(at::kFloat));
    auto tokens = torch::empty({input2}, input1.options());
    auto history = torch::empty({input2, 1024}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    initialize_state<<<1, 1, 0, stream>>>(
        input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(),
        embeddings.data_ptr<at::BFloat16>(), cache.data_ptr<at::BFloat16>(),
        recurrent.data_ptr<float>(), tokens.data_ptr<int64_t>(),
        history.data_ptr<at::BFloat16>(), prompt_size, capacity, input2);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {embeddings, cache, recurrent, tokens, history};
}
