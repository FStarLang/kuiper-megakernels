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

__global__ void update_token_history(const int64_t* old_tokens,
                                     const at::BFloat16* old_history,
                                     const at::BFloat16* hidden, const int64_t* selected,
                                     int64_t* tokens, at::BFloat16* history,
                                     int prompt_size, int steps, int position) {
    int destination = position - prompt_size + 1;
    int64_t best = selected[0];
    for (int i = 0; i < steps; ++i) {
        tokens[i] = i == destination ? best : old_tokens[i];
        for (int j = 0; j < 1024; ++j)
            history[i * 1024 + j] = i == destination ? hidden[j] : old_history[i * 1024 + j];
    }

}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                  torch::Tensor input2, torch::Tensor input3,
                                  int64_t input4, int64_t input5) {
    check_tensor(input0, at::kLong, input0.numel(), input0);
    check_tensor(input1, at::kBFloat16, input0.numel() * 1024, input0);
    check_tensor(input2, at::kBFloat16, 1024, input0);
    check_tensor(input3, at::kLong, 1, input0);
    TORCH_CHECK(input0.dim() == 1 && input0.numel() >= 1 && input1.dim() == 2
                && input1.size(1) == 1024 && input3.dim() == 0 && input4 >= 1
                && input4 + input0.numel() - 1 <= 12 && input5 >= 0 && input5 < 12,
                "Invalid token/history state or position");
    auto tokens = torch::empty({input0.numel()}, input0.options());
    auto history = torch::empty({input0.numel(), 1024}, input1.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    update_token_history<<<1, 1, 0, stream>>>(
        input0.data_ptr<int64_t>(), input1.data_ptr<at::BFloat16>(),
        input2.data_ptr<at::BFloat16>(), input3.data_ptr<int64_t>(),
        tokens.data_ptr<int64_t>(), history.data_ptr<at::BFloat16>(),
        input4, input0.numel(), input5);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {tokens, history};
}
