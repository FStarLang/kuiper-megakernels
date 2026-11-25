#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <climits>

__global__ void select_token(const int64_t* prompt, const at::BFloat16* logits,
                             int64_t* output, int64_t* selected,
                             int columns, int prompt_length) {
    for (int i = 0; i < prompt_length; ++i) output[i] = prompt[i];
    float best = float(logits[0]);
    int64_t index = 0;
    for (int col = 1; col < columns; ++col) {
        float value = float(logits[col]);
        if (value > best || (isnan(value) && !isnan(best))) {
            best = value;
            index = col;
        }
    }
    output[prompt_length] = index;
    selected[0] = index;
}

static void check_tensor(const torch::Tensor& tensor, at::ScalarType dtype,
                         c10::Device device, const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                tensor.scalar_type() == dtype && tensor.device() == device,
                name, " must be contiguous, have the declared dtype, and be on the selected CUDA device");
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1, int64_t input2) {
    check_tensor(input0, at::kLong, input0.device(), "input0 prompt");
    check_tensor(input1, at::kBFloat16, input0.device(), "input1 scores");
    TORCH_CHECK(input0.dim() == 1 && input0.numel() <= INT_MAX, "prompt must be a flat int64 tensor");
    TORCH_CHECK(input1.dim() == 1 && input1.numel() > 0 && input1.numel() <= INT_MAX,
                "scores must be a nonempty flat BF16 tensor");
    TORCH_CHECK(input2 == 1, "the accepted generation contract requires decode_steps=1");
    c10::cuda::CUDAGuard device_guard(input0.device());
    int64_t output_length = input0.numel() + double(input2);
    auto output_tokens = torch::empty({1, output_length}, input0.options());
    auto selected = torch::empty({}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    select_token<<<1, 1, 0, stream>>>(
        input0.data_ptr<int64_t>(), input1.data_ptr<at::BFloat16>(),
        output_tokens.data_ptr<int64_t>(), selected.data_ptr<int64_t>(),
        int(input1.numel()), int(input0.numel()));
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {output_tokens, selected};
}
