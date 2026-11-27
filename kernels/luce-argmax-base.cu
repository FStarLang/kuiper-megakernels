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

__global__ void select_token(const at::BFloat16* logits, int64_t* selected,
                             int prompt_size, int steps, int position) {

    int destination = position - prompt_size + 1;
    int64_t best = 0;
    if (destination >= 0 && destination < steps) {
        float value = float(logits[0]);
        for (int64_t i = 1; i < 248320; ++i) {
            float next = float(logits[i]);
            if (!isnan(value) && (isnan(next) || next > value)) { value = next; best = i; }
        }
    }

    selected[0] = best;
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, int64_t input1,
                                  int64_t input2, int64_t input3) {
    check_tensor(input0, at::kBFloat16, 248320, input0);
    TORCH_CHECK(input1 >= 1 && input2 >= 1 && input1 + input2 - 1 <= 12
                && input3 >= 0 && input3 < 12, "Invalid token-selection position");
    auto selected = torch::empty({}, input0.options().dtype(at::kLong));
    auto stream = at::cuda::getCurrentCUDAStream();
    select_token<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(),
        selected.data_ptr<int64_t>(), input1, input2, input3);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {selected};
}
