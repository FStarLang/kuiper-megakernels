#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

__global__ void router_argmax(
    const float* softmax,
    int32_t* selected_output,
    float* selected_probability_output) {
    int selected = 0;
    float selected_probability = softmax[0];
    for (int expert = 1; expert < 8; ++expert) {
        float probability = softmax[expert];
        if (probability > selected_probability) {
            selected = expert;
            selected_probability = probability;
        }
    }
    selected_output[0] = selected;
    selected_probability_output[0] = selected_probability;
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0) {
    TORCH_CHECK(input0.is_cuda() && input0.is_contiguous() &&
                    input0.scalar_type() == at::kFloat &&
                    input0.sizes() == at::IntArrayRef({8}),
                "input0 must be a contiguous CUDA float32 tensor of shape [8]");
    c10::cuda::CUDAGuard device_guard(input0.device());
    auto selected = torch::empty({}, input0.options().dtype(at::kInt));
    auto selected_probability = torch::empty({}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    router_argmax<<<1, 1, 0, stream>>>(
        input0.data_ptr<float>(), selected.data_ptr<int32_t>(),
        selected_probability.data_ptr<float>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {selected, selected_probability};
}
