#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <vector>

__global__ void argmax_row(
    const float* row, int32_t* output_selected, float* output_selected_probability) {
    int selected = 0;
    float selected_probability = row[0];
    for (int expert = 1; expert < 8; ++expert) {
        if (row[expert] > selected_probability) {
            selected = expert;
            selected_probability = row[expert];
        }
    }
    *output_selected = selected;
    *output_selected_probability = selected_probability;
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0) {
    TORCH_CHECK(
        input0.is_cuda() && input0.is_contiguous() && input0.scalar_type() == at::kFloat &&
        input0.dim() == 1 && input0.size(0) == 8,
        "argmax requires a contiguous CUDA float32 row [8]");
    const c10::cuda::CUDAGuard guard(input0.device());
    auto selected = torch::empty({}, input0.options().dtype(at::kInt));
    auto selected_probability = torch::empty({}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    argmax_row<<<1, 1, 0, stream>>>(
        input0.data_ptr<float>(), selected.data_ptr<int32_t>(), selected_probability.data_ptr<float>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {selected, selected_probability};
}
