#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <vector>

__global__ void softmax_row(const float* scores, int token, float* output_row) {
    float row[8];
    float maximum = -std::numeric_limits<float>::infinity();
    float denominator = 0.0f;
    for (int expert = 0; expert < 8; ++expert) {
        row[expert] = scores[token * 8 + expert];
        float previous = maximum;
        maximum = fmaxf(maximum, row[expert]);
        denominator = __fmaf_rn(
            denominator, __expf(__fsub_rn(previous, maximum)),
            __expf(__fsub_rn(row[expert], maximum)));
    }
    for (int expert = 0; expert < 8; ++expert) {
        row[expert] = __fdividef(
            __expf(__fsub_rn(row[expert], maximum)), denominator);
    }
    for (int expert = 0; expert < 8; ++expert) {
        output_row[expert] = row[expert];
    }
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, int64_t input1) {
    TORCH_CHECK(
        input0.is_cuda() && input0.is_contiguous() && input0.scalar_type() == at::kFloat &&
        input0.dim() == 2 && input0.size(0) == 256 && input0.size(1) == 8,
        "softmax requires contiguous CUDA float32 scores [256, 8]");
    TORCH_CHECK(input1 >= 0 && input1 < 256, "softmax token position must be in [0, 256)");
    const c10::cuda::CUDAGuard guard(input0.device());
    auto row = torch::empty({8}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    softmax_row<<<1, 1, 0, stream>>>(
        input0.data_ptr<float>(), static_cast<int>(input1), row.data_ptr<float>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {row};
}
