#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

__global__ void route_token(
    const float* row,
    const int32_t* input_selected,
    const float* input_selected_probability,
    const int32_t* previous_counts,
    const int64_t* previous_indices,
    const float* previous_probabilities,
    const int32_t* previous_survival,
    int token,
    int32_t* output_counts,
    int64_t* token_indices,
    float* probabilities,
    int32_t* survival) {
    int counts[8];
    for (int expert = 0; expert < 8; ++expert) {
        counts[expert] = previous_counts[expert];
    }
    for (int slot = 0; slot < 256; ++slot) {
        token_indices[slot] = previous_indices[slot];
        probabilities[slot] = previous_probabilities[slot];
    }
    for (int i = 0; i < 256 * 8; ++i) {
        survival[i] = previous_survival[i];
    }
    int selected = *input_selected;
    float selected_probability = *input_selected_probability;
    float combine_weight = __fadd_rn(0.0f, selected_probability);
    float probability = fdividef(row[selected], combine_weight);
    int rank = counts[selected]++;
    if (rank < 32) {
        token_indices[selected * 32 + rank] = token;
        probabilities[selected * 32 + rank] = probability;
        survival[token * 8 + selected] = 1;
    }
    for (int expert = 0; expert < 8; ++expert) {
        output_counts[expert] = counts[expert];
    }
}

std::vector<torch::Tensor> kplay_run(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor input3,
    torch::Tensor input4,
    torch::Tensor input5,
    torch::Tensor input6,
    int64_t input7) {
    TORCH_CHECK(
        input0.is_cuda() && input0.is_contiguous() && input0.scalar_type() == at::kFloat &&
        input0.dim() == 1 && input0.size(0) == 8,
        "routing requires a contiguous CUDA float32 row [8]");
    for (const auto& value : {input1, input2, input3, input4, input5, input6}) {
        TORCH_CHECK(
            value.device() == input0.device() && value.is_contiguous(),
            "routing inputs must be contiguous and on the row device");
    }
    TORCH_CHECK(
        input1.scalar_type() == at::kInt && input1.dim() == 0,
        "routing selected expert must be a rank-zero int32 tensor");
    TORCH_CHECK(
        input2.scalar_type() == at::kFloat && input2.dim() == 0,
        "routing selected probability must be a rank-zero float32 tensor");
    TORCH_CHECK(
        input3.scalar_type() == at::kInt && input3.dim() == 1 && input3.size(0) == 8,
        "routing counts must be int32 [8]");
    TORCH_CHECK(
        input4.scalar_type() == at::kLong &&
        input4.dim() == 2 && input4.size(0) == 8 && input4.size(1) == 32,
        "routing token indices must be int64 [8, 32]");
    TORCH_CHECK(
        input5.scalar_type() == at::kFloat &&
        input5.dim() == 2 && input5.size(0) == 8 && input5.size(1) == 32,
        "routing probabilities must be float32 [8, 32]");
    TORCH_CHECK(
        input6.scalar_type() == at::kInt &&
        input6.dim() == 2 && input6.size(0) == 256 && input6.size(1) == 8,
        "routing survival must be int32 [256, 8]");
    TORCH_CHECK(input7 >= 0 && input7 < 256, "routing token position must be in [0, 256)");
    const c10::cuda::CUDAGuard guard(input0.device());
    auto counts = torch::empty({8}, input3.options());
    auto token_indices = torch::empty({8, 32}, input4.options());
    auto probabilities = torch::empty({8, 32}, input5.options());
    auto survival = torch::empty({256, 8}, input6.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    route_token<<<1, 1, 0, stream>>>(
        input0.data_ptr<float>(), input1.data_ptr<int32_t>(), input2.data_ptr<float>(),
        input3.data_ptr<int32_t>(), input4.data_ptr<int64_t>(), input5.data_ptr<float>(),
        input6.data_ptr<int32_t>(), static_cast<int>(input7), counts.data_ptr<int32_t>(),
        token_indices.data_ptr<int64_t>(), probabilities.data_ptr<float>(), survival.data_ptr<int32_t>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {counts, token_indices, probabilities, survival};
}
