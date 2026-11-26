#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

__global__ void router_capacity_step(
    const int32_t* selected_input,
    const float* selected_probability_input,
    const int64_t* prefix_input,
    const int64_t* token_indices_input,
    const float* probabilities_input,
    const int32_t* survival_input,
    int token,
    int64_t* prefix_output,
    int64_t* token_indices,
    float* probabilities,
    int32_t* survival) {
    int64_t prefix[8];
    for (int expert = 0; expert < 8; ++expert) {
        prefix[expert] = prefix_input[expert];
    }
    for (int slot = 0; slot < 256; ++slot) {
        token_indices[slot] = token_indices_input[slot];
        probabilities[slot] = probabilities_input[slot];
    }
    for (int offset = 0; offset < token * 8; ++offset) {
        survival[offset] = survival_input[offset];
    }

    int selected = selected_input[0];
    float selected_probability = selected_probability_input[0];
    float combine_probability =
        __fdiv_rn(selected_probability, selected_probability);
    int64_t rank = prefix[selected]++;
    for (int expert = 0; expert < 8; ++expert) {
        survival[token * 8 + expert] =
            expert == selected && rank < 32 ? 1 : 0;
    }
    if (rank < 32) {
        token_indices[selected * 32 + rank] = token;
        probabilities[selected * 32 + rank] = combine_probability;
    }
    for (int expert = 0; expert < 8; ++expert) {
        prefix_output[expert] = prefix[expert];
    }
}

void check_state_tensor(
    const torch::Tensor& tensor,
    at::ScalarType dtype,
    at::IntArrayRef shape,
    const torch::Device& device,
    const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == device &&
                    tensor.is_contiguous() && tensor.scalar_type() == dtype &&
                    tensor.sizes() == shape,
                name, " has an invalid routing-state dtype, shape, layout or device");
}

std::vector<torch::Tensor> kplay_run(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor input3,
    torch::Tensor input4,
    torch::Tensor input5,
    int64_t input6) {
    TORCH_CHECK(input6 >= 0 && input6 < 256, "input6 must be a token position in [0, 256)");
    check_state_tensor(input0, at::kInt, {}, input0.device(), "input0");
    check_state_tensor(input1, at::kFloat, {}, input0.device(), "input1");
    check_state_tensor(input2, at::kLong, {8}, input0.device(), "input2");
    check_state_tensor(input3, at::kLong, {8, 32}, input0.device(), "input3");
    check_state_tensor(input4, at::kFloat, {8, 32}, input0.device(), "input4");
    check_state_tensor(input5, at::kInt, {input6, 8}, input0.device(), "input5");
    c10::cuda::CUDAGuard device_guard(input0.device());
    auto counts = torch::empty({8}, input0.options().dtype(at::kLong));
    auto token_indices = torch::empty({8, 32}, input3.options());
    auto probabilities = torch::empty({8, 32}, input4.options());
    auto survival = torch::empty({input6 + 1, 8}, input5.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    router_capacity_step<<<1, 1, 0, stream>>>(
        input0.data_ptr<int32_t>(), input1.data_ptr<float>(),
        input2.data_ptr<int64_t>(), input3.data_ptr<int64_t>(),
        input4.data_ptr<float>(), input5.data_ptr<int32_t>(),
        static_cast<int>(input6),
        counts.data_ptr<int64_t>(),
        token_indices.data_ptr<int64_t>(), probabilities.data_ptr<float>(),
        survival.data_ptr<int32_t>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {counts, token_indices, probabilities, survival};
}
