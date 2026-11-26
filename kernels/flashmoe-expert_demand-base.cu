#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

__global__ void router_expert_demand(
    const int64_t* prefix,
    int32_t* expert_demand) {
    for (int expert = 0; expert < 8; ++expert) {
        expert_demand[expert] = int32_t(prefix[expert]);
    }
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0) {
    TORCH_CHECK(input0.is_cuda() && input0.is_contiguous() &&
                    input0.scalar_type() == at::kLong &&
                    input0.sizes() == at::IntArrayRef({8}),
                "input0 must be a contiguous CUDA int64 tensor of shape [8]");
    c10::cuda::CUDAGuard device_guard(input0.device());
    auto expert_demand = torch::empty({8}, input0.options().dtype(at::kInt));
    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    router_expert_demand<<<1, 1, 0, stream>>>(
        input0.data_ptr<int64_t>(), expert_demand.data_ptr<int32_t>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {expert_demand};
}
