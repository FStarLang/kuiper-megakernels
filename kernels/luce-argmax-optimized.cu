#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <vector>

__device__ bool better(float v, int i, float best, int bi) {
    return i >= 0 && (bi < 0 || v > best || (isnan(v) && !isnan(best)));
}

__global__ void select_token(
    const float* scores, const int64_t* tokens_in, int64_t* selected,
    int64_t* tokens_out, int64_t steps, int decision) {
    for (int i = 0; i < steps; ++i) tokens_out[i] = tokens_in[i];
    float block_value[512], lane_value[32];
    int block_index[512], lane_index[32];
    for (int b = 0; b < 512; ++b) {
        for (int w = 0; w < 32; ++w) {
            float best = -INFINITY;
            int index = -1;
            if (w < 8) {
                for (int m = b * 485 + w; m < (b + 1) * 485; m += 8) {
                    float s = scores[m];
                    if (better(s, m, best, index)) { best = s; index = m; }
                }
            }
            lane_value[w] = best;
            lane_index[w] = index;
        }
        for (int d = 16; d; d /= 2) {
            for (int i = 0; i < d; ++i) {
                if (better(lane_value[i + d], lane_index[i + d], lane_value[i], lane_index[i])) {
                    lane_value[i] = lane_value[i + d];
                    lane_index[i] = lane_index[i + d];
                }
            }
        }
        block_value[b] = lane_value[0];
        block_index[b] = lane_index[0];
    }
    for (int d = 256; d; d /= 2) {
        for (int i = 0; i < d; ++i) {
            if (better(block_value[i + d], block_index[i + d], block_value[i], block_index[i])) {
                block_value[i] = block_value[i + d];
                block_index[i] = block_index[i + d];
            }
        }
    }
    tokens_out[decision] = block_index[0];
    selected[0] = block_index[0];
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1, int64_t input2) {
    TORCH_CHECK(input0.is_cuda() && input0.is_contiguous() &&
                input0.scalar_type() == at::kFloat && input0.sizes() == at::IntArrayRef({248320}),
                "scores must be a contiguous CUDA float32 vector of length 248320");
    TORCH_CHECK(input1.device() == input0.device() && input1.is_contiguous() &&
                input1.scalar_type() == at::kLong && input1.dim() == 1 &&
                input2 >= 0 && input2 < input1.numel(),
                "tokens must be CUDA int64 with an in-range decision offset");
    c10::cuda::CUDAGuard guard(input0.device());
    auto selected = torch::empty({}, input0.options().dtype(at::kLong));
    auto tokens = torch::empty({input1.numel()}, input0.options().dtype(at::kLong));
    auto stream = at::cuda::getCurrentCUDAStream();
    select_token<<<1, 1, 0, stream>>>(
        input0.data_ptr<float>(), input1.data_ptr<int64_t>(), selected.data_ptr<int64_t>(),
        tokens.data_ptr<int64_t>(), input1.numel(), int(input2));
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {selected, tokens};
}
