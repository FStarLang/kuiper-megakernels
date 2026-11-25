#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <vector>
#include <c10/cuda/CUDAGuard.h>

namespace {
void require_tensor(const torch::Tensor& value, const torch::Tensor& device,
                    at::ScalarType dtype, at::IntArrayRef shape) {
  TORCH_CHECK(value.is_cuda() && value.device() == device.device(),
              "All family tensors must be on the same CUDA device");
  TORCH_CHECK(value.is_contiguous() && value.scalar_type() == dtype
                  && value.sizes() == shape,
              "Family tensor has an invalid shape, dtype, or layout");
}
}

__device__ bool argmax_wins(float candidate, int64_t index,
                            float current, int64_t current_index) {
  if (index < 0)
    return false;
  if (current_index < 0)
    return true;
  if (isnan(candidate))
    return !isnan(current) || index < current_index;
  return !isnan(current) && candidate > current;
}

__global__ void select_tokens(const int64_t* prompt, const at::BFloat16* logits,
                              int64_t* tokens, int64_t* selected) {
  float values[256], chunk_values[96];
  int64_t indices[256], chunk_indices[96];
  for (int chunk = 0; chunk < 96; ++chunk) {
    for (int t = 0; t < 128; ++t) {
      values[t] = -INFINITY;
      indices[t] = -1;
      for (int i = t; i < 1600 && chunk * 1600 + i < 151936; i += 128) {
        float candidate = float(logits[chunk * 1600 + i]);
        if (argmax_wins(candidate, i, values[t], indices[t])) {
          values[t] = candidate;
          indices[t] = i;
        }
      }
    }
    for (int warp = 0; warp < 4; ++warp) {
      for (int delta = 16; delta > 0; delta /= 2) {
        for (int t = 0; t < delta; ++t) {
          int a = warp * 32 + t, b = a + delta;
          if (argmax_wins(values[b], indices[b], values[a], indices[a])) {
            values[a] = values[b];
            indices[a] = indices[b];
          }
        }
      }
      values[warp] = values[warp * 32];
      indices[warp] = indices[warp * 32];
    }
    for (int delta = 2; delta > 0; delta /= 2) {
      for (int t = 0; t < delta; ++t) {
        if (argmax_wins(values[t + delta], indices[t + delta],
                       values[t], indices[t])) {
          values[t] = values[t + delta];
          indices[t] = indices[t + delta];
        }
      }
    }
    chunk_values[chunk] = values[0];
    chunk_indices[chunk] = indices[0] < 0
        ? -1 : (int64_t(chunk) << 32) | indices[0];
  }
  for (int t = 0; t < 256; ++t) {
    values[t] = t < 96 ? chunk_values[t] : -INFINITY;
    indices[t] = t < 96 ? chunk_indices[t] : -1;
  }
  for (int warp = 0; warp < 8; ++warp) {
    for (int delta = 16; delta > 0; delta /= 2) {
      for (int t = 0; t < delta; ++t) {
        int a = warp * 32 + t, b = a + delta;
        if (argmax_wins(values[b], indices[b], values[a], indices[a])) {
          values[a] = values[b];
          indices[a] = indices[b];
        }
      }
    }
    values[warp] = values[warp * 32];
    indices[warp] = indices[warp * 32];
  }
  for (int delta = 4; delta > 0; delta /= 2) {
    for (int t = 0; t < delta; ++t) {
      if (argmax_wins(values[t + delta], indices[t + delta],
                     values[t], indices[t])) {
        values[t] = values[t + delta];
        indices[t] = indices[t + delta];
      }
    }
  }
  for (int i = 0; i < 7; ++i)
    tokens[i] = prompt[i];
  tokens[7] = indices[0] < 0 ? 0
      : (indices[0] >> 32) * 1600 + (indices[0] & 0xffffffff);
  selected[0] = tokens[7];
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1) {
  require_tensor(input0, input0, at::kBFloat16, {151936});
  require_tensor(input1, input0, at::kLong, {7});
  const c10::cuda::CUDAGuard device_guard(input0.device());
  auto selected = torch::empty({}, input1.options());
  auto tokens = torch::empty({1, 8}, input1.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  select_tokens<<<1, 1, 0, stream>>>(
      input1.data_ptr<int64_t>(), input0.data_ptr<at::BFloat16>(),
      tokens.data_ptr<int64_t>(), selected.data_ptr<int64_t>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  return {selected, tokens};
}
