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

__global__ void prepare(const at::BFloat16* weights, const int64_t* prompt,
                        at::BFloat16* residual, at::BFloat16* cosine,
                        at::BFloat16* sine) {
  for (int row = 0; row < 8; ++row) {
    int64_t token = row < 7 ? prompt[row] : 0;
    for (int col = 0; col < 1024; ++col)
      residual[row * 1024 + col] = weights[token * 1024 + col];
  }
  for (int col = 0; col < 64; ++col) {
    float inverse = __fdiv_rn(1.0f, powf(1000000.0f, float(col) / 64.0f));
    for (int row = 0; row < 4096; ++row) {
      float angle = __fmul_rn(float(row), inverse);
      at::BFloat16 c(cosf(angle)), s(sinf(angle));
      cosine[row * 128 + col] = c;
      cosine[row * 128 + col + 64] = c;
      sine[row * 128 + col] = s;
      sine[row * 128 + col + 64] = s;
    }
  }
}

__global__ void normalize(const at::BFloat16* weights,
                          const at::BFloat16* input, at::BFloat16* output,
                          int weight_offset) {
  float partial[256];
  for (int row = 0; row < 8; ++row) {
    for (int t = 0; t < 256; ++t) {
      float sum = 0.0f;
      for (int j = 0; j < 4; ++j) {
        float v = float(input[row * 1024 + t + j * 256]);
        sum = __fmaf_rn(v, v, sum);
      }
      partial[t] = sum;
    }
    for (int warp = 0; warp < 8; ++warp) {
      for (int delta = 16; delta > 0; delta /= 2)
        for (int t = 0; t < delta; ++t)
          partial[warp * 32 + t] = __fadd_rn(
              partial[warp * 32 + t], partial[warp * 32 + t + delta]);
      partial[warp] = partial[warp * 32];
    }
    for (int delta = 4; delta > 0; delta /= 2)
      for (int t = 0; t < delta; ++t)
        partial[t] = __fadd_rn(partial[t], partial[t + delta]);
    float inverse = rsqrtf(__fadd_rn(partial[0] / 1024.0f, 1.0e-6f));
    for (int col = 0; col < 1024; ++col)
      output[row * 1024 + col] = at::BFloat16(
          __fmul_rn(float(input[row * 1024 + col]),
                    __fmul_rn(inverse, float(weights[weight_offset + col]))));
  }
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                    torch::Tensor input2) {
  require_tensor(input0, input0, at::kBFloat16, {151936, 1024});
  require_tensor(input1, input0, at::kLong, {7});
  require_tensor(input2, input0, at::kBFloat16, {1024});
  const c10::cuda::CUDAGuard device_guard(input0.device());
  auto residual = torch::empty({8, 1024}, input0.options());
  auto normalized = torch::empty({8, 1024}, input0.options());
  auto cosine = torch::empty({4096, 128}, input0.options());
  auto sine = torch::empty({4096, 128}, input0.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  prepare<<<1, 1, 0, stream>>>(
      input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(),
      residual.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(),
      sine.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  normalize<<<1, 1, 0, stream>>>(
      input2.data_ptr<at::BFloat16>(), residual.data_ptr<at::BFloat16>(),
      normalized.data_ptr<at::BFloat16>(), 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  return {residual, normalized, cosine, sine};
}
