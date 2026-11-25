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

__device__ uint32_t pack_pair(at::BFloat16 a, at::BFloat16 b) {
  return uint32_t(a.x) | (uint32_t(b.x) << 16);
}

__global__ void linear(const at::BFloat16* weights,
                       const at::BFloat16* input, at::BFloat16* output,
                       int weight_offset, int width, int reduction, int layout) {
  int group = int(threadIdx.x) / 4;
  int pair = (int(threadIdx.x) % 4) * 2;
  for (int tile = 0; tile < width; tile += 16) {
    int row0 = tile + group;
    int row1 = row0 + 8;
    int w0 = row0, w1 = row1;
    if (layout == 1) {
      int block0 = row0 / 512, col0 = row0 % 512;
      int block1 = row1 / 512, col1 = row1 % 512;
      w0 = col0 < 256 ? block0 * 256 + col0
                     : 2048 + (col0 - 256) / 128 * 1024
                           + block0 * 128 + col0 % 128;
      w1 = col1 < 256 ? block1 * 256 + col1
                     : 2048 + (col1 - 256) / 128 * 1024
                           + block1 * 128 + col1 % 128;
    } else if (layout == 2) {
      w0 = row0 / 128 * 64 + row0 % 64 + (row0 % 128 / 64) * 3072;
      w1 = row1 / 128 * 64 + row1 % 64 + (row1 % 128 / 64) * 3072;
    }
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    for (int k = 0; k < reduction; k += 16) {
      uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      if (width != 153600 || w0 < 151936) {
        a0 = pack_pair(weights[weight_offset + w0 * reduction + k + pair],
                       weights[weight_offset + w0 * reduction + k + pair + 1]);
        a2 = pack_pair(weights[weight_offset + w0 * reduction + k + pair + 8],
                       weights[weight_offset + w0 * reduction + k + pair + 9]);
      }
      if (width != 153600 || w1 < 151936) {
        a1 = pack_pair(weights[weight_offset + w1 * reduction + k + pair],
                       weights[weight_offset + w1 * reduction + k + pair + 1]);
        a3 = pack_pair(weights[weight_offset + w1 * reduction + k + pair + 8],
                       weights[weight_offset + w1 * reduction + k + pair + 9]);
      }
      uint32_t b0 = pack_pair(input[group * reduction + k + pair],
                              input[group * reduction + k + pair + 1]);
      uint32_t b1 = pack_pair(input[group * reduction + k + pair + 8],
                              input[group * reduction + k + pair + 9]);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    output[pair * width + row0] = at::BFloat16(d0);
    output[(pair + 1) * width + row0] = at::BFloat16(d1);
    output[pair * width + row1] = at::BFloat16(d2);
    output[(pair + 1) * width + row1] = at::BFloat16(d3);
  }
}

__global__ void copy_hidden(const at::BFloat16* normalized,
                            at::BFloat16* hidden) {
  for (int i = 0; i < 7 * 1024; ++i)
    hidden[i] = normalized[i];
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1) {
  require_tensor(input0, input0, at::kBFloat16, {151936, 1024});
  require_tensor(input1, input0, at::kBFloat16, {8, 1024});
  const c10::cuda::CUDAGuard device_guard(input0.device());
  auto logits = torch::empty({8, 153600}, input0.options());
  auto hidden = torch::empty({7, 1024}, input0.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  linear<<<1, 32, 0, stream>>>(
      input0.data_ptr<at::BFloat16>(), input1.data_ptr<at::BFloat16>(),
      logits.data_ptr<at::BFloat16>(), 0, 153600, 1024, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  copy_hidden<<<1, 1, 0, stream>>>(
      input1.data_ptr<at::BFloat16>(), hidden.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  return {logits, hidden};
}
