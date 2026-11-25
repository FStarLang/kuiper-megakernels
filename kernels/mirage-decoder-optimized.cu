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

__global__ void add_normalize(const at::BFloat16* weights,
                              const at::BFloat16* projected,
                              const at::BFloat16* residual,
                              at::BFloat16* next_residual,
                              at::BFloat16* normalized, int weight_offset) {
  at::BFloat16 values[1024];
  float partial[256];
  for (int row = 0; row < 8; ++row) {
    for (int col = 0; col < 1024; ++col) {
      values[col] = at::BFloat16(__fadd_rn(float(projected[row * 1024 + col]),
                                         float(residual[row * 1024 + col])));
      next_residual[row * 1024 + col] = values[col];
    }
    for (int t = 0; t < 256; ++t) {
      float sum = 0.0f;
      for (int j = 0; j < 4; ++j) {
        float v = float(values[t + j * 256]);
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
      normalized[row * 1024 + col] = at::BFloat16(
          __fmul_rn(float(values[col]),
                    __fmul_rn(inverse, float(weights[weight_offset + col]))));
  }
}

__global__ void silu_multiply(const at::BFloat16* input,
                              at::BFloat16* output) {
  for (int row = 0; row < 8; ++row) {
    for (int col = 0; col < 3072; ++col) {
      int offset = row * 6144 + col / 64 * 128 + col % 64;
      float gate = float(input[offset]);
      at::BFloat16 activated(gate / (1.0f + expf(-gate)));
      output[row * 3072 + col] =
          row < 7 ? at::BFloat16(__fmul_rn(float(activated),
                                         float(input[offset + 64])))
                  : at::BFloat16(0.0f);
    }
  }
}

__global__ void attention(const at::BFloat16* weights,
                          const at::BFloat16* input,
                          const at::BFloat16* cosine,
                          const at::BFloat16* sine, at::BFloat16* output,
                          int norm_offset) {
  __shared__ at::BFloat16 q[16 * 128];
  __shared__ at::BFloat16 k[8 * 128];
  __shared__ at::BFloat16 v[8 * 128];
  __shared__ at::BFloat16 probability[16 * 8];
  __shared__ float score[16 * 8];
  int lane = int(threadIdx.x);
  int group = lane / 4;
  int pair = (lane % 4) * 2;
  for (int head = 0; head < 8; ++head) {
    if (lane == 0) {
      float values[128], partial[32];
      at::BFloat16 unit[128];
      for (int row = 0; row < 24; ++row) {
        int token = row < 16 ? row / 2 : row - 16;
        int part = row < 16 ? row % 2 : 2;
        for (int col = 0; col < 128; ++col)
          values[col] = token < 7
              ? float(input[token * 4096 + head * 512 + part * 128 + col])
              : 0.0f;
        for (int t = 0; t < 32; ++t) {
          float sum = 0.0f;
          for (int j = 0; j < 4; ++j) {
            float value = values[t + j * 32];
            sum = __fmaf_rn(value, value, sum);
          }
          partial[t] = sum;
        }
        for (int delta = 16; delta > 0; delta /= 2)
          for (int t = 0; t < delta; ++t)
            partial[t] = __fadd_rn(partial[t], partial[t + delta]);
        float inverse = rsqrtf(__fadd_rn(partial[0] / 128.0f, 1.0e-6f));
        for (int col = 0; col < 128; ++col) {
          at::BFloat16 scaled(__fmul_rn(values[col], inverse));
          unit[col] = at::BFloat16(__fmul_rn(
              float(scaled),
              float(weights[norm_offset + (row < 16 ? 0 : 128) + col])));
        }
        for (int col = 0; col < 128; ++col) {
          float other = float(unit[(col + 64) % 128]);
          if (col < 64)
            other = -other;
          at::BFloat16 a(__fmul_rn(float(unit[col]),
                                   float(cosine[token * 128 + col])));
          at::BFloat16 b(__fmul_rn(other, float(sine[token * 128 + col])));
          at::BFloat16 rotated(__fadd_rn(float(a), float(b)));
          if (row < 16)
            q[row * 128 + col] = rotated;
          else
            k[(row - 16) * 128 + col] = rotated;
        }
      }
      for (int row = 0; row < 8; ++row)
        for (int col = 0; col < 128; ++col)
          v[row * 128 + col] = row < 7
              ? input[row * 4096 + head * 512 + 384 + col]
              : at::BFloat16(0.0f);
    }
    __syncwarp();
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    for (int reduction = 0; reduction < 128; reduction += 16) {
      int col = reduction + pair;
      uint32_t a0 = pack_pair(q[group * 128 + col], q[group * 128 + col + 1]);
      uint32_t a1 = pack_pair(q[(group + 8) * 128 + col],
                              q[(group + 8) * 128 + col + 1]);
      uint32_t a2 = pack_pair(q[group * 128 + col + 8],
                              q[group * 128 + col + 9]);
      uint32_t a3 = pack_pair(q[(group + 8) * 128 + col + 8],
                              q[(group + 8) * 128 + col + 9]);
      uint32_t b0 = pack_pair(k[group * 128 + col], k[group * 128 + col + 1]);
      uint32_t b1 = pack_pair(k[group * 128 + col + 8],
                              k[group * 128 + col + 9]);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    score[group * 8 + pair] = d0;
    score[group * 8 + pair + 1] = d1;
    score[(group + 8) * 8 + pair] = d2;
    score[(group + 8) * 8 + pair + 1] = d3;
    __syncwarp();
    if (lane == 0) {
      float scaled[8], exponent[8];
      at::BFloat16 mask;
      mask.x = 0xff7f;
      for (int row = 0; row < 16; ++row) {
        float maximum = -INFINITY;
        for (int col = 0; col < 8; ++col) {
          at::BFloat16 raw(score[row * 8 + col]);
          at::BFloat16 s(__fmul_rn(float(raw), 1.0f / sqrtf(128.0f)));
          at::BFloat16 masked(__fadd_rn(
              float(s), col > row / 2 ? float(mask) : 0.0f));
          scaled[col] = col < 7 && row < 14 ? float(masked) : -INFINITY;
          maximum = fmaxf(maximum, scaled[col]);
        }
        float shift = maximum == -INFINITY ? 0.0f : maximum;
        for (int col = 0; col < 8; ++col)
          exponent[col] = col < 7 && row < 14
              ? __expf(__fsub_rn(scaled[col], shift)) : 0.0f;
        float sum = __fadd_rn(
            __fadd_rn(__fadd_rn(exponent[0], exponent[1]),
                      __fadd_rn(exponent[2], exponent[3])),
            __fadd_rn(__fadd_rn(exponent[4], exponent[5]),
                      __fadd_rn(exponent[6], exponent[7])));
        float inverse = __fdividef(1.0f, sum);
        for (int col = 0; col < 8; ++col)
          probability[row * 8 + col] = col < 7 && row < 14
              ? at::BFloat16(__fmul_rn(exponent[col], inverse))
              : at::BFloat16(0.0f);
      }
    }
    __syncwarp();
    for (int tile = 0; tile < 128; tile += 8) {
      d0 = 0.0f;
      d1 = 0.0f;
      d2 = 0.0f;
      d3 = 0.0f;
      for (int reduction = 0; reduction < 64; reduction += 16) {
        uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;
        if (reduction == 0) {
          a0 = pack_pair(probability[group * 8 + pair],
                         probability[group * 8 + pair + 1]);
          a1 = pack_pair(probability[(group + 8) * 8 + pair],
                         probability[(group + 8) * 8 + pair + 1]);
          b0 = pack_pair(v[pair * 128 + tile + group],
                         v[(pair + 1) * 128 + tile + group]);
        }
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
      int offset = group / 2 * 2048 + head * 256 + group % 2 * 128 + tile + pair;
      output[offset] = at::BFloat16(d0);
      output[offset + 1] = at::BFloat16(d1);
      if (group + 8 < 14) {
        output[offset + 4 * 2048] = at::BFloat16(d2);
        output[offset + 4 * 2048 + 1] = at::BFloat16(d3);
      }
    }
    __syncwarp();
  }
  if (lane == 0)
    for (int col = 0; col < 2048; ++col)
      output[7 * 2048 + col] = at::BFloat16(0.0f);
}

std::vector<torch::Tensor> kplay_run(
    torch::Tensor input0, torch::Tensor input1, torch::Tensor input2,
    torch::Tensor input3, torch::Tensor input4, torch::Tensor input5,
    torch::Tensor input6, torch::Tensor input7, torch::Tensor input8,
    torch::Tensor input9, torch::Tensor input10, int64_t input11) {
  require_tensor(input0, input0, at::kBFloat16, {4096, 1024});
  require_tensor(input1, input0, at::kBFloat16, {256});
  require_tensor(input2, input0, at::kBFloat16, {1024, 2048});
  require_tensor(input3, input0, at::kBFloat16, {1024});
  require_tensor(input4, input0, at::kBFloat16, {6144, 1024});
  require_tensor(input5, input0, at::kBFloat16, {1024, 3072});
  require_tensor(input6, input0, at::kBFloat16, {1024});
  require_tensor(input7, input0, at::kBFloat16, {8, 1024});
  require_tensor(input8, input0, at::kBFloat16, {8, 1024});
  require_tensor(input9, input0, at::kBFloat16, {4096, 128});
  require_tensor(input10, input0, at::kBFloat16, {4096, 128});
  TORCH_CHECK(input11 >= 0 && input11 <= 4088,
              "The eight padded positions must fit the rotary tables");
  const c10::cuda::CUDAGuard device_guard(input0.device());
  auto residual0 = torch::empty({8, 1024}, input0.options());
  auto residual1 = torch::empty({8, 1024}, input0.options());
  auto normalized = torch::empty({8, 1024}, input0.options());
  auto qkv = torch::empty({8, 4096}, input0.options());
  auto attended = torch::empty({8, 2048}, input0.options());
  auto projected = torch::empty({8, 1024}, input0.options());
  auto gate_up = torch::empty({8, 6144}, input0.options());
  auto activated = torch::empty({8, 3072}, input0.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  linear<<<1, 32, 0, stream>>>(
      input0.data_ptr<at::BFloat16>(), input8.data_ptr<at::BFloat16>(),
      qkv.data_ptr<at::BFloat16>(), 0, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(
      input1.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(),
      input9.data_ptr<at::BFloat16>() + input11 * 128,
      input10.data_ptr<at::BFloat16>() + input11 * 128,
      attended.data_ptr<at::BFloat16>(), 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(
      input2.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(),
      projected.data_ptr<at::BFloat16>(), 0, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(
      input3.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(),
      input7.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(),
      normalized.data_ptr<at::BFloat16>(), 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(
      input4.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(),
      gate_up.data_ptr<at::BFloat16>(), 0, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(
      gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(
      input5.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(),
      projected.data_ptr<at::BFloat16>(), 0, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(
      input6.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(),
      residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(),
      normalized.data_ptr<at::BFloat16>(), 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  return {residual0, normalized};
}
