#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <vector>

__device__ uint32_t pack_pair(at::BFloat16 a, at::BFloat16 b) {
  return uint32_t(a.x) | (uint32_t(b.x) << 16);
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

__global__ void finish(const int64_t* prompt, const at::BFloat16* logits,
                       const at::BFloat16* normalized, int64_t* tokens,
                       at::BFloat16* hidden) {
  float values[256], chunk_values[96];
  int64_t indices[256], chunk_indices[96];
  for (int chunk = 0; chunk < 96; ++chunk) {
    for (int t = 0; t < 128; ++t) {
      values[t] = -INFINITY;
      indices[t] = -1;
      for (int i = t; i < 1600 && chunk * 1600 + i < 151936; i += 128) {
        float candidate = float(logits[6 * 153600 + chunk * 1600 + i]);
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
  for (int i = 0; i < 7 * 1024; ++i)
    hidden[i] = normalized[i];
}


std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1, double input2) {
  auto residual0 = torch::empty({8, 1024}, input0.options());
  auto residual1 = torch::empty({8, 1024}, input0.options());
  auto normalized = torch::empty({8, 1024}, input0.options());
  auto qkv = torch::empty({8, 4096}, input0.options());
  auto attended = torch::empty({8, 2048}, input0.options());
  auto projected = torch::empty({8, 1024}, input0.options());
  auto gate_up = torch::empty({8, 6144}, input0.options());
  auto activated = torch::empty({8, 3072}, input0.options());
  auto cosine = torch::empty({4096, 128}, input0.options());
  auto sine = torch::empty({4096, 128}, input0.options());
  auto logits = torch::empty({8, 153600}, input0.options());
  auto tokens = torch::empty({1, 8}, input1.options());
  auto hidden = torch::empty({7, 1024}, input0.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  prepare<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), residual0.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 155583488);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 155585536, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 161876992);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 159779840, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 155584512);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 161877248, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 168168704, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 171314432);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 171316480, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 177607936);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 175510784, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 171315456);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 177608192, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 183899648, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 187045376);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 187047424, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 193338880);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 191241728, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 187046400);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 193339136, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 199630592, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 202776320);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 202778368, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 209069824);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 206972672, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 202777344);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 209070080, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 215361536, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 218507264);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 218509312, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 224800768);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 222703616, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 218508288);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 224801024, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 231092480, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 234238208);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 234240256, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 240531712);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 238434560, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 234239232);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 240531968, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 246823424, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 249969152);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 249971200, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 256262656);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 254165504, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 249970176);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 256262912, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 262554368, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 265700096);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 265702144, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 271993600);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 269896448, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 265701120);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 271993856, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 278285312, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 281431040);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 281433088, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 287724544);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 285627392, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 281432064);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 287724800, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 294016256, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 297161984);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 297164032, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 303455488);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 301358336, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 297163008);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 303455744, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 309747200, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 312892928);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 312894976, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 319186432);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 317089280, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 312893952);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 319186688, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 325478144, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 328623872);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 328625920, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 334917376);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 332820224, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 328624896);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 334917632, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 341209088, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 344354816);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 344356864, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 350648320);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 348551168, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 344355840);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 350648576, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 356940032, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 360085760);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 360087808, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 366379264);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 364282112, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 360086784);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 366379520, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 372670976, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 375816704);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 375818752, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 382110208);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 380013056, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 375817728);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 382110464, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 388401920, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 391547648);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 391549696, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 397841152);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 395744000, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 391548672);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 397841408, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 404132864, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 407278592);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 407280640, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 413572096);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 411474944, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 407279616);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 413572352, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 419863808, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 423009536);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 423011584, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 429303040);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 427205888, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 423010560);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 429303296, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 435594752, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 438740480);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 438742528, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 445033984);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 442936832, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 438741504);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 445034240, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 451325696, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 454471424);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 454473472, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 460764928);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 458667776, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 454472448);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 460765184, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 467056640, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 470202368);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 470204416, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 476495872);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 474398720, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 470203392);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 476496128, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 482787584, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 485933312);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 485935360, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 492226816);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 490129664, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 485934336);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 492227072, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 498518528, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 501664256);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 501666304, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 507957760);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 505860608, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 501665280);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 507958016, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 514249472, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 517395200);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 517397248, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 523688704);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 521591552, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 517396224);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 523688960, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 529980416, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 533126144);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 533128192, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 539419648);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 537322496, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 533127168);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 539419904, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 545711360, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 548857088);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 548859136, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 555150592);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 553053440, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 548858112);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 555150848, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 561442304, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 564588032);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 564590080, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 570881536);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 568784384, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 564589056);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 570881792, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 577173248, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 580318976);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 580321024, 4096, 1024, 1);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  attention<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 586612480);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 584515328, 1024, 2048, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 580320000);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 586612736, 6144, 1024, 2);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  silu_multiply<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 592904192, 1024, 3072, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  add_normalize<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 155582464);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  linear<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), 0, 153600, 1024, 0);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  finish<<<1, 1, 0, stream>>>(input1.data_ptr<int64_t>(), logits.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), tokens.data_ptr<int64_t>(), hidden.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  return {tokens, hidden};
}
