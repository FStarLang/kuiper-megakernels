#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>

__device__ float rms_inverse(const at::BFloat16* input, int size) {
    // Replay the original four-value partial sums and descending reduction tree.
    float partial[128];
    int width = size == 1024 ? 128 : 32;
    for (int lane = 0; lane < width; ++lane) {
        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int start = lane * 4; start < size; start += width * 4) {
            for (int j = 0; j < 4; ++j) {
                float v = float(input[start + j]);
                sums[j] = __fadd_rn(sums[j], __fmul_rn(v, v));
            }
        }
        partial[lane] = __fadd_rn(__fadd_rn(__fadd_rn(sums[0], sums[1]), sums[2]), sums[3]);
    }
    for (int offset = width / 2; offset > 0; offset /= 2) {
        for (int lane = 0; lane < offset; ++lane) {
            partial[lane] = __fadd_rn(partial[lane], partial[lane + offset]);
        }
    }
    float mean = __fmul_rn(partial[0], 1.0f / float(size));
    return rsqrtf(__fadd_rn(mean, 1.0e-6f));
}

__global__ void embedding_norm(const at::BFloat16* weights, const int64_t* tokens,
                               at::BFloat16* residual, at::BFloat16* normalized,
                               at::BFloat16* rotary) {
    for (int row = 0; row < 7; ++row) {
        for (int col = 0; col < 128; ++col) {
            float inverse = 1.0f / powf(1000000.0f, float(col % 64) / 64.0f);
            float angle = __fmul_rn(float(row), inverse);
            rotary[row * 128 + col] = at::BFloat16(cosf(angle));
            rotary[896 + row * 128 + col] = at::BFloat16(sinf(angle));
        }
    }
    for (int row = 0; row < 7; ++row) {
        int64_t start = tokens[row] * 1024;
        float inverse = rms_inverse(weights + start, 1024);
        for (int col = 0; col < 1024; ++col) {
            at::BFloat16 value = weights[start + col];
            residual[row * 1024 + col] = value;
            at::BFloat16 scaled = at::BFloat16(__fmul_rn(float(value), inverse));
            normalized[row * 1024 + col] =
                at::BFloat16(__fmul_rn(float(scaled), float(weights[155583488 + col])));
        }
    }
}

__global__ void project(const at::BFloat16* input, const at::BFloat16* weights,
                        at::BFloat16* output, int columns, int inner, int weight_offset) {
    int group = int(threadIdx.x) / 4;
    int pair = int(threadIdx.x) % 4 * 2;
    // The original down projection uses two contiguous 1536-term split-K partials.
    int chunk = inner == 3072 ? 1536 : inner;
    for (int row = 0; row < columns; row += 16) {
        float total0 = 0.0f, total1 = 0.0f, total2 = 0.0f, total3 = 0.0f;
        for (int start = 0; start < inner; start += chunk) {
            float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
            for (int k = start; k < start + chunk; k += 16) {
                int a = weight_offset + (row + group) * inner + k + pair;
                uint32_t a0 = uint32_t(weights[a].x) | (uint32_t(weights[a + 1].x) << 16);
                uint32_t a1 = uint32_t(weights[a + 8 * inner].x) |
                              (uint32_t(weights[a + 8 * inner + 1].x) << 16);
                uint32_t a2 = uint32_t(weights[a + 8].x) | (uint32_t(weights[a + 9].x) << 16);
                uint32_t a3 = uint32_t(weights[a + 8 * inner + 8].x) |
                              (uint32_t(weights[a + 8 * inner + 9].x) << 16);
                uint32_t b0 = 0, b1 = 0;
                if (group < 7) {
                    int b = group * inner + k + pair;
                    b0 = uint32_t(input[b].x) | (uint32_t(input[b + 1].x) << 16);
                    b1 = uint32_t(input[b + 8].x) | (uint32_t(input[b + 9].x) << 16);
                }
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
            total0 = start == 0 ? c0 : __fadd_rn(total0, c0);
            total1 = start == 0 ? c1 : __fadd_rn(total1, c1);
            total2 = start == 0 ? c2 : __fadd_rn(total2, c2);
            total3 = start == 0 ? c3 : __fadd_rn(total3, c3);
        }
        output[pair * columns + row + group] = at::BFloat16(total0);
        output[pair * columns + row + group + 8] = at::BFloat16(total2);
        if (pair + 1 < 7) {
            output[(pair + 1) * columns + row + group] = at::BFloat16(total1);
            output[(pair + 1) * columns + row + group + 8] = at::BFloat16(total3);
        }
    }
}

__global__ void attention(const at::BFloat16* qkv, const at::BFloat16* weights,
                          const at::BFloat16* rotary, at::BFloat16* output,
                          int layer_offset) {
    at::BFloat16 qk[21504];
    at::BFloat16 normalized[128];
    for (int row = 0; row < 7; ++row) {
        for (int head = 0; head < 24; ++head) {
            int base = row * 4096 + head * 128;
            int norm_offset = layer_offset + 6293504 + (head < 16 ? 0 : 128);
            float inverse = rms_inverse(qkv + base, 128);
            for (int col = 0; col < 128; ++col) {
                at::BFloat16 scaled = at::BFloat16(__fmul_rn(float(qkv[base + col]), inverse));
                normalized[col] =
                    at::BFloat16(__fmul_rn(float(scaled), float(weights[norm_offset + col])));
            }
            for (int col = 0; col < 128; ++col) {
                float other = float(normalized[(col + 64) % 128]);
                if (col < 64) other = -other;
                at::BFloat16 a = at::BFloat16(__fmul_rn(float(normalized[col]),
                                                     float(rotary[row * 128 + col])));
                at::BFloat16 b = at::BFloat16(__fmul_rn(other, float(rotary[896 + row * 128 + col])));
                qk[(row * 24 + head) * 128 + col] =
                    at::BFloat16(__fadd_rn(float(a), float(b)));
            }
        }
    }
    for (int head = 0; head < 16; ++head) {
        for (int row = 0; row < 7; ++row) {
            float scores[8];
            for (int key = 0; key < 7; ++key) {
                float sum = 0.0f;
                // The original small-N GEMM sums 16 strided partials in order.
                for (int lane = 0; lane < 16; ++lane) {
                    float partial = 0.0f;
                    for (int col = lane; col < 128; col += 16) {
                        partial = __fmaf_rn(float(qk[(row * 24 + head) * 128 + col]),
                                            float(qk[(key * 24 + 16 + head / 2) * 128 + col]), partial);
                    }
                    sum = __fadd_rn(sum, partial);
                }
                at::BFloat16 dot = at::BFloat16(sum);
                at::BFloat16 scaled = at::BFloat16(__fmul_rn(float(dot), 0.08838834764831845f));
                scores[key] = float(at::BFloat16(__fadd_rn(
                    float(scaled), key > row ? -3.3895313892515355e38f : 0.0f)));
            }
            scores[7] = -INFINITY;
            float maximum[8];
            for (int i = 0; i < 8; ++i) maximum[i] = scores[i];
            for (int offset = 4; offset > 0; offset /= 2) {
                for (int i = 0; i < offset; ++i) {
                    maximum[i] = maximum[i] < maximum[i + offset] ? maximum[i + offset] : maximum[i];
                }
            }
            float sums[8];
            for (int i = 0; i < 8; ++i) {
                scores[i] = expf(__fsub_rn(scores[i], maximum[0]));
                sums[i] = scores[i];
            }
            for (int offset = 4; offset > 0; offset /= 2) {
                for (int i = 0; i < offset; ++i) sums[i] = __fadd_rn(sums[i], sums[i + offset]);
            }
            at::BFloat16 probabilities[7];
            for (int i = 0; i < 7; ++i) probabilities[i] = at::BFloat16(scores[i] / sums[0]);
            for (int col = 0; col < 128; ++col) {
                // The value GEMM partitions its seven terms into 4 + 3.
                float partials[2] = {0.0f, 0.0f};
                for (int part = 0; part < 2; ++part) {
                    for (int key = part * 4; key < (part == 0 ? 4 : 7); ++key) {
                        partials[part] = __fmaf_rn(float(probabilities[key]),
                            float(qkv[key * 4096 + 3072 + head / 2 * 128 + col]), partials[part]);
                    }
                }
                output[row * 2048 + head * 128 + col] = at::BFloat16(__fadd_rn(partials[0], partials[1]));
            }
        }
    }
}

__global__ void residual_norm(const at::BFloat16* residual, const at::BFloat16* update,
                              const at::BFloat16* weights, at::BFloat16* summed,
                              at::BFloat16* normalized, int norm_offset) {
    at::BFloat16 row_values[1024];
    for (int row = 0; row < 7; ++row) {
        for (int col = 0; col < 1024; ++col) {
            int i = row * 1024 + col;
            row_values[col] = at::BFloat16(__fadd_rn(float(residual[i]), float(update[i])));
            summed[i] = row_values[col];
        }
        float inverse = rms_inverse(row_values, 1024);
        for (int col = 0; col < 1024; ++col) {
            at::BFloat16 scaled = at::BFloat16(__fmul_rn(float(row_values[col]), inverse));
            normalized[row * 1024 + col] =
                at::BFloat16(__fmul_rn(float(scaled), float(weights[norm_offset + col])));
        }
    }
}

__global__ void silu_product(const at::BFloat16* gate_up, at::BFloat16* output) {
    for (int row = 0; row < 7; ++row) {
        for (int col = 0; col < 3072; ++col) {
            float gate = float(gate_up[row * 6144 + col]);
            at::BFloat16 activation = at::BFloat16(gate / __fadd_rn(1.0f, expf(-gate)));
            output[row * 3072 + col] = at::BFloat16(__fmul_rn(
                float(activation), float(gate_up[row * 6144 + 3072 + col])));
        }
    }
}

__global__ void select_token(const int64_t* prompt, const at::BFloat16* logits,
                             int64_t* output) {
    for (int i = 0; i < 7; ++i) output[i] = prompt[i];
    float best = float(logits[6 * 151936]);
    int64_t index = 0;
    for (int col = 1; col < 151936; ++col) {
        float value = float(logits[6 * 151936 + col]);
        if (value > best || (isnan(value) && !isnan(best))) {
            best = value;
            index = col;
        }
    }
    output[7] = index;
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1, double input2) {
    int64_t output_length = 7 + input2;
    auto residual0 = torch::empty({7, 1024}, input0.options());
    auto residual1 = torch::empty({7, 1024}, input0.options());
    auto normalized = torch::empty({7, 1024}, input0.options());
    auto rotary = torch::empty({2, 7, 128}, input0.options());
    auto qkv = torch::empty({7, 4096}, input0.options());
    auto attended = torch::empty({7, 2048}, input0.options());
    auto projected = torch::empty({7, 1024}, input0.options());
    auto gate_up = torch::empty({7, 6144}, input0.options());
    auto activated = torch::empty({7, 3072}, input0.options());
    auto output_hidden = torch::empty({7, 1024}, input0.options());
    auto logits = torch::empty({7, 151936}, input0.options());
    auto output_tokens = torch::empty({1, output_length}, input1.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    embedding_norm<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 155585536);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 155583488);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 159779840);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 155584512);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 161877248);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 168168704);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 171314432);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 171316480);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 171314432);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 175510784);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 171315456);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 177608192);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 183899648);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 187045376);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 187047424);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 187045376);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 191241728);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 187046400);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 193339136);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 199630592);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 202776320);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 202778368);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 202776320);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 206972672);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 202777344);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 209070080);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 215361536);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 218507264);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 218509312);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 218507264);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 222703616);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 218508288);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 224801024);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 231092480);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 234238208);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 234240256);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 234238208);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 238434560);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 234239232);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 240531968);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 246823424);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 249969152);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 249971200);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 249969152);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 254165504);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 249970176);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 256262912);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 262554368);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 265700096);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 265702144);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 265700096);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 269896448);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 265701120);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 271993856);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 278285312);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 281431040);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 281433088);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 281431040);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 285627392);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 281432064);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 287724800);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 294016256);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 297161984);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 297164032);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 297161984);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 301358336);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 297163008);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 303455744);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 309747200);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 312892928);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 312894976);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 312892928);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 317089280);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 312893952);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 319186688);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 325478144);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 328623872);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 328625920);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 328623872);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 332820224);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 328624896);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 334917632);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 341209088);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 344354816);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 344356864);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 344354816);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 348551168);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 344355840);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 350648576);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 356940032);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 360085760);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 360087808);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 360085760);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 364282112);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 360086784);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 366379520);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 372670976);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 375816704);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 375818752);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 375816704);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 380013056);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 375817728);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 382110464);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 388401920);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 391547648);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 391549696);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 391547648);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 395744000);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 391548672);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 397841408);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 404132864);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 407278592);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 407280640);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 407278592);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 411474944);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 407279616);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 413572352);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 419863808);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 423009536);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 423011584);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 423009536);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 427205888);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 423010560);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 429303296);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 435594752);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 438740480);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 438742528);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 438740480);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 442936832);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 438741504);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 445034240);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 451325696);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 454471424);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 454473472);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 454471424);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 458667776);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 454472448);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 460765184);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 467056640);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 470202368);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 470204416);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 470202368);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 474398720);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 470203392);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 476496128);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 482787584);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 485933312);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 485935360);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 485933312);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 490129664);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 485934336);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 492227072);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 498518528);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 501664256);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 501666304);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 501664256);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 505860608);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 501665280);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 507958016);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 514249472);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 517395200);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 517397248);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 517395200);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 521591552);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 517396224);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 523688960);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 529980416);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 533126144);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 533128192);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 533126144);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 537322496);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 533127168);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 539419904);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 545711360);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 548857088);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 548859136);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 548857088);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 553053440);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 548858112);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 555150848);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 561442304);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 564588032);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 564590080);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 564588032);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 568784384);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 564589056);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 570881792);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 577173248);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 580318976);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, 580321024);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), 580318976);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, 584515328);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 580320000);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, 586612736);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, 592904192);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), output_hidden.data_ptr<at::BFloat16>(), 155582464);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(output_hidden.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), 151936, 1024, 0);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    select_token<<<1, 1, 0, stream>>>(input1.data_ptr<int64_t>(), logits.data_ptr<at::BFloat16>(), output_tokens.data_ptr<int64_t>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {output_tokens, output_hidden};
}
