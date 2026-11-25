#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <climits>

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

static void check_tensor(const torch::Tensor& tensor, at::ScalarType dtype,
                         c10::Device device, const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                tensor.scalar_type() == dtype && tensor.device() == device,
                name, " must be contiguous, have the declared dtype, and be on the selected CUDA device");
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                    torch::Tensor input2, torch::Tensor input3,
                                    int64_t input4, int64_t input5) {
    check_tensor(input0, at::kBFloat16, input0.device(), "input0 weights");
    check_tensor(input1, at::kBFloat16, input0.device(), "input1 residual");
    check_tensor(input2, at::kBFloat16, input0.device(), "input2 normalized");
    check_tensor(input3, at::kBFloat16, input0.device(), "input3 rotary");
    TORCH_CHECK(input0.dim() == 1, "weights must be a flat packed tensor");
    TORCH_CHECK(input1.dim() == 2 && input1.size(0) == 7 && input1.size(1) == 1024,
                "residual must have shape [7, 1024]");
    TORCH_CHECK(input2.sizes() == input1.sizes(), "normalized must have shape [7, 1024]");
    TORCH_CHECK(input3.dim() == 3 && input3.size(0) == 2 &&
                input3.size(1) == 7 && input3.size(2) == 128,
                "rotary must have shape [2, 7, 128]");
    TORCH_CHECK(input4 >= 0 && input4 <= INT_MAX - 15730944 &&
                input4 <= input0.numel() - 15730944, "decoder weight offset is out of bounds");
    TORCH_CHECK(input5 >= 0 && input5 <= INT_MAX - 1024 &&
                input5 <= input0.numel() - 1024, "following RMS weight offset is out of bounds");
    c10::cuda::CUDAGuard device_guard(input0.device());
    int layer_offset = int(input4);
    auto residual0 = torch::empty({7, 1024}, input0.options());
    auto residual1 = torch::empty({7, 1024}, input0.options());
    auto normalized = torch::empty({7, 1024}, input0.options());
    auto qkv = torch::empty({7, 4096}, input0.options());
    auto attended = torch::empty({7, 2048}, input0.options());
    auto projected = torch::empty({7, 1024}, input0.options());
    auto gate_up = torch::empty({7, 6144}, input0.options());
    auto activated = torch::empty({7, 3072}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    project<<<1, 32, 0, stream>>>(input2.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv.data_ptr<at::BFloat16>(), 4096, 1024, layer_offset + 2048);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    attention<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), input3.data_ptr<at::BFloat16>(), attended.data_ptr<at::BFloat16>(), layer_offset);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(attended.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 2048, layer_offset + 4196352);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(input1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), layer_offset + 1024);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gate_up.data_ptr<at::BFloat16>(), 6144, 1024, layer_offset + 6293760);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    silu_product<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    project<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), 1024, 3072, layer_offset + 12585216);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    residual_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), int(input5));
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {residual0, normalized};
}
