#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>
#include <cstdint>
#include <vector>

__device__ float contiguous_sum(const float* values, int size, int width) {
    float partial[256];
    int count = size / width;
    for (int i = 0; i < count; ++i) {
        float value = 0.0f;
        for (int j = 0; j < width; ++j)
            value = __fadd_rn(value, values[i * width + j]);
        partial[i] = value;
    }
    for (int offset = count / 2; offset; offset /= 2)
        for (int i = 0; i < offset; ++i)
            partial[i] = __fadd_rn(partial[i], partial[i + offset]);
    return partial[0];
}

__device__ float recurrent_sum(const float* values) {
    float partial[4];
    for (int y = 0; y < 4; ++y) {
        float a[4] = {};
        for (int i = y; i < 128; i += 16)
            for (int j = 0; j < 4; ++j)
                a[j] = __fadd_rn(a[j], values[i + 4 * j]);
        partial[y] = __fadd_rn(__fadd_rn(__fadd_rn(a[0], a[1]), a[2]), a[3]);
    }
    return __fadd_rn(__fadd_rn(partial[0], partial[2]), __fadd_rn(partial[1], partial[3]));
}

__device__ float sigmoid_value(float x) {
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x)));
}

__device__ float silu_value(float x) {
    return __fdiv_rn(x, __fadd_rn(1.0f, expf(-x)));
}

__global__ void initialize_state(const at::BFloat16* weights, const int64_t* prompt,
                                 at::BFloat16* embeddings, at::BFloat16* cache, float* recurrent,
                                 int64_t* tokens, at::BFloat16* history,
                                 int prompt_size, int capacity, int steps) {
    for (int p = 0; p < prompt_size; ++p)
        for (int i = 0; i < 1024; ++i)
            embeddings[p * 1024 + i] = weights[prompt[p] * 1024 + i];
    for (int i = 0; i < 442368 + capacity * 6144; ++i) cache[i] = at::BFloat16(0.0f);
    for (int i = 0; i < 4718592; ++i) recurrent[i] = 0.0f;
    for (int i = 0; i < steps; ++i) tokens[i] = 0;
    for (int i = 0; i < steps * 1024; ++i) history[i] = at::BFloat16(0.0f);
}

__global__ void decode_token(const at::BFloat16* weights, const at::BFloat16* embeddings,
                             const int64_t* tokens, const at::BFloat16* old_cache,
                             const float* old_recurrent, at::BFloat16* new_cache,
                             float* new_recurrent, at::BFloat16* hidden,
                             at::BFloat16* logits, int prompt_size, int steps, int position) {
    int lane = threadIdx.x;
    int capacity = prompt_size + steps - 1;
    if (position >= capacity) {
        if (lane == 0) {
            for (int i = 0; i < 442368 + capacity * 6144; ++i) new_cache[i] = old_cache[i];
            for (int i = 0; i < 4718592; ++i) new_recurrent[i] = old_recurrent[i];
        }
        return;
    }
    __shared__ at::BFloat16 input[3584], projected[6144];
    __shared__ __nv_bfloat16 ma[256], mb[256];
    __shared__ float mc[256];
    at::BFloat16 residual[1024], normalized[1024], qkv[6144], z[2048], beta[16], gate[3584];
    at::BFloat16 core[2048], cosine[32], sine[32];
    float values[1024], products[128], updated[128], combined[16], scores[12];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> af;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> bf;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cf;
    if (lane == 0) {
        // HF constructs these configuration-only RoPE frequencies on the CPU.
        const float frequency[32] = {
            0x1.000000p+0f, 0x1.356656p-1f, 0x1.75f036p-2f, 0x1.c3f06ap-3f,
            0x1.111aeep-3f, 0x1.4a12acp-4f, 0x1.8eec7ep-5f, 0x1.e222eep-6f,
            0x1.235a72p-6f, 0x1.6020a4p-7f, 0x1.a9942ap-8f, 0x1.012cfap-8f,
            0x1.36d21ap-9f, 0x1.77a7dap-10f, 0x1.c603c4p-11f, 0x1.125c04p-11f,
            0x1.4b96c0p-12f, 0x1.90c182p-13f, 0x1.e459c4p-14f, 0x1.24b0fcp-14f,
            0x1.61bea4p-15f, 0x1.ab8882p-16f, 0x1.025b56p-16f, 0x1.383f88p-17f,
            0x1.796182p-18f, 0x1.c8198cp-19f, 0x1.139e94p-19f, 0x1.4d1c98p-20f,
            0x1.9298aep-21f, 0x1.e6933ap-22f, 0x1.26091cp-22f, 0x1.635e88p-23f
        };
        for (int i = 0; i < 1024; ++i)
            residual[i] = position < prompt_size ? embeddings[position * 1024 + i] : weights[tokens[position - prompt_size] * 1024 + i];
        for (int i = 0; i < 32; ++i) {
            float angle = __fmul_rn(float(position), frequency[i]);
            cosine[i] = at::BFloat16(cosf(angle));
            sine[i] = at::BFloat16(sinf(angle));
        }
    }
    int64_t base = 254280704;
    for (int layer = 0; layer <= 24; ++layer) {
        bool final = layer == 24;
        bool linear = !final && layer % 4 != 3;
        int output_projection = linear ? 4 : 3;
        int projections = final ? 1 : output_projection + 4;
        int64_t attention = base + 11012096;
        if (lane == 0) {
            int64_t norm = final ? 254279680 : base;
            for (int i = 0; i < 1024; ++i) {
                values[i] = __fmul_rn(float(residual[i]), float(residual[i]));
                if (final) hidden[i] = residual[i];
            }
            float inverse = rsqrtf(__fadd_rn(__fmul_rn(contiguous_sum(values, 1024, 4), 1.0f / 1024), 1e-6f));
            for (int i = 0; i < 1024; ++i) {
                normalized[i] = at::BFloat16(__fmul_rn(__fmul_rn(float(residual[i]), inverse), __fadd_rn(1.0f, float(weights[norm + i]))));
                input[i] = normalized[i];
            }
        }
        __syncwarp();
        for (int projection = 0; projection < projections; ++projection) {
            int n = 0, k = 1024;
            int64_t offset = 0;
            if (final) { n = 248320; offset = 0; }
            else if (projection == output_projection) {
                n = 1024; k = 2048;
                offset = attention + (linear ? 8446112 : 5243392);
            } else if (projection > output_projection) {
                int p = projection - output_projection;
                n = p == 3 ? 1024 : 3584;
                k = p == 3 ? 3584 : 1024;
                offset = base + 2048 + (p - 1) * 3670016;
            } else if (linear) {
                n = projection == 0 ? 6144 : (projection == 1 ? 2048 : 16);
                offset = attention + (projection == 0 ? 0 : (projection == 1 ? 6291456 : (projection == 2 ? 8388608 : 8404992)));
            } else {
                n = projection == 0 ? 4096 : 512;
                offset = attention + (projection == 0 ? 0 : (projection == 1 ? 4194304 : 4718592));
            }
            int chunk = k == 3584 ? 1792 : k;
            for (int row = 0; row < n; row += 16) {
                for (int start = 0; start < k; start += chunk) {
                    nvcuda::wmma::fill_fragment(cf, 0.0f);
                    for (int p = start; p < start + chunk; p += 16) {
                        for (int i = lane; i < 256; i += 32) {
                            ma[i] = __ushort_as_bfloat16(weights[offset + int64_t(row + i / 16) * k + p + i % 16].x);
                            mb[i] = __ushort_as_bfloat16(input[p + i % 16].x);
                        }
                        __syncwarp();
                        nvcuda::wmma::load_matrix_sync(af, ma, 16);
                        nvcuda::wmma::load_matrix_sync(bf, mb, 16);
                        nvcuda::wmma::mma_sync(cf, af, bf, cf);
                    }
                    nvcuda::wmma::store_matrix_sync(mc, cf, 16, nvcuda::wmma::mem_row_major);
                    __syncwarp();
                    if (lane == 0)
                        for (int i = 0; i < 16; ++i)
                            combined[i] = start == 0 ? mc[i * 16] : __fadd_rn(combined[i], mc[i * 16]);
                }
                if (lane == 0)
                    for (int i = 0; i < 16; ++i) {
                        if (final) logits[row + i] = at::BFloat16(combined[i]);
                        else projected[row + i] = at::BFloat16(combined[i]);
                    }
            }
            if (!final && !linear && projection == 2 && position == 7) {
                    if (lane == 0) {
                        for (int i = 0; i < 2048; ++i) input[i] = qkv[i];
                        for (int i = 0; i < 512; ++i) projected[512 + i] = qkv[4096 + i];
                    }
                    __syncwarp();
                    int cache = 442368 + (layer / 4) * capacity * 1024;
                    for (int head = 0; head < 8; ++head) {
                        nvcuda::wmma::fill_fragment(cf, 0.0f);
                        for (int p = 0; p < 256; p += 16) {
                            for (int i = lane; i < 256; i += 32) {
                                int t = i / 16, j = p + i % 16;
                                uint16_t key = t >= 8 ? 0 : (t == 7 ? projected[512 + (head / 4) * 256 + j].x : old_cache[cache + ((head / 4) * capacity + t) * 256 + j].x);
                                ma[i] = __ushort_as_bfloat16(key);
                                mb[i] = __ushort_as_bfloat16(input[head * 256 + j].x);
                            }
                            __syncwarp();
                            nvcuda::wmma::load_matrix_sync(af, ma, 16);
                            nvcuda::wmma::load_matrix_sync(bf, mb, 16);
                            nvcuda::wmma::mma_sync(cf, af, bf, cf);
                        }
                        nvcuda::wmma::store_matrix_sync(mc, cf, 16, nvcuda::wmma::mem_row_major);
                        __syncwarp();
                        if (lane == 0)
                            for (int i = 0; i < 8; ++i) projected[1024 + head * 8 + i] = at::BFloat16(mc[i * 16]);
                    }
            }
            if (lane == 0 && !final) {
                if (projection < output_projection) {
                    if (linear) {
                        if (projection == 0) for (int i = 0; i < 6144; ++i) qkv[i] = projected[i];
                        if (projection == 1) for (int i = 0; i < 2048; ++i) z[i] = projected[i];
                        if (projection == 2) for (int i = 0; i < 16; ++i) beta[i] = at::BFloat16(sigmoid_value(float(projected[i])));
                        if (projection == 3) {
                            int index = layer - layer / 4;
                            int conv = index * 24576;
                            for (int i = 0; i < 6144; ++i) {
                                float value = 0.0f;
                                for (int j = 0; j < 4; ++j) {
                                    at::BFloat16 x = j == 3 ? qkv[i] : old_cache[conv + i * 4 + j + 1];
                                    new_cache[conv + i * 4 + j] = x;
                                    value = __fmaf_rn(float(x), float(weights[attention + 8421376 + i * 4 + j]), value);
                                }
                                qkv[i] = at::BFloat16(silu_value(float(at::BFloat16(value))));
                            }
                            for (int head = 0; head < 16; ++head) {
                                for (int part = 0; part < 2; ++part) {
                                    int start = part * 2048 + head * 128;
                                    for (int i = 0; i < 128; ++i) values[i] = float(at::BFloat16(__fmul_rn(float(qkv[start + i]), float(qkv[start + i]))));
                                    float sum = float(at::BFloat16(contiguous_sum(values, 128, 8)));
                                    float inverse = float(at::BFloat16(rsqrtf(float(at::BFloat16(__fadd_rn(sum, 1e-6f))))));
                                    for (int i = 0; i < 128; ++i) qkv[start + i] = at::BFloat16(__fmul_rn(float(qkv[start + i]), inverse));
                                }
                                float a = __fadd_rn(float(projected[head]), float(weights[attention + 8445968 + head]));
                                float softplus = a > 20.0f ? a : log1pf(expf(a));
                                float decay = expf(__fmul_rn(-expf(float(weights[attention + 8445952 + head])), softplus));
                                int state = (index * 16 + head) * 16384;
                                for (int j = 0; j < 128; ++j) {
                                    for (int i = 0; i < 128; ++i) {
                                        updated[i] = __fmul_rn(old_recurrent[state + i * 128 + j], decay);
                                        products[i] = __fmul_rn(updated[i], float(qkv[2048 + head * 128 + i]));
                                    }
                                    float delta = __fmul_rn(__fsub_rn(float(qkv[4096 + head * 128 + j]), recurrent_sum(products)), float(beta[head]));
                                    for (int i = 0; i < 128; ++i) {
                                        float next = __fadd_rn(updated[i], __fmul_rn(float(qkv[2048 + head * 128 + i]), delta));
                                        new_recurrent[state + i * 128 + j] = next;
                                        float query = __fmul_rn(float(qkv[head * 128 + i]), 0.08838834764831845f);
                                        products[i] = __fmul_rn(next, query);
                                    }
                                    core[head * 128 + j] = at::BFloat16(recurrent_sum(products));
                                }
                                for (int i = 0; i < 128; ++i) values[i] = __fmul_rn(float(core[head * 128 + i]), float(core[head * 128 + i]));
                                float inverse = rsqrtf(__fadd_rn(__fmul_rn(contiguous_sum(values, 128, 4), 1.0f / 128), 1e-6f));
                                for (int i = 0; i < 128; ++i) {
                                    at::BFloat16 unit = at::BFloat16(__fmul_rn(float(core[head * 128 + i]), inverse));
                                    at::BFloat16 scaled = at::BFloat16(__fmul_rn(float(weights[attention + 8445984 + i]), float(unit)));
                                    input[head * 128 + i] = at::BFloat16(__fmul_rn(float(scaled), silu_value(float(z[head * 128 + i]))));
                                }
                            }
                        }
                    } else {
                        if (projection < 2) {
                            int heads = projection == 0 ? 8 : 2;
                            int stride = projection == 0 ? 512 : 256;
                            int destination = projection == 0 ? 0 : 4096;
                            int64_t norm = attention + (projection == 0 ? 5242880 : 5243136);
                            for (int head = 0; head < heads; ++head) {
                                for (int i = 0; i < 256; ++i) values[i] = __fmul_rn(float(projected[head * stride + i]), float(projected[head * stride + i]));
                                float inverse = rsqrtf(__fadd_rn(__fmul_rn(contiguous_sum(values, 256, 4), 1.0f / 256), 1e-6f));
                                for (int i = 0; i < 256; ++i)
                                    qkv[destination + head * 256 + i] = at::BFloat16(__fmul_rn(__fmul_rn(float(projected[head * stride + i]), inverse), __fadd_rn(1.0f, float(weights[norm + i]))));
                                for (int i = 0; i < 32; ++i) {
                                    float a = float(qkv[destination + head * 256 + i]);
                                    float b = float(qkv[destination + head * 256 + 32 + i]);
                                    float ac = float(at::BFloat16(__fmul_rn(a, float(cosine[i]))));
                                    float bs = float(at::BFloat16(__fmul_rn(-b, float(sine[i]))));
                                    float bc = float(at::BFloat16(__fmul_rn(b, float(cosine[i]))));
                                    float as = float(at::BFloat16(__fmul_rn(a, float(sine[i]))));
                                    qkv[destination + head * 256 + i] = at::BFloat16(__fadd_rn(ac, bs));
                                    qkv[destination + head * 256 + 32 + i] = at::BFloat16(__fadd_rn(bc, as));
                                }
                            }
                            if (projection == 0)
                                for (int head = 0; head < 8; ++head)
                                    for (int i = 0; i < 256; ++i) qkv[2048 + head * 256 + i] = projected[head * 512 + 256 + i];
                        } else {
                            int cache = 442368 + (layer / 4) * capacity * 1024;
                            for (int head = 0; head < 2; ++head)
                                for (int p = 0; p < capacity; ++p)
                                    for (int i = 0; i < 256; ++i) {
                                        int index = cache + (head * capacity + p) * 256 + i;
                                        new_cache[index] = p == position ? qkv[4096 + head * 256 + i] : old_cache[index];
                                        new_cache[index + capacity * 512] = p == position ? projected[head * 256 + i] : old_cache[index + capacity * 512];
                                    }
                            for (int head = 0; head < 8; ++head) {
                                for (int p = 0; p <= position; ++p) {
                                    at::BFloat16 dot;
                                    if (position == 7) dot = projected[1024 + head * 8 + p];
                                    else {
                                        for (int i = 0; i < 256; ++i) {
                                            at::BFloat16 key = p == position ? qkv[4096 + (head / 4) * 256 + i] : old_cache[cache + ((head / 4) * capacity + p) * 256 + i];
                                            values[i] = __fmul_rn(float(qkv[head * 256 + i]), float(key));
                                        }
                                        dot = at::BFloat16(contiguous_sum(values, 256, 16));
                                    }
                                    scores[p] = at::BFloat16(__fmul_rn(float(dot), 0.0625f));
                                }
                                int width = 1;
                                while (width <= position) width *= 2;
                                for (int p = 0; p < width; ++p) values[p] = p <= position ? scores[p] : -INFINITY;
                                for (int offset = width / 2; offset; offset /= 2) {
                                    for (int p = 0; p < width; ++p) products[p] = values[p] < values[p ^ offset] ? values[p ^ offset] : values[p];
                                    for (int p = 0; p < width; ++p) values[p] = products[p];
                                }
                                for (int p = 0; p < width; ++p) {
                                    if (p <= position) scores[p] = expf(__fsub_rn(scores[p], values[p]));
                                    values[p] = p <= position ? scores[p] : 0.0f;
                                }
                                for (int offset = width / 2; offset; offset /= 2) {
                                    for (int p = 0; p < width; ++p) products[p] = __fadd_rn(values[p], values[p ^ offset]);
                                    for (int p = 0; p < width; ++p) values[p] = products[p];
                                }
                                for (int p = 0; p <= position; ++p) scores[p] = at::BFloat16(values[p] == 0.0f ? NAN : __fdiv_rn(scores[p], values[p]));
                                if (position == 7)
                                    for (int p = 0; p < 8; ++p) input[head * 8 + p] = at::BFloat16(scores[p]);
                                else {
                                    for (int i = 0; i < 256; ++i) {
                                        float partial[4] = {};
                                        for (int p = 0; p <= position; ++p) {
                                            at::BFloat16 v = p == position ? projected[(head / 4) * 256 + i] : old_cache[cache + capacity * 512 + ((head / 4) * capacity + p) * 256 + i];
                                            partial[p / 4] = __fmaf_rn(scores[p], float(v), partial[p / 4]);
                                        }
                                        float value = __fadd_rn(__fadd_rn(partial[0], partial[2]), __fadd_rn(partial[1], partial[3]));
                                        if (position == 0) value = __fmul_rn(scores[0], float(projected[(head / 4) * 256 + i]));
                                        at::BFloat16 probability = at::BFloat16(sigmoid_value(float(qkv[2048 + head * 256 + i])));
                                        input[head * 256 + i] = at::BFloat16(__fmul_rn(float(at::BFloat16(value)), float(probability)));
                                    }
                                }
                            }
                        }
                    }
                } else if (projection == output_projection) {
                    for (int i = 0; i < 1024; ++i) {
                        residual[i] = at::BFloat16(__fadd_rn(float(residual[i]), float(projected[i])));
                        values[i] = __fmul_rn(float(residual[i]), float(residual[i]));
                    }
                    float inverse = rsqrtf(__fadd_rn(__fmul_rn(contiguous_sum(values, 1024, 4), 1.0f / 1024), 1e-6f));
                    for (int i = 0; i < 1024; ++i) {
                        normalized[i] = at::BFloat16(__fmul_rn(__fmul_rn(float(residual[i]), inverse), __fadd_rn(1.0f, float(weights[base + 1024 + i]))));
                        input[i] = normalized[i];
                    }
                } else if (projection == output_projection + 1) {
                    for (int i = 0; i < 3584; ++i) gate[i] = projected[i];
                } else if (projection == output_projection + 2) {
                    for (int i = 0; i < 3584; ++i) input[i] = at::BFloat16(__fmul_rn(float(at::BFloat16(silu_value(float(gate[i])))), float(projected[i])));
                } else {
                    for (int i = 0; i < 1024; ++i) residual[i] = at::BFloat16(__fadd_rn(float(residual[i]), float(projected[i])));
                }
                if (projection < output_projection - 1)
                    for (int i = 0; i < 1024; ++i) input[i] = normalized[i];
            }
            __syncwarp();
            if (!final && !linear && projection == 2 && position == 7) {
                int cache = 442368 + (layer / 4) * capacity * 1024 + capacity * 512;
                for (int head = 0; head < 8; ++head) {
                    for (int row = 0; row < 256; row += 16) {
                        for (int i = lane; i < 256; i += 32) {
                            int p = i % 16, j = row + i / 16;
                            uint16_t value = p >= 8 ? 0 : (p == 7 ? projected[(head / 4) * 256 + j].x : old_cache[cache + ((head / 4) * capacity + p) * 256 + j].x);
                            ma[i] = __ushort_as_bfloat16(value);
                            mb[i] = __ushort_as_bfloat16(p >= 8 ? 0 : input[head * 8 + p].x);
                        }
                        __syncwarp();
                        nvcuda::wmma::fill_fragment(cf, 0.0f);
                        nvcuda::wmma::load_matrix_sync(af, ma, 16);
                        nvcuda::wmma::load_matrix_sync(bf, mb, 16);
                        nvcuda::wmma::mma_sync(cf, af, bf, cf);
                        nvcuda::wmma::store_matrix_sync(mc, cf, 16, nvcuda::wmma::mem_row_major);
                        __syncwarp();
                        if (lane == 0)
                            for (int i = 0; i < 16; ++i) core[head * 256 + row + i] = at::BFloat16(mc[i * 16]);
                    }
                }
                if (lane == 0)
                    for (int i = 0; i < 2048; ++i)
                        input[i] = at::BFloat16(__fmul_rn(float(core[i]), float(at::BFloat16(sigmoid_value(float(qkv[2048 + i]))))));
                __syncwarp();
            }
        }
        base += linear ? 21555360 : 18352640;
    }
}

__global__ void finish_token(const int64_t* old_tokens, const at::BFloat16* old_history,
                             const at::BFloat16* hidden, const at::BFloat16* logits,
                             int64_t* tokens, at::BFloat16* history,
                             int prompt_size, int steps, int position) {
    int destination = position - prompt_size + 1;
    int64_t best = 0;
    if (destination >= 0 && destination < steps) {
        float value = float(logits[0]);
        for (int64_t i = 1; i < 248320; ++i) {
            float next = float(logits[i]);
            if (!isnan(value) && (isnan(next) || next > value)) { value = next; best = i; }
        }
    }
    for (int i = 0; i < steps; ++i) {
        tokens[i] = i == destination ? best : old_tokens[i];
        for (int j = 0; j < 1024; ++j)
            history[i * 1024 + j] = i == destination ? hidden[j] : old_history[i * 1024 + j];
    }
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1, int64_t input2) {
    auto prompt_size = input1.numel();
    auto capacity = prompt_size + input2 - 1;
    auto embeddings = torch::empty({prompt_size, 1024}, input0.options());
    auto cache0 = torch::empty({442368 + capacity * 6144}, input0.options());
    auto cache1 = torch::empty({442368 + capacity * 6144}, input0.options());
    auto recurrent0 = torch::empty({4718592}, input0.options().dtype(at::kFloat));
    auto recurrent1 = torch::empty({4718592}, input0.options().dtype(at::kFloat));
    auto tokens0 = torch::empty({input2}, input1.options());
    auto tokens1 = torch::empty({input2}, input1.options());
    auto history0 = torch::empty({input2, 1024}, input0.options());
    auto history1 = torch::empty({input2, 1024}, input0.options());
    auto hidden = torch::empty({1024}, input0.options());
    auto logits = torch::empty({248320}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    initialize_state<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), embeddings.data_ptr<at::BFloat16>(), cache0.data_ptr<at::BFloat16>(), recurrent0.data_ptr<float>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), prompt_size, capacity, input2);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_token<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), embeddings.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), cache0.data_ptr<at::BFloat16>(), recurrent0.data_ptr<float>(), cache1.data_ptr<at::BFloat16>(), recurrent1.data_ptr<float>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), prompt_size, input2, 0);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    finish_token<<<1, 1, 0, stream>>>(tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), prompt_size, input2, 0);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_token<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), embeddings.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), cache1.data_ptr<at::BFloat16>(), recurrent1.data_ptr<float>(), cache0.data_ptr<at::BFloat16>(), recurrent0.data_ptr<float>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), prompt_size, input2, 1);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    finish_token<<<1, 1, 0, stream>>>(tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), prompt_size, input2, 1);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_token<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), embeddings.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), cache0.data_ptr<at::BFloat16>(), recurrent0.data_ptr<float>(), cache1.data_ptr<at::BFloat16>(), recurrent1.data_ptr<float>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), prompt_size, input2, 2);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    finish_token<<<1, 1, 0, stream>>>(tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), prompt_size, input2, 2);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_token<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), embeddings.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), cache1.data_ptr<at::BFloat16>(), recurrent1.data_ptr<float>(), cache0.data_ptr<at::BFloat16>(), recurrent0.data_ptr<float>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), prompt_size, input2, 3);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    finish_token<<<1, 1, 0, stream>>>(tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), prompt_size, input2, 3);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_token<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), embeddings.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), cache0.data_ptr<at::BFloat16>(), recurrent0.data_ptr<float>(), cache1.data_ptr<at::BFloat16>(), recurrent1.data_ptr<float>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), prompt_size, input2, 4);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    finish_token<<<1, 1, 0, stream>>>(tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), prompt_size, input2, 4);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_token<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), embeddings.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), cache1.data_ptr<at::BFloat16>(), recurrent1.data_ptr<float>(), cache0.data_ptr<at::BFloat16>(), recurrent0.data_ptr<float>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), prompt_size, input2, 5);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    finish_token<<<1, 1, 0, stream>>>(tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), prompt_size, input2, 5);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_token<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), embeddings.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), cache0.data_ptr<at::BFloat16>(), recurrent0.data_ptr<float>(), cache1.data_ptr<at::BFloat16>(), recurrent1.data_ptr<float>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), prompt_size, input2, 6);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    finish_token<<<1, 1, 0, stream>>>(tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), prompt_size, input2, 6);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_token<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), embeddings.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), cache1.data_ptr<at::BFloat16>(), recurrent1.data_ptr<float>(), cache0.data_ptr<at::BFloat16>(), recurrent0.data_ptr<float>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), prompt_size, input2, 7);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    finish_token<<<1, 1, 0, stream>>>(tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), prompt_size, input2, 7);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_token<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), embeddings.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), cache0.data_ptr<at::BFloat16>(), recurrent0.data_ptr<float>(), cache1.data_ptr<at::BFloat16>(), recurrent1.data_ptr<float>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), prompt_size, input2, 8);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    finish_token<<<1, 1, 0, stream>>>(tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), prompt_size, input2, 8);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_token<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), embeddings.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), cache1.data_ptr<at::BFloat16>(), recurrent1.data_ptr<float>(), cache0.data_ptr<at::BFloat16>(), recurrent0.data_ptr<float>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), prompt_size, input2, 9);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    finish_token<<<1, 1, 0, stream>>>(tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), prompt_size, input2, 9);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_token<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), embeddings.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), cache0.data_ptr<at::BFloat16>(), recurrent0.data_ptr<float>(), cache1.data_ptr<at::BFloat16>(), recurrent1.data_ptr<float>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), prompt_size, input2, 10);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    finish_token<<<1, 1, 0, stream>>>(tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), prompt_size, input2, 10);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_token<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), embeddings.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), cache1.data_ptr<at::BFloat16>(), recurrent1.data_ptr<float>(), cache0.data_ptr<at::BFloat16>(), recurrent0.data_ptr<float>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), prompt_size, input2, 11);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    finish_token<<<1, 1, 0, stream>>>(tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), hidden.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), prompt_size, input2, 11);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {tokens0, history0};
}
