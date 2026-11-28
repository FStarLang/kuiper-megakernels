#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <vector>

__device__ float add32(float a, float b) {
    float r;
    asm("add.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ float mul32(float a, float b) {
    float r;
    asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ float sub32(float a, float b) {
    float r;
    asm("sub.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ float divide32(float a, float b) {
    float r;
    asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ float fma32(float a, float b, float c) {
    float r;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

__device__ float log2_32(float x) {
    float r;
    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ float exp2_32(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ float exp32(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(mul32(x, 1.44269504088896340736f)));
    return r;
}

__device__ float reciprocal32(float x) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ float rsqrt32(float x) {
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ float sigmoid32(float x) {
    return reciprocal32(add32(1.0f, exp32(-x)));
}

__device__ float silu32(float x) {
    return mul32(x, sigmoid32(x));
}

// The arrays hold logical lanes; no physical lane communication is used.
__device__ float tree32(float* x) {
    for (int d = 16; d; d /= 2)
        for (int i = 0; i < d; ++i)
            x[i] = add32(x[i], x[i + d]);
    return x[0];
}

__device__ float square_sum(const float* x, int n, int threads) {
    float warps[32] = {}, lanes[32];
    for (int w = 0; w < threads / 32; ++w) {
        for (int l = 0; l < 32; ++l) {
            float s = 0.0f;
            for (int i = w * 32 + l; i < n; i += threads)
                s = fma32(x[i], x[i], s);
            lanes[l] = s;
        }
        warps[w] = tree32(lanes);
    }
    return threads == 32 ? warps[0] : tree32(warps);
}

__device__ float grouped_dot(const at::BFloat16* w, int64_t base, const float* x, int n) {
    float lanes[32];
    for (int l = 0; l < 32; ++l) {
        float s = 0.0f;
        for (int k = l * 8; k < n; k += 256) {
            float part = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                part = fma32(float(w[base + k + j]), x[k + j], part);
            s = add32(s, part);
        }
        lanes[l] = s;
    }
    return tree32(lanes);
}

__device__ float continuous_dot(const at::BFloat16* w, int64_t base, const float* x, int n) {
    float lanes[32];
    for (int l = 0; l < 32; ++l) {
        float s = 0.0f;
        for (int k = l * 8; k < n; k += 256) {
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                s = fma32(float(w[base + k + j]), x[k + j], s);
        }
        lanes[l] = s;
    }
    return tree32(lanes);
}

__device__ bool better(float v, int i, float best, int bi) {
    return i >= 0 && (bi < 0 || v > best || (isnan(v) && !isnan(best)));
}

__global__ void initialize(
    const int64_t* prompt, float* states, float* conv, at::BFloat16* keys, at::BFloat16* values,
    int64_t* tokens, at::BFloat16* history, int64_t steps) {
    for (int i = 0; i < 4718592; ++i) states[i] = 0.0f;
    for (int i = 0; i < 442368; ++i) conv[i] = 0.0f;
    for (int i = 0; i < 6291456; ++i) {
        keys[i] = at::BFloat16(0.0f);
        values[i] = at::BFloat16(0.0f);
    }
    for (int i = 0; i < steps; ++i) tokens[i] = 0;
    // Carry the first prompt token until the first generated decision replaces it.
    tokens[0] = prompt[0];
    for (int i = 0; i < steps * 1024; ++i) history[i] = at::BFloat16(0.0f);
}

__global__ void decode_step(
    const at::BFloat16* weights, const int64_t* prompt,
    const float* state_in, const float* conv_in,
    const at::BFloat16* key_in, const at::BFloat16* value_in,
    const int64_t* tokens_in, const at::BFloat16* history_in,
    float* state_out, float* conv_out,
    at::BFloat16* key_out, at::BFloat16* value_out,
    int64_t* tokens_out, at::BFloat16* history_out,
    int64_t prompt_length, int64_t steps, int position) {
    for (int i = 0; i < 6291456; ++i) {
        key_out[i] = key_in[i];
        value_out[i] = value_in[i];
    }
    for (int i = 0; i < steps; ++i) tokens_out[i] = tokens_in[i];
    for (int i = 0; i < steps * 1024; ++i) history_out[i] = history_in[i];
    if (position >= prompt_length + steps - 1) {
        for (int i = 0; i < 4718592; ++i) state_out[i] = state_in[i];
        for (int i = 0; i < 442368; ++i) conv_out[i] = conv_in[i];
        return;
    }
    int64_t token = position == 0 ? tokens_in[0] :
        (position < prompt_length ? prompt[position] : tokens_in[position - prompt_length]);
    float hidden[1024], norm[1024], projection[6144], z[2048], out[2048], mlp[3584];
    float cosine[32], sine[32];
    for (int i = 0; i < 1024; ++i) hidden[i] = float(weights[token * 1024 + i]);
    // Constant-folded approximate powers differ from the original runtime instructions.
    #pragma unroll 1
    for (int i = 0; i < 32; ++i) {
        float frequency = divide32(float(position),
            exp2_32(mul32(divide32(float(2 * i), 64.0f), log2_32(10000000.0f))));
        asm("cos.approx.ftz.f32 %0, %1;" : "=f"(cosine[i]) : "f"(frequency));
        asm("sin.approx.ftz.f32 %0, %1;" : "=f"(sine[i]) : "f"(frequency));
    }
    int64_t base = 254280704;
    int dn = 0, fa = 0;
    for (int layer = 0; layer < 24; ++layer) {
        int64_t post_norm = base + 1024, gate_weight = base + 2048;
        int64_t up_weight = gate_weight + 3670016, down_weight = up_weight + 3670016;
        int64_t extra = down_weight + 3670016, out_weight;
        float scale = rsqrt32(add32(divide32(square_sum(hidden, 1024, 512), 1024.0f), 1e-6f));
        for (int i = 0; i < 1024; ++i)
            norm[i] = float(at::BFloat16(mul32(mul32(hidden[i], scale), add32(1.0f, float(weights[base + i])))));
        if (layer % 4 != 3) {
            int64_t z_weight = extra + 6291456, beta_weight = z_weight + 2097152;
            int64_t alpha_weight = beta_weight + 16384, conv_weight = alpha_weight + 16384;
            int64_t a_weight = conv_weight + 24576, dt_weight = a_weight + 16;
            int64_t gated_norm = dt_weight + 16;
            out_weight = gated_norm + 128;
            for (int i = 0; i < 6144; ++i)
                projection[i] = grouped_dot(weights, extra + int64_t(i) * 1024, norm, 1024);
            for (int i = 0; i < 2048; ++i)
                z[i] = grouped_dot(weights, z_weight + int64_t(i) * 1024, norm, 1024);
            for (int h = 0; h < 16; ++h) {
                float beta = sigmoid32(grouped_dot(weights, beta_weight + h * 1024, norm, 1024));
                float a = add32(grouped_dot(weights, alpha_weight + h * 1024, norm, 1024), float(weights[dt_weight + h]));
                float softplus = a > 20.0f ? a : mul32(log2_32(add32(1.0f, exp32(a))), 0.6931471805599453f);
                float decay = exp32(-mul32(exp32(float(weights[a_weight + h])), softplus));
                float q[128], k[128], v[128], pk[32], pq[32];
                for (int region = 0; region < 3; ++region) {
                    for (int i = 0; i < 128; ++i) {
                        int channel = region * 2048 + h * 128 + i;
                        int ci = dn * 24576 + channel * 4;
                        float cv[4] = {conv_in[ci + 1], conv_in[ci + 2], conv_in[ci + 3], projection[channel]};
                        float s = 0.0f;
                        for (int t = 0; t < 4; ++t) {
                            conv_out[ci + t] = cv[t];
                            s = fma32(cv[t], float(weights[conv_weight + channel * 4 + t]), s);
                        }
                        if (region == 0) q[i] = silu32(s);
                        else if (region == 1) k[i] = silu32(s);
                        else v[i] = silu32(s);
                    }
                }
                float qs = mul32(rsqrt32(add32(square_sum(q, 128, 32), 1e-6f)), 1.0f / 11.313708498984761f);
                float ks = rsqrt32(add32(square_sum(k, 128, 32), 1e-6f));
                for (int i = 0; i < 128; ++i) {
                    q[i] = mul32(q[i], qs);
                    k[i] = mul32(k[i], ks);
                }
                for (int l = 0; l < 32; ++l) {
                    float s = 0.0f;
                    for (int i = l; i < 128; i += 32) s = fma32(k[i], q[i], s);
                    pk[l] = s;
                }
                float kq = tree32(pk);
                for (int j = 0; j < 128; ++j) {
                    int si = dn * 262144 + h * 16384 + j * 128;
                    for (int l = 0; l < 32; ++l) {
                        float sk = 0.0f, sq = 0.0f;
                        for (int i = l; i < 128; i += 32) {
                            float s = state_in[si + i];
                            sk = fma32(s, k[i], sk);
                            sq = fma32(s, q[i], sq);
                        }
                        pk[l] = sk;
                        pq[l] = sq;
                    }
                    float error = mul32(sub32(v[j], mul32(decay, tree32(pk))), beta);
                    out[h * 128 + j] = fma32(decay, tree32(pq), mul32(error, kq));
                    for (int i = 0; i < 128; ++i)
                        state_out[si + i] = fma32(state_in[si + i], decay, mul32(k[i], error));
                }
                float rstd = rsqrt32(add32(divide32(square_sum(out + h * 128, 128, 512), 128.0f), 1e-6f));
                for (int i = 0; i < 128; ++i)
                    out[h * 128 + i] = mul32(mul32(mul32(out[h * 128 + i], rstd), float(weights[gated_norm + i])), silu32(z[h * 128 + i]));
            }
            ++dn;
        } else {
            int64_t k_weight = extra + 4194304, v_weight = k_weight + 524288;
            int64_t q_norm = v_weight + 524288, k_norm = q_norm + 256;
            out_weight = k_norm + 256;
            for (int i = 0; i < 4096; ++i)
                projection[i] = grouped_dot(weights, extra + int64_t(i) * 1024, norm, 1024);
            for (int i = 0; i < 512; ++i) {
                z[i] = grouped_dot(weights, k_weight + int64_t(i) * 1024, norm, 1024);
                z[512 + i] = grouped_dot(weights, v_weight + int64_t(i) * 1024, norm, 1024);
            }
            at::BFloat16 current_key[512], current_value[512];
            for (int h = 0; h < 10; ++h) {
                float* head = h < 2 ? z + h * 256 : projection + (h - 2) * 512;
                int64_t weight = h < 2 ? k_norm : q_norm;
                float rstd = rsqrt32(add32(divide32(square_sum(head, 256, 32), 256.0f), 1e-6f));
                for (int i = 0; i < 256; ++i)
                    head[i] = mul32(mul32(head[i], rstd), add32(1.0f, float(weights[weight + i])));
                for (int i = 0; i < 32; ++i) {
                    float x0 = head[i], x1 = head[i + 32];
                    head[i] = sub32(mul32(x0, cosine[i]), mul32(x1, sine[i]));
                    head[i + 32] = fma32(x0, sine[i], mul32(x1, cosine[i]));
                }
                if (h < 2) {
                    for (int i = 0; i < 256; ++i) {
                        current_key[h * 256 + i] = at::BFloat16(head[i]);
                        current_value[h * 256 + i] = at::BFloat16(z[512 + h * 256 + i]);
                        int ci = fa * 1048576 + h * 524288 + position * 256 + i;
                        key_out[ci] = current_key[h * 256 + i];
                        value_out[ci] = current_value[h * 256 + i];
                    }
                }
            }
            for (int h = 0; h < 8; ++h) {
                float partial[4096], maximum[16], total[16], lanes[32], factor[16];
                int kvh = h / 4, cb = fa * 1048576 + kvh * 524288;
                for (int w = 0; w < 16; ++w) {
                    float mx = -INFINITY, sum = 0.0f;
                    for (int i = 0; i < 256; ++i) partial[w * 256 + i] = 0.0f;
                    for (int p = w; p <= position; p += 16) {
                        for (int l = 0; l < 32; ++l) {
                            float s = 0.0f;
                            for (int e = 0; e < 8; ++e) {
                                int i = l * 8 + e;
                                float key = float(p == position ? current_key[kvh * 256 + i] : key_in[cb + p * 256 + i]);
                                s = fma32(projection[h * 512 + i], key, s);
                            }
                            lanes[l] = s;
                        }
                        float score = mul32(tree32(lanes), 0.0625f), old = mx;
                        mx = fmaxf(mx, score);
                        float diff = exp32(sub32(old, mx)), weight = exp32(sub32(score, mx));
                        sum = fma32(sum, diff, weight);
                        for (int i = 0; i < 256; ++i) {
                            float val = float(p == position ? current_value[kvh * 256 + i] : value_in[cb + p * 256 + i]);
                            partial[w * 256 + i] = fma32(partial[w * 256 + i], diff, mul32(weight, val));
                        }
                    }
                    maximum[w] = mx;
                    total[w] = sum;
                }
                float gm = -INFINITY, ts = 0.0f;
                for (int w = 0; w < 16; ++w)
                    if (maximum[w] > -INFINITY) gm = fmaxf(gm, maximum[w]);
                for (int w = 0; w < 16; ++w) {
                    if (maximum[w] > -INFINITY) {
                        factor[w] = exp32(sub32(maximum[w], gm));
                        ts = fma32(total[w], factor[w], ts);
                    }
                }
                float rcp = reciprocal32(ts);
                for (int i = 0; i < 256; ++i) {
                    float s = 0.0f;
                    for (int w = 0; w < 16; ++w)
                        if (maximum[w] > -INFINITY) s = fma32(partial[w * 256 + i], factor[w], s);
                    out[h * 256 + i] = mul32(mul32(s, rcp), sigmoid32(projection[h * 512 + 256 + i]));
                }
            }
            ++fa;
        }
        for (int i = 0; i < 1024; ++i)
            hidden[i] = float(at::BFloat16(add32(continuous_dot(weights, out_weight + int64_t(i) * 2048, out, 2048), hidden[i])));
        scale = rsqrt32(add32(divide32(square_sum(hidden, 1024, 512), 1024.0f), 1e-6f));
        for (int i = 0; i < 1024; ++i)
            norm[i] = float(at::BFloat16(mul32(mul32(hidden[i], scale), add32(1.0f, float(weights[post_norm + i])))));
        for (int i = 0; i < 3584; ++i) {
            float gate = grouped_dot(weights, gate_weight + int64_t(i) * 1024, norm, 1024);
            float up = grouped_dot(weights, up_weight + int64_t(i) * 1024, norm, 1024);
            mlp[i] = mul32(silu32(gate), up);
        }
        for (int i = 0; i < 1024; ++i)
            hidden[i] = float(at::BFloat16(add32(continuous_dot(weights, down_weight + int64_t(i) * 3584, mlp, 3584), hidden[i])));
        base = out_weight + 2097152;
    }
    if (position >= prompt_length - 1) {
        int decision = position - prompt_length + 1;
        float scale = rsqrt32(add32(divide32(square_sum(hidden, 1024, 512), 1024.0f), 1e-6f));
        for (int i = 0; i < 1024; ++i) {
            history_out[decision * 1024 + i] = at::BFloat16(hidden[i]);
            norm[i] = mul32(mul32(hidden[i], scale), add32(1.0f, float(weights[254279680 + i])));
        }
        float block_value[512], lane_value[32];
        int block_index[512], lane_index[32];
        for (int b = 0; b < 512; ++b) {
            for (int w = 0; w < 32; ++w) {
                float best = -INFINITY;
                int index = -1;
                if (w < 8) {
                    for (int m = b * 485 + w; m < (b + 1) * 485; m += 8) {
                        float s = continuous_dot(weights, int64_t(m) * 1024, norm, 1024);
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
    }
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1, int64_t input2) {
    auto state0 = torch::empty({18, 16, 128, 128}, input0.options().dtype(at::kFloat));
    auto state1 = torch::empty({18, 16, 128, 128}, input0.options().dtype(at::kFloat));
    auto conv0 = torch::empty({18, 6144, 4}, input0.options().dtype(at::kFloat));
    auto conv1 = torch::empty({18, 6144, 4}, input0.options().dtype(at::kFloat));
    auto key0 = torch::empty({6, 2, 2048, 256}, input0.options());
    auto key1 = torch::empty({6, 2, 2048, 256}, input0.options());
    auto value0 = torch::empty({6, 2, 2048, 256}, input0.options());
    auto value1 = torch::empty({6, 2, 2048, 256}, input0.options());
    auto tokens0 = torch::empty({input2}, input1.options());
    auto tokens1 = torch::empty({input2}, input1.options());
    auto history0 = torch::empty({input2, 1024}, input0.options());
    auto history1 = torch::empty({input2, 1024}, input0.options());
    int64_t prompt_length = input1.numel();
    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    initialize<<<1, 1, 0, stream>>>(input1.data_ptr<int64_t>(), state0.data_ptr<float>(), conv0.data_ptr<float>(), key0.data_ptr<at::BFloat16>(), value0.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), input2);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_step<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), state0.data_ptr<float>(), conv0.data_ptr<float>(), key0.data_ptr<at::BFloat16>(), value0.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), state1.data_ptr<float>(), conv1.data_ptr<float>(), key1.data_ptr<at::BFloat16>(), value1.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), prompt_length, input2, 0);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_step<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), state1.data_ptr<float>(), conv1.data_ptr<float>(), key1.data_ptr<at::BFloat16>(), value1.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), state0.data_ptr<float>(), conv0.data_ptr<float>(), key0.data_ptr<at::BFloat16>(), value0.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), prompt_length, input2, 1);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_step<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), state0.data_ptr<float>(), conv0.data_ptr<float>(), key0.data_ptr<at::BFloat16>(), value0.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), state1.data_ptr<float>(), conv1.data_ptr<float>(), key1.data_ptr<at::BFloat16>(), value1.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), prompt_length, input2, 2);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_step<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), state1.data_ptr<float>(), conv1.data_ptr<float>(), key1.data_ptr<at::BFloat16>(), value1.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), state0.data_ptr<float>(), conv0.data_ptr<float>(), key0.data_ptr<at::BFloat16>(), value0.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), prompt_length, input2, 3);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_step<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), state0.data_ptr<float>(), conv0.data_ptr<float>(), key0.data_ptr<at::BFloat16>(), value0.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), state1.data_ptr<float>(), conv1.data_ptr<float>(), key1.data_ptr<at::BFloat16>(), value1.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), prompt_length, input2, 4);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_step<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), state1.data_ptr<float>(), conv1.data_ptr<float>(), key1.data_ptr<at::BFloat16>(), value1.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), state0.data_ptr<float>(), conv0.data_ptr<float>(), key0.data_ptr<at::BFloat16>(), value0.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), prompt_length, input2, 5);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_step<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), state0.data_ptr<float>(), conv0.data_ptr<float>(), key0.data_ptr<at::BFloat16>(), value0.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), state1.data_ptr<float>(), conv1.data_ptr<float>(), key1.data_ptr<at::BFloat16>(), value1.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), prompt_length, input2, 6);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_step<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), state1.data_ptr<float>(), conv1.data_ptr<float>(), key1.data_ptr<at::BFloat16>(), value1.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), state0.data_ptr<float>(), conv0.data_ptr<float>(), key0.data_ptr<at::BFloat16>(), value0.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), prompt_length, input2, 7);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_step<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), state0.data_ptr<float>(), conv0.data_ptr<float>(), key0.data_ptr<at::BFloat16>(), value0.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), state1.data_ptr<float>(), conv1.data_ptr<float>(), key1.data_ptr<at::BFloat16>(), value1.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), prompt_length, input2, 8);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_step<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), state1.data_ptr<float>(), conv1.data_ptr<float>(), key1.data_ptr<at::BFloat16>(), value1.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), state0.data_ptr<float>(), conv0.data_ptr<float>(), key0.data_ptr<at::BFloat16>(), value0.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), prompt_length, input2, 9);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_step<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), state0.data_ptr<float>(), conv0.data_ptr<float>(), key0.data_ptr<at::BFloat16>(), value0.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), state1.data_ptr<float>(), conv1.data_ptr<float>(), key1.data_ptr<at::BFloat16>(), value1.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), prompt_length, input2, 10);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    decode_step<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), state1.data_ptr<float>(), conv1.data_ptr<float>(), key1.data_ptr<at::BFloat16>(), value1.data_ptr<at::BFloat16>(), tokens1.data_ptr<int64_t>(), history1.data_ptr<at::BFloat16>(), state0.data_ptr<float>(), conv0.data_ptr<float>(), key0.data_ptr<at::BFloat16>(), value0.data_ptr<at::BFloat16>(), tokens0.data_ptr<int64_t>(), history0.data_ptr<at::BFloat16>(), prompt_length, input2, 11);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {tokens0, history0};
}
