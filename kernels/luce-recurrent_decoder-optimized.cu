#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
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

__global__ void recurrent_decoder(
    const at::BFloat16* weights, const float* hidden_in, const float* state_in, const float* conv_in,
    float* hidden_out, float* state_out, float* conv_out, int64_t base, int dn) {
    for (int i = 0; i < 4718592; ++i) state_out[i] = state_in[i];
    for (int i = 0; i < 442368; ++i) conv_out[i] = conv_in[i];
    float hidden[1024], norm[1024], projection[6144], z[2048], out[2048], mlp[3584];
    for (int i = 0; i < 1024; ++i) hidden[i] = hidden_in[i];
    int64_t post_norm = base + 1024, gate_weight = base + 2048;
    int64_t up_weight = gate_weight + 3670016, down_weight = up_weight + 3670016;
    int64_t extra = down_weight + 3670016, out_weight;
    float scale = rsqrt32(add32(divide32(square_sum(hidden, 1024, 512), 1024.0f), 1e-6f));
    for (int i = 0; i < 1024; ++i)
        norm[i] = float(at::BFloat16(mul32(mul32(hidden[i], scale), add32(1.0f, float(weights[base + i])))));
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
    for (int i = 0; i < 1024; ++i) hidden_out[i] = hidden[i];
}

std::vector<torch::Tensor> kplay_run(
    torch::Tensor input0, torch::Tensor input1, torch::Tensor input2, torch::Tensor input3,
    int64_t input4, int64_t input5) {
    TORCH_CHECK(input0.is_cuda() && input0.is_contiguous() && input0.scalar_type() == at::kBFloat16 &&
                input4 >= 0 && input0.numel() >= 21555360 && input4 <= input0.numel() - 21555360,
                "weights must contain a complete packed recurrent layer at the specified offset");
    for (const auto& tensor : {input1, input2, input3})
        TORCH_CHECK(tensor.device() == input0.device() && tensor.is_contiguous() &&
                    tensor.scalar_type() == at::kFloat, "hidden/recurrent/convolution state must be CUDA float32");
    TORCH_CHECK(input1.sizes() == at::IntArrayRef({1024}) &&
                input2.sizes() == at::IntArrayRef({18, 16, 128, 128}) &&
                input3.sizes() == at::IntArrayRef({18, 6144, 4}) &&
                input5 >= 0 && input5 < 18, "invalid recurrent-layer state shape or slot");
    c10::cuda::CUDAGuard guard(input0.device());
    auto hidden = torch::empty({1024}, input0.options().dtype(at::kFloat));
    auto state = torch::empty({18, 16, 128, 128}, input0.options().dtype(at::kFloat));
    auto conv = torch::empty({18, 6144, 4}, input0.options().dtype(at::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream();
    recurrent_decoder<<<1, 1, 0, stream>>>(
        input0.data_ptr<at::BFloat16>(), input1.data_ptr<float>(), input2.data_ptr<float>(),
        input3.data_ptr<float>(), hidden.data_ptr<float>(), state.data_ptr<float>(),
        conv.data_ptr<float>(), input4, int(input5));
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {hidden, state, conv};
}
