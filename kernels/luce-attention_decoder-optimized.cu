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

__global__ void attention_decoder(
    const at::BFloat16* weights, const float* hidden_in,
    const at::BFloat16* key_in, const at::BFloat16* value_in,
    const float* cosine, const float* sine, float* hidden_out,
    at::BFloat16* key_out, at::BFloat16* value_out, int64_t base, int fa, int position) {
    for (int i = 0; i < 6291456; ++i) {
        key_out[i] = key_in[i];
        value_out[i] = value_in[i];
    }
    float hidden[1024], norm[1024], projection[6144], z[2048], out[2048], mlp[3584];
    for (int i = 0; i < 1024; ++i) hidden[i] = hidden_in[i];
    int64_t post_norm = base + 1024, gate_weight = base + 2048;
    int64_t up_weight = gate_weight + 3670016, down_weight = up_weight + 3670016;
    int64_t extra = down_weight + 3670016, out_weight;
    float scale = rsqrt32(add32(divide32(square_sum(hidden, 1024, 512), 1024.0f), 1e-6f));
    for (int i = 0; i < 1024; ++i)
        norm[i] = float(at::BFloat16(mul32(mul32(hidden[i], scale), add32(1.0f, float(weights[base + i])))));
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
    torch::Tensor input4, torch::Tensor input5, int64_t input6, int64_t input7, int64_t input8) {
    TORCH_CHECK(input0.is_cuda() && input0.is_contiguous() && input0.scalar_type() == at::kBFloat16 &&
                input6 >= 0 && input0.numel() >= 18352640 && input6 <= input0.numel() - 18352640,
                "weights must contain a complete packed attention layer at the specified offset");
    for (const auto& tensor : {input1, input4, input5})
        TORCH_CHECK(tensor.device() == input0.device() && tensor.is_contiguous() &&
                    tensor.scalar_type() == at::kFloat, "hidden and rotary inputs must be CUDA float32");
    for (const auto& tensor : {input2, input3})
        TORCH_CHECK(tensor.device() == input0.device() && tensor.is_contiguous() &&
                    tensor.scalar_type() == at::kBFloat16 &&
                    tensor.sizes() == at::IntArrayRef({6, 2, 2048, 256}),
                    "KV state must be contiguous CUDA bfloat16 [6,2,2048,256]");
    TORCH_CHECK(input1.sizes() == at::IntArrayRef({1024}) &&
                input4.sizes() == at::IntArrayRef({32}) && input5.sizes() == input4.sizes() &&
                input7 >= 0 && input7 < 6 && input8 >= 0 && input8 < 2048,
                "invalid attention hidden/rotary shape, KV slot, or sequence position");
    c10::cuda::CUDAGuard guard(input0.device());
    auto hidden = torch::empty({1024}, input0.options().dtype(at::kFloat));
    auto keys = torch::empty({6, 2, 2048, 256}, input0.options().dtype(at::kBFloat16));
    auto values = torch::empty({6, 2, 2048, 256}, input0.options().dtype(at::kBFloat16));
    auto stream = at::cuda::getCurrentCUDAStream();
    attention_decoder<<<1, 1, 0, stream>>>(
        input0.data_ptr<at::BFloat16>(), input1.data_ptr<float>(), input2.data_ptr<at::BFloat16>(),
        input3.data_ptr<at::BFloat16>(), input4.data_ptr<float>(), input5.data_ptr<float>(),
        hidden.data_ptr<float>(), keys.data_ptr<at::BFloat16>(), values.data_ptr<at::BFloat16>(),
        input6, int(input7), int(input8));
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {hidden, keys, values};
}
