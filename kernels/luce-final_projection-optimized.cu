#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
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

__device__ float rsqrt32(float x) {
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
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

__global__ void final_projection(
    const at::BFloat16* weights, const at::BFloat16* norm_weight, const float* hidden,
    const at::BFloat16* history_in, float* scores, at::BFloat16* history_out,
    int64_t steps, int decision) {
    for (int i = 0; i < steps * 1024; ++i) history_out[i] = history_in[i];
    float norm[1024];
    float scale = rsqrt32(add32(divide32(square_sum(hidden, 1024, 512), 1024.0f), 1e-6f));
    for (int i = 0; i < 1024; ++i) {
        history_out[decision * 1024 + i] = at::BFloat16(hidden[i]);
        norm[i] = mul32(mul32(hidden[i], scale), add32(1.0f, float(norm_weight[i])));
    }
    for (int b = 0; b < 512; ++b) {
        for (int w = 0; w < 32; ++w) {
            if (w < 8) {
                for (int m = b * 485 + w; m < (b + 1) * 485; m += 8) {
                    float s = continuous_dot(weights, int64_t(m) * 1024, norm, 1024);
                    scores[m] = s;
                }
            }
        }
    }
}

std::vector<torch::Tensor> kplay_run(
    torch::Tensor input0, torch::Tensor input1, torch::Tensor input2,
    torch::Tensor input3, int64_t input4) {
    TORCH_CHECK(input0.is_cuda() && input0.is_contiguous() &&
                input0.scalar_type() == at::kBFloat16 && input0.numel() == 254279680,
                "projection table must contain 248320 x 1024 contiguous CUDA bfloat16 values");
    for (const auto& tensor : {input1, input3})
        TORCH_CHECK(tensor.device() == input0.device() && tensor.is_contiguous() &&
                    tensor.scalar_type() == at::kBFloat16, "norm weights and history must be CUDA bfloat16");
    TORCH_CHECK(input1.sizes() == at::IntArrayRef({1024}) &&
                input2.device() == input0.device() && input2.is_contiguous() &&
                input2.scalar_type() == at::kFloat && input2.sizes() == at::IntArrayRef({1024}) &&
                input3.dim() == 2 && input3.size(1) == 1024 &&
                input4 >= 0 && input4 < input3.size(0),
                "invalid final-normalization shape or history decision offset");
    c10::cuda::CUDAGuard guard(input0.device());
    auto scores = torch::empty({248320}, input0.options().dtype(at::kFloat));
    auto history = torch::empty({input3.size(0), 1024}, input0.options().dtype(at::kBFloat16));
    auto stream = at::cuda::getCurrentCUDAStream();
    final_projection<<<1, 1, 0, stream>>>(
        input0.data_ptr<at::BFloat16>(), input1.data_ptr<at::BFloat16>(), input2.data_ptr<float>(),
        input3.data_ptr<at::BFloat16>(), scores.data_ptr<float>(), history.data_ptr<at::BFloat16>(),
        input3.size(0), int(input4));
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {scores, history};
}
