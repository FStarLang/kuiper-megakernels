#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

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

__global__ void embedding(
    const at::BFloat16* weights, const int64_t* prompt, const int64_t* tokens_in,
    float* hidden, float* cosine, float* sine, int64_t prompt_length, int position) {
    int64_t token = position == 0 ? tokens_in[0] :
        (position < prompt_length ? prompt[position] : tokens_in[position - prompt_length]);
    for (int i = 0; i < 1024; ++i) hidden[i] = float(weights[token * 1024 + i]);
    // Keep these powers as runtime PTX, as in the accepted program.
    #pragma unroll 1
    for (int i = 0; i < 32; ++i) {
        float frequency = divide32(float(position),
            exp2_32(mul32(divide32(float(2 * i), 64.0f), log2_32(10000000.0f))));
        asm("cos.approx.ftz.f32 %0, %1;" : "=f"(cosine[i]) : "f"(frequency));
        asm("sin.approx.ftz.f32 %0, %1;" : "=f"(sine[i]) : "f"(frequency));
    }
}

std::vector<torch::Tensor> kplay_run(
    torch::Tensor input0, torch::Tensor input1, torch::Tensor input2, int64_t input3) {
    TORCH_CHECK(input0.is_cuda() && input0.is_contiguous() &&
                input0.scalar_type() == at::kBFloat16 && input0.numel() == 254279680,
                "embedding table must contain 248320 x 1024 contiguous CUDA bfloat16 values");
    TORCH_CHECK(input1.device() == input0.device() && input2.device() == input0.device() &&
                input1.is_contiguous() && input2.is_contiguous() &&
                input1.scalar_type() == at::kLong && input2.scalar_type() == at::kLong &&
                input1.dim() == 1 && input2.dim() == 1 &&
                input1.numel() > 0 && input2.numel() > 0,
                "prompt and generated-token state must be nonempty CUDA int64 vectors");
    TORCH_CHECK(input3 >= 0 && input3 < 12 && input3 < input1.numel() + input2.numel() - 1,
                "embedding requires an active accepted decode position");
    c10::cuda::CUDAGuard guard(input0.device());
    auto hidden = torch::empty({1024}, input0.options().dtype(at::kFloat));
    auto cosine = torch::empty({32}, input0.options().dtype(at::kFloat));
    auto sine = torch::empty({32}, input0.options().dtype(at::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream();
    embedding<<<1, 1, 0, stream>>>(
        input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), input2.data_ptr<int64_t>(),
        hidden.data_ptr<float>(), cosine.data_ptr<float>(), sine.data_ptr<float>(),
        input1.numel(), int(input3));
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {hidden, cosine, sine};
}
