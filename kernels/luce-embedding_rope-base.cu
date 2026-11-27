#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>
#include <cstdint>
#include <vector>


void check_tensor(const torch::Tensor& value, at::ScalarType dtype, int64_t count,
                  const torch::Tensor& device) {
    TORCH_CHECK(value.is_cuda() && value.is_contiguous() && value.scalar_type() == dtype
                && value.device() == device.device() && value.numel() == count,
                "Expected a contiguous CUDA tensor with the declared dtype, size and device");
}

__global__ void embedding_rope(const at::BFloat16* weights, const at::BFloat16* embeddings,
                               const int64_t* tokens, at::BFloat16* residual,
                               at::BFloat16* cosine, at::BFloat16* sine,
                               int prompt_size, int position) {
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

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                  torch::Tensor input2, int64_t input3) {
    check_tensor(input0, at::kBFloat16, 254279680, input0);
    check_tensor(input1, at::kBFloat16, input1.numel(), input0);
    check_tensor(input2, at::kLong, input2.numel(), input0);
    TORCH_CHECK(input1.dim() == 2 && input1.size(1) == 1024 && input1.size(0) >= 1
                && input2.dim() == 1 && input2.numel() >= 1 && input3 >= 0
                && input3 < input1.size(0) + input2.numel() - 1 && input3 < 12,
                "Embedding position is outside the active decode range");
    auto residual = torch::empty({1024}, input0.options());
    auto cosine = torch::empty({32}, input0.options());
    auto sine = torch::empty({32}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    embedding_rope<<<1, 1, 0, stream>>>(
        input0.data_ptr<at::BFloat16>(), input1.data_ptr<at::BFloat16>(),
        input2.data_ptr<int64_t>(), residual.data_ptr<at::BFloat16>(),
        cosine.data_ptr<at::BFloat16>(), sine.data_ptr<at::BFloat16>(),
        input1.size(0), input3);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {residual, cosine, sine};
}
