#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include "KPlay_Functional.h"

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1, torch::Tensor input2, int64_t input3) {
  TORCH_CHECK(input0.is_cuda(), "Kuiper requires CUDA tensors");
  TORCH_CHECK(input1.device() == input0.device(), "Kuiper tensors must share one CUDA device");
  TORCH_CHECK(input2.device() == input0.device(), "Kuiper tensors must share one CUDA device");
  TORCH_CHECK(input0.is_contiguous(), "Kuiper requires contiguous typed storage");
  TORCH_CHECK(input0.dim() == 2, "Kuiper input rank mismatch");
  TORCH_CHECK(input0.size(0) == (248320), "Kuiper input extent mismatch");
  TORCH_CHECK(input0.size(1) == (1024), "Kuiper input extent mismatch");
  TORCH_CHECK(input1.is_contiguous(), "Kuiper requires contiguous typed storage");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(input1.data_ptr()) % 2 == 0, "Kuiper input alignment mismatch");
  TORCH_CHECK(input2.is_contiguous(), "Kuiper requires contiguous typed storage");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(input2.data_ptr()) % 8 == 0, "Kuiper input alignment mismatch");
  const c10::cuda::CUDAGuard device_guard(input0.device());
  auto output0 = torch::empty({1024}, input0.options().dtype(torch::kBFloat16));
  auto output1 = torch::empty({32}, input0.options().dtype(torch::kBFloat16));
  auto output2 = torch::empty({32}, input0.options().dtype(torch::kBFloat16));
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  KPlay_Functional_kplay_run(static_cast<uint32_t>(input0.numel()), static_cast<uint32_t>(input1.numel()), static_cast<uint32_t>(input2.numel()), reinterpret_cast<__nv_bfloat16*>(input0.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(input1.data_ptr<at::BFloat16>()), input2.data_ptr<int64_t>(), static_cast<int64_t>(input3), reinterpret_cast<__nv_bfloat16*>(output0.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(output1.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(output2.data_ptr<at::BFloat16>()));
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  return {output0, output1, output2};
}

#pragma push_macro("KRML_CHECK_SIZE")
#undef KRML_CHECK_SIZE
#include "krml/internal/target.h"
#pragma pop_macro("KRML_CHECK_SIZE")
#include "KPlay_Functional.cu"
