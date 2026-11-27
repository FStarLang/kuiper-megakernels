#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include "KPlay_Functional.h"

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1, torch::Tensor input2, torch::Tensor input3, torch::Tensor input4, torch::Tensor input5, int64_t input6, int64_t input7, int64_t input8) {
  TORCH_CHECK(input1.is_cuda(), "Kuiper requires CUDA tensors");
  TORCH_CHECK(input0.device() == input1.device(), "Kuiper tensors must share one CUDA device");
  TORCH_CHECK(input2.device() == input1.device(), "Kuiper tensors must share one CUDA device");
  TORCH_CHECK(input3.device() == input1.device(), "Kuiper tensors must share one CUDA device");
  TORCH_CHECK(input4.device() == input1.device(), "Kuiper tensors must share one CUDA device");
  TORCH_CHECK(input5.device() == input1.device(), "Kuiper tensors must share one CUDA device");
  TORCH_CHECK(input0.is_contiguous(), "Kuiper requires contiguous typed storage");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(input0.data_ptr()) % 2 == 0, "Kuiper input alignment mismatch");
  TORCH_CHECK(input1.is_contiguous(), "Kuiper requires contiguous typed storage");
  TORCH_CHECK(input1.dim() == 1, "Kuiper input rank mismatch");
  TORCH_CHECK(input1.size(0) == (1024), "Kuiper input extent mismatch");
  TORCH_CHECK(input2.is_contiguous(), "Kuiper requires contiguous typed storage");
  TORCH_CHECK(input2.dim() == 4, "Kuiper input rank mismatch");
  TORCH_CHECK(input2.size(0) == (6), "Kuiper input extent mismatch");
  TORCH_CHECK(input2.size(1) == (2), "Kuiper input extent mismatch");
  TORCH_CHECK(input2.size(2) == (2048), "Kuiper input extent mismatch");
  TORCH_CHECK(input2.size(3) == (256), "Kuiper input extent mismatch");
  TORCH_CHECK(input3.is_contiguous(), "Kuiper requires contiguous typed storage");
  TORCH_CHECK(input3.dim() == 4, "Kuiper input rank mismatch");
  TORCH_CHECK(input3.size(0) == (6), "Kuiper input extent mismatch");
  TORCH_CHECK(input3.size(1) == (2), "Kuiper input extent mismatch");
  TORCH_CHECK(input3.size(2) == (2048), "Kuiper input extent mismatch");
  TORCH_CHECK(input3.size(3) == (256), "Kuiper input extent mismatch");
  TORCH_CHECK(input4.is_contiguous(), "Kuiper requires contiguous typed storage");
  TORCH_CHECK(input4.dim() == 1, "Kuiper input rank mismatch");
  TORCH_CHECK(input4.size(0) == (32), "Kuiper input extent mismatch");
  TORCH_CHECK(input5.is_contiguous(), "Kuiper requires contiguous typed storage");
  TORCH_CHECK(input5.dim() == 1, "Kuiper input rank mismatch");
  TORCH_CHECK(input5.size(0) == (32), "Kuiper input extent mismatch");
  const c10::cuda::CUDAGuard device_guard(input1.device());
  auto output0 = torch::empty({1024}, input1.options().dtype(torch::kFloat32));
  auto output1 = torch::empty({6, 2, 2048, 256}, input0.options().dtype(torch::kBFloat16));
  auto output2 = torch::empty({6, 2, 2048, 256}, input0.options().dtype(torch::kBFloat16));
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  KPlay_Functional_kplay_run(reinterpret_cast<__nv_bfloat16*>(input0.data_ptr<at::BFloat16>()), input1.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(input2.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(input3.data_ptr<at::BFloat16>()), input4.data_ptr<float>(), input5.data_ptr<float>(), output0.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(output1.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(output2.data_ptr<at::BFloat16>()), static_cast<uint32_t>(input0.numel()), static_cast<int64_t>(input6), static_cast<int64_t>(input7), static_cast<int64_t>(input8));
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  return {output0, output1, output2};
}

#pragma push_macro("KRML_CHECK_SIZE")
#undef KRML_CHECK_SIZE
#include "krml/internal/target.h"
#pragma pop_macro("KRML_CHECK_SIZE")
#include "KPlay_Functional.cu"
