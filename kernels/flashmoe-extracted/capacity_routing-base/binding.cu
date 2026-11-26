#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include "KPlay_Functional.h"

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1, torch::Tensor input2, torch::Tensor input3, torch::Tensor input4, torch::Tensor input5, int64_t input6) {
  TORCH_CHECK(input2.is_cuda(), "Kuiper requires CUDA tensors");
  TORCH_CHECK(input0.device() == input2.device(), "Kuiper tensors must share one CUDA device");
  TORCH_CHECK(input1.device() == input2.device(), "Kuiper tensors must share one CUDA device");
  TORCH_CHECK(input3.device() == input2.device(), "Kuiper tensors must share one CUDA device");
  TORCH_CHECK(input4.device() == input2.device(), "Kuiper tensors must share one CUDA device");
  TORCH_CHECK(input5.device() == input2.device(), "Kuiper tensors must share one CUDA device");
  TORCH_CHECK(input0.is_contiguous(), "Kuiper requires contiguous typed storage");
  TORCH_CHECK(input0.dim() == 0, "Kuiper input rank mismatch");
  TORCH_CHECK(input1.is_contiguous(), "Kuiper requires contiguous typed storage");
  TORCH_CHECK(input1.dim() == 0, "Kuiper input rank mismatch");
  TORCH_CHECK(input2.is_contiguous(), "Kuiper requires contiguous typed storage");
  TORCH_CHECK(input2.dim() == 1, "Kuiper input rank mismatch");
  TORCH_CHECK(input2.size(0) == (8), "Kuiper input extent mismatch");
  TORCH_CHECK(input3.is_contiguous(), "Kuiper requires contiguous typed storage");
  TORCH_CHECK(input3.dim() == 2, "Kuiper input rank mismatch");
  TORCH_CHECK(input3.size(0) == (8), "Kuiper input extent mismatch");
  TORCH_CHECK(input3.size(1) == (32), "Kuiper input extent mismatch");
  TORCH_CHECK(input4.is_contiguous(), "Kuiper requires contiguous typed storage");
  TORCH_CHECK(input4.dim() == 2, "Kuiper input rank mismatch");
  TORCH_CHECK(input4.size(0) == (8), "Kuiper input extent mismatch");
  TORCH_CHECK(input4.size(1) == (32), "Kuiper input extent mismatch");
  TORCH_CHECK(input5.is_contiguous(), "Kuiper requires contiguous typed storage");
  TORCH_CHECK(input5.dim() == 2, "Kuiper input rank mismatch");
  TORCH_CHECK(input5.size(0) == (input6), "Kuiper input extent mismatch");
  TORCH_CHECK(input5.size(1) == (8), "Kuiper input extent mismatch");
  const c10::cuda::CUDAGuard device_guard(input2.device());
  auto output0 = torch::empty({8}, input2.options().dtype(torch::kInt64));
  auto output1 = torch::empty({8, 32}, input3.options().dtype(torch::kInt64));
  auto output2 = torch::empty({8, 32}, input4.options().dtype(torch::kFloat32));
  auto output3 = torch::empty({([&]() -> int64_t { const __int128 extent = static_cast<__int128>(input6) + 1LL; TORCH_CHECK(extent >= 0, "Kuiper output dimension is negative"); TORCH_CHECK(extent <= 0x7FFFFFFFFFFFFFFFLL, "Kuiper output dimension exceeds int64 range"); return static_cast<int64_t>(extent); }()), 8}, input5.options().dtype(torch::kInt32));
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  KPlay_Functional_kplay_run(input0.data_ptr<int32_t>(), input1.data_ptr<float>(), input2.data_ptr<int64_t>(), input3.data_ptr<int64_t>(), input4.data_ptr<float>(), input5.data_ptr<int32_t>(), static_cast<int64_t>(input6), output0.data_ptr<int64_t>(), output1.data_ptr<int64_t>(), output2.data_ptr<float>(), output3.data_ptr<int32_t>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  return {output0, output1, output2, output3};
}

#pragma push_macro("KRML_CHECK_SIZE")
#undef KRML_CHECK_SIZE
#include "krml/internal/target.h"
#pragma pop_macro("KRML_CHECK_SIZE")
#include "KPlay_Functional.cu"
