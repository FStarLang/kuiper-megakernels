#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include "KPlay_Functional.h"

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1, int64_t input2) {
  TORCH_CHECK(input0.is_cuda(), "Kuiper requires CUDA tensors");
  TORCH_CHECK(input1.device() == input0.device(), "Kuiper tensors must share one CUDA device");
  TORCH_CHECK(input1.is_contiguous(), "Kuiper requires contiguous typed storage");
  TORCH_CHECK(input1.dim() == 2, "Kuiper input rank mismatch");
  TORCH_CHECK(input1.size(0) == (7), "Kuiper input extent mismatch");
  TORCH_CHECK(input1.size(1) == (1024), "Kuiper input extent mismatch");
  const c10::cuda::CUDAGuard device_guard(input0.device());
  auto output0 = torch::empty({7, 151936}, input0.options().dtype(torch::kBFloat16));
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  KPlay_Functional_kplay_run(static_cast<size_t>(input0.numel()), reinterpret_cast<__nv_bfloat16*>(input0.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(input1.data_ptr<at::BFloat16>()), static_cast<int64_t>(input2), reinterpret_cast<__nv_bfloat16*>(output0.data_ptr<at::BFloat16>()));
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  return {output0};
}

#pragma push_macro("KRML_CHECK_SIZE")
#undef KRML_CHECK_SIZE
#include "/fastdata/tsorensen/resolve-mirage-proof-20261125/installs/clean/inst/include/krml/krml/internal/target.h"
#pragma pop_macro("KRML_CHECK_SIZE")
#include "/fastdata/tsorensen/resolve-mirage-final-20261125/recheck/families/base/projection/kuiper/generated/KPlay_Functional.cu"
