#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include "KPlay_Functional.h"

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, int64_t input1, int64_t input2, int64_t input3) {
  TORCH_CHECK(input0.is_cuda(), "Kuiper requires CUDA tensors");
  TORCH_CHECK(input0.is_contiguous(), "Kuiper requires contiguous typed storage");
  TORCH_CHECK(input0.dim() == 1, "Kuiper input rank mismatch");
  TORCH_CHECK(input0.size(0) == (248320), "Kuiper input extent mismatch");
  const c10::cuda::CUDAGuard device_guard(input0.device());
  auto output0 = torch::empty({}, input0.options().dtype(torch::kInt64));
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  KPlay_Functional_kplay_run(reinterpret_cast<__nv_bfloat16*>(input0.data_ptr<at::BFloat16>()), static_cast<int64_t>(input1), static_cast<int64_t>(input2), static_cast<int64_t>(input3), output0.data_ptr<int64_t>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  return {output0};
}

#pragma push_macro("KRML_CHECK_SIZE")
#undef KRML_CHECK_SIZE
#include "krml/internal/target.h"
#pragma pop_macro("KRML_CHECK_SIZE")
#include "KPlay_Functional.cu"
