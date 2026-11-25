#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include "KPlay_Functional.h"

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1) {
  TORCH_CHECK(input1.is_cuda(), "Kuiper requires CUDA tensors");
  TORCH_CHECK(input0.device() == input1.device(), "Kuiper tensors must share one CUDA device");
  const c10::cuda::CUDAGuard device_guard(input1.device());
  auto output0 = torch::empty({}, input1.options().dtype(torch::kInt64));
  auto output1 = torch::empty({1, 8}, input1.options().dtype(torch::kInt64));
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  KPlay_Functional_kplay_run(reinterpret_cast<__nv_bfloat16*>(input0.data_ptr<at::BFloat16>()), input1.data_ptr<int64_t>(), output0.data_ptr<int64_t>(), output1.data_ptr<int64_t>());
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  return {output0, output1};
}

#pragma push_macro("KRML_CHECK_SIZE")
#undef KRML_CHECK_SIZE
#include "/fastdata/tsorensen/resolve-mirage-proof-20261125/installs/clean/inst/include/krml/krml/internal/target.h"
#pragma pop_macro("KRML_CHECK_SIZE")
#include "/fastdata/tsorensen/resolve-mirage-final-20261125/recheck/families/optimized/argmax/kuiper/generated/KPlay_Functional.cu"
