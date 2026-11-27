#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include "KPlay_Functional.h"

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, int64_t input1) {
  TORCH_CHECK(input0.is_cuda(), "Kuiper requires CUDA tensors");
  TORCH_CHECK(input0.is_contiguous(), "Kuiper requires contiguous typed storage");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(input0.data_ptr()) % 8 == 0, "Kuiper input alignment mismatch");
  const c10::cuda::CUDAGuard device_guard(input0.device());
  auto output0 = torch::empty({18, 16, 128, 128}, input0.options().dtype(torch::kFloat32));
  auto output1 = torch::empty({18, 6144, 4}, input0.options().dtype(torch::kFloat32));
  auto output2 = torch::empty({6, 2, 2048, 256}, input0.options().dtype(torch::kBFloat16));
  auto output3 = torch::empty({6, 2, 2048, 256}, input0.options().dtype(torch::kBFloat16));
  auto output4 = torch::empty({input1}, input0.options().dtype(torch::kInt64));
  auto output5 = torch::empty({input1, 1024}, input0.options().dtype(torch::kBFloat16));
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  KPlay_Functional_kplay_run(input0.data_ptr<int64_t>(), static_cast<size_t>(input0.numel()), static_cast<int64_t>(input1), output0.data_ptr<float>(), output1.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(output2.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(output3.data_ptr<at::BFloat16>()), output4.data_ptr<int64_t>(), reinterpret_cast<__nv_bfloat16*>(output5.data_ptr<at::BFloat16>()));
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  return {output0, output1, output2, output3, output4, output5};
}

#pragma push_macro("KRML_CHECK_SIZE")
#undef KRML_CHECK_SIZE
#include "krml/internal/target.h"
#pragma pop_macro("KRML_CHECK_SIZE")
#include "KPlay_Functional.cu"
