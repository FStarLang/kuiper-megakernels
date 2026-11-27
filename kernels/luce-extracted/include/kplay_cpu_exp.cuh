#ifndef KPLAY_CPU_EXP_CUH
#define KPLAY_CPU_EXP_CUH

#if defined(__CUDACC__)
#include <cuda_runtime.h>

// Scalar form of PyTorch's vectorized CPU (AVX512) softmax exponential, for value <= 0.
static __device__ __forceinline__ float kplay_cpu_exp_f32(float value) {
  if (value < -104.0f) {
    return 0.0f;
  }
  int exponent = __float2int_rn(__fmul_rn(value, 0x1.715476p+0f));
  float reduced = __fmaf_rn(float(exponent), -0x1.62e4p-1f, value);
  reduced = __fmaf_rn(float(exponent), -0x1.7f7d1cp-20f, reduced);
  float square = __fmul_rn(reduced, reduced);
  float polynomial = __fmaf_rn(0x1.a057b4p-13f, reduced, 0x1.6d2d92p-10f);
  polynomial = __fmaf_rn(polynomial, reduced, 0x1.11114cp-7f);
  polynomial = __fmaf_rn(polynomial, reduced, 0x1.5554f4p-5f);
  polynomial = __fmaf_rn(polynomial, reduced, 0x1.555556p-3f);
  polynomial = __fmaf_rn(polynomial, reduced, 0.5f);
  float result = __fadd_rn(__fmaf_rn(square, polynomial, reduced), 1.0f);
  int half_exponent = exponent >> 1;
  result = __fmul_rn(result, __int_as_float((half_exponent + 127) << 23));
  return __fmul_rn(
      result, __int_as_float((exponent - half_exponent + 127) << 23));
}

#endif
#endif
