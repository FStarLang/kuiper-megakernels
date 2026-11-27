#ifndef KPLAY_TRANSCENDENTALS_CUH
#define KPLAY_TRANSCENDENTALS_CUH

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <cmath>

static __device__ __forceinline__ float
kplay_flush_subnormal_f32(float value) {
  const unsigned int bits = __float_as_uint(value);
  return ((bits & 0x7f800000u) == 0u)
      ? __uint_as_float(bits & 0x80000000u)
      : value;
}

static __device__ __forceinline__ float
kplay_fast_ftz_swish_f32(float value) {
  const float x = kplay_flush_subnormal_f32(value);
  const float exponential =
      kplay_flush_subnormal_f32(__expf(-x));
  const float denominator =
      kplay_flush_subnormal_f32(__fadd_rn(1.0f, exponential));
  return kplay_flush_subnormal_f32(
      __fdividef(x, denominator));
}

#define Kuiper_KPlay_Transcendentals_fast_ftz_swish_f32(x) \
  kplay_fast_ftz_swish_f32((x))

static __device__ __forceinline__ float
kplay_selu_p027_base_f32(float value) {
  constexpr float alpha = 1.6732632423543772848170429916717f;
  constexpr float scale = 1.0507009873554804934193349852946f;
  constexpr float negative_scale = alpha * scale;
  return value > 0.0f
      ? __fmul_rn(value, scale)
      : __fmul_rn(::expm1f(__fmul_rn(value, 1.0f)), negative_scale);
}

static __device__ __forceinline__ float
kplay_selu_p027_ftz_f32(float input) {
  constexpr float scale = 1.0507009873554804934193349852946f;
  constexpr float alpha = 1.6732632423543772848170429916717f;
  constexpr float negative_scale = scale * alpha;
  float value;
  asm("mul.rn.ftz.f32 %0, %1, 0f3f800000;"
      : "=f"(value) : "f"(input));
  float factor = scale;
  if (!(value > 0.0f)) {
    value = ::expm1f(value);
    factor = negative_scale;
  }
  float result;
  asm("mul.rn.ftz.f32 %0, %1, %2;"
      : "=f"(result) : "f"(factor), "f"(value));
  return result;
}

static __device__ __forceinline__ float
kplay_softplus_p029_base_f32(float x) {
  constexpr float beta = 1.0f;
  constexpr float threshold = 20.0f;
  return (x * beta) > threshold
      ? x : (::log1p(::exp(x * beta))) / beta;
}

static __device__ __forceinline__ float
kplay_softplus_p029_fast_f32(float value) {
  if (value > 20.0f) {
    return value;
  }
  float exponential;
  asm volatile(
      "mul.rn.ftz.f32 %0, %1, 0f3FB8AA3B;\n\t"
      "ex2.approx.ftz.f32 %0, %0;"
      : "=f"(exponential) : "f"(value));
  return ::log1pf(exponential);
}

static __device__ __forceinline__ float
kplay_elu_p031_base_f32(float value) {
  if (value > 0.0f) {
    return __fmul_rn(value, 1.0f);
  } else {
    const float exponent = __fmul_rn(value, 1.0f);
    const float difference = ::expm1f(exponent);
    return __fmul_rn(difference, 1.0f);
  }
}

#define Kuiper_KPlay_Transcendentals_selu_p027_base_f32(x) \
  kplay_selu_p027_base_f32((x))
#define Kuiper_KPlay_Transcendentals_selu_p027_ftz_f32(x) \
  kplay_selu_p027_ftz_f32((x))
#define Kuiper_KPlay_Transcendentals_softplus_p029_base_f32(x) \
  kplay_softplus_p029_base_f32((x))
#define Kuiper_KPlay_Transcendentals_softplus_p029_fast_f32(x) \
  kplay_softplus_p029_fast_f32((x))
#define Kuiper_KPlay_Transcendentals_elu_p031_base_f32(x) \
  kplay_elu_p031_base_f32((x))
#endif

#endif
