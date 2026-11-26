
#include "KPlay_Functional.h"

__device__ static __nv_bfloat16 zero_fn__device(__nv_bfloat16 uu___)
{
    KRML_MAYBE_UNUSED_VAR(uu___);
    return __float2bfloat16(0.0f);
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(__nv_bfloat16 *z)
{
    if (1024U * blockIdx.x + threadIdx.x < 256U)
        z[(1024U * blockIdx.x + threadIdx.x) / 16U * 16U +
            (1024U * blockIdx.x + threadIdx.x) % 16U] =
            zero_fn__device(z[(1024U * blockIdx.x + threadIdx.x) / 16U * 16U +
                              (1024U * blockIdx.x + threadIdx.x) % 16U]);
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_1(
    __nv_bfloat16 *input1, __nv_bfloat16 *input0, __nv_bfloat16 *z, float *s)
{
    auto &fr = KPR_INIT(kpr_mma_fragment);
    uint32_t b = 0U;
    while (b < 9600U) {
        uint32_t bv = b;
        kpr_mma_fill(fr, 0.0f);
        uint32_t k = 0U;
        while (k < 64U) {
            uint32_t kv = k;
            if (bv < 9496U)
                kpr_mma_mma_sync(input0 + (1024U * (bv * 16U) + kv * 16U),
                    1024U, input1 + kv * 16U, 1024U, fr);
            else
                kpr_mma_mma_sync(z, 16U, input1 + kv * 16U, 1024U, fr);
            k = kv + 1U;
        }
        kpr_mma_store(fr, s + 8U * (bv * 16U), 8U);
        b = bv + 1U;
    }
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_2(__nv_bfloat16 *output0, float *s)
{
    if (1024U * blockIdx.x + threadIdx.x < (uint32_t) 1228800U)
        output0[(1024U * blockIdx.x + threadIdx.x) / (uint32_t) 153600U *
                    (uint32_t) 153600U +
                (1024U * blockIdx.x + threadIdx.x) % (uint32_t) 153600U] =
            __float2bfloat16(
                s[(1024U * blockIdx.x + threadIdx.x) % (uint32_t) 153600U * 8U +
                    (1024U * blockIdx.x + threadIdx.x) / (uint32_t) 153600U]);
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_3(__nv_bfloat16 *output1, __nv_bfloat16 *input1)
{
    if (1024U * blockIdx.x + threadIdx.x < 7168U)
        output1[(1024U * blockIdx.x + threadIdx.x) % 7U * 1024U +
                (1024U * blockIdx.x + threadIdx.x) / 7U] =
            input1[(1024U * blockIdx.x + threadIdx.x) % 7U * 1024U +
                   (1024U * blockIdx.x + threadIdx.x) / 7U];
}

__nv_bfloat16 KPlay_Functional_zero_fn(__nv_bfloat16 uu___)
{
    KRML_MAYBE_UNUSED_VAR(uu___);
    return __float2bfloat16(0.0f);
}

void KPlay_Functional_kplay_run(__nv_bfloat16 *input0, __nv_bfloat16 *input1,
    __nv_bfloat16 *output0, __nv_bfloat16 *output1)
{
    __nv_bfloat16 *z =
        (__nv_bfloat16 *) KPR_GPU_ALLOC((uint32_t) sizeof(__nv_bfloat16), 256U);
    float *s =
        (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), (uint32_t) 1228800U);
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_0, 1U, 1024U, 0U, s1, z);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_1, 1U, 32U, 0U, s2, input1, input0, z, s);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    cudaStream_t s3 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_2,
        (uint32_t) 1228800U / 1024U +
            (uint32_t) ((uint32_t) 1228800U % 1024U != 0U),
        1024U, 0U, s3, output0, s);
    MUST(cudaStreamSynchronize(s3));
    MUST(cudaStreamDestroy(s3));
    cudaStream_t s4 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_3, 7U, 1024U, 0U, s4, output1, input1);
    MUST(cudaStreamSynchronize(s4));
    MUST(cudaStreamDestroy(s4));
    MUST(cudaFree(s));
    MUST(cudaFree(z));
}
