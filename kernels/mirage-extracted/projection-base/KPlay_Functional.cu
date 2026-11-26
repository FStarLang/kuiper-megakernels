
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(__nv_bfloat16 *wb, uint32_t a2, __nv_bfloat16 *t0)
{
    if (1024U * blockIdx.x + threadIdx.x < (uint32_t) 155582464U)
        wb[(1024U * blockIdx.x + threadIdx.x) / 1024U * 1024U +
            (1024U * blockIdx.x + threadIdx.x) % 1024U] =
            t0[a2 + ((1024U * blockIdx.x + threadIdx.x) / 1024U * 1024U +
                        (1024U * blockIdx.x + threadIdx.x) % 1024U)];
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_1(__nv_bfloat16 *pd, __nv_bfloat16 *t1)
{
    if (1024U * blockIdx.x + threadIdx.x < 8192U) {
        __nv_bfloat16 ite;
        if ((1024U * blockIdx.x + threadIdx.x) % 8U < 7U)
            ite = t1[(1024U * blockIdx.x + threadIdx.x) % 8U * 1024U +
                     (1024U * blockIdx.x + threadIdx.x) / 8U];
        else
            ite = __float2bfloat16(0.0f);
        pd[(1024U * blockIdx.x + threadIdx.x) % 8U * 1024U +
            (1024U * blockIdx.x + threadIdx.x) / 8U] = ite;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_2(__nv_bfloat16 *wb, __nv_bfloat16 *pd, float *sc)
{
    uint32_t r = 0U;
    while (r < 9496U) {
        uint32_t ri = r;
        auto &fr = KPR_INIT(kpr_mma_fragment);
        kpr_mma_fill(fr, 0.0f);
        uint32_t s = 0U;
        while (s < 64U) {
            uint32_t si = s;
            kpr_mma_mma_sync(wb + (1024U * (ri * 16U) + si * 16U), 1024U,
                pd + si * 16U, 1024U, fr);
            s = si + 1U;
        }
        kpr_mma_store(fr, sc + 8U * (ri * 16U), 8U);
        r = ri + 1U;
    }
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_3(__nv_bfloat16 *tout, float *sc)
{
    if (1024U * blockIdx.x + threadIdx.x < (uint32_t) 1063552U)
        tout[1024U * blockIdx.x + threadIdx.x] = __float2bfloat16(
            sc[(1024U * blockIdx.x + threadIdx.x) % (uint32_t) 151936U * 8U +
                (1024U * blockIdx.x + threadIdx.x) / (uint32_t) 151936U]);
}

krml_checked_int_t KPlay_Functional_nv = 151936;

krml_checked_int_t KPlay_Functional_ntiles = 9496;

void Kuiper_Shareable_double_shareable__Mkshareable(
    __nv_bfloat16 *uu___, __nv_bfloat16 *uu___1)
{
    KRML_MAYBE_UNUSED_VAR(uu___);
    KRML_MAYBE_UNUSED_VAR(uu___1);
}

void KPlay_Functional_inframe_shareable(
    krml_checked_int_t n0, __nv_bfloat16 *uu___, __nv_bfloat16 *uu___1)
{
    KRML_MAYBE_UNUSED_VAR(n0);
    Kuiper_Shareable_double_shareable__Mkshareable(uu___, uu___1);
}

void KPlay_Functional_kplay_run(uint32_t n0, __nv_bfloat16 *in0,
    __nv_bfloat16 *in1, int64_t in2, __nv_bfloat16 *out0)
{
    if (in2 >= 0LL && in2 <= 1991901183LL) {
        uint32_t a2 = (uint32_t) in2;
        if ((uint32_t) 155582464U <= n0)
            if (a2 <= n0 - (uint32_t) 155582464U) {
                __nv_bfloat16 *wb = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                    (uint32_t) sizeof(__nv_bfloat16), (uint32_t) 155582464U);
                cudaStream_t s = KPR_FRESH_STREAM();
                KPR_KCALL(__hoisted_kplay_run_0,
                    (uint32_t) 155582464U / 1024U +
                        (uint32_t) ((uint32_t) 155582464U % 1024U != 0U),
                    1024U, 0U, s, wb, a2, in0);
                MUST(cudaStreamSynchronize(s));
                MUST(cudaStreamDestroy(s));
                __nv_bfloat16 *pd = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                    (uint32_t) sizeof(__nv_bfloat16), 8192U);
                cudaStream_t s1 = KPR_FRESH_STREAM();
                KPR_KCALL(__hoisted_kplay_run_1, 8U, 1024U, 0U, s1, pd, in1);
                MUST(cudaStreamSynchronize(s1));
                MUST(cudaStreamDestroy(s1));
                float *sc = (float *) KPR_GPU_ALLOC(
                    (uint32_t) sizeof(float), (uint32_t) 1215488U);
                cudaStream_t s2 = KPR_FRESH_STREAM();
                KPR_KCALL(__hoisted_kplay_run_2, 1U, 32U, 0U, s2, wb, pd, sc);
                MUST(cudaStreamSynchronize(s2));
                MUST(cudaStreamDestroy(s2));
                cudaStream_t s3 = KPR_FRESH_STREAM();
                KPR_KCALL(__hoisted_kplay_run_3,
                    (uint32_t) 1063552U / 1024U +
                        (uint32_t) ((uint32_t) 1063552U % 1024U != 0U),
                    1024U, 0U, s3, out0, sc);
                MUST(cudaStreamSynchronize(s3));
                MUST(cudaStreamDestroy(s3));
                MUST(cudaFree(wb));
                MUST(cudaFree(pd));
                MUST(cudaFree(sc));
            }
    }
}
