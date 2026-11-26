
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(__nv_bfloat16 *w, __nv_bfloat16 *wt, int64_t *ix,
    float *pr, int32_t *sv, int32_t *cn)
{
    uint32_t k = 0U;
    while (k < 512U) {
        uint32_t kv = k;
        uint32_t n = 0U;
        while (n < 8U) {
            uint32_t nv = n;
            wt[nv * 512U + kv] = w[kv * 8U + nv];
            n = nv + 1U;
        }
        k = kv + 1U;
    }
    uint32_t i = 0U;
    while (i < 256U) {
        uint32_t iv = i;
        ix[iv] = -1LL;
        i = iv + 1U;
    }
    uint32_t i1 = 0U;
    while (i1 < 256U) {
        uint32_t iv = i1;
        pr[iv] = 0.0f;
        i1 = iv + 1U;
    }
    uint32_t i2 = 0U;
    while (i2 < 2048U) {
        uint32_t iv = i2;
        sv[iv] = 0;
        i2 = iv + 1U;
    }
    uint32_t i3 = 0U;
    while (i3 < 8U) {
        uint32_t iv = i3;
        cn[iv] = 0;
        i3 = iv + 1U;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_1(__nv_bfloat16 *x, __nv_bfloat16 *wt, float *sc)
{
    uint32_t ti = 0U;
    while (ti < 16U) {
        uint32_t t = ti;
        auto &fr = KPR_INIT(kpr_mma_fragment);
        kpr_mma_fill(fr, 0.0f);
        uint32_t kb = 0U;
        while (kb < 32U) {
            uint32_t k = kb;
            kpr_mma_mma_sync(
                x + (512U * (t * 16U) + k * 16U), 512U, wt + k * 16U, 512U, fr);
            kb = k + 1U;
        }
        kpr_mma_store(fr, sc + 8U * (t * 16U), 8U);
        ti = t + 1U;
    }
}

void KPlay_Functional_kplay_run(__nv_bfloat16 *x, __nv_bfloat16 *w, float *sc,
    int32_t *cn, int64_t *ix, float *pr, int32_t *sv)
{
    __nv_bfloat16 *wt = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 4096U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, w, wt, ix, pr, sv, cn);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_1, 1U, 32U, 0U, s1, x, wt, sc);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    MUST(cudaFree(wt));
}
