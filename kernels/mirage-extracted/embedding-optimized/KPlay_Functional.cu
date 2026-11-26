
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(int64_t *a1, __nv_bfloat16 *a0, __nv_bfloat16 *o0,
    __nv_bfloat16 *o2, __nv_bfloat16 *o3)
{
    uint32_t i = 0U;
    while (i < 8192U) {
        uint32_t j = i;
        uint32_t row = j / 1024U;
        uint32_t col = j % 1024U;
        int64_t tok = 0LL;
        if (row < 7U)
            tok = a1[row];
        int64_t tv = tok;
        __nv_bfloat16 ite;
        if (0LL <= tv && tv < 151936LL)
            ite = a0[(uint32_t) tv * 1024U + col];
        else
            ite = __float2bfloat16(0.0f);
        o0[j] = ite;
        i = j + 1U;
    }
    uint32_t i1 = 0U;
    while (i1 < (uint32_t) 524288U) {
        uint32_t j = i1;
        int64_t row = (int64_t) (uint32_t) (j / 128U);
        float angle =
            (float) row *
            (1.0f / powf(1000000.0f,
                        (float) (int64_t) (uint32_t) (j % 128U % 64U) / 64.0f));
        __nv_bfloat16 cv = __float2bfloat16(cosf(angle));
        __nv_bfloat16 sv = __float2bfloat16(sinf(angle));
        o2[j] = cv;
        o3[j] = sv;
        i1 = j + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_1(__nv_bfloat16 *o0, __nv_bfloat16 *a2, __nv_bfloat16 *o1)
{
    float p[256U];
    memset(p, 0U, 256U * sizeof(float));
    float onec = 1.0f;
    uint32_t r = 0U;
    while (r < 8U) {
        uint32_t rv = r;
        uint32_t t = 0U;
        while (t < 256U) {
            uint32_t tv = t;
            float sum = 0.0f;
            uint32_t j = 0U;
            while (j < 4U) {
                uint32_t jv = j;
                float xf = __bfloat162float(o0[rv * 1024U + tv + jv * 256U]);
                sum = __fmaf_rn(xf, xf, sum);
                j = jv + 1U;
            }
            p[tv] = sum;
            t = tv + 1U;
        }
        uint32_t w = 0U;
        while (w < 8U) {
            uint32_t wv = w;
            uint32_t d = 16U;
            while (d > 0U) {
                uint32_t dv = d;
                uint32_t t1 = 0U;
                while (t1 < dv) {
                    uint32_t j = t1;
                    p[32U * wv + j] += p[32U * wv + j + dv];
                    t1 = j + 1U;
                }
                d = dv / 2U;
            }
            p[wv] = p[32U * wv];
            w = wv + 1U;
        }
        uint32_t d = 4U;
        while (d > 0U) {
            uint32_t dv = d;
            uint32_t t1 = 0U;
            while (t1 < dv) {
                uint32_t j = t1;
                p[j] += p[j + dv];
                t1 = j + 1U;
            }
            d = dv / 2U;
        }
        float inv = rsqrtf(p[0U] / 1024.0f + onec / 1000000.0f);
        uint32_t c = 0U;
        while (c < 1024U) {
            uint32_t cv = c;
            uint32_t idx = rv * 1024U + cv;
            __nv_bfloat16 x = o0[idx];
            __nv_bfloat16 w1 = a2[cv];
            o1[idx] = __float2bfloat16(
                __bfloat162float(x) * (inv * __bfloat162float(w1)));
            c = cv + 1U;
        }
        r = rv + 1U;
    }
}

void KPlay_Functional_kplay_run(__nv_bfloat16 *a0, int64_t *a1,
    __nv_bfloat16 *a2, __nv_bfloat16 *o0, __nv_bfloat16 *o1, __nv_bfloat16 *o2,
    __nv_bfloat16 *o3)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, a1, a0, o0, o2, o3);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_1, 1U, 1U, 0U, s1, o0, a2, o1);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}
