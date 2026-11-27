
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(float *scores, float *lv, int64_t *li, float *bvv,
    int64_t *bii, int64_t *sel, uint32_t n, int64_t *tokens, int64_t *tout,
    int64_t d)
{
    uint32_t b = 0U;
    while (b < 512U) {
        uint32_t bv = b;
        uint32_t w = 0U;
        while (w < 32U) {
            uint32_t wv = w;
            uint32_t start = bv * 485U + wv;
            uint32_t stop = wv < 8U ? (bv + 1U) * 485U : 0U;
            uint32_t m = start;
            float best = 0.0f;
            int64_t idx = -1LL;
            while (m < stop) {
                uint32_t mv = m;
                float s = scores[mv];
                int64_t mi = (int64_t) (uint32_t) mv;
                float bst = best;
                if (mi >= 0LL &&
                    (idx < 0LL || bst < s || s != s && bst == bst)) {
                    best = s;
                    idx = mi;
                }
                m = mv + 8U;
            }
            int64_t ix = idx;
            lv[wv] = best;
            li[wv] = ix;
            w = wv + 1U;
        }
        uint32_t d1 = 16U;
        while (d1 > 0U) {
            uint32_t dv = d1;
            uint32_t i = 0U;
            while (i < dv) {
                uint32_t iv = i;
                uint32_t jv = iv + dv;
                float a = lv[iv];
                float b1 = lv[jv];
                int64_t bi = li[jv];
                if (bi >= 0LL &&
                    (li[iv] < 0LL || a < b1 || b1 != b1 && a == a)) {
                    lv[iv] = b1;
                    li[iv] = bi;
                }
                i = iv + 1U;
            }
            d1 = dv / 2U;
        }
        int64_t i0 = li[0U];
        bvv[bv] = lv[0U];
        bii[bv] = i0;
        b = bv + 1U;
    }
    uint32_t d1 = 256U;
    while (d1 > 0U) {
        uint32_t dv = d1;
        uint32_t i = 0U;
        while (i < dv) {
            uint32_t iv = i;
            uint32_t jv = iv + dv;
            float a = bvv[iv];
            float b1 = bvv[jv];
            int64_t bi = bii[jv];
            if (bi >= 0LL && (bii[iv] < 0LL || a < b1 || b1 != b1 && a == a)) {
                bvv[iv] = b1;
                bii[iv] = bi;
            }
            i = iv + 1U;
        }
        d1 = dv / 2U;
    }
    int64_t r = bii[0U];
    sel[0U] = r;
    uint32_t i = 0U;
    while (i < n) {
        uint32_t iv = i;
        int64_t x = tokens[iv];
        tout[iv] = (int64_t) (uint32_t) iv == d ? r : x;
        i = iv + 1U;
    }
}

void KPlay_Functional_kplay_run(float *scores, uint32_t n, int64_t *tokens,
    int64_t d, int64_t *sel, int64_t *tout)
{
    float *lv = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 32U);
    int64_t *li = (int64_t *) KPR_GPU_ALLOC((uint32_t) sizeof(int64_t), 32U);
    float *bvv = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 512U);
    int64_t *bii = (int64_t *) KPR_GPU_ALLOC((uint32_t) sizeof(int64_t), 512U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, scores, lv, li, bvv, bii,
        sel, n, tokens, tout, d);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    MUST(cudaFree(lv));
    MUST(cudaFree(li));
    MUST(cudaFree(bvv));
    MUST(cudaFree(bii));
}
