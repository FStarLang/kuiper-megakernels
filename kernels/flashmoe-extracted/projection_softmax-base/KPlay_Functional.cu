
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(__nv_bfloat16 *x, __nv_bfloat16 *w, float *softmax,
    int64_t *token_indices, float *probabilities, int64_t *prefix)
{
    uint32_t t = 0U;
    while (t < 256U) {
        uint32_t tv = t;
        uint32_t base = tv * 8U;
        uint32_t e = 0U;
        while (e < 8U) {
            uint32_t ev = e;
            float acc = 0.0f;
            uint32_t h = 0U;
            while (h < 512U) {
                uint32_t hv = h;
                __nv_bfloat16 xv = x[tv * 512U + hv];
                __nv_bfloat16 wv = w[hv * 8U + ev];
                float a = acc;
                acc = __fmaf_rn(__bfloat162float(xv), __bfloat162float(wv), a);
                h = hv + 1U;
            }
            softmax[base + ev] = acc;
            e = ev + 1U;
        }
        float mx = softmax[base];
        e = 1U;
        while (e < 8U) {
            uint32_t ev = e;
            mx = fmaxf(mx, softmax[base + ev]);
            e = ev + 1U;
        }
        float m = mx;
        e = 0U;
        while (e < 8U) {
            uint32_t ev = e;
            softmax[base + ev] =
                kplay_cpu_exp_f32(__fsub_rn(softmax[base + ev], m));
            e = ev + 1U;
        }
        float den = softmax[base];
        e = 1U;
        while (e < 8U) {
            uint32_t ev = e;
            den += softmax[base + ev];
            e = ev + 1U;
        }
        float inv = 1.0f / den;
        e = 0U;
        while (e < 8U) {
            uint32_t ev = e;
            softmax[base + ev] *= inv;
            e = ev + 1U;
        }
        t = tv + 1U;
    }
    uint32_t k = 0U;
    while (k < 256U) {
        uint32_t kv = k;
        token_indices[kv] = -1LL;
        probabilities[kv] = 0.0f;
        k = kv + 1U;
    }
    uint32_t e = 0U;
    while (e < 8U) {
        uint32_t ev = e;
        prefix[ev] = 0LL;
        e = ev + 1U;
    }
}

void KPlay_Functional_kplay_run(__nv_bfloat16 *x, __nv_bfloat16 *w,
    float *softmax, int64_t *prefix, int64_t *token_indices,
    float *probabilities, int32_t *survival)
{
    KRML_MAYBE_UNUSED_VAR(survival);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, x, w, softmax,
        token_indices, probabilities, prefix);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
