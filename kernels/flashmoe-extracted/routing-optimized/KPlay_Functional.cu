
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(int32_t *counts, int32_t *out_counts, int64_t *idx,
    int64_t *out_idx, float *probs, float *out_probs, int32_t *surv,
    int32_t *out_surv, int32_t *sel, float *sp, float *row, int64_t token)
{
    uint32_t i = 0U;
    while (i < 8U) {
        uint32_t j = i;
        out_counts[j] = counts[j];
        i = j + 1U;
    }
    uint32_t i1 = 0U;
    while (i1 < 256U) {
        uint32_t j = i1;
        out_idx[j] = idx[j];
        i1 = j + 1U;
    }
    uint32_t i2 = 0U;
    while (i2 < 256U) {
        uint32_t j = i2;
        out_probs[j] = probs[j];
        i2 = j + 1U;
    }
    uint32_t i3 = 0U;
    while (i3 < 2048U) {
        uint32_t j = i3;
        out_surv[j] = surv[j];
        i3 = j + 1U;
    }
    int32_t s = sel[0U];
    float p = sp[0U];
    bool vs = 0 <= s && s < 8;
    uint32_t si = vs ? (uint32_t) s : 0U;
    float prob = row[si] / (0.0f + p);
    int32_t r = counts[si];
    out_counts[si] = vs && r < 2147483647 ? r + 1 : r;
    bool keep = vs && 0 <= r && r < 32 && 0LL <= token && token < 256LL;
    uint32_t k = keep ? 32U * si + (uint32_t) r : 0U;
    int64_t oi = idx[k];
    out_idx[k] = keep ? token : oi;
    float op = probs[k];
    out_probs[k] = keep ? prob : op;
    uint32_t j = keep ? 8U * (uint32_t) token + si : 0U;
    int32_t ov = surv[j];
    out_surv[j] = keep ? 1 : ov;
}

void KPlay_Functional_kplay_run(float *row, int32_t *sel, float *sp,
    int32_t *counts, int64_t *idx, float *probs, int32_t *surv, int64_t token,
    int32_t *out_counts, int64_t *out_idx, float *out_probs, int32_t *out_surv)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, counts, out_counts, idx,
        out_idx, probs, out_probs, surv, out_surv, sel, sp, row, token);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
