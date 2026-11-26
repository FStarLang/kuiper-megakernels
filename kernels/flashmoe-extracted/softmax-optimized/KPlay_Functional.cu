
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(float *scores, uint32_t t, float *out)
{
    float maximum = -INFINITY;
    float denominator = 0.0f;
    uint32_t e = 0U;
    while (e < 8U) {
        uint32_t k = e;
        float x = scores[8U * t + k];
        float previous = maximum;
        float mx = fmaxf(previous, x);
        maximum = mx;
        float ea = __expf(__fsub_rn(previous, mx));
        float eb = __expf(__fsub_rn(x, mx));
        denominator = __fmaf_rn(denominator, ea, eb);
        e = k + 1U;
    }
    float mfin = maximum;
    float dfin = denominator;
    uint32_t j = 0U;
    while (j < 8U) {
        uint32_t k = j;
        out[k] = __fdividef(__expf(__fsub_rn(scores[8U * t + k], mfin)), dfin);
        j = k + 1U;
    }
}

void KPlay_Functional_kplay_run(float *scores, int64_t token, float *out)
{
    if (token >= 0LL && token < 256LL) {
        uint32_t t = (uint32_t) token;
        cudaStream_t s = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, scores, t, out);
        MUST(cudaStreamSynchronize(s));
        MUST(cudaStreamDestroy(s));
    }
}
