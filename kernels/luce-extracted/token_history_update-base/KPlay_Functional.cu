
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(int64_t pP, int64_t t, int64_t *selected, uint32_t n,
    int64_t *old_tokens, int64_t *tokens, __nv_bfloat16 *hidden,
    __nv_bfloat16 *old_history, __nv_bfloat16 *history)
{
    int64_t d =
        pP >= 1LL && pP <= 12LL && t >= 0LL && t < 12LL ? t - pP + 1LL : -1LL;
    int64_t best = selected[0U];
    uint32_t i = 0U;
    while (i < n) {
        uint32_t vi = i;
        int64_t ot = old_tokens[vi];
        tokens[vi] = (int64_t) (uint32_t) vi == d ? best : ot;
        i = vi + 1U;
    }
    uint32_t nh = n * 1024U;
    uint32_t k = 0U;
    while (k < nh) {
        uint32_t vk = k;
        __nv_bfloat16 hv = hidden[vk % 1024U];
        __nv_bfloat16 ov = old_history[vk];
        history[vk] = (int64_t) (uint32_t) (vk / 1024U) == d ? hv : ov;
        k = vk + 1U;
    }
}

void KPlay_Functional_kplay_run(uint32_t n, int64_t *old_tokens,
    __nv_bfloat16 *old_history, __nv_bfloat16 *hidden, int64_t *selected,
    int64_t pP, int64_t t, int64_t *tokens, __nv_bfloat16 *history)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, pP, t, selected, n,
        old_tokens, tokens, hidden, old_history, history);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
