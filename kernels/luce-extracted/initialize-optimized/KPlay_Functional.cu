
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(float *rc, float *cv, __nv_bfloat16 *k, __nv_bfloat16 *v,
    uint32_t ntok, int64_t *tokens, uint32_t plen, int64_t *prompt,
    __nv_bfloat16 *history)
{
    uint32_t i = 0U;
    while (i < (uint32_t) 4718592U) {
        uint32_t vi = i;
        rc[vi] = 0.0f;
        i = vi + 1U;
    }
    uint32_t i1 = 0U;
    while (i1 < (uint32_t) 442368U) {
        uint32_t vi = i1;
        cv[vi] = 0.0f;
        i1 = vi + 1U;
    }
    uint32_t i2 = 0U;
    while (i2 < (uint32_t) 6291456U) {
        uint32_t vi = i2;
        k[vi] = __float2bfloat16(0.0f);
        i2 = vi + 1U;
    }
    uint32_t i3 = 0U;
    while (i3 < (uint32_t) 6291456U) {
        uint32_t vi = i3;
        v[vi] = __float2bfloat16(0.0f);
        i3 = vi + 1U;
    }
    uint32_t i4 = 0U;
    while (i4 < ntok) {
        uint32_t vi = i4;
        tokens[vi] = 0LL;
        i4 = vi + 1U;
    }
    if (ntok > 0U && plen > 0U)
        tokens[0U] = prompt[0U];
    uint32_t i5 = 0U;
    while (i5 < ntok * 1024U) {
        uint32_t vi = i5;
        history[vi] = __float2bfloat16(0.0f);
        i5 = vi + 1U;
    }
}

void KPlay_Functional_kplay_run(int64_t *prompt, uint32_t plen, int64_t steps,
    float *rc, float *cv, __nv_bfloat16 *k, __nv_bfloat16 *v, int64_t *tokens,
    __nv_bfloat16 *history)
{
    uint32_t ntok = (uint32_t) (uint64_t) steps;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, rc, cv, k, v, ntok, tokens,
        plen, prompt, history);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
