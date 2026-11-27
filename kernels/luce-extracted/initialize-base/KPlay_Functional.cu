
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(uint32_t p, int64_t steps, int64_t *prompt,
    __nv_bfloat16 *table, __nv_bfloat16 *emb, __nv_bfloat16 *cache,
    float *recurrent, int64_t *tokens, __nv_bfloat16 *history)
{
    if (p >= 1U && p <= 12U && steps >= 1LL && steps <= 12LL) {
        uint32_t s = (uint32_t) steps;
        if (p + s <= 13U) {
            uint32_t q = 0U;
            while (q < p) {
                uint32_t vq = q;
                int64_t tok = prompt[vq];
                if (tok >= 0LL && tok < 248320LL) {
                    uint32_t base = (uint32_t) tok * 1024U;
                    uint32_t row = vq * 1024U;
                    uint32_t i = 0U;
                    while (i < 1024U) {
                        uint32_t vi = i;
                        emb[row + vi] = table[base + vi];
                        i = vi + 1U;
                    }
                }
                q = vq + 1U;
            }
            uint32_t i = 0U;
            while (i < (uint32_t) 442368U + (p + s - 1U) * 6144U) {
                uint32_t vi = i;
                cache[vi] = __float2bfloat16(0.0f);
                i = vi + 1U;
            }
            uint32_t i1 = 0U;
            while (i1 < (uint32_t) 4718592U) {
                uint32_t vi = i1;
                recurrent[vi] = 0.0f;
                i1 = vi + 1U;
            }
            uint32_t i2 = 0U;
            while (i2 < s) {
                uint32_t vi = i2;
                tokens[vi] = 0LL;
                i2 = vi + 1U;
            }
            uint32_t i3 = 0U;
            while (i3 < s * 1024U) {
                uint32_t vi = i3;
                history[vi] = __float2bfloat16(0.0f);
                i3 = vi + 1U;
            }
        }
    }
}

void KPlay_Functional_kplay_run(__nv_bfloat16 *table, int64_t *prompt,
    uint32_t p, int64_t steps, __nv_bfloat16 *emb, __nv_bfloat16 *cache,
    float *recurrent, int64_t *tokens, __nv_bfloat16 *history)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, p, steps, prompt, table,
        emb, cache, recurrent, tokens, history);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
