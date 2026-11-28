
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(int64_t position, uint32_t tlen, int64_t *tokens,
    uint32_t plen, int64_t *prompt, __nv_bfloat16 *table, float *hidden,
    float *cosine, float *sine)
{
    if (position >= 0LL && position < 12LL) {
        uint32_t pos = (uint32_t) position;
        int64_t t;
        if (pos == 0U)
            t = 0U < tlen ? tokens[0U] : 0LL;
        else if (pos < plen)
            t = prompt[pos];
        else {
            uint32_t q = pos - plen;
            t = q < tlen ? tokens[q] : 0LL;
        }
        if (t >= 0LL && t < 248320LL) {
            uint32_t base = (uint32_t) t * 1024U;
            uint32_t j = 0U;
            while (j < 1024U) {
                uint32_t vj = j;
                hidden[vj] = __bfloat162float(table[base + vj]);
                j = vj + 1U;
            }
        }
    }
    float pf = (float) position;
    float l = logf(10000000.0f) / logf(2.0f);
    uint32_t i = (uint32_t) (plen == 0U);
    while (i < 32U) {
        uint32_t vi = i;
        float fi = (float) (int64_t) (uint32_t) (2U * vi);
        float fr = kpr_f32_mul_rn_ftz(
            pf, kpr_f32_exp2_approx_ftz(-kpr_f32_mul_rn_ftz(
                    kpr_f32_mul_rn_ftz(fi, kpr_f32_rcp_approx_ftz(64.0f)), l)));
        float cv = __cosf(fr);
        float sv = __sinf(fr);
        cosine[vi] = cv;
        sine[vi] = sv;
        i = vi + 1U;
    }
}

void KPlay_Functional_kplay_run(__nv_bfloat16 *table, int64_t *prompt,
    uint32_t plen, int64_t *tokens, uint32_t tlen, int64_t position,
    float *hidden, float *cosine, float *sine)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, position, tlen, tokens,
        plen, prompt, table, hidden, cosine, sine);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
