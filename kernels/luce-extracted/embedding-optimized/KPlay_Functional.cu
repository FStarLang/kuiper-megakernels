
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
    uint32_t i = (uint32_t) (plen == 0U);
    while (i < 32U) {
        uint32_t vi = i;
        float fr = kpr_f32_mul_rn_ftz(
            pf, vi == 0U    ? 1.0f
                : vi == 1U  ? 0.604296386241912841796875f
                : vi == 2U  ? 0.3651741445064544677734375f
                : vi == 3U  ? 0.22067339718341827392578125f
                : vi == 4U  ? 0.13335214555263519287109375f
                : vi == 5U  ? 0.08058421313762664794921875f
                : vi == 6U  ? 0.0486967526376247406005859375f
                : vi == 7U  ? 0.02942727319896221160888671875f
                : vi == 8U  ? 0.01778279431164264678955078125f
                : vi == 9U  ? 0.01074607856571674346923828125f
                : vi == 10U ? 0.0064938166178762912750244140625f
                : vi == 11U ? 0.0039241896010935306549072265625f
                : vi == 12U ? 0.00237137381918728351593017578125f
                : vi == 13U ? 0.001433012657798826694488525390625f
                : vi == 14U ? 0.000865964335389435291290283203125f
                : vi == 15U ? 0.000523299095220863819122314453125f
                : vi == 16U ? 0.000316227786242961883544921875f
                : vi == 17U ? 0.000191095299669541418552398681640625f
                : vi == 18U ? 0.000115478193038143217563629150390625f
                : vi == 19U ? 0.000069783054641447961330413818359375f
                : vi == 20U ? 0.0000421696531702764332294464111328125f
                : vi == 21U ? 0.000025482966520939953625202178955078125f
                : vi == 22U ? 0.000015399264157167635858058929443359375f
                : vi == 23U ? 0.00000930572059587575495243072509765625f
                : vi == 24U ? 0.00000562341347176698036491870880126953125f
                : vi == 25U ? 0.00000339820826411596499383449554443359375f
                : vi == 26U ? 0.00000205352489501819945871829986572265625f
                : vi == 27U ? 0.00000124093776321387849748134613037109375f
                : vi == 28U ? 0.00000074989424092564149759709835052490234375f
                : vi == 29U ? 0.000000453158378377338522113859653472900390625f
                : vi == 30U ? 0.00000027384197665014653466641902923583984375f
                            : 0.00000016548170833630138076841831207275390625f);
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
