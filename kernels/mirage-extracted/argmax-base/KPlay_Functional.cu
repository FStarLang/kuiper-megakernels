
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(uint32_t n0, int64_t *input0, int64_t *output0,
    uint32_t n1, __nv_bfloat16 *input1, int64_t *output1)
{
    uint32_t i = 0U;
    while (i < n0) {
        uint32_t iv = i;
        output0[iv] = input0[iv];
        i = iv + 1U;
    }
    if (0U < n1) {
        float best = __bfloat162float(input1[0U]);
        uint32_t idx = 0U;
        uint32_t col = 1U;
        while (col < n1) {
            uint32_t c = col;
            float fv = __bfloat162float(input1[c]);
            float fb = best;
            if (fb < fv || fv != fv && fb == fb) {
                best = fv;
                idx = c;
            }
            col = c + 1U;
        }
        int64_t r = (int64_t) (uint32_t) idx;
        output0[n0] = r;
        output1[0U] = r;
    } else {
        output0[n0] = 0LL;
        output1[0U] = 0LL;
    }
}

void KPlay_Functional_kplay_run(uint32_t n0, uint32_t n1, int64_t *input0,
    __nv_bfloat16 *input1, int64_t input2, int64_t *output0, int64_t *output1)
{
    KRML_MAYBE_UNUSED_VAR(input2);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, n0, input0, output0, n1,
        input1, output1);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
