
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(int64_t input1, int64_t input2, int64_t input3,
    __nv_bfloat16 *input0, int64_t *output0)
{
    int64_t ite;
    if (input1 >= 1LL && input1 <= 12LL && input2 >= 1LL && input2 <= 12LL &&
        input3 >= 0LL && input3 < 12LL && input1 <= input3 + 1LL &&
        input3 + 1LL < input2 + input1) {
        float value = __bfloat162float(input0[0U]);
        int64_t best = 0LL;
        uint32_t i = 1U;
        int64_t ii = 1LL;
        for (; i < (uint32_t) 248320U; ii++) {
            uint32_t iv = i;
            float nx = __bfloat162float(input0[iv]);
            float v = value;
            if (v == v && (nx != nx || v < nx)) {
                value = nx;
                best = ii;
            }
            i = iv + 1U;
        }
        ite = best;
    } else
        ite = 0LL;
    output0[0U] = ite;
}

void KPlay_Functional_kplay_run(__nv_bfloat16 *input0, int64_t input1,
    int64_t input2, int64_t input3, int64_t *output0)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, input1, input2, input3,
        input0, output0);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
