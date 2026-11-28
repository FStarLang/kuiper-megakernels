
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(float *row, int32_t *sel, float *sp)
{
    uint32_t selected = 0U;
    float best = -INFINITY;
    uint32_t e = 0U;
    while (e < 8U) {
        uint32_t j = e;
        float x = row[j];
        if (best < x) {
            selected = j;
            best = x;
        }
        e = j + 1U;
    }
    float bv = best;
    sel[0U] = (int32_t) (uint32_t) selected;
    sp[0U] = bv;
}

void KPlay_Functional_kplay_run(float *row, int32_t *sel, float *sp)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, row, sel, sp);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
