
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(int64_t *input0, int32_t *output0)
{
    uint32_t i = 0U;
    while (i < 8U) {
        uint32_t iv = i;
        output0[iv] = (int32_t) input0[iv];
        i = iv + 1U;
    }
}

int32_t KPlay_Functional_demand_of(int64_t x) { return (int32_t) x; }

void KPlay_Functional_kplay_run(int64_t *input0, int32_t *output0)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, input0, output0);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
