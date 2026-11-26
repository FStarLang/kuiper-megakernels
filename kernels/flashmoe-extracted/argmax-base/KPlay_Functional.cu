
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(float *input0, int32_t *output0, float *output1)
{
    int32_t selected = 0;
    float selected_probability = input0[0U];
    uint32_t expert = 1U;
    int32_t expert_i = 1;
    while (expert < 8U) {
        uint32_t j = expert;
        int32_t ji = expert_i;
        float probability = input0[j];
        if (selected_probability < probability) {
            selected = ji;
            selected_probability = probability;
        }
        expert = j + 1U;
        expert_i = ji + 1;
    }
    float b = selected_probability;
    output0[0U] = selected;
    output1[0U] = b;
}

void KPlay_Functional_kplay_run(float *input0, int32_t *output0, float *output1)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, input0, output0, output1);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
