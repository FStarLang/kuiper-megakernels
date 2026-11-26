
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(int64_t input6, int32_t *input0, float *input1,
    int64_t *input2, int64_t *output0, int64_t *input3, int64_t *output1,
    float *input4, float *output2, int32_t *input5, int32_t *output3)
{
    if (0LL <= input6 && input6 < 256LL) {
        int32_t sel = input0[0U];
        if (0 <= sel && sel < 8) {
            uint32_t tsz = (uint32_t) input6;
            uint32_t ssz = (uint32_t) sel;
            float sp = input1[0U];
            float c = sp / sp;
            uint32_t i = 0U;
            while (i < 8U) {
                uint32_t j = i;
                output0[j] = input2[j];
                i = j + 1U;
            }
            uint32_t i1 = 0U;
            while (i1 < 256U) {
                uint32_t j = i1;
                output1[j] = input3[j];
                i1 = j + 1U;
            }
            uint32_t i2 = 0U;
            while (i2 < 256U) {
                uint32_t j = i2;
                output2[j] = input4[j];
                i2 = j + 1U;
            }
            uint32_t base = 8U * tsz;
            uint32_t i3 = 0U;
            while (i3 < base) {
                uint32_t j = i3;
                output3[j] = input5[j];
                i3 = j + 1U;
            }
            int64_t rank = input2[ssz];
            output0[ssz] = rank < 9223372036854775807LL ? rank + 1LL : rank;
            uint32_t i4 = 0U;
            while (i4 < 8U) {
                uint32_t j = i4;
                output3[base + j] =
                    (int32_t) !!(j <= ssz && ssz <= j && rank < 32LL);
                i4 = j + 1U;
            }
            bool ok = 0LL <= rank && rank < 32LL;
            uint32_t slot = ok && 0LL <= rank && rank < 32LL
                                ? 32U * ssz + (uint32_t) rank
                                : 0U;
            if (ok)
                output1[slot] = input6;
            if (ok)
                output2[slot] = c;
        }
    }
}

void KPlay_Functional_kplay_run(int32_t *input0, float *input1, int64_t *input2,
    int64_t *input3, float *input4, int32_t *input5, int64_t input6,
    int64_t *output0, int64_t *output1, float *output2, int32_t *output3)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, input6, input0, input1,
        input2, output0, input3, output1, input4, output2, input5, output3);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
