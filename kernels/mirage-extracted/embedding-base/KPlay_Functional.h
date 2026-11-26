
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

void KPlay_Functional_kplay_run(uint32_t n, __nv_bfloat16 *input0,
    int64_t *input1, int64_t input2, int64_t input3, int64_t input4,
    __nv_bfloat16 *output0, __nv_bfloat16 *output1, __nv_bfloat16 *output2);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
