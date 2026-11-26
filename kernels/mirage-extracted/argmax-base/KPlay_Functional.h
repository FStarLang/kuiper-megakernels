
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

void KPlay_Functional_kplay_run(uint32_t n0, uint32_t n1, int64_t *input0,
    __nv_bfloat16 *input1, int64_t input2, int64_t *output0, int64_t *output1);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
