
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

__nv_bfloat16 KPlay_Functional_zero_fn(__nv_bfloat16 uu___);

void KPlay_Functional_kplay_run(__nv_bfloat16 *input0, __nv_bfloat16 *input1,
    __nv_bfloat16 *output0, __nv_bfloat16 *output1);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
