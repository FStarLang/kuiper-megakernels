
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

void KPlay_Functional_kplay_run(__nv_bfloat16 *table, __nv_bfloat16 *normw,
    __nv_bfloat16 *residual, __nv_bfloat16 *hidden, __nv_bfloat16 *logits);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
