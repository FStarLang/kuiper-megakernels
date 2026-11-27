
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

void KPlay_Functional_kplay_run(__nv_bfloat16 *table, int64_t *prompt,
    uint32_t plen, int64_t *tokens, uint32_t tlen, int64_t position,
    float *hidden, float *cosine, float *sine);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
