
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

void KPlay_Functional_kplay_run(__nv_bfloat16 *table, int64_t *prompt,
    uint32_t p, int64_t steps, __nv_bfloat16 *emb, __nv_bfloat16 *cache,
    float *recurrent, int64_t *tokens, __nv_bfloat16 *history);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
