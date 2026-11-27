
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

__nv_bfloat16 KPlay_Functional_cos_lane(int64_t t, uint32_t i);

__nv_bfloat16 KPlay_Functional_sin_lane(int64_t t, uint32_t i);

void KPlay_Functional_kplay_run(uint32_t tab_len, uint32_t emb_len,
    uint32_t tok_len, __nv_bfloat16 *table, __nv_bfloat16 *emb, int64_t *tokens,
    int64_t t, __nv_bfloat16 *residual, __nv_bfloat16 *cos_out,
    __nv_bfloat16 *sin_out);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
