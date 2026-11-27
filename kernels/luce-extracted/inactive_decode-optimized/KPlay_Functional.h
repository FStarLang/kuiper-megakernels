
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

void KPlay_Functional_kplay_run(float *state_in, float *conv_in,
    __nv_bfloat16 *key_in, __nv_bfloat16 *value_in, uint32_t steps,
    uint32_t hist_n, int64_t *tokens_in, __nv_bfloat16 *history_in,
    float *state_out, float *conv_out, __nv_bfloat16 *key_out,
    __nv_bfloat16 *value_out, int64_t *tokens_out, __nv_bfloat16 *history_out,
    int64_t p, int64_t l);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
