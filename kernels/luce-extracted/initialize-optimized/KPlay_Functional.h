
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

void KPlay_Functional_kplay_run(int64_t *prompt, uint32_t plen, int64_t steps,
    float *rc, float *cv, __nv_bfloat16 *k, __nv_bfloat16 *v, int64_t *tokens,
    __nv_bfloat16 *history);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
