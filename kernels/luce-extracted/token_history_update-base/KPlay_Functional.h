
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

void KPlay_Functional_kplay_run(uint32_t n, int64_t *old_tokens,
    __nv_bfloat16 *old_history, __nv_bfloat16 *hidden, int64_t *selected,
    int64_t pP, int64_t t, int64_t *tokens, __nv_bfloat16 *history);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
