
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

void KPlay_Functional_kplay_run(__nv_bfloat16 *x, __nv_bfloat16 *w, float *sc,
    int32_t *cn, int64_t *ix, float *pr, int32_t *sv);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
