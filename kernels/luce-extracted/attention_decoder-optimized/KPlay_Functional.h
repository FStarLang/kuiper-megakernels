
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

void KPlay_Functional_kplay_run(__nv_bfloat16 *w, float *x, __nv_bfloat16 *k,
    __nv_bfloat16 *v, float *cs, float *sn, float *xo, __nv_bfloat16 *ko,
    __nv_bfloat16 *vo, uint32_t wn, int64_t base, int64_t fa, int64_t pos);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
