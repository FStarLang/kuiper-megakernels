
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

void KPlay_Functional_kplay_run(__nv_bfloat16 *w, float *x, float *rc,
    float *cv, float *xo, float *rco, float *cvo, uint32_t wn, int64_t base,
    int64_t dn);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
