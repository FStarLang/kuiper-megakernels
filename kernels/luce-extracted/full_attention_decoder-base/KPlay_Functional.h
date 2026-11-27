
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

uint32_t KPlay_Functional_i64_to_sz(int64_t t);

void KPlay_Functional_t7_scores(uint32_t nkv, uint32_t c, __nv_bfloat16 *kvo,
    __nv_bfloat16 *qn, __nv_bfloat16 *dots);

void KPlay_Functional_t7_values(uint32_t nkv, uint32_t c, __nv_bfloat16 *kvo,
    __nv_bfloat16 *probs, float *vals);

void KPlay_Functional_run_main(uint32_t nkv, uint32_t c, uint32_t ts,
    __nv_bfloat16 *w, __nv_bfloat16 *x, __nv_bfloat16 *kv, __nv_bfloat16 *cs,
    __nv_bfloat16 *sn, __nv_bfloat16 *xo, __nv_bfloat16 *kvo);

void KPlay_Functional_kplay_run(uint32_t nkv, __nv_bfloat16 *w,
    __nv_bfloat16 *x, __nv_bfloat16 *kv, __nv_bfloat16 *cs, __nv_bfloat16 *sn,
    int64_t t, __nv_bfloat16 *xo, __nv_bfloat16 *kvo);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
