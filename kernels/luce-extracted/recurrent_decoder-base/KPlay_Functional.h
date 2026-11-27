
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

void KPlay_Functional_h_copy(__nv_bfloat16 *x, __nv_bfloat16 *o);

void KPlay_Functional_h_norm(__nv_bfloat16 *res, __nv_bfloat16 *w,
    uint32_t woff, __nv_bfloat16 *inp, float *vals, float *part);

void KPlay_Functional_h_bfill(
    __nv_bfloat16 *inp, __nv_bfloat16 *bb, uint32_t k);

void KPlay_Functional_h_beta(__nv_bfloat16 *prj, __nv_bfloat16 *beta);

void KPlay_Functional_h_conv(__nv_bfloat16 *w, __nv_bfloat16 *conv,
    __nv_bfloat16 *out1, __nv_bfloat16 *qkv);

void KPlay_Functional_h_qknorm(__nv_bfloat16 *qkv, float *vals, float *part);

void KPlay_Functional_h_rec(__nv_bfloat16 *w, __nv_bfloat16 *qkv,
    __nv_bfloat16 *prj, __nv_bfloat16 *beta, float *st, float *out2,
    __nv_bfloat16 *core, float *upd, float *prod);

void KPlay_Functional_h_gnorm(__nv_bfloat16 *w, __nv_bfloat16 *core,
    __nv_bfloat16 *z, __nv_bfloat16 *inp, float *vals, float *part);

void KPlay_Functional_h_resadd(__nv_bfloat16 *res, __nv_bfloat16 *prj);

void KPlay_Functional_h_silumul(
    __nv_bfloat16 *gate, __nv_bfloat16 *prj, __nv_bfloat16 *inp);

void KPlay_Functional_proj(__nv_bfloat16 *w, __nv_bfloat16 *bb, float *cc,
    __nv_bfloat16 *dst, uint32_t woff, uint32_t kk, uint32_t nt, uint32_t nch,
    uint32_t chunk);

void KPlay_Functional_kplay_run(__nv_bfloat16 *w, __nv_bfloat16 *x,
    __nv_bfloat16 *conv, float *st, __nv_bfloat16 *out0, __nv_bfloat16 *out1,
    float *out2);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
