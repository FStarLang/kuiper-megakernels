
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

void KPlay_Functional_h_wcopy(uint32_t rt, uint32_t kt, __nv_bfloat16 *src,
    uint32_t off, __nv_bfloat16 *dst);

void KPlay_Functional_h_pad(
    uint32_t kt, __nv_bfloat16 *src, __nv_bfloat16 *dst);

void KPlay_Functional_h_mma(uint32_t rt, uint32_t kt, uint32_t k0, uint32_t k1,
    __nv_bfloat16 *w, __nv_bfloat16 *x, float *c);

void KPlay_Functional_h_proj(uint32_t rt, uint32_t kt, __nv_bfloat16 *w,
    __nv_bfloat16 *x, float *c, __nv_bfloat16 *out);

void KPlay_Functional_h_proj_split(uint32_t rt, uint32_t kt, uint32_t kh,
    __nv_bfloat16 *w, __nv_bfloat16 *x, float *c, float *c_,
    __nv_bfloat16 *out);

void KPlay_Functional_h_silu(__nv_bfloat16 *gu, __nv_bfloat16 *out);

void KPlay_Functional_h_attqk(__nv_bfloat16 *w, uint32_t off,
    __nv_bfloat16 *qkv, __nv_bfloat16 *rot, __nv_bfloat16 *qk,
    __nv_bfloat16 *nrm, float *partial);

void KPlay_Functional_h_attsv(
    __nv_bfloat16 *qk, __nv_bfloat16 *qkv, __nv_bfloat16 *out, float *sc);

void KPlay_Functional_h_resnorm(__nv_bfloat16 *w, uint32_t noff,
    __nv_bfloat16 *res, __nv_bfloat16 *ua, __nv_bfloat16 *summed,
    __nv_bfloat16 *normd, float *partial);

void KPlay_Functional_kplay_run(uint32_t n0, __nv_bfloat16 *input0,
    __nv_bfloat16 *input1, __nv_bfloat16 *input2, __nv_bfloat16 *input3,
    int64_t input4, int64_t input5, __nv_bfloat16 *output0,
    __nv_bfloat16 *output1);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
