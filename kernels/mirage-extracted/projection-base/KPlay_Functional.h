
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

extern krml_checked_int_t KPlay_Functional_nv;

extern krml_checked_int_t KPlay_Functional_ntiles;

void Kuiper_Shareable_double_shareable__Mkshareable(
    __nv_bfloat16 *uu___, __nv_bfloat16 *uu___1);

void KPlay_Functional_inframe_shareable(
    krml_checked_int_t n0, __nv_bfloat16 *uu___, __nv_bfloat16 *uu___1);

void KPlay_Functional_kplay_run(uint32_t n0, __nv_bfloat16 *in0,
    __nv_bfloat16 *in1, int64_t in2, __nv_bfloat16 *out0);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
