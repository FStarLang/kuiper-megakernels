
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

#include "kplay_cpu_exp.cuh"
__device__ extern float kplay_cpu_exp_f32(float x0);

void KPlay_Functional_kplay_run(__nv_bfloat16 *x, __nv_bfloat16 *w,
    float *softmax, int64_t *prefix, int64_t *token_indices,
    float *probabilities, int32_t *survival);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
