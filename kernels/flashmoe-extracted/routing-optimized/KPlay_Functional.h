
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

void KPlay_Functional_kplay_run(float *row, int32_t *sel, float *sp,
    int32_t *counts, int64_t *idx, float *probs, int32_t *surv, int64_t token,
    int32_t *out_counts, int64_t *out_idx, float *out_probs, int32_t *out_surv);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
