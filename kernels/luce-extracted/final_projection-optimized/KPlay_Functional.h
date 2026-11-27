
#ifndef KPlay_Functional_H
#define KPlay_Functional_H

#include <kuiper.h>

typedef struct prims_list_____s prims_list____;

typedef enum { prims_Nil, prims_Cons } prims_list_____tags;

typedef struct prims_list_____s {
    prims_list_____tags tag;
    prims_list____ *tl;
} prims_list____;

typedef struct KPlay_Functional_env_s {
    prims_list____ *rt;
    prims_list____ *rn;
    prims_list____ *rx;
} KPlay_Functional_env;

prims_list____ *KPlay_Functional_et(KPlay_Functional_env r);

void KPlay_Functional_kplay_run(uint32_t hn, __nv_bfloat16 *tab,
    __nv_bfloat16 *nw, float *x, __nv_bfloat16 *hin, float *sc,
    __nv_bfloat16 *hout, int64_t d);

#define KPlay_Functional_H_DEFINED
#endif /* KPlay_Functional_H */
