
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(
    __nv_bfloat16 *input0, int64_t *input1, int64_t *output1, int64_t *output0)
{
    float values[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        values[_i] = -INFINITY;
    int64_t indices[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        indices[_i] = -1LL;
    float cvals[96U];
    for (uint32_t _i = 0U; _i < 96U; ++_i)
        cvals[_i] = -INFINITY;
    int64_t cidx[96U];
    for (uint32_t _i = 0U; _i < 96U; ++_i)
        cidx[_i] = -1LL;
    uint32_t c = 0U;
    while (c < 96U) {
        uint32_t vc = c;
        uint32_t lim = vc * 1600U + 1600U <= (uint32_t) 151936U ? 1600U
                       : vc * 1600U <= (uint32_t) 151936U
                           ? (uint32_t) 151936U - vc * 1600U
                           : 0U;
        uint32_t t = 0U;
        while (t < 128U) {
            uint32_t vt = t;
            float bv = -INFINITY;
            int64_t bi = -1LL;
            uint32_t i = vt;
            while (i < lim) {
                uint32_t vi = i;
                float cand = __bfloat162float(input0[vc * 1600U + vi]);
                int64_t ci = (int64_t) (uint32_t) vi;
                float cv = bv;
                int64_t cx = bi;
                bool w = !(ci < 0LL) &&
                         (cx < 0LL || (cand != cand ? cv == cv || ci < cx
                                                    : cv == cv && cv < cand));
                bv = w ? cand : cv;
                bi = w ? ci : cx;
                i = vi + 128U;
            }
            int64_t fi = bi;
            values[vt] = bv;
            indices[vt] = fi;
            t = vt + 1U;
        }
        uint32_t w = 0U;
        while (w < 4U) {
            uint32_t vw = w;
            uint32_t delta = 16U;
            while (delta > 0U) {
                uint32_t vd = delta;
                uint32_t t1 = 0U;
                while (t1 < vd) {
                    uint32_t vt = t1;
                    float va = values[32U * vw + vt];
                    int64_t ia = indices[32U * vw + vt];
                    float vb = values[32U * vw + vt + vd];
                    int64_t ib = indices[32U * vw + vt + vd];
                    bool w1 = !(ib < 0LL) &&
                              (ia < 0LL || (vb != vb ? va == va || ib < ia
                                                     : va == va && va < vb));
                    values[32U * vw + vt] = w1 ? vb : va;
                    indices[32U * vw + vt] = w1 ? ib : ia;
                    t1 = vt + 1U;
                }
                delta = vd / 2U;
            }
            int64_t i0 = indices[32U * vw];
            values[vw] = values[32U * vw];
            indices[vw] = i0;
            w = vw + 1U;
        }
        uint32_t delta = 2U;
        while (delta > 0U) {
            uint32_t vd = delta;
            uint32_t t1 = 0U;
            while (t1 < vd) {
                uint32_t vt = t1;
                float va = values[vt];
                int64_t ia = indices[vt];
                float vb = values[vt + vd];
                int64_t ib = indices[vt + vd];
                bool w1 = !(ib < 0LL) &&
                          (ia < 0LL || (vb != vb ? va == va || ib < ia
                                                 : va == va && va < vb));
                values[vt] = w1 ? vb : va;
                indices[vt] = w1 ? ib : ia;
                t1 = vt + 1U;
            }
            delta = vd / 2U;
        }
        float v0 = values[0U];
        int64_t i0 = indices[0U];
        int64_t e =
            i0 < 0LL ? -1LL : (int64_t) (uint32_t) vc * 4294967296LL + i0;
        cvals[vc] = v0;
        cidx[vc] = e;
        c = vc + 1U;
    }
    uint32_t t = 0U;
    while (t < 256U) {
        uint32_t vt = t;
        float __anf0 = vt < 96U ? cvals[vt] : -INFINITY;
        int64_t __anf01 = vt < 96U ? cidx[vt] : -1LL;
        values[vt] = __anf0;
        indices[vt] = __anf01;
        t = vt + 1U;
    }
    uint32_t w = 0U;
    while (w < 8U) {
        uint32_t vw = w;
        uint32_t delta = 16U;
        while (delta > 0U) {
            uint32_t vd = delta;
            uint32_t t1 = 0U;
            while (t1 < vd) {
                uint32_t vt = t1;
                float va = values[32U * vw + vt];
                int64_t ia = indices[32U * vw + vt];
                float vb = values[32U * vw + vt + vd];
                int64_t ib = indices[32U * vw + vt + vd];
                bool w1 = !(ib < 0LL) &&
                          (ia < 0LL || (vb != vb ? va == va || ib < ia
                                                 : va == va && va < vb));
                values[32U * vw + vt] = w1 ? vb : va;
                indices[32U * vw + vt] = w1 ? ib : ia;
                t1 = vt + 1U;
            }
            delta = vd / 2U;
        }
        int64_t i0 = indices[32U * vw];
        values[vw] = values[32U * vw];
        indices[vw] = i0;
        w = vw + 1U;
    }
    uint32_t delta = 4U;
    while (delta > 0U) {
        uint32_t vd = delta;
        uint32_t t1 = 0U;
        while (t1 < vd) {
            uint32_t vt = t1;
            float va = values[vt];
            int64_t ia = indices[vt];
            float vb = values[vt + vd];
            int64_t ib = indices[vt + vd];
            bool w1 =
                !(ib < 0LL) && (ia < 0LL || (vb != vb ? va == va || ib < ia
                                                      : va == va && va < vb));
            values[vt] = w1 ? vb : va;
            indices[vt] = w1 ? ib : ia;
            t1 = vt + 1U;
        }
        delta = vd / 2U;
    }
    int64_t e = indices[0U];
    int64_t tok = e < 0LL ? 0LL : e / 4294967296LL * 1600LL + e % 4294967296LL;
    uint32_t i = 0U;
    while (i < 7U) {
        uint32_t vi = i;
        output1[vi] = input1[vi];
        i = vi + 1U;
    }
    output1[7U] = tok;
    output0[0U] = output1[7U];
}

void KPlay_Functional_kplay_run(
    __nv_bfloat16 *input0, int64_t *input1, int64_t *output0, int64_t *output1)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(
        __hoisted_kplay_run_0, 1U, 1U, 0U, s, input0, input1, output1, output0);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
