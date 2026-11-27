
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(float *x, __nv_bfloat16 *nw, bool valid,
    __nv_bfloat16 *hout, uint32_t du, __nv_bfloat16 *tab, float *sc)
{
    float warps[32U];
    memset(warps, 0U, 32U * sizeof(float));
    float lanes[32U];
    memset(lanes, 0U, 32U * sizeof(float));
    uint32_t w = 0U;
    while (w < 16U) {
        uint32_t vw = w;
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t vl = l;
            uint32_t i0 = 32U * vw + vl;
            float a = x[i0];
            float s1 = kpr_f32_fma_rn_ftz(a, a, 0.0f);
            float b = x[i0 + 512U];
            lanes[vl] = kpr_f32_fma_rn_ftz(b, b, s1);
            l = vl + 1U;
        }
        uint32_t i = 0U;
        while (i < 16U) {
            uint32_t vi = i;
            lanes[vi] = kpr_f32_add_rn_ftz(lanes[vi], lanes[vi + 16U]);
            i = vi + 1U;
        }
        uint32_t i1 = 0U;
        while (i1 < 8U) {
            uint32_t vi = i1;
            lanes[vi] = kpr_f32_add_rn_ftz(lanes[vi], lanes[vi + 8U]);
            i1 = vi + 1U;
        }
        uint32_t i2 = 0U;
        while (i2 < 4U) {
            uint32_t vi = i2;
            lanes[vi] = kpr_f32_add_rn_ftz(lanes[vi], lanes[vi + 4U]);
            i2 = vi + 1U;
        }
        uint32_t i3 = 0U;
        while (i3 < 2U) {
            uint32_t vi = i3;
            lanes[vi] = kpr_f32_add_rn_ftz(lanes[vi], lanes[vi + 2U]);
            i3 = vi + 1U;
        }
        uint32_t i4 = 0U;
        while (i4 < 1U) {
            uint32_t vi = i4;
            lanes[vi] = kpr_f32_add_rn_ftz(lanes[vi], lanes[vi + 1U]);
            i4 = vi + 1U;
        }
        warps[vw] = lanes[0U];
        w = vw + 1U;
    }
    uint32_t i = 0U;
    while (i < 16U) {
        uint32_t vi = i;
        warps[vi] = kpr_f32_add_rn_ftz(warps[vi], warps[vi + 16U]);
        i = vi + 1U;
    }
    uint32_t i1 = 0U;
    while (i1 < 8U) {
        uint32_t vi = i1;
        warps[vi] = kpr_f32_add_rn_ftz(warps[vi], warps[vi + 8U]);
        i1 = vi + 1U;
    }
    uint32_t i2 = 0U;
    while (i2 < 4U) {
        uint32_t vi = i2;
        warps[vi] = kpr_f32_add_rn_ftz(warps[vi], warps[vi + 4U]);
        i2 = vi + 1U;
    }
    uint32_t i3 = 0U;
    while (i3 < 2U) {
        uint32_t vi = i3;
        warps[vi] = kpr_f32_add_rn_ftz(warps[vi], warps[vi + 2U]);
        i3 = vi + 1U;
    }
    uint32_t i4 = 0U;
    while (i4 < 1U) {
        uint32_t vi = i4;
        warps[vi] = kpr_f32_add_rn_ftz(warps[vi], warps[vi + 1U]);
        i4 = vi + 1U;
    }
    float scale = kpr_f32_rsqrt_approx_ftz(
        kpr_f32_add_rn_ftz(__fdividef(warps[0U], 1024.0f),
            0.0000009999999974752427078783512115478515625f));
    float norm[1024U];
    memset(norm, 0U, 1024U * sizeof(float));
    uint32_t i5 = 0U;
    while (i5 < 1024U) {
        uint32_t vi = i5;
        float t1 = kpr_f32_mul_rn_ftz(x[vi], scale);
        norm[vi] = kpr_f32_mul_rn_ftz(
            t1, kpr_f32_add_rn_ftz(1.0f, __bfloat162float(nw[vi])));
        i5 = vi + 1U;
    }
    if (valid) {
        uint32_t j = 0U;
        while (j < 1024U) {
            uint32_t vj = j;
            hout[du * 1024U + vj] = __float2bfloat16(x[vj]);
            j = vj + 1U;
        }
    }
    uint32_t m = 0U;
    while (m < (uint32_t) 248320U) {
        uint32_t vm = m;
        float lanes1[32U];
        memset(lanes1, 0U, 32U * sizeof(float));
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t vl = l;
            float s = 0.0f;
            uint32_t c = 0U;
            while (c < 4U) {
                uint32_t vc = c;
                uint32_t j = 0U;
                while (j < 8U) {
                    uint32_t vj = j;
                    uint32_t k = 8U * vl + 256U * vc + vj;
                    float a1 = __bfloat162float(tab[vm * 1024U + k]);
                    s = kpr_f32_fma_rn_ftz(a1, norm[k], s);
                    j = vj + 1U;
                }
                c = vc + 1U;
            }
            lanes1[vl] = s;
            l = vl + 1U;
        }
        uint32_t i6 = 0U;
        while (i6 < 16U) {
            uint32_t vi = i6;
            lanes1[vi] = kpr_f32_add_rn_ftz(lanes1[vi], lanes1[vi + 16U]);
            i6 = vi + 1U;
        }
        uint32_t i7 = 0U;
        while (i7 < 8U) {
            uint32_t vi = i7;
            lanes1[vi] = kpr_f32_add_rn_ftz(lanes1[vi], lanes1[vi + 8U]);
            i7 = vi + 1U;
        }
        uint32_t i8 = 0U;
        while (i8 < 4U) {
            uint32_t vi = i8;
            lanes1[vi] = kpr_f32_add_rn_ftz(lanes1[vi], lanes1[vi + 4U]);
            i8 = vi + 1U;
        }
        uint32_t i9 = 0U;
        while (i9 < 2U) {
            uint32_t vi = i9;
            lanes1[vi] = kpr_f32_add_rn_ftz(lanes1[vi], lanes1[vi + 2U]);
            i9 = vi + 1U;
        }
        uint32_t i10 = 0U;
        while (i10 < 1U) {
            uint32_t vi = i10;
            lanes1[vi] = kpr_f32_add_rn_ftz(lanes1[vi], lanes1[vi + 1U]);
            i10 = vi + 1U;
        }
        sc[vm] = lanes1[0U];
        m = vm + 1U;
    }
}

prims_list____ *KPlay_Functional_et(KPlay_Functional_env r) { return r.rt; }

void KPlay_Functional_kplay_run(uint32_t hn, __nv_bfloat16 *tab,
    __nv_bfloat16 *nw, float *x, __nv_bfloat16 *hin, float *sc,
    __nv_bfloat16 *hout, int64_t d)
{
    MUST(cudaMemcpy(hout, hin, (uint64_t) sizeof(__nv_bfloat16) * (uint64_t) hn,
        cudaMemcpyDeviceToDevice));
    bool ok = d >= 0LL && d < 4194304LL;
    uint32_t du = ok ? (uint32_t) d : 0U;
    bool valid = ok && du < hn / 1024U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(
        __hoisted_kplay_run_0, 1U, 1U, 0U, s, x, nw, valid, hout, du, tab, sc);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
