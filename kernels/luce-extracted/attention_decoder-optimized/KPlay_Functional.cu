
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(float *x, float *lanes, float *warps, __nv_bfloat16 *w,
    uint32_t b, float *norm1)
{
    uint32_t w1 = 0U;
    while (w1 < 16U) {
        uint32_t vw = w1;
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t vl = l;
            float s = 0.0f;
            uint32_t k = 0U;
            while (k < 2U) {
                uint32_t vk = k;
                float a = x[vw * 32U + vl + vk * 512U];
                s = kpr_f32_fma_rn_ftz(a, a, s);
                k = vk + 1U;
            }
            lanes[vl] = s;
            l = vl + 1U;
        }
        uint32_t i = 0U;
        while (i < 16U) {
            uint32_t k = i;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 16U]);
            i = k + 1U;
        }
        uint32_t i1 = 0U;
        while (i1 < 8U) {
            uint32_t k = i1;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 8U]);
            i1 = k + 1U;
        }
        uint32_t i2 = 0U;
        while (i2 < 4U) {
            uint32_t k = i2;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 4U]);
            i2 = k + 1U;
        }
        uint32_t i3 = 0U;
        while (i3 < 2U) {
            uint32_t k = i3;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 2U]);
            i3 = k + 1U;
        }
        uint32_t i4 = 0U;
        while (i4 < 1U) {
            uint32_t k = i4;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 1U]);
            i4 = k + 1U;
        }
        warps[vw] = lanes[0U];
        w1 = vw + 1U;
    }
    uint32_t j = 16U;
    while (j < 32U) {
        uint32_t vj = j;
        warps[vj] = 0.0f;
        j = vj + 1U;
    }
    uint32_t i = 0U;
    while (i < 16U) {
        uint32_t k = i;
        warps[k] = kpr_f32_add_rn_ftz(warps[k], warps[k + 16U]);
        i = k + 1U;
    }
    uint32_t i1 = 0U;
    while (i1 < 8U) {
        uint32_t k = i1;
        warps[k] = kpr_f32_add_rn_ftz(warps[k], warps[k + 8U]);
        i1 = k + 1U;
    }
    uint32_t i2 = 0U;
    while (i2 < 4U) {
        uint32_t k = i2;
        warps[k] = kpr_f32_add_rn_ftz(warps[k], warps[k + 4U]);
        i2 = k + 1U;
    }
    uint32_t i3 = 0U;
    while (i3 < 2U) {
        uint32_t k = i3;
        warps[k] = kpr_f32_add_rn_ftz(warps[k], warps[k + 2U]);
        i3 = k + 1U;
    }
    uint32_t i4 = 0U;
    while (i4 < 1U) {
        uint32_t k = i4;
        warps[k] = kpr_f32_add_rn_ftz(warps[k], warps[k + 1U]);
        i4 = k + 1U;
    }
    float ss = warps[0U];
    float scale = kpr_f32_rsqrt_approx_ftz(kpr_f32_add_rn_ftz(
        kpr_f32_mul_rn_ftz(ss, kpr_f32_rcp_approx_ftz(1024.0f)),
        0.0000009999999974752427078783512115478515625f));
    uint32_t i5 = 0U;
    while (i5 < 1024U) {
        uint32_t vi = i5;
        float xi = x[vi];
        float cw = __bfloat162float(w[b + vi]);
        float m1 = kpr_f32_mul_rn_ftz(xi, scale);
        norm1[vi] = __bfloat162float(__float2bfloat16(
            kpr_f32_mul_rn_ftz(m1, kpr_f32_add_rn_ftz(1.0f, cw))));
        i5 = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_1(
    __nv_bfloat16 *w, uint32_t mix, float *norm1, float *lanes, float *proj)
{
    uint32_t i = 0U;
    while (i < 4096U) {
        uint32_t vi = i;
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t vl = l;
            float s = 0.0f;
            uint32_t c = 0U;
            while (c < 4U) {
                uint32_t vc = c;
                uint32_t kk = vc * 256U + vl * 8U;
                float part = 0.0f;
                uint32_t j = 0U;
                while (j < 8U) {
                    uint32_t vj = j;
                    float b = norm1[kk + vj];
                    float vp = part;
                    part = kpr_f32_fma_rn_ftz(
                        __bfloat162float(w[mix + vi * 1024U + kk + vj]), b, vp);
                    j = vj + 1U;
                }
                s = kpr_f32_add_rn_ftz(s, part);
                c = vc + 1U;
            }
            lanes[vl] = s;
            l = vl + 1U;
        }
        uint32_t i1 = 0U;
        while (i1 < 16U) {
            uint32_t k = i1;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 16U]);
            i1 = k + 1U;
        }
        uint32_t i2 = 0U;
        while (i2 < 8U) {
            uint32_t k = i2;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 8U]);
            i2 = k + 1U;
        }
        uint32_t i3 = 0U;
        while (i3 < 4U) {
            uint32_t k = i3;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 4U]);
            i3 = k + 1U;
        }
        uint32_t i4 = 0U;
        while (i4 < 2U) {
            uint32_t k = i4;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 2U]);
            i4 = k + 1U;
        }
        uint32_t i5 = 0U;
        while (i5 < 1U) {
            uint32_t k = i5;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 1U]);
            i5 = k + 1U;
        }
        proj[vi] = lanes[0U];
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_2(
    __nv_bfloat16 *w, uint32_t mix, float *norm1, float *lanes, float *zk)
{
    uint32_t i = 0U;
    while (i < 512U) {
        uint32_t vi = i;
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t vl = l;
            float s = 0.0f;
            uint32_t c = 0U;
            while (c < 4U) {
                uint32_t vc = c;
                uint32_t kk = vc * 256U + vl * 8U;
                float part = 0.0f;
                uint32_t j = 0U;
                while (j < 8U) {
                    uint32_t vj = j;
                    float b = norm1[kk + vj];
                    float vp = part;
                    part = kpr_f32_fma_rn_ftz(
                        __bfloat162float(w[mix + (uint32_t) 4194304U +
                                           vi * 1024U + kk + vj]),
                        b, vp);
                    j = vj + 1U;
                }
                s = kpr_f32_add_rn_ftz(s, part);
                c = vc + 1U;
            }
            lanes[vl] = s;
            l = vl + 1U;
        }
        uint32_t i1 = 0U;
        while (i1 < 16U) {
            uint32_t k = i1;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 16U]);
            i1 = k + 1U;
        }
        uint32_t i2 = 0U;
        while (i2 < 8U) {
            uint32_t k = i2;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 8U]);
            i2 = k + 1U;
        }
        uint32_t i3 = 0U;
        while (i3 < 4U) {
            uint32_t k = i3;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 4U]);
            i3 = k + 1U;
        }
        uint32_t i4 = 0U;
        while (i4 < 2U) {
            uint32_t k = i4;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 2U]);
            i4 = k + 1U;
        }
        uint32_t i5 = 0U;
        while (i5 < 1U) {
            uint32_t k = i5;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 1U]);
            i5 = k + 1U;
        }
        zk[vi] = lanes[0U];
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_3(
    __nv_bfloat16 *w, uint32_t mix, float *norm1, float *lanes, float *zv)
{
    uint32_t i = 0U;
    while (i < 512U) {
        uint32_t vi = i;
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t vl = l;
            float s = 0.0f;
            uint32_t c = 0U;
            while (c < 4U) {
                uint32_t vc = c;
                uint32_t kk = vc * 256U + vl * 8U;
                float part = 0.0f;
                uint32_t j = 0U;
                while (j < 8U) {
                    uint32_t vj = j;
                    float b = norm1[kk + vj];
                    float vp = part;
                    part = kpr_f32_fma_rn_ftz(
                        __bfloat162float(w[mix + (uint32_t) 4718592U +
                                           vi * 1024U + kk + vj]),
                        b, vp);
                    j = vj + 1U;
                }
                s = kpr_f32_add_rn_ftz(s, part);
                c = vc + 1U;
            }
            lanes[vl] = s;
            l = vl + 1U;
        }
        uint32_t i1 = 0U;
        while (i1 < 16U) {
            uint32_t k = i1;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 16U]);
            i1 = k + 1U;
        }
        uint32_t i2 = 0U;
        while (i2 < 8U) {
            uint32_t k = i2;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 8U]);
            i2 = k + 1U;
        }
        uint32_t i3 = 0U;
        while (i3 < 4U) {
            uint32_t k = i3;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 4U]);
            i3 = k + 1U;
        }
        uint32_t i4 = 0U;
        while (i4 < 2U) {
            uint32_t k = i4;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 2U]);
            i4 = k + 1U;
        }
        uint32_t i5 = 0U;
        while (i5 < 1U) {
            uint32_t k = i5;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 1U]);
            i5 = k + 1U;
        }
        zv[vi] = lanes[0U];
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_4(uint32_t mix, float *zk, float *lanes, __nv_bfloat16 *w,
    float *cs, float *sn, float *kn, float *proj, float *qn)
{
    uint32_t h = 0U;
    while (h < 2U) {
        uint32_t vh = h;
        uint32_t off = vh * 256U;
        uint32_t wk = mix + (uint32_t) 5243136U;
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t vl = l;
            float s = 0.0f;
            uint32_t k = 0U;
            while (k < 8U) {
                uint32_t vk = k;
                float a = zk[off + vl + vk * 32U];
                s = kpr_f32_fma_rn_ftz(a, a, s);
                k = vk + 1U;
            }
            lanes[vl] = s;
            l = vl + 1U;
        }
        uint32_t i = 0U;
        while (i < 16U) {
            uint32_t k = i;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 16U]);
            i = k + 1U;
        }
        uint32_t i1 = 0U;
        while (i1 < 8U) {
            uint32_t k = i1;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 8U]);
            i1 = k + 1U;
        }
        uint32_t i2 = 0U;
        while (i2 < 4U) {
            uint32_t k = i2;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 4U]);
            i2 = k + 1U;
        }
        uint32_t i3 = 0U;
        while (i3 < 2U) {
            uint32_t k = i3;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 2U]);
            i3 = k + 1U;
        }
        uint32_t i4 = 0U;
        while (i4 < 1U) {
            uint32_t k = i4;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 1U]);
            i4 = k + 1U;
        }
        float ss = lanes[0U];
        float rstd = kpr_f32_rsqrt_approx_ftz(kpr_f32_add_rn_ftz(
            kpr_f32_mul_rn_ftz(ss, kpr_f32_rcp_approx_ftz(256.0f)),
            0.0000009999999974752427078783512115478515625f));
        uint32_t i5 = 0U;
        while (i5 < 256U) {
            uint32_t vi = i5;
            float ite;
            if (vi < 32U) {
                float a1 = zk[off + vi];
                float cb = __bfloat162float(w[wk + vi]);
                float m1 = kpr_f32_mul_rn_ftz(a1, rstd);
                float u0 = kpr_f32_mul_rn_ftz(m1, kpr_f32_add_rn_ftz(1.0f, cb));
                float a2 = zk[off + (vi + 32U)];
                float cb1 = __bfloat162float(w[wk + (vi + 32U)]);
                float m11 = kpr_f32_mul_rn_ftz(a2, rstd);
                float u1 =
                    kpr_f32_mul_rn_ftz(m11, kpr_f32_add_rn_ftz(1.0f, cb1));
                float c1 = cs[vi];
                ite =
                    kpr_f32_fma_rn_ftz(u0, c1, -kpr_f32_mul_rn_ftz(u1, sn[vi]));
            } else if (vi < 64U) {
                float a1 = zk[off + (vi - 32U)];
                float cb = __bfloat162float(w[wk + (vi - 32U)]);
                float m1 = kpr_f32_mul_rn_ftz(a1, rstd);
                float u0 = kpr_f32_mul_rn_ftz(m1, kpr_f32_add_rn_ftz(1.0f, cb));
                float a2 = zk[off + vi];
                float cb1 = __bfloat162float(w[wk + vi]);
                float m11 = kpr_f32_mul_rn_ftz(a2, rstd);
                float u1 =
                    kpr_f32_mul_rn_ftz(m11, kpr_f32_add_rn_ftz(1.0f, cb1));
                float sv = sn[vi - 32U];
                ite = kpr_f32_fma_rn_ftz(
                    u0, sv, kpr_f32_mul_rn_ftz(u1, cs[vi - 32U]));
            } else {
                float a1 = zk[off + vi];
                float cb = __bfloat162float(w[wk + vi]);
                float m1 = kpr_f32_mul_rn_ftz(a1, rstd);
                ite = kpr_f32_mul_rn_ftz(m1, kpr_f32_add_rn_ftz(1.0f, cb));
            }
            kn[off + vi] = ite;
            i5 = vi + 1U;
        }
        h = vh + 1U;
    }
    uint32_t q = 0U;
    while (q < 8U) {
        uint32_t vq = q;
        uint32_t ro = vq * 512U;
        uint32_t off = vq * 256U;
        uint32_t wq = mix + (uint32_t) 5242880U;
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t vl = l;
            float s = 0.0f;
            uint32_t k = 0U;
            while (k < 8U) {
                uint32_t vk = k;
                float a = proj[ro + vl + vk * 32U];
                s = kpr_f32_fma_rn_ftz(a, a, s);
                k = vk + 1U;
            }
            lanes[vl] = s;
            l = vl + 1U;
        }
        uint32_t i = 0U;
        while (i < 16U) {
            uint32_t k = i;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 16U]);
            i = k + 1U;
        }
        uint32_t i1 = 0U;
        while (i1 < 8U) {
            uint32_t k = i1;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 8U]);
            i1 = k + 1U;
        }
        uint32_t i2 = 0U;
        while (i2 < 4U) {
            uint32_t k = i2;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 4U]);
            i2 = k + 1U;
        }
        uint32_t i3 = 0U;
        while (i3 < 2U) {
            uint32_t k = i3;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 2U]);
            i3 = k + 1U;
        }
        uint32_t i4 = 0U;
        while (i4 < 1U) {
            uint32_t k = i4;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 1U]);
            i4 = k + 1U;
        }
        float ss = lanes[0U];
        float rstd = kpr_f32_rsqrt_approx_ftz(kpr_f32_add_rn_ftz(
            kpr_f32_mul_rn_ftz(ss, kpr_f32_rcp_approx_ftz(256.0f)),
            0.0000009999999974752427078783512115478515625f));
        uint32_t i5 = 0U;
        while (i5 < 256U) {
            uint32_t vi = i5;
            float ite;
            if (vi < 32U) {
                float a1 = proj[ro + vi];
                float cb = __bfloat162float(w[wq + vi]);
                float m1 = kpr_f32_mul_rn_ftz(a1, rstd);
                float u0 = kpr_f32_mul_rn_ftz(m1, kpr_f32_add_rn_ftz(1.0f, cb));
                float a2 = proj[ro + (vi + 32U)];
                float cb1 = __bfloat162float(w[wq + (vi + 32U)]);
                float m11 = kpr_f32_mul_rn_ftz(a2, rstd);
                float u1 =
                    kpr_f32_mul_rn_ftz(m11, kpr_f32_add_rn_ftz(1.0f, cb1));
                float c1 = cs[vi];
                ite =
                    kpr_f32_fma_rn_ftz(u0, c1, -kpr_f32_mul_rn_ftz(u1, sn[vi]));
            } else if (vi < 64U) {
                float a1 = proj[ro + (vi - 32U)];
                float cb = __bfloat162float(w[wq + (vi - 32U)]);
                float m1 = kpr_f32_mul_rn_ftz(a1, rstd);
                float u0 = kpr_f32_mul_rn_ftz(m1, kpr_f32_add_rn_ftz(1.0f, cb));
                float a2 = proj[ro + vi];
                float cb1 = __bfloat162float(w[wq + vi]);
                float m11 = kpr_f32_mul_rn_ftz(a2, rstd);
                float u1 =
                    kpr_f32_mul_rn_ftz(m11, kpr_f32_add_rn_ftz(1.0f, cb1));
                float sv = sn[vi - 32U];
                ite = kpr_f32_fma_rn_ftz(
                    u0, sv, kpr_f32_mul_rn_ftz(u1, cs[vi - 32U]));
            } else {
                float a1 = proj[ro + vi];
                float cb = __bfloat162float(w[wq + vi]);
                float m1 = kpr_f32_mul_rn_ftz(a1, rstd);
                ite = kpr_f32_mul_rn_ftz(m1, kpr_f32_add_rn_ftz(1.0f, cb));
            }
            qn[off + vi] = ite;
            i5 = vi + 1U;
        }
        q = vq + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_5(uint32_t f, uint32_t p, float *kn, float *zv,
    __nv_bfloat16 *ko, __nv_bfloat16 *vo)
{
    uint32_t c = 0U;
    while (c < 512U) {
        uint32_t vc = c;
        uint32_t ci = f * (uint32_t) 1048576U + vc / 256U * (uint32_t) 524288U +
                      p * 256U + vc % 256U;
        float b = zv[vc];
        ko[ci] = __float2bfloat16(kn[vc]);
        vo[ci] = __float2bfloat16(b);
        c = vc + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_6(uint32_t f, float *partial, uint32_t p, float *qn,
    __nv_bfloat16 *ko, float *lanes, __nv_bfloat16 *vo, float *maxs,
    float *totals, float *factor, float *proj, float *attn)
{
    uint32_t h = 0U;
    while (h < 8U) {
        uint32_t vh = h;
        uint32_t cb = f * (uint32_t) 1048576U + vh / 4U * (uint32_t) 524288U;
        uint32_t wi = 0U;
        while (wi < 16U) {
            uint32_t vw = wi;
            uint32_t i = 0U;
            while (i < 256U) {
                uint32_t vi = i;
                partial[vw * 256U + vi] = 0.0f;
                i = vi + 1U;
            }
            float mx = -INFINITY;
            float sum = 0.0f;
            uint32_t pp = vw;
            while (pp <= p) {
                uint32_t vp = pp;
                uint32_t kb = cb + vp * 256U;
                uint32_t l = 0U;
                while (l < 32U) {
                    uint32_t vl = l;
                    float s = 0.0f;
                    uint32_t e = 0U;
                    while (e < 8U) {
                        uint32_t ve = e;
                        uint32_t i1 = vl * 8U + ve;
                        float a = qn[vh * 256U + i1];
                        float vs = s;
                        s = kpr_f32_fma_rn_ftz(
                            a, __bfloat162float(ko[kb + i1]), vs);
                        e = ve + 1U;
                    }
                    lanes[vl] = s;
                    l = vl + 1U;
                }
                uint32_t i1 = 0U;
                while (i1 < 16U) {
                    uint32_t k = i1;
                    lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 16U]);
                    i1 = k + 1U;
                }
                uint32_t i2 = 0U;
                while (i2 < 8U) {
                    uint32_t k = i2;
                    lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 8U]);
                    i2 = k + 1U;
                }
                uint32_t i3 = 0U;
                while (i3 < 4U) {
                    uint32_t k = i3;
                    lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 4U]);
                    i3 = k + 1U;
                }
                uint32_t i4 = 0U;
                while (i4 < 2U) {
                    uint32_t k = i4;
                    lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 2U]);
                    i4 = k + 1U;
                }
                uint32_t i5 = 0U;
                while (i5 < 1U) {
                    uint32_t k = i5;
                    lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 1U]);
                    i5 = k + 1U;
                }
                float t = lanes[0U];
                float sc = kpr_f32_mul_rn_ftz(t, kpr_f32_rcp_approx_ftz(16.0f));
                float old = mx;
                float m = fmaxf(old, sc);
                mx = m;
                float diff = __expf(kpr_f32_add_rn_ftz(old, -m));
                float wt = __expf(kpr_f32_add_rn_ftz(sc, -m));
                sum = kpr_f32_fma_rn_ftz(sum, diff, wt);
                uint32_t i6 = 0U;
                while (i6 < 256U) {
                    uint32_t vi = i6;
                    uint32_t ix = vw * 256U + vi;
                    float old_p = partial[ix];
                    partial[ix] = kpr_f32_fma_rn_ftz(old_p, diff,
                        kpr_f32_mul_rn_ftz(wt, __bfloat162float(vo[kb + vi])));
                    i6 = vi + 1U;
                }
                pp = vp + 16U;
            }
            float vs = sum;
            maxs[vw] = mx;
            totals[vw] = vs;
            wi = vw + 1U;
        }
        float gm = -INFINITY;
        uint32_t w2 = 0U;
        while (w2 < 16U) {
            uint32_t vw = w2;
            if (vw <= p)
                gm = fmaxf(gm, maxs[vw]);
            w2 = vw + 1U;
        }
        float g = gm;
        float ts = 0.0f;
        uint32_t w3 = 0U;
        while (w3 < 16U) {
            uint32_t vw = w3;
            if (vw <= p) {
                float f1 = __expf(kpr_f32_add_rn_ftz(maxs[vw], -g));
                factor[vw] = f1;
                ts = kpr_f32_fma_rn_ftz(totals[vw], f1, ts);
            }
            w3 = vw + 1U;
        }
        float rcp = kpr_f32_rcp_approx_ftz(ts);
        uint32_t i = 0U;
        while (i < 256U) {
            uint32_t vi = i;
            float s = 0.0f;
            uint32_t w4 = 0U;
            while (w4 < 16U) {
                uint32_t vw = w4;
                if (vw <= p)
                    s = kpr_f32_fma_rn_ftz(
                        partial[vw * 256U + vi], factor[vw], s);
                w4 = vw + 1U;
            }
            float vs = s;
            float sg = kpr_f32_rcp_approx_ftz(
                kpr_f32_add_rn_ftz(1.0f, __expf(-proj[vh * 512U + 256U + vi])));
            attn[vh * 256U + vi] =
                kpr_f32_mul_rn_ftz(kpr_f32_mul_rn_ftz(vs, rcp), sg);
            i = vi + 1U;
        }
        h = vh + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_7(__nv_bfloat16 *w, uint32_t mix, float *attn, float *lanes,
    float *x, float *h1)
{
    uint32_t i = 0U;
    while (i < 1024U) {
        uint32_t vi = i;
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t vl = l;
            float s = 0.0f;
            uint32_t c = 0U;
            while (c < 8U) {
                uint32_t vc = c;
                uint32_t kk = vc * 256U + vl * 8U;
                uint32_t j = 0U;
                while (j < 8U) {
                    uint32_t vj = j;
                    float b = attn[kk + vj];
                    float vs = s;
                    s = kpr_f32_fma_rn_ftz(
                        __bfloat162float(w[mix + (uint32_t) 5243392U +
                                           vi * 2048U + kk + vj]),
                        b, vs);
                    j = vj + 1U;
                }
                c = vc + 1U;
            }
            lanes[vl] = s;
            l = vl + 1U;
        }
        uint32_t i1 = 0U;
        while (i1 < 16U) {
            uint32_t k = i1;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 16U]);
            i1 = k + 1U;
        }
        uint32_t i2 = 0U;
        while (i2 < 8U) {
            uint32_t k = i2;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 8U]);
            i2 = k + 1U;
        }
        uint32_t i3 = 0U;
        while (i3 < 4U) {
            uint32_t k = i3;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 4U]);
            i3 = k + 1U;
        }
        uint32_t i4 = 0U;
        while (i4 < 2U) {
            uint32_t k = i4;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 2U]);
            i4 = k + 1U;
        }
        uint32_t i5 = 0U;
        while (i5 < 1U) {
            uint32_t k = i5;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 1U]);
            i5 = k + 1U;
        }
        h1[vi] = __bfloat162float(
            __float2bfloat16(kpr_f32_add_rn_ftz(lanes[0U], x[vi])));
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_8(float *h1, float *lanes, float *warps, __nv_bfloat16 *w,
    uint32_t b, float *norm2)
{
    uint32_t w1 = 0U;
    while (w1 < 16U) {
        uint32_t vw = w1;
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t vl = l;
            float s = 0.0f;
            uint32_t k = 0U;
            while (k < 2U) {
                uint32_t vk = k;
                float a = h1[vw * 32U + vl + vk * 512U];
                s = kpr_f32_fma_rn_ftz(a, a, s);
                k = vk + 1U;
            }
            lanes[vl] = s;
            l = vl + 1U;
        }
        uint32_t i = 0U;
        while (i < 16U) {
            uint32_t k = i;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 16U]);
            i = k + 1U;
        }
        uint32_t i1 = 0U;
        while (i1 < 8U) {
            uint32_t k = i1;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 8U]);
            i1 = k + 1U;
        }
        uint32_t i2 = 0U;
        while (i2 < 4U) {
            uint32_t k = i2;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 4U]);
            i2 = k + 1U;
        }
        uint32_t i3 = 0U;
        while (i3 < 2U) {
            uint32_t k = i3;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 2U]);
            i3 = k + 1U;
        }
        uint32_t i4 = 0U;
        while (i4 < 1U) {
            uint32_t k = i4;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 1U]);
            i4 = k + 1U;
        }
        warps[vw] = lanes[0U];
        w1 = vw + 1U;
    }
    uint32_t j = 16U;
    while (j < 32U) {
        uint32_t vj = j;
        warps[vj] = 0.0f;
        j = vj + 1U;
    }
    uint32_t i = 0U;
    while (i < 16U) {
        uint32_t k = i;
        warps[k] = kpr_f32_add_rn_ftz(warps[k], warps[k + 16U]);
        i = k + 1U;
    }
    uint32_t i1 = 0U;
    while (i1 < 8U) {
        uint32_t k = i1;
        warps[k] = kpr_f32_add_rn_ftz(warps[k], warps[k + 8U]);
        i1 = k + 1U;
    }
    uint32_t i2 = 0U;
    while (i2 < 4U) {
        uint32_t k = i2;
        warps[k] = kpr_f32_add_rn_ftz(warps[k], warps[k + 4U]);
        i2 = k + 1U;
    }
    uint32_t i3 = 0U;
    while (i3 < 2U) {
        uint32_t k = i3;
        warps[k] = kpr_f32_add_rn_ftz(warps[k], warps[k + 2U]);
        i3 = k + 1U;
    }
    uint32_t i4 = 0U;
    while (i4 < 1U) {
        uint32_t k = i4;
        warps[k] = kpr_f32_add_rn_ftz(warps[k], warps[k + 1U]);
        i4 = k + 1U;
    }
    float ss = warps[0U];
    float scale = kpr_f32_rsqrt_approx_ftz(kpr_f32_add_rn_ftz(
        kpr_f32_mul_rn_ftz(ss, kpr_f32_rcp_approx_ftz(1024.0f)),
        0.0000009999999974752427078783512115478515625f));
    uint32_t i5 = 0U;
    while (i5 < 1024U) {
        uint32_t vi = i5;
        float xi = h1[vi];
        float cw = __bfloat162float(w[b + 1024U + vi]);
        float m1 = kpr_f32_mul_rn_ftz(xi, scale);
        norm2[vi] = __bfloat162float(__float2bfloat16(
            kpr_f32_mul_rn_ftz(m1, kpr_f32_add_rn_ftz(1.0f, cw))));
        i5 = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_9(
    __nv_bfloat16 *w, uint32_t b, float *norm2, float *lanes, float *gate)
{
    uint32_t i = 0U;
    while (i < 3584U) {
        uint32_t vi = i;
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t vl = l;
            float s = 0.0f;
            uint32_t c = 0U;
            while (c < 4U) {
                uint32_t vc = c;
                uint32_t kk = vc * 256U + vl * 8U;
                float part = 0.0f;
                uint32_t j = 0U;
                while (j < 8U) {
                    uint32_t vj = j;
                    float b1 = norm2[kk + vj];
                    float vp = part;
                    part = kpr_f32_fma_rn_ftz(
                        __bfloat162float(w[b + 2048U + vi * 1024U + kk + vj]),
                        b1, vp);
                    j = vj + 1U;
                }
                s = kpr_f32_add_rn_ftz(s, part);
                c = vc + 1U;
            }
            lanes[vl] = s;
            l = vl + 1U;
        }
        uint32_t i1 = 0U;
        while (i1 < 16U) {
            uint32_t k = i1;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 16U]);
            i1 = k + 1U;
        }
        uint32_t i2 = 0U;
        while (i2 < 8U) {
            uint32_t k = i2;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 8U]);
            i2 = k + 1U;
        }
        uint32_t i3 = 0U;
        while (i3 < 4U) {
            uint32_t k = i3;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 4U]);
            i3 = k + 1U;
        }
        uint32_t i4 = 0U;
        while (i4 < 2U) {
            uint32_t k = i4;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 2U]);
            i4 = k + 1U;
        }
        uint32_t i5 = 0U;
        while (i5 < 1U) {
            uint32_t k = i5;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 1U]);
            i5 = k + 1U;
        }
        gate[vi] = lanes[0U];
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_10(
    __nv_bfloat16 *w, uint32_t b, float *norm2, float *lanes, float *up)
{
    uint32_t i = 0U;
    while (i < 3584U) {
        uint32_t vi = i;
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t vl = l;
            float s = 0.0f;
            uint32_t c = 0U;
            while (c < 4U) {
                uint32_t vc = c;
                uint32_t kk = vc * 256U + vl * 8U;
                float part = 0.0f;
                uint32_t j = 0U;
                while (j < 8U) {
                    uint32_t vj = j;
                    float b1 = norm2[kk + vj];
                    float vp = part;
                    part = kpr_f32_fma_rn_ftz(
                        __bfloat162float(
                            w[b + (uint32_t) 3672064U + vi * 1024U + kk + vj]),
                        b1, vp);
                    j = vj + 1U;
                }
                s = kpr_f32_add_rn_ftz(s, part);
                c = vc + 1U;
            }
            lanes[vl] = s;
            l = vl + 1U;
        }
        uint32_t i1 = 0U;
        while (i1 < 16U) {
            uint32_t k = i1;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 16U]);
            i1 = k + 1U;
        }
        uint32_t i2 = 0U;
        while (i2 < 8U) {
            uint32_t k = i2;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 8U]);
            i2 = k + 1U;
        }
        uint32_t i3 = 0U;
        while (i3 < 4U) {
            uint32_t k = i3;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 4U]);
            i3 = k + 1U;
        }
        uint32_t i4 = 0U;
        while (i4 < 2U) {
            uint32_t k = i4;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 2U]);
            i4 = k + 1U;
        }
        uint32_t i5 = 0U;
        while (i5 < 1U) {
            uint32_t k = i5;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 1U]);
            i5 = k + 1U;
        }
        up[vi] = lanes[0U];
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_11(float *gate, float *up, float *act)
{
    uint32_t i = 0U;
    while (i < 3584U) {
        uint32_t vi = i;
        float g = gate[vi];
        float u = up[vi];
        act[vi] = kpr_f32_mul_rn_ftz(
            kpr_f32_mul_rn_ftz(g,
                kpr_f32_rcp_approx_ftz(kpr_f32_add_rn_ftz(1.0f, __expf(-g)))),
            u);
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_12(__nv_bfloat16 *w, uint32_t b, float *act, float *lanes,
    float *h1, float *xo)
{
    uint32_t i = 0U;
    while (i < 1024U) {
        uint32_t vi = i;
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t vl = l;
            float s = 0.0f;
            uint32_t c = 0U;
            while (c < 14U) {
                uint32_t vc = c;
                uint32_t kk = vc * 256U + vl * 8U;
                uint32_t j = 0U;
                while (j < 8U) {
                    uint32_t vj = j;
                    float b1 = act[kk + vj];
                    float vs = s;
                    s = kpr_f32_fma_rn_ftz(
                        __bfloat162float(
                            w[b + (uint32_t) 7342080U + vi * 3584U + kk + vj]),
                        b1, vs);
                    j = vj + 1U;
                }
                c = vc + 1U;
            }
            lanes[vl] = s;
            l = vl + 1U;
        }
        uint32_t i1 = 0U;
        while (i1 < 16U) {
            uint32_t k = i1;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 16U]);
            i1 = k + 1U;
        }
        uint32_t i2 = 0U;
        while (i2 < 8U) {
            uint32_t k = i2;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 8U]);
            i2 = k + 1U;
        }
        uint32_t i3 = 0U;
        while (i3 < 4U) {
            uint32_t k = i3;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 4U]);
            i3 = k + 1U;
        }
        uint32_t i4 = 0U;
        while (i4 < 2U) {
            uint32_t k = i4;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 2U]);
            i4 = k + 1U;
        }
        uint32_t i5 = 0U;
        while (i5 < 1U) {
            uint32_t k = i5;
            lanes[k] = kpr_f32_add_rn_ftz(lanes[k], lanes[k + 1U]);
            i5 = k + 1U;
        }
        xo[vi] = __bfloat162float(
            __float2bfloat16(kpr_f32_add_rn_ftz(lanes[0U], h1[vi])));
        i = vi + 1U;
    }
}

void KPlay_Functional_kplay_run(__nv_bfloat16 *w, float *x, __nv_bfloat16 *k,
    __nv_bfloat16 *v, float *cs, float *sn, float *xo, __nv_bfloat16 *ko,
    __nv_bfloat16 *vo, uint32_t wn, int64_t base, int64_t fa, int64_t pos)
{
    if (!(base < 0LL || base > 4294967295LL || fa < 0LL || fa >= 6LL ||
            pos < 0LL || pos >= 2048LL)) {
        uint32_t b = (uint32_t) base;
        uint32_t f = (uint32_t) fa;
        uint32_t p = (uint32_t) pos;
        if (!(b > wn || wn - b < (uint32_t) 18352640U)) {
            MUST(cudaMemcpy(ko, k,
                (uint64_t) sizeof(__nv_bfloat16) *
                    (uint64_t) (uint32_t) 6291456U,
                cudaMemcpyDeviceToDevice));
            MUST(cudaMemcpy(vo, v,
                (uint64_t) sizeof(__nv_bfloat16) *
                    (uint64_t) (uint32_t) 6291456U,
                cudaMemcpyDeviceToDevice));
            float *norm1 =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 1024U);
            float *lanes =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 32U);
            float *warps =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 32U);
            float *proj =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 4096U);
            float *zk = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 512U);
            float *zv = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 512U);
            float *qn =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 2048U);
            float *kn = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 512U);
            float *partial =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 4096U);
            float *maxs =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 16U);
            float *totals =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 16U);
            float *factor =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 16U);
            float *attn =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 2048U);
            float *h1 =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 1024U);
            float *norm2 =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 1024U);
            float *gate =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 3584U);
            float *up =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 3584U);
            float *act =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 3584U);
            uint32_t mix = b + (uint32_t) 11012096U;
            cudaStream_t s = KPR_FRESH_STREAM();
            KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, x, lanes, warps, w,
                b, norm1);
            MUST(cudaStreamSynchronize(s));
            MUST(cudaStreamDestroy(s));
            cudaStream_t s1 = KPR_FRESH_STREAM();
            KPR_KCALL(__hoisted_kplay_run_1, 1U, 1U, 0U, s1, w, mix, norm1,
                lanes, proj);
            MUST(cudaStreamSynchronize(s1));
            MUST(cudaStreamDestroy(s1));
            cudaStream_t s2 = KPR_FRESH_STREAM();
            KPR_KCALL(__hoisted_kplay_run_2, 1U, 1U, 0U, s2, w, mix, norm1,
                lanes, zk);
            MUST(cudaStreamSynchronize(s2));
            MUST(cudaStreamDestroy(s2));
            cudaStream_t s3 = KPR_FRESH_STREAM();
            KPR_KCALL(__hoisted_kplay_run_3, 1U, 1U, 0U, s3, w, mix, norm1,
                lanes, zv);
            MUST(cudaStreamSynchronize(s3));
            MUST(cudaStreamDestroy(s3));
            cudaStream_t s4 = KPR_FRESH_STREAM();
            KPR_KCALL(__hoisted_kplay_run_4, 1U, 1U, 0U, s4, mix, zk, lanes, w,
                cs, sn, kn, proj, qn);
            MUST(cudaStreamSynchronize(s4));
            MUST(cudaStreamDestroy(s4));
            cudaStream_t s5 = KPR_FRESH_STREAM();
            KPR_KCALL(
                __hoisted_kplay_run_5, 1U, 1U, 0U, s5, f, p, kn, zv, ko, vo);
            MUST(cudaStreamSynchronize(s5));
            MUST(cudaStreamDestroy(s5));
            cudaStream_t s6 = KPR_FRESH_STREAM();
            KPR_KCALL(__hoisted_kplay_run_6, 1U, 1U, 0U, s6, f, partial, p, qn,
                ko, lanes, vo, maxs, totals, factor, proj, attn);
            MUST(cudaStreamSynchronize(s6));
            MUST(cudaStreamDestroy(s6));
            cudaStream_t s7 = KPR_FRESH_STREAM();
            KPR_KCALL(__hoisted_kplay_run_7, 1U, 1U, 0U, s7, w, mix, attn,
                lanes, x, h1);
            MUST(cudaStreamSynchronize(s7));
            MUST(cudaStreamDestroy(s7));
            cudaStream_t s8 = KPR_FRESH_STREAM();
            KPR_KCALL(__hoisted_kplay_run_8, 1U, 1U, 0U, s8, h1, lanes, warps,
                w, b, norm2);
            MUST(cudaStreamSynchronize(s8));
            MUST(cudaStreamDestroy(s8));
            cudaStream_t s9 = KPR_FRESH_STREAM();
            KPR_KCALL(__hoisted_kplay_run_9, 1U, 1U, 0U, s9, w, b, norm2, lanes,
                gate);
            MUST(cudaStreamSynchronize(s9));
            MUST(cudaStreamDestroy(s9));
            cudaStream_t s10 = KPR_FRESH_STREAM();
            KPR_KCALL(__hoisted_kplay_run_10, 1U, 1U, 0U, s10, w, b, norm2,
                lanes, up);
            MUST(cudaStreamSynchronize(s10));
            MUST(cudaStreamDestroy(s10));
            cudaStream_t s11 = KPR_FRESH_STREAM();
            KPR_KCALL(__hoisted_kplay_run_11, 1U, 1U, 0U, s11, gate, up, act);
            MUST(cudaStreamSynchronize(s11));
            MUST(cudaStreamDestroy(s11));
            cudaStream_t s12 = KPR_FRESH_STREAM();
            KPR_KCALL(__hoisted_kplay_run_12, 1U, 1U, 0U, s12, w, b, act, lanes,
                h1, xo);
            MUST(cudaStreamSynchronize(s12));
            MUST(cudaStreamDestroy(s12));
            MUST(cudaFree(norm1));
            MUST(cudaFree(lanes));
            MUST(cudaFree(warps));
            MUST(cudaFree(proj));
            MUST(cudaFree(zk));
            MUST(cudaFree(zv));
            MUST(cudaFree(qn));
            MUST(cudaFree(kn));
            MUST(cudaFree(partial));
            MUST(cudaFree(maxs));
            MUST(cudaFree(totals));
            MUST(cudaFree(factor));
            MUST(cudaFree(attn));
            MUST(cudaFree(h1));
            MUST(cudaFree(norm2));
            MUST(cudaFree(gate));
            MUST(cudaFree(up));
            MUST(cudaFree(act));
        }
    }
}
