
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(float *rc, float *rco, float *cv, float *cvo, float *x,
    __nv_bfloat16 *w, uint32_t b, uint32_t d, float *xo)
{
    uint32_t i = 0U;
    while (i < (uint32_t) 4718592U) {
        uint32_t iv = i;
        rco[iv] = rc[iv];
        i = iv + 1U;
    }
    uint32_t i1 = 0U;
    while (i1 < (uint32_t) 442368U) {
        uint32_t iv = i1;
        cvo[iv] = cv[iv];
        i1 = iv + 1U;
    }
    float norm[1024U];
    memset(norm, 0U, 1024U * sizeof(float));
    float proj[6144U];
    memset(proj, 0U, 6144U * sizeof(float));
    float z[2048U];
    memset(z, 0U, 2048U * sizeof(float));
    float out[2048U];
    memset(out, 0U, 2048U * sizeof(float));
    float q[128U];
    memset(q, 0U, 128U * sizeof(float));
    float k[128U];
    memset(k, 0U, 128U * sizeof(float));
    float co[6144U];
    memset(co, 0U, 6144U * sizeof(float));
    float core[128U];
    memset(core, 0U, 128U * sizeof(float));
    float hid[1024U];
    memset(hid, 0U, 1024U * sizeof(float));
    float mlp[3584U];
    memset(mlp, 0U, 3584U * sizeof(float));
    float warps[32U];
    memset(warps, 0U, 32U * sizeof(float));
    float lanes[32U];
    memset(lanes, 0U, 32U * sizeof(float));
    uint32_t g = 0U;
    while (g < 16U) {
        uint32_t gv = g;
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t lv = l;
            float acc = 0.0f;
            uint32_t t = 0U;
            while (t < 2U) {
                uint32_t tv = t;
                float e = x[gv * 32U + lv + tv * 512U];
                acc = kpr_f32_fma_rn_ftz(e, e, acc);
                t = tv + 1U;
            }
            lanes[lv] = acc;
            l = lv + 1U;
        }
        uint32_t d1 = 16U;
        while (d1 > 0U) {
            uint32_t dv = d1;
            uint32_t i2 = 0U;
            while (i2 < dv) {
                uint32_t iv = i2;
                lanes[iv] = kpr_f32_add_rn_ftz(lanes[iv], lanes[iv + dv]);
                i2 = iv + 1U;
            }
            d1 = dv / 2U;
        }
        warps[gv] = lanes[0U];
        g = gv + 1U;
    }
    uint32_t d1 = 16U;
    while (d1 > 0U) {
        uint32_t dv = d1;
        uint32_t i2 = 0U;
        while (i2 < dv) {
            uint32_t iv = i2;
            warps[iv] = kpr_f32_add_rn_ftz(warps[iv], warps[iv + dv]);
            i2 = iv + 1U;
        }
        d1 = dv / 2U;
    }
    float scale = kpr_f32_rsqrt_approx_ftz(
        kpr_f32_add_rn_ftz(kpr_f32_mul_rn_ftz(warps[0U], 0.0009765625f),
            0.0000009999999974752427078783512115478515625f));
    uint32_t i2 = 0U;
    while (i2 < 1024U) {
        uint32_t iv = i2;
        float h = x[iv];
        float w1 = kpr_f32_add_rn_ftz(1.0f, __bfloat162float(w[b + iv]));
        norm[iv] = __bfloat162float(__float2bfloat16(
            kpr_f32_mul_rn_ftz(kpr_f32_mul_rn_ftz(h, scale), w1)));
        i2 = iv + 1U;
    }
    uint32_t mix = b + (uint32_t) 11012096U;
    uint32_t i3 = 0U;
    while (i3 < 6144U) {
        uint32_t iv = i3;
        float lanes1[32U];
        memset(lanes1, 0U, 32U * sizeof(float));
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t lv = l;
            float acc = 0.0f;
            uint32_t m1 = 0U;
            while (m1 < 4U) {
                uint32_t mv = m1;
                uint32_t k1 = lv * 8U + mv * 256U;
                float a0 = acc;
                float part = 0.0f;
                uint32_t j = 0U;
                while (j < 8U) {
                    uint32_t jv = j;
                    float xv = norm[k1 + jv];
                    float pv = part;
                    part = kpr_f32_fma_rn_ftz(
                        __bfloat162float(w[mix + iv * 1024U + k1 + jv]), xv,
                        pv);
                    j = jv + 1U;
                }
                acc = kpr_f32_add_rn_ftz(a0, part);
                m1 = mv + 1U;
            }
            lanes1[lv] = acc;
            l = lv + 1U;
        }
        uint32_t d2 = 16U;
        while (d2 > 0U) {
            uint32_t dv = d2;
            uint32_t i4 = 0U;
            while (i4 < dv) {
                uint32_t iv1 = i4;
                lanes1[iv1] = kpr_f32_add_rn_ftz(lanes1[iv1], lanes1[iv1 + dv]);
                i4 = iv1 + 1U;
            }
            d2 = dv / 2U;
        }
        proj[iv] = lanes1[0U];
        i3 = iv + 1U;
    }
    uint32_t i4 = 0U;
    while (i4 < 2048U) {
        uint32_t iv = i4;
        float lanes1[32U];
        memset(lanes1, 0U, 32U * sizeof(float));
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t lv = l;
            float acc = 0.0f;
            uint32_t m1 = 0U;
            while (m1 < 4U) {
                uint32_t mv = m1;
                uint32_t k1 = lv * 8U + mv * 256U;
                float a0 = acc;
                float part = 0.0f;
                uint32_t j = 0U;
                while (j < 8U) {
                    uint32_t jv = j;
                    float xv = norm[k1 + jv];
                    float pv = part;
                    part = kpr_f32_fma_rn_ftz(
                        __bfloat162float(w[mix + (uint32_t) 6291456U +
                                           iv * 1024U + k1 + jv]),
                        xv, pv);
                    j = jv + 1U;
                }
                acc = kpr_f32_add_rn_ftz(a0, part);
                m1 = mv + 1U;
            }
            lanes1[lv] = acc;
            l = lv + 1U;
        }
        uint32_t d2 = 16U;
        while (d2 > 0U) {
            uint32_t dv = d2;
            uint32_t i5 = 0U;
            while (i5 < dv) {
                uint32_t iv1 = i5;
                lanes1[iv1] = kpr_f32_add_rn_ftz(lanes1[iv1], lanes1[iv1 + dv]);
                i5 = iv1 + 1U;
            }
            d2 = dv / 2U;
        }
        z[iv] = lanes1[0U];
        i4 = iv + 1U;
    }
    uint32_t i5 = 0U;
    while (i5 < 6144U) {
        uint32_t iv = i5;
        uint32_t ci = d * 24576U + iv * 4U;
        uint32_t wc = mix + (uint32_t) 8421376U + iv * 4U;
        float c0 = cv[ci + 1U];
        float c1 = cv[ci + 2U];
        float c2 = cv[ci + 3U];
        float c3 = proj[iv];
        cvo[ci] = c0;
        cvo[ci + 1U] = c1;
        cvo[ci + 2U] = c2;
        cvo[ci + 3U] = c3;
        __nv_bfloat16 w1 = w[wc + 1U];
        __nv_bfloat16 w2 = w[wc + 2U];
        __nv_bfloat16 w3 = w[wc + 3U];
        float s0 = kpr_f32_fma_rn_ftz(c0, __bfloat162float(w[wc]), 0.0f);
        float s1 = kpr_f32_fma_rn_ftz(c1, __bfloat162float(w1), s0);
        float s2 = kpr_f32_fma_rn_ftz(c2, __bfloat162float(w2), s1);
        float s3 = kpr_f32_fma_rn_ftz(c3, __bfloat162float(w3), s2);
        co[iv] = kpr_f32_mul_rn_ftz(
            s3, kpr_f32_rcp_approx_ftz(kpr_f32_add_rn_ftz(1.0f, __expf(-s3))));
        i5 = iv + 1U;
    }
    uint32_t h = 0U;
    while (h < 16U) {
        uint32_t hv = h;
        float lanes1[32U];
        memset(lanes1, 0U, 32U * sizeof(float));
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t lv = l;
            float acc = 0.0f;
            uint32_t m1 = 0U;
            while (m1 < 4U) {
                uint32_t mv = m1;
                uint32_t k1 = lv * 8U + mv * 256U;
                float a0 = acc;
                float part = 0.0f;
                uint32_t j = 0U;
                while (j < 8U) {
                    uint32_t jv = j;
                    float xv = norm[k1 + jv];
                    float pv = part;
                    part = kpr_f32_fma_rn_ftz(
                        __bfloat162float(w[mix + (uint32_t) 8388608U +
                                           hv * 1024U + k1 + jv]),
                        xv, pv);
                    j = jv + 1U;
                }
                acc = kpr_f32_add_rn_ftz(a0, part);
                m1 = mv + 1U;
            }
            lanes1[lv] = acc;
            l = lv + 1U;
        }
        uint32_t d2 = 16U;
        while (d2 > 0U) {
            uint32_t dv = d2;
            uint32_t i6 = 0U;
            while (i6 < dv) {
                uint32_t iv = i6;
                lanes1[iv] = kpr_f32_add_rn_ftz(lanes1[iv], lanes1[iv + dv]);
                i6 = iv + 1U;
            }
            d2 = dv / 2U;
        }
        float beta = kpr_f32_rcp_approx_ftz(
            kpr_f32_add_rn_ftz(1.0f, __expf(-lanes1[0U])));
        float lanes2[32U];
        memset(lanes2, 0U, 32U * sizeof(float));
        uint32_t l1 = 0U;
        while (l1 < 32U) {
            uint32_t lv = l1;
            float acc = 0.0f;
            uint32_t m1 = 0U;
            while (m1 < 4U) {
                uint32_t mv = m1;
                uint32_t k1 = lv * 8U + mv * 256U;
                float a0 = acc;
                float part = 0.0f;
                uint32_t j = 0U;
                while (j < 8U) {
                    uint32_t jv = j;
                    float xv = norm[k1 + jv];
                    float pv = part;
                    part = kpr_f32_fma_rn_ftz(
                        __bfloat162float(w[mix + (uint32_t) 8404992U +
                                           hv * 1024U + k1 + jv]),
                        xv, pv);
                    j = jv + 1U;
                }
                acc = kpr_f32_add_rn_ftz(a0, part);
                m1 = mv + 1U;
            }
            lanes2[lv] = acc;
            l1 = lv + 1U;
        }
        uint32_t d4 = 16U;
        while (d4 > 0U) {
            uint32_t dv = d4;
            uint32_t i6 = 0U;
            while (i6 < dv) {
                uint32_t iv = i6;
                lanes2[iv] = kpr_f32_add_rn_ftz(lanes2[iv], lanes2[iv + dv]);
                i6 = iv + 1U;
            }
            d4 = dv / 2U;
        }
        float d5 = lanes2[0U];
        float a2 = kpr_f32_add_rn_ftz(
            d5, __bfloat162float(w[mix + (uint32_t) 8445968U + hv]));
        float sp;
        if (20.0f < a2)
            sp = a2;
        else
            sp = __logf(kpr_f32_add_rn_ftz(
                1.0f, kpr_f32_exp2_approx_ftz(
                          kpr_f32_mul_rn_ftz(a2, 1.44269504088896340736f))));
        float decay = __expf(-kpr_f32_mul_rn_ftz(
            __expf(__bfloat162float(w[mix + (uint32_t) 8445952U + hv])), sp));
        float warps1[32U];
        memset(warps1, 0U, 32U * sizeof(float));
        float lanes3[32U];
        memset(lanes3, 0U, 32U * sizeof(float));
        uint32_t g1 = 0U;
        while (g1 < 1U) {
            uint32_t gv = g1;
            uint32_t l2 = 0U;
            while (l2 < 32U) {
                uint32_t lv = l2;
                float acc = 0.0f;
                uint32_t t = 0U;
                while (t < 4U) {
                    uint32_t tv = t;
                    float e1 = co[hv * 128U + gv * 32U + lv + tv * 32U];
                    acc = kpr_f32_fma_rn_ftz(e1, e1, acc);
                    t = tv + 1U;
                }
                lanes3[lv] = acc;
                l2 = lv + 1U;
            }
            uint32_t d6 = 16U;
            while (d6 > 0U) {
                uint32_t dv = d6;
                uint32_t i6 = 0U;
                while (i6 < dv) {
                    uint32_t iv = i6;
                    lanes3[iv] =
                        kpr_f32_add_rn_ftz(lanes3[iv], lanes3[iv + dv]);
                    i6 = iv + 1U;
                }
                d6 = dv / 2U;
            }
            warps1[gv] = lanes3[0U];
            g1 = gv + 1U;
        }
        float qsc = kpr_f32_mul_rn_ftz(
            kpr_f32_rsqrt_approx_ftz(kpr_f32_add_rn_ftz(
                warps1[0U], 0.0000009999999974752427078783512115478515625f)),
            0.088388346135616302490234375f);
        uint32_t i6 = 0U;
        while (i6 < 128U) {
            uint32_t iv = i6;
            q[iv] = kpr_f32_mul_rn_ftz(co[hv * 128U + iv], qsc);
            i6 = iv + 1U;
        }
        float warps2[32U];
        memset(warps2, 0U, 32U * sizeof(float));
        float lanes4[32U];
        memset(lanes4, 0U, 32U * sizeof(float));
        uint32_t g2 = 0U;
        while (g2 < 1U) {
            uint32_t gv = g2;
            uint32_t l2 = 0U;
            while (l2 < 32U) {
                uint32_t lv = l2;
                float acc = 0.0f;
                uint32_t t = 0U;
                while (t < 4U) {
                    uint32_t tv = t;
                    float e1 = co[2048U + hv * 128U + gv * 32U + lv + tv * 32U];
                    acc = kpr_f32_fma_rn_ftz(e1, e1, acc);
                    t = tv + 1U;
                }
                lanes4[lv] = acc;
                l2 = lv + 1U;
            }
            uint32_t d6 = 16U;
            while (d6 > 0U) {
                uint32_t dv = d6;
                uint32_t i7 = 0U;
                while (i7 < dv) {
                    uint32_t iv = i7;
                    lanes4[iv] =
                        kpr_f32_add_rn_ftz(lanes4[iv], lanes4[iv + dv]);
                    i7 = iv + 1U;
                }
                d6 = dv / 2U;
            }
            warps2[gv] = lanes4[0U];
            g2 = gv + 1U;
        }
        float ksc = kpr_f32_rsqrt_approx_ftz(kpr_f32_add_rn_ftz(
            warps2[0U], 0.0000009999999974752427078783512115478515625f));
        uint32_t i7 = 0U;
        while (i7 < 128U) {
            uint32_t iv = i7;
            k[iv] = kpr_f32_mul_rn_ftz(co[2048U + hv * 128U + iv], ksc);
            i7 = iv + 1U;
        }
        float lanes5[32U];
        memset(lanes5, 0U, 32U * sizeof(float));
        uint32_t l2 = 0U;
        while (l2 < 32U) {
            uint32_t lv = l2;
            float acc = 0.0f;
            uint32_t t = 0U;
            while (t < 4U) {
                uint32_t tv = t;
                uint32_t idx = lv + tv * 32U;
                acc = kpr_f32_fma_rn_ftz(k[idx], q[idx], acc);
                t = tv + 1U;
            }
            lanes5[lv] = acc;
            l2 = lv + 1U;
        }
        uint32_t d6 = 16U;
        while (d6 > 0U) {
            uint32_t dv = d6;
            uint32_t i8 = 0U;
            while (i8 < dv) {
                uint32_t iv = i8;
                lanes5[iv] = kpr_f32_add_rn_ftz(lanes5[iv], lanes5[iv + dv]);
                i8 = iv + 1U;
            }
            d6 = dv / 2U;
        }
        float kqv = lanes5[0U];
        uint32_t j = 0U;
        while (j < 128U) {
            uint32_t jv = j;
            uint32_t si_ = d * (uint32_t) 262144U + hv * 16384U + jv * 128U;
            float lanes6[32U];
            memset(lanes6, 0U, 32U * sizeof(float));
            uint32_t l3 = 0U;
            while (l3 < 32U) {
                uint32_t lv = l3;
                float acc = 0.0f;
                uint32_t t = 0U;
                while (t < 4U) {
                    uint32_t tv = t;
                    uint32_t idx = lv + tv * 32U;
                    acc = kpr_f32_fma_rn_ftz(rc[si_ + idx], k[idx], acc);
                    t = tv + 1U;
                }
                lanes6[lv] = acc;
                l3 = lv + 1U;
            }
            uint32_t d7 = 16U;
            while (d7 > 0U) {
                uint32_t dv = d7;
                uint32_t i8 = 0U;
                while (i8 < dv) {
                    uint32_t iv = i8;
                    lanes6[iv] =
                        kpr_f32_add_rn_ftz(lanes6[iv], lanes6[iv + dv]);
                    i8 = iv + 1U;
                }
                d7 = dv / 2U;
            }
            float sk1 = lanes6[0U];
            float lanes7[32U];
            memset(lanes7, 0U, 32U * sizeof(float));
            uint32_t l4 = 0U;
            while (l4 < 32U) {
                uint32_t lv = l4;
                float acc = 0.0f;
                uint32_t t = 0U;
                while (t < 4U) {
                    uint32_t tv = t;
                    uint32_t idx = lv + tv * 32U;
                    acc = kpr_f32_fma_rn_ftz(rc[si_ + idx], q[idx], acc);
                    t = tv + 1U;
                }
                lanes7[lv] = acc;
                l4 = lv + 1U;
            }
            uint32_t d8 = 16U;
            while (d8 > 0U) {
                uint32_t dv = d8;
                uint32_t i8 = 0U;
                while (i8 < dv) {
                    uint32_t iv = i8;
                    lanes7[iv] =
                        kpr_f32_add_rn_ftz(lanes7[iv], lanes7[iv + dv]);
                    i8 = iv + 1U;
                }
                d8 = dv / 2U;
            }
            float sq1 = lanes7[0U];
            float err = kpr_f32_mul_rn_ftz(
                kpr_f32_fma_rn_ftz(-decay, sk1, co[4096U + hv * 128U + jv]),
                beta);
            core[jv] =
                kpr_f32_fma_rn_ftz(decay, sq1, kpr_f32_mul_rn_ftz(err, kqv));
            uint32_t i8 = 0U;
            while (i8 < 128U) {
                uint32_t iv = i8;
                float s1 = rc[si_ + iv];
                rco[si_ + iv] = kpr_f32_fma_rn_ftz(
                    s1, decay, kpr_f32_mul_rn_ftz(k[iv], err));
                i8 = iv + 1U;
            }
            j = jv + 1U;
        }
        float warps3[32U];
        memset(warps3, 0U, 32U * sizeof(float));
        float lanes6[32U];
        memset(lanes6, 0U, 32U * sizeof(float));
        uint32_t g3 = 0U;
        while (g3 < 4U) {
            uint32_t gv = g3;
            uint32_t l3 = 0U;
            while (l3 < 32U) {
                uint32_t lv = l3;
                float acc = 0.0f;
                uint32_t t = 0U;
                while (t < 1U) {
                    uint32_t tv = t;
                    float e1 = core[gv * 32U + lv + tv * 128U];
                    acc = kpr_f32_fma_rn_ftz(e1, e1, acc);
                    t = tv + 1U;
                }
                lanes6[lv] = acc;
                l3 = lv + 1U;
            }
            uint32_t d7 = 16U;
            while (d7 > 0U) {
                uint32_t dv = d7;
                uint32_t i8 = 0U;
                while (i8 < dv) {
                    uint32_t iv = i8;
                    lanes6[iv] =
                        kpr_f32_add_rn_ftz(lanes6[iv], lanes6[iv + dv]);
                    i8 = iv + 1U;
                }
                d7 = dv / 2U;
            }
            warps3[gv] = lanes6[0U];
            g3 = gv + 1U;
        }
        uint32_t d7 = 16U;
        while (d7 > 0U) {
            uint32_t dv = d7;
            uint32_t i8 = 0U;
            while (i8 < dv) {
                uint32_t iv = i8;
                warps3[iv] = kpr_f32_add_rn_ftz(warps3[iv], warps3[iv + dv]);
                i8 = iv + 1U;
            }
            d7 = dv / 2U;
        }
        float rstd = kpr_f32_rsqrt_approx_ftz(
            kpr_f32_add_rn_ftz(kpr_f32_mul_rn_ftz(warps3[0U], 0.0078125f),
                0.0000009999999974752427078783512115478515625f));
        uint32_t i8 = 0U;
        while (i8 < 128U) {
            uint32_t iv = i8;
            float c = core[iv];
            __nv_bfloat16 g4 = w[mix + (uint32_t) 8445984U + iv];
            float zv = z[hv * 128U + iv];
            float szv = kpr_f32_mul_rn_ftz(zv,
                kpr_f32_rcp_approx_ftz(kpr_f32_add_rn_ftz(1.0f, __expf(-zv))));
            float m11 = kpr_f32_mul_rn_ftz(c, rstd);
            out[hv * 128U + iv] = kpr_f32_mul_rn_ftz(
                kpr_f32_mul_rn_ftz(m11, __bfloat162float(g4)), szv);
            i8 = iv + 1U;
        }
        h = hv + 1U;
    }
    uint32_t i6 = 0U;
    while (i6 < 1024U) {
        uint32_t iv = i6;
        float lanes1[32U];
        memset(lanes1, 0U, 32U * sizeof(float));
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t lv = l;
            float acc = 0.0f;
            uint32_t m1 = 0U;
            while (m1 < 8U) {
                uint32_t mv = m1;
                uint32_t k1 = lv * 8U + mv * 256U;
                float part = acc;
                uint32_t j = 0U;
                while (j < 8U) {
                    uint32_t jv = j;
                    float xv = out[k1 + jv];
                    float pv = part;
                    part = kpr_f32_fma_rn_ftz(
                        __bfloat162float(w[mix + (uint32_t) 8446112U +
                                           iv * 2048U + k1 + jv]),
                        xv, pv);
                    j = jv + 1U;
                }
                acc = part;
                m1 = mv + 1U;
            }
            lanes1[lv] = acc;
            l = lv + 1U;
        }
        uint32_t d2 = 16U;
        while (d2 > 0U) {
            uint32_t dv = d2;
            uint32_t i7 = 0U;
            while (i7 < dv) {
                uint32_t iv1 = i7;
                lanes1[iv1] = kpr_f32_add_rn_ftz(lanes1[iv1], lanes1[iv1 + dv]);
                i7 = iv1 + 1U;
            }
            d2 = dv / 2U;
        }
        hid[iv] = __bfloat162float(
            __float2bfloat16(kpr_f32_add_rn_ftz(lanes1[0U], x[iv])));
        i6 = iv + 1U;
    }
    float warps1[32U];
    memset(warps1, 0U, 32U * sizeof(float));
    float lanes1[32U];
    memset(lanes1, 0U, 32U * sizeof(float));
    uint32_t g1 = 0U;
    while (g1 < 16U) {
        uint32_t gv = g1;
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t lv = l;
            float acc = 0.0f;
            uint32_t t = 0U;
            while (t < 2U) {
                uint32_t tv = t;
                float e = hid[gv * 32U + lv + tv * 512U];
                acc = kpr_f32_fma_rn_ftz(e, e, acc);
                t = tv + 1U;
            }
            lanes1[lv] = acc;
            l = lv + 1U;
        }
        uint32_t d2 = 16U;
        while (d2 > 0U) {
            uint32_t dv = d2;
            uint32_t i7 = 0U;
            while (i7 < dv) {
                uint32_t iv = i7;
                lanes1[iv] = kpr_f32_add_rn_ftz(lanes1[iv], lanes1[iv + dv]);
                i7 = iv + 1U;
            }
            d2 = dv / 2U;
        }
        warps1[gv] = lanes1[0U];
        g1 = gv + 1U;
    }
    uint32_t d2 = 16U;
    while (d2 > 0U) {
        uint32_t dv = d2;
        uint32_t i7 = 0U;
        while (i7 < dv) {
            uint32_t iv = i7;
            warps1[iv] = kpr_f32_add_rn_ftz(warps1[iv], warps1[iv + dv]);
            i7 = iv + 1U;
        }
        d2 = dv / 2U;
    }
    float scale1 = kpr_f32_rsqrt_approx_ftz(
        kpr_f32_add_rn_ftz(kpr_f32_mul_rn_ftz(warps1[0U], 0.0009765625f),
            0.0000009999999974752427078783512115478515625f));
    uint32_t i7 = 0U;
    while (i7 < 1024U) {
        uint32_t iv = i7;
        float h1 = hid[iv];
        float w1 =
            kpr_f32_add_rn_ftz(1.0f, __bfloat162float(w[b + 1024U + iv]));
        norm[iv] = __bfloat162float(__float2bfloat16(
            kpr_f32_mul_rn_ftz(kpr_f32_mul_rn_ftz(h1, scale1), w1)));
        i7 = iv + 1U;
    }
    uint32_t i8 = 0U;
    while (i8 < 3584U) {
        uint32_t iv = i8;
        float lanes2[32U];
        memset(lanes2, 0U, 32U * sizeof(float));
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t lv = l;
            float acc = 0.0f;
            uint32_t m2 = 0U;
            while (m2 < 4U) {
                uint32_t mv = m2;
                uint32_t k1 = lv * 8U + mv * 256U;
                float a0 = acc;
                float part = 0.0f;
                uint32_t j = 0U;
                while (j < 8U) {
                    uint32_t jv = j;
                    float xv = norm[k1 + jv];
                    float pv = part;
                    part = kpr_f32_fma_rn_ftz(
                        __bfloat162float(w[b + 2048U + iv * 1024U + k1 + jv]),
                        xv, pv);
                    j = jv + 1U;
                }
                acc = kpr_f32_add_rn_ftz(a0, part);
                m2 = mv + 1U;
            }
            lanes2[lv] = acc;
            l = lv + 1U;
        }
        uint32_t d3 = 16U;
        while (d3 > 0U) {
            uint32_t dv = d3;
            uint32_t i9 = 0U;
            while (i9 < dv) {
                uint32_t iv1 = i9;
                lanes2[iv1] = kpr_f32_add_rn_ftz(lanes2[iv1], lanes2[iv1 + dv]);
                i9 = iv1 + 1U;
            }
            d3 = dv / 2U;
        }
        float g2 = lanes2[0U];
        float lanes3[32U];
        memset(lanes3, 0U, 32U * sizeof(float));
        uint32_t l1 = 0U;
        while (l1 < 32U) {
            uint32_t lv = l1;
            float acc = 0.0f;
            uint32_t m2 = 0U;
            while (m2 < 4U) {
                uint32_t mv = m2;
                uint32_t k1 = lv * 8U + mv * 256U;
                float a0 = acc;
                float part = 0.0f;
                uint32_t j = 0U;
                while (j < 8U) {
                    uint32_t jv = j;
                    float xv = norm[k1 + jv];
                    float pv = part;
                    part = kpr_f32_fma_rn_ftz(
                        __bfloat162float(
                            w[b + (uint32_t) 3672064U + iv * 1024U + k1 + jv]),
                        xv, pv);
                    j = jv + 1U;
                }
                acc = kpr_f32_add_rn_ftz(a0, part);
                m2 = mv + 1U;
            }
            lanes3[lv] = acc;
            l1 = lv + 1U;
        }
        uint32_t d4 = 16U;
        while (d4 > 0U) {
            uint32_t dv = d4;
            uint32_t i9 = 0U;
            while (i9 < dv) {
                uint32_t iv1 = i9;
                lanes3[iv1] = kpr_f32_add_rn_ftz(lanes3[iv1], lanes3[iv1 + dv]);
                i9 = iv1 + 1U;
            }
            d4 = dv / 2U;
        }
        float u = lanes3[0U];
        mlp[iv] = kpr_f32_mul_rn_ftz(
            kpr_f32_mul_rn_ftz(g2,
                kpr_f32_rcp_approx_ftz(kpr_f32_add_rn_ftz(1.0f, __expf(-g2)))),
            u);
        i8 = iv + 1U;
    }
    uint32_t i9 = 0U;
    while (i9 < 1024U) {
        uint32_t iv = i9;
        float lanes2[32U];
        memset(lanes2, 0U, 32U * sizeof(float));
        uint32_t l = 0U;
        while (l < 32U) {
            uint32_t lv = l;
            float acc = 0.0f;
            uint32_t m2 = 0U;
            while (m2 < 14U) {
                uint32_t mv = m2;
                uint32_t k1 = lv * 8U + mv * 256U;
                float part = acc;
                uint32_t j = 0U;
                while (j < 8U) {
                    uint32_t jv = j;
                    float xv = mlp[k1 + jv];
                    float pv = part;
                    part = kpr_f32_fma_rn_ftz(
                        __bfloat162float(
                            w[b + (uint32_t) 7342080U + iv * 3584U + k1 + jv]),
                        xv, pv);
                    j = jv + 1U;
                }
                acc = part;
                m2 = mv + 1U;
            }
            lanes2[lv] = acc;
            l = lv + 1U;
        }
        uint32_t d3 = 16U;
        while (d3 > 0U) {
            uint32_t dv = d3;
            uint32_t i10 = 0U;
            while (i10 < dv) {
                uint32_t iv1 = i10;
                lanes2[iv1] = kpr_f32_add_rn_ftz(lanes2[iv1], lanes2[iv1 + dv]);
                i10 = iv1 + 1U;
            }
            d3 = dv / 2U;
        }
        xo[iv] = __bfloat162float(
            __float2bfloat16(kpr_f32_add_rn_ftz(lanes2[0U], hid[iv])));
        i9 = iv + 1U;
    }
}

void KPlay_Functional_kplay_run(__nv_bfloat16 *w, float *x, float *rc,
    float *cv, float *xo, float *rco, float *cvo, uint32_t wn, int64_t base,
    int64_t dn)
{
    if (base >= 0LL && base < 4294967296LL && dn >= 0LL && dn < 18LL) {
        uint32_t b = (uint32_t) base;
        uint32_t d = (uint32_t) dn;
        if (wn >= (uint32_t) 21555360U && b <= wn - (uint32_t) 21555360U) {
            cudaStream_t s = KPR_FRESH_STREAM();
            KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, rc, rco, cv, cvo, x,
                w, b, d, xo);
            MUST(cudaStreamSynchronize(s));
            MUST(cudaStreamDestroy(s));
        }
    }
}
