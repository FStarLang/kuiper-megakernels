
#include "KPlay_Functional.h"

__global__ __launch_bounds__(32)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(__nv_bfloat16 *input0, __nv_bfloat16 *input8, float *cqkv)
{
    auto &fr = KPR_INIT(kpr_mma_fragment);
    uint32_t t = 0U;
    while (t < 256U) {
        uint32_t tv = t;
        uint32_t rat = tv % 32U < 16U ? tv / 32U * 16U + tv % 32U
                                      : 128U + (tv % 32U - 16U) / 8U * 64U +
                                            tv / 32U * 8U + tv % 8U;
        uint32_t cbt = 0U;
        kpr_mma_fill(fr, 0.0f);
        uint32_t ch = 0U;
        while (ch < 64U) {
            uint32_t chv = ch;
            kpr_mma_mma_sync(input0 + (1024U * (rat * 16U) + chv * 16U), 1024U,
                input8 + (1024U * (cbt * 8U) + chv * 16U), 1024U, fr);
            ch = chv + 1U;
        }
        kpr_mma_store(fr, cqkv + 8U * (tv * 16U), 8U);
        t = tv + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_1(float *cqkv, __nv_bfloat16 *qkv)
{
    uint32_t q = 0U;
    while (q < 32768U) {
        uint32_t qv = q;
        qkv[qv] = __float2bfloat16(cqkv[qv % 4096U * 8U + qv / 4096U]);
        q = qv + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_2(__nv_bfloat16 *qkv, float *part32, __nv_bfloat16 *input1,
    __nv_bfloat16 *un, uint32_t off, __nv_bfloat16 *input9,
    __nv_bfloat16 *input10, __nv_bfloat16 *qb, __nv_bfloat16 *kb)
{
    uint32_t i = 0U;
    while (i < 192U) {
        uint32_t iv = i;
        uint32_t h = iv / 24U;
        uint32_t r = iv % 24U;
        uint32_t tok = r < 16U ? r / 2U : r - 16U;
        uint32_t vb = tok * 4096U + h * 512U + (r < 16U ? r % 2U : 2U) * 128U;
        uint32_t t = 0U;
        while (t < 32U) {
            uint32_t tv = t;
            __nv_bfloat16 x = qkv[vb + tv];
            float v0;
            if (tok < 7U)
                v0 = __bfloat162float(x);
            else
                v0 = 0.0f;
            float a0 = __fmaf_rn(v0, v0, 0.0f);
            __nv_bfloat16 x1 = qkv[vb + tv + 32U];
            float v1;
            if (tok < 7U)
                v1 = __bfloat162float(x1);
            else
                v1 = 0.0f;
            float a1 = __fmaf_rn(v1, v1, a0);
            __nv_bfloat16 x2 = qkv[vb + tv + 64U];
            float v2;
            if (tok < 7U)
                v2 = __bfloat162float(x2);
            else
                v2 = 0.0f;
            float a2 = __fmaf_rn(v2, v2, a1);
            __nv_bfloat16 x3 = qkv[vb + tv + 96U];
            float v3;
            if (tok < 7U)
                v3 = __bfloat162float(x3);
            else
                v3 = 0.0f;
            part32[tv] = __fmaf_rn(v3, v3, a2);
            t = tv + 1U;
        }
        uint32_t d = 16U;
        while (d > 0U) {
            uint32_t dv = d;
            uint32_t t1 = 0U;
            while (t1 < dv) {
                uint32_t tv = t1;
                part32[tv] += part32[tv + dv];
                t1 = tv + 1U;
            }
            d = dv / 2U;
        }
        float buf = 1000000.0f;
        float inv = rsqrtf(part32[0U] / 128.0f + 1.0f / buf);
        uint32_t wb = r < 16U ? 0U : 128U;
        uint32_t c = 0U;
        while (c < 128U) {
            uint32_t cv = c;
            __nv_bfloat16 x = qkv[vb + cv];
            float ite;
            if (tok < 7U)
                ite = __bfloat162float(x);
            else
                ite = 0.0f;
            __nv_bfloat16 scd = __float2bfloat16(ite * inv);
            float wc = __bfloat162float(input1[wb + cv]);
            un[cv] = __float2bfloat16(__bfloat162float(scd) * wc);
            c = cv + 1U;
        }
        uint32_t tb = (off + tok) * 128U;
        uint32_t c3 = 0U;
        while (c3 < 128U) {
            uint32_t cv = c3;
            float ot = __bfloat162float(un[(cv + 64U) % 128U]);
            float other = cv < 64U ? -ot : ot;
            float u = __bfloat162float(un[cv]);
            float co = __bfloat162float(input9[tb + cv]);
            float si = __bfloat162float(input10[tb + cv]);
            __nv_bfloat16 a = __float2bfloat16(u * co);
            __nv_bfloat16 bb = __float2bfloat16(other * si);
            __nv_bfloat16 rot =
                __float2bfloat16(__bfloat162float(a) + __bfloat162float(bb));
            if (r < 16U)
                qb[(h * 16U + r) * 128U + cv] = rot;
            else
                kb[(h * 8U + (r - 16U)) * 128U + cv] = rot;
            c3 = cv + 1U;
        }
        i = iv + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_3(__nv_bfloat16 *qkv, __nv_bfloat16 *vt)
{
    uint32_t i = 0U;
    while (i < (uint32_t) 65536U) {
        uint32_t iv = i;
        __nv_bfloat16 ite;
        if (iv % 64U < 7U) {
            uint32_t hc = iv / 64U;
            ite = qkv[iv % 64U * 4096U + hc / 128U * 512U + 384U + hc % 128U];
        } else
            ite = __float2bfloat16(0.0f);
        vt[iv] = ite;
        i = iv + 1U;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_4(__nv_bfloat16 *qb, __nv_bfloat16 *kb, float *cqk)
{
    auto &fr = KPR_INIT(kpr_mma_fragment);
    uint32_t t = 0U;
    while (t < 8U) {
        uint32_t tv = t;
        uint32_t rat = tv;
        kpr_mma_fill(fr, 0.0f);
        uint32_t ch = 0U;
        while (ch < 8U) {
            uint32_t chv = ch;
            kpr_mma_mma_sync(qb + (128U * (rat * 16U) + chv * 16U), 128U,
                kb + (128U * (tv * 8U) + chv * 16U), 128U, fr);
            ch = chv + 1U;
        }
        kpr_mma_store(fr, cqk + 8U * (tv * 16U), 8U);
        t = tv + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_5(float *cqk, __nv_bfloat16 *pr)
{
    uint32_t i = 0U;
    while (i < 128U) {
        uint32_t rr = i;
        uint32_t r = rr % 16U;
        uint32_t b = rr * 8U;
        float a = 255.0f;
        float buf = 1152921504606846976.0f;
        float bv = buf;
        float mk = -(a * (bv * bv));
        float x0 = cqk[b];
        float s0;
        if (r < 14U)
            s0 = __bfloat162float(
                __float2bfloat16(__bfloat162float(__float2bfloat16(
                                     __bfloat162float(__float2bfloat16(x0)) *
                                     (1.0f / sqrtf(128.0f)))) +
                                 (0U > r / 2U ? mk : 0.0f)));
        else
            s0 = -INFINITY;
        float x1 = cqk[b + 1U];
        float s1;
        if (r < 14U)
            s1 = __bfloat162float(
                __float2bfloat16(__bfloat162float(__float2bfloat16(
                                     __bfloat162float(__float2bfloat16(x1)) *
                                     (1.0f / sqrtf(128.0f)))) +
                                 (1U > r / 2U ? mk : 0.0f)));
        else
            s1 = -INFINITY;
        float x2 = cqk[b + 2U];
        float s2;
        if (r < 14U)
            s2 = __bfloat162float(
                __float2bfloat16(__bfloat162float(__float2bfloat16(
                                     __bfloat162float(__float2bfloat16(x2)) *
                                     (1.0f / sqrtf(128.0f)))) +
                                 (2U > r / 2U ? mk : 0.0f)));
        else
            s2 = -INFINITY;
        float x3 = cqk[b + 3U];
        float s3;
        if (r < 14U)
            s3 = __bfloat162float(
                __float2bfloat16(__bfloat162float(__float2bfloat16(
                                     __bfloat162float(__float2bfloat16(x3)) *
                                     (1.0f / sqrtf(128.0f)))) +
                                 (3U > r / 2U ? mk : 0.0f)));
        else
            s3 = -INFINITY;
        float x4 = cqk[b + 4U];
        float s4;
        if (r < 14U)
            s4 = __bfloat162float(
                __float2bfloat16(__bfloat162float(__float2bfloat16(
                                     __bfloat162float(__float2bfloat16(x4)) *
                                     (1.0f / sqrtf(128.0f)))) +
                                 (4U > r / 2U ? mk : 0.0f)));
        else
            s4 = -INFINITY;
        float x5 = cqk[b + 5U];
        float s5;
        if (r < 14U)
            s5 = __bfloat162float(
                __float2bfloat16(__bfloat162float(__float2bfloat16(
                                     __bfloat162float(__float2bfloat16(x5)) *
                                     (1.0f / sqrtf(128.0f)))) +
                                 (5U > r / 2U ? mk : 0.0f)));
        else
            s5 = -INFINITY;
        float x6 = cqk[b + 6U];
        float s6;
        if (r < 14U)
            s6 = __bfloat162float(
                __float2bfloat16(__bfloat162float(__float2bfloat16(
                                     __bfloat162float(__float2bfloat16(x6)) *
                                     (1.0f / sqrtf(128.0f)))) +
                                 (6U > r / 2U ? mk : 0.0f)));
        else
            s6 = -INFINITY;
        float mx = fmaxf(
            fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(-INFINITY, s0), s1), s2),
                                  s3),
                            s4),
                      s5),
                s6),
            -INFINITY);
        float shift = mx == -INFINITY ? 0.0f : mx;
        bool vr = r < 14U;
        float e0;
        if (vr)
            e0 = __expf(__fsub_rn(s0, shift));
        else
            e0 = 0.0f;
        float e1;
        if (vr)
            e1 = __expf(__fsub_rn(s1, shift));
        else
            e1 = 0.0f;
        float e2;
        if (vr)
            e2 = __expf(__fsub_rn(s2, shift));
        else
            e2 = 0.0f;
        float e3;
        if (vr)
            e3 = __expf(__fsub_rn(s3, shift));
        else
            e3 = 0.0f;
        float e4;
        if (vr)
            e4 = __expf(__fsub_rn(s4, shift));
        else
            e4 = 0.0f;
        float e5;
        if (vr)
            e5 = __expf(__fsub_rn(s5, shift));
        else
            e5 = 0.0f;
        float e6;
        if (vr)
            e6 = __expf(__fsub_rn(s6, shift));
        else
            e6 = 0.0f;
        float inv =
            __fdividef(1.0f, e0 + e1 + (e2 + e3) + (e4 + e5 + (e6 + 0.0f)));
        uint32_t o = rr * 64U;
        __nv_bfloat16 ite0;
        if (vr)
            ite0 = __float2bfloat16(e0 * inv);
        else
            ite0 = __float2bfloat16(0.0f);
        pr[o] = ite0;
        uint32_t ni7 = o + 1U;
        __nv_bfloat16 ite1;
        if (vr)
            ite1 = __float2bfloat16(e1 * inv);
        else
            ite1 = __float2bfloat16(0.0f);
        pr[ni7] = ite1;
        uint32_t ni8 = o + 2U;
        __nv_bfloat16 ite2;
        if (vr)
            ite2 = __float2bfloat16(e2 * inv);
        else
            ite2 = __float2bfloat16(0.0f);
        pr[ni8] = ite2;
        uint32_t ni9 = o + 3U;
        __nv_bfloat16 ite3;
        if (vr)
            ite3 = __float2bfloat16(e3 * inv);
        else
            ite3 = __float2bfloat16(0.0f);
        pr[ni9] = ite3;
        uint32_t ni10 = o + 4U;
        __nv_bfloat16 ite4;
        if (vr)
            ite4 = __float2bfloat16(e4 * inv);
        else
            ite4 = __float2bfloat16(0.0f);
        pr[ni10] = ite4;
        uint32_t ni11 = o + 5U;
        __nv_bfloat16 ite5;
        if (vr)
            ite5 = __float2bfloat16(e5 * inv);
        else
            ite5 = __float2bfloat16(0.0f);
        pr[ni11] = ite5;
        uint32_t ni12 = o + 6U;
        __nv_bfloat16 ite;
        if (vr)
            ite = __float2bfloat16(e6 * inv);
        else
            ite = __float2bfloat16(0.0f);
        pr[ni12] = ite;
        uint32_t o1 = rr * 64U;
        uint32_t k = 7U;
        while (k < 64U) {
            uint32_t kv = k;
            uint32_t ni13 = o1 + kv;
            pr[ni13] = __float2bfloat16(0.0f);
            k = kv + 1U;
        }
        i = rr + 1U;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_6(__nv_bfloat16 *pr, __nv_bfloat16 *vt, float *cpv)
{
    auto &fr = KPR_INIT(kpr_mma_fragment);
    uint32_t t = 0U;
    while (t < 128U) {
        uint32_t tv = t;
        uint32_t rat = tv / 16U;
        uint32_t cbt = tv;
        kpr_mma_fill(fr, 0.0f);
        uint32_t ch = 0U;
        while (ch < 4U) {
            uint32_t chv = ch;
            kpr_mma_mma_sync(pr + (64U * (rat * 16U) + chv * 16U), 64U,
                vt + (64U * (cbt * 8U) + chv * 16U), 64U, fr);
            ch = chv + 1U;
        }
        kpr_mma_store(fr, cpv + 8U * (tv * 16U), 8U);
        t = tv + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_7(float *cpv, __nv_bfloat16 *att)
{
    uint32_t i = 0U;
    while (i < 14336U) {
        uint32_t iv = i;
        uint32_t c = iv % 128U;
        att[iv] =
            __float2bfloat16(cpv[((iv % 2048U / 256U * 16U + c / 8U) * 16U +
                                     (iv / 2048U * 2U + iv % 256U / 128U)) *
                                     8U +
                                 c % 8U]);
        i = iv + 1U;
    }
    uint32_t j = 0U;
    while (j < 2048U) {
        uint32_t jv = j;
        uint32_t ni = 14336U + jv;
        att[ni] = __float2bfloat16(0.0f);
        j = jv + 1U;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_8(__nv_bfloat16 *input2, __nv_bfloat16 *att, float *cpj)
{
    auto &fr = KPR_INIT(kpr_mma_fragment);
    uint32_t t = 0U;
    while (t < 64U) {
        uint32_t tv = t;
        uint32_t rat = tv;
        uint32_t cbt = 0U;
        kpr_mma_fill(fr, 0.0f);
        uint32_t ch = 0U;
        while (ch < 128U) {
            uint32_t chv = ch;
            kpr_mma_mma_sync(input2 + (2048U * (rat * 16U) + chv * 16U), 2048U,
                att + (2048U * (cbt * 8U) + chv * 16U), 2048U, fr);
            ch = chv + 1U;
        }
        kpr_mma_store(fr, cpj + 8U * (tv * 16U), 8U);
        t = tv + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_9(float *cpj, __nv_bfloat16 *pj)
{
    uint32_t q = 0U;
    while (q < 8192U) {
        uint32_t qv = q;
        pj[qv] = __float2bfloat16(cpj[qv % 1024U * 8U + qv / 1024U]);
        q = qv + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_10(__nv_bfloat16 *pj, __nv_bfloat16 *input7,
    __nv_bfloat16 *res1, float *part256, __nv_bfloat16 *input3,
    __nv_bfloat16 *output1)
{
    uint32_t r = 0U;
    while (r < 8U) {
        uint32_t rv = r;
        uint32_t b = rv * 1024U;
        uint32_t c = 0U;
        while (c < 1024U) {
            uint32_t cv = c;
            float x = __bfloat162float(pj[b + cv]);
            uint32_t ni2 = b + cv;
            res1[ni2] = __float2bfloat16(x + __bfloat162float(input7[b + cv]));
            c = cv + 1U;
        }
        uint32_t t = 0U;
        while (t < 256U) {
            uint32_t tv = t;
            float v0 = __bfloat162float(res1[b + tv]);
            float a0 = __fmaf_rn(v0, v0, 0.0f);
            float v1 = __bfloat162float(res1[b + (tv + 256U)]);
            float a1 = __fmaf_rn(v1, v1, a0);
            float v2 = __bfloat162float(res1[b + (tv + 512U)]);
            float a2 = __fmaf_rn(v2, v2, a1);
            float v3 = __bfloat162float(res1[b + (tv + 768U)]);
            part256[tv] = __fmaf_rn(v3, v3, a2);
            t = tv + 1U;
        }
        uint32_t wi = 0U;
        while (wi < 8U) {
            uint32_t wv = wi;
            uint32_t d = 16U;
            while (d > 0U) {
                uint32_t dv = d;
                uint32_t t1 = 0U;
                while (t1 < dv) {
                    uint32_t tv = t1;
                    part256[wv * 32U + tv] += part256[wv * 32U + tv + dv];
                    t1 = tv + 1U;
                }
                d = dv / 2U;
            }
            part256[wv] = part256[wv * 32U];
            wi = wv + 1U;
        }
        uint32_t d = 4U;
        while (d > 0U) {
            uint32_t dv = d;
            uint32_t t1 = 0U;
            while (t1 < dv) {
                uint32_t tv = t1;
                part256[tv] += part256[tv + dv];
                t1 = tv + 1U;
            }
            d = dv / 2U;
        }
        float buf = 1000000.0f;
        float inv = rsqrtf(part256[0U] / 1024.0f + 1.0f / buf);
        uint32_t b1 = rv * 1024U;
        uint32_t c2 = 0U;
        while (c2 < 1024U) {
            uint32_t cv = c2;
            float v = __bfloat162float(res1[b1 + cv]);
            uint32_t ni1 = b1 + cv;
            output1[ni1] =
                __float2bfloat16(v * (inv * __bfloat162float(input3[cv])));
            c2 = cv + 1U;
        }
        r = rv + 1U;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_11(
    __nv_bfloat16 *input4, __nv_bfloat16 *output1, float *cgu)
{
    auto &fr = KPR_INIT(kpr_mma_fragment);
    uint32_t t = 0U;
    while (t < 384U) {
        uint32_t tv = t;
        uint32_t rat = tv / 8U * 4U + tv % 4U + tv % 8U / 4U * 192U;
        uint32_t cbt = 0U;
        kpr_mma_fill(fr, 0.0f);
        uint32_t ch = 0U;
        while (ch < 64U) {
            uint32_t chv = ch;
            kpr_mma_mma_sync(input4 + (1024U * (rat * 16U) + chv * 16U), 1024U,
                output1 + (1024U * (cbt * 8U) + chv * 16U), 1024U, fr);
            ch = chv + 1U;
        }
        kpr_mma_store(fr, cgu + 8U * (tv * 16U), 8U);
        t = tv + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_12(float *cgu, __nv_bfloat16 *gu)
{
    uint32_t q = 0U;
    while (q < 49152U) {
        uint32_t qv = q;
        gu[qv] = __float2bfloat16(cgu[qv % 6144U * 8U + qv / 6144U]);
        q = qv + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_13(__nv_bfloat16 *gu, __nv_bfloat16 *act)
{
    uint32_t q = 0U;
    while (q < 24576U) {
        uint32_t qv = q;
        uint32_t rv = qv / 3072U;
        uint32_t cv = qv % 3072U;
        uint32_t off = rv * 6144U + cv / 64U * 128U + cv % 64U;
        float g = __bfloat162float(gu[off]);
        __nv_bfloat16 a = __float2bfloat16(g / (1.0f + expf(-g)));
        float u = __bfloat162float(gu[off + 64U]);
        __nv_bfloat16 ite;
        if (rv < 7U)
            ite = __float2bfloat16(__bfloat162float(a) * u);
        else
            ite = __float2bfloat16(0.0f);
        act[qv] = ite;
        q = qv + 1U;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_14(__nv_bfloat16 *input5, __nv_bfloat16 *act, float *cpj)
{
    auto &fr = KPR_INIT(kpr_mma_fragment);
    uint32_t t = 0U;
    while (t < 64U) {
        uint32_t tv = t;
        uint32_t rat = tv;
        uint32_t cbt = 0U;
        kpr_mma_fill(fr, 0.0f);
        uint32_t ch = 0U;
        while (ch < 192U) {
            uint32_t chv = ch;
            kpr_mma_mma_sync(input5 + (3072U * (rat * 16U) + chv * 16U), 3072U,
                act + (3072U * (cbt * 8U) + chv * 16U), 3072U, fr);
            ch = chv + 1U;
        }
        kpr_mma_store(fr, cpj + 8U * (tv * 16U), 8U);
        t = tv + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_15(float *cpj, __nv_bfloat16 *pj)
{
    uint32_t q = 0U;
    while (q < 8192U) {
        uint32_t qv = q;
        pj[qv] = __float2bfloat16(cpj[qv % 1024U * 8U + qv / 1024U]);
        q = qv + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_16(__nv_bfloat16 *pj, __nv_bfloat16 *res1,
    __nv_bfloat16 *output0, float *part256, __nv_bfloat16 *input6,
    __nv_bfloat16 *output1)
{
    uint32_t r = 0U;
    while (r < 8U) {
        uint32_t rv = r;
        uint32_t b = rv * 1024U;
        uint32_t c = 0U;
        while (c < 1024U) {
            uint32_t cv = c;
            float x = __bfloat162float(pj[b + cv]);
            uint32_t ni2 = b + cv;
            output0[ni2] = __float2bfloat16(x + __bfloat162float(res1[b + cv]));
            c = cv + 1U;
        }
        uint32_t t = 0U;
        while (t < 256U) {
            uint32_t tv = t;
            float v0 = __bfloat162float(output0[b + tv]);
            float a0 = __fmaf_rn(v0, v0, 0.0f);
            float v1 = __bfloat162float(output0[b + (tv + 256U)]);
            float a1 = __fmaf_rn(v1, v1, a0);
            float v2 = __bfloat162float(output0[b + (tv + 512U)]);
            float a2 = __fmaf_rn(v2, v2, a1);
            float v3 = __bfloat162float(output0[b + (tv + 768U)]);
            part256[tv] = __fmaf_rn(v3, v3, a2);
            t = tv + 1U;
        }
        uint32_t wi = 0U;
        while (wi < 8U) {
            uint32_t wv = wi;
            uint32_t d = 16U;
            while (d > 0U) {
                uint32_t dv = d;
                uint32_t t1 = 0U;
                while (t1 < dv) {
                    uint32_t tv = t1;
                    part256[wv * 32U + tv] += part256[wv * 32U + tv + dv];
                    t1 = tv + 1U;
                }
                d = dv / 2U;
            }
            part256[wv] = part256[wv * 32U];
            wi = wv + 1U;
        }
        uint32_t d = 4U;
        while (d > 0U) {
            uint32_t dv = d;
            uint32_t t1 = 0U;
            while (t1 < dv) {
                uint32_t tv = t1;
                part256[tv] += part256[tv + dv];
                t1 = tv + 1U;
            }
            d = dv / 2U;
        }
        float buf = 1000000.0f;
        float inv = rsqrtf(part256[0U] / 1024.0f + 1.0f / buf);
        uint32_t b1 = rv * 1024U;
        uint32_t c2 = 0U;
        while (c2 < 1024U) {
            uint32_t cv = c2;
            float v = __bfloat162float(output0[b1 + cv]);
            uint32_t ni1 = b1 + cv;
            output1[ni1] =
                __float2bfloat16(v * (inv * __bfloat162float(input6[cv])));
            c2 = cv + 1U;
        }
        r = rv + 1U;
    }
}

void KPlay_Functional_kplay_run(__nv_bfloat16 *input0, __nv_bfloat16 *input1,
    __nv_bfloat16 *input2, __nv_bfloat16 *input3, __nv_bfloat16 *input4,
    __nv_bfloat16 *input5, __nv_bfloat16 *input6, __nv_bfloat16 *input7,
    __nv_bfloat16 *input8, __nv_bfloat16 *input9, __nv_bfloat16 *input10,
    int64_t input11, __nv_bfloat16 *output0, __nv_bfloat16 *output1)
{
    if (0LL <= input11 && input11 <= 4088LL) {
        uint32_t off = (uint32_t) input11;
        __nv_bfloat16 *qkv = (__nv_bfloat16 *) KPR_GPU_ALLOC(
            (uint32_t) sizeof(__nv_bfloat16), 32768U);
        float *cqkv = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 32768U);
        cudaStream_t s = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_kplay_run_0, 1U, 32U, 0U, s, input0, input8, cqkv);
        MUST(cudaStreamSynchronize(s));
        MUST(cudaStreamDestroy(s));
        cudaStream_t s1 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_kplay_run_1, 1U, 1U, 0U, s1, cqkv, qkv);
        MUST(cudaStreamSynchronize(s1));
        MUST(cudaStreamDestroy(s1));
        MUST(cudaFree(cqkv));
        float *part32 = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 32U);
        __nv_bfloat16 *un = (__nv_bfloat16 *) KPR_GPU_ALLOC(
            (uint32_t) sizeof(__nv_bfloat16), 128U);
        __nv_bfloat16 *qb = (__nv_bfloat16 *) KPR_GPU_ALLOC(
            (uint32_t) sizeof(__nv_bfloat16), 16384U);
        __nv_bfloat16 *kb = (__nv_bfloat16 *) KPR_GPU_ALLOC(
            (uint32_t) sizeof(__nv_bfloat16), 8192U);
        cudaStream_t s2 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_kplay_run_2, 1U, 1U, 0U, s2, qkv, part32, input1,
            un, off, input9, input10, qb, kb);
        MUST(cudaStreamSynchronize(s2));
        MUST(cudaStreamDestroy(s2));
        MUST(cudaFree(part32));
        MUST(cudaFree(un));
        __nv_bfloat16 *vt = (__nv_bfloat16 *) KPR_GPU_ALLOC(
            (uint32_t) sizeof(__nv_bfloat16), (uint32_t) 65536U);
        cudaStream_t s3 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_kplay_run_3, 1U, 1U, 0U, s3, qkv, vt);
        MUST(cudaStreamSynchronize(s3));
        MUST(cudaStreamDestroy(s3));
        MUST(cudaFree(qkv));
        float *cqk = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 1024U);
        cudaStream_t s4 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_kplay_run_4, 1U, 32U, 0U, s4, qb, kb, cqk);
        MUST(cudaStreamSynchronize(s4));
        MUST(cudaStreamDestroy(s4));
        MUST(cudaFree(qb));
        MUST(cudaFree(kb));
        __nv_bfloat16 *pr = (__nv_bfloat16 *) KPR_GPU_ALLOC(
            (uint32_t) sizeof(__nv_bfloat16), 8192U);
        cudaStream_t s5 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_kplay_run_5, 1U, 1U, 0U, s5, cqk, pr);
        MUST(cudaStreamSynchronize(s5));
        MUST(cudaStreamDestroy(s5));
        MUST(cudaFree(cqk));
        float *cpv = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 16384U);
        cudaStream_t s6 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_kplay_run_6, 1U, 32U, 0U, s6, pr, vt, cpv);
        MUST(cudaStreamSynchronize(s6));
        MUST(cudaStreamDestroy(s6));
        MUST(cudaFree(pr));
        MUST(cudaFree(vt));
        __nv_bfloat16 *att = (__nv_bfloat16 *) KPR_GPU_ALLOC(
            (uint32_t) sizeof(__nv_bfloat16), 16384U);
        cudaStream_t s7 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_kplay_run_7, 1U, 1U, 0U, s7, cpv, att);
        MUST(cudaStreamSynchronize(s7));
        MUST(cudaStreamDestroy(s7));
        MUST(cudaFree(cpv));
        float *cpj = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 8192U);
        cudaStream_t s8 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_kplay_run_8, 1U, 32U, 0U, s8, input2, att, cpj);
        MUST(cudaStreamSynchronize(s8));
        MUST(cudaStreamDestroy(s8));
        MUST(cudaFree(att));
        __nv_bfloat16 *pj = (__nv_bfloat16 *) KPR_GPU_ALLOC(
            (uint32_t) sizeof(__nv_bfloat16), 8192U);
        cudaStream_t s9 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_kplay_run_9, 1U, 1U, 0U, s9, cpj, pj);
        MUST(cudaStreamSynchronize(s9));
        MUST(cudaStreamDestroy(s9));
        __nv_bfloat16 *res1 = (__nv_bfloat16 *) KPR_GPU_ALLOC(
            (uint32_t) sizeof(__nv_bfloat16), 8192U);
        float *part256 =
            (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 256U);
        cudaStream_t s10 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_kplay_run_10, 1U, 1U, 0U, s10, pj, input7, res1,
            part256, input3, output1);
        MUST(cudaStreamSynchronize(s10));
        MUST(cudaStreamDestroy(s10));
        float *cgu = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 49152U);
        cudaStream_t s11 = KPR_FRESH_STREAM();
        KPR_KCALL(
            __hoisted_kplay_run_11, 1U, 32U, 0U, s11, input4, output1, cgu);
        MUST(cudaStreamSynchronize(s11));
        MUST(cudaStreamDestroy(s11));
        __nv_bfloat16 *gu = (__nv_bfloat16 *) KPR_GPU_ALLOC(
            (uint32_t) sizeof(__nv_bfloat16), 49152U);
        cudaStream_t s12 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_kplay_run_12, 1U, 1U, 0U, s12, cgu, gu);
        MUST(cudaStreamSynchronize(s12));
        MUST(cudaStreamDestroy(s12));
        MUST(cudaFree(cgu));
        __nv_bfloat16 *act = (__nv_bfloat16 *) KPR_GPU_ALLOC(
            (uint32_t) sizeof(__nv_bfloat16), 24576U);
        cudaStream_t s13 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_kplay_run_13, 1U, 1U, 0U, s13, gu, act);
        MUST(cudaStreamSynchronize(s13));
        MUST(cudaStreamDestroy(s13));
        MUST(cudaFree(gu));
        cudaStream_t s14 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_kplay_run_14, 1U, 32U, 0U, s14, input5, act, cpj);
        MUST(cudaStreamSynchronize(s14));
        MUST(cudaStreamDestroy(s14));
        MUST(cudaFree(act));
        cudaStream_t s15 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_kplay_run_15, 1U, 1U, 0U, s15, cpj, pj);
        MUST(cudaStreamSynchronize(s15));
        MUST(cudaStreamDestroy(s15));
        MUST(cudaFree(cpj));
        cudaStream_t s16 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_kplay_run_16, 1U, 1U, 0U, s16, pj, res1, output0,
            part256, input6, output1);
        MUST(cudaStreamSynchronize(s16));
        MUST(cudaStreamDestroy(s16));
        MUST(cudaFree(pj));
        MUST(cudaFree(res1));
        MUST(cudaFree(part256));
    }
}
