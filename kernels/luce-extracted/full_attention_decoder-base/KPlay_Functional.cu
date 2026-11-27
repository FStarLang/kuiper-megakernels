
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting t7_scores
*/
static void
__hoisted_t7_scores_0(
    uint32_t vh, uint32_t c, __nv_bfloat16 *kvo, __nv_bfloat16 *ka)
{
    uint32_t i = 0U;
    while (i < 4096U) {
        uint32_t vi = i;
        uint32_t r = vi / 256U;
        uint32_t j = vi % 256U;
        __nv_bfloat16 x = kvo[r < 8U ? (vh / 4U * c + r) * 256U + j : 0U];
        __nv_bfloat16 z = __float2bfloat16(0.0f);
        ka[vi] = r < 8U ? x : z;
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting t7_scores
*/
static void
__hoisted_t7_scores_1(__nv_bfloat16 *qn, uint32_t vh, __nv_bfloat16 *ubq)
{
    uint32_t i = 0U;
    while (i < 2048U) {
        uint32_t vi = i;
        ubq[vi] = qn[vh * 256U + vi % 256U];
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting t7_scores
*/
static void
__hoisted_t7_scores_2(__nv_bfloat16 *ka, __nv_bfloat16 *ubq, float *csc)
{
    auto &fr = KPR_INIT(kpr_mma_fragment);
    uint32_t r = 0U;
    while (r < 1U) {
        uint32_t vr = r;
        kpr_mma_fill(fr, 0.0f);
        uint32_t j = 0U;
        while (j < 16U) {
            uint32_t vj = j;
            kpr_mma_mma_sync(ka + (256U * (vr * 16U) + vj * 16U), 256U,
                ubq + vj * 16U, 256U, fr);
            j = vj + 1U;
        }
        kpr_mma_store(fr, csc + 8U * (vr * 16U), 8U);
        r = vr + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting t7_scores
*/
static void
__hoisted_t7_scores_3(float *csc, __nv_bfloat16 *dots, uint32_t vh)
{
    uint32_t i = 0U;
    while (i < 16U) {
        uint32_t vi = i;
        dots[vh * 16U + vi] = __float2bfloat16(csc[vi * 8U]);
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting t7_values
*/
static void
__hoisted_t7_values_0(
    uint32_t c, uint32_t vh, __nv_bfloat16 *kvo, __nv_bfloat16 *va)
{
    uint32_t half = c * 512U;
    uint32_t i = 0U;
    while (i < 4096U) {
        uint32_t vi = i;
        uint32_t j = vi / 16U;
        uint32_t pp = vi % 16U;
        __nv_bfloat16 x =
            kvo[pp < 8U ? half + (vh / 4U * c + pp) * 256U + j : 0U];
        __nv_bfloat16 z = __float2bfloat16(0.0f);
        va[vi] = pp < 8U ? x : z;
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting t7_values
*/
static void
__hoisted_t7_values_1(__nv_bfloat16 *probs, uint32_t vh, __nv_bfloat16 *ubp)
{
    uint32_t i = 0U;
    while (i < 128U) {
        uint32_t vi = i;
        ubp[vi] = probs[vh * 16U + vi % 16U];
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting t7_values
*/
static void
__hoisted_t7_values_2(__nv_bfloat16 *va, __nv_bfloat16 *ubp, float *cvl)
{
    auto &fr = KPR_INIT(kpr_mma_fragment);
    uint32_t r = 0U;
    while (r < 16U) {
        uint32_t vr = r;
        kpr_mma_fill(fr, 0.0f);
        uint32_t j = 0U;
        while (j < 1U) {
            uint32_t vj = j;
            kpr_mma_mma_sync(va + (16U * (vr * 16U) + vj * 16U), 16U,
                ubp + vj * 16U, 16U, fr);
            j = vj + 1U;
        }
        kpr_mma_store(fr, cvl + 8U * (vr * 16U), 8U);
        r = vr + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting t7_values
*/
static void
__hoisted_t7_values_3(float *cvl, float *vals, uint32_t vh)
{
    uint32_t i = 0U;
    while (i < 256U) {
        uint32_t vi = i;
        vals[vh * 256U + vi] = cvl[vi * 8U];
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_0(
    __nv_bfloat16 *x, float *tmp, __nv_bfloat16 *w, __nv_bfloat16 *hin)
{
    uint32_t g = 0U;
    while (g < 256U) {
        uint32_t vg = g;
        float acc = 0.0f;
        uint32_t j = 0U;
        while (j < 4U) {
            uint32_t vj = j;
            float v = __bfloat162float(x[4U * vg + vj]);
            acc += v * v;
            j = vj + 1U;
        }
        tmp[vg] = acc;
        g = vg + 1U;
    }
    uint32_t off = 128U;
    while (off > 0U) {
        uint32_t o = off;
        uint32_t i = 0U;
        while (i < o) {
            uint32_t vi = i;
            tmp[vi] += tmp[vi + o];
            i = vi + 1U;
        }
        off = o / 2U;
    }
    float inv = rsqrtf(tmp[0U] * 0.0009765625f +
                       0.0000009999999974752427078783512115478515625f);
    uint32_t i = 0U;
    while (i < 1024U) {
        uint32_t vi = i;
        float xv = __bfloat162float(x[vi]);
        hin[vi] = __float2bfloat16(xv * inv * (1.0f + __bfloat162float(w[vi])));
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_1(__nv_bfloat16 *hin, __nv_bfloat16 *ub1)
{
    uint32_t i = 0U;
    while (i < 8192U) {
        uint32_t vi = i;
        ub1[vi] = hin[vi % 1024U];
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_2(__nv_bfloat16 *wq, __nv_bfloat16 *ub1, float *cq)
{
    auto &fr = KPR_INIT(kpr_mma_fragment);
    uint32_t r = 0U;
    while (r < 256U) {
        uint32_t vr = r;
        kpr_mma_fill(fr, 0.0f);
        uint32_t j = 0U;
        while (j < 64U) {
            uint32_t vj = j;
            kpr_mma_mma_sync(wq + (1024U * (vr * 16U) + vj * 16U), 1024U,
                ub1 + vj * 16U, 1024U, fr);
            j = vj + 1U;
        }
        kpr_mma_store(fr, cq + 8U * (vr * 16U), 8U);
        r = vr + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_3(float *cq, __nv_bfloat16 *pq)
{
    uint32_t i = 0U;
    while (i < 4096U) {
        uint32_t vi = i;
        pq[vi] = __float2bfloat16(cq[vi * 8U]);
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_4(__nv_bfloat16 *wk, __nv_bfloat16 *ub1, float *ck)
{
    auto &fr = KPR_INIT(kpr_mma_fragment);
    uint32_t r = 0U;
    while (r < 32U) {
        uint32_t vr = r;
        kpr_mma_fill(fr, 0.0f);
        uint32_t j = 0U;
        while (j < 64U) {
            uint32_t vj = j;
            kpr_mma_mma_sync(wk + (1024U * (vr * 16U) + vj * 16U), 1024U,
                ub1 + vj * 16U, 1024U, fr);
            j = vj + 1U;
        }
        kpr_mma_store(fr, ck + 8U * (vr * 16U), 8U);
        r = vr + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_5(float *ck, __nv_bfloat16 *pk)
{
    uint32_t i = 0U;
    while (i < 512U) {
        uint32_t vi = i;
        pk[vi] = __float2bfloat16(ck[vi * 8U]);
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_6(__nv_bfloat16 *wv, __nv_bfloat16 *ub1, float *cv)
{
    auto &fr = KPR_INIT(kpr_mma_fragment);
    uint32_t r = 0U;
    while (r < 32U) {
        uint32_t vr = r;
        kpr_mma_fill(fr, 0.0f);
        uint32_t j = 0U;
        while (j < 64U) {
            uint32_t vj = j;
            kpr_mma_mma_sync(wv + (1024U * (vr * 16U) + vj * 16U), 1024U,
                ub1 + vj * 16U, 1024U, fr);
            j = vj + 1U;
        }
        kpr_mma_store(fr, cv + 8U * (vr * 16U), 8U);
        r = vr + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_7(float *cv, __nv_bfloat16 *pv)
{
    uint32_t i = 0U;
    while (i < 512U) {
        uint32_t vi = i;
        pv[vi] = __float2bfloat16(cv[vi * 8U]);
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_8(__nv_bfloat16 *pq, float *tmp, __nv_bfloat16 *w,
    __nv_bfloat16 *qn, __nv_bfloat16 *cs, __nv_bfloat16 *sn)
{
    uint32_t h = 0U;
    while (h < 8U) {
        uint32_t vh = h;
        uint32_t base = vh * 512U;
        uint32_t ob = vh * 256U;
        uint32_t g = 0U;
        while (g < 64U) {
            uint32_t vg = g;
            float acc = 0.0f;
            uint32_t j = 0U;
            while (j < 4U) {
                uint32_t vj = j;
                float v = __bfloat162float(pq[base + 4U * vg + vj]);
                acc += v * v;
                j = vj + 1U;
            }
            tmp[vg] = acc;
            g = vg + 1U;
        }
        uint32_t off = 32U;
        while (off > 0U) {
            uint32_t o = off;
            uint32_t i = 0U;
            while (i < o) {
                uint32_t vi = i;
                tmp[vi] += tmp[vi + o];
                i = vi + 1U;
            }
            off = o / 2U;
        }
        float inv = rsqrtf(tmp[0U] * 0.00390625f +
                           0.0000009999999974752427078783512115478515625f);
        uint32_t i = 0U;
        while (i < 256U) {
            uint32_t vi = i;
            float xv = __bfloat162float(pq[base + vi]);
            qn[ob + vi] = __float2bfloat16(
                xv * inv * (1.0f + __bfloat162float(w[16254976U + vi])));
            i = vi + 1U;
        }
        uint32_t r = 0U;
        while (r < 32U) {
            uint32_t vr = r;
            float a = __bfloat162float(qn[ob + vr]);
            float b = __bfloat162float(qn[ob + 32U + vr]);
            float c = __bfloat162float(cs[vr]);
            float sv = __bfloat162float(sn[vr]);
            float ac = __bfloat162float(__float2bfloat16(a * c));
            float bs = __bfloat162float(__float2bfloat16(-b * sv));
            float bc = __bfloat162float(__float2bfloat16(b * c));
            float as_ = __bfloat162float(__float2bfloat16(a * sv));
            __nv_bfloat16 va = __float2bfloat16(ac + bs);
            __nv_bfloat16 vb = __float2bfloat16(bc + as_);
            qn[ob + vr] = va;
            qn[ob + 32U + vr] = vb;
            r = vr + 1U;
        }
        h = vh + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_9(__nv_bfloat16 *pk, float *tmp, __nv_bfloat16 *w,
    __nv_bfloat16 *kn, __nv_bfloat16 *cs, __nv_bfloat16 *sn)
{
    uint32_t h = 0U;
    while (h < 2U) {
        uint32_t vh = h;
        uint32_t base = vh * 256U;
        uint32_t ob = vh * 256U;
        uint32_t g = 0U;
        while (g < 64U) {
            uint32_t vg = g;
            float acc = 0.0f;
            uint32_t j = 0U;
            while (j < 4U) {
                uint32_t vj = j;
                float v = __bfloat162float(pk[base + 4U * vg + vj]);
                acc += v * v;
                j = vj + 1U;
            }
            tmp[vg] = acc;
            g = vg + 1U;
        }
        uint32_t off = 32U;
        while (off > 0U) {
            uint32_t o = off;
            uint32_t i = 0U;
            while (i < o) {
                uint32_t vi = i;
                tmp[vi] += tmp[vi + o];
                i = vi + 1U;
            }
            off = o / 2U;
        }
        float inv = rsqrtf(tmp[0U] * 0.00390625f +
                           0.0000009999999974752427078783512115478515625f);
        uint32_t i = 0U;
        while (i < 256U) {
            uint32_t vi = i;
            float xv = __bfloat162float(pk[base + vi]);
            kn[ob + vi] = __float2bfloat16(
                xv * inv * (1.0f + __bfloat162float(w[16255232U + vi])));
            i = vi + 1U;
        }
        uint32_t r = 0U;
        while (r < 32U) {
            uint32_t vr = r;
            float a = __bfloat162float(kn[ob + vr]);
            float b = __bfloat162float(kn[ob + 32U + vr]);
            float c = __bfloat162float(cs[vr]);
            float sv = __bfloat162float(sn[vr]);
            float ac = __bfloat162float(__float2bfloat16(a * c));
            float bs = __bfloat162float(__float2bfloat16(-b * sv));
            float bc = __bfloat162float(__float2bfloat16(b * c));
            float as_ = __bfloat162float(__float2bfloat16(a * sv));
            __nv_bfloat16 va = __float2bfloat16(ac + bs);
            __nv_bfloat16 vb = __float2bfloat16(bc + as_);
            kn[ob + vr] = va;
            kn[ob + 32U + vr] = vb;
            r = vr + 1U;
        }
        h = vh + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_10(uint32_t c, uint32_t nkv, __nv_bfloat16 *kn,
    __nv_bfloat16 *pv, __nv_bfloat16 *kv, uint32_t ts, __nv_bfloat16 *kvo)
{
    uint32_t half = c * 512U;
    uint32_t i = 0U;
    while (i < nkv) {
        uint32_t vi = i;
        uint32_t comp = vi % (c * 512U) / (c * 256U) * 256U + vi % 256U;
        __nv_bfloat16 a = kn[comp];
        __nv_bfloat16 b = pv[comp];
        __nv_bfloat16 o = kv[vi];
        kvo[vi] =
            vi % (c * 512U) % (c * 256U) / 256U == ts ? vi < half ? a : b : o;
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_11(uint32_t ts, uint32_t c, __nv_bfloat16 *qn,
    __nv_bfloat16 *kvo, float *tmp, __nv_bfloat16 *dots)
{
    uint32_t h = 0U;
    while (h < 8U) {
        uint32_t vh = h;
        uint32_t p = 0U;
        while (p <= ts) {
            uint32_t vp = p;
            uint32_t kb = (vh / 4U * c + vp) * 256U;
            uint32_t g = 0U;
            while (g < 16U) {
                uint32_t vg = g;
                float acc = 0.0f;
                uint32_t j = 0U;
                while (j < 16U) {
                    uint32_t vj = j;
                    uint32_t ii = 16U * vg + vj;
                    float q = __bfloat162float(qn[vh * 256U + ii]);
                    float kk = __bfloat162float(kvo[kb + ii]);
                    acc += q * kk;
                    j = vj + 1U;
                }
                tmp[vg] = acc;
                g = vg + 1U;
            }
            uint32_t off = 8U;
            while (off > 0U) {
                uint32_t o = off;
                uint32_t i = 0U;
                while (i < o) {
                    uint32_t vi = i;
                    tmp[vi] += tmp[vi + o];
                    i = vi + 1U;
                }
                off = o / 2U;
            }
            dots[vh * 16U + vp] = __float2bfloat16(tmp[0U]);
            p = vp + 1U;
        }
        h = vh + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_12(
    uint32_t ts, __nv_bfloat16 *dots, float *tmp, __nv_bfloat16 *probs)
{
    uint32_t wd = 1U;
    while (wd <= ts)
        wd *= 2U;
    uint32_t width = wd;
    uint32_t h = 0U;
    while (h < 8U) {
        uint32_t vh = h;
        uint32_t p = 0U;
        while (p <= ts) {
            uint32_t vp = p;
            tmp[16U + vp] = __bfloat162float(__float2bfloat16(
                __bfloat162float(dots[vh * 16U + vp]) * 0.0625f));
            p = vp + 1U;
        }
        float mx = tmp[16U];
        uint32_t p2 = 1U;
        while (p2 <= ts) {
            uint32_t vp = p2;
            float sv = tmp[16U + vp];
            float m = mx;
            mx = m < sv ? sv : m;
            p2 = vp + 1U;
        }
        float m = mx;
        uint32_t p3 = 0U;
        while (p3 < width) {
            uint32_t vp = p3;
            float sv = tmp[16U + vp];
            float e = expf(sv - m);
            bool le = vp <= ts;
            tmp[16U + vp] = le ? e : sv;
            tmp[vp] = le ? e : 0.0f;
            p3 = vp + 1U;
        }
        uint32_t off = width / 2U;
        while (off > 0U) {
            uint32_t o = off;
            uint32_t i = 0U;
            while (i < o) {
                uint32_t vi = i;
                tmp[vi] += tmp[vi + o];
                i = vi + 1U;
            }
            off = o / 2U;
        }
        float sum = tmp[0U];
        uint32_t p4 = 0U;
        while (p4 < 16U) {
            uint32_t vp = p4;
            __nv_bfloat16 r = __float2bfloat16(tmp[16U + vp] / sum);
            __nv_bfloat16 z = __float2bfloat16(0.0f);
            probs[vh * 16U + vp] = vp <= ts ? r : z;
            p4 = vp + 1U;
        }
        h = vh + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_13(uint32_t c, float *tmp, uint32_t ts, __nv_bfloat16 *kvo,
    __nv_bfloat16 *probs, float *vals)
{
    uint32_t half = c * 512U;
    uint32_t h = 0U;
    while (h < 8U) {
        uint32_t vh = h;
        uint32_t i = 0U;
        while (i < 256U) {
            uint32_t vi = i;
            tmp[0U] = 0.0f;
            tmp[1U] = 0.0f;
            tmp[2U] = 0.0f;
            tmp[3U] = 0.0f;
            uint32_t p = 0U;
            while (p <= ts) {
                uint32_t vp = p;
                float v = __bfloat162float(
                    kvo[half + (vh / 4U * c + vp) * 256U + vi]);
                float pr = __bfloat162float(probs[vh * 16U + vp]);
                uint32_t g = vp / 4U;
                tmp[g] = __fmaf_rn(pr, v, tmp[g]);
                p = vp + 1U;
            }
            float a0 = tmp[0U];
            float a1 = tmp[1U];
            float a2 = tmp[2U];
            float a3 = tmp[3U];
            float v0 = __bfloat162float(kvo[half + vh / 4U * c * 256U + vi]);
            float pr0 = __bfloat162float(probs[vh * 16U]);
            vals[vh * 256U + vi] = ts == 0U ? pr0 * v0 : a0 + a2 + (a1 + a3);
            i = vi + 1U;
        }
        h = vh + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_14(float *vals, __nv_bfloat16 *pq, __nv_bfloat16 *gated)
{
    uint32_t i = 0U;
    while (i < 2048U) {
        uint32_t vi = i;
        float v = vals[vi];
        float g = __bfloat162float(pq[vi / 256U * 512U + 256U + vi % 256U]);
        gated[vi] = __float2bfloat16(
            __bfloat162float(__float2bfloat16(v)) *
            __bfloat162float(__float2bfloat16(1.0f / (1.0f + expf(-g)))));
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_15(__nv_bfloat16 *gated, __nv_bfloat16 *ubg)
{
    uint32_t i = 0U;
    while (i < 16384U) {
        uint32_t vi = i;
        ubg[vi] = gated[vi % 2048U];
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_16(__nv_bfloat16 *wo, __nv_bfloat16 *ubg, float *co)
{
    auto &fr = KPR_INIT(kpr_mma_fragment);
    uint32_t r = 0U;
    while (r < 64U) {
        uint32_t vr = r;
        kpr_mma_fill(fr, 0.0f);
        uint32_t j = 0U;
        while (j < 128U) {
            uint32_t vj = j;
            kpr_mma_mma_sync(wo + (2048U * (vr * 16U) + vj * 16U), 2048U,
                ubg + vj * 16U, 2048U, fr);
            j = vj + 1U;
        }
        kpr_mma_store(fr, co + 8U * (vr * 16U), 8U);
        r = vr + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_17(float *co, __nv_bfloat16 *po)
{
    uint32_t i = 0U;
    while (i < 1024U) {
        uint32_t vi = i;
        po[vi] = __float2bfloat16(co[vi * 8U]);
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_18(__nv_bfloat16 *x, __nv_bfloat16 *po, __nv_bfloat16 *x1)
{
    uint32_t i = 0U;
    while (i < 1024U) {
        uint32_t vi = i;
        float u = __bfloat162float(x[vi]);
        x1[vi] = __float2bfloat16(u + __bfloat162float(po[vi]));
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_19(
    __nv_bfloat16 *x1, float *tmp, __nv_bfloat16 *w, __nv_bfloat16 *h2)
{
    uint32_t g = 0U;
    while (g < 256U) {
        uint32_t vg = g;
        float acc = 0.0f;
        uint32_t j = 0U;
        while (j < 4U) {
            uint32_t vj = j;
            float v = __bfloat162float(x1[4U * vg + vj]);
            acc += v * v;
            j = vj + 1U;
        }
        tmp[vg] = acc;
        g = vg + 1U;
    }
    uint32_t off = 128U;
    while (off > 0U) {
        uint32_t o = off;
        uint32_t i = 0U;
        while (i < o) {
            uint32_t vi = i;
            tmp[vi] += tmp[vi + o];
            i = vi + 1U;
        }
        off = o / 2U;
    }
    float inv = rsqrtf(tmp[0U] * 0.0009765625f +
                       0.0000009999999974752427078783512115478515625f);
    uint32_t i = 0U;
    while (i < 1024U) {
        uint32_t vi = i;
        float xv = __bfloat162float(x1[vi]);
        h2[vi] = __float2bfloat16(
            xv * inv * (1.0f + __bfloat162float(w[1024U + vi])));
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_20(__nv_bfloat16 *h2, __nv_bfloat16 *ub2)
{
    uint32_t i = 0U;
    while (i < 8192U) {
        uint32_t vi = i;
        ub2[vi] = h2[vi % 1024U];
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_21(__nv_bfloat16 *wg, __nv_bfloat16 *ub2, float *cg)
{
    auto &fr = KPR_INIT(kpr_mma_fragment);
    uint32_t r = 0U;
    while (r < 224U) {
        uint32_t vr = r;
        kpr_mma_fill(fr, 0.0f);
        uint32_t j = 0U;
        while (j < 64U) {
            uint32_t vj = j;
            kpr_mma_mma_sync(wg + (1024U * (vr * 16U) + vj * 16U), 1024U,
                ub2 + vj * 16U, 1024U, fr);
            j = vj + 1U;
        }
        kpr_mma_store(fr, cg + 8U * (vr * 16U), 8U);
        r = vr + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_22(float *cg, __nv_bfloat16 *g)
{
    uint32_t i = 0U;
    while (i < 3584U) {
        uint32_t vi = i;
        g[vi] = __float2bfloat16(cg[vi * 8U]);
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_23(__nv_bfloat16 *wu, __nv_bfloat16 *ub2, float *cg)
{
    auto &fr = KPR_INIT(kpr_mma_fragment);
    uint32_t r = 0U;
    while (r < 224U) {
        uint32_t vr = r;
        kpr_mma_fill(fr, 0.0f);
        uint32_t j = 0U;
        while (j < 64U) {
            uint32_t vj = j;
            kpr_mma_mma_sync(wu + (1024U * (vr * 16U) + vj * 16U), 1024U,
                ub2 + vj * 16U, 1024U, fr);
            j = vj + 1U;
        }
        kpr_mma_store(fr, cg + 8U * (vr * 16U), 8U);
        r = vr + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_24(float *cg, __nv_bfloat16 *u)
{
    uint32_t i = 0U;
    while (i < 3584U) {
        uint32_t vi = i;
        u[vi] = __float2bfloat16(cg[vi * 8U]);
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_25(__nv_bfloat16 *g, __nv_bfloat16 *u, __nv_bfloat16 *act)
{
    uint32_t i = 0U;
    while (i < 3584U) {
        uint32_t vi = i;
        float gv = __bfloat162float(g[vi]);
        float uv = __bfloat162float(u[vi]);
        act[vi] = __float2bfloat16(
            __bfloat162float(__float2bfloat16(gv / (1.0f + expf(-gv)))) * uv);
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_26(__nv_bfloat16 *act, __nv_bfloat16 *uba)
{
    uint32_t i = 0U;
    while (i < 28672U) {
        uint32_t vi = i;
        uba[vi] = act[vi % 3584U];
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_27(__nv_bfloat16 *wd, __nv_bfloat16 *uba, float *cd0)
{
    auto &fr = KPR_INIT(kpr_mma_fragment);
    uint32_t r = 0U;
    while (r < 64U) {
        uint32_t vr = r;
        kpr_mma_fill(fr, 0.0f);
        uint32_t j = 0U;
        while (j < 112U) {
            uint32_t vj = j;
            kpr_mma_mma_sync(wd + (3584U * (vr * 16U) + vj * 16U), 3584U,
                uba + vj * 16U, 3584U, fr);
            j = vj + 1U;
        }
        kpr_mma_store(fr, cd0 + 8U * (vr * 16U), 8U);
        r = vr + 1U;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_28(__nv_bfloat16 *wd, __nv_bfloat16 *uba, float *cd1)
{
    auto &fr = KPR_INIT(kpr_mma_fragment);
    uint32_t r = 0U;
    while (r < 64U) {
        uint32_t vr = r;
        kpr_mma_fill(fr, 0.0f);
        uint32_t j = 112U;
        while (j < 224U) {
            uint32_t vj = j;
            kpr_mma_mma_sync(wd + (3584U * (vr * 16U) + vj * 16U), 3584U,
                uba + vj * 16U, 3584U, fr);
            j = vj + 1U;
        }
        kpr_mma_store(fr, cd1 + 8U * (vr * 16U), 8U);
        r = vr + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_29(float *cd0, float *cd1, __nv_bfloat16 *pd)
{
    uint32_t i = 0U;
    while (i < 1024U) {
        uint32_t vi = i;
        pd[vi] = __float2bfloat16(cd0[vi * 8U] + cd1[vi * 8U]);
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_main
*/
static void
__hoisted_run_main_30(__nv_bfloat16 *x1, __nv_bfloat16 *pd, __nv_bfloat16 *xo)
{
    uint32_t i = 0U;
    while (i < 1024U) {
        uint32_t vi = i;
        float u = __bfloat162float(x1[vi]);
        xo[vi] = __float2bfloat16(u + __bfloat162float(pd[vi]));
        i = vi + 1U;
    }
}

uint32_t KPlay_Functional_i64_to_sz(int64_t t)
{
    if (t == 0LL)
        return 0U;
    else if (t == 1LL)
        return 1U;
    else if (t == 2LL)
        return 2U;
    else if (t == 3LL)
        return 3U;
    else if (t == 4LL)
        return 4U;
    else if (t == 5LL)
        return 5U;
    else if (t == 6LL)
        return 6U;
    else if (t == 7LL)
        return 7U;
    else if (t == 8LL)
        return 8U;
    else if (t == 9LL)
        return 9U;
    else if (t == 10LL)
        return 10U;
    else
        return 11U;
}

void KPlay_Functional_t7_scores(uint32_t nkv, uint32_t c, __nv_bfloat16 *kvo,
    __nv_bfloat16 *qn, __nv_bfloat16 *dots)
{
    KRML_MAYBE_UNUSED_VAR(nkv);
    __nv_bfloat16 *ka = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 4096U);
    __nv_bfloat16 *ubq = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 2048U);
    float *csc = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 128U);
    uint32_t h = 0U;
    while (h < 8U) {
        uint32_t vh = h;
        cudaStream_t s = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_t7_scores_0, 1U, 1U, 0U, s, vh, c, kvo, ka);
        MUST(cudaStreamSynchronize(s));
        MUST(cudaStreamDestroy(s));
        cudaStream_t s1 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_t7_scores_1, 1U, 1U, 0U, s1, qn, vh, ubq);
        MUST(cudaStreamSynchronize(s1));
        MUST(cudaStreamDestroy(s1));
        cudaStream_t s2 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_t7_scores_2, 1U, 32U, 0U, s2, ka, ubq, csc);
        MUST(cudaStreamSynchronize(s2));
        MUST(cudaStreamDestroy(s2));
        cudaStream_t s3 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_t7_scores_3, 1U, 1U, 0U, s3, csc, dots, vh);
        MUST(cudaStreamSynchronize(s3));
        MUST(cudaStreamDestroy(s3));
        h = vh + 1U;
    }
    MUST(cudaFree(ka));
    MUST(cudaFree(ubq));
    MUST(cudaFree(csc));
}

void KPlay_Functional_t7_values(uint32_t nkv, uint32_t c, __nv_bfloat16 *kvo,
    __nv_bfloat16 *probs, float *vals)
{
    KRML_MAYBE_UNUSED_VAR(nkv);
    __nv_bfloat16 *va = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 4096U);
    __nv_bfloat16 *ubp =
        (__nv_bfloat16 *) KPR_GPU_ALLOC((uint32_t) sizeof(__nv_bfloat16), 128U);
    float *cvl = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 2048U);
    uint32_t h = 0U;
    while (h < 8U) {
        uint32_t vh = h;
        cudaStream_t s = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_t7_values_0, 1U, 1U, 0U, s, c, vh, kvo, va);
        MUST(cudaStreamSynchronize(s));
        MUST(cudaStreamDestroy(s));
        cudaStream_t s1 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_t7_values_1, 1U, 1U, 0U, s1, probs, vh, ubp);
        MUST(cudaStreamSynchronize(s1));
        MUST(cudaStreamDestroy(s1));
        cudaStream_t s2 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_t7_values_2, 1U, 32U, 0U, s2, va, ubp, cvl);
        MUST(cudaStreamSynchronize(s2));
        MUST(cudaStreamDestroy(s2));
        cudaStream_t s3 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_t7_values_3, 1U, 1U, 0U, s3, cvl, vals, vh);
        MUST(cudaStreamSynchronize(s3));
        MUST(cudaStreamDestroy(s3));
        h = vh + 1U;
    }
    MUST(cudaFree(va));
    MUST(cudaFree(ubp));
    MUST(cudaFree(cvl));
}

void KPlay_Functional_run_main(uint32_t nkv, uint32_t c, uint32_t ts,
    __nv_bfloat16 *w, __nv_bfloat16 *x, __nv_bfloat16 *kv, __nv_bfloat16 *cs,
    __nv_bfloat16 *sn, __nv_bfloat16 *xo, __nv_bfloat16 *kvo)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 18352640U);
    __nv_bfloat16 *hw =
        (__nv_bfloat16 *) KRML_HOST_MALLOC(sizeof(__nv_bfloat16) * 18352640U);
    if (hw != NULL)
        for (uint32_t _i = 0U; _i < 18352640U; ++_i)
            hw[_i] = __float2bfloat16(0.0f);
    MUST(cudaMemcpy(hw, w,
        (uint64_t) sizeof(__nv_bfloat16) * (uint64_t) 18352640U,
        cudaMemcpyDeviceToHost));
    float *tmp = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 1024U);
    __nv_bfloat16 *hin = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 1024U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_0, 1U, 1U, 0U, s, x, tmp, w, hin);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    __nv_bfloat16 *ub1 = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 8192U);
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_1, 1U, 1U, 0U, s1, hin, ub1);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    __nv_bfloat16 *a = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 4194304U);
    MUST(cudaMemcpy(a, hw + 11012096U,
        (uint64_t) sizeof(__nv_bfloat16) * (uint64_t) 4194304U,
        cudaMemcpyHostToDevice));
    float *cq = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 32768U);
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_2, 1U, 32U, 0U, s2, a, ub1, cq);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    MUST(cudaFree(a));
    __nv_bfloat16 *pq = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 4096U);
    cudaStream_t s3 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_3, 1U, 1U, 0U, s3, cq, pq);
    MUST(cudaStreamSynchronize(s3));
    MUST(cudaStreamDestroy(s3));
    MUST(cudaFree(cq));
    __nv_bfloat16 *a1 = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 524288U);
    MUST(cudaMemcpy(a1, hw + 15206400U,
        (uint64_t) sizeof(__nv_bfloat16) * (uint64_t) 524288U,
        cudaMemcpyHostToDevice));
    float *ck = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 4096U);
    cudaStream_t s4 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_4, 1U, 32U, 0U, s4, a1, ub1, ck);
    MUST(cudaStreamSynchronize(s4));
    MUST(cudaStreamDestroy(s4));
    MUST(cudaFree(a1));
    __nv_bfloat16 *pk =
        (__nv_bfloat16 *) KPR_GPU_ALLOC((uint32_t) sizeof(__nv_bfloat16), 512U);
    cudaStream_t s5 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_5, 1U, 1U, 0U, s5, ck, pk);
    MUST(cudaStreamSynchronize(s5));
    MUST(cudaStreamDestroy(s5));
    MUST(cudaFree(ck));
    __nv_bfloat16 *a2 = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 524288U);
    MUST(cudaMemcpy(a2, hw + 15730688U,
        (uint64_t) sizeof(__nv_bfloat16) * (uint64_t) 524288U,
        cudaMemcpyHostToDevice));
    float *cv = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 4096U);
    cudaStream_t s6 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_6, 1U, 32U, 0U, s6, a2, ub1, cv);
    MUST(cudaStreamSynchronize(s6));
    MUST(cudaStreamDestroy(s6));
    MUST(cudaFree(a2));
    __nv_bfloat16 *pv =
        (__nv_bfloat16 *) KPR_GPU_ALLOC((uint32_t) sizeof(__nv_bfloat16), 512U);
    cudaStream_t s7 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_7, 1U, 1U, 0U, s7, cv, pv);
    MUST(cudaStreamSynchronize(s7));
    MUST(cudaStreamDestroy(s7));
    MUST(cudaFree(cv));
    MUST(cudaFree(ub1));
    __nv_bfloat16 *qn = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 2048U);
    cudaStream_t s8 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_8, 1U, 1U, 0U, s8, pq, tmp, w, qn, cs, sn);
    MUST(cudaStreamSynchronize(s8));
    MUST(cudaStreamDestroy(s8));
    __nv_bfloat16 *kn =
        (__nv_bfloat16 *) KPR_GPU_ALLOC((uint32_t) sizeof(__nv_bfloat16), 512U);
    cudaStream_t s9 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_9, 1U, 1U, 0U, s9, pk, tmp, w, kn, cs, sn);
    MUST(cudaStreamSynchronize(s9));
    MUST(cudaStreamDestroy(s9));
    cudaStream_t s10 = KPR_FRESH_STREAM();
    KPR_KCALL(
        __hoisted_run_main_10, 1U, 1U, 0U, s10, c, nkv, kn, pv, kv, ts, kvo);
    MUST(cudaStreamSynchronize(s10));
    MUST(cudaStreamDestroy(s10));
    __nv_bfloat16 *dots =
        (__nv_bfloat16 *) KPR_GPU_ALLOC((uint32_t) sizeof(__nv_bfloat16), 128U);
    if (ts == 7U)
        KPlay_Functional_t7_scores(nkv, c, kvo, qn, dots);
    else {
        cudaStream_t s11 = KPR_FRESH_STREAM();
        KPR_KCALL(
            __hoisted_run_main_11, 1U, 1U, 0U, s11, ts, c, qn, kvo, tmp, dots);
        MUST(cudaStreamSynchronize(s11));
        MUST(cudaStreamDestroy(s11));
    }
    __nv_bfloat16 *probs =
        (__nv_bfloat16 *) KPR_GPU_ALLOC((uint32_t) sizeof(__nv_bfloat16), 128U);
    cudaStream_t s11 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_12, 1U, 1U, 0U, s11, ts, dots, tmp, probs);
    MUST(cudaStreamSynchronize(s11));
    MUST(cudaStreamDestroy(s11));
    float *vals = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 2048U);
    if (ts == 7U)
        KPlay_Functional_t7_values(nkv, c, kvo, probs, vals);
    else {
        cudaStream_t s12 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_run_main_13, 1U, 1U, 0U, s12, c, tmp, ts, kvo,
            probs, vals);
        MUST(cudaStreamSynchronize(s12));
        MUST(cudaStreamDestroy(s12));
    }
    __nv_bfloat16 *gated = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 2048U);
    cudaStream_t s12 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_14, 1U, 1U, 0U, s12, vals, pq, gated);
    MUST(cudaStreamSynchronize(s12));
    MUST(cudaStreamDestroy(s12));
    __nv_bfloat16 *ubg = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 16384U);
    cudaStream_t s13 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_15, 1U, 1U, 0U, s13, gated, ubg);
    MUST(cudaStreamSynchronize(s13));
    MUST(cudaStreamDestroy(s13));
    __nv_bfloat16 *a3 = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 2097152U);
    MUST(cudaMemcpy(a3, hw + 16255488U,
        (uint64_t) sizeof(__nv_bfloat16) * (uint64_t) 2097152U,
        cudaMemcpyHostToDevice));
    float *co = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 8192U);
    cudaStream_t s14 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_16, 1U, 32U, 0U, s14, a3, ubg, co);
    MUST(cudaStreamSynchronize(s14));
    MUST(cudaStreamDestroy(s14));
    MUST(cudaFree(a3));
    __nv_bfloat16 *po = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 1024U);
    cudaStream_t s15 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_17, 1U, 1U, 0U, s15, co, po);
    MUST(cudaStreamSynchronize(s15));
    MUST(cudaStreamDestroy(s15));
    MUST(cudaFree(co));
    __nv_bfloat16 *x1 = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 1024U);
    cudaStream_t s16 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_18, 1U, 1U, 0U, s16, x, po, x1);
    MUST(cudaStreamSynchronize(s16));
    MUST(cudaStreamDestroy(s16));
    __nv_bfloat16 *h2 = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 1024U);
    cudaStream_t s17 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_19, 1U, 1U, 0U, s17, x1, tmp, w, h2);
    MUST(cudaStreamSynchronize(s17));
    MUST(cudaStreamDestroy(s17));
    __nv_bfloat16 *ub2 = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 8192U);
    cudaStream_t s18 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_20, 1U, 1U, 0U, s18, h2, ub2);
    MUST(cudaStreamSynchronize(s18));
    MUST(cudaStreamDestroy(s18));
    __nv_bfloat16 *a4 = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 3670016U);
    MUST(cudaMemcpy(a4, hw + 2048U,
        (uint64_t) sizeof(__nv_bfloat16) * (uint64_t) 3670016U,
        cudaMemcpyHostToDevice));
    float *cg = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 28672U);
    cudaStream_t s19 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_21, 1U, 32U, 0U, s19, a4, ub2, cg);
    MUST(cudaStreamSynchronize(s19));
    MUST(cudaStreamDestroy(s19));
    MUST(cudaFree(a4));
    __nv_bfloat16 *g = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 3584U);
    cudaStream_t s20 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_22, 1U, 1U, 0U, s20, cg, g);
    MUST(cudaStreamSynchronize(s20));
    MUST(cudaStreamDestroy(s20));
    __nv_bfloat16 *a5 = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 3670016U);
    MUST(cudaMemcpy(a5, hw + 3672064U,
        (uint64_t) sizeof(__nv_bfloat16) * (uint64_t) 3670016U,
        cudaMemcpyHostToDevice));
    cudaStream_t s21 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_23, 1U, 32U, 0U, s21, a5, ub2, cg);
    MUST(cudaStreamSynchronize(s21));
    MUST(cudaStreamDestroy(s21));
    MUST(cudaFree(a5));
    __nv_bfloat16 *u = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 3584U);
    cudaStream_t s22 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_24, 1U, 1U, 0U, s22, cg, u);
    MUST(cudaStreamSynchronize(s22));
    MUST(cudaStreamDestroy(s22));
    MUST(cudaFree(cg));
    __nv_bfloat16 *act = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 3584U);
    cudaStream_t s23 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_25, 1U, 1U, 0U, s23, g, u, act);
    MUST(cudaStreamSynchronize(s23));
    MUST(cudaStreamDestroy(s23));
    __nv_bfloat16 *uba = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 28672U);
    cudaStream_t s24 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_26, 1U, 1U, 0U, s24, act, uba);
    MUST(cudaStreamSynchronize(s24));
    MUST(cudaStreamDestroy(s24));
    __nv_bfloat16 *a6 = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 3670016U);
    MUST(cudaMemcpy(a6, hw + 7342080U,
        (uint64_t) sizeof(__nv_bfloat16) * (uint64_t) 3670016U,
        cudaMemcpyHostToDevice));
    float *cd0 = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 8192U);
    float *cd1 = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 8192U);
    cudaStream_t s25 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_27, 1U, 32U, 0U, s25, a6, uba, cd0);
    MUST(cudaStreamSynchronize(s25));
    MUST(cudaStreamDestroy(s25));
    cudaStream_t s26 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_28, 1U, 32U, 0U, s26, a6, uba, cd1);
    MUST(cudaStreamSynchronize(s26));
    MUST(cudaStreamDestroy(s26));
    MUST(cudaFree(a6));
    __nv_bfloat16 *pd = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 1024U);
    cudaStream_t s27 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_29, 1U, 1U, 0U, s27, cd0, cd1, pd);
    MUST(cudaStreamSynchronize(s27));
    MUST(cudaStreamDestroy(s27));
    cudaStream_t s28 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_main_30, 1U, 1U, 0U, s28, x1, pd, xo);
    MUST(cudaStreamSynchronize(s28));
    MUST(cudaStreamDestroy(s28));
    MUST(cudaFree(cd0));
    MUST(cudaFree(cd1));
    MUST(cudaFree(pd));
    MUST(cudaFree(uba));
    MUST(cudaFree(act));
    MUST(cudaFree(u));
    MUST(cudaFree(g));
    MUST(cudaFree(ub2));
    MUST(cudaFree(h2));
    MUST(cudaFree(x1));
    MUST(cudaFree(po));
    MUST(cudaFree(ubg));
    MUST(cudaFree(gated));
    MUST(cudaFree(vals));
    MUST(cudaFree(probs));
    MUST(cudaFree(dots));
    MUST(cudaFree(kn));
    MUST(cudaFree(qn));
    MUST(cudaFree(pv));
    MUST(cudaFree(pk));
    MUST(cudaFree(pq));
    MUST(cudaFree(hin));
    MUST(cudaFree(tmp));
    KRML_HOST_FREE(hw);
}

void KPlay_Functional_kplay_run(uint32_t nkv, __nv_bfloat16 *w,
    __nv_bfloat16 *x, __nv_bfloat16 *kv, __nv_bfloat16 *cs, __nv_bfloat16 *sn,
    int64_t t, __nv_bfloat16 *xo, __nv_bfloat16 *kvo)
{
    uint32_t c = nkv / 1024U;
    if (nkv % 1024U == 0U && 1U <= c && c <= 12U && 0LL <= t && t < 12LL) {
        uint32_t ts = KPlay_Functional_i64_to_sz(t);
        if (ts < c)
            KPlay_Functional_run_main(nkv, c, ts, w, x, kv, cs, sn, xo, kvo);
    }
}
