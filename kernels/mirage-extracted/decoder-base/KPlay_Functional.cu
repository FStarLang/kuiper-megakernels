
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting h_wcopy
*/
static void
__hoisted_h_wcopy_0(uint32_t kt, uint32_t rt, uint32_t off, __nv_bfloat16 *src,
    __nv_bfloat16 *dst)
{
    uint32_t kk = 16U * kt;
    uint32_t rr = 16U * rt;
    uint32_t i = 0U;
    uint32_t base = off;
    while (i < rr) {
        uint32_t iv = i;
        uint32_t bv = base;
        uint32_t k = 0U;
        while (k < kk) {
            uint32_t kv = k;
            dst[iv * (16U * kt) + kv] = src[bv + kv];
            k = kv + 1U;
        }
        base = bv + kk;
        i = iv + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting h_pad
*/
static void
__hoisted_h_pad_0(uint32_t kt, __nv_bfloat16 *src, __nv_bfloat16 *dst)
{
    uint32_t kk = 16U * kt;
    uint32_t t = 0U;
    uint32_t base = 0U;
    while (t < 7U) {
        uint32_t tv = t;
        uint32_t bv = base;
        uint32_t k = 0U;
        while (k < kk) {
            uint32_t kv = k;
            dst[tv * (16U * kt) + kv] = src[bv + kv];
            k = kv + 1U;
        }
        base = bv + kk;
        t = tv + 1U;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting h_mma
*/
static void
__hoisted_h_mma_0(uint32_t rt, uint32_t k0, uint32_t k1, __nv_bfloat16 *w,
    __nv_bfloat16 *x, uint32_t kt, float *c)
{
    auto &fr = KPR_INIT(kpr_mma_fragment);
    uint32_t i = 0U;
    while (i < rt) {
        uint32_t tr = i;
        kpr_mma_fill(fr, 0.0f);
        uint32_t k = k0;
        while (k < k1) {
            uint32_t kv = k;
            kpr_mma_mma_sync(w + (16U * kt * (tr * 16U) + kv * 16U), 16U * kt,
                x + kv * 16U, 16U * kt, fr);
            k = kv + 1U;
        }
        kpr_mma_store(fr, c + 8U * (tr * 16U), 8U);
        i = tr + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting h_proj
*/
static void
__hoisted_h_proj_0(uint32_t rt, float *c, __nv_bfloat16 *out)
{
    uint32_t rr = 16U * rt;
    uint32_t t = 0U;
    uint32_t base = 0U;
    while (t < 7U) {
        uint32_t tv = t;
        uint32_t bv = base;
        uint32_t r = 0U;
        while (r < rr) {
            uint32_t rv = r;
            out[bv + rv] = __float2bfloat16(c[rv * 8U + tv]);
            r = rv + 1U;
        }
        base = bv + rr;
        t = tv + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting h_proj_split
*/
static void
__hoisted_h_proj_split_0(uint32_t rt, float *c, float *c_, __nv_bfloat16 *out)
{
    uint32_t rr = 16U * rt;
    uint32_t t = 0U;
    uint32_t base = 0U;
    while (t < 7U) {
        uint32_t tv = t;
        uint32_t bv = base;
        uint32_t r = 0U;
        while (r < rr) {
            uint32_t rv = r;
            out[bv + rv] = __float2bfloat16(c[rv * 8U + tv] + c_[rv * 8U + tv]);
            r = rv + 1U;
        }
        base = bv + rr;
        t = tv + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting h_silu
*/
static void
__hoisted_h_silu_0(__nv_bfloat16 *gu, __nv_bfloat16 *out)
{
    uint32_t row = 0U;
    uint32_t gbase = 0U;
    uint32_t obase = 0U;
    while (row < 7U) {
        uint32_t rv = row;
        uint32_t gb = gbase;
        uint32_t ob = obase;
        uint32_t col = 0U;
        while (col < 3072U) {
            uint32_t cv = col;
            uint32_t gi = gb + cv;
            float gate = __bfloat162float(gu[gi]);
            __nv_bfloat16 act = __float2bfloat16(gate / (1.0f + expf(-gate)));
            __nv_bfloat16 u = gu[gi + 3072U];
            out[ob + cv] =
                __float2bfloat16(__bfloat162float(act) * __bfloat162float(u));
            col = cv + 1U;
        }
        gbase = gb + 6144U;
        obase = ob + 3072U;
        row = rv + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting h_attqk
*/
static void
__hoisted_h_attqk_0(uint32_t off, __nv_bfloat16 *qkv, float *partial,
    __nv_bfloat16 *w, __nv_bfloat16 *nrm, __nv_bfloat16 *rot, __nv_bfloat16 *qk)
{
    uint32_t row = 0U;
    while (row < 7U) {
        uint32_t r = row;
        uint32_t head = 0U;
        while (head < 24U) {
            uint32_t h = head;
            uint32_t base = r * 4096U + h * 128U;
            uint32_t noff = off + (uint32_t) 6293504U + (h < 16U ? 0U : 128U);
            uint32_t lane = 0U;
            while (lane < 32U) {
                uint32_t lv = lane;
                float s0 = 0.0f;
                float s1 = 0.0f;
                float s2 = 0.0f;
                float s3 = 0.0f;
                uint32_t start = lv * 4U;
                while (start < 128U) {
                    uint32_t sv = start;
                    uint32_t b = base + sv;
                    float v0 = __bfloat162float(qkv[b]);
                    s0 += v0 * v0;
                    float v1 = __bfloat162float(qkv[b + 1U]);
                    s1 += v1 * v1;
                    float v2 = __bfloat162float(qkv[b + 2U]);
                    s2 += v2 * v2;
                    float v3 = __bfloat162float(qkv[b + 3U]);
                    s3 += v3 * v3;
                    start = sv + 128U;
                }
                partial[lv] = s0 + s1 + s2 + s3;
                lane = lv + 1U;
            }
            uint32_t off1 = 16U;
            while (off1 > 0U) {
                uint32_t ov = off1;
                uint32_t l2 = 0U;
                while (l2 < ov) {
                    uint32_t lv = l2;
                    partial[lv] += partial[lv + ov];
                    l2 = lv + 1U;
                }
                off1 = ov / 2U;
            }
            int64_t szc = (int64_t) (uint32_t) 128U;
            int64_t epc = 1000000LL;
            float inv =
                rsqrtf(partial[0U] * (1.0f / (float) szc) + 1.0f / (float) epc);
            uint32_t col = 0U;
            while (col < 128U) {
                uint32_t c = col;
                __nv_bfloat16 sc =
                    __float2bfloat16(__bfloat162float(qkv[base + c]) * inv);
                float wv = __bfloat162float(w[noff + c]);
                nrm[c] = __float2bfloat16(__bfloat162float(sc) * wv);
                col = c + 1U;
            }
            uint32_t qbase = (r * 24U + h) * 128U;
            col = 0U;
            while (col < 128U) {
                uint32_t c = col;
                float xn = __bfloat162float(nrm[c]);
                float other;
                if (c < 64U)
                    other = -__bfloat162float(nrm[c + 64U]);
                else
                    other = __bfloat162float(nrm[c - 64U]);
                __nv_bfloat16 sn = rot[896U + r * 128U + c];
                __nv_bfloat16 a =
                    __float2bfloat16(xn * __bfloat162float(rot[r * 128U + c]));
                __nv_bfloat16 b =
                    __float2bfloat16(other * __bfloat162float(sn));
                qk[qbase + c] =
                    __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
                col = c + 1U;
            }
            head = h + 1U;
        }
        row = r + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting h_attsv
*/
static void
__hoisted_h_attsv_0(
    __nv_bfloat16 *qk, float *sc, __nv_bfloat16 *qkv, __nv_bfloat16 *out)
{
    float scale = 1.0f / sqrtf(128.0f);
    float buf = 1073741824.0f;
    float p30 = buf;
    float mask = -(255.0f * (p30 * p30 * (p30 * p30)));
    uint32_t head = 0U;
    while (head < 16U) {
        uint32_t h = head;
        uint32_t row = 0U;
        while (row < 7U) {
            uint32_t t = row;
            uint32_t key = 0U;
            while (key < 7U) {
                uint32_t k = key;
                uint32_t qb = (t * 24U + h) * 128U;
                uint32_t kb = (k * 24U + 16U + h / 2U) * 128U;
                float sum = 0.0f;
                uint32_t lane = 0U;
                while (lane < 16U) {
                    uint32_t l = lane;
                    float partial = 0.0f;
                    uint32_t col = l;
                    while (col < 128U) {
                        uint32_t c = col;
                        __nv_bfloat16 x = qk[qb + c];
                        __nv_bfloat16 y = qk[kb + c];
                        float pv = partial;
                        partial = __fmaf_rn(
                            __bfloat162float(x), __bfloat162float(y), pv);
                        col = c + 16U;
                    }
                    sum += partial;
                    lane = l + 1U;
                }
                __nv_bfloat16 scaled = __float2bfloat16(
                    __bfloat162float(__float2bfloat16(sum)) * scale);
                float ms = t < k ? mask : 0.0f;
                sc[k] = __bfloat162float(
                    __float2bfloat16(__bfloat162float(scaled) + ms));
                key = k + 1U;
            }
            float s0 = sc[0U];
            float s1 = sc[1U];
            float s2 = sc[2U];
            float s3 = sc[3U];
            float s4 = sc[4U];
            float s5 = sc[5U];
            float s6 = sc[6U];
            float m =
                ((s0 < s4 ? s4 : s0) < (s2 < s6 ? s6 : s2) ? s2 < s6 ? s6 : s2
                    : s0 < s4                              ? s4
                              : s0) < ((s1 < s5 ? s5 : s1) < s3 ? s3
                                          : s1 < s5             ? s5
                                                                : s1)
                    ? (s1 < s5 ? s5 : s1) < s3 ? s3
                      : s1 < s5                ? s5
                                               : s1
                : (s0 < s4 ? s4 : s0) < (s2 < s6 ? s6 : s2) ? s2 < s6 ? s6 : s2
                : s0 < s4                                   ? s4
                                                            : s0;
            float e0 = expf(s0 - m);
            float e1 = expf(s1 - m);
            float e2 = expf(s2 - m);
            float e3 = expf(s3 - m);
            float e4 = expf(s4 - m);
            float e5 = expf(s5 - m);
            float e6 = expf(s6 - m);
            float tot = e0 + e4 + (e2 + e6) + (e1 + e5 + e3);
            __nv_bfloat16 p0 = __float2bfloat16(e0 / tot);
            __nv_bfloat16 p1 = __float2bfloat16(e1 / tot);
            __nv_bfloat16 p2 = __float2bfloat16(e2 / tot);
            __nv_bfloat16 p3 = __float2bfloat16(e3 / tot);
            __nv_bfloat16 p4 = __float2bfloat16(e4 / tot);
            __nv_bfloat16 p5 = __float2bfloat16(e5 / tot);
            __nv_bfloat16 p6 = __float2bfloat16(e6 / tot);
            uint32_t hh = h / 2U;
            uint32_t col = 0U;
            while (col < 128U) {
                uint32_t c = col;
                uint32_t vb = 3072U + hh * 128U + c;
                __nv_bfloat16 v0 = qkv[vb];
                __nv_bfloat16 v1 = qkv[4096U + vb];
                __nv_bfloat16 v2 = qkv[8192U + vb];
                __nv_bfloat16 v3 = qkv[12288U + vb];
                __nv_bfloat16 v4 = qkv[16384U + vb];
                __nv_bfloat16 v5 = qkv[20480U + vb];
                __nv_bfloat16 v6 = qkv[24576U + vb];
                float a0 =
                    __fmaf_rn(__bfloat162float(p0), __bfloat162float(v0), 0.0f);
                float a1 =
                    __fmaf_rn(__bfloat162float(p1), __bfloat162float(v1), a0);
                float a2 =
                    __fmaf_rn(__bfloat162float(p2), __bfloat162float(v2), a1);
                float a3 =
                    __fmaf_rn(__bfloat162float(p3), __bfloat162float(v3), a2);
                float b0 =
                    __fmaf_rn(__bfloat162float(p4), __bfloat162float(v4), 0.0f);
                float b1 =
                    __fmaf_rn(__bfloat162float(p5), __bfloat162float(v5), b0);
                out[t * 2048U + h * 128U + c] =
                    __float2bfloat16(a3 + __fmaf_rn(__bfloat162float(p6),
                                              __bfloat162float(v6), b1));
                col = c + 1U;
            }
            row = t + 1U;
        }
        head = h + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting h_resnorm
*/
static void
__hoisted_h_resnorm_0(__nv_bfloat16 *res, __nv_bfloat16 *ua,
    __nv_bfloat16 *summed, float *partial, uint32_t noff, __nv_bfloat16 *w,
    __nv_bfloat16 *normd)
{
    uint32_t row = 0U;
    while (row < 7U) {
        uint32_t r = row;
        uint32_t base = r * 1024U;
        uint32_t col = 0U;
        while (col < 1024U) {
            uint32_t c = col;
            float x = __bfloat162float(res[base + c]);
            summed[base + c] =
                __float2bfloat16(x + __bfloat162float(ua[base + c]));
            col = c + 1U;
        }
        uint32_t lane = 0U;
        while (lane < 128U) {
            uint32_t lv = lane;
            float s0 = 0.0f;
            float s1 = 0.0f;
            float s2 = 0.0f;
            float s3 = 0.0f;
            uint32_t start = lv * 4U;
            while (start < 1024U) {
                uint32_t sv = start;
                uint32_t b = base + sv;
                float v0 = __bfloat162float(summed[b]);
                s0 += v0 * v0;
                float v1 = __bfloat162float(summed[b + 1U]);
                s1 += v1 * v1;
                float v2 = __bfloat162float(summed[b + 2U]);
                s2 += v2 * v2;
                float v3 = __bfloat162float(summed[b + 3U]);
                s3 += v3 * v3;
                start = sv + 512U;
            }
            partial[lv] = s0 + s1 + s2 + s3;
            lane = lv + 1U;
        }
        uint32_t off = 64U;
        while (off > 0U) {
            uint32_t ov = off;
            uint32_t l2 = 0U;
            while (l2 < ov) {
                uint32_t lv = l2;
                partial[lv] += partial[lv + ov];
                l2 = lv + 1U;
            }
            off = ov / 2U;
        }
        int64_t szc = (int64_t) (uint32_t) 1024U;
        int64_t epc = 1000000LL;
        float inv =
            rsqrtf(partial[0U] * (1.0f / (float) szc) + 1.0f / (float) epc);
        col = 0U;
        while (col < 1024U) {
            uint32_t c = col;
            __nv_bfloat16 sc =
                __float2bfloat16(__bfloat162float(summed[base + c]) * inv);
            float wv = __bfloat162float(w[noff + c]);
            normd[base + c] = __float2bfloat16(__bfloat162float(sc) * wv);
            col = c + 1U;
        }
        row = r + 1U;
    }
}

void KPlay_Functional_h_wcopy(uint32_t rt, uint32_t kt, __nv_bfloat16 *src,
    uint32_t off, __nv_bfloat16 *dst)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_h_wcopy_0, 1U, 1U, 0U, s, kt, rt, off, src, dst);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void KPlay_Functional_h_pad(uint32_t kt, __nv_bfloat16 *src, __nv_bfloat16 *dst)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_h_pad_0, 1U, 1U, 0U, s, kt, src, dst);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void KPlay_Functional_h_mma(uint32_t rt, uint32_t kt, uint32_t k0, uint32_t k1,
    __nv_bfloat16 *w, __nv_bfloat16 *x, float *c)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_h_mma_0, 1U, 32U, 0U, s, rt, k0, k1, w, x, kt, c);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void KPlay_Functional_h_proj(uint32_t rt, uint32_t kt, __nv_bfloat16 *w,
    __nv_bfloat16 *x, float *c, __nv_bfloat16 *out)
{
    KPlay_Functional_h_mma(rt, kt, 0U, kt, w, x, c);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_h_proj_0, 1U, 1U, 0U, s, rt, c, out);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void KPlay_Functional_h_proj_split(uint32_t rt, uint32_t kt, uint32_t kh,
    __nv_bfloat16 *w, __nv_bfloat16 *x, float *c, float *c_, __nv_bfloat16 *out)
{
    KPlay_Functional_h_mma(rt, kt, 0U, kh, w, x, c);
    KPlay_Functional_h_mma(rt, kt, kh, kt, w, x, c_);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_h_proj_split_0, 1U, 1U, 0U, s, rt, c, c_, out);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void KPlay_Functional_h_silu(__nv_bfloat16 *gu, __nv_bfloat16 *out)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_h_silu_0, 1U, 1U, 0U, s, gu, out);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void KPlay_Functional_h_attqk(__nv_bfloat16 *w, uint32_t off,
    __nv_bfloat16 *qkv, __nv_bfloat16 *rot, __nv_bfloat16 *qk,
    __nv_bfloat16 *nrm, float *partial)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(
        __hoisted_h_attqk_0, 1U, 1U, 0U, s, off, qkv, partial, w, nrm, rot, qk);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void KPlay_Functional_h_attsv(
    __nv_bfloat16 *qk, __nv_bfloat16 *qkv, __nv_bfloat16 *out, float *sc)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_h_attsv_0, 1U, 1U, 0U, s, qk, sc, qkv, out);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void KPlay_Functional_h_resnorm(__nv_bfloat16 *w, uint32_t noff,
    __nv_bfloat16 *res, __nv_bfloat16 *ua, __nv_bfloat16 *summed,
    __nv_bfloat16 *normd, float *partial)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_h_resnorm_0, 1U, 1U, 0U, s, res, ua, summed, partial,
        noff, w, normd);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void KPlay_Functional_kplay_run(uint32_t n0, __nv_bfloat16 *input0,
    __nv_bfloat16 *input1, __nv_bfloat16 *input2, __nv_bfloat16 *input3,
    int64_t input4, int64_t input5, __nv_bfloat16 *output0,
    __nv_bfloat16 *output1)
{
    if (input4 >= 0LL && input4 <= 2131752703LL && input5 >= 0LL &&
        input5 <= 2147482623LL) {
        uint32_t off = (uint32_t) input4;
        uint32_t a5 = (uint32_t) input5;
        if ((uint32_t) 15730944U <= n0 && off <= n0 - (uint32_t) 15730944U &&
            1024U <= n0 && a5 <= n0 - 1024U) {
            __nv_bfloat16 *w1 = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                (uint32_t) sizeof(__nv_bfloat16), (uint32_t) 4194304U);
            KPlay_Functional_h_wcopy(256U, 64U, input0, off + 2048U, w1);
            __nv_bfloat16 *w2 = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                (uint32_t) sizeof(__nv_bfloat16), (uint32_t) 2097152U);
            KPlay_Functional_h_wcopy(
                64U, 128U, input0, off + (uint32_t) 4196352U, w2);
            __nv_bfloat16 *w3 = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                (uint32_t) sizeof(__nv_bfloat16), (uint32_t) 6291456U);
            KPlay_Functional_h_wcopy(
                384U, 64U, input0, off + (uint32_t) 6293760U, w3);
            __nv_bfloat16 *w4 = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                (uint32_t) sizeof(__nv_bfloat16), (uint32_t) 3145728U);
            KPlay_Functional_h_wcopy(
                64U, 192U, input0, off + (uint32_t) 12585216U, w4);
            __nv_bfloat16 *xa = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                (uint32_t) sizeof(__nv_bfloat16), 8192U);
            KPlay_Functional_h_pad(64U, input2, xa);
            float *ca =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 32768U);
            __nv_bfloat16 *qkv = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                (uint32_t) sizeof(__nv_bfloat16), 28672U);
            KPlay_Functional_h_proj(256U, 64U, w1, xa, ca, qkv);
            __nv_bfloat16 *qk = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                (uint32_t) sizeof(__nv_bfloat16), 21504U);
            __nv_bfloat16 *nrm = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                (uint32_t) sizeof(__nv_bfloat16), 128U);
            float *part =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 128U);
            KPlay_Functional_h_attqk(input0, off, qkv, input3, qk, nrm, part);
            __nv_bfloat16 *att = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                (uint32_t) sizeof(__nv_bfloat16), 14336U);
            float *sc = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 8U);
            KPlay_Functional_h_attsv(qk, qkv, att, sc);
            __nv_bfloat16 *xb = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                (uint32_t) sizeof(__nv_bfloat16), 16384U);
            KPlay_Functional_h_pad(128U, att, xb);
            float *cb =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 8192U);
            __nv_bfloat16 *proj = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                (uint32_t) sizeof(__nv_bfloat16), 7168U);
            KPlay_Functional_h_proj(64U, 128U, w2, xb, cb, proj);
            __nv_bfloat16 *res = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                (uint32_t) sizeof(__nv_bfloat16), 7168U);
            __nv_bfloat16 *norm = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                (uint32_t) sizeof(__nv_bfloat16), 7168U);
            KPlay_Functional_h_resnorm(
                input0, off + 1024U, input1, proj, res, norm, part);
            __nv_bfloat16 *xc = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                (uint32_t) sizeof(__nv_bfloat16), 8192U);
            KPlay_Functional_h_pad(64U, norm, xc);
            float *cc =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 49152U);
            __nv_bfloat16 *gu = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                (uint32_t) sizeof(__nv_bfloat16), 43008U);
            KPlay_Functional_h_proj(384U, 64U, w3, xc, cc, gu);
            __nv_bfloat16 *act = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                (uint32_t) sizeof(__nv_bfloat16), 21504U);
            KPlay_Functional_h_silu(gu, act);
            __nv_bfloat16 *xd = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                (uint32_t) sizeof(__nv_bfloat16), 24576U);
            KPlay_Functional_h_pad(192U, act, xd);
            float *cd =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 8192U);
            float *cd2 =
                (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 8192U);
            __nv_bfloat16 *proj2 = (__nv_bfloat16 *) KPR_GPU_ALLOC(
                (uint32_t) sizeof(__nv_bfloat16), 7168U);
            KPlay_Functional_h_proj_split(
                64U, 192U, 96U, w4, xd, cd, cd2, proj2);
            KPlay_Functional_h_resnorm(
                input0, a5, res, proj2, output0, output1, part);
            MUST(cudaFree(w1));
            MUST(cudaFree(w2));
            MUST(cudaFree(w3));
            MUST(cudaFree(w4));
            MUST(cudaFree(xa));
            MUST(cudaFree(xb));
            MUST(cudaFree(xc));
            MUST(cudaFree(xd));
            MUST(cudaFree(ca));
            MUST(cudaFree(cb));
            MUST(cudaFree(cc));
            MUST(cudaFree(cd));
            MUST(cudaFree(cd2));
            MUST(cudaFree(qkv));
            MUST(cudaFree(qk));
            MUST(cudaFree(nrm));
            MUST(cudaFree(part));
            MUST(cudaFree(att));
            MUST(cudaFree(sc));
            MUST(cudaFree(proj));
            MUST(cudaFree(res));
            MUST(cudaFree(norm));
            MUST(cudaFree(gu));
            MUST(cudaFree(act));
            MUST(cudaFree(proj2));
        }
    }
}
