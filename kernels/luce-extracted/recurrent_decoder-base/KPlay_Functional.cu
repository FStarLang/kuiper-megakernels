
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting h_copy
*/
static void
__hoisted_h_copy_0(__nv_bfloat16 *x, __nv_bfloat16 *o)
{
    uint32_t i = 0U;
    while (i < 1024U) {
        uint32_t vi = i;
        o[vi] = x[vi];
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting h_norm
*/
static void
__hoisted_h_norm_0(__nv_bfloat16 *res, float *vals, float *part,
    __nv_bfloat16 *w, uint32_t woff, __nv_bfloat16 *inp)
{
    uint32_t i = 0U;
    while (i < 1024U) {
        uint32_t vi = i;
        float r = __bfloat162float(res[vi]);
        vals[vi] = r * r;
        i = vi + 1U;
    }
    uint32_t i1 = 0U;
    while (i1 < 256U) {
        uint32_t vi = i1;
        float acc = 0.0f;
        uint32_t j = 0U;
        while (j < 4U) {
            uint32_t vj = j;
            acc += vals[vi * 4U + vj];
            j = vj + 1U;
        }
        part[vi] = acc;
        i1 = vi + 1U;
    }
    uint32_t off = 128U;
    while (off > 0U) {
        uint32_t vo = off;
        uint32_t k = 0U;
        while (k < vo) {
            uint32_t vk = k;
            part[vk] += part[vk + vo];
            k = vk + 1U;
        }
        off = vo / 2U;
    }
    float inv = rsqrtf(part[0U] * 0.0009765625f +
                       0.0000009999999974752427078783512115478515625f);
    uint32_t i2 = 0U;
    while (i2 < 1024U) {
        uint32_t vi = i2;
        float r = __bfloat162float(res[vi]);
        inp[vi] =
            __float2bfloat16(r * inv * (1.0f + __bfloat162float(w[woff + vi])));
        i2 = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting h_bfill
*/
static void
__hoisted_h_bfill_0(uint32_t k, __nv_bfloat16 *inp, __nv_bfloat16 *bb)
{
    uint32_t n = 16U * k;
    uint32_t i = 0U;
    while (i < n) {
        uint32_t vi = i;
        bb[vi] = inp[vi % k];
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting h_beta
*/
static void
__hoisted_h_beta_0(__nv_bfloat16 *prj, __nv_bfloat16 *beta)
{
    uint32_t i = 0U;
    while (i < 16U) {
        uint32_t vi = i;
        beta[vi] =
            __float2bfloat16(1.0f / (1.0f + expf(-__bfloat162float(prj[vi]))));
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting h_conv
*/
static void
__hoisted_h_conv_0(__nv_bfloat16 *conv, __nv_bfloat16 *out1, __nv_bfloat16 *w,
    __nv_bfloat16 *qkv)
{
    uint32_t i = 0U;
    while (i < 6144U) {
        uint32_t vi = i;
        uint32_t b4 = vi * 4U;
        uint32_t wb = (uint32_t) 19433472U + b4;
        __nv_bfloat16 x0 = conv[b4 + 1U];
        out1[b4] = x0;
        __nv_bfloat16 __anf0 = w[wb];
        float v0 =
            __fmaf_rn(__bfloat162float(x0), __bfloat162float(__anf0), 0.0f);
        __nv_bfloat16 x1 = conv[b4 + 2U];
        out1[b4 + 1U] = x1;
        __nv_bfloat16 __anf01 = w[wb + 1U];
        float v1 =
            __fmaf_rn(__bfloat162float(x1), __bfloat162float(__anf01), v0);
        __nv_bfloat16 x2 = conv[b4 + 3U];
        out1[b4 + 2U] = x2;
        __nv_bfloat16 __anf02 = w[wb + 2U];
        float v2 =
            __fmaf_rn(__bfloat162float(x2), __bfloat162float(__anf02), v1);
        __nv_bfloat16 x3 = qkv[vi];
        out1[b4 + 3U] = x3;
        __nv_bfloat16 __anf03 = w[wb + 3U];
        float v3 =
            __fmaf_rn(__bfloat162float(x3), __bfloat162float(__anf03), v2);
        qkv[vi] = __float2bfloat16(
            __bfloat162float(__float2bfloat16(v3)) /
            (1.0f + expf(-__bfloat162float(__float2bfloat16(v3)))));
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting h_qknorm
*/
static void
__hoisted_h_qknorm_0(__nv_bfloat16 *qkv, float *vals, float *part)
{
    uint32_t hp = 0U;
    while (hp < 32U) {
        uint32_t vh = hp;
        uint32_t start = vh % 2U * 2048U + vh / 2U * 128U;
        uint32_t i = 0U;
        while (i < 128U) {
            uint32_t vi = i;
            float q = __bfloat162float(qkv[start + vi]);
            vals[vi] = __bfloat162float(__float2bfloat16(q * q));
            i = vi + 1U;
        }
        uint32_t i1 = 0U;
        while (i1 < 16U) {
            uint32_t vi = i1;
            float acc = 0.0f;
            uint32_t j = 0U;
            while (j < 8U) {
                uint32_t vj = j;
                acc += vals[vi * 8U + vj];
                j = vj + 1U;
            }
            part[vi] = acc;
            i1 = vi + 1U;
        }
        uint32_t off = 8U;
        while (off > 0U) {
            uint32_t vo = off;
            uint32_t k = 0U;
            while (k < vo) {
                uint32_t vk = k;
                part[vk] += part[vk + vo];
                k = vk + 1U;
            }
            off = vo / 2U;
        }
        float inv = __bfloat162float(
            __float2bfloat16(rsqrtf(__bfloat162float(__float2bfloat16(
                __bfloat162float(__float2bfloat16(part[0U])) +
                0.0000009999999974752427078783512115478515625f)))));
        uint32_t i2 = 0U;
        while (i2 < 128U) {
            uint32_t vi = i2;
            qkv[start + vi] =
                __float2bfloat16(__bfloat162float(qkv[start + vi]) * inv);
            i2 = vi + 1U;
        }
        hp = vh + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting h_rec
*/
static void
__hoisted_h_rec_0(__nv_bfloat16 *prj, __nv_bfloat16 *w, __nv_bfloat16 *beta,
    float *st, float *upd, __nv_bfloat16 *qkv, float *prod, float *out2,
    __nv_bfloat16 *core)
{
    uint32_t h = 0U;
    while (h < 16U) {
        uint32_t vh = h;
        __nv_bfloat16 __anf1 = prj[vh];
        __nv_bfloat16 __anf0 = w[(uint32_t) 19458064U + vh];
        float a = __bfloat162float(__anf1) + __bfloat162float(__anf0);
        float spv;
        if (20.0f < a)
            spv = a;
        else
            spv = log1pf(expf(a));
        float decay =
            expf(-expf(__bfloat162float(w[(uint32_t) 19458048U + vh])) * spv);
        float bh = __bfloat162float(beta[vh]);
        uint32_t sbase = vh * 16384U;
        uint32_t hb = vh * 128U;
        uint32_t j = 0U;
        while (j < 128U) {
            uint32_t vj = j;
            uint32_t i = 0U;
            while (i < 128U) {
                uint32_t vi = i;
                float u = st[sbase + vi * 128U + vj] * decay;
                upd[vi] = u;
                prod[vi] = u * __bfloat162float(qkv[2048U + hb + vi]);
                i = vi + 1U;
            }
            float pa = 0.0f;
            float pb = 0.0f;
            float pc = 0.0f;
            float pd = 0.0f;
            uint32_t y = 0U;
            while (y < 4U) {
                uint32_t vy = y;
                float a0 = 0.0f;
                float a1 = 0.0f;
                float a2 = 0.0f;
                float a3 = 0.0f;
                uint32_t t = 0U;
                while (t < 8U) {
                    uint32_t vt = t;
                    uint32_t i1 = vy + 16U * vt;
                    float e1 = prod[i1 + 4U];
                    float e2 = prod[i1 + 8U];
                    float e3 = prod[i1 + 12U];
                    a0 += prod[i1];
                    a1 += e1;
                    a2 += e2;
                    a3 += e3;
                    t = vt + 1U;
                }
                float s = a0 + a1 + a2 + a3;
                if (vy == 0U)
                    pa = s;
                else if (vy == 1U)
                    pb = s;
                else if (vy == 2U)
                    pc = s;
                else
                    pd = s;
                y = vy + 1U;
            }
            float m = pa + pc + (pb + pd);
            float delta = (__bfloat162float(qkv[4096U + hb + vj]) - m) * bh;
            uint32_t i1 = 0U;
            while (i1 < 128U) {
                uint32_t vi = i1;
                float u = upd[vi];
                float nx = u + __bfloat162float(qkv[2048U + hb + vi]) * delta;
                float qv = __bfloat162float(qkv[hb + vi]) *
                           0.088388346135616302490234375f;
                out2[sbase + vi * 128U + vj] = nx;
                prod[vi] = nx * qv;
                i1 = vi + 1U;
            }
            float pa1 = 0.0f;
            float pb1 = 0.0f;
            float pc1 = 0.0f;
            float pd1 = 0.0f;
            uint32_t y1 = 0U;
            while (y1 < 4U) {
                uint32_t vy = y1;
                float a0 = 0.0f;
                float a11 = 0.0f;
                float a2 = 0.0f;
                float a3 = 0.0f;
                uint32_t t = 0U;
                while (t < 8U) {
                    uint32_t vt = t;
                    uint32_t i2 = vy + 16U * vt;
                    float e1 = prod[i2 + 4U];
                    float e2 = prod[i2 + 8U];
                    float e3 = prod[i2 + 12U];
                    a0 += prod[i2];
                    a11 += e1;
                    a2 += e2;
                    a3 += e3;
                    t = vt + 1U;
                }
                float s = a0 + a11 + a2 + a3;
                if (vy == 0U)
                    pa1 = s;
                else if (vy == 1U)
                    pb1 = s;
                else if (vy == 2U)
                    pc1 = s;
                else
                    pd1 = s;
                y1 = vy + 1U;
            }
            core[hb + vj] = __float2bfloat16(pa1 + pc1 + (pb1 + pd1));
            j = vj + 1U;
        }
        h = vh + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting h_gnorm
*/
static void
__hoisted_h_gnorm_0(__nv_bfloat16 *core, float *vals, float *part,
    __nv_bfloat16 *w, __nv_bfloat16 *z, __nv_bfloat16 *inp)
{
    uint32_t h = 0U;
    while (h < 16U) {
        uint32_t vh = h;
        uint32_t hb = vh * 128U;
        uint32_t i = 0U;
        while (i < 128U) {
            uint32_t vi = i;
            float c = __bfloat162float(core[hb + vi]);
            vals[vi] = c * c;
            i = vi + 1U;
        }
        uint32_t i1 = 0U;
        while (i1 < 32U) {
            uint32_t vi = i1;
            float acc = 0.0f;
            uint32_t j = 0U;
            while (j < 4U) {
                uint32_t vj = j;
                acc += vals[vi * 4U + vj];
                j = vj + 1U;
            }
            part[vi] = acc;
            i1 = vi + 1U;
        }
        uint32_t off = 16U;
        while (off > 0U) {
            uint32_t vo = off;
            uint32_t k = 0U;
            while (k < vo) {
                uint32_t vk = k;
                part[vk] += part[vk + vo];
                k = vk + 1U;
            }
            off = vo / 2U;
        }
        float inv = rsqrtf(part[0U] * 0.0078125f +
                           0.0000009999999974752427078783512115478515625f);
        uint32_t i2 = 0U;
        while (i2 < 128U) {
            uint32_t vi = i2;
            __nv_bfloat16 unit =
                __float2bfloat16(__bfloat162float(core[hb + vi]) * inv);
            __nv_bfloat16 __anf01 = w[(uint32_t) 19458080U + vi];
            __nv_bfloat16 scaled = __float2bfloat16(
                __bfloat162float(__anf01) * __bfloat162float(unit));
            __nv_bfloat16 __anf02 = z[hb + vi];
            inp[hb + vi] = __float2bfloat16(
                __bfloat162float(scaled) *
                (__bfloat162float(__anf02) /
                    (1.0f + expf(-__bfloat162float(__anf02)))));
            i2 = vi + 1U;
        }
        h = vh + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting h_resadd
*/
static void
__hoisted_h_resadd_0(__nv_bfloat16 *res, __nv_bfloat16 *prj)
{
    uint32_t i = 0U;
    while (i < 1024U) {
        uint32_t vi = i;
        float r = __bfloat162float(res[vi]);
        res[vi] = __float2bfloat16(r + __bfloat162float(prj[vi]));
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting h_silumul
*/
static void
__hoisted_h_silumul_0(
    __nv_bfloat16 *gate, __nv_bfloat16 *prj, __nv_bfloat16 *inp)
{
    uint32_t i = 0U;
    while (i < 3584U) {
        uint32_t vi = i;
        __nv_bfloat16 __anf0 = gate[vi];
        float g = __bfloat162float(
            __float2bfloat16(__bfloat162float(__anf0) /
                             (1.0f + expf(-__bfloat162float(__anf0)))));
        inp[vi] = __float2bfloat16(g * __bfloat162float(prj[vi]));
        i = vi + 1U;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting proj
*/
static void
__hoisted_proj_0(uint32_t nt, uint32_t nch, uint32_t chunk, uint32_t kk,
    uint32_t woff, __nv_bfloat16 *w, __nv_bfloat16 *bb, float *cc)
{
    auto &fc = KPR_INIT(kpr_mma_fragment);
    uint32_t ng = nt * nch;
    uint32_t nj = chunk / 16U;
    uint32_t g = 0U;
    while (g < ng) {
        uint32_t vg = g;
        uint32_t boff = vg % nch * chunk;
        uint32_t aoff = woff + vg / nch * 16U * kk + boff;
        kpr_mma_fill(fc, 0.0f);
        uint32_t j = 0U;
        while (j < nj) {
            uint32_t vj = j;
            kpr_mma_mma_sync(
                w + (aoff + 16U * vj), kk, bb + (boff + 16U * vj), kk, fc);
            j = vj + 1U;
        }
        kpr_mma_store(fc, cc + vg * 128U, 8U);
        g = vg + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting proj
*/
static void
__hoisted_proj_1(uint32_t nt, uint32_t nch, float *cc, __nv_bfloat16 *dst)
{
    uint32_t m = nt * 16U;
    uint32_t r = 0U;
    while (r < m) {
        uint32_t vr = r;
        uint32_t i = vr % 16U;
        uint32_t g = vr / 16U * nch;
        float c0 = cc[g * 128U + i * 8U];
        float cv = c0;
        if (nch == 2U)
            cv = c0 + cc[(g + 1U) * 128U + i * 8U];
        dst[vr] = __float2bfloat16(cv);
        r = vr + 1U;
    }
}

void KPlay_Functional_h_copy(__nv_bfloat16 *x, __nv_bfloat16 *o)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_h_copy_0, 1U, 1U, 0U, s, x, o);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void KPlay_Functional_h_norm(__nv_bfloat16 *res, __nv_bfloat16 *w,
    uint32_t woff, __nv_bfloat16 *inp, float *vals, float *part)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_h_norm_0, 1U, 1U, 0U, s, res, vals, part, w, woff, inp);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void KPlay_Functional_h_bfill(__nv_bfloat16 *inp, __nv_bfloat16 *bb, uint32_t k)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_h_bfill_0, 1U, 1U, 0U, s, k, inp, bb);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void KPlay_Functional_h_beta(__nv_bfloat16 *prj, __nv_bfloat16 *beta)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_h_beta_0, 1U, 1U, 0U, s, prj, beta);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void KPlay_Functional_h_conv(__nv_bfloat16 *w, __nv_bfloat16 *conv,
    __nv_bfloat16 *out1, __nv_bfloat16 *qkv)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_h_conv_0, 1U, 1U, 0U, s, conv, out1, w, qkv);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void KPlay_Functional_h_qknorm(__nv_bfloat16 *qkv, float *vals, float *part)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_h_qknorm_0, 1U, 1U, 0U, s, qkv, vals, part);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void KPlay_Functional_h_rec(__nv_bfloat16 *w, __nv_bfloat16 *qkv,
    __nv_bfloat16 *prj, __nv_bfloat16 *beta, float *st, float *out2,
    __nv_bfloat16 *core, float *upd, float *prod)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_h_rec_0, 1U, 1U, 0U, s, prj, w, beta, st, upd, qkv,
        prod, out2, core);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void KPlay_Functional_h_gnorm(__nv_bfloat16 *w, __nv_bfloat16 *core,
    __nv_bfloat16 *z, __nv_bfloat16 *inp, float *vals, float *part)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_h_gnorm_0, 1U, 1U, 0U, s, core, vals, part, w, z, inp);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void KPlay_Functional_h_resadd(__nv_bfloat16 *res, __nv_bfloat16 *prj)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_h_resadd_0, 1U, 1U, 0U, s, res, prj);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void KPlay_Functional_h_silumul(
    __nv_bfloat16 *gate, __nv_bfloat16 *prj, __nv_bfloat16 *inp)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_h_silumul_0, 1U, 1U, 0U, s, gate, prj, inp);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void KPlay_Functional_proj(__nv_bfloat16 *w, __nv_bfloat16 *bb, float *cc,
    __nv_bfloat16 *dst, uint32_t woff, uint32_t kk, uint32_t nt, uint32_t nch,
    uint32_t chunk)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(
        __hoisted_proj_0, 1U, 32U, 0U, s, nt, nch, chunk, kk, woff, w, bb, cc);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_proj_1, 1U, 1U, 0U, s1, nt, nch, cc, dst);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

void KPlay_Functional_kplay_run(__nv_bfloat16 *w, __nv_bfloat16 *x,
    __nv_bfloat16 *conv, float *st, __nv_bfloat16 *out0, __nv_bfloat16 *out1,
    float *out2)
{
    __nv_bfloat16 *inp = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 3584U);
    __nv_bfloat16 *bb = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 57344U);
    __nv_bfloat16 *prj = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 6144U);
    __nv_bfloat16 *qkv = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 6144U);
    __nv_bfloat16 *z = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 2048U);
    __nv_bfloat16 *beta =
        (__nv_bfloat16 *) KPR_GPU_ALLOC((uint32_t) sizeof(__nv_bfloat16), 16U);
    __nv_bfloat16 *gate = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 3584U);
    __nv_bfloat16 *core = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 2048U);
    float *cc = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 49152U);
    float *vals = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 1024U);
    float *part = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 256U);
    float *upd = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 128U);
    float *prod = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 128U);
    KPlay_Functional_h_copy(x, out0);
    KPlay_Functional_h_norm(out0, w, 0U, inp, vals, part);
    KPlay_Functional_h_bfill(inp, bb, 1024U);
    KPlay_Functional_proj(
        w, bb, cc, qkv, (uint32_t) 11012096U, 1024U, 384U, 1U, 1024U);
    KPlay_Functional_proj(
        w, bb, cc, z, (uint32_t) 17303552U, 1024U, 128U, 1U, 1024U);
    KPlay_Functional_proj(
        w, bb, cc, prj, (uint32_t) 19400704U, 1024U, 1U, 1U, 1024U);
    KPlay_Functional_h_beta(prj, beta);
    KPlay_Functional_proj(
        w, bb, cc, prj, (uint32_t) 19417088U, 1024U, 1U, 1U, 1024U);
    KPlay_Functional_h_conv(w, conv, out1, qkv);
    KPlay_Functional_h_qknorm(qkv, vals, part);
    KPlay_Functional_h_rec(w, qkv, prj, beta, st, out2, core, upd, prod);
    KPlay_Functional_h_gnorm(w, core, z, inp, vals, part);
    KPlay_Functional_h_bfill(inp, bb, 2048U);
    KPlay_Functional_proj(
        w, bb, cc, prj, (uint32_t) 19458208U, 2048U, 64U, 1U, 2048U);
    KPlay_Functional_h_resadd(out0, prj);
    KPlay_Functional_h_norm(out0, w, 1024U, inp, vals, part);
    KPlay_Functional_h_bfill(inp, bb, 1024U);
    KPlay_Functional_proj(w, bb, cc, gate, 2048U, 1024U, 224U, 1U, 1024U);
    KPlay_Functional_proj(
        w, bb, cc, prj, (uint32_t) 3672064U, 1024U, 224U, 1U, 1024U);
    KPlay_Functional_h_silumul(gate, prj, inp);
    KPlay_Functional_h_bfill(inp, bb, 3584U);
    KPlay_Functional_proj(
        w, bb, cc, prj, (uint32_t) 7342080U, 3584U, 64U, 2U, 1792U);
    KPlay_Functional_h_resadd(out0, prj);
    MUST(cudaFree(inp));
    MUST(cudaFree(bb));
    MUST(cudaFree(prj));
    MUST(cudaFree(qkv));
    MUST(cudaFree(z));
    MUST(cudaFree(beta));
    MUST(cudaFree(gate));
    MUST(cudaFree(core));
    MUST(cudaFree(cc));
    MUST(cudaFree(vals));
    MUST(cudaFree(part));
    MUST(cudaFree(upd));
    MUST(cudaFree(prod));
}
