
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(__nv_bfloat16 *residual, __nv_bfloat16 *hidden,
    float *part, __nv_bfloat16 *normw, __nv_bfloat16 *bsc)
{
    uint32_t i = 0U;
    while (i < 1024U) {
        uint32_t vi = i;
        hidden[vi] = residual[vi];
        i = vi + 1U;
    }
    uint32_t c = 0U;
    while (c < 256U) {
        uint32_t vc = c;
        uint32_t b = 4U * vc;
        float x0 = __bfloat162float(residual[b]);
        float x1 = __bfloat162float(residual[b + 1U]);
        float x2 = __bfloat162float(residual[b + 2U]);
        float x3 = __bfloat162float(residual[b + 3U]);
        part[vc] = 0.0f + x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
        c = vc + 1U;
    }
    uint32_t off = 128U;
    while (0U < off) {
        uint32_t vo = off;
        uint32_t j = 0U;
        while (j < vo) {
            uint32_t vj = j;
            part[vj] += part[vj + vo];
            j = vj + 1U;
        }
        off = vo / 2U;
    }
    float inv = rsqrtf(part[0U] * 0.0009765625f +
                       0.0000009999999974752427078783512115478515625f);
    uint32_t k = 0U;
    while (k < 1024U) {
        uint32_t vk = k;
        float r = __bfloat162float(residual[vk]);
        __nv_bfloat16 nv =
            __float2bfloat16(r * inv * (1.0f + __bfloat162float(normw[vk])));
        uint32_t n = 0U;
        while (n < 8U) {
            uint32_t vn = n;
            bsc[vn * 1024U + vk] = nv;
            n = vn + 1U;
        }
        k = vk + 1U;
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_1(__nv_bfloat16 *table, __nv_bfloat16 *bsc, float *csc)
{
    auto &fc = KPR_INIT(kpr_mma_fragment);
    uint32_t rb = 0U;
    while (rb < 15520U) {
        uint32_t vrb = rb;
        kpr_mma_fill(fc, 0.0f);
        uint32_t p = 0U;
        while (p < 64U) {
            uint32_t vp = p;
            kpr_mma_mma_sync(table + (1024U * (vrb * 16U) + vp * 16U), 1024U,
                bsc + vp * 16U, 1024U, fc);
            p = vp + 1U;
        }
        kpr_mma_store(fc, csc + 8U * (vrb * 16U), 8U);
        rb = vrb + 1U;
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_2(float *csc, __nv_bfloat16 *logits)
{
    uint32_t m = 0U;
    while (m < (uint32_t) 248320U) {
        uint32_t vm = m;
        logits[vm] = __float2bfloat16(csc[vm * 8U]);
        m = vm + 1U;
    }
}

void KPlay_Functional_kplay_run(__nv_bfloat16 *table, __nv_bfloat16 *normw,
    __nv_bfloat16 *residual, __nv_bfloat16 *hidden, __nv_bfloat16 *logits)
{
    float *part0 = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 256U);
    __nv_bfloat16 *bsc0 = (__nv_bfloat16 *) KPR_GPU_ALLOC(
        (uint32_t) sizeof(__nv_bfloat16), 8192U);
    float *csc0 =
        (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), (uint32_t) 1986560U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, residual, hidden, part0,
        normw, bsc0);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_1, 1U, 32U, 0U, s1, table, bsc0, csc0);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_kplay_run_2, 1U, 1U, 0U, s2, csc0, logits);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    MUST(cudaFree(part0));
    MUST(cudaFree(bsc0));
    MUST(cudaFree(csc0));
}
