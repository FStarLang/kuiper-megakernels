
#include "KPlay_Functional.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting kplay_run
*/
static void
__hoisted_kplay_run_0(int64_t input4, __nv_bfloat16 *output2, int64_t *input1,
    uint32_t eoff, __nv_bfloat16 *input0, int64_t kd, int64_t ke, uint32_t noff,
    __nv_bfloat16 *output0, __nv_bfloat16 *output1)
{
    uint32_t i = 0U;
    while (i < 896U) {
        uint32_t iv = i;
        int64_t p = (int64_t) (uint32_t) (iv / 128U) + input4;
        float a = (float) p *
                  (1.0f / powf(1000000.0f,
                              (float) (int64_t) (uint32_t) (iv % 128U % 64U) /
                                  64.0f));
        output2[iv] = __float2bfloat16(cosf(a));
        output2[iv + 896U] = __float2bfloat16(sinf(a));
        i = iv + 1U;
    }
    uint32_t r = 0U;
    while (r < 7U) {
        uint32_t rv = r;
        int64_t tk = input1[rv];
        uint32_t b =
            eoff + (uint32_t) (tk >= 0LL && tk < 151936LL ? tk : 0LL) * 1024U;
        float partial[128U];
        memset(partial, 0U, 128U * sizeof(float));
        uint32_t lane = 0U;
        while (lane < 128U) {
            uint32_t l = lane;
            uint32_t k = b + 4U * l;
            __nv_bfloat16 __anf1 = input0[k];
            __nv_bfloat16 __anf0 = input0[k + 512U];
            float x0 = __bfloat162float(__anf1);
            float x1 = __bfloat162float(__anf0);
            float s0 = 0.0f + x0 * x0 + x1 * x1;
            __nv_bfloat16 __anf11 = input0[k + 1U];
            __nv_bfloat16 __anf01 = input0[k + 513U];
            float x2 = __bfloat162float(__anf11);
            float x3 = __bfloat162float(__anf01);
            float s1 = 0.0f + x2 * x2 + x3 * x3;
            __nv_bfloat16 __anf12 = input0[k + 2U];
            __nv_bfloat16 __anf02 = input0[k + 514U];
            float x4 = __bfloat162float(__anf12);
            float x5 = __bfloat162float(__anf02);
            float s2 = 0.0f + x4 * x4 + x5 * x5;
            __nv_bfloat16 __anf13 = input0[k + 3U];
            __nv_bfloat16 __anf03 = input0[k + 515U];
            float x6 = __bfloat162float(__anf13);
            float x = __bfloat162float(__anf03);
            partial[l] = s0 + s1 + s2 + (0.0f + x6 * x6 + x * x);
            lane = l + 1U;
        }
        uint32_t off = 128U;
        while (off > 1U) {
            uint32_t h = off / 2U;
            uint32_t i1 = 0U;
            while (i1 < h) {
                uint32_t iv = i1;
                partial[iv] += partial[iv + h];
                i1 = iv + 1U;
            }
            off = h;
        }
        float inv =
            rsqrtf(partial[0U] * (1.0f / (float) kd) + 1.0f / (float) ke);
        uint32_t c = 0U;
        while (c < 1024U) {
            uint32_t cv = c;
            uint32_t idx = rv * 1024U + cv;
            __nv_bfloat16 v = input0[b + cv];
            __nv_bfloat16 wv = input0[noff + cv];
            output0[idx] = v;
            __nv_bfloat16 sc = __float2bfloat16(__bfloat162float(v) * inv);
            output1[idx] =
                __float2bfloat16(__bfloat162float(sc) * __bfloat162float(wv));
            c = cv + 1U;
        }
        r = rv + 1U;
    }
}

void KPlay_Functional_kplay_run(uint32_t n, __nv_bfloat16 *input0,
    int64_t *input1, int64_t input2, int64_t input3, int64_t input4,
    __nv_bfloat16 *output0, __nv_bfloat16 *output1, __nv_bfloat16 *output2)
{
    int64_t n64 = (int64_t) (uint32_t) n;
    if (input2 >= 0LL && input2 <= 1991901183LL &&
        input2 <= n64 - 155582464LL && input3 >= 0LL &&
        input3 <= 2147482623LL && input3 <= n64 - 1024LL && input4 >= 0LL &&
        input4 <= 2147483641LL) {
        uint32_t eoff = (uint32_t) input2;
        uint32_t noff = (uint32_t) input3;
        cudaStream_t s = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_kplay_run_0, 1U, 1U, 0U, s, input4, output2, input1,
            eoff, input0, 1024LL, 1000000LL, noff, output0, output1);
        MUST(cudaStreamSynchronize(s));
        MUST(cudaStreamDestroy(s));
    }
}
