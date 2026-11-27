
#include "KPlay_Functional.h"

void KPlay_Functional_kplay_run(float *state_in, float *conv_in,
    __nv_bfloat16 *key_in, __nv_bfloat16 *value_in, uint32_t steps,
    uint32_t hist_n, int64_t *tokens_in, __nv_bfloat16 *history_in,
    float *state_out, float *conv_out, __nv_bfloat16 *key_out,
    __nv_bfloat16 *value_out, int64_t *tokens_out, __nv_bfloat16 *history_out,
    int64_t p, int64_t l)
{
    KRML_MAYBE_UNUSED_VAR(p);
    KRML_MAYBE_UNUSED_VAR(l);
    MUST(cudaMemcpy(key_out, key_in,
        (uint64_t) sizeof(__nv_bfloat16) * (uint64_t) (uint32_t) 6291456U,
        cudaMemcpyDeviceToDevice));
    MUST(cudaMemcpy(value_out, value_in,
        (uint64_t) sizeof(__nv_bfloat16) * (uint64_t) (uint32_t) 6291456U,
        cudaMemcpyDeviceToDevice));
    MUST(cudaMemcpy(tokens_out, tokens_in,
        (uint64_t) sizeof(int64_t) * (uint64_t) steps,
        cudaMemcpyDeviceToDevice));
    MUST(cudaMemcpy(history_out, history_in,
        (uint64_t) sizeof(__nv_bfloat16) * (uint64_t) hist_n,
        cudaMemcpyDeviceToDevice));
    MUST(cudaMemcpy(state_out, state_in,
        (uint64_t) sizeof(float) * (uint64_t) (uint32_t) 4718592U,
        cudaMemcpyDeviceToDevice));
    MUST(cudaMemcpy(conv_out, conv_in,
        (uint64_t) sizeof(float) * (uint64_t) (uint32_t) 442368U,
        cudaMemcpyDeviceToDevice));
}
