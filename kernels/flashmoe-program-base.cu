#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <vector>

__device__ float softmax_exp(float value) {
    // Scalar form of the installed CPU SLEEF AVX512 exponential, for value <= 0.
    if (value < -104.0f) {
        return 0.0f;
    }
    int exponent = __float2int_rn(__fmul_rn(value, 0x1.715476p+0f));
    float reduced = __fmaf_rn(float(exponent), -0x1.62e4p-1f, value);
    reduced = __fmaf_rn(float(exponent), -0x1.7f7d1cp-20f, reduced);
    float square = __fmul_rn(reduced, reduced);
    float polynomial = __fmaf_rn(0x1.a057b4p-13f, reduced, 0x1.6d2d92p-10f);
    polynomial = __fmaf_rn(polynomial, reduced, 0x1.11114cp-7f);
    polynomial = __fmaf_rn(polynomial, reduced, 0x1.5554f4p-5f);
    polynomial = __fmaf_rn(polynomial, reduced, 0x1.555556p-3f);
    polynomial = __fmaf_rn(polynomial, reduced, 0.5f);
    float result = __fadd_rn(__fmaf_rn(square, polynomial, reduced), 1.0f);
    int half_exponent = exponent >> 1;
    result = __fmul_rn(result, __int_as_float((half_exponent + 127) << 23));
    return __fmul_rn(
        result, __int_as_float((exponent - half_exponent + 127) << 23));
}

__global__ void router_scores(
    const at::BFloat16* tokens,
    const at::BFloat16* router_weights,
    float* scores) {
    for (int token = 0; token < 256; ++token) {
        for (int expert = 0; expert < 8; ++expert) {
            float score = 0.0f;
            for (int hidden = 0; hidden < 512; ++hidden) {
                score = __fmaf_rn(
                    float(tokens[token * 512 + hidden]),
                    float(router_weights[hidden * 8 + expert]), score);
            }
            scores[token * 8 + expert] = score;
        }
    }
}

__global__ void router_softmax(const float* scores, float* softmax) {
    for (int token = 0; token < 256; ++token) {
        float maximum = scores[token * 8];
        for (int expert = 1; expert < 8; ++expert) {
            maximum = fmaxf(maximum, scores[token * 8 + expert]);
        }
        float exponentials[8];
        for (int expert = 0; expert < 8; ++expert) {
            exponentials[expert] =
                softmax_exp(__fsub_rn(scores[token * 8 + expert], maximum));
        }
        float denominator = exponentials[0];
        for (int expert = 1; expert < 8; ++expert) {
            denominator = __fadd_rn(denominator, exponentials[expert]);
        }
        float inverse = __fdiv_rn(1.0f, denominator);
        for (int expert = 0; expert < 8; ++expert) {
            softmax[token * 8 + expert] =
                __fmul_rn(exponentials[expert], inverse);
        }
    }
}

__global__ void router_capacity(
    const float* softmax,
    int32_t* expert_demand,
    int64_t* token_indices,
    float* probabilities,
    int32_t* survival) {
    int64_t prefix[8] = {};
    for (int slot = 0; slot < 256; ++slot) {
        token_indices[slot] = -1;
        probabilities[slot] = 0.0f;
    }
    for (int token = 0; token < 256; ++token) {
        int selected = 0;
        float selected_probability = softmax[token * 8];
        for (int expert = 1; expert < 8; ++expert) {
            float probability = softmax[token * 8 + expert];
            if (probability > selected_probability) {
                selected = expert;
                selected_probability = probability;
            }
        }
        float combine_probability =
            __fdiv_rn(selected_probability, selected_probability);
        int64_t rank = prefix[selected]++;
        for (int expert = 0; expert < 8; ++expert) {
            survival[token * 8 + expert] =
                expert == selected && rank < 32 ? 1 : 0;
        }
        if (rank < 32) {
            token_indices[selected * 32 + rank] = token;
            probabilities[selected * 32 + rank] = combine_probability;
        }
    }
    for (int expert = 0; expert < 8; ++expert) {
        expert_demand[expert] = int32_t(prefix[expert]);
    }
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1) {
    auto scores = torch::empty({256, 8}, input0.options().dtype(at::kFloat));
    auto softmax = torch::empty({256, 8}, input0.options().dtype(at::kFloat));
    auto expert_demand = torch::empty({8}, input0.options().dtype(at::kInt));
    auto token_indices = torch::empty({8, 32}, input0.options().dtype(at::kLong));
    auto probabilities = torch::empty({8, 32}, input0.options().dtype(at::kFloat));
    auto survival = torch::empty({256, 8}, input0.options().dtype(at::kInt));
    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    router_scores<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<at::BFloat16>(), scores.data_ptr<float>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    router_softmax<<<1, 1, 0, stream>>>(scores.data_ptr<float>(), softmax.data_ptr<float>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    router_capacity<<<1, 1, 0, stream>>>(softmax.data_ptr<float>(), expert_demand.data_ptr<int32_t>(), token_indices.data_ptr<int64_t>(), probabilities.data_ptr<float>(), survival.data_ptr<int32_t>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {expert_demand, token_indices, probabilities, survival};
}
