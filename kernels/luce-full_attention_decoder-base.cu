#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>
#include <cstdint>
#include <vector>

__device__ float contiguous_sum(const float* values, int size, int width) {
    float partial[256];
    int count = size / width;
    for (int i = 0; i < count; ++i) {
        float value = 0.0f;
        for (int j = 0; j < width; ++j)
            value = __fadd_rn(value, values[i * width + j]);
        partial[i] = value;
    }
    for (int offset = count / 2; offset; offset /= 2)
        for (int i = 0; i < offset; ++i)
            partial[i] = __fadd_rn(partial[i], partial[i + offset]);
    return partial[0];
}

__device__ float recurrent_sum(const float* values) {
    float partial[4];
    for (int y = 0; y < 4; ++y) {
        float a[4] = {};
        for (int i = y; i < 128; i += 16)
            for (int j = 0; j < 4; ++j)
                a[j] = __fadd_rn(a[j], values[i + 4 * j]);
        partial[y] = __fadd_rn(__fadd_rn(__fadd_rn(a[0], a[1]), a[2]), a[3]);
    }
    return __fadd_rn(__fadd_rn(partial[0], partial[2]), __fadd_rn(partial[1], partial[3]));
}

__device__ float sigmoid_value(float x) {
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x)));
}

__device__ float silu_value(float x) {
    return __fdiv_rn(x, __fadd_rn(1.0f, expf(-x)));
}


void check_tensor(const torch::Tensor& value, at::ScalarType dtype, int64_t count,
                  const torch::Tensor& device) {
    TORCH_CHECK(value.is_cuda() && value.is_contiguous() && value.scalar_type() == dtype
                && value.device() == device.device() && value.numel() == count,
                "Expected a contiguous CUDA tensor with the declared dtype, size and device");
}

__global__ void full_attention_decoder(
    const at::BFloat16* weights, const at::BFloat16* old_residual,
    const at::BFloat16* old_cache, const at::BFloat16* old_cosine, const at::BFloat16* old_sine,
    at::BFloat16* next_residual, at::BFloat16* new_cache, int capacity, int position) {
    int lane = threadIdx.x;
    int64_t base = 0;
    __shared__ at::BFloat16 input[3584], projected[6144];
    __shared__ __nv_bfloat16 ma[256], mb[256];
    __shared__ float mc[256];
    at::BFloat16 residual[1024], normalized[1024], qkv[6144], z[2048], beta[16], gate[3584];
    at::BFloat16 core[2048], cosine[32], sine[32];
    float values[1024], products[128], updated[128], combined[16], scores[12];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> af;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> bf;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cf;

    if (lane == 0) {
        for (int i = 0; i < 1024; ++i) residual[i] = old_residual[i];
        for (int i = 0; i < 32; ++i) {
            cosine[i] = old_cosine[i];
            sine[i] = old_sine[i];
        }
    }
    int output_projection = 3;
    int projections = output_projection + 4;
    int64_t attention = base + 11012096;
    if (lane == 0) {
        int64_t norm = base;
        for (int i = 0; i < 1024; ++i) {
            values[i] = __fmul_rn(float(residual[i]), float(residual[i]));
        }
        float inverse = rsqrtf(__fadd_rn(__fmul_rn(contiguous_sum(values, 1024, 4), 1.0f / 1024), 1e-6f));
        for (int i = 0; i < 1024; ++i) {
            normalized[i] = at::BFloat16(__fmul_rn(__fmul_rn(float(residual[i]), inverse), __fadd_rn(1.0f, float(weights[norm + i]))));
            input[i] = normalized[i];
        }
    }
    __syncwarp();
    for (int projection = 0; projection < projections; ++projection) {
        int n = 0, k = 1024;
        int64_t offset = 0;
        if (projection == output_projection) {
            n = 1024; k = 2048;
            offset = attention + 5243392;
        } else if (projection > output_projection) {
            int p = projection - output_projection;
            n = p == 3 ? 1024 : 3584;
            k = p == 3 ? 3584 : 1024;
            offset = base + 2048 + (p - 1) * 3670016;
        } else {
            n = projection == 0 ? 4096 : 512;
            offset = attention + (projection == 0 ? 0 : (projection == 1 ? 4194304 : 4718592));
        }
        int chunk = k == 3584 ? 1792 : k;
        for (int row = 0; row < n; row += 16) {
            for (int start = 0; start < k; start += chunk) {
                nvcuda::wmma::fill_fragment(cf, 0.0f);
                for (int p = start; p < start + chunk; p += 16) {
                    for (int i = lane; i < 256; i += 32) {
                        ma[i] = __ushort_as_bfloat16(weights[offset + int64_t(row + i / 16) * k + p + i % 16].x);
                        mb[i] = __ushort_as_bfloat16(input[p + i % 16].x);
                    }
                    __syncwarp();
                    nvcuda::wmma::load_matrix_sync(af, ma, 16);
                    nvcuda::wmma::load_matrix_sync(bf, mb, 16);
                    nvcuda::wmma::mma_sync(cf, af, bf, cf);
                }
                nvcuda::wmma::store_matrix_sync(mc, cf, 16, nvcuda::wmma::mem_row_major);
                __syncwarp();
                if (lane == 0)
                    for (int i = 0; i < 16; ++i)
                        combined[i] = start == 0 ? mc[i * 16] : __fadd_rn(combined[i], mc[i * 16]);
            }
            if (lane == 0)
                for (int i = 0; i < 16; ++i) {
                    projected[row + i] = at::BFloat16(combined[i]);
                }
        }
        if (projection == 2 && position == 7) {
                if (lane == 0) {
                    for (int i = 0; i < 2048; ++i) input[i] = qkv[i];
                    for (int i = 0; i < 512; ++i) projected[512 + i] = qkv[4096 + i];
                }
                __syncwarp();
                int cache = 0;
                for (int head = 0; head < 8; ++head) {
                    nvcuda::wmma::fill_fragment(cf, 0.0f);
                    for (int p = 0; p < 256; p += 16) {
                        for (int i = lane; i < 256; i += 32) {
                            int t = i / 16, j = p + i % 16;
                            uint16_t key = t >= 8 ? 0 : (t == 7 ? projected[512 + (head / 4) * 256 + j].x : old_cache[cache + ((head / 4) * capacity + t) * 256 + j].x);
                            ma[i] = __ushort_as_bfloat16(key);
                            mb[i] = __ushort_as_bfloat16(input[head * 256 + j].x);
                        }
                        __syncwarp();
                        nvcuda::wmma::load_matrix_sync(af, ma, 16);
                        nvcuda::wmma::load_matrix_sync(bf, mb, 16);
                        nvcuda::wmma::mma_sync(cf, af, bf, cf);
                    }
                    nvcuda::wmma::store_matrix_sync(mc, cf, 16, nvcuda::wmma::mem_row_major);
                    __syncwarp();
                    if (lane == 0)
                        for (int i = 0; i < 8; ++i) projected[1024 + head * 8 + i] = at::BFloat16(mc[i * 16]);
                }
        }
        if (lane == 0) {
            if (projection < output_projection) {
                {
                    if (projection < 2) {
                        int heads = projection == 0 ? 8 : 2;
                        int stride = projection == 0 ? 512 : 256;
                        int destination = projection == 0 ? 0 : 4096;
                        int64_t norm = attention + (projection == 0 ? 5242880 : 5243136);
                        for (int head = 0; head < heads; ++head) {
                            for (int i = 0; i < 256; ++i) values[i] = __fmul_rn(float(projected[head * stride + i]), float(projected[head * stride + i]));
                            float inverse = rsqrtf(__fadd_rn(__fmul_rn(contiguous_sum(values, 256, 4), 1.0f / 256), 1e-6f));
                            for (int i = 0; i < 256; ++i)
                                qkv[destination + head * 256 + i] = at::BFloat16(__fmul_rn(__fmul_rn(float(projected[head * stride + i]), inverse), __fadd_rn(1.0f, float(weights[norm + i]))));
                            for (int i = 0; i < 32; ++i) {
                                float a = float(qkv[destination + head * 256 + i]);
                                float b = float(qkv[destination + head * 256 + 32 + i]);
                                float ac = float(at::BFloat16(__fmul_rn(a, float(cosine[i]))));
                                float bs = float(at::BFloat16(__fmul_rn(-b, float(sine[i]))));
                                float bc = float(at::BFloat16(__fmul_rn(b, float(cosine[i]))));
                                float as = float(at::BFloat16(__fmul_rn(a, float(sine[i]))));
                                qkv[destination + head * 256 + i] = at::BFloat16(__fadd_rn(ac, bs));
                                qkv[destination + head * 256 + 32 + i] = at::BFloat16(__fadd_rn(bc, as));
                            }
                        }
                        if (projection == 0)
                            for (int head = 0; head < 8; ++head)
                                for (int i = 0; i < 256; ++i) qkv[2048 + head * 256 + i] = projected[head * 512 + 256 + i];
                    } else {
                        int cache = 0;
                        for (int head = 0; head < 2; ++head)
                            for (int p = 0; p < capacity; ++p)
                                for (int i = 0; i < 256; ++i) {
                                    int index = cache + (head * capacity + p) * 256 + i;
                                    new_cache[index] = p == position ? qkv[4096 + head * 256 + i] : old_cache[index];
                                    new_cache[index + capacity * 512] = p == position ? projected[head * 256 + i] : old_cache[index + capacity * 512];
                                }
                        for (int head = 0; head < 8; ++head) {
                            for (int p = 0; p <= position; ++p) {
                                at::BFloat16 dot;
                                if (position == 7) dot = projected[1024 + head * 8 + p];
                                else {
                                    for (int i = 0; i < 256; ++i) {
                                        at::BFloat16 key = p == position ? qkv[4096 + (head / 4) * 256 + i] : old_cache[cache + ((head / 4) * capacity + p) * 256 + i];
                                        values[i] = __fmul_rn(float(qkv[head * 256 + i]), float(key));
                                    }
                                    dot = at::BFloat16(contiguous_sum(values, 256, 16));
                                }
                                scores[p] = at::BFloat16(__fmul_rn(float(dot), 0.0625f));
                            }
                            int width = 1;
                            while (width <= position) width *= 2;
                            for (int p = 0; p < width; ++p) values[p] = p <= position ? scores[p] : -INFINITY;
                            for (int offset = width / 2; offset; offset /= 2) {
                                for (int p = 0; p < width; ++p) products[p] = values[p] < values[p ^ offset] ? values[p ^ offset] : values[p];
                                for (int p = 0; p < width; ++p) values[p] = products[p];
                            }
                            for (int p = 0; p < width; ++p) {
                                if (p <= position) scores[p] = expf(__fsub_rn(scores[p], values[p]));
                                values[p] = p <= position ? scores[p] : 0.0f;
                            }
                            for (int offset = width / 2; offset; offset /= 2) {
                                for (int p = 0; p < width; ++p) products[p] = __fadd_rn(values[p], values[p ^ offset]);
                                for (int p = 0; p < width; ++p) values[p] = products[p];
                            }
                            for (int p = 0; p <= position; ++p) scores[p] = at::BFloat16(values[p] == 0.0f ? NAN : __fdiv_rn(scores[p], values[p]));
                            if (position == 7)
                                for (int p = 0; p < 8; ++p) input[head * 8 + p] = at::BFloat16(scores[p]);
                            else {
                                for (int i = 0; i < 256; ++i) {
                                    float partial[4] = {};
                                    for (int p = 0; p <= position; ++p) {
                                        at::BFloat16 v = p == position ? projected[(head / 4) * 256 + i] : old_cache[cache + capacity * 512 + ((head / 4) * capacity + p) * 256 + i];
                                        partial[p / 4] = __fmaf_rn(scores[p], float(v), partial[p / 4]);
                                    }
                                    float value = __fadd_rn(__fadd_rn(partial[0], partial[2]), __fadd_rn(partial[1], partial[3]));
                                    if (position == 0) value = __fmul_rn(scores[0], float(projected[(head / 4) * 256 + i]));
                                    at::BFloat16 probability = at::BFloat16(sigmoid_value(float(qkv[2048 + head * 256 + i])));
                                    input[head * 256 + i] = at::BFloat16(__fmul_rn(float(at::BFloat16(value)), float(probability)));
                                }
                            }
                        }
                    }
                }
            } else if (projection == output_projection) {
                for (int i = 0; i < 1024; ++i) {
                    residual[i] = at::BFloat16(__fadd_rn(float(residual[i]), float(projected[i])));
                    values[i] = __fmul_rn(float(residual[i]), float(residual[i]));
                }
                float inverse = rsqrtf(__fadd_rn(__fmul_rn(contiguous_sum(values, 1024, 4), 1.0f / 1024), 1e-6f));
                for (int i = 0; i < 1024; ++i) {
                    normalized[i] = at::BFloat16(__fmul_rn(__fmul_rn(float(residual[i]), inverse), __fadd_rn(1.0f, float(weights[base + 1024 + i]))));
                    input[i] = normalized[i];
                }
            } else if (projection == output_projection + 1) {
                for (int i = 0; i < 3584; ++i) gate[i] = projected[i];
            } else if (projection == output_projection + 2) {
                for (int i = 0; i < 3584; ++i) input[i] = at::BFloat16(__fmul_rn(float(at::BFloat16(silu_value(float(gate[i])))), float(projected[i])));
            } else {
                for (int i = 0; i < 1024; ++i) residual[i] = at::BFloat16(__fadd_rn(float(residual[i]), float(projected[i])));
            }
            if (projection < output_projection - 1)
                for (int i = 0; i < 1024; ++i) input[i] = normalized[i];
        }
        __syncwarp();
        if (projection == 2 && position == 7) {
            int cache = 0 + capacity * 512;
            for (int head = 0; head < 8; ++head) {
                for (int row = 0; row < 256; row += 16) {
                    for (int i = lane; i < 256; i += 32) {
                        int p = i % 16, j = row + i / 16;
                        uint16_t value = p >= 8 ? 0 : (p == 7 ? projected[(head / 4) * 256 + j].x : old_cache[cache + ((head / 4) * capacity + p) * 256 + j].x);
                        ma[i] = __ushort_as_bfloat16(value);
                        mb[i] = __ushort_as_bfloat16(p >= 8 ? 0 : input[head * 8 + p].x);
                    }
                    __syncwarp();
                    nvcuda::wmma::fill_fragment(cf, 0.0f);
                    nvcuda::wmma::load_matrix_sync(af, ma, 16);
                    nvcuda::wmma::load_matrix_sync(bf, mb, 16);
                    nvcuda::wmma::mma_sync(cf, af, bf, cf);
                    nvcuda::wmma::store_matrix_sync(mc, cf, 16, nvcuda::wmma::mem_row_major);
                    __syncwarp();
                    if (lane == 0)
                        for (int i = 0; i < 16; ++i) core[head * 256 + row + i] = at::BFloat16(mc[i * 16]);
                }
            }
            if (lane == 0)
                for (int i = 0; i < 2048; ++i)
                    input[i] = at::BFloat16(__fmul_rn(float(core[i]), float(at::BFloat16(sigmoid_value(float(qkv[2048 + i]))))));
            __syncwarp();
        }
    }

    if (lane == 0)
        for (int i = 0; i < 1024; ++i) next_residual[i] = residual[i];
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                  torch::Tensor input2, torch::Tensor input3,
                                  torch::Tensor input4, int64_t input5) {
    check_tensor(input0, at::kBFloat16, 18352640, input0);
    check_tensor(input1, at::kBFloat16, 1024, input0);
    check_tensor(input2, at::kBFloat16, input2.numel(), input0);
    check_tensor(input3, at::kBFloat16, 32, input0);
    check_tensor(input4, at::kBFloat16, 32, input0);
    TORCH_CHECK(input2.numel() % 1024 == 0 && input2.numel() / 1024 >= 1
                && input2.numel() / 1024 <= 12 && input5 >= 0
                && input5 < input2.numel() / 1024, "Invalid KV capacity or position");
    auto residual = torch::empty({1024}, input0.options());
    auto cache = torch::empty({input2.numel()}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    full_attention_decoder<<<1, 32, 0, stream>>>(
        input0.data_ptr<at::BFloat16>(), input1.data_ptr<at::BFloat16>(),
        input2.data_ptr<at::BFloat16>(), input3.data_ptr<at::BFloat16>(),
        input4.data_ptr<at::BFloat16>(), residual.data_ptr<at::BFloat16>(),
        cache.data_ptr<at::BFloat16>(), input2.numel() / 1024, input5);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {residual, cache};
}
