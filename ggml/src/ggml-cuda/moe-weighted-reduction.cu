#include "moe-weighted-reduction.cuh"

__device__ __forceinline__ float moe_load_nt(const float * p) {
#if defined(GGML_USE_HIP) || defined(__clang__)
    return __builtin_nontemporal_load(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float4 moe_load_nt_f4(const float4 * p) {
#if defined(GGML_USE_HIP) || defined(__clang__)
    typedef float moe_nt_f4 __attribute__((ext_vector_type(4)));
    const moe_nt_f4 v = __builtin_nontemporal_load((const moe_nt_f4 *) p);
    return make_float4(v.x, v.y, v.z, v.w);
#else
    return *p;
#endif
}

static __global__ void moe_weighted_reduction_f32_vec4(const float4 * __restrict__ experts,
                                                       const float * __restrict__ expert_scale,
                                                       const float * __restrict__ weights,
                                                       float4 * __restrict__ dst,
                                                       const int64_t n_embd4,
                                                       const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col4  = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col4 >= n_embd4) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    const float    first_w     = first_scale * weights[first_row];
    const float4   first_val   = moe_load_nt_f4(&experts[first_row * n_embd4 + col4]);
    float4         sum         = make_float4(
        first_val.x * first_w,
        first_val.y * first_w,
        first_val.z * first_w,
        first_val.w * first_w);

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float    scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        const float    w     = scale * weights[row];
        const float4   val   = moe_load_nt_f4(&experts[row * n_embd4 + col4]);
        sum.x += val.x * w;
        sum.y += val.y * w;
        sum.z += val.z * w;
        sum.w += val.w * w;
    }
    dst[token * n_embd4 + col4] = sum;
}

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (moe_load_nt(&experts[first_row * n_embd + col]) * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float    scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (moe_load_nt(&experts[row * n_embd + col]) * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    if (n_embd % 4 == 0 &&
        ((uintptr_t) experts % sizeof(float4) == 0) &&
        ((uintptr_t) dst % sizeof(float4) == 0)) {
        const int64_t n_embd4 = n_embd / 4;
        const dim3    blocks(n_tokens, (n_embd4 + threads - 1) / threads, 1);
        moe_weighted_reduction_f32_vec4<<<blocks, threads, 0, stream>>>(
            (const float4 *) experts, expert_scale, weights, (float4 *) dst, n_embd4, n_expert_used);
        return;
    }
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}
