#include "gated_delta_net.cuh"
#include "gated-delta-net-mma.cuh"
#include "ggml-cuda/common.cuh"

#include <cstdint>

constexpr int gdn_cols_per_warp = 4;

struct gdn_prologue_params {
    const float * dt_bias;
    const float * ssm_a;
    float         q_eps;
    float         q_scale;
    float         k_eps;
    float         k_scale;
    bool          has_qk_norm;
    bool          has_gate;
    bool          has_beta_sigmoid;
};

template <int S_v, bool KDA, bool keep_rs_t, bool fuse_prologue = false, bool vec4_io = true, int cols_per_warp = gdn_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     gdn_prologue_params prologue) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    constexpr bool use_vec4 = vec4_io && (rows_per_lane % 4 == 0);

    ggml_cuda_pdl_sync();
    if constexpr (use_vec4) {
        const float4 * curr_state4 = reinterpret_cast<const float4 *>(curr_state);
#pragma unroll
        for (int r4 = 0; r4 < rows_per_lane / 4; r4++) {
            const float4 v4 = curr_state4[r4 * lanes_per_col + lane_in_col];
            s_shard[r4 * 4 + 0] = v4.x;
            s_shard[r4 * 4 + 1] = v4.y;
            s_shard[r4 * 4 + 2] = v4.z;
            s_shard[r4 * 4 + 3] = v4.w;
        }
    } else {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            s_shard[r]  = curr_state[i];
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        float beta_val = *beta_t;
        if constexpr (fuse_prologue) {
            if (prologue.has_beta_sigmoid) {
                beta_val = 1.0f / (1.0f + expf(-beta_val));
            }
        }

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
        if constexpr (use_vec4) {
            const float4 * k_t4 = reinterpret_cast<const float4 *>(k_t);
            const float4 * q_t4 = reinterpret_cast<const float4 *>(q_t);
#pragma unroll
            for (int r4 = 0; r4 < rows_per_lane / 4; r4++) {
                const int idx4  = r4 * lanes_per_col + lane_in_col;
                const float4 kv = k_t4[idx4];
                const float4 qv = q_t4[idx4];
                k_reg[r4 * 4 + 0] = kv.x;
                k_reg[r4 * 4 + 1] = kv.y;
                k_reg[r4 * 4 + 2] = kv.z;
                k_reg[r4 * 4 + 3] = kv.w;
                q_reg[r4 * 4 + 0] = qv.x;
                q_reg[r4 * 4 + 1] = qv.y;
                q_reg[r4 * 4 + 2] = qv.z;
                q_reg[r4 * 4 + 3] = qv.w;
            }
        } else {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                k_reg[r] = k_t[i];
                q_reg[r] = q_t[i];
            }
        }

        if constexpr (fuse_prologue) {
            if (prologue.has_qk_norm) {
                float q_ss = 0.0f;
                float k_ss = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    q_ss += q_reg[r] * q_reg[r];
                    k_ss += k_reg[r] * k_reg[r];
                }
                q_ss = warp_reduce_sum<lanes_per_col>(q_ss);
                k_ss = warp_reduce_sum<lanes_per_col>(k_ss);
                const float q_inv = prologue.q_scale * rsqrtf(q_ss * (1.0f / float(S_v)) + prologue.q_eps);
                const float k_inv = prologue.k_scale * rsqrtf(k_ss * (1.0f / float(S_v)) + prologue.k_eps);
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    q_reg[r] *= q_inv;
                    k_reg[r] *= k_inv;
                }
            }
        }

        if constexpr (!KDA) {
            float g_raw = *g_t;
            if constexpr (fuse_prologue) {
                if (prologue.has_gate) {
                    const float x = g_raw + (prologue.dt_bias != nullptr ? prologue.dt_bias[h_idx] : 0.0f);
                    const float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
                    g_raw = sp * prologue.ssm_a[h_idx];
                }
            }
            const float g_val = expf(g_raw);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
            if constexpr (use_vec4) {
#pragma unroll
                for (int c = 0; c < 4; c++) {
#pragma unroll
                    for (int r4 = 0; r4 < rows_per_lane / 4; r4++) {
                        kv_shard += s_shard[r4 * 4 + c] * k_reg[r4 * 4 + c];
                    }
                }
            } else {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += s_shard[r] * k_reg[r];
                }
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
            if constexpr (use_vec4) {
#pragma unroll
                for (int c = 0; c < 4; c++) {
#pragma unroll
                    for (int r4 = 0; r4 < rows_per_lane / 4; r4++) {
                        const int r = r4 * 4 + c;
                        s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                        attn_partial += s_shard[r] * q_reg[r];
                    }
                }
            } else {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                    attn_partial += s_shard[r] * q_reg[r];
                }
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = use_vec4 ? ((r / 4) * lanes_per_col + lane_in_col) * 4 + (r % 4) : (r * lanes_per_col + lane_in_col);
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = use_vec4 ? ((r / 4) * lanes_per_col + lane_in_col) * 4 + (r % 4) : (r * lanes_per_col + lane_in_col);
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
                if constexpr (use_vec4) {
                    float4 * dst_state4 = reinterpret_cast<float4 *>(curr_state + col * S_v);
#pragma unroll
                    for (int r4 = 0; r4 < rows_per_lane / 4; r4++) {
                        dst_state4[r4 * lanes_per_col + lane_in_col] = make_float4(
                            s_shard[r4 * 4 + 0], s_shard[r4 * 4 + 1], s_shard[r4 * 4 + 2], s_shard[r4 * 4 + 3]);
                    }
                } else {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * lanes_per_col + lane_in_col;
                        curr_state[col * S_v + i] = s_shard[r];
                    }
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
        if constexpr (use_vec4) {
            float4 * dst_state4 = reinterpret_cast<float4 *>(state + col * S_v);
#pragma unroll
            for (int r4 = 0; r4 < rows_per_lane / 4; r4++) {
                dst_state4[r4 * lanes_per_col + lane_in_col] = make_float4(
                    s_shard[r4 * 4 + 0], s_shard[r4 * 4 + 1], s_shard[r4 * 4 + 2], s_shard[r4 * 4 + 3]);
            }
        } else {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i          = r * lanes_per_col + lane_in_col;
                state[col * S_v + i] = s_shard[r];
            }
        }
    }
}

template <bool KDA, bool keep_rs_t, bool fuse_prologue = false>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K,
        gdn_prologue_params prologue, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // four columns per warp (see the kernel); shrink the CTA when the wider CTA would leave
    // SMs without a CTA, so small head counts keep the device filled
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int cols_per_warp = gdn_cols_per_warp;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one CTA covers cols_per_warp*num_warps columns (see the kernel)
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const bool vec4_aligned =
        (((uintptr_t) q_d | (uintptr_t) k_d | (uintptr_t) s_d | (uintptr_t) dst_d | (uintptr_t) state_d) & 15) == 0 &&
        (sq1 & 3) == 0 && (sq2 & 3) == 0 && (sq3 & 3) == 0 &&
        (!keep_rs_t || (state_slot_stride & 3) == 0);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    const auto launch_sv = [&](auto sv_tag, auto vec4_tag) {
        constexpr int  sv_val   = decltype(sv_tag)::value;
        constexpr bool vec4_val = decltype(vec4_tag)::value;
        ggml_cuda_kernel_launch(gated_delta_net_cuda<sv_val, KDA, keep_rs_t, fuse_prologue, vec4_val>, launch_params,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, prologue);
    };
    const auto dispatch_sv = [&](auto sv_tag) {
        if (vec4_aligned) {
            launch_sv(sv_tag, std::true_type{});
        } else {
            launch_sv(sv_tag, std::false_type{});
        }
    };
    switch (S_v) {
        case 16:
            dispatch_sv(std::integral_constant<int, 16>{});
            break;
        case 32:
            dispatch_sv(std::integral_constant<int, 32>{});
            break;
        case 64:
            dispatch_sv(std::integral_constant<int, 64>{});
            break;
        case 128:
            dispatch_sv(std::integral_constant<int, 128>{});
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static ggml_cuda_gdn_mma_args ggml_cuda_gdn_mma_args_from_tensor(const ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    const int64_t S_v = v->ne[0];
    const int64_t H   = v->ne[1];
    return {
        (const float *) q->data,
        (const float *) k->data,
        (const float *) v->data,
        (const float *) g->data,
        (const float *) beta->data,
        (const float *) state->data,
        (float *) dst->data,
        nullptr,
        H,
        q->ne[1],
        v->ne[2],
        v->ne[3],
        v->ne[3] / q->ne[3],
        (int64_t) (q->nb[1] / sizeof(float)),
        (int64_t) (q->nb[2] / sizeof(float)),
        (int64_t) (q->nb[3] / sizeof(float)),
        (int64_t) (v->nb[1] / sizeof(float)),
        (int64_t) (v->nb[2] / sizeof(float)),
        (int64_t) (v->nb[3] / sizeof(float)),
        (int64_t) (beta->nb[1] / sizeof(float)),
        (int64_t) (beta->nb[2] / sizeof(float)),
        (int64_t) (beta->nb[3] / sizeof(float)),
        1.0f / sqrtf((float) S_v),
        g->ne[0] == 1 && ggml_get_op_params_i32(dst, 0) <= 1 && S_v == 128 && q->ne[0] == 128,
    };
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fusion_args * fusion) {
    const bool has_prologue = fusion != nullptr &&
        (fusion->fuse_qk_norm || fusion->fuse_beta_sigmoid || fusion->fuse_gate);

    const ggml_tensor * src_q     = (fusion && fusion->fuse_qk_norm && fusion->raw_q) ? fusion->raw_q : dst->src[0];
    const ggml_tensor * src_k     = (fusion && fusion->fuse_qk_norm && fusion->raw_k) ? fusion->raw_k : dst->src[1];
    const ggml_tensor * src_v     = dst->src[2];
    const ggml_tensor * src_g     = (fusion && fusion->fuse_gate && fusion->raw_alpha) ? fusion->raw_alpha : dst->src[3];
    const ggml_tensor * src_beta  = (fusion && fusion->fuse_beta_sigmoid && fusion->raw_beta) ? fusion->raw_beta : dst->src[4];
    const ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, dst->src[4], nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (dst->src[3]->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    GGML_ASSERT(neq3 > 0 && nev3 >= neq3 && nev3 % neq3 == 0);
    GGML_ASSERT(nek3 == neq3);
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(dst->src[3]->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (fusion != nullptr && fusion->has_cache_fusion) {
        state_d           = fusion->cache.data;
        state_slot_stride = fusion->cache.slot_stride;
    }

    if (!has_prologue) {
        ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
        args.state_out = state_d;
        if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
            return;
        }
    }

    gdn_prologue_params prologue = {};
    if (has_prologue) {
        prologue.has_qk_norm      = fusion->fuse_qk_norm;
        prologue.q_eps            = fusion->fuse_qk_norm ? fusion->q_eps : 0.0f;
        prologue.q_scale          = fusion->fuse_qk_norm ? fusion->q_scale : 1.0f;
        prologue.k_eps            = fusion->fuse_qk_norm ? fusion->k_eps : 0.0f;
        prologue.k_scale          = fusion->fuse_qk_norm ? fusion->k_scale : 1.0f;
        prologue.has_beta_sigmoid = fusion->fuse_beta_sigmoid;
        prologue.has_gate         = fusion->fuse_gate && fusion->ssm_a != nullptr;
        prologue.dt_bias          = (fusion->fuse_gate && fusion->dt_bias) ? (const float *) fusion->dt_bias->data : nullptr;
        prologue.ssm_a            = (fusion->fuse_gate && fusion->ssm_a)   ? (const float *) fusion->ssm_a->data   : nullptr;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, prologue, stream);
        } else {
            launch_gated_delta_net<true, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, prologue, stream);
        }
    } else if (has_prologue) {
        if (keep_rs) {
            launch_gated_delta_net<false, true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, prologue, stream);
        } else {
            launch_gated_delta_net<false, false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, prologue, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, prologue, stream);
        } else {
            launch_gated_delta_net<false, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, prologue, stream);
        }
    }
}

bool ggml_cuda_gdn_mma_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_gdn_mma_supported(device, ggml_cuda_gdn_mma_args_from_tensor(dst));
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_gated_delta_net_fusion_args fusion = {};
    fusion.has_cache_fusion = true;
    fusion.cache            = cache;
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &fusion);
}

void ggml_cuda_op_gated_delta_net_fused(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fusion_args & fusion) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &fusion);
}
