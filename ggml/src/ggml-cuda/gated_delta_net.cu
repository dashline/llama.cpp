#include "gated_delta_net.cuh"
#include "gated-delta-net-mma.cuh"
#include "ggml-cuda/common.cuh"
#include "mmvq.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

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

// ---------------------------------------------------------------------------------------------
// Fused GDN decode layer: conv (+ new conv state) -> gated delta rule (+ recurrent state)
// -> gated RMS norm (+ q8_1 deposit), one launch per layer for small token counts.
//
// Grid (H, n_seqs, S_v / 16), block (warp, 4): the same (head, column slice) decomposition as
// gated_delta_net_cuda, so the delta-rule arithmetic is copied verbatim and stays bit-identical.
// The conv is recomputed per block from the cache state and x (cheap: 4 FMAs per channel) into
// shared memory instead of being read from a materialized SSM_CONV output. The new conv state
// and the gated norm need every block of a k-head group to be done, so the last block to retire
// (per-(seq, k-head) counter, atomicInc wrapping back to 0) performs them.
// ---------------------------------------------------------------------------------------------

struct gdn_decode_params {
    // conv: input sequence of channel c is [state0, state1, state2, x_0 .. x_{T-1}]
    const float * x;         // x[seq*x_s2 + t*C + c]
    const float * cs_in;     // cs_in[seq*cs_in_s2 + c*3 + j]
    const float * conv_w;    // conv_w[c*4 + j]
    float *       cs_out;    // cs_out[seq*cs_out_s1 + c*3 + j]
    int64_t       x_s2;
    int64_t       cs_in_s2;
    int64_t       cs_out_s1;
    int           C;
    int           T;
    int           H;
    int           Hk;
    int           q_off;     // conv channel of (head 0, dim 0) for q / k / v
    int           k_off;
    int           v_off;
    int           qk_s1;     // channels per q/k head
    int           v_s1;      // channels per v head
    // gated delta rule (prologue folded in)
    const float * alpha;     // raw alpha, indexed like beta
    const float * beta;      // raw beta: beta[seq*sb3 + t*sb2 + h*sb1]
    const float * dt_bias;   // [H] or nullptr
    const float * ssm_a;     // [H]
    int64_t       sb1, sb2, sb3;
    float         q_eps, q_scale, k_eps, k_scale;
    float         attn_scale;
    const float * state_in;  // [S_v, S_v, H, n_seqs]
    float *       state_out; // same layout (recurrent cache)
    float *       gdn_dst;   // [S_v, H, T, n_seqs]
    // gated RMS norm
    const float * norm_w;    // [S_v]
    const float * z;         // [S_v, H, T, n_seqs]
    float         norm_eps;
    float *       out;       // [S_v, H, T, n_seqs]
    block_q8_1 *  q8_out;    // q8_1 of `out`, or nullptr
    uint32_t *    counters;  // [Hk, n_seqs], zero between launches
};

template <int S_v>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gdn_decode_fused_cuda(const gdn_decode_params p) {
    constexpr int warp_size     = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    constexpr int cols_per_warp = gdn_cols_per_warp;
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    constexpr int num_warps     = 4;
    constexpr int threads       = warp_size * num_warps;
    constexpr int cols_per_blk  = cols_per_warp * num_warps;
    static_assert(S_v % cols_per_blk == 0, "S_v must be a multiple of the columns per block");
    static_assert(rows_per_lane % 4 == 0, "vec4 state path");

    const int h    = blockIdx.x;
    const int seq  = blockIdx.y;
    const int hk   = h % p.Hk;
    const int col0 = blockIdx.z * cols_per_blk;

    const int lane        = threadIdx.x;
    const int tid         = threadIdx.y * warp_size + lane;
    const int col_in_warp = lane / lanes_per_col;
    const int lane_in_col = lane - col_in_warp * lanes_per_col;
    const int col         = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    __shared__ float sh_q[S_v];
    __shared__ float sh_k[S_v];
    __shared__ float sh_v[cols_per_blk];
    __shared__ int   sh_last;

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    auto conv_in = [&](int c, int idx) -> float {
        return idx < 3 ? p.cs_in[seq * p.cs_in_s2 + (int64_t) c * 3 + idx]
                       : p.x[seq * p.x_s2 + (int64_t) (idx - 3) * p.C + c];
    };
    // same statement order as ssm_conv_f32 (bias is absent there, but the add is still emitted)
    auto conv_silu = [&](int c, int t) -> float {
        const float * w = p.conv_w + (int64_t) c * 4;
        float sumf = 0.0f;
        sumf += conv_in(c, t + 0) * w[0];
        sumf += conv_in(c, t + 1) * w[1];
        sumf += conv_in(c, t + 2) * w[2];
        sumf += conv_in(c, t + 3) * w[3];
        sumf += 0.0f;
        return ggml_cuda_op_silu_single(sumf);
    };

    const int64_t state_off  = ((int64_t) seq * p.H + h) * S_v * S_v;
    const float * curr_state = p.state_in + state_off + col * S_v;
    float *       attn_data  = p.gdn_dst + ((int64_t) seq * p.T * p.H + h) * S_v;

    float s_shard[rows_per_lane];
    {
        const float4 * curr_state4 = reinterpret_cast<const float4 *>(curr_state);
#pragma unroll
        for (int r4 = 0; r4 < rows_per_lane / 4; r4++) {
            const float4 v4 = curr_state4[r4 * lanes_per_col + lane_in_col];
            s_shard[r4 * 4 + 0] = v4.x;
            s_shard[r4 * 4 + 1] = v4.y;
            s_shard[r4 * 4 + 2] = v4.z;
            s_shard[r4 * 4 + 3] = v4.w;
        }
    }

    for (int t = 0; t < p.T; t++) {
        for (int i = tid; i < S_v; i += threads) {
            sh_q[i] = conv_silu(p.q_off + hk * p.qk_s1 + i, t);
            sh_k[i] = conv_silu(p.k_off + hk * p.qk_s1 + i, t);
        }
        if (tid < cols_per_blk) {
            sh_v[tid] = conv_silu(p.v_off + h * p.v_s1 + col0 + tid, t);
        }
        __syncthreads();

        const int64_t gb_offset = seq * p.sb3 + t * p.sb2 + h * p.sb1;

        float beta_val = p.beta[gb_offset];
        beta_val = 1.0f / (1.0f + expf(-beta_val));

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r4 = 0; r4 < rows_per_lane / 4; r4++) {
            const int idx4 = r4 * lanes_per_col + lane_in_col;
#pragma unroll
            for (int c = 0; c < 4; c++) {
                k_reg[r4 * 4 + c] = sh_k[idx4 * 4 + c];
                q_reg[r4 * 4 + c] = sh_q[idx4 * 4 + c];
            }
        }

        {
            float q_ss = 0.0f;
            float k_ss = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                q_ss += q_reg[r] * q_reg[r];
                k_ss += k_reg[r] * k_reg[r];
            }
            q_ss = warp_reduce_sum<lanes_per_col>(q_ss);
            k_ss = warp_reduce_sum<lanes_per_col>(k_ss);
            const float q_inv = p.q_scale * rsqrtf(q_ss * (1.0f / float(S_v)) + p.q_eps);
            const float k_inv = p.k_scale * rsqrtf(k_ss * (1.0f / float(S_v)) + p.k_eps);
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                q_reg[r] *= q_inv;
                k_reg[r] *= k_inv;
            }
        }

        float g_raw = p.alpha[gb_offset];
        {
            const float x  = g_raw + (p.dt_bias != nullptr ? p.dt_bias[h] : 0.0f);
            const float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
            g_raw = sp * p.ssm_a[h];
        }
        const float g_val = expf(g_raw);

        float kv_shard = 0.0f;
#pragma unroll
        for (int c = 0; c < 4; c++) {
#pragma unroll
            for (int r4 = 0; r4 < rows_per_lane / 4; r4++) {
                kv_shard += s_shard[r4 * 4 + c] * k_reg[r4 * 4 + c];
            }
        }
        const float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

        const float delta_col = (sh_v[col - col0] - g_val * kv_col) * beta_val;

        float attn_partial = 0.0f;
#pragma unroll
        for (int c = 0; c < 4; c++) {
#pragma unroll
            for (int r4 = 0; r4 < rows_per_lane / 4; r4++) {
                const int r = r4 * 4 + c;
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }
        }
        const float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

        if (lane_in_col == 0) {
            attn_data[col] = attn_col * p.attn_scale;
        }
        attn_data += S_v * p.H;

        __syncthreads(); // sh_q / sh_k / sh_v are rewritten for the next token
    }

    {
        float4 * dst_state4 = reinterpret_cast<float4 *>(p.state_out + state_off + col * S_v);
#pragma unroll
        for (int r4 = 0; r4 < rows_per_lane / 4; r4++) {
            dst_state4[r4 * lanes_per_col + lane_in_col] = make_float4(
                s_shard[r4 * 4 + 0], s_shard[r4 * 4 + 1], s_shard[r4 * 4 + 2], s_shard[r4 * 4 + 3]);
        }
    }

    // ---- last block of the k-head group: new conv state + gated norm ------------------------
    const int group  = p.H / p.Hk;
    const int n_rows = group * p.T; // (token, head) rows of this group that the gated norm covers

    constexpr int norm_warps = 2;
    constexpr int conv_warps = num_warps - norm_warps;
    static_assert(conv_warps > 0, "need warps for the conv-state shift");

    // the rows a would-be last block handles are known up front: pull in the norm weight and gate
    // for the first row of each norm warp before the retire atomic (independent of other blocks)
    float4 pre_m = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 pre_g = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if constexpr (warp_size * 4 == S_v) {
        if (threadIdx.y < norm_warps && threadIdx.y < n_rows) {
            const int     r       = threadIdx.y;
            const int     t       = r / group;
            const int     hh      = hk + (r - t * group) * p.Hk;
            const int64_t row_idx = ((int64_t) seq * p.T + t) * p.H + hh;
            pre_m = reinterpret_cast<const float4 *>(p.norm_w)[lane];
            pre_g = reinterpret_cast<const float4 *>(p.z + row_idx * S_v)[lane];
        }
    }

    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const uint32_t target = (uint32_t) group * gridDim.z;
        sh_last = (atomicInc(&p.counters[seq * p.Hk + hk], target - 1) == target - 1);
    }
    __syncthreads();
    if (!sh_last) {
        return;
    }
    __threadfence();

    if (threadIdx.y >= norm_warps) {
        // every reader of this group's conv-state channels has retired: shift the window. One
        // thread per channel; all loads of a batch are issued before its stores since cs_out may
        // alias cs_in.
        constexpr int batch    = 4;
        const int     n_ch     = (2 + group) * S_v;
        const int     ctid     = (threadIdx.y - norm_warps) * warp_size + lane;
        const int     cthreads = conv_warps * warp_size;
        for (int base = ctid; base < n_ch; base += batch * cthreads) {
            int   ch[batch];
            float n0[batch], n1[batch], n2[batch];
#pragma unroll
            for (int b = 0; b < batch; b++) {
                const int ci = base + b * cthreads;
                int c = 0;
                if (ci < S_v) {
                    c = p.q_off + hk * p.qk_s1 + ci;
                } else if (ci < 2 * S_v) {
                    c = p.k_off + hk * p.qk_s1 + (ci - S_v);
                } else if (ci < n_ch) {
                    const int m = (ci - 2 * S_v) / S_v;
                    c = p.v_off + (hk + m * p.Hk) * p.v_s1 + (ci - 2 * S_v - m * S_v);
                }
                ch[b] = c;
                if (ci < n_ch) {
                    n0[b] = conv_in(c, p.T + 0);
                    n1[b] = conv_in(c, p.T + 1);
                    n2[b] = conv_in(c, p.T + 2);
                }
            }
#pragma unroll
            for (int b = 0; b < batch; b++) {
                const int ci = base + b * cthreads;
                if (ci < n_ch) {
                    float * o = p.cs_out + seq * p.cs_out_s1 + (int64_t) ch[b] * 3;
                    o[0] = n0[b];
                    o[1] = n1[b];
                    o[2] = n2[b];
                }
            }
        }
        return;
    }

    if constexpr (warp_size * 4 != S_v) {
        NO_DEVICE_CODE;
    } else {
        // rms_norm_gated_f32<warp, 1, vec, silu, q8_1> for every (token, head) row of the group
        for (int r = threadIdx.y; r < n_rows; r += norm_warps) {
            const int     t       = r / group;
            const int     hh      = hk + (r - t * group) * p.Hk;
            const int64_t row_idx = ((int64_t) seq * p.T + t) * p.H + hh;

            const volatile float * xr = p.gdn_dst + row_idx * S_v + lane * 4;
            float4 xs;
            xs.x = xr[0];
            xs.y = xr[1];
            xs.z = xr[2];
            xs.w = xr[3];

            float tmp = 0.0f;
            tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
            tmp = block_reduce<block_reduce_method::SUM, warp_size>(tmp, (float *) nullptr);
            const int   ncols = S_v;
            const float scale = rsqrtf(tmp / ncols + p.norm_eps);

            const float4 m = r < norm_warps ? pre_m : reinterpret_cast<const float4 *>(p.norm_w)[lane];
            const float4 g = r < norm_warps ? pre_g : reinterpret_cast<const float4 *>(p.z + row_idx * S_v)[lane];
            const float gx = g.x / (1.0f + expf(-g.x));
            const float gy = g.y / (1.0f + expf(-g.y));
            const float gz = g.z / (1.0f + expf(-g.z));
            const float gw = g.w / (1.0f + expf(-g.w));
            const float4 res = make_float4(
                scale * xs.x * m.x * gx,
                scale * xs.y * m.y * gy,
                scale * xs.z * m.z * gz,
                scale * xs.w * m.w * gw);
            reinterpret_cast<float4 *>(p.out + row_idx * S_v)[lane] = res;
            if (p.q8_out != nullptr) {
                quantize_float4_to_q8_1(res, p.q8_out + row_idx * (S_v / QK8_1), lane);
            }
        }
    }
}

static bool gdn_decode_aligned16(const ggml_tensor * t) {
    return t != nullptr && t->data != nullptr && (((uintptr_t) t->data) & 15) == 0;
}

bool ggml_cuda_gdn_decode_supported(const ggml_cuda_gdn_decode_match & m) {
    if (!m.gdn || !m.x || !m.conv_state || !m.conv_w || !m.conv_cpy || !m.conv_out || !m.rms_norm || !m.norm_w || !m.z || !m.out) {
        return false;
    }
    const auto & info = ggml_cuda_info().devices[ggml_cuda_get_device()];
    if (info.warp_size != 32) {
        return false;
    }

    const ggml_tensor * q = m.gdn_args.raw_q;
    const ggml_tensor * k = m.gdn_args.raw_k;
    const ggml_tensor * v = m.gdn->src[2];
    const ggml_tensor * s = m.gdn->src[5];
    if (!m.gdn_args.fuse_qk_norm || !m.gdn_args.fuse_gate || !m.gdn_args.fuse_beta_sigmoid ||
        !m.gdn_args.has_cache_fusion || m.gdn_args.cache.slot_stride != 0 ||
        ggml_get_op_params_i32(m.gdn, 0) > 1 || !q || !k || !v || !s ||
        !m.gdn_args.raw_alpha || !m.gdn_args.raw_beta || !m.gdn_args.ssm_a) {
        return false;
    }

    const int64_t S_v    = v->ne[0];
    const int64_t H      = v->ne[1];
    const int64_t T      = v->ne[2];
    const int64_t n_seqs = v->ne[3];
    const int64_t Hk     = q->ne[1];
    const int64_t C      = m.x->ne[0];
    if (S_v != 128 || Hk <= 0 || H % Hk != 0 || T < 1 || T > GGML_CUDA_GDN_DECODE_MAX_TOKENS || n_seqs < 1) {
        return false;
    }
    if (m.x->ne[1] != T || m.x->ne[2] != n_seqs || !ggml_is_contiguous(m.x) ||
        m.conv_state->ne[0] != 3 || m.conv_state->ne[1] != C || m.conv_state->ne[2] != n_seqs || !ggml_is_contiguous(m.conv_state) ||
        m.conv_w->ne[0] != 4 || m.conv_w->ne[1] != C || !ggml_is_contiguous(m.conv_w)) {
        return false;
    }
    // Qwen3.5/3.6 conv channel layout: [q heads | k heads | v heads], nothing else
    if (C != 2 * Hk * S_v + H * S_v) {
        return false;
    }
    auto view_ok = [&](const ggml_tensor * t, int64_t n_heads, int64_t off_expected) {
        return t->view_src == m.conv_out && t->type == GGML_TYPE_F32 &&
               t->ne[0] == S_v && t->ne[1] == n_heads && t->ne[2] == T && t->ne[3] == n_seqs &&
               t->nb[0] == sizeof(float) && t->nb[1] == S_v * sizeof(float) &&
               t->nb[2] == (size_t) C * sizeof(float) && t->nb[3] == (size_t) C * T * sizeof(float) &&
               t->view_offs == (size_t) off_expected * sizeof(float);
    };
    if (!view_ok(q, Hk, 0) || !view_ok(k, Hk, Hk * S_v) || !view_ok(v, H, 2 * Hk * S_v)) {
        return false;
    }
    // new conv state: CPY of the last 3 concat columns into a per-seq contiguous cache row
    const ggml_tensor * tail = m.conv_cpy->src[0];
    const ggml_tensor * cdst = m.conv_cpy->src[1];
    if (!tail || !cdst || tail->view_offs != (size_t) T * sizeof(float) ||
        tail->ne[0] != 3 || tail->ne[1] != C || tail->ne[2] != n_seqs || tail->ne[3] != 1 ||
        cdst->type != GGML_TYPE_F32 || cdst->ne[0] != 3 * C || cdst->ne[1] != n_seqs || cdst->ne[2] != 1 || cdst->ne[3] != 1 ||
        cdst->nb[0] != sizeof(float) || cdst->nb[1] % sizeof(float) != 0 || (m.conv_cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    // recurrent state and gdn output
    if (!ggml_is_contiguous(s) || s->ne[0] != S_v || s->ne[1] != S_v || s->ne[2] != H || s->ne[3] != n_seqs) {
        return false;
    }
    if (m.gdn->type != GGML_TYPE_F32 || (m.gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    // gated norm over [S_v, H, T, n_seqs] rows of the gdn output
    const ggml_tensor * nsrc = m.rms_norm->src[0];
    if ((nsrc != m.gdn && nsrc->view_src != m.gdn) || (nsrc->view_src == m.gdn && nsrc->view_offs != 0) ||
        nsrc->ne[0] != S_v || nsrc->ne[1] != H || nsrc->ne[2] != T || nsrc->ne[3] != n_seqs || !ggml_is_contiguous(nsrc)) {
        return false;
    }
    if (m.norm_w->type != GGML_TYPE_F32 || m.norm_w->ne[0] != S_v || ggml_nrows(m.norm_w) != 1 || !ggml_is_contiguous(m.norm_w)) {
        return false;
    }
    if (m.z->type != GGML_TYPE_F32 || !ggml_are_same_shape(m.z, nsrc) || !ggml_is_contiguous(m.z)) {
        return false;
    }
    if (m.out->type != GGML_TYPE_F32 || !ggml_are_same_shape(m.out, nsrc) || !ggml_is_contiguous(m.out)) {
        return false;
    }
    // alpha / beta as the prologue fusion validated them, indexed with beta's strides
    const ggml_tensor * sb = m.gdn->src[4];
    if (sb->nb[0] != sizeof(float) || sb->nb[1] % sizeof(float) != 0 || sb->nb[2] % sizeof(float) != 0 || sb->nb[3] % sizeof(float) != 0) {
        return false;
    }
    // vec4 paths (data pointers are only known once the graph is allocated)
    if (m.gdn->data != nullptr) {
        if (!gdn_decode_aligned16(s) || m.gdn_args.cache.data == nullptr || (((uintptr_t) m.gdn_args.cache.data) & 15) != 0 ||
            !gdn_decode_aligned16(m.gdn) || !gdn_decode_aligned16(m.z) || !gdn_decode_aligned16(m.norm_w) ||
            !gdn_decode_aligned16(m.out)) {
            return false;
        }
    }
    return true;
}

void ggml_cuda_op_gdn_decode_fused(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_decode_match & m) {
    GGML_ASSERT(ggml_cuda_gdn_decode_supported(m));

    const ggml_tensor * q    = m.gdn_args.raw_q;
    const ggml_tensor * k    = m.gdn_args.raw_k;
    const ggml_tensor * v    = m.gdn->src[2];
    const ggml_tensor * sb   = m.gdn->src[4];
    const ggml_tensor * s    = m.gdn->src[5];
    const ggml_tensor * cdst = m.conv_cpy->src[1];

    constexpr int S_v    = 128;
    const int     H      = (int) v->ne[1];
    const int     T      = (int) v->ne[2];
    const int     n_seqs = (int) v->ne[3];
    const int     Hk     = (int) q->ne[1];
    const int     C      = (int) m.x->ne[0];

    cudaStream_t stream = ctx.stream();

    gdn_decode_params p = {};
    p.x         = (const float *) m.x->data;
    p.cs_in     = (const float *) m.conv_state->data;
    p.conv_w    = (const float *) m.conv_w->data;
    p.cs_out    = (float *) cdst->data;
    p.x_s2      = (int64_t) C * T;
    p.cs_in_s2  = (int64_t) 3 * C;
    p.cs_out_s1 = (int64_t) (cdst->nb[1] / sizeof(float));
    p.C         = C;
    p.T         = T;
    p.H         = H;
    p.Hk        = Hk;
    p.q_off     = (int) (q->view_offs / sizeof(float));
    p.k_off     = (int) (k->view_offs / sizeof(float));
    p.v_off     = (int) (v->view_offs / sizeof(float));
    p.qk_s1     = (int) (q->nb[1] / sizeof(float));
    p.v_s1      = (int) (v->nb[1] / sizeof(float));

    p.alpha      = (const float *) m.gdn_args.raw_alpha->data;
    p.beta       = (const float *) m.gdn_args.raw_beta->data;
    p.dt_bias    = m.gdn_args.dt_bias ? (const float *) m.gdn_args.dt_bias->data : nullptr;
    p.ssm_a      = (const float *) m.gdn_args.ssm_a->data;
    p.sb1        = (int64_t) (sb->nb[1] / sizeof(float));
    p.sb2        = (int64_t) (sb->nb[2] / sizeof(float));
    p.sb3        = (int64_t) (sb->nb[3] / sizeof(float));
    p.q_eps      = m.gdn_args.q_eps;
    p.q_scale    = m.gdn_args.q_scale;
    p.k_eps      = m.gdn_args.k_eps;
    p.k_scale    = m.gdn_args.k_scale;
    p.attn_scale = 1.0f / sqrtf((float) S_v);
    p.state_in   = (const float *) s->data;
    p.state_out  = m.gdn_args.cache.data;
    p.gdn_dst    = (float *) m.gdn->data;

    memcpy(&p.norm_eps, m.rms_norm->op_params, sizeof(float));
    p.norm_w = (const float *) m.norm_w->data;
    p.z      = (const float *) m.z->data;
    p.out    = (float *) m.out->data;

    const int64_t flat_row = (int64_t) S_v * H;
    if (flat_row % MATRIX_ROW_PADDING == 0 && (int64_t) T * n_seqs <= MMVQ_MAX_BATCH_SIZE) {
        p.q8_out = ctx.alloc_mmvq_q8_1_cache_slot(m.out, flat_row, flat_row, T, n_seqs, 1,
                                                  flat_row, flat_row * T, flat_row * T * n_seqs);
    }
    p.counters = ctx.mmvq_retire_counters(ggml_backend_cuda_context::MMVQ_RETIRE_REGION_GDN, (size_t) n_seqs * Hk, stream);

    const int  warp_size = 32;
    const dim3 grid_dims(H, n_seqs, S_v / (gdn_cols_per_warp * 4));
    const dim3 block_dims(warp_size, 4, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    ggml_cuda_kernel_launch(gdn_decode_fused_cuda<S_v>, launch_params, p);
}
