#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

// Direct 2-source GDN causal conv1d + SiLU fusion across CONCAT(state, transpose(x)) -> SSM_CONV -> SILU.
// Snapshots the 3-column prefix state and any tail columns read by conv_state_last VIEWs into concat->data
// before intermediate CPY ops overwrite the zero-copy recurrent state view, avoiding materialization of
// the full [T + 3, C] transposed concat tensor.
static __global__ void gdn_concat_head_and_tail(
        const float * __restrict__ x,
        const float * __restrict__ st,
        float * __restrict__ cc,
        const int C,
        const int T,
        const int tail_from,
        const int n_tail) {
    const int idx         = blockIdx.x * blockDim.x + threadIdx.x;
    const int cols_per_ch = 3 + n_tail;
    if (idx >= C * cols_per_ch) {
        return;
    }
    const int c  = idx / cols_per_ch;
    const int k  = idx % cols_per_ch;
    const int T3 = T + 3;
    if (k < 3) {
        cc[(size_t) c * T3 + k] = st[(size_t) c * 3 + k];
    } else {
        const int s = tail_from + (k - 3);
        cc[(size_t) c * T3 + s] = (s < 3) ? st[(size_t) c * 3 + s]
                                          : x[(size_t) (s - 3) * C + c];
    }
}

template <int TC, int TT, bool apply_silu>
__launch_bounds__(TC, 2)
static __global__ void gdn_conv_direct(
        const float * __restrict__ x,
        const float * __restrict__ prefix_src,
        const float * __restrict__ w,
        float * __restrict__ y,
        const int C,
        const int T,
        const int prefix_stride) {
    __shared__ float sh[(TT + 3) * TC];

    const int tid   = threadIdx.x;
    const int c0    = blockIdx.x * TC;
    const int t0    = blockIdx.y * TT;
    const int t_len = min(TT, T - t0);
    if (t_len <= 0) {
        return;
    }

    const int ch = c0 + tid;
    if (t0 == 0) {
        const float * p = prefix_src + (size_t) ch * prefix_stride;
        sh[0 * TC + tid] = p[0];
        sh[1 * TC + tid] = p[1];
        sh[2 * TC + tid] = p[2];
    } else {
        sh[0 * TC + tid] = x[(size_t) (t0 - 3) * C + ch];
        sh[1 * TC + tid] = x[(size_t) (t0 - 2) * C + ch];
        sh[2 * TC + tid] = x[(size_t) (t0 - 1) * C + ch];
    }

    const int n_elem = TC * t_len;
    for (int i = tid; i < n_elem; i += TC) {
        const int c = i & (TC - 1);
        const int t = i >> 8;
        sh[(t + 3) * TC + c] = x[(size_t) (t0 + t) * C + (c0 + c)];
    }
    __syncthreads();

    const float * wc = w + (size_t) ch * 4;
    const float w0 = wc[0];
    const float w1 = wc[1];
    const float w2 = wc[2];
    const float w3 = wc[3];

    for (int t = 0; t < t_len; ++t) {
        const float s = sh[(t + 0) * TC + tid] * w0
                      + sh[(t + 1) * TC + tid] * w1
                      + sh[(t + 2) * TC + tid] * w2
                      + sh[(t + 3) * TC + tid] * w3;
        y[(size_t) (t0 + t) * C + ch] = apply_silu ? ggml_cuda_op_silu_single(s) : s;
    }
}

static bool gdn_ranges_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b || !a->data || !b->data) {
        return false;
    }
    const uintptr_t a0 = (uintptr_t) a->data;
    const uintptr_t a1 = a0 + ggml_nbytes(a);
    const uintptr_t b0 = (uintptr_t) b->data;
    const uintptr_t b1 = b0 + ggml_nbytes(b);
    return a0 < b1 && b0 < a1;
}

bool ggml_cuda_gdn_conv_match_at_concat(const ggml_cgraph * cgraph, int concat_idx, ggml_cuda_gdn_conv_match & out) {
    static const bool disable = getenv("GGML_CUDA_DISABLE_GDN_CONV_DIRECT") != nullptr;
    if (disable) {
        return false;
    }
    if (concat_idx < 0 || concat_idx >= cgraph->n_nodes) {
        return false;
    }

    ggml_tensor * cc = cgraph->nodes[concat_idx];
    if (cc->op != GGML_OP_CONCAT || cc->type != GGML_TYPE_F32) {
        return false;
    }
    if (ggml_get_op_params_i32(cc, 0) != 0) {
        return false;
    }

    const ggml_tensor * st = cc->src[0];
    const ggml_tensor * tr = cc->src[1];
    if (!st || !tr || st->type != GGML_TYPE_F32 || tr->type != GGML_TYPE_F32) {
        return false;
    }
    if (tr->op != GGML_OP_TRANSPOSE || !tr->src[0]) {
        return false;
    }

    const ggml_tensor * x = tr->src[0];
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || !ggml_is_contiguous(st) || !ggml_is_contiguous(cc)) {
        return false;
    }
    if (st->ne[2] != 1 || st->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1) {
        return false;
    }

    const int64_t C = x->ne[0];
    const int64_t T = x->ne[1];
    if (C <= 0 || C % 256 != 0 || T < 64) {
        return false;
    }
    if (st->ne[0] != 3 || st->ne[1] != C) {
        return false;
    }
    if (tr->ne[0] != T || tr->ne[1] != C) {
        return false;
    }
    if (cc->ne[0] != T + 3 || cc->ne[1] != C) {
        return false;
    }

    int     conv_idx  = -1;
    int64_t tail_from = T + 3;
    int     cc_uses   = 1; // SSM_CONV itself
    const int limit   = std::min(cgraph->n_nodes, concat_idx + 16);

    for (int j = concat_idx + 1; j < limit; ++j) {
        ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_SSM_CONV && n->src[0] == cc) {
            conv_idx = j;
            break;
        }
        if (n->op == GGML_OP_VIEW && n->view_src == cc) {
            if (n->ne[1] != C || n->ne[2] != 1 || n->ne[3] != 1) {
                return false;
            }
            if (n->nb[0] != sizeof(float) || n->nb[1] != cc->nb[1]) {
                return false;
            }
            const size_t byte_off = n->view_offs;
            if (byte_off % sizeof(float) != 0) {
                return false;
            }
            const int64_t col0 = (int64_t) (byte_off / sizeof(float));
            if (col0 < 3 || col0 + n->ne[0] > T + 3) {
                return false;
            }
            if (ggml_node_get_use_count(cgraph, j) != 1 || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                return false;
            }
            if (col0 < tail_from) {
                tail_from = col0;
            }
            ++cc_uses;
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] == cc) {
                return false;
            }
        }
        if (n->view_src == cc) {
            return false;
        }
    }

    if (conv_idx < 0 || conv_idx + 1 >= cgraph->n_nodes) {
        return false;
    }
    if (!ggml_node_has_n_uses(cgraph, concat_idx, cc_uses)) {
        return false;
    }

    if (conv_idx != concat_idx + 1) {
        if (cgraph->nodes[concat_idx + 1]->op != GGML_OP_VIEW || cgraph->nodes[concat_idx + 1]->view_src != cc) {
            return false;
        }
    }

    // Verify every tail VIEW of cc is consumed exclusively by a CPY node before conv_idx.
    for (int j = concat_idx + 1; j < conv_idx; ++j) {
        ggml_tensor * v = cgraph->nodes[j];
        if (v->op == GGML_OP_VIEW && v->view_src == cc) {
            bool consumed_before_conv = false;
            for (int k = j + 1; k < conv_idx; ++k) {
                ggml_tensor * cpy = cgraph->nodes[k];
                if (cpy->op == GGML_OP_CPY && cpy->src[0] == v) {
                    consumed_before_conv = true;
                    break;
                }
            }
            if (!consumed_before_conv) {
                return false;
            }
        }
    }

    if (!ggml_can_fuse(cgraph, conv_idx, { GGML_OP_SSM_CONV, GGML_OP_UNARY })) {
        return false;
    }

    ggml_tensor * conv = cgraph->nodes[conv_idx];
    ggml_tensor * silu = cgraph->nodes[conv_idx + 1];
    if (ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU) {
        return false;
    }

    const ggml_tensor * w = conv->src[1];
    if (!w || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w)) {
        return false;
    }
    if (w->ne[0] != 4 || w->ne[1] != C || w->ne[2] != 1 || w->ne[3] != 1) {
        return false;
    }
    if (conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32 || !ggml_is_contiguous(silu)) {
        return false;
    }
    if (silu->ne[0] != C || silu->ne[1] != T || silu->ne[2] != 1 || silu->ne[3] != 1) {
        return false;
    }

    // When buffers are allocated, verify no memory aliasing between inputs (x, w, cc) and outputs.
    if (x->data != nullptr && silu->data != nullptr) {
        if (gdn_ranges_overlap(x, cc) || gdn_ranges_overlap(x, silu) ||
            gdn_ranges_overlap(w, cc) || gdn_ranges_overlap(w, silu) ||
            gdn_ranges_overlap(cc, silu)) {
            return false;
        }
        for (int j = concat_idx + 1; j < conv_idx; ++j) {
            ggml_tensor * n = cgraph->nodes[j];
            if (n->op == GGML_OP_CPY) {
                if (gdn_ranges_overlap(x, n) || gdn_ranges_overlap(w, n) || gdn_ranges_overlap(cc, n)) {
                    return false;
                }
            }
        }
    }

    out.x          = x;
    out.state      = st;
    out.w          = w;
    out.concat     = cc;
    out.conv_out   = silu;
    out.concat_idx = concat_idx;
    out.conv_idx   = conv_idx;
    out.C          = C;
    out.T          = T;
    out.tail_from  = tail_from;
    return true;
}

int ggml_cuda_gdn_conv_try_fuse(ggml_backend_cuda_context * ctx, const ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor * node = cgraph->nodes[node_idx];

    if (node->op == GGML_OP_CONCAT) {
        ggml_cuda_gdn_conv_match gm;
        if (!ggml_cuda_gdn_conv_match_at_concat(cgraph, node_idx, gm)) {
            return 0;
        }
        cudaStream_t stream = ctx->stream();
        const int    C      = (int) gm.C;
        const int    T      = (int) gm.T;

        if (gm.conv_idx == node_idx + 1) {
            constexpr int TC = 256;
            constexpr int TT = 32;
            const dim3 grid((C + TC - 1) / TC, (T + TT - 1) / TT, 1);
            gdn_conv_direct<TC, TT, true><<<grid, TC, 0, stream>>>(
                (const float *) gm.x->data,
                (const float *) gm.state->data,
                (const float *) gm.w->data,
                (float *)       gm.conv_out->data,
                C, T, 3);
            return 2;
        }

        const int n_tail = std::max(0, (T + 3) - (int) gm.tail_from);
        const int total  = C * (3 + n_tail);
        gdn_concat_head_and_tail<<<(total + 255) / 256, 256, 0, stream>>>(
            (const float *) gm.x->data,
            (const float *) gm.state->data,
            (float *)       gm.concat->data,
            C, T, (int) gm.tail_from, n_tail);
        return 1;
    }

    if (node->op == GGML_OP_SSM_CONV) {
        const ggml_tensor * cc = node->src[0];
        if (!cc || cc->op != GGML_OP_CONCAT) {
            return 0;
        }
        int concat_idx = -1;
        const int lo = std::max(0, node_idx - 16);
        for (int j = node_idx - 1; j >= lo; --j) {
            if (cgraph->nodes[j] == cc) {
                concat_idx = j;
                break;
            }
        }
        ggml_cuda_gdn_conv_match gm;
        if (concat_idx < 0 || !ggml_cuda_gdn_conv_match_at_concat(cgraph, concat_idx, gm) || gm.conv_idx != node_idx) {
            return 0;
        }

        cudaStream_t  stream = ctx->stream();
        constexpr int TC     = 256;
        constexpr int TT     = 32;
        const int     C      = (int) gm.C;
        const int     T      = (int) gm.T;
        const dim3    grid((C + TC - 1) / TC, (T + TT - 1) / TT, 1);

        gdn_conv_direct<TC, TT, true><<<grid, TC, 0, stream>>>(
            (const float *) gm.x->data,
            (const float *) gm.concat->data,
            (const float *) gm.w->data,
            (float *)       gm.conv_out->data,
            C, T, T + 3);
        return 1;
    }

    return 0;
}

