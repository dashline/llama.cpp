#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

struct ggml_cuda_gdn_conv_match {
    const ggml_tensor * x;          // [C, T, 1, 1] F32 contiguous (pre-transpose QKV)
    const ggml_tensor * state;      // [3, C, 1, 1] F32 contiguous
    const ggml_tensor * w;          // [4, C, 1, 1] F32 contiguous
    ggml_tensor *       concat;     // [T + 3, C, 1, 1] F32 contiguous
    ggml_tensor *       conv_out;   // [C, T, 1, 1] F32 contiguous (SiLU output)
    int                 concat_idx;
    int                 conv_idx;
    int64_t             C;
    int64_t             T;
    int64_t             tail_from;  // earliest concat column read by a tail VIEW (T + 3 if none)
};

bool ggml_cuda_gdn_conv_match_at_concat(const ggml_cgraph * cgraph, int concat_idx, ggml_cuda_gdn_conv_match & out);
int  ggml_cuda_gdn_conv_try_fuse(ggml_backend_cuda_context * ctx, const ggml_cgraph * cgraph, int node_idx);

