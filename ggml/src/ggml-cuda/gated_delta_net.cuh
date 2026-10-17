#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data        = nullptr; // rollback slot 0
    int64_t slot_stride = 0;       // between rollback slots (0 when K==1)
};

// optional prologue ops fused directly into gated_delta_net_cuda
struct ggml_cuda_gated_delta_net_fusion_args {
    // q / k L2 norm (RMS_NORM + SCALE)
    bool  fuse_qk_norm = false;
    float q_eps        = 0.0f;
    float q_scale      = 1.0f;
    float k_eps        = 0.0f;
    float k_scale      = 1.0f;
    const ggml_tensor * raw_q = nullptr;
    const ggml_tensor * raw_k = nullptr;

    // beta = sigmoid(raw_beta)
    bool fuse_beta_sigmoid = false;
    const ggml_tensor * raw_beta = nullptr;

    // g = softplus(raw_alpha + dt_bias) * ssm_a
    bool fuse_gate = false;
    const ggml_tensor * raw_alpha = nullptr;
    const ggml_tensor * dt_bias   = nullptr;
    const ggml_tensor * ssm_a     = nullptr;

    // optional state snapshot write into recurrent cache
    bool has_cache_fusion = false;
    ggml_cuda_gated_delta_net_fused_cache cache = {};
};

bool ggml_cuda_gdn_mma_supported(int device, const ggml_tensor * dst);

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

void ggml_cuda_op_gated_delta_net_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                        const ggml_cuda_gated_delta_net_fusion_args & fusion);

// Decode-time GDN layer fusion (small n_tokens): one launch covers
//   CONCAT(conv state, x^T) -> tail VIEW -> CPY (new conv state) -> SSM_CONV -> SILU
//   -> q/k/v views -> GDN prologue -> GATED_DELTA_NET -> state CPY
//   -> RMS_NORM -> MUL(norm weight) -> SILU(z) -> MUL                     (rms_norm_gated -> q8_1)
// The conv output is never materialized; the gated norm runs in the last block to finish a
// k-head group (per-head retire counter), so no intermediate launches or memsets are needed.
struct ggml_cuda_gdn_decode_match {
    // conv
    const ggml_tensor * x          = nullptr; // qkv_mixed [C, T, n_seqs], contiguous
    const ggml_tensor * conv_state = nullptr; // [3, C, n_seqs]
    const ggml_tensor * conv_w     = nullptr; // [4, C]
    const ggml_tensor * conv_cpy   = nullptr; // CPY writing the new conv state; src[1] is the cache view
    const ggml_tensor * conv_out   = nullptr; // SILU(SSM_CONV) output (elided)
    // gdn
    ggml_tensor *                          gdn = nullptr;
    ggml_cuda_gated_delta_net_fusion_args  gdn_args; // prologue (q/k norm, gate, beta) + cache snapshot
    // gated norm
    ggml_tensor *       rms_norm = nullptr;
    const ggml_tensor * norm_w   = nullptr;   // [S_v]
    const ggml_tensor * z        = nullptr;   // [S_v, H, T, n_seqs], contiguous
    ggml_tensor *       out      = nullptr;   // final MUL
    int concat_idx = -1;
    int last_idx   = -1;                      // index of `out`
};

// true when the device kernel can run this match (shapes/alignment); call after the graph match
bool ggml_cuda_gdn_decode_supported(const ggml_cuda_gdn_decode_match & m);

void ggml_cuda_op_gdn_decode_fused(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_decode_match & m);
