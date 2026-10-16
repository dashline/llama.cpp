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
