#include "common.cuh"

// Native f32 GEMM for "thin" weights: dst[ne01 x ne11] = src0[ne01 x ne00]^T-dot src1[ne11 x ne00], with ne01 <= 64.
// Targets the f32 GDN head projections (ssm_alpha / ssm_beta: ne00 = n_embd, ne01 = n_heads) of the Qwen3.5/3.6/3.8
// hybrid models during prefill, where cuBLAS/rocBLAS sgemm is otherwise the only path for ne11 > MMVF_MAX_BATCH_SIZE.

bool ggml_cuda_should_use_mmf32s(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

void ggml_cuda_mul_mat_f32s(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
