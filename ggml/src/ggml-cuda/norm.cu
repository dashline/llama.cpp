#include "norm.cuh"
#include "mmvq.cuh"
#include <cstdint>

template <int block_size>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps) {
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    extern __shared__ float2 s_sum2[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;

            float2 mean_var = make_float2(0.0f, 0.0f);

            for (int col = tid; col < ncols; col += block_size) {
                const float xi = xc[col];
                mean_var.x += xi;
                mean_var.y += xi * xi;
            }

            // sum up partial sums
            mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

            const float mean = mean_var.x / ncols;
            const float var = mean_var.y / ncols - mean * mean;
            const float inv_std = rsqrtf(var + eps);

            for (int col = tid; col < ncols; col += block_size) {
                dstc[col] = (xc[col] - mean) * inv_std;
            }

            if constexpr (block_size > WARP_SIZE) {
                // sync is needed as we reuse s_sum2 across block_reduce invocations, see #26385
                __syncthreads();
            }
        }
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start =     blockIdx.x*group_size + threadIdx.x;
    const int end   = min(blockIdx.x*group_size + group_size,  ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum + 32);

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

static __device__ __forceinline__ void quantize_float4_to_q8_1(
        const float4 r, block_q8_1 * y_row, const int i4) {
    const int ib   = i4 / 8;
    const int sub  = i4 % 8;
    const float v0 = r.x;
    const float v1 = r.y;
    const float v2 = r.z;
    const float v3 = r.w;

    float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    float sum  = v0 + v1 + v2 + v3;

    amax = warp_reduce_max<8>(amax);
    sum  = warp_reduce_sum<8>(sum);

    const float d = amax / 127.0f;
    int32_t q4 = 0;
    if (amax != 0.0f) {
        const float id = 127.0f / amax;
        const uint8_t q0 = (uint8_t)(int8_t) roundf(v0 * id);
        const uint8_t q1 = (uint8_t)(int8_t) roundf(v1 * id);
        const uint8_t q2 = (uint8_t)(int8_t) roundf(v2 * id);
        const uint8_t q3 = (uint8_t)(int8_t) roundf(v3 * id);
        q4 = (int32_t)(uint32_t(q0) | (uint32_t(q1) << 8) | (uint32_t(q2) << 16) | (uint32_t(q3) << 24));
    }
    reinterpret_cast<int32_t *>(y_row[ib].qs)[sub] = q4;
    if (sub == 0) {
        y_row[ib].ds = make_half2(d, sum);
    }
}

template <int block_size, int vpt, bool do_vec, bool do_multiply = false, bool do_add = false, bool do_scale = false, bool do_pre_add = false, bool do_q8_1 = false>
static __global__ void rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int     nrows,
                                    const int     nchannels,
                                    const int     nsamples,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float   scale_out            = 1.0f,
                                    const float * pre_add              = nullptr,
                                    float *       pre_add_dst          = nullptr,
                                    block_q8_1 *  q8_dst               = nullptr) {
    ggml_cuda_pdl_lc();
    const int row = block_size == WARP_SIZE ? (int)(blockIdx.x * blockDim.y + threadIdx.y) : (int)blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= nrows) {
        return;
    }

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
    static_assert(!do_scale || !do_multiply, "fusing scale is not supported with multiplying");

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const int64_t row_idx = (sample*nchannels + channel)*nrows + row;
            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
            float       * dstc = dst + row_idx*ncols;
            [[maybe_unused]] const float * pac = do_pre_add ? (pre_add + row_idx*ncols) : nullptr;
            [[maybe_unused]] float * padstc    = (do_pre_add && pre_add_dst) ? (pre_add_dst + row_idx*ncols) : nullptr;
            [[maybe_unused]] block_q8_1 * q8c  = do_q8_1 ? (q8_dst + row_idx*(ncols / QK8_1)) : nullptr;

            [[maybe_unused]] const float * mulc = nullptr;
            if constexpr (do_multiply) {
                const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
                const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
                const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
                mulc = mul + mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
            }

            [[maybe_unused]] const float * addc = nullptr;
            if constexpr (do_add) {
                const int add_row     = fastmodulo(row, add_nrows_packed);
                const int add_channel = fastmodulo(channel, add_nchannels_packed);
                const int add_sample  = fastmodulo(sample, add_nsamples_packed);
                addc = add + add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
            }

            if constexpr (do_vec) {
                const int      ncols4 = ncols / 4;
                const float4 * x4     = (const float4 *) xc;
                const float4 * pa4    = (const float4 *) pac;
                float4 *       padst4 = (float4 *) padstc;
                const float4 * mul4   = (const float4 *) mulc;
                const float4 * add4   = (const float4 *) addc;
                float4 *       dst4   = (float4 *) dstc;

                float4 xs[vpt];
                float  tmp = 0.0f;

#pragma unroll
                for (int k = 0; k < vpt; k++) {
                    const int i = tid + k*block_size;
                    if (i < ncols4) {
                        float4 xv = x4[i];
                        if constexpr (do_pre_add) {
                            const float4 pv = pa4[i];
                            xv.x += pv.x;
                            xv.y += pv.y;
                            xv.z += pv.z;
                            xv.w += pv.w;
                            if (padst4) {
                                padst4[i] = xv;
                            }
                        }
                        xs[k] = xv;
                        tmp += xv.x*xv.x + xv.y*xv.y + xv.z*xv.z + xv.w*xv.w;
                    }
                }
                tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
                const float scale = rsqrtf(tmp / ncols + eps);

#pragma unroll
                for (int k = 0; k < vpt; k++) {
                    const int i = tid + k*block_size;
                    if (i < ncols4) {
                        float4 r = make_float4(scale*xs[k].x, scale*xs[k].y, scale*xs[k].z, scale*xs[k].w);
                        if constexpr (do_multiply) {
                            const float4 m = mul4[i];
                            r.x *= m.x;
                            r.y *= m.y;
                            r.z *= m.z;
                            r.w *= m.w;
                        }
                        if constexpr (do_add) {
                            const float4 a = add4[i];
                            r.x += a.x;
                            r.y += a.y;
                            r.z += a.z;
                            r.w += a.w;
                        }
                        if constexpr (do_scale) {
                            r.x *= scale_out;
                            r.y *= scale_out;
                            r.z *= scale_out;
                            r.w *= scale_out;
                        }
                        dst4[i] = r;
                        if constexpr (do_q8_1) {
                            quantize_float4_to_q8_1(r, q8c, i);
                        }
                    }
                }
            } else {
                float tmp = 0.0f; // partial sum for thread in warp

                for (int col = tid; col < ncols; col += block_size) {
                    float xi = xc[col];
                    if constexpr (do_pre_add) {
                        xi += pac[col];
                        if (padstc) {
                            padstc[col] = xi;
                        }
                    }
                    tmp += xi * xi;
                }

                // sum up partial sums
                tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

                const float mean = tmp / ncols;
                const float scale = rsqrtf(mean + eps);

                for (int col = tid; col < ncols; col += block_size) {
                    const float xi = do_pre_add ? (padstc ? padstc[col] : xc[col] + pac[col]) : xc[col];
                    if constexpr (do_multiply && do_add) {
                        const int mul_col = fastmodulo(col, mul_ncols_packed);
                        const int add_col = fastmodulo(col, add_ncols_packed);
                        dstc[col]         = scale * xi * mulc[mul_col] + addc[add_col];
                    } else if constexpr (do_multiply) {
                        const int mul_col = fastmodulo(col, mul_ncols_packed);
                        dstc[col]         = scale * xi * mulc[mul_col];
                    } else if constexpr (do_scale) {
                        dstc[col] = scale_out * (scale * xi);
                    } else {
                        dstc[col] = scale * xi;
                    }
                }
            }

            if constexpr (block_size > WARP_SIZE) {
                // sync is needed as we reuse s_sum across block_reduce invocations, see #26385
                __syncthreads();
            }
        }
    }
}

template <int block_size, int vpt, bool do_vec, bool is_sigmoid = false, bool do_q8_1 = false>
static __global__ void rms_norm_gated_f32(const float * x,
                                          const float * mul,
                                          const float * gate,
                                          float *       dst,
                                          const int     ncols,
                                          const int     nrows,
                                          const int     nchannels,
                                          const int     nsamples,
                                          const int64_t stride_row,
                                          const int64_t stride_channel,
                                          const int64_t stride_sample,
                                          const int64_t mul_stride_row,
                                          const int64_t mul_stride_channel,
                                          const int64_t mul_stride_sample,
                                          const uint3   mul_ncols_packed,
                                          const uint3   mul_nrows_packed,
                                          const uint3   mul_nchannels_packed,
                                          const uint3   mul_nsamples_packed,
                                          const int64_t gate_stride_row,
                                          const int64_t gate_stride_channel,
                                          const int64_t gate_stride_sample,
                                          const float   eps,
                                          block_q8_1 *  q8_dst = nullptr) {
    ggml_cuda_pdl_lc();
    const int row = block_size == WARP_SIZE ? (int)(blockIdx.x * blockDim.y + threadIdx.y) : (int)blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= nrows) {
        return;
    }

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const int64_t row_idx = (sample*nchannels + channel)*nrows + row;
            const float * xc    = x    + sample*stride_sample      + channel*stride_channel      + row*stride_row;
            const float * gatec = gate + sample*gate_stride_sample + channel*gate_stride_channel + row*gate_stride_row;
            float       * dstc  = dst  + row_idx*ncols;
            [[maybe_unused]] block_q8_1 * q8c = do_q8_1 ? (q8_dst + row_idx*(ncols / QK8_1)) : nullptr;

            const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
            const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
            const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
            const float * mulc = mul + mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;

            if constexpr (do_vec) {
                const int      ncols4 = ncols / 4;
                const float4 * x4     = (const float4 *) xc;
                const float4 * mul4   = (const float4 *) mulc;
                const float4 * gate4  = (const float4 *) gatec;
                float4 *       dst4   = (float4 *) dstc;

                float4 xs[vpt];
                float  tmp = 0.0f;

#pragma unroll
                for (int k = 0; k < vpt; k++) {
                    const int i = tid + k*block_size;
                    if (i < ncols4) {
                        xs[k] = x4[i];
                        tmp += xs[k].x*xs[k].x + xs[k].y*xs[k].y + xs[k].z*xs[k].z + xs[k].w*xs[k].w;
                    }
                }
                tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
                const float scale = rsqrtf(tmp / ncols + eps);

#pragma unroll
                for (int k = 0; k < vpt; k++) {
                    const int i = tid + k*block_size;
                    if (i < ncols4) {
                        const float4 m = mul4[i];
                        const float4 g = gate4[i];
                        const float gx = is_sigmoid ? (1.0f / (1.0f + expf(-g.x))) : (g.x / (1.0f + expf(-g.x)));
                        const float gy = is_sigmoid ? (1.0f / (1.0f + expf(-g.y))) : (g.y / (1.0f + expf(-g.y)));
                        const float gz = is_sigmoid ? (1.0f / (1.0f + expf(-g.z))) : (g.z / (1.0f + expf(-g.z)));
                        const float gw = is_sigmoid ? (1.0f / (1.0f + expf(-g.w))) : (g.w / (1.0f + expf(-g.w)));
                        float4 r = make_float4(
                            scale * xs[k].x * m.x * gx,
                            scale * xs[k].y * m.y * gy,
                            scale * xs[k].z * m.z * gz,
                            scale * xs[k].w * m.w * gw);
                        dst4[i] = r;
                        if constexpr (do_q8_1) {
                            quantize_float4_to_q8_1(r, q8c, i);
                        }
                    }
                }
            } else {
                float tmp = 0.0f;
                for (int col = tid; col < ncols; col += block_size) {
                    const float xi = xc[col];
                    tmp += xi * xi;
                }
                tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
                const float scale = rsqrtf(tmp / ncols + eps);

                for (int col = tid; col < ncols; col += block_size) {
                    const int mul_col = fastmodulo(col, mul_ncols_packed);
                    const float g = gatec[col];
                    const float act_g = is_sigmoid ? (1.0f / (1.0f + expf(-g))) : (g / (1.0f + expf(-g)));
                    dstc[col] = scale * xc[col] * mulc[mul_col] * act_g;
                }
            }

            if constexpr (block_size > WARP_SIZE) {
                __syncthreads();
            }
        }
    }
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size>
static __global__ void l2_norm_f32(
        const float * x, float * dst, const int ncols, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps) {
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;

            float tmp = 0.0f; // partial sum for thread in warp

            for (int col = tid; col < ncols; col += block_size) {
                const float xi = xc[col];
                tmp += xi * xi;
            }

            // sum up partial sums
            tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

            // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
            const float scale = rsqrtf(fmaxf(tmp, eps * eps));

            for (int col = tid; col < ncols; col += block_size) {
                dstc[col] = scale * xc[col];
            }

            if constexpr (block_size > WARP_SIZE) {
                // sync is needed as we reuse s_sum across block_reduce invocations, see #26385
                __syncthreads();
            }
        }
    }
}

static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    }
}

static void group_norm_f32_cuda(
        const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, block_dims.x > WARP_SIZE ? 2 * 32 * sizeof(float): 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

#define RMS_NORM_LAUNCH(BLOCK_SIZE, VPT, VEC) \
    do { \
        const int warps_per_block = (BLOCK_SIZE == WARP_SIZE && nrows >= 4) ? 4 : 1; \
        dim3 grid_dims((nrows + warps_per_block - 1) / warps_per_block, blocks_num.y, blocks_num.z); \
        const dim3 block_dims(BLOCK_SIZE, warps_per_block, 1); \
        ggml_cuda_grid_stall_pad(grid_dims, block_dims); \
        const ggml_cuda_kernel_launch_params launch_params{ grid_dims, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, stream }; \
        ggml_cuda_kernel_launch(rms_norm_f32<BLOCK_SIZE, VPT, VEC, do_multiply, do_add, do_scale, do_pre_add, VEC && do_q8_1>, launch_params, \
            x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, \
            mul, mul_stride_row, mul_stride_channel, mul_stride_sample, \
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, \
            add, add_stride_row, add_stride_channel, add_stride_sample, \
            add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, scale_out, \
            pre_add, pre_add_dst, q8_dst); \
    } while (0)

template <bool do_multiply, bool do_add, bool do_scale, bool do_pre_add = false, bool do_q8_1 = false>
static void rms_norm_f32_dispatch(
        const dim3     blocks_num,
        const ggml_tensor * x_t,   float * dst,   const int ncols, const int nchannels, const int nsamples,
        const int64_t  stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps,
        const ggml_tensor * mul_t, const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const uint3    mul_ncols_packed, const uint3 mul_nrows_packed, const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const ggml_tensor * add_t, const int64_t add_stride_row, const int64_t add_stride_channel, const int64_t add_stride_sample,
        const uint3    add_ncols_packed, const uint3 add_nrows_packed, const uint3 add_nchannels_packed, const uint3 add_nsamples_packed,
        cudaStream_t   stream, const float scale_out = 1.0f,
        const ggml_tensor * pre_add_t = nullptr, float * pre_add_dst = nullptr, block_q8_1 * q8_dst = nullptr) {
    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
    static_assert(!do_scale || !do_multiply, "fusing scale is not supported with multiplying");

    const int nrows   = blocks_num.x;
    const float * x   = (const float *) x_t->data;
    const float * mul = mul_t ? (const float *) mul_t->data : nullptr;
    const float * add = add_t ? (const float *) add_t->data : nullptr;
    const float * pre_add = pre_add_t ? (const float *) pre_add_t->data : nullptr;

    // dst is written at row * ncols, i.e. always at a multiple of ncols, so only its base needs checking here
    bool vec = ncols % 4 == 0 && ncols > 0 && ((uintptr_t) dst & 15) == 0;
    vec = vec && ggml_cuda_is_aligned(x_t, 16);
    if constexpr (do_pre_add) {
        vec = vec && ggml_cuda_is_aligned(pre_add_t, 16)
            && (pre_add_dst == nullptr || ((uintptr_t) pre_add_dst & 15) == 0);
    }
    if constexpr (do_multiply) {
        vec = vec && mul_ncols_packed.z == (uint32_t) ncols
            && ggml_cuda_is_aligned(mul_t, 16);
    }
    if constexpr (do_add) {
        vec = vec && add_ncols_packed.z == (uint32_t) ncols
            && ggml_cuda_is_aligned(add_t, 16);
    }

    const int n4 = ncols / 4;
    if (vec) {
        if (n4 <= 32) {
            RMS_NORM_LAUNCH(32,   1, true);
        } else if (n4 <= 128) {
            RMS_NORM_LAUNCH(128,  1, true);
        } else if (n4 <= 256) {
            RMS_NORM_LAUNCH(256,  1, true);
        } else if (n4 <= 512) {
            RMS_NORM_LAUNCH(512,  1, true);
        } else if (n4 <= 1024) {
            RMS_NORM_LAUNCH(1024, 1, true);
        } else if (n4 <= 4096) {
            RMS_NORM_LAUNCH(1024, 4, true);
        } else {
            RMS_NORM_LAUNCH(1024, 1, false);
        }
    } else if (ncols < 1024) {
        RMS_NORM_LAUNCH(256,  1, false);
    } else {
        RMS_NORM_LAUNCH(1024, 1, false);
    }
}
#undef RMS_NORM_LAUNCH

template <bool do_scale = false>
static void rms_norm_f32_cuda(
        const ggml_tensor * x_t, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream,
        const float scale_out = 1.0f) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    rms_norm_f32_dispatch</*do_multiply*/false, /*do_add*/false, do_scale>(
        blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        stream, scale_out);
}

static void rms_norm_mul_f32_cuda(const ggml_tensor * x_t,
                                  const ggml_tensor * mul_t,
                                  const ggml_tensor * add_t,
                                  float             * dst,
                                  const int      ncols,
                                  const int      nrows,
                                  const int      nchannels,
                                  const int      nsamples,
                                  const int64_t  stride_row,
                                  const int64_t  stride_channel,
                                  const int64_t  stride_sample,
                                  const int64_t  mul_stride_row,
                                  const int64_t  mul_stride_channel,
                                  const int64_t  mul_stride_sample,
                                  const uint32_t mul_ncols,
                                  const uint32_t mul_nrows,
                                  const uint32_t mul_nchannels,
                                  const uint32_t mul_nsamples,
                                  const int64_t  add_stride_row,
                                  const int64_t  add_stride_channel,
                                  const int64_t  add_stride_sample,
                                  const uint32_t add_ncols,
                                  const uint32_t add_nrows,
                                  const uint32_t add_nchannels,
                                  const uint32_t add_nsamples,
                                  const float    eps,
                                  cudaStream_t   stream,
                                  const ggml_tensor * pre_add_t = nullptr,
                                  float *        pre_add_dst = nullptr,
                                  block_q8_1 *   q8_dst = nullptr) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    if (mul_t == nullptr) {
        rms_norm_f32_cuda(x_t, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
    const uint3 zero_packed          = make_uint3(0, 0, 0);

    if (add_t == nullptr) {
        if (pre_add_t != nullptr) {
            if (q8_dst != nullptr) {
                rms_norm_f32_dispatch</*do_multiply*/true, /*do_add*/false, /*do_scale*/false, /*do_pre_add*/true, /*do_q8_1*/true>(
                    blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
                    mul_t, mul_stride_row, mul_stride_channel, mul_stride_sample,
                    mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                    nullptr, 0, 0, 0, zero_packed, zero_packed, zero_packed, zero_packed, stream, 1.0f,
                    pre_add_t, pre_add_dst, q8_dst);
            } else {
                rms_norm_f32_dispatch</*do_multiply*/true, /*do_add*/false, /*do_scale*/false, /*do_pre_add*/true, /*do_q8_1*/false>(
                    blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
                    mul_t, mul_stride_row, mul_stride_channel, mul_stride_sample,
                    mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                    nullptr, 0, 0, 0, zero_packed, zero_packed, zero_packed, zero_packed, stream, 1.0f,
                    pre_add_t, pre_add_dst, nullptr);
            }
        } else if (q8_dst != nullptr) {
            rms_norm_f32_dispatch</*do_multiply*/true, /*do_add*/false, /*do_scale*/false, /*do_pre_add*/false, /*do_q8_1*/true>(
                blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
                mul_t, mul_stride_row, mul_stride_channel, mul_stride_sample,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                nullptr, 0, 0, 0, zero_packed, zero_packed, zero_packed, zero_packed, stream, 1.0f,
                nullptr, nullptr, q8_dst);
        } else {
            rms_norm_f32_dispatch</*do_multiply*/true, /*do_add*/false, /*do_scale*/false>(
                blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
                mul_t, mul_stride_row, mul_stride_channel, mul_stride_sample,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                nullptr, 0, 0, 0, zero_packed, zero_packed, zero_packed, zero_packed, stream);
        }
    } else {
        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        rms_norm_f32_dispatch</*do_multiply*/true, /*do_add*/true, /*do_scale*/false>(
            blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
            mul_t, mul_stride_row, mul_stride_channel, mul_stride_sample,
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
            add_t, add_stride_row, add_stride_channel, add_stride_sample,
            add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, stream);
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

static void l2_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<WARP_SIZE>, launch_params, x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<1024>, launch_params, x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda(src0, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor) {
    const ggml_tensor * src0   = dst->src[0];
    float *             dst_d  = (float *) scale_tensor->data;
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    float scale;
    memcpy(&scale, (const float *) scale_tensor->op_params + 0, sizeof(float));

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda<true>(src0, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream, scale);
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const bool can_vec_q8_1 = ne00 % 4 == 0 && ne00 > 0 && ne00 / 4 <= 4096
        && ((uintptr_t) dst_d & 15) == 0
        && ggml_cuda_is_aligned(rms_norm_src, 16)
        && mul_ncols == ne00
        && ggml_cuda_is_aligned(mul_src, 16);

    block_q8_1 * q8_dst = nullptr;
    if (can_vec_q8_1 && ne00 % MATRIX_ROW_PADDING == 0 && ne01 * ne02 * ne03 <= MMVQ_MAX_BATCH_SIZE && ggml_is_contiguous(mul_tensor)) {
        q8_dst = ctx.alloc_mmvq_q8_1_cache_slot(mul_tensor, ne00, ne00, ne01, ne02, ne03, ne00, ne00 * ne01, ne00 * ne01 * ne02);
    }

    rms_norm_mul_f32_cuda(rms_norm_src, mul_src, nullptr, dst_d,
                          ne00, ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ 0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream, nullptr, nullptr, q8_dst);
}

void ggml_cuda_op_rms_norm_pre_add_fused(ggml_backend_cuda_context & ctx,
                                         ggml_tensor *               pre_add_tensor,
                                         ggml_tensor *               rms_norm_tensor,
                                         ggml_tensor *               mul_tensor) {
    const ggml_tensor * x_src       = pre_add_tensor->src[0];
    const ggml_tensor * pre_add_src = pre_add_tensor->src[1];
    float               eps         = 0.0f;
    memcpy(&eps, rms_norm_tensor->op_params, sizeof(float));

    const ggml_tensor * mul_src = nullptr;
    if (mul_tensor->src[0] == rms_norm_tensor) {
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == rms_norm_tensor) {
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float *      pre_add_dst = (float *) pre_add_tensor->data;
    float *      dst_d       = (float *) mul_tensor->data;
    cudaStream_t stream      = ctx.stream();

    GGML_ASSERT(x_src->type == GGML_TYPE_F32);
    GGML_ASSERT(pre_add_src->type == GGML_TYPE_F32);
    GGML_ASSERT(pre_add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(rms_norm_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = x_src->ne[0];
    const int64_t ne01 = x_src->ne[1];
    const int64_t ne02 = x_src->ne[2];
    const int64_t ne03 = x_src->ne[3];

    const size_t ts0 = ggml_type_size(x_src->type);
    GGML_ASSERT(x_src->nb[0] == ts0);
    const int64_t s01 = x_src->nb[1] / ts0;
    const int64_t s02 = x_src->nb[2] / ts0;
    const int64_t s03 = x_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const bool can_vec_q8_1 = ne00 % 4 == 0 && ne00 > 0 && ne00 / 4 <= 4096
        && ((uintptr_t) dst_d & 15) == 0
        && ggml_cuda_is_aligned(x_src, 16)
        && ggml_cuda_is_aligned(pre_add_src, 16)
        && (pre_add_dst == nullptr || ((uintptr_t) pre_add_dst & 15) == 0)
        && mul_ncols == ne00
        && ggml_cuda_is_aligned(mul_src, 16);

    block_q8_1 * q8_dst = nullptr;
    if (can_vec_q8_1 && ne00 % MATRIX_ROW_PADDING == 0 && ne01 * ne02 * ne03 <= MMVQ_MAX_BATCH_SIZE && ggml_is_contiguous(mul_tensor)) {
        q8_dst = ctx.alloc_mmvq_q8_1_cache_slot(mul_tensor, ne00, ne00, ne01, ne02, ne03, ne00, ne00 * ne01, ne00 * ne01 * ne02);
    }

    rms_norm_mul_f32_cuda(x_src, mul_src, nullptr, dst_d,
                          ne00, ne01, ne02, ne03,
                          s01, s02, s03,
                          mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          0, 0, 0, 0, 0, 0, 0,
                          eps, stream, pre_add_src, pre_add_dst, q8_dst);
}

// Keep a value in a lane register so later ops cannot be rescheduled across it.
// On RDNA the gate is uniform across the row; without this, expf is scalarized
// and mul+add contract into fma. Neither matches the split unary/mul/add kernels.
static __device__ __forceinline__ float shexp_lane(float x) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__)
    asm volatile("" : "+v"(x));
#elif defined(__CUDA_ARCH__)
    asm volatile("" : "+f"(x));
#endif
    return x;
}

static __global__ void shexp_gate_add_f32(
        const float * __restrict__ moe,
        const float * __restrict__ shexp,
        const float * __restrict__ gate_inp,
        float *       __restrict__ dst,
        const int                  ncols,
        const int                  nrows) {
    const int row = (int) blockIdx.y;
    const int col = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= nrows || col >= ncols) {
        return;
    }

    // op_sigmoid, then src0*src1, then src0+src1. Separate statements so this
    // stays a mul and an add, matching k_bin_bcast rather than an fma.
    const float gate = shexp_lane(gate_inp[row]);
    const float sig  = shexp_lane(1.0f / (1.0f + expf(-gate)));
    const int64_t i  = (int64_t) row * ncols + col;
    const float prod = shexp_lane(shexp[i] * sig);
    dst[i] = moe[i] + prod;
}

void ggml_cuda_op_rms_norm_shexp_pre_add_fused(ggml_backend_cuda_context & ctx,
                                               ggml_tensor *               sig_tensor,
                                               ggml_tensor *               shexp_mul_tensor,
                                               ggml_tensor *               add1_tensor,
                                               ggml_tensor *               add2_tensor,
                                               ggml_tensor *               rms_tensor,
                                               ggml_tensor *               mul_tensor) {
    const ggml_tensor * gate_inp = sig_tensor->src[0];
    const ggml_tensor * shexp    = (shexp_mul_tensor->src[0] == sig_tensor) ? shexp_mul_tensor->src[1] : shexp_mul_tensor->src[0];
    const ggml_tensor * moe_out  = (add1_tensor->src[0] == shexp_mul_tensor) ? add1_tensor->src[1] : add1_tensor->src[0];

    cudaStream_t stream = ctx.stream();
    const int ncols = (int) moe_out->ne[0];
    const int nrows = (int) ggml_nrows(moe_out);

    dim3 block_dims(256, 1, 1);
    dim3 grid_dims((ncols + block_dims.x - 1) / block_dims.x, nrows, 1);
    ggml_cuda_grid_stall_pad(grid_dims, block_dims);
    const ggml_cuda_kernel_launch_params launch_params{ grid_dims, block_dims, 0, stream };
    ggml_cuda_kernel_launch(shexp_gate_add_f32, launch_params,
        (const float *) moe_out->data, (const float *) shexp->data, (const float *) gate_inp->data,
        (float *) add1_tensor->data, ncols, nrows);

    // Residual add + RMS + norm weight stay on the existing kernel so the
    // reduction and q8_1 epilogue match the unfused graph bit for bit.
    ggml_cuda_op_rms_norm_pre_add_fused(ctx, add2_tensor, rms_tensor, mul_tensor);
}


#define RMS_NORM_GATED_LAUNCH(BLOCK_SIZE, VPT, VEC, IS_SIGMOID, DO_Q8_1) \
    do { \
        const int warps_per_block = (BLOCK_SIZE == WARP_SIZE && nrows >= 4) ? 4 : 1; \
        dim3 grid_dims((nrows + warps_per_block - 1) / warps_per_block, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX)); \
        const dim3 block_dims(BLOCK_SIZE, warps_per_block, 1); \
        ggml_cuda_grid_stall_pad(grid_dims, block_dims); \
        const ggml_cuda_kernel_launch_params launch_params{ grid_dims, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, stream }; \
        ggml_cuda_kernel_launch(rms_norm_gated_f32<BLOCK_SIZE, VPT, VEC, IS_SIGMOID, VEC && DO_Q8_1>, launch_params, \
            x, mul, gate, dst_d, ncols, nrows, nchannels, nsamples, \
            s01, s02, s03, mul_s01, mul_s02, mul_s03, \
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, \
            gate_s01, gate_s02, gate_s03, eps, q8_dst); \
    } while (0)

template <bool is_sigmoid, bool do_q8_1>
static void rms_norm_gated_dispatch(
        const ggml_tensor * rms_norm_src, const ggml_tensor * mul_src, const ggml_tensor * gate_src,
        float * dst_d, block_q8_1 * q8_dst,
        const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed, const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const int64_t gate_s01, const int64_t gate_s02, const int64_t gate_s03,
        const float eps, const bool vec, cudaStream_t stream) {
    const float * x    = (const float *) rms_norm_src->data;
    const float * mul  = (const float *) mul_src->data;
    const float * gate = (const float *) gate_src->data;
    const int n4 = ncols / 4;
    if (vec) {
        if (n4 <= 32) {
            RMS_NORM_GATED_LAUNCH(32,   1, true, is_sigmoid, do_q8_1);
        } else if (n4 <= 128) {
            RMS_NORM_GATED_LAUNCH(128,  1, true, is_sigmoid, do_q8_1);
        } else if (n4 <= 256) {
            RMS_NORM_GATED_LAUNCH(256,  1, true, is_sigmoid, do_q8_1);
        } else if (n4 <= 512) {
            RMS_NORM_GATED_LAUNCH(512,  1, true, is_sigmoid, do_q8_1);
        } else if (n4 <= 1024) {
            RMS_NORM_GATED_LAUNCH(1024, 1, true, is_sigmoid, do_q8_1);
        } else if (n4 <= 4096) {
            RMS_NORM_GATED_LAUNCH(1024, 4, true, is_sigmoid, do_q8_1);
        } else {
            RMS_NORM_GATED_LAUNCH(1024, 1, false, is_sigmoid, false);
        }
    } else if (ncols < 1024) {
        RMS_NORM_GATED_LAUNCH(256,  1, false, is_sigmoid, false);
    } else {
        RMS_NORM_GATED_LAUNCH(1024, 1, false, is_sigmoid, false);
    }
}
#undef RMS_NORM_GATED_LAUNCH

void ggml_cuda_op_rms_norm_gated(ggml_backend_cuda_context & ctx,
                                 ggml_tensor *               rms_norm_tensor,
                                 ggml_tensor *               mul1_tensor,
                                 ggml_tensor *               act_tensor,
                                 ggml_tensor *               mul2_tensor) {
    const ggml_tensor * rms_norm_src = rms_norm_tensor->src[0];
    float               eps          = 0.0f;
    memcpy(&eps, rms_norm_tensor->op_params, sizeof(float));

    const ggml_tensor * mul_src = (mul1_tensor->src[0] == rms_norm_tensor) ? mul1_tensor->src[1] : mul1_tensor->src[0];
    const ggml_tensor * gate_src = act_tensor->src[0];
    const bool          is_sigmoid = ggml_get_unary_op(act_tensor) == GGML_UNARY_OP_SIGMOID;

    float *      dst_d  = (float *) mul2_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_src->type == GGML_TYPE_F32);
    GGML_ASSERT(gate_src->type == GGML_TYPE_F32);
    GGML_ASSERT(mul2_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int ncols     = rms_norm_src->ne[0];
    const int nrows     = rms_norm_src->ne[1];
    const int nchannels = rms_norm_src->ne[2];
    const int nsamples  = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const size_t ts_gate = ggml_type_size(gate_src->type);
    GGML_ASSERT(gate_src->nb[0] == ts_gate);
    const int64_t gate_s01 = gate_src->nb[1] / ts_gate;
    const int64_t gate_s02 = gate_src->nb[2] / ts_gate;
    const int64_t gate_s03 = gate_src->nb[3] / ts_gate;

    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_src->ne[0]);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_src->ne[1]);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_src->ne[2]);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_src->ne[3]);

    const bool vec = ncols % 4 == 0 && ncols > 0 && ncols / 4 <= 4096
        && ((uintptr_t) dst_d & 15) == 0
        && ggml_cuda_is_aligned(rms_norm_src, 16)
        && mul_ncols_packed.z == (uint32_t) ncols
        && ggml_cuda_is_aligned(mul_src, 16)
        && ggml_cuda_is_aligned(gate_src, 16);

    block_q8_1 * q8_dst = nullptr;
    const int64_t flat_row = (int64_t) ncols * nrows;
    if (vec && ncols % QK8_1 == 0 && flat_row % MATRIX_ROW_PADDING == 0 && nchannels * nsamples <= MMVQ_MAX_BATCH_SIZE && ggml_is_contiguous(mul2_tensor)) {
        q8_dst = ctx.alloc_mmvq_q8_1_cache_slot(mul2_tensor, flat_row, flat_row, nchannels, nsamples, 1, flat_row, flat_row * nchannels, flat_row * nchannels * nsamples);
    }

    if (is_sigmoid) {
        if (q8_dst != nullptr) {
            rms_norm_gated_dispatch<true, true>(rms_norm_src, mul_src, gate_src, dst_d, q8_dst,
                ncols, nrows, nchannels, nsamples, s01, s02, s03, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                gate_s01, gate_s02, gate_s03, eps, vec, stream);
        } else {
            rms_norm_gated_dispatch<true, false>(rms_norm_src, mul_src, gate_src, dst_d, nullptr,
                ncols, nrows, nchannels, nsamples, s01, s02, s03, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                gate_s01, gate_s02, gate_s03, eps, vec, stream);
        }
    } else {
        if (q8_dst != nullptr) {
            rms_norm_gated_dispatch<false, true>(rms_norm_src, mul_src, gate_src, dst_d, q8_dst,
                ncols, nrows, nchannels, nsamples, s01, s02, s03, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                gate_s01, gate_s02, gate_s03, eps, vec, stream);
        } else {
            rms_norm_gated_dispatch<false, false>(rms_norm_src, mul_src, gate_src, dst_d, nullptr,
                ncols, nrows, nchannels, nsamples, s01, s02, s03, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                gate_s01, gate_s02, gate_s03, eps, vec, stream);
        }
    }
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float               eps          = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float *      dst_d  = (float *) add_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    rms_norm_mul_f32_cuda(rms_norm_src, mul_src,add_src,dst_d,
                          ne00,ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    l2_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}
