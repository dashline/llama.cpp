#include "mmf32s.cuh"
#include "mmvf.cuh"

#define MMF32S_NT    256   // threads per block
#define MMF32S_MAX_ROWS_DEFAULT 64 // dispatch limit on ne01 (GDN head projections m=16/32); env GGML_CUDA_F32_SMALLM=<m> overrides, 0 disables

// dst[ne01 x ne11] = x[ne01 x ne00] . y[ne11 x ne00]^T for thin x (ne01 <= 64). The work is memory-bound on streaming
// y, so a block owns a ROWS-row slab of x times COLS columns of y, stages both operands transposed in shared memory
// ([k][row] / [k][col], padded so float4 reads over rows/cols are aligned and the staging writes are at most 2-way
// bank conflicted), and every thread accumulates a 4x4 (rows x cols) micro tile over its own slice of k. The next
// k chunk is prefetched into registers while the current one is consumed. The k-slices are reduced through shared
// memory at the end, so the result is deterministic (no atomics).
template <int ROWS, int COLS>
__global__ void __launch_bounds__(MMF32S_NT, 1) mul_mat_f32s_kernel(
        const float * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int ncols_y,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst) {
    constexpr int NT    = MMF32S_NT;
    constexpr int KC    = (COLS <= 8 || ROWS <= 16) ? 256 : 128;  // k chunk per stage; keeps static shared memory <= 48 KB
    constexpr int TILES = ROWS*COLS/16;           // 4x4 micro tiles per block tile
    constexpr int KS    = NT/TILES;               // k-slices (threads sharing one micro tile)
    constexpr int KPT   = KC/KS;                  // k steps per thread per chunk
    constexpr int XP    = ROWS + 4;               // padded [k][row] stride (16B aligned float4 reads, 2-way staging conflicts)
    constexpr int YP    = COLS + 4;
    constexpr int XA    = ROWS*(KC/4)/NT;         // float4 of x staged per thread per chunk
    constexpr int YA    = COLS*(KC/4)/NT;
    static_assert(NT % TILES == 0 && KC % KS == 0 && KPT % 4 == 0, "bad mmf32s tile shape");
    static_assert(XA >= 1 && YA >= 1 && (ROWS*(KC/4)) % NT == 0 && (COLS*(KC/4)) % NT == 0, "bad mmf32s staging shape");

    __shared__ float xs[KC*XP];
    __shared__ float ys[KC*YP];

    const int tid  = threadIdx.x;
    const int tile = tid % TILES;
    const int ks   = tid / TILES;
    const int tr   = tile % (ROWS/4);             // micro tile row group
    const int tc   = tile / (ROWS/4);             // micro tile col group
    const int row0 = blockIdx.y*ROWS;
    const int col0 = blockIdx.x*COLS;

    float4 xa[XA];
    float4 ya[YA];

    auto load_chunk = [&](const int k0) {
        const int kc = min(KC, ncols_x - k0);
#pragma unroll
        for (int a = 0; a < XA; ++a) {
            const int idx = tid + a*NT;
            const int r   = idx / (KC/4);
            const int k4  = (idx % (KC/4))*4;
            const int gr  = row0 + r;
            xa[a] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (gr < nrows_x && k4 < kc) {
                xa[a] = *(const float4 *) (x + (size_t) gr*stride_row_x + k0 + k4);
            }
        }
#pragma unroll
        for (int a = 0; a < YA; ++a) {
            const int idx = tid + a*NT;
            const int c   = idx / (KC/4);
            const int k4  = (idx % (KC/4))*4;
            const int gc  = col0 + c;
            ya[a] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (gc < ncols_y && k4 < kc) {
                ya[a] = *(const float4 *) (y + (size_t) gc*stride_col_y + k0 + k4);
            }
        }
    };
    auto store_chunk = [&]() {
#pragma unroll
        for (int a = 0; a < XA; ++a) {
            const int idx = tid + a*NT;
            const int r   = idx / (KC/4);
            const int k4  = (idx % (KC/4))*4;
            xs[(k4 + 0)*XP + r] = xa[a].x;
            xs[(k4 + 1)*XP + r] = xa[a].y;
            xs[(k4 + 2)*XP + r] = xa[a].z;
            xs[(k4 + 3)*XP + r] = xa[a].w;
        }
#pragma unroll
        for (int a = 0; a < YA; ++a) {
            const int idx = tid + a*NT;
            const int c   = idx / (KC/4);
            const int k4  = (idx % (KC/4))*4;
            ys[(k4 + 0)*YP + c] = ya[a].x;
            ys[(k4 + 1)*YP + c] = ya[a].y;
            ys[(k4 + 2)*YP + c] = ya[a].z;
            ys[(k4 + 3)*YP + c] = ya[a].w;
        }
    };

    float acc[4][4] = {{0.0f}};

    load_chunk(0);
    for (int k0 = 0; k0 < ncols_x; k0 += KC) {
        store_chunk();
        __syncthreads();
        if (k0 + KC < ncols_x) {
            load_chunk(k0 + KC); // in flight while this chunk is consumed
        }

        const float * xk = xs + (ks*KPT)*XP + tr*4;
        const float * yk = ys + (ks*KPT)*YP + tc*4;
#pragma unroll 4
        for (int k = 0; k < KPT; ++k) {
            const float4 a = *(const float4 *) (xk + k*XP);
            const float4 b = *(const float4 *) (yk + k*YP);
            const float av[4] = {a.x, a.y, a.z, a.w};
            const float bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc[j][i] = fmaf(av[i], bv[j], acc[j][i]);
                }
            }
        }
        __syncthreads();
    }

    // reduce the KS k-slices of every micro tile: red[ks][tile][16]
    float * red = xs;
    static_assert(KS*TILES*16 <= KC*XP, "reduction scratch does not fit");
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            red[(ks*TILES + tile)*16 + j*4 + i] = acc[j][i];
        }
    }
    __syncthreads();

    for (int o = tid; o < ROWS*COLS; o += NT) {
        const int r = o % ROWS;
        const int c = o / ROWS;
        const int t = (r/4) + (c/4)*(ROWS/4);
        const int e = (c%4)*4 + (r%4);
        float sum = 0.0f;
#pragma unroll
        for (int s = 0; s < KS; ++s) {
            sum += red[(s*TILES + t)*16 + e];
        }
        const int row = row0 + r;
        const int col = col0 + c;
        if (row < nrows_x && col < ncols_y) {
            dst[(size_t) col*stride_col_dst + row] = sum;
        }
    }
}

bool ggml_cuda_should_use_mmf32s(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const int max_rows = getenv("GGML_CUDA_F32_SMALLM") != nullptr ? atoi(getenv("GGML_CUDA_F32_SMALLM")) : MMF32S_MAX_ROWS_DEFAULT;
    if (max_rows <= 0) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[1] > max_rows || src1->ne[1] <= MMVF_MAX_BATCH_SIZE) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->ne[0] % 4 != 0) {
        return false;
    }
    // float4 loads along k
    if (src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (src0->nb[1] % 16 != 0 || src1->nb[1] % 16 != 0 || dst->nb[1] % sizeof(float) != 0) {
        return false;
    }
    if (!ggml_cuda_is_aligned(src0, 16) || !ggml_cuda_is_aligned(src1, 16)) {
        return false;
    }
    return true;
}

template <int ROWS>
static void launch_mmf32s(const float * x, const float * y, float * dst, const int ncols_x, const int nrows_x, const int ncols_y,
        const int s0, const int s1, const int sd, cudaStream_t stream) {
    const dim3 block_dims(MMF32S_NT, 1, 1);
    const int grid_y = (nrows_x + ROWS - 1)/ROWS;
    // narrow the column tile until there are enough blocks to fill the GPU (64 CUs on gfx1201, 2 blocks each)
    constexpr int min_blocks = 128;
    if ((ncols_y + 15)/16*grid_y >= min_blocks) {
        mul_mat_f32s_kernel<ROWS, 16><<<dim3((ncols_y + 15)/16, grid_y, 1), block_dims, 0, stream>>>(x, y, dst, ncols_x, nrows_x, ncols_y, s0, s1, sd);
    } else if ((ncols_y + 7)/8*grid_y >= min_blocks) {
        mul_mat_f32s_kernel<ROWS, 8><<<dim3((ncols_y + 7)/8, grid_y, 1), block_dims, 0, stream>>>(x, y, dst, ncols_x, nrows_x, ncols_y, s0, s1, sd);
    } else {
        mul_mat_f32s_kernel<ROWS, 4><<<dim3((ncols_y + 3)/4, grid_y, 1), block_dims, 0, stream>>>(x, y, dst, ncols_x, nrows_x, ncols_y, s0, s1, sd);
    }
}

void ggml_cuda_mul_mat_f32s(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_should_use_mmf32s(src0, src1, dst));

    const int ncols_x = src0->ne[0];
    const int nrows_x = src0->ne[1];
    const int ncols_y = src1->ne[1];

    const float * x_d = (const float *) src0->data;
    const float * y_d = (const float *) src1->data;
    float * dst_d = (float *) dst->data;
    const int s0 = src0->nb[1]/sizeof(float);
    const int s1 = src1->nb[1]/sizeof(float);
    const int sd = dst->nb[1]/sizeof(float);

    if (nrows_x <= 16) {
        launch_mmf32s<16>(x_d, y_d, dst_d, ncols_x, nrows_x, ncols_y, s0, s1, sd, ctx.stream());
    } else {
        launch_mmf32s<32>(x_d, y_d, dst_d, ncols_x, nrows_x, ncols_y, s0, s1, sd, ctx.stream());
    }
}
