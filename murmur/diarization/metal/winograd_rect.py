"""FP32 Winograd convolution with two frequency and four time outputs per tile."""

import mlx.core as mx


def pack_weights(weights):
    freq = mx.array([[1, 0, 0], [.5, .5, .5], [.5, -.5, .5], [0, 0, 1]],
                    dtype=mx.float32)
    time = mx.array([[.25, 0, 0], [-1/6, -1/6, -1/6], [-1/6, 1/6, -1/6],
                     [1/24, 1/12, 1/6], [1/24, -1/12, 1/6], [0, 0, 1]],
                    dtype=mx.float32)
    kernels = weights.transpose(0, 3, 1, 2)
    transformed = (freq @ kernels) @ time.T
    return mx.contiguous(transformed.transpose(2, 3, 1, 0).flatten(0, 1))


_kernel = mx.fast.metal_kernel(
    name="wespeaker_f2x4_onchip",
    input_names=["x", "u", "bias", "skip"], output_names=["output"],
    header="#include <metal_simdgroup_matrix>\n",
    source="""
    constexpr int Groups = 12, N = C, M = 8, K = 32, TW = (W + 3) / 4;
    int lid = thread_position_in_threadgroup.x, sg = lid / 32;
    int first = threadgroup_position_in_grid.x * M;
    int b = threadgroup_position_in_grid.z;
    threadgroup float shared[24 * M * 32];
    // Each SIMD group accumulates independent transform coordinates. The
    // same scratch holds transformed inputs and then accumulator fragments.
    simdgroup_float8x8 acc[24 / Groups][N / 8];
    #pragma clang loop unroll(full)
    for (int p = 0; p < 24 / Groups; ++p)
        for (int j = 0; j < N / 8; ++j)
            acc[p][j] = simdgroup_float8x8(0.0f);
    for (int k0 = 0; k0 < C; k0 += K) {
        if (lid < M * K) {
            int tile = first + lid / K, c = k0 + lid % K;
            int ox = tile % TW * 4 - 1, oy = tile / TW * 2 - 1;
            // Apply the long transform along time, the short one along frequency.
            float rows[4][6];
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; ++i) {
                float d[6];
                #pragma clang loop unroll(full)
                for (int j = 0; j < 6; ++j) {
                    int y = oy + i, col = ox + j;
                    d[j] = (y >= 0 && y < H && col >= 0 && col < W)
                        ? x[((b * H + y) * W + col) * C + c] : 0.0f;
                }
                // Shared sums for the six horizontal transform coordinates.
                float a = d[4] - 4*d[2], e = d[3] - 4*d[1];
                float h = d[4] - d[2], f = 2*(d[3] - d[1]);
                rows[i][0] = 4*d[0] - 5*d[2] + d[4];
                rows[i][1] = a + e; rows[i][2] = a - e;
                rows[i][3] = h + f; rows[i][4] = h - f;
                rows[i][5] = 4*d[1] - 5*d[3] + d[5];
            }
            #pragma clang loop unroll(full)
            for (int j = 0; j < 6; ++j) {
                shared[(0*6+j)*M*K+lid] = rows[0][j]-rows[2][j];
                shared[(1*6+j)*M*K+lid] = rows[1][j]+rows[2][j];
                shared[(2*6+j)*M*K+lid] = rows[2][j]-rows[1][j];
                shared[(3*6+j)*M*K+lid] = rows[1][j]-rows[3][j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (int p = 0; p < 24 / Groups; ++p) {
            int q = p*Groups + sg;
            for (int k = 0; k < K; k += 8) {
                simdgroup_float8x8 a;
                simdgroup_load(a, shared + q*M*K + k, K);
                #pragma clang loop unroll(full)
                for (int j = 0; j < N / 8; ++j) {
                    simdgroup_float8x8 weight;
                    simdgroup_load(weight, u + (q*C+k0+k)*C+8*j, C);
                    simdgroup_multiply_accumulate(acc[p][j], a, weight, acc[p][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The inverse uses 32 output channels at a time to fit 24 KiB.
    #pragma clang loop unroll(full)
    for (int slab = 0; slab < N / 32; ++slab) {
        #pragma clang loop unroll(full)
        for (int p = 0; p < 24 / Groups; ++p)
            for (int j = 0; j < 4; ++j)
                simdgroup_store(acc[p][slab*4+j], shared+(p*Groups+sg)*M*32+8*j,32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < M*32) {
            int tile = first + lid/32, c = slab*32 + lid%32;
            int ox = tile%TW*4, oy = tile/TW*2;
            float4 v[4];
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; ++i) {
                float d[6];
                #pragma clang loop unroll(full)
                for (int j = 0; j < 6; ++j)
                    d[j] = shared[(i*6+j)*M*32+lid];
                float a = d[1]+d[2], e = d[1]-d[2], h = d[3]+d[4], f = d[3]-d[4];
                v[i] = float4(d[0]+a+h,e+2*f,a+4*h,e+8*f+d[5]);
            }
            float4 block[2] = {v[0]+v[1]+v[2], v[1]-v[2]-v[3]};
            #pragma clang loop unroll(full)
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 4; ++j)
                    if (oy+i < H && ox+j < W) {
                        int idx = ((b*H+oy+i)*W+ox+j)*C+c;
                        float value = block[i][j] + bias[c];
                        if (Residual) value += skip[idx];
                        output[idx] = metal::max(value, 0.0f);
                    }
        }
        if (slab + 1 < N / 32)
            threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    """,
)


def convolution(x, weights, bias, skip=None):
    batch, h, w, channels = x.shape
    tiles = ((h + 1) // 2) * ((w + 3) // 4)
    return _kernel(
        inputs=[x, weights, bias, bias if skip is None else skip],
        template=[("H", h), ("W", w), ("C", channels),
                  ("Residual", skip is not None)],
        grid=(384 * ((tiles + 7) // 8), 1, batch),
        threadgroup=(384, 1, 1),
        output_shapes=[x.shape], output_dtypes=[mx.float32],
    )[0]
