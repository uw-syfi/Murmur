"""Channel GEMM and spatial inverse for pretransformed FP32 F5 tiles."""

import mlx.core as mx


_kernel = mx.fast.metal_kernel(
    name="wespeaker_f5_mma_inverse", input_names=["v", "u", "bias", "skip"], output_names=["y"],
    header="""
        #include <metal_simdgroup_matrix>
        inline void inverse7(thread const float* x, thread float* z) {
            float s1=x[0]+x[1], d1=x[0]-x[1], s2=x[2]+x[3], d2=x[2]-x[3];
            float sh=x[4]+x[5], dh=x[4]-x[5];
            z[0]=s1+s2+sh;
            z[1]=d1+2*d2+0.5f*dh;
            z[2]=s1+4*s2+0.25f*sh;
            z[3]=d1+8*d2+0.125f*dh;
            z[4]=s1+16*s2+0.0625f*sh+x[6];
        }
    """,
    source="""
        // Each SIMDgroup owns one of seven transformed rows for a 32-channel panel.
        // All seven horizontal products stay in registers until the first inverse.
        constexpr int Threads = 7*(N/Panel)*32;
        constexpr int Band = 8192/(7*M*N) > 5 ? 5 : 8192/(7*M*N);
        constexpr int TileCols=(W+4)/5, TileRows=(H+4)/5;
        int lid=thread_position_in_threadgroup.x, sg=lid/32;
        int tx0=threadgroup_position_in_grid.x*M;
        int oc0=threadgroup_position_in_grid.y*N;
        int ty=sg/(N/Panel), oc=sg%(N/Panel)*Panel;
        threadgroup float scratch[Band*7*M*N];
        simdgroup_float8x8 acc[7][M/8][Panel/8];
        #pragma clang loop unroll(full)
        for (int coord=0; coord<7; ++coord) {
            #pragma clang loop unroll(full)
            for (int m=0; m<M/8; ++m)
                #pragma clang loop unroll(full)
                for (int n=0; n<Panel/8; ++n)
                    acc[coord][m][n]=simdgroup_float8x8(0.0f);
            int idx=ty*7+coord;
            for (int k=0; k<C; k+=8) {
                simdgroup_float8x8 a[M/8], w[Panel/8];
                #pragma clang loop unroll(full)
                for (int m=0; m<M/8; ++m) {
                    int t=tx0+m*8;
                    // An inactive eight-row fragment can read the last full
                    // fragment. Each row accumulates independently and the
                    // output store is guarded by the true tile count.
                    int row=Tiles%M==0 ? t : metal::min(t,Tiles-8);
                    simdgroup_load(a[m],v+(idx*Tiles+row)*C+k,C);
                }
                #pragma clang loop unroll(full)
                for (int n=0; n<Panel/8; ++n) {
                    simdgroup_load(w[n],u+(idx*C+k)*C+oc0+oc+n*8,C);
                    #pragma clang loop unroll(full)
                    for (int m=0; m<M/8; ++m)
                        simdgroup_multiply_accumulate(acc[coord][m][n],a[m],w[n],acc[coord][m][n]);
                }
            }
        }
        // Corresponding elements in identically shaped fragments have the same
        // distributed layout. Neither inverse needs a lane-to-coordinate map.
        simdgroup_float8x8 reduced[5][M/8][Panel/8];
        #pragma clang loop unroll(full)
        for (int m=0; m<M/8; ++m)
            #pragma clang loop unroll(full)
            for (int n=0; n<Panel/8; ++n)
                #pragma clang loop unroll(full)
                for (int j=0; j<2; ++j) {
                    float vals[7], outs[5];
                    #pragma clang loop unroll(full)
                    for (int q=0; q<7; ++q) vals[q]=acc[q][m][n].thread_elements()[j];
                    inverse7(vals,outs);
                    #pragma clang loop unroll(full)
                    for (int q=0; q<5; ++q) reduced[q][m][n].thread_elements()[j]=outs[q];
                }
        // Store a horizontal band in threadgroup memory for the vertical inverse.
        #pragma clang loop unroll(full)
        for (int band=0; band<5; band+=Band) {
            #pragma clang loop unroll(full)
            for (int q=0; q<Band; ++q)
                if (q+band<5)
                    #pragma clang loop unroll(full)
                    for (int m=0; m<M/8; ++m)
                        #pragma clang loop unroll(full)
                        for (int n=0; n<Panel/8; ++n)
                            simdgroup_store(reduced[band+q][m][n],
                                scratch+((q*7+ty)*M+8*m)*N+oc+8*n,N);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i=lid; i<Band*M*N; i+=Threads) {
                int q=i/(M*N), spatial=tx0+(i/N)%M, c=oc0+i%N;
                if (q+band>=5 || spatial>=Tiles) continue;
                int xx=(spatial%TileCols)*5+band+q;
                int yy=(spatial/TileCols%TileRows)*5;
                int b=spatial/(TileCols*TileRows);
                if (xx>=W) continue;
                float vals[7], outs[5];
                #pragma clang loop unroll(full)
                for (int r=0; r<7; ++r) vals[r]=scratch[(q*7+r)*M*N+i%(M*N)];
                inverse7(vals,outs);
                #pragma clang loop unroll(full)
                for (int r=0; r<5; ++r)
                    if (yy+r<H) {
                        int dst=((b*H+yy+r)*W+xx)*C+c;
                        float value=outs[r]+bias[c];
                        if (Residual) value+=skip[dst];
                        y[dst]=metal::max(0.0f,value);
                    }
            }
            if (band+Band<5) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """,
)


def output(transformed, packed, bias, skip, shape):
    """Compute the C128 tile product, inverse transform and ReLU epilogue."""
    _, height, width, channels = shape
    tiles = transformed.shape[1]
    tile_rows, tile_channels, panel = 16, 64, 32
    threads = 7 * (tile_channels // panel) * 32
    groups = (tiles + tile_rows - 1) // tile_rows
    result, = _kernel(
        inputs=[transformed, packed, bias, bias if skip is None else skip],
        template=[
            ("Tiles", tiles), ("H", height), ("W", width), ("C", channels),
            ("M", tile_rows), ("N", tile_channels), ("Panel", panel),
            ("Residual", skip is not None),
        ],
        grid=(groups * threads, channels // tile_channels, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[shape],
        output_dtypes=[mx.float32],
    )
    return result
