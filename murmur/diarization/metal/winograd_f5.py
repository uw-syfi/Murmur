"""FP32 F(5,3) WeSpeaker convolutions using a staged batched channel GEMM."""

import mlx.core as mx

from . import f5_fused_output


def pack_weights(filters):
    g = mx.array([[-10, -10, -10], [10, -10, 10], [1, 2, 4], [-1, 2, -4],
                  [16, 8, 4], [-16, 8, -4], [0, 0, 45]], dtype=mx.float32) / 45
    spatial = filters.transpose(0, 3, 1, 2)
    transformed = (g @ spatial) @ g.T
    return mx.contiguous(transformed.transpose(2, 3, 1, 0).flatten(0, 1))


_input = mx.fast.metal_kernel(
    name="wespeaker_f5_input", input_names=["x"], output_names=["v"],
    header="""
        // Cook-Toom evaluation points +1,-1,+2,-2,+1/2,-1/2,infinity.
        inline void f5_input(thread const float* x, thread float* y) {
            float odd = x[1] - 4.25f*x[3] + x[5];
            float even = x[0] - 4.25f*x[2] + x[4];
            y[0] = odd + even; y[1] = odd - even;
            odd = 0.25f*x[1] - 1.25f*x[3] + x[5];
            even = 0.5f*x[0] - 2.5f*x[2] + 2*x[4];
            y[2] = odd + even; y[3] = odd - even;
            odd = 4*x[1] - 5*x[3] + x[5];
            even = 2*x[0] - 2.5f*x[2] + 0.5f*x[4];
            y[4] = odd + even; y[5] = odd - even;
            y[6] = (x[6]-x[0]) + 5.25f*(x[2]-x[4]);
        }
    """,
    source="""
        uint id = thread_position_in_grid.x;
        uint tile = id / C, c = id % C;
        constexpr int cols = (W+4)/5, rows = (H+4)/5;
        int ox = int(tile%cols)*5-1, oy = int((tile/cols)%rows)*5-1;
        int b = tile/(rows*cols);
        float horizontal[7][7];
        for (int i=0; i<7; ++i) {
            float raw[7];
            for (int j=0; j<7; ++j) {
                int xx = ox+j, yy = oy+i;
                raw[j] = xx>=0 && xx<W && yy>=0 && yy<H ?
                    x[((b*H+yy)*W+xx)*C+c] : 0.0f;
            }
            f5_input(raw,horizontal[i]);
        }
        for (int j=0; j<7; ++j) {
            float raw[7], transformed[7];
            for (int i=0; i<7; ++i) raw[i]=horizontal[i][j];
            f5_input(raw,transformed);
            for (int i=0; i<7; ++i) v[(i*7+j)*Tiles*C+id]=transformed[i];
        }
    """,
)


_output = mx.fast.metal_kernel(
    name="wespeaker_f5_output", input_names=["products", "bias", "skip"], output_names=["y"],
    header="""
        inline void f5_output(thread const float* x, thread float* y) {
            float s1=x[0]+x[1], d1=x[0]-x[1], s2=x[2]+x[3], d2=x[2]-x[3];
            float sh=x[4]+x[5], dh=x[4]-x[5];
            y[0]=s1+s2+sh;
            y[1]=d1+2*d2+0.5f*dh;
            y[2]=s1+4*s2+0.25f*sh;
            y[3]=d1+8*d2+0.125f*dh;
            y[4]=s1+16*s2+0.0625f*sh+x[6];
        }
    """,
    source="""
        uint id = thread_position_in_grid.x;
        int tile=id/C, c=id%C;
        constexpr int cols=(W+4)/5, rows=(H+4)/5;
        int ox=(tile%cols)*5, oy=(tile/cols%rows)*5, b=tile/(rows*cols);
        float horizontal[7][5];
        for (int i=0; i<7; ++i) {
            float row[7];
            for (int j=0; j<7; ++j) row[j]=products[(i*7+j)*Tiles*C+id];
            f5_output(row,horizontal[i]);
        }
        for (int j=0; j<5 && ((C==128 && W%5==0) || ox+j<W); ++j) {
            float col[7], values[5];
            for (int i=0; i<7; ++i) col[i]=horizontal[i][j];
            f5_output(col,values);
            for (int i=0; i<5 && ((C==128 && H%5==0) || oy+i<H); ++i) {
                int dst=((b*H+oy+i)*W+ox+j)*C+c;
                float value=values[i]+bias[c];
                if (Residual) value+=skip[dst];
                y[dst]=metal::max(0.0f,value);
            }
        }
    """,
)


def convolution(x, packed, bias, skip=None):
    batch, h, w, c = x.shape
    tiles = batch * ((h + 4) // 5) * ((w + 4) // 5)
    template = [("H", h), ("W", w), ("Tiles", tiles), ("C", c)]
    launch = dict(grid=(tiles*c, 1, 1), threadgroup=(256, 1, 1), output_dtypes=[mx.float32])
    transformed, = _input(inputs=[x], template=template, output_shapes=[(49, tiles, c)], **launch)
    if c == 128 and tiles % 8 == 0:
        return f5_fused_output.output(transformed, packed, bias, skip, x.shape)
    products = transformed @ packed
    result, = _output(inputs=[products, bias, bias if skip is None else skip],
                      template=[*template, ("Residual", skip is not None)],
                      output_shapes=[x.shape], **launch)
    return result
