"""Fused small-width VibeVoice FFN for FP32 activations on Metal."""

from functools import lru_cache

import mlx.core as mx

from .block_math import BLOCK_MATH


# Same FP32 erf and expm1 implementations used by MLX GELU.
_ERF = r"""
/*
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Original license copied below:
//  Copyright (c) 2015-2023 Norbert Juffa
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//
//  1. Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/* Compute exponential base e minus 1. Maximum ulp error = 0.997458

   i = rint(a/log(2)), f = a-i*log(2). Then expm1(a) = 2**i * (expm1(f)+1) - 1.
   Compute r = expm1(f). Then expm1(a)= 2 * (0.5 * 2**i * r + 0.5 * 2**i - 0.5).
   With t = 0.5*2**i, expm1(a) = 2*(r * t + t-0.5). However, for best accuracy,
   when i == 1, expm1(a)= 2*(r + 0.5), and when i == 0, expm1(a) = r.

   NOTE: Scale factor b is only applied if i < 0 or i > 1 (should be power of 2)
*/
float expm1f_scaled_unchecked(float a, float b) {
  float f, j, r, s, t, u, v, x, y;
  int i;

  // exp(a) = 2**i * exp(f); i = rintf (a / log(2))
  j = fma(1.442695f, a, 12582912.f); // 0x1.715476p0, 0x1.8p23
  j = j - 12582912.0f; // 0x1.8p23
  i = (int)j;
  f = fma(j, -6.93145752e-1f, a);
  f = fma(j, -1.42860677e-6f, f);

  // approximate r = exp(f)-1 on interval [-log(2)/2, +log(2)/2]
  s = f * f;
  if (a == 0.0f)
    s = a; // ensure -0 is passed through
  // err = 0.997458  ulp1 = 11081805
  r = 1.97350979e-4f; // 0x1.9de000p-13
  r = fma(r, f, 1.39309070e-3f); // 0x1.6d30bcp-10
  r = fma(r, f, 8.33343994e-3f); // 0x1.1111f6p-7
  r = fma(r, f, 4.16668020e-2f); // 0x1.55559ep-5
  r = fma(r, f, 1.66666716e-1f); // 0x1.55555cp-3
  r = fma(r, f, 4.99999970e-1f); // 0x1.fffffep-2
  u = (j == 1) ? (f + 0.5f) : f;
  v = fma(r, s, u);
  s = 0.5f * b;
  t = ldexp(s, i);
  y = t - s;
  x = (t - y) - s; // double-float canonicalization of difference
  r = fma(v, t, x) + y;
  r = r + r;
  if (j == 0)
    r = v;
  if (j == 1)
    r = v + v;
  return r;
}

/* Compute exponential base e minus 1. max ulp err = 0.99746 */
float expm1f(float a) {
  float r;

  r = expm1f_scaled_unchecked(a, 1.0f);
  /* handle severe overflow and underflow */
  if (abs(a - 1.0f) > 88.0f) {
    r = pow(2, a);
    r = fma(r, r, -1.0f);
  }
  return r;
}

/*
 * Approximation to the error function.
 * Based on code from:
 * https://stackoverflow.com/questions/35148198/efficient-faithfully-rounded-implementation-of-error-function-erff#answer-35148199
 */
float erf(float a) {
  float r, s, t, u;
  t = metal::abs(a);
  s = a * a;
  if (t > 0.927734375f) {
    // maximum error 0.99527 ulp
    r = metal::fma(
        -1.72853470e-5f, t, 3.83197126e-4f); // -0x1.220000p-16,0x1.91cfb2p-12
    u = metal::fma(
        -3.88396438e-3f, t, 2.42546219e-2f); // -0x1.fd1438p-9, 0x1.8d6342p-6
    r = metal::fma(r, s, u);
    r = metal::fma(r, t, -1.06777877e-1f); // -0x1.b55cb8p-4
    r = metal::fma(r, t, -6.34846687e-1f); // -0x1.450aa0p-1
    r = metal::fma(r, t, -1.28717512e-1f); // -0x1.079d0cp-3
    r = metal::fma(r, t, -t);
    r = -expm1f(r);
    r = metal::copysign(r, a);
  } else {
    // maximum error 0.98929 ulp
    r = -5.96761703e-4f; // -0x1.38e000p-11
    r = metal::fma(r, s, 4.99119423e-3f); //  0x1.471a58p-8
    r = metal::fma(r, s, -2.67681349e-2f); // -0x1.b691b2p-6
    r = metal::fma(r, s, 1.12819925e-1f); //  0x1.ce1c44p-4
    r = metal::fma(r, s, -3.76125336e-1f); // -0x1.812700p-2
    r = metal::fma(r, s, 1.28379166e-1f); //  0x1.06eba8p-3
    r = metal::fma(r, a, a);
  }
  return r;
}

"""


# Shared-memory input tile -> exact GELU -> projected output in the input tile.
_FFN_PROJECTIONS = """
            simdgroup_float8x8 totals[4];
            for (uint n = 0; n < 4; ++n) {
                totals[n] = make_filled_simdgroup_matrix<float, 8>(0.f);
            }
            for (uint k = 0; k < C; k += 8) {
                simdgroup_float8x8 a;
                simdgroup_load(a, input_tile + row_tile * C + k, C);
                for (uint n = 0; n < 4; ++n) {
                    simdgroup_float8x8 b;
                    uint col = col_tile + n * 8;
                    simdgroup_load(b, up + (k * H + col * 8), 8);
                    simdgroup_multiply_accumulate(totals[n], a, b, totals[n]);
                }
            }
            for (uint n = 0; n < 4; ++n) {
                simdgroup_store(totals[n], hidden + row_tile * H + col_tile + n * 8, H);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < ROWS * H; i += 512) {
                float v = hidden[i] + up_bias[i % H];
                hidden[i] = v * (1.f + erf(v / 1.4142135623730951f)) * 0.5f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // The same group now computes eight output channels for those rows.
            uint out_col = (sg % (C / 8)) * 8;
            simdgroup_float8x8 result = make_filled_simdgroup_matrix<float, 8>(0.f);
            for (uint k = 0; k < H; k += 8) {
                simdgroup_float8x8 a, b;
                simdgroup_load(a, hidden + row_tile * H + k, H);
                simdgroup_load(b, down + (k * C + out_col * 8), 8);
                simdgroup_multiply_accumulate(result, a, b, result);
            }
            simdgroup_store(result, input_tile + row_tile * C + out_col, C);
            threadgroup_barrier(mem_flags::mem_threadgroup);
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="murmur_vibevoice_ffn",
        input_names=["x", "residual", "up", "up_bias", "down", "down_bias", "scale"],
        output_names=["y"],
        header=_ERF + BLOCK_MATH,
        source="""
            // Each SIMD group computes eight rows and 32 hidden channels.
            uint tid = thread_index_in_threadgroup;
            uint sg = simdgroup_index_in_threadgroup;
            uint row0 = threadgroup_position_in_grid.x * ROWS;
            uint rows = x_shape[0] * x_shape[1];
            uint row_tile = (sg / (C / 8)) * 8;
            uint col_tile = (sg % (C / 8)) * 32;
            threadgroup float input_tile[ROWS * C];
            threadgroup float hidden[ROWS * H];
            for (uint i = tid; i < ROWS * C; i += 512) {
                input_tile[i] = row0 + i / C < rows ? x[row0 * C + i] : 0.f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

        """ + _FFN_PROJECTIONS + """
            for (uint i = tid; i < ROWS * C; i += 512) {
                if (row0 + i / C < rows) {
                    uint j = row0 * C + i;
                    y[j] = vv_add_scaled(residual[j], input_tile[i], down_bias[i % C], scale[i % C]);
                }
            }
        """,
        compile_options={"math_mode": "safe"},
    )


def _projection_weight(weight):
    """Lay each reduction/output tile out in the order the SIMD group reads."""
    values = weight.astype(mx.float32)
    outputs, inputs = weight.shape
    tiles = values.reshape(outputs // 8, 8, inputs // 8, 8)
    return mx.contiguous(tiles.transpose(2, 0, 3, 1))


def try_fused_ffn(block, x, residual):
    """Return the fused result for compatible dense FFNs, or None."""
    channels = x.shape[-1]
    if channels not in (32, 64, 128) or x.dtype != mx.float32 or residual.dtype != mx.float32:
        return None
    up = block.ffn.linear1
    down = block.ffn.linear2
    hidden = 4 * channels
    if (up.weight.shape != (hidden, channels) or down.weight.shape != (channels, hidden)
            or getattr(up, "bias", None) is None or getattr(down, "bias", None) is None
            or block.ffn_gamma is None):
        return None
    rows = x.size // channels
    # Keep one 20 KiB tile across widths while retaining 16 SIMD groups.
    tile_rows = 1024 // channels
    return _kernel()(
        inputs=[x, residual, _projection_weight(up.weight), up.bias,
                _projection_weight(down.weight), down.bias, block.ffn_gamma],
        template=[("C", channels), ("H", hidden), ("ROWS", tile_rows)],
        grid=(((rows + tile_rows - 1) // tile_rows) * 512, 1, 1), threadgroup=(512, 1, 1),
        output_shapes=[x.shape], output_dtypes=[mx.float32],
    )[0]
