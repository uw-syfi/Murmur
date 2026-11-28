"""Time-tiled causal VibeVoice mixer and its two RMS normalizations."""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from .block_math import BLOCK_MATH
from .wide_mixer import wide_mixer_kernel


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="murmur_vibevoice_tiled_rms_mixer",
        input_names=["x", "pre_weight", "taps", "bias", "scale", "post_weight", "eps"],
        output_names=["residual", "ffn_input"],
        header=BLOCK_MATH,
        source="""
            constexpr uint rows_per_group = 16;
            // MLX rms_single_row<float, 4> accumulates four consecutive
            // channels per lane and shares lane zero's reduction with the row.
            constexpr uint channels_per_lane = 4;
            uint row = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            uint batch = threadgroup_position_in_grid.y;
            int first = int(threadgroup_position_in_grid.x * rows_per_group);
            int length = x_shape[1];
            size_t base = size_t(batch) * length * C;
            threadgroup float normalized[(rows_per_group + 6) * C];
            float original[channels_per_lane];

            // One SIMD group owns a complete row. Stage the tile plus its left halo.
            for (uint step = 0; step < 2; ++step) {
                uint local_row = row + (step == 0 ? 6 : 0);
                if (step == 1 && row >= 6) break;
                int time = first + int(local_row) - 6;
                float values[channels_per_lane];
                float squares = 0.f;
                for (uint j = 0; j < channels_per_lane; ++j) {
                    uint c = lane * 4 + j;
                    float v = c < C && time >= 0 && time < length ? x[base + size_t(time) * C + c] : 0.f;
                    values[j] = v;
                    if (step == 0) original[j] = v;
                    squares = vv_square_sum(squares, v);
                }
                float inv = vv_inverse_rms(squares, C, eps[0]);
                for (uint j = 0; j < channels_per_lane; ++j) {
                    uint c = lane * 4 + j;
                    if (c < C) normalized[local_row * C + c] = (values[j] * inv) * float(pre_weight[c]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (first + int(row) >= length) return;
            float mixed[channels_per_lane];
            float squares = 0.f;
            for (uint j = 0; j < channels_per_lane; ++j) {
                uint c = lane * 4 + j;
                if (c >= C) continue;
                float acc = 0.f;
                for (uint k = 0; k < 7; ++k) {
                    float product = normalized[(row + k) * C + c] * float(taps[c * 7 + k]);
                    acc = acc + product;
                }
                mixed[j] = vv_add_scaled(original[j], acc, bias[c], scale[c]);
                squares = vv_square_sum(squares, mixed[j]);
            }
            float inv = vv_inverse_rms(squares, C, eps[1]);
            for (uint j = 0; j < channels_per_lane; ++j) {
                uint c = lane * 4 + j;
                if (c >= C) continue;
                size_t index = base + size_t(first + row) * C + c;
                residual[index] = mixed[j];
                ffn_input[index] = (mixed[j] * inv) * float(post_weight[c]);
            }
        """,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=16)
def _epsilons(pre, post):
    return mx.array([pre, post], dtype=mx.float32)


def try_tiled_mixer(block, conv, x):
    channels = x.shape[-1]
    if channels not in (32, 64, 128, 256, 512, 1024, 2048) or x.dtype != mx.float32:
        return None
    if type(block.norm) is not nn.RMSNorm or type(block.ffn_norm) is not nn.RMSNorm:
        return None
    if getattr(conv, "bias", None) is None or block.gamma is None:
        return None
    vectors = (block.norm.weight, conv.bias, block.gamma, block.ffn_norm.weight)
    if any(v.shape != (channels,) for v in vectors):
        return None
    weights = (conv.weight, *vectors)
    if any(w.dtype not in (mx.bfloat16, mx.float16, mx.float32) for w in weights):
        return None
    batch, length, _ = x.shape
    if channels >= 256:
        kernel = wide_mixer_kernel()
        threads = channels // 4
        groups = length
    else:
        kernel = _kernel()
        threads = 512
        groups = (length + 15) // 16
    return kernel(
        inputs=[x, block.norm.weight, conv.weight, conv.bias, block.gamma,
                block.ffn_norm.weight, _epsilons(block.norm.eps, block.ffn_norm.eps)],
        template=[("C", channels)],
        grid=(groups * threads, batch, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype],
    )
