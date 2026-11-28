"""One-dispatch FP32 VibeVoice residual blocks at widths 32, 64 and 128."""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from .vibevoice_block_math import BLOCK_MATH
from .vibevoice_ffn import _ERF, _FFN_PROJECTIONS, _projection_weight
from .vibevoice_tiled_mixer import _epsilons


@lru_cache(maxsize=1)
def _kernel():
    row_reduction = """
        template <uint WIDTH>
        inline float vv_row_inverse(thread const float* values, uint lane, float eps) {
            constexpr uint owned = WIDTH / 32;
            float squares = 0.f;
            for (uint j = 0; j < 4; ++j) {
                float v = values[j % owned];
                if constexpr (WIDTH < 128) {
                    // Gather the original four consecutive channels per reduction lane.
                    uint reduction_channel = 4 * lane + j;
                    ushort owner = ushort((reduction_channel / owned) % 32);
                    v = simd_shuffle(v, owner);
                }
                if (lane < WIDTH / 4) squares = vv_square_sum(squares, v);
            }
            return vv_inverse_rms(squares, WIDTH, eps);
        }
    """
    return mx.fast.metal_kernel(
        name="murmur_vibevoice_full_block",
        input_names=["x", "pre_weight", "taps", "mix_bias", "mix_scale", "post_weight",
                     "eps", "up", "up_bias", "down", "down_bias", "scale"],
        output_names=["y"],
        header=_ERF + BLOCK_MATH + row_reduction,
        source="""
            // Use every lane for taps; restore MLX channel order at each RMS reduction.
            constexpr uint CHANNELS_PER_LANE = C / 32;
            constexpr uint SLOTS = (ROWS + 15) / 16;
            uint tid = thread_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            uint row_tile = (sg / (C / 8)) * 8;
            uint col_tile = (sg % (C / 8)) * 32;
            int first = int(threadgroup_position_in_grid.x * ROWS);
            int length = x_shape[1];
            size_t base = size_t(threadgroup_position_in_grid.y) * length * C;
            threadgroup float input_tile[ROWS * C];
            threadgroup float hidden[ROWS * H];
            float residual[SLOTS][CHANNELS_PER_LANE];

            // Append the six left-halo rows to the work queue. Each SIMD group
            // owns all channels of a row, including its RMS reduction.
            for (uint task = sg; task < ROWS + 6; task += 16) {
                int local_row = task < ROWS ? int(task) : int(task - ROWS) - 6;
                int time = first + local_row;
                float v[CHANNELS_PER_LANE];
                for (uint j = 0; j < CHANNELS_PER_LANE; ++j) {
                    uint c = lane * CHANNELS_PER_LANE + j;
                    v[j] = time >= 0 && time < length ? x[base + size_t(time) * C + c] : 0.f;
                    if (task < ROWS) residual[task / 16][j] = v[j];
                }
                float inv = vv_row_inverse<C>(v, lane, eps[0]);
                for (uint j = 0; j < CHANNELS_PER_LANE; ++j) {
                    uint c = lane * CHANNELS_PER_LANE + j;
                    hidden[(local_row + 6) * C + c] = (v[j] * inv) * float(pre_weight[c]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint row = sg; row < ROWS; row += 16) {
                for (uint j = 0; j < CHANNELS_PER_LANE; ++j) {
                    uint c = lane * CHANNELS_PER_LANE + j;
                    float acc = 0.f;
                    for (uint k = 0; k < 7; ++k) {
                        float product = hidden[(row + k) * C + c] * float(taps[c * 7 + k]);
                        acc = acc + product;
                    }
                    float mixed = vv_add_scaled(residual[row / 16][j], acc, mix_bias[c], mix_scale[c]);
                    residual[row / 16][j] = mixed;
                }
                float inv = vv_row_inverse<C>(residual[row / 16], lane, eps[1]);
                for (uint j = 0; j < CHANNELS_PER_LANE; ++j) {
                    uint c = lane * CHANNELS_PER_LANE + j;
                    input_tile[row * C + c] = (residual[row / 16][j] * inv) * float(post_weight[c]);
                }
            }
            // All halo reads and all FFN input writes must finish before the GEMMs.
            threadgroup_barrier(mem_flags::mem_threadgroup);
        """ + _FFN_PROJECTIONS + """
            for (uint row = sg; row < ROWS; row += 16) {
                int time = first + int(row);
                if (time < length) {
                    for (uint j = 0; j < CHANNELS_PER_LANE; ++j) {
                        uint c = lane * CHANNELS_PER_LANE + j;
                        size_t pos = base + size_t(time) * C + c;
                        y[pos] = vv_add_scaled(residual[row / 16][j], input_tile[row * C + c],
                                               down_bias[c], scale[c]);
                    }
                }
            }
        """,
        compile_options={"math_mode": "safe"},
    )


def try_full_block(block, conv, x):
    channels = x.shape[-1]
    if x.dtype != mx.float32 or channels not in (32, 64, 128):
        return None
    if type(block.norm) is not nn.RMSNorm or type(block.ffn_norm) is not nn.RMSNorm:
        return None
    up, down = block.ffn.linear1, block.ffn.linear2
    hidden = 4 * channels
    if (conv.weight.shape != (channels, 7, 1) or up.weight.shape != (hidden, channels)
            or down.weight.shape != (channels, hidden)):
        return None
    vectors = (block.norm.weight, getattr(conv, "bias", None), block.gamma,
               block.ffn_norm.weight, getattr(down, "bias", None), block.ffn_gamma)
    if any(v is None or v.shape != (channels,) for v in vectors):
        return None
    if getattr(up, "bias", None) is None or up.bias.shape != (hidden,):
        return None
    weights = (conv.weight, up.weight, down.weight, up.bias, *vectors)
    if any(w.dtype not in (mx.float32, mx.bfloat16, mx.float16) for w in weights):
        return None
    batch, length, _ = x.shape
    rows = 1024 // channels
    return _kernel()(
        inputs=[x, block.norm.weight, conv.weight, conv.bias, block.gamma,
                block.ffn_norm.weight, _epsilons(block.norm.eps, block.ffn_norm.eps),
                _projection_weight(up.weight), up.bias, _projection_weight(down.weight),
                down.bias, block.ffn_gamma],
        template=[("C", channels), ("ROWS", rows), ("H", hidden)],
        grid=(((length + rows - 1) // rows) * 512, batch, 1), threadgroup=(512, 1, 1),
        output_shapes=[x.shape], output_dtypes=[x.dtype],
    )[0]
