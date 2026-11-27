"""Time-axis operations for concatenated variable-length TitaNet excerpts on Metal."""

import mlx.core as mx
import numpy as np


# Each output frame uses only taps that fall inside the same excerpt.
_depthwise = mx.fast.metal_kernel(
    name="titanet_packed_depthwise",
    input_names=["frames", "weights", "owners", "offsets"],
    output_names=["out"],
    source="""
        int channel = thread_position_in_grid.x;
        int frame = thread_position_in_grid.y;
        if (channel >= frames_shape[1] || frame >= frames_shape[0]) return;
        int clip = owners[frame];
        int width = weights_shape[1];
        int first = frame - width / 2;
        int begin = metal::max(0, offsets[clip] - first);
        int end = metal::min(width, offsets[clip + 1] - first);
        float value = 0;
        for (int tap = begin; tap < end; ++tap) {
            value += frames[(first + tap) * frames_shape[1] + channel]
                     * weights[channel * width + tap];
        }
        out[frame * frames_shape[1] + channel] = value;
    """,
)

# One group reduces 32 adjacent channels of one excerpt. Eight threads per
# channel share the frame reads and produce a single sum or maximum.
_reduce = mx.fast.metal_kernel(
    name="titanet_packed_reduce",
    input_names=["frames", "offsets"],
    output_names=["out"],
    source="""
        uint channel = thread_position_in_grid.x;
        uint clip = threadgroup_position_in_grid.y;
        uint lane = thread_position_in_threadgroup.x;
        uint stripe = thread_position_in_threadgroup.y;
        float value = IS_MAX ? -INFINITY : 0.0f;
        if (channel < uint(frames_shape[1])) {
            for (int t = offsets[clip] + stripe; t < offsets[clip + 1]; t += 8) {
                float x = frames[t * frames_shape[1] + channel];
                value = IS_MAX ? metal::max(value, x) : value + x;
            }
        }
        threadgroup float partial[8][32];
        partial[stripe][lane] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (stripe == 0 && channel < uint(frames_shape[1])) {
            for (uint i = 1; i < 8; ++i) {
                float x = partial[i][lane];
                value = IS_MAX ? metal::max(value, x) : value + x;
            }
            out[clip * frames_shape[1] + channel] = value;
        }
    """,
)


class PackedFrames:
    def __init__(self, lengths):
        offsets = np.empty(len(lengths) + 1, dtype=np.int32)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        self.offsets = mx.array(offsets)
        self.lengths = mx.array(lengths[:, None])
        self.owners = mx.array(np.repeat(np.arange(len(lengths), dtype=np.int32), lengths))

    def convolve(self, x, weight):
        channels = x.shape[1]
        return _depthwise(
            inputs=[x, weight, self.owners, self.offsets],
            grid=(channels, x.shape[0], 1), threadgroup=(32, 8, 1),
            output_shapes=[x.shape], output_dtypes=[x.dtype],
        )[0]

    def reduce(self, x, maximum=False):
        clips, channels = self.lengths.shape[0], x.shape[1]
        return _reduce(
            inputs=[x, self.offsets], template=[("IS_MAX", maximum)],
            grid=(channels, clips * 8, 1), threadgroup=(32, 8, 1),
            output_shapes=[(clips, channels)], output_dtypes=[x.dtype],
        )[0]

    def expand(self, x):
        return x[self.owners]

    def average(self, x, weights=None):
        if weights is None:
            return self.reduce(x) / self.lengths
        return self.reduce(x * weights)

    def softmax(self, x):
        numerator = mx.exp(x - self.expand(self.reduce(x, maximum=True)))
        return numerator / self.expand(self.reduce(numerator))
