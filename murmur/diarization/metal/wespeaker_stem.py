"""FP32 single-channel 3x3 WeSpeaker stem on time-major filterbank features."""

import mlx.core as mx


def pack_weights(filters):
    return mx.contiguous(filters[:, :, :, 0].transpose(1, 2, 0))


_stem = mx.fast.metal_kernel(
    name="wespeaker_single_channel_stem",
    input_names=["features", "weights", "bias"],
    output_names=["result"],
    source="""
        // Eight lanes cover all output channels. Each lane applies the same
        // nine filter vectors to four adjacent time positions.
        uint id = thread_position_in_grid.x;
        uint group = id / 8;
        uint channels = id % 8;
        constexpr uint tiles = (Time + Strip - 1) / Strip;
        uint first = (group % tiles) * Strip;
        uint frequency = (group / tiles) % Freq;
        uint batch = group / (tiles * Freq);
        device const float4* w = reinterpret_cast<device const float4*>(weights) + channels;
        device float4* out = reinterpret_cast<device float4*>(result) + channels;
        float4 value[Strip];
        for (uint k = 0; k < Strip; ++k) value[k] = 0;
        for (int f = 0; f < 3; ++f) {
            int source_f = int(frequency) + f - 1;
            if (source_f < 0 || source_f >= Freq) continue;
            float samples[Strip + 2];
            for (uint j = 0; j < Strip + 2; ++j) {
                int t = int(first + j) - 1;
                samples[j] = t >= 0 && t < Time ?
                    features[(batch * Time + uint(t)) * Freq + uint(source_f)] : 0.0f;
            }
            for (int t = 0; t < 3; ++t) {
                float4 weights4 = w[(f * 3 + t) * 8];
                for (uint k = 0; k < Strip; ++k) {
                    value[k] = fma(float4(samples[k+t]), weights4, value[k]);
                }
            }
        }
        float4 b = reinterpret_cast<device const float4*>(bias)[channels];
        for (uint k = 0; k < Strip && first + k < Time; ++k) {
            out[((batch * Freq + frequency) * Time + first + k) * 8] =
                metal::max(float4(0), value[k] + b);
        }
    """,
)


def convolution(features, packed, bias):
    batch, steps, bands = features.shape
    tiles = (steps + 3) // 4
    return _stem(inputs=[features, packed, bias],
                 template=[("Time", steps), ("Freq", bands), ("Strip", 4)],
                 grid=(batch * bands * tiles * 8, 1, 1), threadgroup=(128, 1, 1),
                 output_shapes=[(batch, bands, steps, 32)], output_dtypes=[mx.float32])[0]
