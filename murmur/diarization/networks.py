"""FP32 MLX inference for the released pyannote 3.1 neural networks."""

import math

import mlx.core as mx
import mlx.nn as nn


# One threadgroup owns a single sequence. All four gates read the previous
# hidden state before any lane replaces it. Input projections use MLX GEMM.
# Precise gate activations avoid NaNs for saturated gates in the released LSTM.
_lstm = mx.fast.metal_kernel(
    name="pyannet_recurrence", input_names=["projected", "recurrent"], output_names=["result"],
    source="""
        uint gate = thread_position_in_threadgroup.x;
        uint batch = threadgroup_position_in_grid.y;
        uint steps = projected_shape[1];
        threadgroup float hidden[128];
        threadgroup float values[512];
        float cell = 0;
        if (gate < 128) hidden[gate] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < steps; ++i) {
            uint t = Reverse ? steps - 1 - i : i;
            float v = projected[(batch * steps + t) * 512 + gate];
            for (uint j = 0; j < 128; ++j)
                v = fma(hidden[j], recurrent[j * 512 + gate], v);
            values[gate] = v;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (gate < 128) {
                float input = 1 / (1 + metal::precise::exp(-values[gate]));
                float forget = 1 / (1 + metal::precise::exp(-values[128 + gate]));
                float output = 1 / (1 + metal::precise::exp(-values[384 + gate]));
                cell = forget * cell + input * metal::precise::tanh(values[256 + gate]);
                hidden[gate] = output * metal::precise::tanh(cell);
                result[(batch * steps + t) * 128 + gate] = hidden[gate];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """,
)


class Segmentation:
    def __init__(self, weights):
        self.w = weights

    def norm(self, x, prefix):
        return ((x - x.mean(axis=1, keepdims=True)) * mx.rsqrt(x.var(axis=1, keepdims=True) + 1e-5)
                * self.w[prefix + ".weight"] + self.w[prefix + ".bias"])

    def direction(self, x, layer, reverse):
        suffix = f"_l{layer}" + ("_reverse" if reverse else "")
        bias = self.w["lstm.bias" + suffix]
        projected = mx.addmm(bias, x, self.w["lstm.weight_ih" + suffix])
        batch, steps, _ = projected.shape
        return _lstm(inputs=[projected, self.w["lstm.weight_hh" + suffix]],
                     template=[("Reverse", reverse)], grid=(512, batch, 1), threadgroup=(512, 1, 1),
                     output_shapes=[(batch, steps, 128)], output_dtypes=[mx.float32])[0]

    def __call__(self, windows):
        x = self.norm(windows[:, :, None], "sincnet.wav_norm1d")
        for block in range(3):
            if block == 0:
                x = mx.abs(mx.conv1d(x, self.w["sincnet.filters"], stride=10))
            else:
                prefix = f"sincnet.conv1d.{block}"
                x = mx.conv1d(x, self.w[prefix + ".weight"]) + self.w[prefix + ".bias"]
            x = nn.leaky_relu(self.norm(nn.MaxPool1d(3, 3)(x), f"sincnet.norm1d.{block}"))
        for layer in range(4):
            x = mx.concatenate([self.direction(x, layer, False), self.direction(x, layer, True)], axis=-1)
        for layer in ("linear.0", "linear.1"):
            x = nn.leaky_relu(mx.addmm(self.w[layer + ".bias"], x, self.w[layer + ".weight"]))
        logits = mx.addmm(self.w["classifier.bias"], x, self.w["classifier.weight"])
        powerset = mx.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                             [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=mx.float32)
        return powerset[logits.argmax(axis=-1)]


class WeSpeaker:
    def __init__(self, weights):
        self.w = weights

        # Kaldi's 80 triangular filters have uniformly spaced vertices in mel space.
        mel = 1127 * mx.log1p(mx.arange(257) * (16000 / 512) / 700)
        lo, hi = (1127 * math.log1p(hz / 700) for hz in (20, 8000))
        step = (hi - lo) / 81
        center = lo + step * mx.arange(1, 81)
        self.mel = mx.maximum(1 - mx.abs(mel[:, None] - center) / step, 0)
        self.window = 0.54 - 0.46 * mx.cos(2 * math.pi * mx.arange(400) / 399)

    def features(self, audio):
        frames = mx.as_strided(audio, (len(audio), (audio.shape[1] - 400) // 160 + 1, 400),
                               strides=(audio.shape[1], 160, 1))
        frames = frames * 32768
        frames = frames - frames.mean(axis=-1, keepdims=True)
        previous = mx.concatenate([frames[..., :1], frames[..., :-1]], axis=-1)
        spectrum = mx.fft.rfft((frames - 0.97 * previous) * self.window, n=512)
        power = mx.square(spectrum.real) + mx.square(spectrum.imag)
        logmel = mx.log(mx.maximum(power @ self.mel, 1.1920929e-7))
        return logmel - logmel.mean(axis=1, keepdims=True)

    def conv(self, x, name, stride=1):
        kernel, bias = self.w[name + ".weight"], self.w[name + ".bias"]
        return mx.conv2d(x, kernel, stride=stride, padding=kernel.shape[1] // 2) + bias

    def __call__(self, audio, activity):
        x = self.features(audio).transpose(0, 2, 1)[..., None]
        x = nn.relu(self.conv(x, "conv1"))
        for layer, blocks in enumerate((3, 4, 6, 3), 1):
            for block in range(blocks):
                prefix = f"layer{layer}.{block}."
                stride = 2 if layer > 1 and block == 0 else 1
                skip = self.conv(x, prefix + "shortcut.0", stride) if stride == 2 else x
                x = nn.relu(self.conv(x, prefix + "conv1", stride))
                x = nn.relu(self.conv(x, prefix + "conv2") + skip)

        # The three speakers share the ResNet forward. Apply per-speaker
        # masks only at the trained statistics pooling layer.
        batch, _, steps, _ = x.shape
        features = x.transpose(0, 2, 3, 1).reshape(batch, steps, -1)
        clean = activity * (activity.sum(axis=-1, keepdims=True) == 1)
        masks = mx.where(clean.sum(axis=1, keepdims=True) > math.ceil(activity.shape[1] * 400 / audio.shape[1]),
                         clean, activity)
        masks = masks[:, mx.arange(steps) * activity.shape[1] // steps].transpose(0, 2, 1)
        count = masks.sum(axis=-1, keepdims=True)
        mean = (masks @ features) / mx.maximum(count, 1)
        delta = features[:, None] - mean[:, :, None]
        var = (mx.square(delta) * masks[..., None]).sum(axis=2) / mx.maximum(count - 1, 1)
        stats = mx.concatenate([mean, mx.sqrt(var)], axis=-1)
        return mx.addmm(self.w["seg_1.bias"], stats, self.w["seg_1.weight"])
