"""FP32 MLX inference for the released pyannote 3.1 neural networks."""

from functools import partial
import math

import mlx.core as mx
import mlx.nn as nn

from .metal import wespeaker_stem, winograd_f5, winograd_rect
from .metal.recurrence import batched_lstm


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

    def directions(self, x, layer):
        batch, steps, _ = x.shape
        if batch % 8:
            return mx.concatenate([self.direction(x, layer, False),
                                   self.direction(x, layer, True)], axis=-1)
        inputs = []
        for suffix in (f"_l{layer}", f"_l{layer}_reverse"):
            projected = mx.addmm(self.w["lstm.bias" + suffix], x, self.w["lstm.weight_ih" + suffix])
            inputs.extend([projected, self.w["lstm.weight_hh" + suffix]])
        return batched_lstm(inputs=inputs, grid=(512, batch // 8, 2), threadgroup=(512, 1, 1),
                             output_shapes=[(batch, steps, 256)], output_dtypes=[mx.float32])[0]

    def sincnet(self, windows):
        x = self.norm(windows[:, :, None], "sincnet.wav_norm1d")
        for block in range(3):
            if block == 0:
                x = mx.abs(mx.conv1d(x, self.w["sincnet.filters"], stride=10))
            else:
                prefix = f"sincnet.conv1d.{block}"
                x = mx.conv1d(x, self.w[prefix + ".weight"]) + self.w[prefix + ".bias"]
            x = nn.leaky_relu(self.norm(nn.MaxPool1d(3, 3)(x), f"sincnet.norm1d.{block}"))
        return x

    def classify(self, x):
        for layer in range(4):
            x = self.directions(x, layer)
        for layer in ("linear.0", "linear.1"):
            x = nn.leaky_relu(mx.addmm(self.w[layer + ".bias"], x, self.w[layer + ".weight"]))
        logits = mx.addmm(self.w["classifier.bias"], x, self.w["classifier.weight"])
        powerset = mx.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                             [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=mx.float32)
        return powerset[logits.argmax(axis=-1)]

    def __call__(self, windows):
        # SincNet's first convolution has a much longer time axis than the
        # LSTM input. Finish each front-end slice before starting the next.
        if len(windows) <= 32:
            features = self.sincnet(windows)
        else:
            slices = []
            for start in range(0, len(windows), 32):
                part = self.sincnet(windows[start:start + 32])
                mx.eval(part)
                slices.append(part)
            features = mx.concatenate(slices)

        # The batched recurrent kernel consumes complete groups of eight.
        # Route only the final incomplete group through the scalar recurrence.
        aligned = len(features) // 8 * 8
        if 0 < aligned < len(features):
            return mx.concatenate([self.classify(features[:aligned]),
                                   self.classify(features[aligned:])])
        return self.classify(features)


@partial(mx.compile, shapeless=True)
def _bias_relu(x, bias):
    return nn.relu(x + bias)


@partial(mx.compile, shapeless=True)
def _bias_residual_relu(x, bias, residual):
    return nn.relu((x + bias) + residual)


class WeSpeaker:
    def __init__(self, weights):
        self.w = weights

        # Apple-family matrix instructions operate on FP32 throughout.
        # Pack immutable filters once and retain stock convolution elsewhere.
        self.stem_weights = None
        self.winograd = {}
        if mx.metal.is_available() and mx.device_info(mx.gpu)["device_name"].startswith("Apple M"):
            stem = weights.get("conv1.weight")
            bias = weights.get("conv1.bias")
            if (stem is not None and stem.shape == (32, 3, 3, 1) and stem.dtype == mx.float32
                    and bias is not None and bias.shape == (32,) and bias.dtype == mx.float32):
                self.stem_weights = wespeaker_stem.pack_weights(stem)
            self.winograd = {
                name.removesuffix(".weight"): (
                    winograd_f5.pack_weights(value) if value.shape[0] in (128, 256)
                    else winograd_rect.pack_weights(value)
                )
                for name, value in weights.items()
                if name.endswith(".weight") and value.ndim == 4
                and value.shape[0] in (32, 64, 128, 256)
                and value.shape == (value.shape[0], 3, 3, value.shape[0])
                and value.dtype == mx.float32
            }

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

    def conv(self, x, name, stride=1, *, relu=False, residual=None):
        kernel, bias = self.w[name + ".weight"], self.w[name + ".bias"]
        # Preserve native MLX vertical border behavior on single-column inputs.
        single_column = x.shape[2] == 1 and x.shape[1] > 1 and x.shape[-1] in (64, 128, 256)
        if (name in self.winograd and not single_column and stride == 1 and (relu or residual is not None)
                and x.dtype == mx.float32 and mx.default_device() == mx.Device(mx.gpu)):
            operator = winograd_f5 if kernel.shape[0] in (128, 256) else winograd_rect
            return operator.convolution(x, self.winograd[name], bias, residual)
        out = mx.conv2d(x, kernel, stride=stride, padding=kernel.shape[1] // 2)
        if residual is not None:
            return _bias_residual_relu(out, bias, residual)
        if relu:
            return _bias_relu(out, bias)
        return out + bias

    def __call__(self, audio, activity):
        features = self.features(audio)
        if (self.stem_weights is not None and features.dtype == mx.float32
                and mx.default_device() == mx.Device(mx.gpu)):
            x = wespeaker_stem.convolution(features, self.stem_weights, self.w["conv1.bias"])
        else:
            x = self.conv(features.transpose(0, 2, 1)[..., None], "conv1", relu=True)
        for layer, blocks in enumerate((3, 4, 6, 3), 1):
            for block in range(blocks):
                prefix = f"layer{layer}.{block}."
                stride = 2 if layer > 1 and block == 0 else 1
                skip = self.conv(x, prefix + "shortcut.0", stride) if stride == 2 else x
                x = self.conv(x, prefix + "conv1", stride, relu=True)
                x = self.conv(x, prefix + "conv2", residual=skip)

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
