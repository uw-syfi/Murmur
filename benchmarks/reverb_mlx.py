"""Native MLX segmentation for sherpa-onnx-reverb-diarization-v1/model.int8.onnx.

Run: python benchmarks/reverb_mlx.py CHECKPOINT AUDIO [--batch 32] [--compare]
"""
import argparse
import math
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import onnx
from onnx import numpy_helper

WINDOW = 160000
HOP = 16000


def quantize(x, axes):
    """ONNX asymmetric uint8 activation quantization, independently per window."""
    low = mx.minimum(x.min(axis=axes, keepdims=True), 0)
    high = mx.maximum(x.max(axis=axes, keepdims=True), 0)
    scale = mx.where(high > low, (high - low) / 255, 1)
    zero = mx.clip(mx.round(-low / scale), 0, 255)
    q = mx.clip(mx.round(x / scale) + zero, 0, 255)
    return q - zero, scale


# Each group advances one sequence; 8 SIMD groups reduce the hidden-state
# range every timestep. Recurrent weights stay signed bytes in device memory.
_recurrence = mx.fast.metal_kernel(
    name="reverb_dynamic_int8_lstm", input_names=["projected", "weights", "weight_scale", "bias"],
    output_names=["output"],
    source="""
        uint lane = thread_position_in_threadgroup.x;
        uint b = threadgroup_position_in_grid.y;
        uint steps = projected_shape[1];
        threadgroup float low[8], high[8];
        threadgroup short hidden_q[256];
        float h = 0, c = 0;
        for (uint t = 0; t < steps; ++t) {
            float lo = simd_min(min(h, 0.0f)), hi = simd_max(max(h, 0.0f));
            if (lane % 32 == 0) { low[lane/32] = lo; high[lane/32] = hi; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            lo = low[0]; hi = high[0];
            for (uint k=1; k<8; ++k) { lo=min(lo,low[k]); hi=max(hi,high[k]); }
            float scale = hi > lo ? (hi-lo)/255.0f : 1.0f;
            float zero = rint(-lo/scale);
            hidden_q[lane] = short(clamp(rint(h/scale)+zero,0.0f,255.0f)-zero);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            int4 sum = int4(0);
            for (uint k = 0; k < 256; ++k) {
                uint w = (k*256 + lane)*4;
                sum += int(hidden_q[k]) * int4(weights[w],weights[w+1],weights[w+2],weights[w+3]);
            }
            uint frame = Reverse ? steps-1-t : t;
            uint p = ((b*steps+frame)*256+lane)*4;
            float4 gates = float4(projected[p],projected[p+1],projected[p+2],projected[p+3])
                         + float4(sum) * (scale * weight_scale)
                         + float4(bias[lane*4],bias[lane*4+1],bias[lane*4+2],bias[lane*4+3]);
            // ONNX gate layout: input, output, forget, cell.
            float3 sigmoid = 1.0f/(1.0f + metal::precise::exp(-gates.xyz));
            c = sigmoid.z*c + sigmoid.x*metal::precise::tanh(gates.w);
            h = sigmoid.y*metal::precise::tanh(c);
            output[(b*steps+frame)*256+lane] = h;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """,
)


def interleave(gates):
    """Store the four ONNX gates adjacent in memory for each of the 256 neurons."""
    return gates.reshape(*gates.shape[:-1], 4, 256).swapaxes(-2, -1).reshape(gates.shape)


class QuantizedProjection:
    """Pack the original 8-bit weight codes into the layout of MLX affine QMM."""

    def __init__(self, weights, zero_point):
        # weights arrive as (input, output). Bias the signed LSTM codes by
        # 128 so every layer can use the same unsigned MLX affine format.
        if weights.dtype == np.int8:
            weights = weights.view(np.uint8) ^ np.uint8(128)
            zero_point = int(zero_point) + 128
        self.inputs, self.outputs = weights.shape
        self.padded_inputs = (self.inputs + 31) // 32 * 32
        padded_outputs = (self.outputs + 31) // 32 * 32
        codes = np.full((padded_outputs, self.padded_inputs), zero_point, dtype=np.uint8)
        codes[:self.outputs, :self.inputs] = weights.T
        self.weights = mx.array(codes.view("<u4"))
        groups = (padded_outputs, self.padded_inputs // 32)
        self.scales = mx.ones(groups)
        self.biases = mx.full(groups, -float(zero_point))

    def __call__(self, inputs):
        if self.inputs != self.padded_inputs:
            inputs = mx.pad(inputs, [(0, 0)] * (inputs.ndim - 1) + [(0, self.padded_inputs - self.inputs)])
        projected = mx.quantized_matmul(inputs, self.weights, self.scales, self.biases, group_size=32, bits=8)
        return projected[..., :self.outputs]


class ReverbInt8:
    """Execute the fixed Reverb v1 INT8 export, with seven powerset output classes."""

    def __init__(self, checkpoint):
        graph = onnx.load(str(checkpoint)).graph
        weights = {w.name: numpy_helper.to_array(w) for w in graph.initializer}
        nodes = {n.name: n for n in graph.node}
        recurrent = [n for n in graph.node if n.op_type == "DynamicQuantizeLSTM"]
        if len(recurrent) != 2 or any(weights[n.input[2]].shape != (2, 256, 1024) for n in recurrent):
            raise ValueError("Expected the Reverb v1 dynamically quantized ONNX checkpoint.")

        def param(name):
            return mx.array(weights[name])

        def centered(name, zero):
            return (param(name).astype(mx.int32) - param(zero).astype(mx.int32)).astype(mx.float32)

        self.norms = {}
        for layer in ("wav_norm1d", "norm1d.0", "norm1d.1", "norm1d.2"):
            self.norms[layer] = tuple(param(f"sincnet.{layer}.{k}") for k in ("weight", "bias"))

        prefix = "sincnet.conv1d.0.filterbank."
        low = 50 + mx.abs(param(prefix + "low_hz_"))
        high = mx.minimum(low + 50 + mx.abs(param(prefix + "band_hz_")), 8000)
        bandwidth = 2 * (high - low)
        n = centered(prefix + "n__quantized", prefix + "n__zero_point")
        angles = []
        for hz in (low, high):
            codes, scale = quantize(hz, (0, 1))
            angles.append((codes * n) * (scale * param(prefix + "n__scale")))
        lo, hi = angles
        div = param(nodes["/sincnet/conv1d.0/Div"].input[1])
        window = param(prefix + "window_")
        even = (mx.sin(hi) - mx.sin(lo)) / div * window
        odd = (mx.cos(lo) - mx.cos(hi)) / div * window
        even = mx.concatenate([even, bandwidth, even[:, ::-1]], axis=1) / bandwidth
        odd = mx.concatenate([odd, mx.zeros_like(bandwidth), -odd[:, ::-1]], axis=1) / bandwidth
        self.filters = mx.concatenate([even, odd])[:, :, None]

        self.convs = []
        for i in (1, 2):
            key = f"sincnet.conv1d.{i}.weight"
            w = weights[key + "_quantized"]
            weight = QuantizedProjection(w.transpose(2, 1, 0).reshape(-1, w.shape[0]), weights[key + "_zero_point"])
            self.convs.append((weight, param(key + "_scale"), param(f"sincnet.conv1d.{i}.bias")))

        self.layers = []
        for node in recurrent:
            _, w, r, b, _, _, _, _, ws, wz, rs, rz = node.input
            if np.any(weights[rz]) or np.any(weights[wz]):
                raise ValueError("Reverb LSTM weights must use symmetric signed INT8 quantization.")
            directions = []
            for d in (0, 1):
                bias = param(b)[d].reshape(2, 1024).sum(axis=0)
                projection = QuantizedProjection(interleave(weights[w][d]), weights[wz][d])
                directions.append((projection, param(ws)[d],
                                   interleave(param(r)[d]), param(rs)[d], interleave(bias)))
            self.layers.append(directions)

        self.head = []
        for name in ("linear.0", "linear.1", "classifier"):
            node = nodes[f"/{name}/MatMul_quant"]
            key = node.input[1].removesuffix("_quantized")
            projection = QuantizedProjection(weights[node.input[1]], weights[node.input[3]])
            self.head.append((projection, param(key + "_scale"), param(name + ".bias")))

    def norm(self, x, layer):
        weight, bias = self.norms[layer]
        return (x - x.mean(axis=1, keepdims=True)) * mx.rsqrt(x.var(axis=1, keepdims=True) + 1e-5) * weight + bias

    def __call__(self, windows):
        """Return log probabilities [window, frame, class] on the MLX GPU."""
        x = self.norm(windows[..., None], "wav_norm1d")
        x = mx.abs(mx.conv1d(x, self.filters, stride=10))
        pool = nn.MaxPool1d(3, 3)
        x = nn.leaky_relu(self.norm(pool(x), "norm1d.0"))
        # Lower the two valid width-five convolutions to packed MLX QMM.
        for i, (weight, ws, bias) in enumerate(self.convs, 1):
            q, scale = quantize(x, (1, 2))
            batch, steps, channels = q.shape
            patches = mx.as_strided(mx.contiguous(q), (batch, steps - 4, 5 * channels),
                                    strides=(steps * channels, channels, 1))
            x = weight(patches) * (scale * ws) + bias
            x = nn.leaky_relu(self.norm(pool(x), f"norm1d.{i}"))
        batch, steps, _ = x.shape
        for directions in self.layers:
            q, scale = quantize(x, (1, 2))
            outputs = []
            for d, (w, ws, r, rs, bias) in enumerate(directions):
                projected = w(q) * (scale * ws)
                out, = _recurrence(inputs=[projected, r, rs, bias], template=[("Reverse", bool(d))],
                                   grid=(256, batch, 1), threadgroup=(256, 1, 1),
                                   output_shapes=[(batch, steps, 256)], output_dtypes=[mx.float32])
                outputs.append(out)
            x = mx.concatenate(outputs, axis=-1)
        for i, (weight, ws, bias) in enumerate(self.head):
            q, scale = quantize(x, (1, 2))
            x = weight(q) * (scale * ws) + bias
            if i < 2:
                x = nn.leaky_relu(x)
        return nn.log_softmax(x, axis=-1)

    def segment(self, audio, batch_size=32):
        if audio.ndim != 1 or not len(audio):
            raise ValueError("Expected a nonempty mono 16 kHz waveform")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        with mx.stream(mx.gpu):
            n = 1 + max(0, math.ceil((len(audio) - WINDOW) / HOP))
            padded = mx.pad(mx.array(audio, dtype=mx.float32), (0, WINDOW + (n - 1) * HOP - len(audio)))
            windows = mx.as_strided(padded, (n, WINDOW), strides=(HOP, 1))
            classes = []
            for i in range(0, n, batch_size):
                scores = self(mx.contiguous(windows[i:i + batch_size]))
                classes.append(np.array(scores.argmax(axis=-1)))
            return np.concatenate(classes)


def main():
    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--compare", action="store_true", help="Time and compare the four-thread ONNX CPU reference")
    args = parser.parse_args()
    audio, rate = sf.read(args.audio, dtype="float32")
    if rate != 16000 or audio.ndim != 1 or not len(audio):
        parser.error("expected nonempty, mono 16 kHz audio")
    with mx.stream(mx.gpu):
        model = ReverbInt8(args.checkpoint)
        model.segment(audio[:WINDOW], args.batch)
        start = perf_counter()
        classes = model.segment(audio, args.batch)
        print(f"MLX GPU: {perf_counter()-start:.2f} s, {len(classes)} windows", flush=True)
    if args.compare:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        cpu = ort.InferenceSession(str(args.checkpoint), opts, providers=["CPUExecutionProvider"])
        padded = np.pad(audio, (0, WINDOW + (len(classes) - 1) * HOP - len(audio)))

        def reference(i):
            x = padded[i * HOP:i * HOP + WINDOW].reshape(1, 1, WINDOW)
            return cpu.run(["y"], {"x": x})[0][0].argmax(axis=-1)

        reference(0)
        start = perf_counter()
        ref = np.stack([reference(i) for i in range(len(classes))])
        print(f"ONNX CPU: {perf_counter()-start:.2f} s, frame agreement: {(classes == ref).mean():.6%}")


if __name__ == "__main__":
    main()
