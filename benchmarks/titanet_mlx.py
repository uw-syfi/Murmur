"""Batched Apple GPU inference for the sherpa-onnx NeMo TitaNet Small export."""

import mlx.core as mx
import numpy as np
import onnx
from onnx import numpy_helper


class PaddedFrames:
    def __init__(self, frames, lengths):
        self.valid = mx.arange(frames)[None, :, None] < lengths[:, None, None]
        self.weights = self.valid.astype(mx.float32) / lengths[:, None, None]

    def convolve(self, x, weight):
        return mx.conv1d(x * self.valid, weight, padding=weight.shape[1] // 2,
                         groups=x.shape[-1])

    def expand(self, x):
        return x[:, None, :]

    def average(self, x, weights=None):
        return (x * (self.weights if weights is None else weights)).sum(axis=1)

    def softmax(self, x):
        return mx.softmax(mx.where(self.valid, x, -float("inf")), axis=1)


class TitaNetSmall:
    """Read the encoder and embedding weights directly from the NeMo ONNX graph."""

    def __init__(self, checkpoint):
        model = onnx.load(checkpoint)
        weights = {w.name: w for w in model.graph.initializer}
        self.layers = {}
        for node in model.graph.node:
            if node.op_type not in ("Conv", "MatMul", "BatchNormalization"):
                continue
            if node.name == "/decoder/final/MatMul":  # Training speaker classifier
                continue
            params = [mx.array(numpy_helper.to_array(weights[name]))
                      for name in node.input[1:]]
            if node.op_type == "Conv":
                # MLX convolutions use (out_channels, kernel, in_channels).
                params[0] = params[0].transpose(0, 2, 1)
            self.layers[node.name] = params
        mx.eval(self.layers)

    def pointwise(self, x, name):
        weight, bias = self.layers[name]
        return x @ weight[:, 0, :].T + bias

    def batch_norm(self, x, name):
        scale, bias, mean, variance = self.layers[name]
        return (x - mean) * mx.rsqrt(variance + 1e-5) * scale + bias

    def __call__(self, x, lengths):
        return self._forward(x, PaddedFrames(x.shape[1], lengths))

    def _forward(self, x, layout):
        for block, repeats in enumerate((1, 3, 3, 3, 1)):
            prefix = f"/encoder/encoder/encoder.{block}"
            residual = x
            for i in range(repeats):
                layer = 5 * i
                weight, = self.layers[f"{prefix}/mconv.{layer}/conv/Conv"]
                x = layout.convolve(x, weight)
                x = self.pointwise(x, f"{prefix}/mconv.{layer + 1}/conv/Conv")
                if i + 1 < repeats:
                    x = mx.maximum(x, 0)
            gate = f"{prefix}/mconv.{layer + 3}/fc"
            mean = layout.average(x)
            hidden = mx.maximum(mean @ self.layers[f"{gate}/fc.0/MatMul"][0], 0)
            x = x * layout.expand(mx.sigmoid(hidden @ self.layers[f"{gate}/fc.2/MatMul"][0]))
            if repeats > 1:
                x = x + self.pointwise(residual, f"{prefix}/res.0.0/conv/Conv")
            x = mx.maximum(x, 0)

        def moments(weights=None):
            mean = layout.average(x, weights)
            variance = layout.average((x - layout.expand(mean)).square(), weights)
            return mean, mx.sqrt(mx.maximum(variance, 1e-10))

        mean, std = moments()
        attention = "/decoder/_pooling/attention_layer"
        weights, bias = self.layers[f"{attention}/attention_layer.0/conv_layer/Conv"]
        # The ONNX graph tiles both statistics across time before a 1x1 conv.
        # Project the two constant terms once per speaker excerpt instead.
        frame_w, mean_w, std_w = mx.split(weights[:, 0, :], 3, axis=1)
        hidden = x @ frame_w.T + layout.expand(mean @ mean_w.T + std @ std_w.T + bias)
        hidden = self.batch_norm(mx.maximum(hidden, 0),
                                 f"{attention}/attention_layer.0/bn/BatchNormalization")
        logits = self.pointwise(mx.tanh(hidden), f"{attention}/attention_layer.2/Conv")
        mean, std = moments(layout.softmax(logits))
        pooled = mx.concatenate((mean, std), axis=-1)
        head = "/decoder/emb_layers.0"
        pooled = self.batch_norm(pooled, f"{head}/emb_layers.0.0/BatchNormalization")
        return self.pointwise(pooled, f"{head}/emb_layers.0.1/Conv")

    def embed(self, features, batch_size, *, packed=False):
        """Return embeddings in input order. Packed batches use only real frames."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if packed:
            if __package__:
                from .titanet_packed import PackedFrames
            else:
                from titanet_packed import PackedFrames
        order = sorted(range(len(features)), key=lambda i: len(features[i]))
        embeddings = np.empty((len(order), 192), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            lengths = np.array([len(features[i]) for i in indices], dtype=np.int32)
            if packed:
                if np.any(lengths == 0):
                    raise ValueError("packed excerpts must contain at least one frame")
                inputs = np.concatenate([features[i] for i in indices]).astype(np.float32, copy=False)
                vectors = self._forward(mx.array(inputs), PackedFrames(lengths))
                embeddings[indices] = np.asarray(vectors)
                continue
            frames = int(lengths.max() + 31) // 32 * 32
            inputs = np.zeros((len(indices), frames, 80), dtype=np.float32)
            for row, i in enumerate(indices):
                inputs[row, :lengths[row]] = features[i]
            embeddings[indices] = np.asarray(self(mx.array(inputs), mx.array(lengths)))
        return embeddings
