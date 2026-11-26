"""Boundary coverage for the optional MLX/Metal TitaNet benchmark."""

import numpy as np
import pytest


def test_packed_convolution_and_pooling_respect_clip_boundaries():
    try:
        import mlx.core as mx
    except ImportError as exc:
        pytest.skip(f"MLX unavailable: {exc}")
    if not mx.metal.is_available():
        pytest.skip("requires Apple Metal")
    from benchmarks.titanet_packed import PackedFrames

    lengths = np.array([1, 3, 17, 2], np.int32)
    rng = np.random.default_rng(41)
    clips = [rng.normal(size=(size, 35)).astype(np.float32) for size in lengths]
    x = mx.array(np.concatenate(clips))
    layout = PackedFrames(lengths)
    kernel = rng.normal(size=(35, 7, 1)).astype(np.float32)
    convolved = []
    for clip in clips:
        extended = np.pad(clip, ((3, 3), (0, 0)))
        windows = np.lib.stride_tricks.sliding_window_view(extended, 7, axis=0)
        convolved.append((windows * kernel[:, :, 0]).sum(axis=-1))
    np.testing.assert_allclose(np.asarray(layout.convolve(x, mx.array(kernel))),
                               np.concatenate(convolved), atol=3e-6)

    weights = []
    for clip in clips:
        exp = np.exp(clip - clip.max(axis=0))
        weights.append(exp / exp.sum(axis=0))
    attention = layout.softmax(x)
    np.testing.assert_allclose(np.asarray(attention), np.concatenate(weights), atol=1e-6)
    pooled = np.stack([(clip * weight).sum(axis=0) for clip, weight in zip(clips, weights)])
    np.testing.assert_allclose(np.asarray(layout.average(x, attention)), pooled, atol=1e-6)
