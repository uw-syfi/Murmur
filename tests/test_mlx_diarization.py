"""Regression tests for the Metal recurrence and MLX speaker reconstruction."""

import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires an Apple GPU")


@pytest.mark.parametrize("reverse", [False, True])
def test_recurrence_with_saturated_gates(reverse):
    import torch
    from murmur.diarization.networks import Segmentation

    rng = np.random.default_rng(15)
    x = rng.normal(size=(2, 37, 17)).astype("float32")
    layer = torch.nn.LSTM(17, 128, batch_first=True)
    weights = {}
    suffix = "_l0_reverse" if reverse else "_l0"
    for name, param in layer.named_parameters():
        # Saturation exposed NaNs in the default fast Metal tanh implementation.
        values = rng.normal(size=tuple(param.shape)).astype("float32") * 0.15
        if name == "bias_ih_l0":
            values[256:320] = np.linspace(-100, 100, 64)
        param.data.copy_(torch.from_numpy(values))
        if name.startswith("weight"):
            weights["lstm." + name.removesuffix("_l0") + suffix] = mx.array(values.T.copy())
        else:
            key = "lstm.bias" + suffix
            weights[key] = weights.get(key, 0) + mx.array(values)
    model = Segmentation(weights)
    result = np.array(model.direction(mx.array(x), 0, reverse))
    sequence = torch.from_numpy(x)
    with torch.no_grad():
        expected, _ = layer(sequence.flip(1) if reverse else sequence)
    if reverse:
        expected = expected.flip(1)
    np.testing.assert_allclose(result, expected.numpy(), atol=1e-5)


@pytest.mark.parametrize("channels", [64, 128])
def test_wespeaker_single_column_convolution(channels):
    from murmur.diarization.networks import WeSpeaker

    rng = np.random.default_rng(17)
    x = rng.normal(size=(1, 3, 1, channels)).astype(np.float32)
    kernel = rng.normal(0, 0.05, size=(channels, 3, 3, channels)).astype(np.float32)
    bias = rng.normal(size=channels).astype(np.float32)
    residual = rng.normal(size=x.shape).astype(np.float32)
    model = WeSpeaker({"conv.weight": mx.array(kernel), "conv.bias": mx.array(bias)})
    actual = model.conv(mx.array(x), "conv", residual=mx.array(residual))

    # Only the middle column of the filter overlaps this one-column input.
    padded = np.pad(x[0, :, 0].astype(np.float64), ((1, 1), (0, 0)))
    expected = sum(padded[row:row + 3] @ kernel[:, row, 1].astype(np.float64).T
                   for row in range(3))
    expected = np.maximum(expected[None, :, None, :] + bias + residual, 0)
    assert actual.shape == x.shape
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize(("channels", "with_residual"), [(32, False), (64, True)])
def test_wespeaker_rectangular_convolution_partial_tiles(channels, with_residual):
    from unittest.mock import patch
    from murmur.diarization.metal import winograd_rect
    from murmur.diarization.networks import WeSpeaker

    if not mx.device_info(mx.gpu)["device_name"].startswith("Apple M"):
        pytest.skip("rectangular convolution requires an Apple M GPU")

    # Nine 2-by-4 tiles require a second threadgroup with one valid tile.
    rng = np.random.default_rng(32)
    x = rng.normal(size=(2, 5, 11, channels)).astype(np.float32)
    kernel = rng.normal(0, 0.05, size=(channels, 3, 3, channels)).astype(np.float32)
    bias = rng.normal(size=channels).astype(np.float32)
    residual = rng.normal(size=x.shape).astype(np.float32) if with_residual else None
    model = WeSpeaker({"conv.weight": mx.array(kernel), "conv.bias": mx.array(bias)})
    with patch.object(winograd_rect, "convolution", wraps=winograd_rect.convolution) as routed:
        actual = model.conv(mx.array(x), "conv", relu=True,
                            residual=mx.array(residual) if with_residual else None)
        mx.eval(actual)
        routed.assert_called_once()

    h, w = x.shape[1:3]
    padded = np.pad(x.astype(np.float64), ((0, 0), (1, 1), (1, 1), (0, 0)))
    expected = sum(padded[:, row:row + h, col:col + w] @ kernel[:, row, col].T
                   for row in range(3) for col in range(3)) + bias
    if with_residual:
        expected += residual
    assert actual.shape == x.shape
    assert actual.dtype == mx.float32
    np.testing.assert_allclose(np.asarray(actual), np.maximum(expected, 0), rtol=1e-5, atol=5e-6)


def test_wespeaker_stem_odd_time_strip(monkeypatch):
    from murmur.diarization.networks import WeSpeaker

    if not mx.device_info(mx.gpu)["device_name"].startswith("Apple M"):
        pytest.skip("fused stem requires an Apple M GPU")

    # Seven frames span one full strip and a final three-frame strip.
    rng = np.random.default_rng(40)
    audio = mx.array(rng.normal(size=(2, 400 + 6 * 160)).astype(np.float32))
    kernel = rng.normal(0, 0.2, size=(32, 3, 3, 1)).astype(np.float32)
    bias = rng.normal(size=32).astype(np.float32)
    model = WeSpeaker({"conv1.weight": mx.array(kernel), "conv1.bias": mx.array(bias)})

    # Capture the actual tensor passed from the production stem to ResNet.
    class ReachedResidualBlock(Exception):
        pass

    def capture(activation, name, *args, **kwargs):
        assert name == "layer1.0.conv1"
        raise ReachedResidualBlock(activation)

    monkeypatch.setattr(model, "conv", capture)
    with pytest.raises(ReachedResidualBlock) as captured:
        model(audio, None)
    actual = captured.value.args[0]

    features = np.asarray(model.features(audio)).transpose(0, 2, 1).astype(np.float64)
    batch, bands, steps = features.shape
    padded = np.pad(features, ((0, 0), (1, 1), (1, 1)))
    expected = sum(padded[:, row:row + bands, col:col + steps, None] * kernel[:, row, col, 0]
                   for row in range(3) for col in range(3)) + bias
    assert actual.shape == (batch, bands, steps, 32)
    assert actual.dtype == mx.float32
    np.testing.assert_allclose(np.asarray(actual), np.maximum(expected, 0), rtol=1e-5, atol=3e-6)


def test_centroid_linkage_matches_scipy():
    from scipy.cluster.hierarchy import linkage as reference
    from murmur.diarization.clustering import linkage, unit

    rng = np.random.default_rng(123)
    points = unit(mx.array(rng.normal(size=(85, 16)).astype("float32")))
    expected = reference(np.array(points), method="centroid")
    children, heights, sizes, _ = linkage(points)
    np.testing.assert_array_equal(children, expected[:, :2])
    np.testing.assert_allclose(heights, expected[:, 2], atol=1e-5)
    assert sizes.dtype == mx.int32
    np.testing.assert_array_equal(sizes[len(points):], expected[:, 3])


def test_timeline_keeps_overlap_and_clips_partial_window():
    from murmur.diarization.mlx import timeline

    activity = mx.array([[[1, 0], [1, 1], [0, 1], [0, 1]]], dtype=mx.float32)
    turns = timeline(activity, mx.array([[0, 1]]), 2, duration=.08)
    assert turns == [(0.03096875, 0.06471875, "SPEAKER_00"), (0.04784375, .08, "SPEAKER_01")]


def test_speaker_count_bounds():
    from murmur.diarization.clustering import assign

    embeddings = mx.array([[[1., 0.]], [[1., .01]], [[0., 1.]], [[.01, 1.]]])
    activity = mx.ones((4, 10, 1))
    assert assign(embeddings, activity)[1] == 2
    assert assign(embeddings, activity, requested=1)[1] == 1
    assert assign(embeddings, activity, minimum=3)[1] == 3
    assert assign(embeddings, mx.zeros_like(activity))[1] == 1


def test_full_diarizer_against_mps_on_meeting_excerpt():
    import soundfile as sf
    import torch
    from murmur.audio import Diarizer
    from murmur.diarization.mlx import MLXDiarizer

    path = os.getenv("MURMUR_AMI_ES2004A")
    if not path:
        pytest.skip("set MURMUR_AMI_ES2004A to compare the released checkpoints")
    samples, sr = sf.read(path, start=80 * 16000, frames=65 * 16000 + 123, dtype="float32")
    assert sr == 16000
    waveform = torch.from_numpy(samples)
    expected = Diarizer(device="mps")(waveform, num_speakers=4)
    # The reference can return activity in the padding of the final 10 s
    # window. Murmur clips its output to actual audio samples.
    duration = len(waveform) / sr
    expected = [(a, min(b, duration), k) for a, b, k in expected if a < duration]
    actual = MLXDiarizer()(waveform, num_speakers=4)
    assert all(0 <= a < b <= duration for a, b, _ in actual)
    # Cluster numbers are local identifiers. Compare speaker partitions and
    # boundaries independently of a consistent renumbering.
    assert len(actual) == len(expected) > 0
    np.testing.assert_allclose([t[:2] for t in actual], [t[:2] for t in expected], rtol=0, atol=1e-6)
    pairs = {(a[2], b[2]) for a, b in zip(actual, expected)}
    assert len(pairs) == len({a for a, _ in pairs}) == len({b for _, b in pairs})
