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


def test_centroid_linkage_matches_scipy():
    from scipy.cluster.hierarchy import linkage as reference
    from murmur.diarization.clustering import linkage, unit

    rng = np.random.default_rng(123)
    points = unit(mx.array(rng.normal(size=(85, 16)).astype("float32")))
    expected = reference(np.array(points), method="centroid")
    children, heights, sizes, _ = linkage(points)
    np.testing.assert_array_equal(children, expected[:, :2])
    np.testing.assert_allclose(heights, expected[:, 2], atol=1e-5)
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
