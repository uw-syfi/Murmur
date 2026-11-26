"""Packed Reverb GPU projections and independence of window quantization."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("onnx")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Apple Metal")


@pytest.mark.parametrize("dtype,zero_point", [(np.uint8, 137), (np.int8, 0)])
def test_projection_preserves_checkpoint_codes(dtype, zero_point):
    from benchmarks.reverb_mlx import QuantizedProjection

    rng = np.random.default_rng(8)
    limits = np.iinfo(dtype)
    weights = rng.integers(limits.min, limits.max, size=(60, 7), dtype=dtype)
    x = rng.integers(-100, 150, size=(2, 9, 60), dtype=np.int32)
    projection = QuantizedProjection(weights, zero_point)
    expected = x @ (weights.astype(np.int32) - zero_point)
    np.testing.assert_array_equal(np.asarray(projection(mx.array(x, dtype=mx.float32))), expected)


def test_dynamic_quantization_keeps_quiet_windows_independent():
    from benchmarks.reverb_mlx import quantize

    inputs = mx.array([[[0., 0., 0.]], [[-0.0001, 0., 0.0001]], [[-1., 0., 3.]]])
    q, scales = quantize(inputs, (1, 2))
    assert np.isfinite(np.asarray(q)).all()
    for i in range(3):
        single, scale = quantize(inputs[i:i+1], (1, 2))
        np.testing.assert_array_equal(np.asarray(q[i]), np.asarray(single[0]))
        np.testing.assert_array_equal(np.asarray(scales[i]), np.asarray(scale[0]))


@pytest.mark.parametrize("reverse", [False, True])
def test_integer_recurrence_matches_per_timestep_reference(reverse):
    from benchmarks.reverb_mlx import _recurrence

    rng = np.random.default_rng(45)
    batch, steps, hidden = 2, 11, 256
    projected = rng.normal(size=(batch, steps, hidden, 4)).astype(np.float32)
    weights = rng.integers(-20, 21, size=(hidden, hidden, 4), dtype=np.int8)
    bias = rng.normal(size=(hidden, 4)).astype(np.float32)
    weight_scale = np.float32(.003)
    state = np.zeros((batch, hidden), dtype=np.float32)
    cell = np.zeros_like(state)
    expected = np.empty((batch, steps, hidden), dtype=np.float32)
    for t in (reversed(range(steps)) if reverse else range(steps)):
        low = np.minimum(state.min(axis=1, keepdims=True), 0)
        high = np.maximum(state.max(axis=1, keepdims=True), 0)
        scale = np.where(high > low, (high - low) / 255, 1)
        zero = np.rint(-low / scale)
        q = (np.clip(np.rint(state / scale) + zero, 0, 255) - zero).astype(np.int32)
        sums = q @ weights.reshape(hidden, -1).astype(np.int32)
        gates = projected[:, t] + (sums.astype(np.float32) * (scale * weight_scale)).reshape(batch, hidden, 4) + bias
        sigmoid = 1 / (1 + np.exp(-gates[..., :3]))
        cell = sigmoid[..., 2] * cell + sigmoid[..., 0] * np.tanh(gates[..., 3])
        state = sigmoid[..., 1] * np.tanh(cell)
        expected[:, t] = state
    actual, = _recurrence(inputs=[mx.array(projected.reshape(batch, steps, -1)), mx.array(weights),
                                  mx.array(weight_scale), mx.array(bias)],
                          template=[("Reverse", reverse)], grid=(hidden, batch, 1), threadgroup=(hidden, 1, 1),
                          output_shapes=[expected.shape], output_dtypes=[mx.float32])
    np.testing.assert_allclose(np.array(actual), expected, rtol=0, atol=1e-5)
