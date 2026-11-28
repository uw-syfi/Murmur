"""Check the Metal residual block against mlx-audio without loading a checkpoint."""

import pytest

mx = pytest.importorskip("mlx.core")
encoder = pytest.importorskip("mlx_audio.stt.models.vibevoice_asr.audio_encoder")


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("channels", [32, 64, 128, 256])
def test_causal_block_matches_reference_across_tile_boundary(channels):
    from murmur.models.vibevoice_speech import _block

    mx.random.seed(17)
    block = encoder.Block1D(channels, layer_scale_init_value=1.0)
    # Each batch starts with an empty causal history. The last tile is partial.
    x = mx.random.normal((2, 35, channels))
    expected = block(x)
    actual = _block(block, x)
    assert actual.shape == expected.shape
    assert mx.allclose(actual, expected, atol=2e-5, rtol=1e-5).item()
