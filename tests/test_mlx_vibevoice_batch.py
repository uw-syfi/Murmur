"""Batched Qwen2 verification must preserve per-chunk context across rejections."""

import pytest

mx = pytest.importorskip("mlx.core")
qwen = pytest.importorskip("mlx_audio.lm.models.qwen2")


@pytest.mark.parametrize("tied,bits", [(True, None), (False, None), (False, 4)])
def test_packed_verification_matches_independent_greedy(tied, bits):
    import numpy as np
    from mlx_audio.lm.models.cache import make_prompt_cache
    from mlx_audio.stt.models.vibevoice_asr.vibevoice_asr import LanguageModel
    from murmur.models.vibevoice_batch import _verify_blocks

    mx.random.seed(17)
    model = LanguageModel(qwen.ModelArgs(model_type="qwen2", hidden_size=64, num_hidden_layers=2,
                                         intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                                         rms_norm_eps=1e-6, vocab_size=128, tie_word_embeddings=tied))
    if bits:
        import mlx.nn as nn
        nn.quantize(model, bits=bits, group_size=64)
    grouped = [make_prompt_cache(model) for _ in range(3)]
    serial = [make_prompt_cache(model) for _ in grouped]
    batches = [
        ([[1, 9, 12], [1, 4], [2]], [0, 0, 0]),
        ([[7], [15, 16, 19, 11], [6, 8]], [0, 2, 1]),
        ([[4, 5, 6], [25], [10, 11]], [1, 0, 0]),
    ]
    for blocks, rejected in batches:
        pred = _verify_blocks(model, blocks, grouped)
        for i, block in enumerate(blocks):
            expected = mx.argmax(model(mx.array([block]), cache=serial[i]), axis=-1).tolist()[0]
            assert pred[i] == expected
            for cached, reference in zip(grouped[i], serial[i]):
                assert cached.trim(rejected[i]) == reference.trim(rejected[i])
                for x, y in zip(cached.state, reference.state):
                    np.testing.assert_allclose(np.array(x), np.array(y), atol=2e-5, rtol=1e-5)
