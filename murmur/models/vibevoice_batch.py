"""Batched speculative verification for the offline MLX VibeVoice Qwen2 decoder."""

from itertools import accumulate
from time import perf_counter

import mlx.core as mx


def _verify_blocks(lm, blocks, caches):
    if len(blocks) == 1:
        return mx.argmax(lm(mx.array(blocks), cache=caches[0]), axis=-1).tolist()

    # Apply the decoder's linear layers to every real token in one matrix.
    # Separate attention calls give each chunk its own positions and KV history.
    lengths = [len(block) for block in blocks]
    ends = list(accumulate(lengths))
    ids = mx.array([[token for block in blocks for token in block]])
    h = lm.embed_tokens(ids)
    masks = [cache[0].make_mask(n, return_array=False, window_size=None) for cache, n in zip(caches, lengths)]
    for layer_idx, layer in enumerate(lm.layers):
        attn = layer.self_attn
        x = layer.input_layernorm(h)

        def heads(projection, count):
            projected = projection(x).reshape(1, ends[-1], count, -1).transpose(0, 2, 1, 3)
            return mx.split(projected, ends[:-1], axis=2)

        queries = heads(attn.q_proj, attn.n_heads)
        keys = heads(attn.k_proj, attn.n_kv_heads)
        values = heads(attn.v_proj, attn.n_kv_heads)
        attended = []
        for q, k, v, mask, cache in zip(queries, keys, values, masks, caches):
            kv = cache[layer_idx]
            q = attn.rope(q, offset=kv.offset)
            k = attn.rope(k, offset=kv.offset)
            k, v = kv.update_and_fetch(k, v)
            attended.append(mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mask))
        merged = mx.concatenate(attended, axis=2).transpose(0, 2, 1, 3).reshape(1, ends[-1], -1)
        h = h + attn.o_proj(merged)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    h = lm.model.norm(h)
    logits = lm.embed_tokens.as_linear(h) if lm.config.tie_word_embeddings else lm.lm_head(h)
    predictions = mx.argmax(logits[0], axis=-1).tolist()
    return [predictions[end - n:end] for end, n in zip(ends, lengths)]


def verify_batch(lm, sequences, caches):
    finished = [perf_counter() if seq.done else None for seq in sequences]
    active = [i for i, seq in enumerate(sequences) if not seq.done]
    while active:
        blocks = [sequences[i].block() for i in active]
        predictions = _verify_blocks(lm, blocks, [caches[i] for i in active])
        continuing = []
        for i, block, pred in zip(active, blocks, predictions):
            seq = sequences[i]
            rejected = seq.accept(block, pred)
            if seq.done:
                finished[i] = perf_counter()
                caches[i] = None
            else:
                if rejected:
                    for kv in caches[i]:
                        if kv.trim(rejected) != rejected:
                            raise RuntimeError("VibeVoice could not rewind the speculative token cache.")
                continuing.append(i)
        active = continuing
    mx.clear_cache()
    return finished
