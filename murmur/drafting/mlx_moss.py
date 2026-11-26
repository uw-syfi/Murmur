"""Greedy verification of Murmur's MOSS proposals on Apple Silicon."""

from .vllm_proposer import DraftTranscript, MossDrafter


def verify_greedy(first, proposer, verify, rewind, eos, budget):
    """Keep the last emitted token pending. The cache holds everything before it."""
    tokens = []
    stats = {"proposed": 0, "accepted": 0, "verification_calls": 0}
    if budget <= 0 or first in eos:
        return tokens, stats
    tokens.append(first)
    while len(tokens) < budget:
        # Leave room for the target's correction (or a bonus after a full match).
        k = min(8, budget - len(tokens) - 1)
        draft = proposer.propose(tokens, k)[:k] if k else []
        predictions = verify([tokens[-1], *draft])
        matched = 0
        while matched < len(draft) and predictions[matched] == draft[matched]:
            matched += 1
        stats["verification_calls"] += 1
        stats["proposed"] += len(draft)
        stats["accepted"] += matched
        if matched < len(draft):
            rewind(len(draft) - matched)
        for token in predictions[:matched + 1]:
            if token in eos:
                return tokens, stats
            tokens.append(token)
    return tokens, stats


def generate(model, audio, words, language, diar_segments, offset, budget):
    import mlx.core as mx
    from mlx_lm.models.cache import trim_prompt_cache

    tokenizer = model._tokenizer
    transcript = DraftTranscript(words, tokenizer, sep="" if language == "zh" else " ")
    proposer = MossDrafter(transcript, tokenizer, language, diar_segments, offset)
    ids, embeddings, length, _ = model._prepare_generation_inputs(audio, None)
    cache = model.make_cache()

    # Mirror the runtime's chunked audio prefill, leaving the final prompt token
    # for a one-position forward pass just like the ordinary greedy generator.
    for start in range(0, length - 1, 2048):
        end = min(start + 2048, length - 1)
        model(ids[None, start:end], input_embeddings=embeddings[None, start:end], cache=cache)
        mx.eval([layer.state for layer in cache])
    logits = model(ids[None, -1:], input_embeddings=embeddings[None, -1:], cache=cache)
    first = mx.argmax(logits[0, -1]).item()

    def verify(block):
        logits = model(mx.array([block]), cache=cache)
        return mx.argmax(logits[0], axis=-1).tolist()

    def rewind(count):
        if trim_prompt_cache(cache, count) != count:
            raise RuntimeError("MOSS could not rewind the speculative token cache.")

    tokens, stats = verify_greedy(first, proposer, verify, rewind, model._eos_token_ids(), budget)
    return tokenizer.decode(tokens, skip_special_tokens=True).strip(), stats
