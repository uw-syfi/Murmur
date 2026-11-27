"""Profile warm VibeVoice speculative decoding and WhisperX on the same recording.

Run vibevoice first with --output vibe.json; then run whisperx in its own
environment with --timeline vibe.json and --output whisperx.json. Use the same
16 kHz mono WAV. Timings are wall seconds; all interval endpoints are relative
to the start of each call to transcribe. Run one GPU workload at a time.
"""

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch


def profile_batch(pipeline, args):
    """Record batch occupancy, synchronized verifier time and overlapping Whisper work."""
    import mlx.core as mx
    from murmur.models import vibevoice_batch

    drafts, diarizations, calls = [], [], Counter()
    verify_s = 0.0
    verify = vibevoice_batch._verify_blocks
    drafter, diarizer = pipeline.drafter, pipeline.diarizer

    def verify_batch(lm, blocks, caches):
        nonlocal verify_s
        begin = perf_counter()
        result = verify(lm, blocks, caches)
        verify_s += perf_counter() - begin
        calls[len(blocks)] += 1
        return result

    def draft(audio):
        begin = perf_counter() - start
        words = drafter(audio)
        drafts.append(dict(start=begin, end=perf_counter() - start, words=len(words)))
        return words

    def diarize(*a, **kw):
        begin = perf_counter() - start
        turns = diarizer(*a, **kw)
        diarizations.append(dict(start=begin, end=perf_counter() - start, turns=turns))
        return turns

    with patch.object(vibevoice_batch, "_verify_blocks", verify_batch), \
            patch.object(pipeline, "drafter", draft), patch.object(pipeline, "diarizer", diarize):
        start = perf_counter()
        tr = pipeline.transcribe(args.audio)
        mx.synchronize()
        wall = perf_counter() - start
    if tr.stats["chunks_failed"] or len(drafts) != tr.stats["n_chunks"]:
        raise RuntimeError("A draft or target chunk failed during the benchmark")
    assert sum(n * count for n, count in calls.items()) == tr.stats["speculative"]["verification_calls"]
    return dict(wall_s=wall, duration_s=tr.duration_s, stats=tr.stats, drafts=drafts,
                diarizations=diarizations, verification_s=verify_s, calls_by_active_chunks=dict(calls),
                segments=tr.to_dict()["segments"])


def vibevoice(args, warm):
    import mlx.core as mx
    from murmur import Murmur
    from murmur.models import vibevoice as vv

    with Murmur.load("vibevoice", backend="mlx", checkpoint=args.checkpoint, speculative="murmur",
                     diarize=True, language="en", chunk_s=args.chunk_s, batch_size=args.batch_size) as pipeline:
        if args.draft_precision == "cpu-int8":
            from murmur.drafting.whisper_drafts import WhisperDrafter
            pipeline.drafter = WhisperDrafter("en", device="cpu", compute_type="int8")
        elif args.draft_precision in ("int4", "int8"):
            import mlx.nn as nn
            nn.quantize(pipeline.drafter.model, bits=int(args.draft_precision[3:]), group_size=64,
                        class_predicate=lambda _, layer: isinstance(layer, (nn.Linear, nn.Embedding)))
            mx.eval(pipeline.drafter.model.parameters())
        mx.random.seed(27)
        pipeline.transcribe(warm, sr=16000)
        mx.synchronize()
        print("VibeVoice warm-up complete", flush=True)

        mx.clear_cache()
        mx.random.seed(48)
        if args.batch_size > 1:
            return profile_batch(pipeline, args)

        chunks, drafts, diarizations = [], [], []
        current = {}
        target = pipeline.client.model
        draft_fn, diar_fn = pipeline.drafter, pipeline.diarizer
        decode_fn = pipeline.client.decode_with_draft
        encode_fn = type(target).encode_speech
        loop_fn = vv.verify_greedy

        def elapsed():
            return perf_counter() - start

        def draft(*a, **kw):
            begin = elapsed()
            words = draft_fn(*a, **kw)
            drafts.append(dict(start=begin, end=elapsed(), words=len(words)))
            return words

        def diarize(*a, **kw):
            begin = elapsed()
            turns = diar_fn(*a, **kw)
            diarizations.append(dict(start=begin, end=elapsed(), turns=turns))
            print("Diarization finished", round(elapsed(), 2), "s", flush=True)
            return turns

        def decode(*a, **kw):
            current.clear()
            current["start"] = elapsed()
            reply = decode_fn(*a, **kw)
            current["end"] = elapsed()
            current["speculative"] = reply.speculative
            chunks.append(dict(current))
            print("Chunk", len(chunks), "finished", round(elapsed(), 2), "s", flush=True)
            return reply

        def encode(model, waveform):
            mx.eval(waveform)
            current["encoder_start"] = elapsed()
            encoded = encode_fn(model, waveform)
            mx.eval(encoded)
            current["encoder_end"] = elapsed()
            return encoded

        def loop(first, proposer, verify, rewind, eos, budget):
            # The caller reads the first greedy token with .item() before this point.
            current["prefill_end"] = elapsed()
            times = Counter()
            lengths = Counter()
            accepted = Counter()
            propose_fn = proposer.propose

            def propose(*a):
                before = perf_counter()
                proposal = propose_fn(*a)
                times["propose"] += perf_counter() - before
                return proposal

            def verify_timed(block):
                before = perf_counter()
                predictions = verify(block)  # .tolist() synchronizes the target forward.
                times["verify"] += perf_counter() - before
                lengths[len(block) - 1] += 1
                match = next((i for i, (p, d) in enumerate(zip(predictions, block[1:])) if p != d), len(block) - 1)
                accepted[match] += 1
                return predictions

            def rewind_timed(count):
                before = perf_counter()
                rewind(count)
                times["rewind"] += perf_counter() - before

            with patch.object(proposer, "propose", propose):
                result = loop_fn(first, proposer, verify_timed, rewind_timed, eos, budget)
            current.update(loop_end=elapsed(), loop_s=dict(times), draft_lengths=dict(lengths),
                           accepted_lengths=dict(accepted))
            return result

        with ExitStack() as patches:
            for obj, attr, fn in ((pipeline, "drafter", draft), (pipeline, "diarizer", diarize),
                                  (pipeline.client, "decode_with_draft", decode),
                                  (type(target), "encode_speech", encode), (vv, "verify_greedy", loop)):
                patches.enter_context(patch.object(obj, attr, fn))
            start = perf_counter()
            tr = pipeline.transcribe(args.audio)
            wall = elapsed()
        assert tr.segments and tr.stats["chunks_failed"] == 0
        assert sum(sum(c["draft_lengths"].values()) for c in chunks) == tr.stats["speculative"]["verification_calls"]
        return dict(wall_s=wall, chunks=chunks, drafts=drafts, diarizations=diarizations, stats=tr.stats,
                    duration_s=tr.duration_s, segments=len(tr.segments))


def whisperx(args, warm):
    import pandas as pd
    import torch
    import whisperx as wx

    data = json.loads(args.timeline.read_text())
    if data["audio_sha256"] != args.audio_hash:
        raise ValueError("The saved timeline was measured on a different recording")
    torch.set_num_threads(8)
    model = wx.load_model(args.checkpoint, "mps", language="en", compute_type="float16", threads=8)
    align, meta = wx.load_align_model(language_code="en", device="mps")
    timeline = data["diarizations"][-1]["turns"]
    vad_type = type(model.vad_model)
    vad_fn = vad_type.__call__

    def run(audio):
        start = perf_counter()
        vad_s = 0.0

        def timed_vad(vad, *args, **kw):
            nonlocal vad_s
            before = perf_counter()
            result = vad_fn(vad, *args, **kw)
            torch.mps.synchronize()
            vad_s += perf_counter() - before
            return result

        with patch.object(vad_type, "__call__", timed_vad):
            output = model.transcribe(audio, batch_size=16)
        torch.mps.synchronize()
        asr = perf_counter()
        aligned = wx.align(output["segments"], align, meta, audio, "mps")
        torch.mps.synchronize()
        aligned_at = perf_counter()
        result = wx.assign_word_speakers(pd.DataFrame(timeline, columns=["start", "end", "speaker"]), aligned)
        end = perf_counter()
        return dict(asr_s=asr-start, asr_vad_s=vad_s, alignment_s=aligned_at-asr, assignment_s=end-aligned_at,
                    segments=len(result["segments"]))
    run(warm)
    print("WhisperX warm-up complete", flush=True)
    start = perf_counter()
    audio = wx.load_audio(str(args.audio))
    loaded = perf_counter()
    stages = run(audio)
    wall = perf_counter() - start
    assert stages["segments"]
    shared = sum(d["end"] - d["start"] for d in data["diarizations"])
    return dict(stages=stages, audio_load_s=loaded-start, pipeline_s=wall, shared_diarization_s=shared,
                wall_with_shared_diarization_s=wall+shared, duration_s=len(audio)/16000)


if __name__ == "__main__":
    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=["vibevoice", "whisperx"])
    parser.add_argument("audio", type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--timeline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1, help="maximum VibeVoice target batch size")
    parser.add_argument("--chunk-s", type=float, default=330, help="VibeVoice target chunk length in seconds")
    parser.add_argument("--draft-precision", choices=["fp16", "int8", "int4", "cpu-int8"], default="fp16",
                        help="Whisper drafter used by VibeVoice; cpu-int8 requires the drafts extra")
    args = parser.parse_args()
    if args.batch_size < 1 or args.chunk_s <= 0:
        parser.error("batch size and chunk length must be positive")
    if args.system == "whisperx" and not args.timeline:
        parser.error("whisperx requires the matching --timeline profile")
    args.audio_hash = hashlib.sha256(args.audio.read_bytes()).hexdigest()
    warm, sr = sf.read(args.audio, frames=60*16000, dtype="float32")
    if sr != 16000 or warm.ndim != 1:
        parser.error("supply a mono 16 kHz WAV")
    result = (vibevoice if args.system == "vibevoice" else whisperx)(args, warm)
    result.update(system=args.system, audio=args.audio.name, audio_sha256=args.audio_hash)
    if args.system == "vibevoice":
        result.update(batch_size=args.batch_size, chunk_s=args.chunk_s, draft_precision=args.draft_precision)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Saved", args.output, flush=True)
