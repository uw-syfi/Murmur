"""Time Murmur's diarizers on in-memory audio and score it against RTTM.

Example: python benchmarks/pyannote_mac.py meeting.wav meeting.rttm --device mps --speakers 4 --output run.json
Install eval and the model extra (diarize, mlx-diarize, or mlx-nemotron).
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from meeting_data import parse_rttm
from murmur.audio import Diarizer, SR, load_audio


def annotation(turns):
    result = Annotation()
    for track, (start, end, speaker) in enumerate(turns):
        result[Segment(start, end), track] = speaker
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("rttm", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "mlx"), help="Default to automatic PyTorch device selection")
    parser.add_argument("--diarizer", choices=("pyannote", "nemotron"), default="pyannote")
    parser.add_argument("--speakers", type=int, help="Known speaker count, if available")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1 or args.threads < 1 or (args.speakers is not None and args.speakers < 1):
        parser.error("runs, threads and speaker count must be positive")

    if args.diarizer == "nemotron" and (args.device != "mlx" or args.speakers is not None):
        parser.error("Nemotron requires --device mlx and infers the speaker count")

    torch.set_num_threads(args.threads)
    waveform = load_audio(args.audio)
    reference = annotation(parse_rttm(args.rttm))
    if not reference:
        parser.error("expected a nonempty RTTM reference")
    if args.device == "mlx":
        import mlx.core as mx
        device = "mlx"
        if args.diarizer == "nemotron":
            from murmur.diarization.nemotron_mlx import MLXNemotronDiarizer
            diarizer = MLXNemotronDiarizer()
            placements = {"sortformer": "mlx:gpu"}
        else:
            from murmur.diarization.mlx import MLXDiarizer
            diarizer = MLXDiarizer()
            placements = dict.fromkeys(("segmentation", "embedding", "clustering"), "mlx:gpu")
        pipe = diarizer
    else:
        diarizer = Diarizer(device=args.device)
        device = diarizer.device.type
        pipe = diarizer.pipe
        placements = {"segmentation": str(next(pipe._segmentation.model.parameters()).device),
                      "embedding": str(next(pipe._embedding.model_.parameters()).device)}
    print(f"Running {args.audio.name} on {device} with {placements}", flush=True)
    options = {"num_speakers": args.speakers} if args.diarizer == "pyannote" else {}
    diarizer(waveform[:20 * SR], **options)
    result = dict(audio=str(args.audio), reference=str(args.rttm), device=device, placements=placements,
                  model=diarizer.MODEL if args.diarizer == "nemotron" else "pyannote/speaker-diarization-3.1",
                  torch=torch.__version__, threads=args.threads, speakers=args.speakers,
                  duration_s=len(waveform) / SR, runs=[])
    if args.diarizer == "pyannote":
        result.update(segmentation_batch=pipe.segmentation_batch_size, embedding_batch=pipe.embedding_batch_size)
    else:
        result.update(revision=diarizer.REVISION, mlx_audio=version("mlx-audio"))

    if device == "mlx":
        result["mlx"] = version("mlx")
    else:
        result["pyannote_audio"] = version("pyannote.audio")

    def synchronize():
        if device == "mlx":
            mx.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(args.runs):
        starts, stages = {}, {}

        def hook(step_name, step_artifact, *, completed=None, **kwargs):
            if step_name not in ("segmentation", "embeddings"):
                return
            if completed == 0:
                synchronize()
                starts[step_name] = perf_counter()
            elif completed is None and step_name in starts:
                synchronize()
                stages[step_name] = perf_counter() - starts.pop(step_name)

        if args.diarizer == "pyannote":
            options["hook"] = hook
        synchronize()
        start = perf_counter()
        timeline = diarizer(waveform, **options)
        synchronize()
        total_s = perf_counter() - start
        metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
        der = metric(reference, annotation(timeline)) * 100
        row = dict(total_s=total_s, stages_s=stages, other_s=total_s - sum(stages.values()),
                   der_percent=der, turns=len(timeline), timeline=timeline)
        result["runs"].append(row)
        args.output.write_text(json.dumps(result, indent=2))
        print({k: v for k, v in row.items() if k != "timeline"}, flush=True)


if __name__ == "__main__":
    main()
