"""Time Murmur's pyannote diarizer on in-memory audio and score it against RTTM.

Example: python benchmarks/pyannote_mac.py meeting.wav meeting.rttm --device mps --speakers 4 --output run.json
Install eval and the backend extra (diarize for PyTorch or mlx-diarize for MLX).
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
    parser.add_argument("--speakers", type=int, help="Known speaker count, if available")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1 or args.threads < 1 or (args.speakers is not None and args.speakers < 1):
        parser.error("runs, threads and speaker count must be positive")

    torch.set_num_threads(args.threads)
    waveform = load_audio(args.audio)
    reference = annotation(parse_rttm(args.rttm))
    if not reference:
        parser.error("expected a nonempty RTTM reference")
    if args.device == "mlx":
        import mlx.core as mx
        from murmur.diarization.mlx import MLXDiarizer
        diarizer = MLXDiarizer()
        device = "mlx"
        pipe = diarizer
        placements = dict.fromkeys(("segmentation", "embedding", "clustering"), "mlx:gpu")
    else:
        diarizer = Diarizer(device=args.device)
        device = diarizer.device.type
        pipe = diarizer.pipe
        placements = {"segmentation": str(next(pipe._segmentation.model.parameters()).device),
                      "embedding": str(next(pipe._embedding.model_.parameters()).device)}
    print(f"Running {args.audio.name} on {device} with {placements}", flush=True)
    diarizer(waveform[:20 * SR], num_speakers=args.speakers)
    result = dict(audio=str(args.audio), reference=str(args.rttm), device=device, placements=placements,
                  model="pyannote/speaker-diarization-3.1",
                  torch=torch.__version__, threads=args.threads, speakers=args.speakers,
                  duration_s=len(waveform) / SR, segmentation_batch=pipe.segmentation_batch_size,
                  embedding_batch=pipe.embedding_batch_size, runs=[])

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

        synchronize()
        start = perf_counter()
        timeline = diarizer(waveform, num_speakers=args.speakers, hook=hook)
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
