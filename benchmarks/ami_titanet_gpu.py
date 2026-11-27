"""Benchmark Reverb + batched TitaNet on a mono 16 kHz AMI recording on Apple silicon.

Requires sherpa-onnx 2.0.3, onnx, onnxruntime, mlx, numpy and soundfile.
Run from the repo root: python benchmarks/ami_titanet_gpu.py AUDIO.wav --output run.json
"""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import sherpa_onnx as sherpa
import soundfile as sf

WINDOW = 160000
HOP = 16000
FRAME_SHIFT = 270
SAMPLE_RATE = 16000
POWERSET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                     [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.int32)


def cpu_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def segment(audio, model):
    classes = []
    n_windows = max(1, math.ceil((len(audio) - WINDOW) / HOP) + 1)
    for i in range(n_windows):
        x = audio[i * HOP:i * HOP + WINDOW]
        if len(x) < WINDOW:
            x = np.pad(x, (0, WINDOW - len(x)))
        scores, = model.run(["y"], {"x": x[None, None, :]})
        classes.append(scores[0].argmax(axis=-1))
    return np.stack(classes)


def speaker_features(audio, classes, extractor):
    features, pairs, sample_counts = [], [], []
    for window_id, local in enumerate(classes):
        for speaker in range(3):
            # Singleton powerset classes contain clean speech for this speaker.
            mask = local == speaker + 1
            if mask.sum() < 10:
                continue
            edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
            edges = np.minimum(edges, len(local) - 1).astype(np.float32)
            samples = (edges / len(local) * WINDOW + np.float32(window_id * HOP)).astype(int)
            crop = np.concatenate([audio[a:b] for a, b in samples.reshape(-1, 2)])
            frames = 1 + (len(crop) - 400) // 160  # 25 ms window, 10 ms shift
            if frames < 1:
                continue
            stream = extractor.create_stream()
            stream.accept_waveform(SAMPLE_RATE, crop)
            stream.input_finished()
            feat = np.asarray(stream.get_frames(0, frames), np.float32).reshape(frames, 80)
            feat -= feat.mean(axis=0)
            feat /= np.sqrt(np.mean(feat * feat, axis=0)) + 1e-5
            features.append(feat)
            pairs.append((window_id, speaker))
            sample_counts.append(len(crop))
    return features, pairs, np.array(sample_counts)


def speaker_timeline(classes, embeddings, pairs, sample_counts, n_samples, speakers):
    if not len(embeddings):
        return []
    # Match sherpa 2.0.3's 20% minimum active ratio for clustering.
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    cluster = sherpa.FastClustering(sherpa.FastClusteringConfig(num_clusters=speakers))
    keep = sample_counts >= 0.2 * WINDOW
    if max(2, speakers) < keep.sum() < len(keep):
        retained = embeddings[keep]
        initial = np.array(cluster(retained))
        centers = np.stack([retained[initial == i].sum(axis=0) for i in range(initial.max() + 1)])
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        groups = (embeddings @ centers.T).argmax(axis=1)
    else:
        groups = np.array(cluster(embeddings))
    n_groups = groups.max() + 1
    assignments = np.zeros((len(classes), 3, n_groups), np.int32)
    for (window, speaker), group in zip(pairs, groups):
        assignments[window, speaker, group] = 1

    size = (WINDOW + (len(classes) - 1) * HOP) // FRAME_SHIFT + 1
    votes = np.zeros((size, n_groups), np.int32)
    coverage = np.zeros(size, np.int32)
    speaker_counts = np.zeros(size, np.int32)
    for i, (local, assignment) in enumerate(zip(classes, assignments)):
        start = int(np.float32(i * HOP) / np.float32(FRAME_SHIFT) + 0.5)
        region = slice(start, start + len(local))
        active = POWERSET[local]
        votes[region] += (active @ assignment > 0)
        speaker_counts[region] += active.sum(axis=1)
        coverage[region] += 1
    count = (speaker_counts / np.maximum(1, coverage) + 0.5).astype(int)
    ranking = (-votes).argsort(axis=1, kind="stable").argsort(axis=1)
    active = ranking < count[:, None]
    if (n_samples - WINDOW) % HOP:
        active = active[:n_samples // FRAME_SHIFT]
    turns = []
    for group, mask in enumerate(active.T):
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        edges = np.minimum(edges, len(mask) - 1).astype(np.float32)
        times = edges * np.float32(FRAME_SHIFT / SAMPLE_RATE) + np.float32(991 / (2 * SAMPLE_RATE))
        merged = []
        for start, end in times.reshape(-1, 2).tolist():
            if merged and start - merged[-1][1] < 0.5:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        turns.extend([s, e, f"S{group}"] for s, e in merged if e - s > 0.3)
    return sorted(turns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--segmentation", default="models/sherpa-onnx-reverb-diarization-v1/model.int8.onnx")
    parser.add_argument("--segmentation-device", choices=("cpu", "mlx"), default="cpu",
                        help="Reverb INT8 inference backend (default: ONNX CPU)")
    parser.add_argument("--segmentation-batch", type=int, default=32, help="Windows per MLX Reverb batch")
    parser.add_argument("--titanet", default="models/nemo_en_titanet_small.onnx")
    parser.add_argument("--speakers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cpu", action="store_true", help="Use serial ONNX TitaNet as a matched control")
    mode.add_argument("--packed", action="store_true",
                      help="Concatenate unequal-length TitaNet inputs on GPU without batch padding")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audio, rate = sf.read(args.audio, dtype="float32")
    if rate != SAMPLE_RATE or audio.ndim != 1 or len(audio) <= WINDOW:
        parser.error("This meeting benchmark expects mono 16 kHz audio longer than 10 seconds")
    if args.batch < 1 or args.segmentation_batch < 1 or args.speakers < 1:
        parser.error("batch size and speaker count must be positive")

    if args.segmentation_device == "mlx":
        if __package__:
            from .reverb_mlx import ReverbInt8
        else:
            from reverb_mlx import ReverbInt8
        segmentation = ReverbInt8(args.segmentation)

        def segment_audio(samples):
            return segmentation.segment(samples, args.segmentation_batch)
    else:
        segmentation = cpu_session(args.segmentation)

        def segment_audio(samples):
            return segment(samples, segmentation)

    extractor = sherpa.SpeakerEmbeddingExtractor(sherpa.SpeakerEmbeddingExtractorConfig(model=args.titanet))
    if args.cpu:
        model = cpu_session(args.titanet)
        device = "CPU"

        def embed(features):
            vectors = []
            for feature in features:
                inputs = {"audio_signal": feature.T[None],
                          "length": np.array([len(feature)], np.int64)}
                output, = model.run(["embs"], inputs)
                vectors.append(output[0])
            return np.asarray(vectors)
    else:
        import mlx.core as mx
        if __package__:
            from .titanet_mlx import TitaNetSmall
        else:
            from titanet_mlx import TitaNetSmall

        mx.set_default_device(mx.gpu)
        model = TitaNetSmall(args.titanet)
        device = mx.device_info()["device_name"]

        def embed(features):
            return model.embed(features, args.batch, packed=args.packed)

    def run(samples):
        start = perf_counter()
        classes = segment_audio(samples)
        segmented = perf_counter()
        features, pairs, counts = speaker_features(samples, classes, extractor)
        prepared = perf_counter()
        vectors = embed(features)
        embedded = perf_counter()
        turns = speaker_timeline(classes, vectors, pairs, counts, len(samples), args.speakers)
        finished = perf_counter()
        return dict(segmentation_s=segmented - start, features_s=prepared - segmented,
                    titanet_s=embedded - prepared, postprocess_s=finished - embedded,
                    total_s=finished - start, excerpts=len(pairs), timeline=turns)

    run(audio[:20 * SAMPLE_RATE])
    result = run(audio)
    result.update(audio=str(args.audio), device=device, segmentation_device=args.segmentation_device,
                  segmentation_batch=args.segmentation_batch if args.segmentation_device == "mlx" else 1,
                  batch_size=1 if args.cpu else args.batch, packed=args.packed,
                  onnxruntime_version=ort.__version__)
    args.output.write_text(json.dumps(result, indent=2))
    print({k: v for k, v in result.items() if k != "timeline"}, flush=True)


if __name__ == "__main__":
    main()
