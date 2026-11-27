# Pyannote 3.1 diarization on Apple M4 Max

## Native MLX diarization

The M4 Max (128 GB) ran each backend once per capture after a 20-second warm-up. These runs used Python 3.12, MLX 0.32.5, PyTorch 2.15.0 and pyannote.audio 4.1.1. Wall time starts with a mono 16 kHz waveform in memory and ends with speaker turns. DER includes overlap and uses a 0.25-second collar.

| AMI capture | Speaker count setting | Speakers returned | PyTorch MPS | Native MLX | Speedup | DER on both | Turns |
|---|---|---:|---:|---:|---:|---:|---:|
| ES2004a headset | 4 supplied | 4 | 46.94 s | 18.74 s | 2.51× | 13.76% | 262 |
| ES2004a distant | 4 supplied | 4 | 46.93 s | 17.79 s | 2.64× | 20.60% | 275 |
| IS1009a headset | Automatic | 4 | 35.66 s | 14.21 s | 2.51× | 14.05% | 239 |
| TS3003a headset | Automatic | 2 | 73.93 s | 29.55 s | 2.50× | 15.46% | 281 |

All four comparisons matched speaker labels and turn boundaries to within one microsecond. Both backends identified two speakers on TS3003a with automatic counting. The reference contains four.

| MLX capture | Segmentation | Feature extraction and embeddings | Clustering, reconstruction and remaining time |
|---|---:|---:|---:|
| ES2004a headset | 1.37 s | 16.27 s | 1.10 s |
| ES2004a distant | 1.60 s | 15.15 s | 1.04 s |
| IS1009a headset | 1.12 s | 12.24 s | 0.85 s |
| TS3003a headset | 5.06 s | 23.20 s | 1.29 s |

MLX runs the PyanNet recurrent layers, filterbank, WeSpeaker ResNet34-LM, masked statistics pooling, centroid linkage and overlap reconstruction on the GPU. Each embedding batch contains eight windows. Local speakers in a window share the same ResNet activations and receive separate pooling masks. The PyTorch reference uses its released batch settings. MLX clips turns in the padded final window to the input audio duration. File I/O, checkpoint loading and DER scoring are outside the timer.

## Earlier CPU and MPS baselines

| AMI ES2004a capture | Pyannote CPU | Pyannote MPS | Speedup | Pyannote DER on both | Reverb/TitaNet GPU | Reverb/TitaNet DER |
|---|---:|---:|---:|---:|---:|---:|
| Headset mix | 640.90 s | 50.49 s | 12.69× | 13.76% | 28.47 s | 65.74% |
| Distant mic (Array1-01) | 641.59 s | 55.26 s | 11.61× | 20.60% | 30.32 s | 60.60% |

CPU and MPS produced identical pyannote timestamps and speaker labels for each recording, with 262 turns on the headset mix and 275 on the distant mic. Pyannote on MPS took 22.03 s and 24.94 s longer than the earlier batch-8 Reverb/TitaNet system, respectively. Its DER was lower by 51.98 and 39.99 percentage points.

| Stage | Headset CPU | Headset MPS | Distant CPU | Distant MPS |
|---|---:|---:|---:|---:|
| Segmentation | 34.81 s | 4.32 s | 34.84 s | 4.10 s |
| Speaker embeddings | 605.76 s | 45.79 s | 606.43 s | 50.88 s |
| Remaining pipeline, including clustering | 0.34 s | 0.39 s | 0.33 s | 0.29 s |

### Method

Both captures cover the same 1,049.35-second AMI ES2004a meeting. Each device ran one timed pass per microphone after a 20-second warm-up. Timing starts with a mono 16 kHz waveform in memory and ends when Murmur returns speaker turns. Model loading, file I/O, scoring and transcription are outside that interval. GPU stage boundaries were synchronized.

The machine has an Apple M4 Max and 128 GB of unified memory. Measurements used Python 3.12, PyTorch 2.8.0, pyannote.audio 4.1.1 and four Torch CPU threads. The official 3.1 pipeline uses pyannote segmentation-3.0, WeSpeaker ResNet34-LM speaker embeddings and agglomerative clustering. Both neural networks used MPS for the GPU runs. The released pipeline uses a batch size of 32 for both stages.

Both diarizers received the known count of four speakers. Diarization error rate (DER) uses the same reference RTTM and pyannote.metrics settings as the earlier Reverb/TitaNet benchmark, including overlapping speech and a 0.25-second collar. The microphones are two views of a single meeting.

## Reproduce

Install the `mlx-diarize` and `eval` extras for the MLX benchmark. The CPU and MPS controls also require the `diarize` extra and access to the gated pyannote repositories. Run the following from the repository root, substituting the waveform and reference paths for the other microphone. Omit `--speakers` to measure automatic speaker counting. The benchmark defaults to the PyTorch device selector. Pass `--device mlx` for the native MLX implementation.

```sh
audio=benchmarks/meeting_datasets/_ami_raw/amicorpus/ES2004a/audio/ES2004a.Mix-Headset.wav
for device in mlx mps cpu; do
    .venv/bin/python benchmarks/pyannote_mac.py "$audio" results/ami-mac/ihm.rttm \
        --device "$device" --speakers 4 --threads 4 --output "results/ami-pyannote-mac/headset.$device.json"
done
```
