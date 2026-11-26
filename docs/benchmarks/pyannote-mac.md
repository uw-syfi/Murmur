# Pyannote 3.1 diarization on Apple M4 Max

These measurements come from an experimental Apple GPU build of Murmur. The comparison pipeline uses Reverb Diarization V1 for segmentation on the CPU and NeMo TitaNet Small for speaker embeddings on the GPU.

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

## Method

Both captures cover the same 1,049.35-second AMI ES2004a meeting. Each device ran one timed pass per microphone after a 20-second warm-up. Timing starts with a mono 16 kHz waveform in memory and ends when Murmur returns speaker turns. Model loading, file I/O, scoring and transcription are outside that interval. GPU stage boundaries were synchronized.

The machine has an Apple M4 Max and 128 GB of unified memory. Measurements used Python 3.12, PyTorch 2.8.0, pyannote.audio 4.1.1 and four Torch CPU threads. The official 3.1 pipeline uses pyannote segmentation-3.0, WeSpeaker ResNet34-LM speaker embeddings and agglomerative clustering. Both neural networks used MPS for the GPU runs. The released pipeline uses a batch size of 32 for both stages.

Both diarizers received the known count of four speakers. Diarization error rate (DER) uses the same reference RTTM and pyannote.metrics settings as the earlier Reverb/TitaNet benchmark, including overlapping speech and a 0.25-second collar. The microphones are two views of a single meeting.

## VibeVoice CLI check

The regular `murmur transcribe` command ran on a 20-second headset excerpt with MLX VibeVoice BF16, Whisper drafts, automatic pyannote device selection and automatic speaker counting. It returned a speaker-labeled transcript with zero failed chunks. Pyannote took 1.23 s. The speculative decoder accepted 23 of 84 proposed tokens across 62 verification calls. This integration check used the project environment and was timed separately from the full-meeting diarizer measurements.
