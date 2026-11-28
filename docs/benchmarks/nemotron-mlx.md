# Nemotron-3 diarization on Apple M4 Max

Murmur ran both diarizers on the 1,049.35-second ES2004a AMI meeting, using the headset mix and the first distant microphone. The machine had 128 GB of unified memory and used MLX 0.32.5 and mlx-audio 0.6.8. Nemotron used the public BF16 checkpoint, with eight output channels, the checkpoint's offline chunk configuration, a 0.5 activity threshold, and Murmur's 0.5 s per-speaker pause merge. [Checkpoint](https://huggingface.co/mlx-community/Nemotron-3-Diarization/tree/59ed2dbfc1346dcea9d423c71306a3a2499c568f)

## Diarizer latency and accuracy

| Capture | Diarizer | Median time | DER | Speakers | Turns |
| --- | --- | ---: | ---: | ---: | ---: |
| Headset mix | MLX pyannote | 8.364 s | 13.76% | 4 | 262 |
| Headset mix | MLX Nemotron | 0.886 s | 19.66% | 4 | 393 |
| Distant mic | MLX pyannote | 8.502 s | 23.73% | 5 | 276 |
| Distant mic | MLX Nemotron | 0.899 s | 26.37% | 4 | 412 |

Each row is the median of three full-recording calls after a 20-second audio warm-up, from an in-memory mono 16 kHz waveform to speaker turns. GPU work was synchronized around each call. Audio loading, model loading and scoring were outside the timer. Both diarizers inferred the speaker count. All three runs in each row returned identical turns. DER includes overlap and uses a 0.25-second collar against the reference RTTM. The two microphones capture the same four-speaker conversation.

Nemotron ran 9.44× faster on the headset and 9.46× faster on the distant mic. Its DER increased by 5.90 and 2.64 percentage points, respectively.

## End-to-end transcription

Murmur transcribed the full headset recording with MOSS INT4, using the public MLX backend, 330-second target chunks, batch size one, and greedy decoding. Each diarizer ran in a fresh process with the same checkpoint after transcribing the same 20-second excerpt. The table reports one full run per model. The wall timer excludes model loading, file I/O, warm-up, and scoring.

| Diarizer | Diarization | Decoding | End-to-end | WER | Time-constrained speaker-attributed WER (tcpWER) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLX pyannote | 8.43 s | 35.23 s | 43.67 s | 15.21% | 17.49% |
| MLX Nemotron | 0.97 s | 34.10 s | 35.08 s | 15.65% | 13.17% |

Both processed four chunks with zero failures. Nemotron also changed the chunk boundaries and the final speaker assignments. On this recording, total wall time fell by 8.59 s (19.7%), tcpWER fell by 4.31 percentage points, and WER rose by 0.44 points.

## Reproduce the diarizer benchmark

Install the optional eval, mlx-diarize, and mlx-nemotron dependencies. Run the same command for each capture using its matching reference. The output JSON includes per-run wall time, DER and the full speaker timeline.

```bash
for model in pyannote nemotron; do
    python benchmarks/pyannote_mac.py meeting.wav meeting.rttm \
        --device mlx --diarizer "$model" --runs 3 --output "results/$model.json"
done
```
