# Transcribe recordings

Activate the environment from the installation guide for your machine. The examples
use MOSS, which produces timestamped speaker turns on both backends.
[Apple Silicon installation](install/mlx.md) · [NVIDIA GPU installation](install/vllm.md)

## From the terminal

For Apple Silicon, select `mlx`. For an NVIDIA GPU, change the backend to `vllm`.
Replace `meeting.wav` with your audio file.

```bash
murmur transcribe meeting.wav --model moss --backend mlx --no-diarize
```

For JSON, add `--json`. To save one transcript, redirect standard output to a file.

```bash
murmur transcribe meeting.wav --model moss --backend mlx --no-diarize --json > meeting.json
```

You can supply several paths in one invocation to keep the model loaded. With
`--json`, each recording produces a separate JSON object. Run
`murmur transcribe --help` for all CLI options.

## From Python

Set `backend` to the runtime you installed. Keep the model context open to reuse it
for subsequent files. Each call returns a `Transcript`.

```python
from murmur import Murmur

with Murmur.load("moss", backend="mlx", diarize=False) as model:
    transcript = model.transcribe("meeting.wav")
    print(transcript)       # One line per speaker turn
    print(transcript.text)  # The words in time order
```

Use `transcript.segments` to iterate over turns, `transcript.speakers` for the
distinct speaker labels, or `transcript.to_dict()` for the same fields as the CLI JSON.

| JSON field | Meaning |
|---|---|
| `duration_s` | Total duration of the input recording in seconds. |
| `segments` | Speaker turns in time order. Each has a `speaker` label, `start` and `end` in seconds from the start of the recording, and the transcribed `text`. |
| `stats` | Processing details, including the number of chunks, the number flagged as failed, and time spent in each stage. |

## Follow speakers across chunks

With `--no-diarize` (Python: `diarize=False`), MOSS assigns speaker labels within
each chunk. A label such as `c002:S01` identifies a speaker in that chunk. For
consistent labels across an entire recording, configure pyannote as described in
your installation guide and omit the flag. Murmur uses its timeline to link
speaker labels between chunks. If you already have a timeline, pass a list of
`(start_seconds, end_seconds, speaker_label)` tuples to
`model.transcribe("meeting.wav", timeline=timeline)` with `diarize=False`.

## Adjust decoding

Murmur targets 330-second chunks by default. Set `--chunk-s` to change that
target. On vLLM, `--batch-size` controls how many chunks decode concurrently
(default 8). MLX processes one chunk at a time. Set `--language en` or
`--language zh` to select the draft language when using draft windows.

Use `--single-pass` (Python: `model.transcribe("meeting.wav", single_pass=True)`)
to run the baseline. Recordings longer than the model's audio limit are still
split to fit. [Model limits](../README.md#supported-models)
