# Install Murmur on Apple Silicon

Murmur runs MOSS-Transcribe-Diarize locally with MLX. You'll need an Apple Silicon Mac,
Git, and `uv`. From a fresh checkout, create and activate a Python 3.12 environment.

```bash
git clone https://github.com/uw-syfi/Murmur.git
cd Murmur
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[diarize]' 'moss-transcribe-diarize[mlx-runtime] @ git+https://github.com/vanch007/mlx-MOSS-Transcribe-Diarize.git@d2546316f76e93947e8d99a17fb88652f8ad6ab2'
```

Accept the model terms for [speaker diarization](https://huggingface.co/pyannote/speaker-diarization-3.1)
and [segmentation](https://huggingface.co/pyannote/segmentation-3.0), then set `HF_TOKEN`
in your shell to a Hugging Face read token with access to both models.
With pyannote.audio 4.1.1, also accept the [Community-1 terms](https://huggingface.co/pyannote/speaker-diarization-community-1)
for the files that version downloads during setup.

## Transcribe a file

Replace `meeting.wav` with the path to a recording on your Mac.

```bash
murmur transcribe meeting.wav --model moss --backend mlx
```

The default MOSS checkpoint is about 1.8 GB. The first run also downloads the
diarization models. Murmur prints each speaker turn with its start and end times.
[Default MOSS checkpoint](https://huggingface.co/vanch007/mlx-MOSS-Transcribe-Diarize)

## Choose a checkpoint or precision

Use `--quantization int4` or `--quantization int8` to download the existing packed MOSS
checkpoint directly. Python accepts the same values in `Murmur.load(..., quantization="int4")`.
Both checkpoints quantize the text backbone and retain the original audio encoder and VQ adaptor.

```bash
murmur transcribe meeting.wav --model moss --backend mlx --quantization int4
```

To load your own **packed** checkpoint, pass its local directory or Hugging Face ID via
`--checkpoint` alone. To quantize explicit **unquantized** weights in memory, supply both
`--checkpoint` and `--quantization`. Other presets quantize the selected weights at load time.

Available presets are affine integers (`int1`, `int2`, `int3`, `int4`, `int5`, `int6`,
`int8`, group size 64), `mxfp4` and `mxfp8` (group size 32), and `nvfp4` (group size 16).
The 1–3 bit affine presets produced unusable transcripts in a short English smoke test on an M4 Max.

## Enable speculative decoding

Install the draft dependencies and enable draft windows.

```bash
uv pip install -e '.[drafts]'
murmur transcribe meeting.wav --model moss --backend mlx --draft-windows on --json
```

In Python, use `Murmur.load("moss", backend="mlx", draft_windows=True)`.
Whisper runs on the CPU and supplies timestamped words to Murmur’s MOSS proposer. The MLX model
checks up to eight proposed tokens per pass. JSON output includes proposed and accepted token counts
and the number of verification passes in `stats.speculative`. The default for MOSS is greedy decoding.
Use the same chunk size and diarization settings with `--draft-windows off` for a latency comparison.
`--single-pass` selects the greedy baseline even when draft windows are enabled.
