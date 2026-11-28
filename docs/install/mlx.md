# Install Murmur on Apple Silicon

Murmur runs MOSS-Transcribe-Diarize, VibeVoice-ASR and both Voxtral sizes locally with MLX. You'll need an Apple Silicon Mac,
Git, and `uv`. From a fresh checkout, create and activate a Python 3.12 environment, then install the runtimes
for the models you plan to use. These commands can share one environment.

```bash
git clone https://github.com/uw-syfi/Murmur.git
cd Murmur
uv venv --python 3.12
source .venv/bin/activate
# VibeVoice and both Voxtral sizes
uv pip install -e '.[mlx-diarize,mlx-vibevoice,mlx-voxtral]'
# MOSS
uv pip install -e '.[mlx-diarize]' 'moss-transcribe-diarize[mlx-runtime] @ git+https://github.com/vanch007/mlx-MOSS-Transcribe-Diarize.git@d2546316f76e93947e8d99a17fb88652f8ad6ab2'
```

The `mlx-diarize` extra runs speaker diarization on the Apple GPU with public weights, so no
Hugging Face token is needed. See [Diarize on the Apple GPU](#diarize-on-the-apple-gpu).

## Transcribe a file

Replace `meeting.wav` with the path to a recording on your Mac.

```bash
murmur transcribe meeting.wav --model moss --backend mlx
murmur transcribe meeting.wav --model vibevoice --backend mlx
murmur transcribe meeting.wav --model voxtral --backend mlx
```

Voxtral (`voxtral` is Mini 3B, `voxtral-small` is 24B) generates text and uses MLX Whisper for word
times, even with draft windows off. Whisper runs on the Apple GPU through mlx-audio. On first use it
downloads `openai/whisper-tiny` for English or `openai/whisper-small` for Mandarin.

The default MOSS checkpoint is about 1.8 GB. The first run also downloads the
diarization models. Murmur prints each speaker turn with its start and end times.
[Default MOSS checkpoint (BF16)](https://huggingface.co/efficient-speech/MOSS-Transcribe-Diarize-MLX-BF16)

## Diarize on the Apple GPU

The MLX backend runs pyannote segmentation, WeSpeaker embeddings, clustering, and turn
reconstruction on the Apple GPU to link speakers across chunks. It downloads 32.5 MB of FP32
weights once and reuses the Hugging Face cache. [Public MLX checkpoints](https://huggingface.co/efficient-speech/Pyannote-Diarization-3.1-MLX-FP32)
Pass `--no-diarize` to skip it; MOSS and VibeVoice then label speakers within each chunk.

On an M4 Max, MLX diarized the 17.5-minute AMI ES2004a meeting in 18.74 s on the headset mix
and 17.79 s on the distant mic. PyTorch MPS took 46.94 s and 46.93 s, respectively, with matching
speaker turns. [Benchmark and scoring details](../benchmarks/pyannote-mac.md)

Install the optional Nemotron runtime with `uv pip install -e '.[mlx-nemotron]'` (Python 3.11+).
Select Nemotron-3 with `--diarizer nemotron` or `Murmur.load(..., backend="mlx", diarizer="nemotron")`.
Murmur downloads the converted weights on first use and runs the eight-speaker Sortformer through
mlx-audio. It infers the speaker count and joins pauses shorter than half a second for each speaker.
[MLX checkpoint](https://huggingface.co/mlx-community/Nemotron-3-Diarization) · [Mac benchmark](../benchmarks/nemotron-mlx.md)

A separate benchmark runs Reverb INT8 segmentation and TitaNet Small embeddings on the Apple GPU.
Feature extraction and clustering run on the CPU. [Experimental setup and accuracy](../benchmarks/reverb-mlx.md)

## Choose a checkpoint or precision

Every model supports `--quantization bf16` (default), `int8` and `int4`. On first use, Murmur
fetches the corresponding checkpoint from [efficient-speech on Hugging Face](https://huggingface.co/efficient-speech);
for example, `voxtral-small` with `int4` loads `efficient-speech/Voxtral-Small-24B-2507-MLX-INT4`.
Python accepts the same values in `Murmur.load(..., quantization="int4")`. The packed MOSS
checkpoints quantize the text backbone and retain the original audio encoder and VQ adaptor.

```bash
murmur transcribe meeting.wav --model vibevoice --backend mlx --quantization int4
```

To load your own **packed** checkpoint, pass its local directory or Hugging Face ID via
`--checkpoint` alone; it loads in its stored precision. For MOSS, you can also quantize explicit
**unquantized** weights in memory by supplying both `--checkpoint` and `--quantization`.
Other presets quantize the selected weights at load time.

Available presets are affine integers (`int1`, `int2`, `int3`, `int4`, `int5`, `int6`,
`int8`, group size 64), `mxfp4` and `mxfp8` (group size 32), and `nvfp4` (group size 16).
The 1–3 bit affine presets produced unusable transcripts in a short English smoke test on an M4 Max.

## Enable speculative decoding

Speculative decoding on MLX is available for MOSS and VibeVoice. Install the draft dependencies and enable draft windows.

```bash
uv pip install -e '.[mlx-drafts]'
murmur transcribe meeting.wav --model moss --backend mlx --draft-windows on --json
```

In Python, use `Murmur.load("moss", backend="mlx", draft_windows=True)`.
MLX Whisper supplies timestamped words to the selected model’s proposer. The MLX model
checks up to eight proposed tokens per pass. JSON output includes proposed and accepted token counts
and the number of verification passes in `stats.speculative`. Both MLX models default to greedy decoding.

For long recordings on Apple silicon, VibeVoice can verify drafts across several target chunks
in one GPU forward pass. MLX decodes one chunk at a time by default.

```bash
murmur transcribe meeting.wav --model vibevoice --backend mlx --draft-windows on --batch-size 4 --json
```

In Python, pass `speculative="murmur", batch_size=4` to `Murmur.load`. The batch size is a
maximum. Each active chunk has its own KV cache, so increasing it uses more memory. A recording
with one target chunk still runs at batch 1. In one warm M4 Max BF16 run on the 17.5-minute AMI
ES2004a headset recording, four chunks took 350.8 s at batch 4 and 529.2 s at batch 1, including
diarization. WER was 19.16% and 19.24%, respectively.

Use the same chunk size and diarization settings with `--draft-windows off` for a latency comparison.
`--single-pass` selects the greedy baseline even when draft windows are enabled.
