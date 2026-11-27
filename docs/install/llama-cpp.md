# Run Murmur with llama.cpp (GGUF)

Murmur can run VibeVoice-ASR, Voxtral Mini 3B and Voxtral Small 24B with an audio-capable
`llama-server`. Install llama.cpp (tested with build **b12803**) and put `llama-server` on
`PATH`, set `MURMUR_LLAMA_SERVER`, or pass `--llama-bin`. The server runs the GGUF decoder and
its matching audio projector using the devices available to that llama.cpp build, including Metal on Mac.

```sh
pip install -e .
murmur transcribe meeting.wav --backend llama.cpp --model vibevoice --no-diarize --json
```

This starts a loopback server, downloads the default GGUF pair on first use and releases
the server on exit. VibeVoice supplies segment times and speaker labels itself. The optional
`diarize` extra and pyannote access enable Murmur's cross-chunk speaker linking. For Voxtral,
install `pip install -e '.[drafts]'` for CPU Whisper word timing, including when `--no-diarize`
is set. Voxtral's GGUF target uses the model's transcription tokens and the requested language.

| Murmur model | Default GGUF repository |
|---|---|
| `vibevoice` | [ggml-org/VibeVoice-ASR-GGUF](https://huggingface.co/ggml-org/VibeVoice-ASR-GGUF) |
| `voxtral` | [ggml-org/Voxtral-Mini-3B-2507-GGUF](https://huggingface.co/ggml-org/Voxtral-Mini-3B-2507-GGUF) |
| `voxtral-small` | [bartowski/mistralai_Voxtral-Small-24B-2507-GGUF](https://huggingface.co/bartowski/mistralai_Voxtral-Small-24B-2507-GGUF) |

Use `--checkpoint org/repo:Q4_K_M` to select a Hub quantization, or supply both local files.
The backend uses greedy decoding. `--batch-size` controls concurrent chunks and managed server
slots (default 1). Each additional slot reserves another context. MOSS and speculative draft
verification are available through other Murmur backends.

```python
from murmur import Murmur

with Murmur.load("vibevoice", backend="llama.cpp", checkpoint="model.gguf",
                 mmproj="mmproj.gguf", diarize=False) as pipeline:
    transcript = pipeline.transcribe("meeting.wav")
    print(transcript.text)
```

To reuse a running server across client processes, start it once and connect by URL. For a
third-party llama-server with a custom model alias, pass `--served-name` to the client.

```sh
murmur serve voxtral --backend llama.cpp --port 8080
murmur transcribe meeting.wav --backend llama.cpp --model voxtral --url http://localhost:8080 --no-diarize
```

## CPU-only Linux

Install a CPU build of `llama-server` on `PATH`, or set `MURMUR_LLAMA_SERVER` to its path.
From a [llama.cpp b12803 source checkout](https://github.com/ggml-org/llama.cpp/tree/b12803),
build the CPU server with:

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF
cmake --build build-cpu --target llama-server --parallel
export MURMUR_LLAMA_SERVER="$PWD/build-cpu/bin/llama-server"
```

In the Murmur checkout, install the CPU wheels in a fresh Python environment. Pyannote also
requires FFmpeg's shared libraries and Hugging Face access to its gated speaker-diarization-3.1
pipeline and dependencies.

```sh
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[diarize,drafts]'
murmur transcribe meeting.wav --backend llama.cpp --model voxtral --cpu-only --json
```

| Stage | CPU implementation |
|---|---|
| Target ASR and audio encoder | llama.cpp and libmtmd with a GGUF decoder and audio projector |
| Speaker diarization and chunk boundaries | pyannote 3.1 with CPU PyTorch |
| Voxtral word timestamps | faster-whisper with CPU INT8 CTranslate2, Whisper-tiny for English and Whisper-small for Mandarin |
| Speaker attribution and stitching | Murmur running on the host CPU |

Use `cpu_only=True` in `Murmur.load`. For an existing server, run `murmur serve MODEL --backend llama.cpp --cpu-only`
on the server and use `--cpu-only --url URL` on the client. VibeVoice supplies its own segment timestamps
and speaker labels and can run with `--no-diarize` without the diarization or Whisper extras.
Murmur's Whisper-guided speculative draft verification is not implemented in the llama.cpp backend.

To exercise the full managed pipeline on the deployment machine, set `MURMUR_AMI_ES2004A` to the
ES2004a Mix-Headset WAV, install the test extra and run
`MURMUR_TEST_LLAMA_CPU_MODEL=voxtral python -m pytest -q tests/test_llama_cpp.py`.
Set `MURMUR_TEST_LLAMA_GGUF` and `MURMUR_TEST_LLAMA_MMPROJ` to use local weights for this test.
