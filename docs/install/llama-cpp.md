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
