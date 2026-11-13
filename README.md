# Murmur: Efficient Long-Context ASR Inference with Hierarchical Chunking

[![arXiv](https://img.shields.io/badge/arXiv-2606.01483-b31b1b.svg)](https://arxiv.org/abs/2606.01483)

**News**
- 2026/11/04: [Preprint v2](https://arxiv.org/abs/2606.01483v2) is out.
- 2026/06/01: [Preprint](https://arxiv.org/abs/2606.01483v1) is released on arXiv.

Murmur is an inference system that serves long-context ASR models, such as **VibeVoice-ASR** and
**MOSS-Transcribe-Diarize**, on long audio. These models transcribe an hour-long recording in a single
pass, which is slow and can fail entirely on a repetition loop.

- **Across chunks:** Murmur decodes ~5-minute target chunks in parallel. This intermediate size gives
  the model enough context to match single pass in most cases, and a loop costs at most one chunk.
- **Within each chunk:** hierarchical chunking splits each chunk into 30 s draft windows. A small
  Whisper model transcribes them, and structure-aware drafting adds the timestamps and speaker tags,
  giving drafts for speculative decoding. The long-context model verifies every token, so decoding is
  faster and the output is unchanged.

Overall, Murmur is **2.5x faster** than single pass on average (2.0-2.9x), end to end including
diarization, and lowers tcpWER by 2.0 points on average.

![Hierarchical chunking](https://raw.githubusercontent.com/uw-syfi/Murmur/main/docs/figures/hierarchy.png)

## Demo

Single pass, Murmur and WhisperX transcribing the same 32-minute AMI meeting on one H100, replayed at 8x.

https://github.com/user-attachments/assets/e2897c2a-cc33-4021-8e2d-7dfbac46f8c9

## Results

Murmur against each model's own single pass on four meeting datasets. Each cell is tcpWER / latency
per recording (lower is better), on one H100 with vLLM, with latency including diarization.

| Model | System | AMI-IHM | AMI-SDM | AliMeeting | AISHELL-4 |
|---|---|---|---|---|---|
| VibeVoice-ASR | single pass | 30.1% / 109 s | 39.6% / 100 s | 41.7% / 123 s | **30.3%** / 111 s |
| | **Murmur** | **24.5% / 38 s** | **32.9% / 39 s** | **38.7% / 47 s** | 30.5% / **46 s** |
| MOSS-Transcribe-Diarize | single pass | 17.5% / 54 s | 29.6% / 77 s | **13.6%** / 91 s | **14.1%** / 51 s |
| | **Murmur** | **17.3% / 26 s** | **22.8% / 26 s** | 17.5% / **32 s** | 16.6% / **26 s** |

<sub>Bold: better of single pass and Murmur. WER, cpWER and DER are in the paper. AMI scores use a
corrected reference reader (2-6 points lower than earlier arXiv versions).</sub>

## Supported models

Murmur runs each model on its own vLLM server and talks to it through a small client class.

| Model | `Murmur.load(...)` | Draft windows | vLLM server |
|---|---|---|---|
| [VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) (8.5B) | `"vibevoice"` | on | vLLM 0.14 + VibeVoice plugin |
| [MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) (0.9B) | `"moss"` | off | vLLM nightly |

Both are tested on English and Mandarin. To add a model, add a client class in `murmur/models/`.

## Install

The `murmur` client and each model's vLLM server use separate environments, because the two models
need different vLLM builds.

```bash
conda create -n murmur python=3.10 -y && conda activate murmur
pip install "murmur-asr[all]"
# or, from source: git clone https://github.com/uw-syfi/Murmur.git && pip install -e "Murmur[all]"
huggingface-cli login   # accept the terms of pyannote/speaker-diarization-3.1 first
```

<details>
<summary>vLLM servers (VibeVoice-ASR, MOSS-Transcribe-Diarize)</summary>

```bash
# VibeVoice-ASR: vLLM 0.14 + the VibeVoice plugin
conda create -n vibevoice-vllm python=3.10 -y && conda activate vibevoice-vllm
pip install vllm==0.14.0
git clone https://github.com/microsoft/VibeVoice.git && pip install -e VibeVoice
pip install "murmur-asr[server]"

# MOSS-Transcribe-Diarize: vLLM nightly (tested: 0.23.1rc1.dev949)
conda create -n moss-vllm python=3.10 -y && conda activate moss-vllm
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
pip install "murmur-asr[server]"

# so the client can start the servers itself
export MURMUR_VLLM_VIBEVOICE=$CONDA_PREFIX/../vibevoice-vllm/bin/vllm
export MURMUR_VLLM_MOSS=$CONDA_PREFIX/../moss-vllm/bin/vllm
```
</details>

## Usage

`Murmur.load` starts the model's vLLM server, and `transcribe` returns segments with speakers and
timestamps.

```python
from murmur import Murmur

with Murmur.load("vibevoice") as m:   # or Murmur.load("moss", language="zh")
    print(m.transcribe("meeting.wav"))  # [  12.40-  15.10] SPEAKER_01: ...
```

<details>
<summary>Options, CLI and a separately started server</summary>

`Murmur.load` options: `chunk_s` (default 330), `batch_size` (default 8), `draft_windows`,
`language` (`"en"` or `"zh"`), `diarize`, `gpu`. `transcribe(..., single_pass=True)` runs the
single-pass baseline.

```bash
# in the vibevoice-vllm environment
CUDA_VISIBLE_DEVICES=0 murmur serve vibevoice --port 8000 --draft-dir /tmp/murmur_drafts
# in the murmur environment
murmur transcribe meeting.wav --model vibevoice --url http://localhost:8000 --draft-dir /tmp/murmur_drafts
```
Client and server must share the draft directory (same machine).
</details>

## Benchmarks

`benchmarks/run.py` reproduces the paper's results, one model, dataset and system at a time.

```bash
python benchmarks/run.py --model vibevoice --dataset ami_ihm --system murmur --mode latency
```

<details>
<summary>Datasets, systems and scoring</summary>

```bash
bash benchmarks/prepare/download_ami.sh && python benchmarks/prepare/build_ami_sessions.py   # needs lhotse
python benchmarks/prepare/download_alimeeting_refs.py --archives Eval_Ali.tar.gz Test_Ali.tar.gz --raw_dir /tmp/ali   # OpenSLR 119
python benchmarks/prepare/download_aishell4_refs.py --archive test.tar.gz --raw_dir /tmp/aishell4                    # OpenSLR 111
```

`--system`: `single`, `chunked` (greedy), `murmur` (as in the paper; `--draft-windows on|off`
overrides) or `ngram` (vLLM prompt lookup). `--mode latency` reports end-to-end latency per
recording; `--mode throughput` reports decoding time for a whole dataset. Scoring: WER (CER for
Mandarin) with jiwer, cpWER and tcpWER (5 s collar) with MeetEval, DER with pyannote.metrics.
</details>

## Citation

```bibtex
@article{murmur2026lee,
  title={Murmur: Efficient Long-Context ASR Inference with Hierarchical Chunking},
  author={Lee, Wei-Tzu and Kamahori, Keisuke and Kasikci, Baris},
  journal={arXiv preprint arXiv:2606.01483},
  year={2026},
  url={https://arxiv.org/abs/2606.01483}
}
```
