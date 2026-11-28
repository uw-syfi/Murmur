# Install Murmur on an NVIDIA GPU

Murmur runs the transcription pipeline in a client environment. A second environment
holds the vLLM version that serves your chosen model. The steps below use Conda and
install Murmur from the same checkout in both environments.

## Install the client

```bash
git clone https://github.com/uw-syfi/Murmur.git
cd Murmur
conda create -n murmur python=3.10 -y
conda activate murmur
python -m pip install -e '.[all]'
```

For the published client release, install `murmur-asr[all]` from PyPI. The
examples here follow the source checkout.

Accept the model terms for [speaker diarization](https://huggingface.co/pyannote/speaker-diarization-3.1)
and [segmentation](https://huggingface.co/pyannote/segmentation-3.0), then set `HF_TOKEN`
in your shell to a Hugging Face read token with access to both models.
With pyannote.audio 4.1.1, also accept the [Community-1 terms](https://huggingface.co/pyannote/speaker-diarization-community-1)
for the files that version downloads during setup.

To diarize with [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
instead (`diarizer="nemotron"`, CLI `--diarizer nemotron`), install NeMo 3.1 or newer, which is
not on PyPI yet (3.0.0 cannot load the model). The model is not gated, so it needs no token.
NeMo installs next to pyannote (it downgrades `lightning` to 2.4, which pyannote supports).

```bash
python -m pip install "nemo-toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"
```

## Install the MOSS server and transcribe a file

Run these commands from the Murmur checkout in the same shell. The project tested
MOSS with vLLM nightly build `0.23.1rc1.dev949`.

```bash
conda create -n moss-vllm python=3.10 -y
conda activate moss-vllm
python -m pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
python -m pip install -e '.[server]'
export MURMUR_VLLM_MOSS="$CONDA_PREFIX/bin/vllm"

conda activate murmur
murmur transcribe meeting.wav --model moss --backend vllm
```

Replace `meeting.wav` with the path to your recording. Murmur starts the model
server, prints timestamped speaker turns, and stops the server when the command
exits. In a new shell, set `MURMUR_VLLM_MOSS` again to the absolute path of the
server environment's `vllm` executable.

## Use a different model

**VibeVoice-ASR** uses vLLM 0.14 and the VibeVoice plugin. From the Murmur checkout,
create its server environment, then return to the client to transcribe.

```bash
conda create -n vibevoice-vllm python=3.10 -y
conda activate vibevoice-vllm
python -m pip install vllm==0.14.0
git clone https://github.com/microsoft/VibeVoice.git
python -m pip install -e VibeVoice
python -m pip install -e '.[server]'
export MURMUR_VLLM_VIBEVOICE="$CONDA_PREFIX/bin/vllm"

conda activate murmur
murmur transcribe meeting.wav --model vibevoice --backend vllm
```

**Voxtral Mini and Small** use vLLM 0.14. Set `MURMUR_VLLM_VOXTRAL` to the
vLLM executable in their server environment. Use `--model voxtral` for Mini
or `--model voxtral-small` for Small. Murmur uses Whisper to add word times
and pyannote to assign speakers to Voxtral's text. [Model details](../../README.md#supported-models)

## Connect to an existing vLLM server

To keep the VibeVoice server running across CLI invocations, run this in a separate
terminal on the same machine:

```bash
conda activate vibevoice-vllm
CUDA_VISIBLE_DEVICES=0 murmur serve vibevoice --port 8000 --draft-dir /tmp/murmur_drafts
```

In the client terminal, pass the server URL and the same directory so the server can
read the client's drafts:

```bash
conda activate murmur
murmur transcribe meeting.wav --model vibevoice --url http://localhost:8000 --draft-dir /tmp/murmur_drafts
```
