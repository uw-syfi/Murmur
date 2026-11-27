# Continuous integration

## Workflows

| Workflow | Runs | Checks |
|---|---|---|
| CI | every pull request and push to `main` | CPU tests on Python 3.10 and 3.12; the built wheel installs in a clean environment |
| Hardware inference, `smoke` | nightly | Voxtral Mini INT4 on Apple Silicon; VibeVoice with speculative decoding on CUDA |
| Hardware inference, `full` | manually, before tagging a release | all 12 MLX checkpoints and all four vLLM models |
| Hardware inference, `benchmark` | weekly, or manually | tcpWER, failed chunks and end-to-end latency for VibeVoice on AMI-IHM ES2004a (workflow summary and artifacts) |
| Release | when a release is published | reruns the enabled hardware suites on the release commit, then uploads to PyPI the same wheel that passed the installation check |

Run the manual modes from **Run workflow** on the Hardware inference workflow.

## Set up hardware runners

Each hardware suite turns on separately: provision a dedicated runner, set its environment for the
runner service, then set the repository variable to `true`.

| Runner label | Machine | Repository variable |
|---|---|---|
| `murmur-mlx` | macOS ARM64 with Metal and enough unified memory for Voxtral Small BF16 (about 48 GB for weights plus inference memory) | `MURMUR_MLX_CI` |
| `murmur-cuda` | Linux x64 GPU sized for Voxtral Small and Whisper | `MURMUR_CUDA_CI` |

| Variable | Runner | Value |
|---|---|---|
| `MURMUR_CI_PYTHON` | both | Absolute path to the Python executable. MLX: Python 3.12 with the three MLX runtimes from the Apple Silicon setup and the `test` extra. CUDA: the client Python with the `all` and `test` extras. |
| `MURMUR_VLLM_VIBEVOICE`, `MURMUR_VLLM_MOSS`, `MURMUR_VLLM_VOXTRAL` | CUDA | Each model's vLLM executable |
| `MURMUR_AMI_ES2004A` | both | Path to AMI `ES2004a.Mix-Headset.wav` |
| `MURMUR_DATA` | CUDA | Benchmark dataset root with the ES2004a audio and STM/RTTM references |

On CUDA, the runner user also needs Hugging Face access to pyannote.

- Keep the benchmark on the same GPU so timings stay comparable.
- Hardware jobs run the checked-out source against these provisioned dependencies. Use a runner
  group restricted to the trusted hardware workflow, or isolated ephemeral runners.

## Run the CI suites locally

`scripts/test.sh` runs the same commands as the workflows. It uses `python` from your
environment; set `PYTHON` to use another interpreter. Extra arguments go to pytest.

```bash
scripts/test.sh            # CPU tests, as on every pull request
scripts/test.sh package    # build the wheel and smoke-test it in a fresh virtualenv
scripts/test.sh cpu -k stitch -x
```

| Suite | Runs | Needs |
|---|---|---|
| `cpu` (default) | CPU tests from CI | the `test` extra |
| `package` | wheel build, `twine check`, clean install and `murmur --help` | network access for the CPU PyTorch wheels |
| `mlx` / `mlx-full` | Voxtral Mini INT4 / all 12 MLX checkpoints | `MURMUR_AMI_ES2004A`, the three MLX runtimes from the [Apple Silicon setup](install/mlx.md); `mlx-full` needs about 48 GB of memory for the 24B BF16 checkpoint |
| `cuda` / `cuda-full` | VibeVoice / all four models on vLLM | `MURMUR_AMI_ES2004A`, the `MURMUR_VLLM_*` executables from the [vLLM setup](install/vllm.md) |
| `llama-cpu` | llama.cpp managed pipeline on CPU (Voxtral) | `MURMUR_AMI_ES2004A`, the [CPU-only llama.cpp setup](install/llama-cpp.md#cpu-only-linux); `MURMUR_TEST_LLAMA_GGUF` and `MURMUR_TEST_LLAMA_MMPROJ` select local weights |
| `benchmark` | VibeVoice tcpWER and latency on AMI-IHM ES2004a, failing on any failed chunk | `MURMUR_DATA`, `MURMUR_VLLM_VIBEVOICE`, pyannote access |

`MURMUR_AMI_ES2004A` is the path to AMI's `ES2004a.Mix-Headset.wav`.
