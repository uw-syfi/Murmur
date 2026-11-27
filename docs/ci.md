# Continuous integration

Pull requests run CPU tests on Python 3.10 and 3.12 and install the built wheel in a clean environment. The release workflow publishes the same wheel that passed the installation check.

The **Hardware inference** workflow runs Voxtral Mini INT4 on Apple Silicon and VibeVoice with speculative decoding on CUDA each night. Select `full` in **Run workflow** to test all 12 MLX checkpoints and all four vLLM models before tagging a release. Publishing a release repeats the enabled suites against the release commit before uploading distributions to PyPI. A weekly CUDA benchmark records tcpWER, failed chunks, and end-to-end latency for VibeVoice on AMI-IHM ES2004a in the workflow summary and artifacts. Select `benchmark` to reproduce it on demand.

Hardware suites activate independently. Provision a dedicated runner for each backend, configure the following environment for the runner service, then set the corresponding GitHub Actions repository variable to `true`.

| Runner label | Runner environment | Repository variable |
| --- | --- | --- |
| `murmur-mlx` | macOS ARM64 with Metal and enough unified memory for Voxtral Small BF16 (about 48 GB for weights plus inference memory). Set `MURMUR_CI_PYTHON` to the absolute Python 3.12 executable with the three MLX runtimes from the Apple Silicon setup and the `test` extra installed. | `MURMUR_MLX_CI` |
| `murmur-cuda` | Linux x64 GPU sized for Voxtral Small and Whisper. Set `MURMUR_CI_PYTHON` to the absolute client Python executable with the `all` and `test` extras. Set `MURMUR_VLLM_VIBEVOICE`, `MURMUR_VLLM_MOSS`, and `MURMUR_VLLM_VOXTRAL` to their model-specific vLLM executables. | `MURMUR_CUDA_CI` |

Set `MURMUR_AMI_ES2004A` on both runners to the readable AMI `ES2004a.Mix-Headset.wav`. On CUDA also set `MURMUR_DATA` to the prepared benchmark dataset root with the ES2004a audio and STM/RTTM references, and configure the runner user’s Hugging Face access for pyannote. Keep the benchmark on the same GPU for comparable timings. Hardware jobs run the checked-out source against these provisioned dependencies. Use a runner group restricted to the trusted hardware workflow or isolated ephemeral runners.
