"""Starting a vLLM server for a model, with Murmur's drafter plugged in.

VibeVoice-ASR and MOSS-Transcribe-Diarize need different vLLM builds (0.14 with the
VibeVoice plugin; a 0.23 nightly), so each server runs with the ``vllm`` executable
of its own environment. Point Murmur at it with ``vllm_bin=``, or the environment
variable ``MURMUR_VLLM_VIBEVOICE`` / ``MURMUR_VLLM_MOSS`` / ``MURMUR_VLLM``; otherwise
the ``vllm`` on ``PATH`` is used.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import requests

from .drafting import MOSS_SPEC_CONFIG, NGRAM_SPEC_CONFIG, SITE_DIR, VIBEVOICE_SPEC_CONFIG
from .models import ModelClient

log = logging.getLogger(__name__)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

SPECULATIVE_MODES = (None, "murmur", "ngram")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def find_vllm(model: str, vllm_bin: str | None = None) -> str:
    for cand in (vllm_bin, os.environ.get(f"MURMUR_VLLM_{model.upper()}"), os.environ.get("MURMUR_VLLM"),
                 shutil.which("vllm")):
        if cand:
            return cand
    raise FileNotFoundError(f"No vllm executable for {model}: pass vllm_bin=, set MURMUR_VLLM_{model.upper()}, "
                            f"or start the server yourself (`murmur serve {model}`) and pass url=.")


def server_command(client: type[ModelClient], *, port: int, batch_size: int = 8, speculative: str | None = "murmur",
                   draft_dir: str | Path | None = None, vllm_bin: str | None = None,
                   single_pass: bool = False, prepare_model: bool = True,
                   gpu_memory_utilization: float | None = None) -> tuple[list[str], dict]:
    """The ``vllm serve`` command line and environment for one model and speculative mode.

    ``prepare_model`` resolves the local model folder (and any one-time setup it needs);
    set False to get the command with the Hugging Face id (e.g. in tests)."""
    if speculative not in SPECULATIVE_MODES:
        raise ValueError(f"speculative must be one of {SPECULATIVE_MODES}")
    name = client.name
    spec = None
    env = {"PYTHONPATH": os.pathsep.join(filter(None, [str(PACKAGE_ROOT), os.environ.get("PYTHONPATH", "")]))}
    if speculative == "ngram":
        spec = NGRAM_SPEC_CONFIG
    elif speculative == "murmur":
        if draft_dir is None:
            raise ValueError("speculative='murmur' needs a draft_dir shared with the client")
        env["MURMUR_DRAFT_DIR"] = str(draft_dir)
        if name == "vibevoice":  # vLLM 0.14 has no custom-proposer hook: patch the n-gram proposer
            spec = VIBEVOICE_SPEC_CONFIG
            env["MURMUR_MODEL"] = "vibevoice"
            env["PYTHONPATH"] = os.pathsep.join([str(SITE_DIR), env["PYTHONPATH"]])
        else:
            spec = MOSS_SPEC_CONFIG
    mem = {"gpu_memory_utilization": gpu_memory_utilization}
    model_args, model_env = (client.server_args(batch_size, spec, single_pass=single_pass, **mem)
                             if name == "moss" else client.server_args(batch_size, spec, **mem))
    env.update(model_env)
    vllm = find_vllm(name, vllm_bin)
    cmd = [vllm, "serve", client.model_path(vllm) if prepare_model else client.hf_id, *model_args,
           "--port", str(port)]
    return cmd, env


class VLLMServer:
    """A vLLM server process owned by this Python process (stopped on ``stop()`` / exit)."""

    def __init__(self, client: type[ModelClient], *, port: int | None = None, gpu: int | str | None = None,
                 batch_size: int = 8, speculative: str | None = "murmur", draft_dir: str | Path | None = None,
                 vllm_bin: str | None = None, log_path: str | Path | None = None, single_pass: bool = False,
                 gpu_memory_utilization: float | None = None, startup_timeout: float = 1800.0):
        self.port = port or free_port()
        self.url = f"http://localhost:{self.port}"
        cmd, extra_env = server_command(client, port=self.port, batch_size=batch_size, speculative=speculative,
                                        draft_dir=draft_dir, vllm_bin=vllm_bin, single_pass=single_pass,
                                        gpu_memory_utilization=gpu_memory_utilization)
        env = {**os.environ, **extra_env}
        env.pop("PYTHONNOUSERSITE", None)  # the server keeps its own environment's site settings
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.log_path = Path(log_path) if log_path else Path(tempfile.gettempdir()) / f"murmur_vllm_{self.port}.log"
        log.info("starting vLLM (%s, log %s): %s", client.name, self.log_path, " ".join(cmd))
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, env=env, stdout=self._log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self._wait(startup_timeout)

    def _wait(self, timeout: float) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM server exited during startup; see {self.log_path}")
            try:
                if requests.get(f"{self.url}/health", timeout=5).ok:
                    log.info("vLLM up on %s after %.0f s", self.url, time.time() - t0)
                    return
            except requests.RequestException:
                pass
            time.sleep(5)
        self.stop()
        raise TimeoutError(f"vLLM server not ready after {timeout:.0f} s; see {self.log_path}")

    def stop(self) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=60)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                os.killpg(self.proc.pid, signal.SIGKILL)
        self._log.close()
