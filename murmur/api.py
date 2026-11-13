"""``Murmur``: long-form speaker-attributed ASR with hierarchical chunking.

    from murmur import Murmur
    m = Murmur.load("vibevoice")          # starts a vLLM server (or pass url=...)
    t = m.transcribe("meeting.wav")
    print(t)                               # [  12.40-  15.10] SPEAKER_01: ...

Pipeline per recording (paper Sec. 4):
  1. diarization (pyannote) gives speech regions and speakers;
  2. target chunks of up to ``chunk_s`` seconds are cut between speech regions;
  3. draft windows (optional): a short-window ASR transcribes each chunk, and the
     server-side drafter turns that transcript, format rules and the model's own
     output into drafts that the model verifies (output unchanged);
  4. all chunks are decoded in parallel on the vLLM server;
  5. chunk outputs are stitched: timestamps shifted, speakers linked across chunks.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .audio import SR, Diarizer, Timer, chunk_bounds, load_audio, resample, slice_audio
from .models import ModelClient, get_model
from .stitch import chunk_speaker_count, stitch
from .types import Transcript

log = logging.getLogger(__name__)


class Murmur:
    """A loaded model plus the Murmur pipeline around it. Create with :meth:`load`."""

    def __init__(self, client: ModelClient, *, server=None, chunk_s: float = 330.0, batch_size: int = 8,
                 draft_windows: bool = False, draft_dir: str | Path | None = None, language: str = "en",
                 diarizer: Diarizer | None = None, drafter=None, keep_failed_output: bool = False,
                 owns_draft_dir: bool = False):
        self.client, self.server = client, server
        self.chunk_s, self.batch_size = chunk_s, batch_size
        self.draft_windows, self.draft_dir, self.language = draft_windows, draft_dir, language
        self.diarizer, self.drafter = diarizer, drafter
        self.keep_failed_output = keep_failed_output
        self._owns_draft_dir = owns_draft_dir

    @classmethod
    def load(cls, model: str = "vibevoice", *, url: str | None = None, draft_windows: bool | None = None,
             speculative: str | None = None, chunk_s: float = 330.0, batch_size: int = 8, language: str = "en",
             gpu: int | str | None = None, vllm_bin: str | None = None, draft_dir: str | Path | None = None,
             diarize: bool = True, hf_token: str | None = None, served_name: str | None = None,
             server_log: str | Path | None = None, single_pass_server: bool = False,
             gpu_memory_utilization: float | None = None, keep_failed_output: bool = False) -> "Murmur":
        """Load a model.

        model          "vibevoice" (VibeVoice-ASR) or "moss" (MOSS-Transcribe-Diarize).
        url            an already running server (``murmur serve``); otherwise one is started
                       with the ``vllm`` of this model's environment (see :mod:`murmur.server`).
        draft_windows  speculative decoding from 30 s draft windows; default on for VibeVoice-ASR
                       (7B decoder) and off for MOSS-Transcribe-Diarize (0.6B), as in the paper,
                       and off when ``speculative`` is "ngram".
        speculative    server-side mode when starting a server: "murmur" (implied by
                       draft_windows), "ngram" (vLLM prompt lookup, a baseline) or None.
        chunk_s        target chunk length in seconds (330 in the paper).
        batch_size     chunks decoded concurrently (and the server's --max-num-seqs).
        language       "en" or "zh"; picks the draft model (Whisper-tiny / Whisper-small).
        draft_dir      directory the client and server share for drafts (required with url=
                       and draft windows; it must match the server's MURMUR_DRAFT_DIR).
        diarize        use pyannote for chunk boundaries and speaker linking (needs a HF token).
        gpu_memory_utilization  share of the GPU the started server takes (default 0.70, leaving
                       room for pyannote and the draft model on the same GPU).
        """
        client_cls = get_model(model)
        if draft_windows is None:
            draft_windows = client_cls.draft_windows_default and speculative in (None, "murmur")
        if speculative is None and draft_windows:
            speculative = "murmur"
        if draft_windows and speculative != "murmur":
            raise ValueError("draft_windows=True needs speculative='murmur'")
        owns = False
        if draft_windows and draft_dir is None:
            if url is not None:
                raise ValueError("with url= and draft windows, pass the server's draft_dir (MURMUR_DRAFT_DIR)")
            draft_dir, owns = tempfile.mkdtemp(prefix="murmur_drafts_"), True
        server = None
        if url is None:
            from .server import VLLMServer
            server = VLLMServer(client_cls, gpu=gpu, batch_size=batch_size, speculative=speculative,
                                draft_dir=draft_dir, vllm_bin=vllm_bin, log_path=server_log,
                                single_pass=single_pass_server, gpu_memory_utilization=gpu_memory_utilization)
            url = server.url
        client = client_cls(url, served_name=served_name)
        diarizer = Diarizer(hf_token) if diarize else None
        drafter = None
        if draft_windows:
            from .drafting.whisper_drafts import WhisperDrafter
            drafter = WhisperDrafter(language)
        return cls(client, server=server, chunk_s=chunk_s, batch_size=batch_size, draft_windows=draft_windows,
                   draft_dir=draft_dir, language=language, diarizer=diarizer, drafter=drafter,
                   keep_failed_output=keep_failed_output, owns_draft_dir=owns)

    def transcribe(self, audio, sr: int | None = None, *, single_pass: bool = False,
                   timeline=None) -> Transcript:
        """Transcribe one recording (a path, or a waveform with ``sr``).

        single_pass  decode the whole recording in one request (the baseline; no chunking).
        timeline     a precomputed diarization timeline [(start, end, speaker), ...] to use
                     instead of running pyannote (its time is then not counted).
        """
        return self.transcribe_batch([audio], sr=sr, single_pass=single_pass,
                                     timelines=None if timeline is None else [timeline])[0]

    def transcribe_batch(self, audios: list, sr: int | None = None, *, single_pass: bool = False,
                         timelines: list | None = None) -> list[Transcript]:
        """Transcribe several recordings, decoding all their chunks through one shared pool
        of ``batch_size`` concurrent requests (dataset throughput)."""
        timer = Timer()
        wavs = [load_audio(a, sr) for a in audios]
        durs = [w.shape[0] / SR for w in wavs]
        cs = self.client.sample_rate

        # 1. diarization and chunk boundaries
        tls: list = [None] * len(wavs)
        if single_pass:
            jobs = [(r, 0.0, durs[r]) for r in range(len(wavs))]
        else:
            with timer("diarization"):
                for r, w in enumerate(wavs):
                    if timelines is not None and timelines[r] is not None:
                        tls[r] = [tuple(t) for t in timelines[r]]
                    elif self.diarizer is not None:
                        tls[r] = self.diarizer(w)
            jobs = [(r, a, b) for r in range(len(wavs)) for a, b in chunk_bounds(durs[r], self.chunk_s, tls[r])]
        arrays = [resample(wavs[r], cs) if single_pass else slice_audio(wavs[r], a, b, cs) for r, a, b in jobs]
        run_id = uuid.uuid4().hex[:8]
        keys = [f"{run_id}_{i}" for i in range(len(jobs))]

        # 2. draft windows (a background thread, overlapped with decoding)
        ready = None
        draft_thread = None
        if self.draft_windows and not single_pass:
            from .drafting.whisper_drafts import write_draft
            ready = [threading.Event() for _ in jobs]

            def drafting():
                t0 = time.perf_counter()
                for i, (r, a, b) in enumerate(jobs):
                    try:
                        words = self.drafter(slice_audio(wavs[r], a, b, SR))
                        diar = ([list(t) for t in tls[r] if t[1] > a and t[0] < b]
                                if self.client.name == "moss" and tls[r] else [])
                        write_draft(Path(self.draft_dir), keys[i], words, self.language, a, diar)
                    except Exception:  # a chunk without drafts still decodes (structure + self copy)
                        log.exception("drafting failed for chunk %d", i)
                    ready[i].set()
                timer.stages["drafting"] = time.perf_counter() - t0
            draft_thread = threading.Thread(target=drafting, daemon=True)

        # 3. decode all chunks in parallel
        def decode(i):
            if ready is not None:
                ready[i].wait()
            return self.client.decode(arrays[i], request_id=keys[i] if ready is not None else None)

        t0 = time.perf_counter()
        if draft_thread is not None:
            draft_thread.start()
        with ThreadPoolExecutor(max_workers=self.batch_size) as ex:
            replies = list(ex.map(decode, range(len(jobs))))
        if draft_thread is not None:
            draft_thread.join()
        timer.stages["decoding"] = time.perf_counter() - t0
        self._cleanup_drafts(keys)

        # 4. stitch per recording (with the model-informed speaker-count hint)
        out = []
        for r in range(len(wavs)):
            idx = [i for i, j in enumerate(jobs) if j[0] == r]
            pieces = [(jobs[i][1], jobs[i][2], self.client.parse(replies[i].text, keep_raw=self.keep_failed_output))
                      for i in idx]
            tl = tls[r]
            if tl and self.diarizer is not None and not single_pass:
                k = chunk_speaker_count([p[2] for p in pieces])
                if k > 1 and len({t[2] for t in tl}) < k:
                    with timer("rediarization"):
                        tl = self.diarizer(wavs[r], min_speakers=k) or tl
            segs = stitch(pieces, None if single_pass else tl, merge_gap_s=self.client.merge_gap_s,
                          clamp=self.client.name == "vibevoice")
            stats = {"n_chunks": len(idx), "chunks_failed": sum(self.client.flagged(replies[i]) for i in idx),
                     "request_s": [round(replies[i].seconds, 3) for i in idx],
                     "chunks": [(jobs[i][1], jobs[i][2]) for i in idx], "stages_s": dict(timer.stages)}
            out.append(Transcript(segs, durs[r], stats))
        return out

    def _cleanup_drafts(self, keys):
        if self.draft_dir is None:
            return
        for k in keys:
            try:
                (Path(self.draft_dir) / f"{k}.json").unlink()
            except FileNotFoundError:
                pass

    def close(self) -> None:
        """Stop the server Murmur started (if any) and remove its temporary draft directory."""
        if self.server is not None:
            self.server.stop()
            self.server = None
        if self._owns_draft_dir and self.draft_dir:
            shutil.rmtree(self.draft_dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load(model: str = "vibevoice", **kw) -> Murmur:
    """Shortcut for :meth:`Murmur.load`."""
    return Murmur.load(model, **kw)

