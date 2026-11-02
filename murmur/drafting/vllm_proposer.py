"""
Murmur's structure-aware drafter, running inside the vLLM server.

Loaded by the server that ``murmur serve`` / ``Murmur.load`` starts:

  MOSS-Transcribe-Diarize (vLLM nightly, custom-proposer hook):
    --speculative-config '{"model": "murmur.drafting.vllm_proposer.MurmurMossProposer",
                           "num_speculative_tokens": 8}'
  VibeVoice-ASR (vLLM 0.14, no custom-proposer hook): served with method "ngram"
    and MURMUR_MODEL=vibevoice; murmur/drafting/_vllm_site/sitecustomize.py
    (put on the server's PYTHONPATH) replaces NgramProposer with Murmur's.

  env: MURMUR_DRAFT_DIR = directory where the client writes each chunk's
       short-window transcript (<key>.json); must be visible to the server.

Drafting rules (all drafts are verified, so the output is the model's greedy output):
  text       short-window ASR words: start at the word nearest the segment's own
             start time, re-anchor on the first word, then n-gram lookup in the
             drafter transcript (target-tokenized)
  structure  VibeVoice: after "Start": propose the previous "End" value.
             MOSS: after a segment's [end]: "[next start][speaker]".
  self copy  copy ahead from the model's own output (2-8 grams), including
             repetition loops inside a text field ("loop copy").

vLLM does not pass request ids to a proposer, so this module wraps
GPUModelRunner.propose_draft_token_ids to hand over input_batch.req_ids. The
client tags each chunk with ``X-Request-Id: murmur_<key>``. A request without a
draft file still gets structure and self-copy drafts; the file is re-checked
every _RECHECK steps, so a chunk can start decoding before its draft is written
(write it atomically).

This module needs only ``transformers`` (it runs in the vLLM server environment).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_SEG_START = re.compile(r"\[(\d+(?:\.\d+)?)\]\[(S\d+)\]$")
_END_TS = re.compile(r"\[S\d+\][^\[]*\[(\d+(?:\.\d+)?)\](\[?)$")
_KEY = re.compile(r"murmur_([A-Za-z0-9_.]+)")
_VV_MARK = '"Content":"'
# Inside a text field, prefer copying the model's own output over the drafter
# transcript when the self match is longer (>= LOOP_COPY_MIN tokens). The
# transcript lookup backs off to 1-grams, so without this a repetition loop is
# drafted from the transcript, rejected every step, and decoded 1 token/step.
LOOP_COPY = os.environ.get("MURMUR_LOOP_COPY", "1") == "1"
LOOP_COPY_MIN = 4
# Draft-source ablation (VibeVoice drafter): comma list of text, struct, copy.
_SOURCES = set(os.environ.get("MURMUR_SOURCES", "text,struct,copy").split(","))
USE_TEXT, USE_STRUCT, USE_COPY = ("text" in _SOURCES), ("struct" in _SOURCES), ("copy" in _SOURCES)


def _json_escape(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)[1:-1]


class DraftTranscript:
    """Drafter words + times, tokenized with the target's tokenizer, plus an
    n-gram index over that token stream."""

    NMAX = 4
    HEAD_WINDOW = 8

    def __init__(self, words, tok, sep: str = " ", escape=None):
        self.tok, self.sep = tok, sep
        self.escape = escape or (lambda t: t)
        self.times = [t for t, _ in words]
        self.words = [w for _, w in words]
        self.ids: list[int] = []
        self.wstart: list[int] = []
        if sep:
            for w in self.words:
                self.wstart.append(len(self.ids))
                self.ids += tok.encode(self.escape(sep + w), add_special_tokens=False)
        else:  # CJK: tokens merge across words; tokenize the stream once
            text, starts = "", []
            for w in self.words:
                starts.append(len(text)); text += self.escape(w)
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            self.ids = enc["input_ids"]
            offs = [o[0] for o in enc["offset_mapping"]]
            j = 0
            for c in starts:
                while j < len(offs) - 1 and offs[j + 1] <= c:
                    j += 1
                self.wstart.append(j)
        self.index: dict[tuple, list[int]] = {}
        for n in range(1, self.NMAX + 1):
            for j in range(len(self.ids) - n + 1):
                self.index.setdefault(tuple(self.ids[j:j + n]), []).append(j + n)

    def word_at(self, t):
        return min(range(len(self.times)), key=lambda i: abs(self.times[i] - t)) if self.times else 0

    def from_time(self, t, k, lead=""):
        if not self.words:
            return [], 0
        w = self.word_at(t)
        return self._enc(lead + self.sep.join(self.words[w:w + 16]), k), self.wstart[w]

    def _enc(self, text, k):
        return self.tok.encode(self.escape(text), add_special_tokens=False)[:k]

    @staticmethod
    def _norm(w):
        return re.sub(r"[^\w']", "", w).lower()

    def head(self, content, t, k):
        if not self.words or not content.strip() or not self.sep:
            return None
        c = content.strip()
        cn = self._norm(c)
        if not cn:
            return None
        w0 = self.word_at(t)
        lo, hi = max(0, w0 - self.HEAD_WINDOW), min(len(self.words), w0 + self.HEAD_WINDOW + 1)
        cands = [i for i in range(lo, hi) if self._norm(self.words[i]).startswith(cn)]
        if not cands:
            return None
        p = min(cands, key=lambda i: abs(i - w0))
        word = self.words[p]
        if content.endswith(" "):
            rest = " ".join(self.words[p + 1:p + 16])
        elif word.startswith(c):
            rest = word[len(c):] + " " + " ".join(self.words[p + 1:p + 16])
            rest = rest if word != c else " " + " ".join(self.words[p + 1:p + 16])
        else:
            return None
        return (self._enc(rest, k),
                self.wstart[p + 1] if p + 1 < len(self.wstart) else len(self.ids))

    def lookup(self, accepted, ptr, k):
        d, ptr, _ = self.lookup_n(accepted, ptr, k)
        return d, ptr

    def lookup_n(self, accepted, ptr, k):
        """lookup() plus the length of the n-gram that matched (0 = none)."""
        for n in range(min(self.NMAX, len(accepted)), 0, -1):
            ends = self.index.get(tuple(accepted[-n:]))
            if ends:
                ptr = min(ends, key=lambda e: abs(e - ptr))
                return self.ids[ptr:ptr + k], ptr, n
        return [], ptr, 0


class MossDrafter:
    """Drafts for MOSS-Transcribe-Diarize's ``[start][Sxx]text[end]`` output."""

    SELF_NMAX = 8

    def __init__(self, dt: DraftTranscript, tok, language: str,
                 diar_segments=None, chunk_offset: float = 0.0):
        self.dt, self.tok = dt, tok
        self.lead = "" if language == "zh" else " "
        self.text, self.in_content, self.content = "", False, ""
        self.seg_start, self.ptr, self.n_seen = None, 0, 0
        self.self_index: dict[tuple, int] = {}
        # (abs_start, abs_end, pyannote label) in this chunk, to predict the next speaker
        self.diar_segments = diar_segments or []
        self.chunk_offset = chunk_offset
        self._pyan_to_moss: dict[str, str] = {}

    def _advance(self, gen):
        for i in range(self.n_seen, len(gen)):
            for n in range(2, self.SELF_NMAX + 1):
                if i - n >= 0:
                    self.self_index[tuple(gen[i - n:i])] = i
        for ch in self.tok.decode(gen[self.n_seen:]):
            self.text += ch
            if self.in_content:
                if ch == "[":
                    self.in_content = False
                else:
                    self.content += ch
            else:
                m = _SEG_START.search(self.text)
                if m:
                    self.in_content, self.content, self.seg_start = True, "", float(m.group(1))
                    if self.diar_segments:
                        abs_t = self.chunk_offset + float(m.group(1))
                        pyan_spk = self._diar_speaker_at(abs_t + 0.05)
                        if pyan_spk:
                            self._pyan_to_moss[pyan_spk] = m.group(2)
        self.text = self.text[-400:]
        self.n_seen = len(gen)

    def _diar_speaker_at(self, abs_t: float):
        """Return pyannote speaker label active at absolute time abs_t, or None."""
        for s, e, lbl in self.diar_segments:
            if s <= abs_t < e:
                return lbl
        return None

    def _next_start_ts(self, end_t: float) -> float:
        """Best estimate of the next segment's start, using faster-whisper words."""
        if self.dt.times:
            future = [t for t in self.dt.times if t > end_t + 0.01]
            if future:
                return future[0]
        return end_t + 0.03

    def _next_speaker(self, end_t: float, cur_spk: str) -> str:
        """Predict the next MOSS speaker label using pyannote diarization."""
        if not self.diar_segments:
            return cur_spk
        next_ts = self._next_start_ts(end_t)
        abs_next = self.chunk_offset + next_ts
        pyan_spk = self._diar_speaker_at(abs_next + 0.05)
        if pyan_spk and pyan_spk in self._pyan_to_moss:
            return self._pyan_to_moss[pyan_spk]
        # unknown pyannote speaker: guess another speaker the model has used
        others = [ms for ps, ms in self._pyan_to_moss.items() if ms != cur_spk]
        return others[0] if others else cur_spk

    def propose(self, gen, k):
        self._advance(gen)
        if self.in_content:
            t = self.seg_start
            if t is not None and self.content == "":
                # In a loop the same segment header repeats; self-copy handles it
                # faster than the transcript (which would be rejected every time).
                for n in range(min(self.SELF_NMAX, len(gen)), 1, -1):
                    se = self.self_index.get(tuple(gen[-n:]))
                    if se is not None:
                        return gen[se:se + k]
                d, self.ptr = self.dt.from_time(t, k, lead=self.lead)
                return d
            if t is not None and len(self.content.split()) <= 1:
                hit = self.dt.head(self.content, t, k)
                if hit is not None:
                    d, self.ptr = hit
                    return d
            d, self.ptr = self.dt.lookup(gen, self.ptr, k)
            if d:
                return d
        # structure: after a segment's [end], propose the next [start][speaker]
        m = _END_TS.search(self.text)
        if m and m.group(2) != "[":
            end_t = float(m.group(1))
            spk_m = re.search(r'\[S\d+\]', m.group(0))
            cur_spk = spk_m.group(0)[1:-1] if spk_m else "S00"
            next_start = self._next_start_ts(end_t)
            next_spk = self._next_speaker(end_t, cur_spk)
            return self.tok.encode(f"[{next_start:.2f}][{next_spk}]", add_special_tokens=False)[:k]
        for n in range(min(self.SELF_NMAX, len(gen)), 1, -1):
            end = self.self_index.get(tuple(gen[-n:]))
            if end is not None:
                return gen[end:end + k]
        return []


class VibeVoiceDrafter:
    """Drafts for VibeVoice-ASR's JSON output."""

    SELF_NMAX = 8

    def __init__(self, dt: DraftTranscript, tok, language: str):
        self.dt, self.tok = dt, tok
        self.text, self.in_content, self.escape, self.content = "", False, False, ""
        self.seg_start, self.ptr, self.n_seen = None, 0, 0
        self.self_index: dict[tuple, int] = {}

    def _advance(self, gen):
        for i in range(self.n_seen, len(gen)):
            for n in range(2, self.SELF_NMAX + 1):
                if i - n >= 0:
                    self.self_index[tuple(gen[i - n:i])] = i
        for ch in self.tok.decode(gen[self.n_seen:]):
            self.text += ch
            if self.in_content:
                if self.escape:
                    self.escape = False
                elif ch == "\\":
                    self.escape = True
                elif ch == '"':
                    self.in_content = False
                    continue
                self.content += ch
            elif self.text.endswith(_VV_MARK):
                self.in_content, self.content = True, ""
                m = re.findall(r'"Start":\s*([0-9.]+)', self.text)
                self.seg_start = float(m[-1]) if m else None
        self.text = self.text[-400:]
        self.n_seen = len(gen)

    def propose(self, gen, k):
        self._advance(gen)
        if self.in_content:
            t = self.seg_start
            if t is not None and self.content == "":
                # In a loop the same segment header repeats; self-copy handles it
                # faster than the transcript (which would be rejected every time).
                if USE_COPY:
                    for n in range(min(self.SELF_NMAX, len(gen)), 1, -1):
                        se = self.self_index.get(tuple(gen[-n:]))
                        if se is not None:
                            return gen[se:se + k]
                if USE_TEXT:
                    d, self.ptr = self.dt.from_time(t, k)
                    return d
            if USE_TEXT and t is not None and len(self.content.split()) <= 1:
                hit = self.dt.head(self.content, t, k)
                if hit is not None:
                    d, self.ptr = hit
                    return d
            if USE_TEXT:
                d, self.ptr, n_fw = self.dt.lookup_n(gen, self.ptr, k)
            else:
                d, n_fw = [], 0
            if LOOP_COPY and USE_COPY:
                for n in range(min(self.SELF_NMAX, len(gen)), max(n_fw, LOOP_COPY_MIN - 1), -1):
                    se = self.self_index.get(tuple(gen[-n:]))
                    if se is not None:
                        # the latest match ends just before the suffix, so repeat
                        # its continuation with the loop's period to fill k
                        p = len(gen) - se
                        return [gen[se + j % p] for j in range(k)]
            if d:
                return d
        if USE_STRUCT and self.text.endswith('"Start":'):
            m = re.findall(r'"End":\s*([0-9.]+)', self.text)
            if m:
                return self.tok.encode(m[-1] + ',"End":', add_special_tokens=False)[:k]
        if not USE_COPY:
            return []
        for n in range(min(self.SELF_NMAX, len(gen)), 1, -1):
            end = self.self_index.get(tuple(gen[-n:]))
            if end is not None:
                return gen[end:end + k]
        return []


_RECHECK = 16


class _Req:
    def __init__(self, prompt_len, drafter, missing=None):
        self.prompt_len, self.drafter = prompt_len, drafter
        self.missing, self.calls = missing, 0  # draft file not there yet (path)


class MurmurCore:
    """Per-request drafter bank; propose() matches vLLM's ngram signature."""

    _murmur = True
    kind = "moss"

    def _setup(self, vllm_config, kind):
        from transformers import AutoTokenizer
        mc = vllm_config.model_config
        self.kind = kind
        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.tok = AutoTokenizer.from_pretrained(mc.tokenizer, revision=mc.tokenizer_revision,
                                                 trust_remote_code=True)
        self.dir = Path(os.environ["MURMUR_DRAFT_DIR"])
        self.reqs: dict[str, _Req] = {}
        self._req_ids: list[str] = []

    def _new(self, rid, prompt_len):
        m = _KEY.search(rid)
        words, lang, diar_segs, chunk_offset = [], "en", [], 0.0
        missing = None
        if m and not (self.dir / f"{m.group(1)}.json").exists():
            missing = self.dir / f"{m.group(1)}.json"
        if m and missing is None:
            d = json.loads((self.dir / f"{m.group(1)}.json").read_text())
            words, lang = [tuple(x) for x in d["words"]], d.get("language", "en")
            diar_segs = [tuple(x) for x in d.get("diar_segments", [])]
            chunk_offset = float(d.get("chunk_offset", 0.0))
        sep = "" if lang == "zh" else " "
        if self.kind == "vibevoice":
            dt = DraftTranscript(words, self.tok, sep=sep, escape=_json_escape)
            return _Req(prompt_len, VibeVoiceDrafter(dt, self.tok, lang), missing)
        dt = DraftTranscript(words, self.tok, sep=sep)
        return _Req(prompt_len, MossDrafter(dt, self.tok, lang,
                                            diar_segments=diar_segs,
                                            chunk_offset=chunk_offset), missing)

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None, **_):
        out = []
        for i, sampled in enumerate(sampled_token_ids):
            rid = self._req_ids[i] if i < len(self._req_ids) else None
            if not sampled or rid is None:
                out.append([])
                continue
            n = int(num_tokens_no_spec[i])
            r = self.reqs.get(rid)
            if r is None:
                r = self.reqs[rid] = self._new(rid, n - len(sampled))
            elif r.missing is not None:
                r.calls += 1
                if r.calls % _RECHECK == 0 and r.missing.exists():
                    # draft arrived: rebuild; the drafter replays the output so far
                    r = self.reqs[rid] = self._new(rid, r.prompt_len)
            out.append(r.drafter.propose(token_ids_cpu[i, r.prompt_len:n].tolist(), self.k)[:self.k])
        live = set(self._req_ids)
        if len(self.reqs) > 4 * max(1, len(live)):
            self.reqs = {k: v for k, v in self.reqs.items() if k in live}
        return out


class MurmurMossProposer(MurmurCore):
    def __init__(self, vllm_config):
        self._setup(vllm_config, "moss")


def patch_ngram(kind: str = "vibevoice"):
    """vLLM without a custom-proposer hook: turn NgramProposer into Murmur."""
    from vllm.v1.spec_decode import ngram_proposer as npm
    cls = npm.NgramProposer
    if getattr(cls, "_murmur", False):
        return

    def __init__(self, vllm_config):
        self._setup(vllm_config, kind)

    for name in ("_setup", "_new", "propose"):
        setattr(cls, name, getattr(MurmurCore, name))
    cls.__init__ = __init__
    cls._murmur = True


def _install_req_id_hook():
    try:
        from vllm.v1.worker import gpu_model_runner as g
    except ImportError:  # imported outside a vLLM worker (offline tests)
        return
    orig = g.GPUModelRunner.propose_draft_token_ids
    if getattr(orig, "_murmur", False):
        return

    def wrapped(self, *a, **kw):
        if getattr(getattr(self, "drafter", None), "_murmur", False):
            self.drafter._req_ids = list(self.input_batch.req_ids)
        return orig(self, *a, **kw)

    wrapped._murmur = True
    g.GPUModelRunner.propose_draft_token_ids = wrapped


_install_req_id_hook()
