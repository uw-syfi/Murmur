# VibeVoice speculative decoding and WhisperX on M4 Max

Warm, file-to-speaker-labeled-transcript wall time on the AMI ES2004a headset mix (1,049.35 s). Both systems ran sequentially on one M4 Max with 128 GB, macOS 15.1. Each warmed up on the first 60 seconds and ran the full meeting once. The measurements exclude model loading, downloads, warm-up and scoring. This comparison measures latency.

Murmur used VibeVoice-ASR MLX BF16, batch size 1, 330-second target chunks, up to eight proposed tokens, Whisper-tiny English CPU INT8 drafts and the default native MLX pyannote diarizer. Automatic speaker counting produced four speakers and 262 turns in one diarization pass. All four transcription chunks completed. WhisperX used large-v3, MPS FP16, batch size 16, the English forced aligner, and its normal voice activity detection. It received the same measured speaker timeline for word attribution. Its comparable total adds that diarization interval once to the directly measured WhisperX pipeline time.

## VibeVoice wall time

| Stage | Seconds | Share |
|---|---:|---:|
| Target verification forwards, greedy argmax and result readback | 439.31 | 83.73% |
| Acoustic and semantic speech encoding | 44.19 | 8.42% |
| Prompt construction and language model prefill through the first token | 20.09 | 3.83% |
| Speaker diarization | 18.04 | 3.44% |
| CPU Whisper drafting outside target decoding | 2.44 | 0.46% |
| Python token proposals | 0.18 | 0.03% |
| KV cache rewind calls on the host | 0.02 | <0.01% |
| Remaining audio I/O, setup, loop bookkeeping and stitching | 0.37 | 0.07% |
| **Total** | **524.64** | **100%** |

The draft worker spent 33.64 s in Whisper, of which 31.20 s overlapped a target decode call. Only the first chunk waited appreciably for its draft. The table subtracts the intersection of those wall-clock intervals. Simultaneous CPU and GPU work can still contend for hardware resources.

| Target chunk | Speech encoder | Prompt and prefill | Verification | Verification calls |
|---|---:|---:|---:|---:|
| 1 | 14.32 s | 7.02 s | 89.72 s | 751 |
| 2 | 14.34 s | 5.57 s | 174.87 s | 1,364 |
| 3 | 13.07 s | 5.60 s | 138.56 s | 1,025 |
| 4 | 2.46 s | 1.89 s | 36.17 s | 313 |

VibeVoice spent a mean 127.2 ms per target verification call. It accepted 3,765 of 20,776 proposed tokens (18.1%). Of 3,453 calls, 848 ran with an empty proposal and 1,303 of the 2,605 nonempty proposals missed on their first token. Of 2,582 eight-token proposals, 76 matched all eight tokens. The most direct follow-up is to measure verification time and accepted-prefix length across proposal lengths and proposal sources, holding the Whisper words and chunk boundaries fixed.

The encoder timer explicitly materializes its input and output. Prefill ends at the synchronous read of the first target token. Each verification timer includes its existing synchronous token readback. Rewind measures the host call; any deferred device work appears at a later synchronization. The remaining time is the wall-clock residual.

## WhisperX baseline

| Stage | Seconds |
|---|---:|
| Shared speaker diarization | 18.04 |
| Audio loading | 0.17 |
| Voice activity detection inside transcribe | 1.87 |
| Other ASR work, including batched large-v3 recognition | 90.73 |
| Forced word alignment | 9.28 |
| Assigning words to speakers | 0.02 |
| Other overhead | <0.01 |
| **Measured WhisperX pipeline** | **102.06** |
| **Total including shared diarization** | **120.10** |

VibeVoice took 4.37 times as long end to end. Excluding shared diarization, the comparison was 506.60 s versus 102.06 s (4.96×). Real-time factors for the complete systems were 0.500 and 0.114, respectively. VibeVoice's target verification alone took 439.31 s, exceeding WhisperX's complete 120.10 s pipeline including diarization.

## What the target verifies

The Mac decoded four chunks serially with target batch size 1. Across all 3,453 target calls, the mean draft length was 6.02. The target scored 7.02 positions per call including the pending token, accepted 1.09 draft tokens, and advanced by about 2.09 tokens including its own correction or bonus. Among the 2,605 nonempty proposals, the means were 7.98 proposed and 1.45 accepted. On vLLM's counter convention, this gives a mean acceptance length of 2.45 including the target token. The 848 ordinary one-token calls still count towards real wall time. [vLLM 0.14 scheduler](https://github.com/vllm-project/vllm/blob/v0.14.0/vllm/v1/core/sched/scheduler.py) · [Acceptance metric](https://github.com/vllm-project/vllm/blob/v0.14.0/vllm/v1/spec_decode/metrics.py)

A replay of the first real chunk timed 192 consecutive target calls. After discarding the first eight, the median one-position forward took 46.6 ms (81 calls), and the median nine-position forward took 164.6 ms (101 calls). Two intermediate-length calls completed the sample. The full block costs 3.53× as much on this replay, and the target evaluates it in full even if the first proposed token is subsequently rejected.

Two separate decoder-only sweeps used the BF16 target and independent synthetic KV histories of 4,096 tokens. At nine positions, batching four rows gave 1.77× and 2.52× the row throughput of batch 1. Equal-length rows and warmed shapes favor this microbenchmark. In a separate diagnostic with synchronization at attention and feed-forward boundaries, the feed-forward submodules took 61–64% of the instrumented nine-position step across the two runs. These barriers increase absolute times. Timings varied across runs, and the original meeting profile did not record individual kernel timings.

On one H100 80 GB, the paper reports 2.31 tokens per nonempty speculative step for VibeVoice on AMI-IHM at target batch 8 and maximum draft length 8. It reports 1.78× speculative-over-greedy decoding speedup at batch 1 and 1.83× at batch 8. In the per-recording latency experiment vLLM could process up to eight chunks concurrently. Mean end-to-end latency over all 16 headset recordings was 38.0 s for VibeVoice with drafts, 42.2 s for chunked greedy, and 36.3 s for WhisperX. The paper reports a speculative step cost of about 1.25–1.30× a plain step in vLLM. Its BF16 serving measurements do not give absolute milliseconds per verifier call for the specific ES2004a meeting used in the Mac trace. [Murmur v2, sections 4.3 and 5.3 and Tables 2, 3, 11 and 12](https://arxiv.org/html/2606.01483v2)

## End-to-end chunk batching

The same M4 Max ran a fresh batch-1 control and then batch 4, warming up on 60 s and seeding MLX
identically before each timed run. Both processed the same four chunks and the same audio file
with automatic MLX diarization. Timings include CPU drafting, prefill, verification and stitching.

| Target batch | Wall time | Target forward passes | Peak active MLX memory | WER | tcpWER | Failed chunks |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 529.21 s | 3,453 | 19.30 GiB | 19.24% | 19.24% | 0 |
| 4 | 350.80 s | 1,367 | 19.52 GiB | 19.16% | 19.24% | 0 |

Batch 4 took 33.7% less time (1.51× speedup). Its 1,367 target passes comprised 318 with four
active chunks, 447 with three, 252 with two and 350 with one. The public verification counter
counts per-chunk checks (3,467 at batch 4). The larger batch applies the decoder projections and
feed-forward layers to tokens from multiple chunks together. Attention, positions and cache
rollback stay independent for each chunk. The BF16 transcripts differed slightly. The memory
figures are MLX active allocations during the timed pipeline, excluding other process and system
allocations. These latency and accuracy figures are from one meeting and one run per setting.

### Larger batch sizes

On the same M4 Max and ES2004a headset recording, a 60-second target chunk limit produced
19 chunks. All three runs used identical chunk boundaries, a 60-second warm-up, seed 48,
the BF16 model and automatic diarization. Each size ran once, in the order 8, 4, 16. The
original 330-second setting produced four chunks on this recording.

| Maximum target batch | End-to-end time | Target forward passes | Passes at full batch | Peak active MLX memory | WER | tcpWER |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 289.16 s | 1,192 | 677 | 18.15 GiB | 19.48% | 20.20% |
| 8 | 230.90 s | 753 | 189 | 18.15 GiB | 19.52% | 20.24% |
| 16 | 210.11 s | 506 | 81 | 18.15 GiB | 19.52% | 20.28% |

All 19 chunks completed at each setting. Batch 16 cut latency by 27.3% relative to batch 4 and 9.0% relative to batch 8.

## Quantized VibeVoice with batch 16

The M4 Max processed the 1,049.35-second ES2004a Mix-Headset recording in two target batches (16 chunks, then 3). Each timed run includes automatic MLX diarization, live CPU Whisper-tiny INT8 drafting, and final stitching. Model loading and the 60-second warm-up precede the timer. INT8 and INT4 use quantized text-backbone weights and BF16 speech encoder weights. The BF16 column uses the instrumented replay of the earlier batch-16 benchmark. Each column is one measured run.

| End-to-end stage | BF16 (s) | INT8 (s) | INT4 (s) |
|---|---:|---:|---:|
| Speaker diarization | 19.26 | 17.93 | 16.62 |
| Target waiting for a draft | 2.77 | 0.93 | 0.28 |
| Speech encoder | 32.01 | 33.48 | 30.38 |
| Prompt preparation and decoder prefill | 40.28 | 37.60 | 35.32 |
| Target token verification | 120.27 | 87.70 | 78.42 |
| Proposals, acceptance, stitching and remaining overhead | 0.63 | 0.43 | 0.42 |
| **Total wall time** | **215.22** | **178.07** | **161.45** |

The verifier ran 506, 505 and 494 batched forward passes for BF16, INT8 and INT4 respectively. Live CPU Whisper occupied 70.91 seconds on INT8 and 58.02 seconds on INT4. Of those times, 61.91 and 56.29 seconds overlapped target prefill (including speech encoding), and 7.97 and 1.38 seconds overlapped verification. The draft-wait row counts the time the target spent waiting for input. The WhisperX baseline measured earlier on this recording took 120.10 seconds including shared diarization.

Peak active MLX allocations were 11.50 GiB for INT8 and 7.96 GiB for INT4. WER was 19.52% for BF16, 19.44% for INT8 and 19.48% for INT4 on this recording.

### CPU Whisper interference

Four INT4 replays provided identical saved Whisper words immediately to the target. In two replays, a background CPU Whisper worker also transcribed all 19 original clips, starting with the first draft request. The trials ran in idle, active, active, idle order. All transcripts and speculative verification counters matched the live INT4 output. The first 16 chunks' speech encoding and prefill overlapped the background worker essentially in full in both active runs. Whisper overlapped just 0.50 and 1.91 seconds of verification and started after diarization.

| Mean of two replays per condition | Whisper idle (s) | Whisper active (s) | Change |
|---|---:|---:|---:|
| Speech encoding, first 16 chunks | 24.75 | 25.68 | +3.8% |
| Prompt preparation and decoder prefill, first 16 chunks | 27.71 | 32.43 | +17.1% |
| Combined first-batch encoding and prefill | 52.45 | 58.12 | +10.8% |
| Whole meeting, saved drafts available | 156.74 | 160.29 | +2.3% |

The individual paired wall-time increases were 2.01 and 5.10 seconds. The prefill measurements include host work and synchronized MLX execution. These trials establish a concurrent-workload effect for this INT4 configuration. They do not identify the hardware mechanism or measure sustained verifier contention.

### Native MLX Whisper drafter

The same full-meeting benchmark with VibeVoice INT4 and MLX Whisper-tiny FP16 took 180.70 s on the M4 Max. All 19 chunks completed. The drafter returned 2,117 timestamped words and the target accepted 3,499 of 16,097 proposed tokens. WER was 19.64%. Python imports of faster-whisper and CTranslate2 were blocked during the run.

| Stage | Earlier CPU Whisper (s) | MLX Whisper (s) |
|---|---:|---:|
| Speaker diarization | 16.62 | 23.22 |
| Waiting for drafts | 0.28 | 1.07 |
| Speech encoder | 30.38 | 35.43 |
| Prompt and decoder prefill | 35.32 | 37.17 |
| Verification | 78.42 | 83.17 |
| Proposals, stitching and other overhead | 0.42 | 0.64 |
| **End to end** | **161.45** | **180.70** |

These are separate warm runs. Diarization precedes Whisper and varied by 6.60 s between runs. The Whisper calls themselves totaled 58.02 s on CPU and 59.65 s with MLX, much of it overlapping target execution. The two Whisper implementations produced different drafts. The CPU results above used faster-whisper on the CPU.

### Quantizing the MLX Whisper drafter

On the M4 Max, three warm runs transcribed the same 1,049.35 s meeting using VibeVoice INT4, MLX diarization, 60 s target chunks, and a maximum batch of 16. Only the precision of multilingual Whisper-tiny varied. All 19 chunks completed in each run and all returned word starts were monotonic and within the corresponding chunk. The trials ran in INT8, FP16, INT4 order with a fresh process and 60 s warm-up each time. Checkpoint loading and in-memory quantization preceded the timer.

| Whisper precision | Whisper parameters (MiB) | Whisper alone (s) | Whisper during pipeline (s) | End to end (s) | Whisper draft WER | Target accepted / proposed | Final transcript WER |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 70.9 | 6.23 | 37.58 | 161.08 | 33.21% | 3,499 / 16,097 (21.74%) | 19.64% |
| INT8 | 38.4 | 6.28 | 39.11 | 157.49 | 32.93% | 3,535 / 15,913 (22.21%) | 19.44% |
| INT4 | 21.0 | 6.50 | 41.00 | 164.52 | 45.23% | 3,128 / 17,068 (18.33%) | 19.64% |

Whisper-alone times are means of two runs over the same 19 pre-sliced clips, in reverse order on the second pass. Pipeline Whisper times sum elapsed time in the synchronous drafter calls, including GPU contention while VibeVoice was active. The target spent 0.54, 0.48 and 0.40 seconds waiting for drafts respectively. Drafter WER scores its concatenated words against the meeting reference. The final column scores VibeVoice's verified transcript. The earlier FP16 meeting run took 180.70 s and produced the same drafts and final transcript as the 161.08 s control. One full run per precision in this comparison limits conclusions about end-to-end speed.

To reproduce the quantization, apply the following to a loaded MLX pipeline with drafting enabled before warming the target. The experiment packed all 64 Whisper linear layers and its tied token embedding with affine weight-only quantization, in groups of 64.

```python
import mlx.core as mx
import mlx.nn as nn

whisper = pipeline.drafter.model
nn.quantize(whisper, bits=8, group_size=64, mode="affine",
            class_predicate=lambda _, layer: isinstance(layer, (nn.Linear, nn.Embedding)))
mx.eval(whisper.parameters())
```

## Reproduction

The reported diarization timings use the native MLX diarizer from [PR #21](https://github.com/uw-syfi/Murmur/pull/21). Reproducing those timings requires a checkout with both the diarization and decoding changes. The profiler otherwise uses the configured Murmur diarizer. Install the `drafts` extra for the historical CPU Whisper comparison. Run from the repository root with the Mac environments and checkpoints installed. Use the same recording and run the commands sequentially. The profile saves the actual speaker timeline, per-chunk intervals, proposal-length counts, and accepted-prefix counts. WhisperX checks the recording's SHA-256 before using the timeline. VibeVoice used MLX 0.32.5, mlx-audio 0.6.8, faster-whisper 1.2.1 and CTranslate2 4.9.1. WhisperX used version 3.8.6, Torch 2.8.0 and CTranslate2 4.9.1.

```sh
PYTHONPATH=. "$MURMUR_PYTHON" benchmarks/profile_vibevoice_mac.py vibevoice "$WAV" \
  --checkpoint "$VIBEVOICE_BF16" --draft-precision cpu-int8 --output vibevoice.json
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 "$WHISPERX_PYTHON" benchmarks/profile_vibevoice_mac.py whisperx "$WAV" \
  --checkpoint "$WHISPERX_LARGE_V3" --timeline vibevoice.json --output whisperx.json
```

For the quantized Whisper comparison, run all three precisions sequentially in the same environment. The target checkpoint is INT4 in all three runs. The JSON records the target transcript for scoring, the draft intervals, and the time and batch occupancy of target verification.

```sh
for precision in int8 fp16 int4; do
  PYTHONPATH=. "$MURMUR_PYTHON" benchmarks/profile_vibevoice_mac.py vibevoice "$WAV" \
    --checkpoint "$VIBEVOICE_INT4" --batch-size 16 --chunk-s 60 \
    --draft-precision "$precision" --output "whisper-${precision}.json" || break
done
```
