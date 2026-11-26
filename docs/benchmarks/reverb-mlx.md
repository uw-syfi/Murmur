# Reverb INT8 segmentation on Apple silicon

The experimental Reverb backend loads the sherpa-onnx Reverb v1 INT8 checkpoint directly. MLX runs segmentation on the GPU. The SincNet front end uses the export's quantized filter parameters. The convolutions and dense projections use packed 8-bit MLX matrix multiplication. A Metal kernel dynamically quantizes each LSTM hidden state and accumulates signed recurrent weight products in INT32. Normalization, gates and output scores use FP32. Each window has its own activation scale, matching the one-window ONNX control.

## M4 Max results

One warm run per configuration on two microphones capturing the same 1,049.35-second AMI ES2004a meeting. Reverb processes 1,041 overlapping windows. The ONNX Runtime 1.32.0 control uses four CPU threads. MLX 0.32.5 batches 32 windows. Both pipelines use the same MLX TitaNet Small embedder (batch 8), CPU feature extraction and four-speaker clustering. Timings exclude disk I/O, model loading, transcription and scoring.

| Microphone | Reverb ONNX CPU | Reverb MLX GPU | Total with ONNX Reverb | Total with MLX Reverb | Frame agreement | ONNX / MLX pipeline DER |
|---|---:|---:|---:|---:|---:|---:|
| Headset mix | 16.47 s | 4.83 s | 29.05 s | 18.25 s | 99.790% | 65.74% / 44.63% |
| Distant mic | 17.97 s | 5.08 s | 31.24 s | 18.22 s | 99.811% | 60.60% / 75.20% |

Frame agreement compares the winning powerset class on all 613,149 window frames per capture. MLX differed on 1,287 headset frames and 1,158 distant-mic frames. The speech excerpts selected for TitaNet and the final speaker assignments changed. Headset turns went from 109 to 123 and distant-mic turns from 80 to 65. DER includes overlapping speech and uses a 0.25-second collar. The regular Murmur MLX transcription path uses pyannote 3.1. [Pyannote comparison](pyannote-mac.md)

## Run

Install `mlx`, `onnx`, `onnxruntime`, `numpy` and `soundfile` on an Apple silicon Mac. The combined benchmark also uses `sherpa-onnx` (tested with 2.0.3). Supply the original Reverb INT8 and NeMo TitaNet Small ONNX checkpoints and mono 16 kHz audio.

```sh
python benchmarks/reverb_mlx.py model.int8.onnx meeting.wav --compare

python benchmarks/ami_titanet_gpu.py meeting.wav \
    --segmentation model.int8.onnx --segmentation-device mlx \
    --titanet nemo_en_titanet_small.onnx --output meeting.mlx.json
```

The first command measures standalone segmentation and compares frame predictions. The second writes stage timings and speaker turns. Use `--segmentation-device cpu` for the matched ONNX control while keeping TitaNet on the GPU.
