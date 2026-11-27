"""Export the two pinned pyannote 3.1 checkpoints for Murmur's MLX runtime.

Run python -m murmur.diarization.convert OUTPUT with the diarize and mlx-diarize
extras and Hugging Face access to pyannote/segmentation-3.0.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import mlx.core as mx

SOURCES = {
    "segmentation": ("pyannote/segmentation-3.0", "e66f3d3b9eb0873085418a7b813d3b369bf160bb", "mit"),
    "wespeaker": ("pyannote/wespeaker-voxceleb-resnet34-LM", "837717ddb9ff5507820346191109dc79c958d614", "cc-by-4.0"),
}


def segmentation_weights(state):
    prefix = "sincnet.conv1d.0.filterbank."
    n, window = state[prefix + "n_"], state[prefix + "window_"]
    low = 50 + mx.abs(state[prefix + "low_hz_"])
    high = mx.minimum(low + 50 + mx.abs(state[prefix + "band_hz_"]), 8000)
    bandwidth = 2 * (high - low)
    even = (mx.sin(high * n) - mx.sin(low * n)) * (window / (n / 2))
    odd = (mx.cos(low * n) - mx.cos(high * n)) * (window / (n / 2))
    even = mx.concatenate([even, bandwidth, even[:, ::-1]], axis=1) / bandwidth
    odd = mx.concatenate([odd, mx.zeros_like(bandwidth), -odd[:, ::-1]], axis=1) / bandwidth
    weights = {"sincnet.filters": mx.concatenate([even, odd])[:, :, None]}
    for key, value in state.items():
        if key.startswith(prefix) or key.startswith("lstm.bias_hh"):
            continue
        if key.startswith("lstm.bias_ih"):
            weights[key.replace("bias_ih", "bias")] = value + state[key.replace("bias_ih", "bias_hh")]
        elif value.ndim == 3 and key.endswith(".weight"):
            weights[key] = value.transpose(0, 2, 1)
        elif value.ndim == 2 and "weight" in key:
            weights[key] = value.T
        else:
            weights[key] = value
    return weights


def wespeaker_weights(state):
    state = {key.removeprefix("resnet."): value for key, value in state.items()}
    weights = {"seg_1.weight": state["seg_1.weight"].T, "seg_1.bias": state["seg_1.bias"]}
    for key in state:
        if key.endswith(".running_mean"):
            bn = key.removesuffix(".running_mean")
            conv = bn.replace("bn", "conv") if "bn" in bn else bn.replace("shortcut.1", "shortcut.0")
            gain = state[bn + ".weight"] * mx.rsqrt(state[bn + ".running_var"] + 1e-5)
            weights[conv + ".weight"] = state[conv + ".weight"].transpose(0, 2, 3, 1) * gain[:, None, None, None]
            weights[conv + ".bias"] = state[bn + ".bias"] - state[key] * gain
    return weights


def export(directory):
    import torch
    from huggingface_hub import hf_hub_download
    from pyannote.audio.core.task import Problem, Resolution, Specifications

    directory.mkdir(parents=True, exist_ok=True)
    config = {"format": "murmur-diarization-mlx-v1", "dtype": "float32", "sample_rate": 16000,
              "layout": {"conv1d": "OKI", "conv2d": "OHWI", "linear_and_lstm": "IO"},
              "components": {}}
    converters = {"segmentation": segmentation_weights, "wespeaker": wespeaker_weights}
    for name, (repo, revision, license_id) in SOURCES.items():
        path = hf_hub_download(repo, "pytorch_model.bin", revision=revision)
        safe = [Problem, Resolution, Specifications, torch.torch_version.TorchVersion]
        with torch.serialization.safe_globals(safe):
            state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        arrays = {k: mx.array(v.numpy()) for k, v in state.items() if not k.endswith("num_batches_tracked")}
        weights = converters[name](arrays)
        if any(v.dtype != mx.float32 for v in weights.values()):
            raise ValueError(f"Expected FP32 weights in {name}")
        file = directory / f"{name}.safetensors"
        mx.save_safetensors(str(file), weights)
        config["components"][name] = {"file": file.name, "repo": repo, "revision": revision,
                                      "license": license_id, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
    repo, revision, _ = SOURCES["segmentation"]
    shutil.copyfile(hf_hub_download(repo, "LICENSE", revision=revision), directory / "LICENSE-segmentation")
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    export(parser.parse_args().directory)
