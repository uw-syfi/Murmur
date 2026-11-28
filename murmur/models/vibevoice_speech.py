"""Offline VibeVoice speech encoding on Apple GPUs."""
from functools import lru_cache

import mlx.core as mx

from .vibevoice_full_block import try_full_block
from .vibevoice_ffn import try_fused_ffn
from .vibevoice_tiled_mixer import try_tiled_mixer


@lru_cache(maxsize=1)
def _causal_depthwise_kernel():
    return mx.fast.metal_kernel(
        name="murmur_vibevoice_depthwise7",
        input_names=["x", "weight"], output_names=["y"],
        source="""
            uint pos = thread_position_in_grid.x;
            uint channel = pos % C;
            uint time = (pos / C) % x_shape[1];
            T acc = T(0);
            for (int k = 0; k < 7; ++k) {
                if (int(time) + k >= 6) {
                    T v = T(x[pos - (6-k)*C]) * T(weight[channel*7+k]);
                    acc = T(acc + v);
                }
            }
            y[pos] = acc;
        """,
        compile_options={"math_mode": "safe"},
    )


def _block(block, x):
    conv = getattr(block.mixer, "conv", None)
    if (not hasattr(conv, "kernel_size") or not conv.causal or conv.kernel_size != 7
            or conv.stride != 1 or conv.dilation != 1 or conv.groups != x.shape[-1]
            or x.size > 0xFFFFFFFF
            or x.shape[1] * x.shape[2] > block._MAX_BLOCK_ELEMENTS):
        return block(x)
    if x.shape[-1] in (32, 64, 128):
        output = try_full_block(block, conv.conv, x)
        if output is not None:
            return output
    pair = try_tiled_mixer(block, conv.conv, x)
    if pair is None:
        normed = block.norm(x)
        weight = conv.conv.weight
        dtype = mx.result_type(normed, weight)
        mixed = _causal_depthwise_kernel()(
            inputs=[normed, weight], template=[("T", dtype), ("C", x.shape[-1])],
            grid=(x.size, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[x.shape], output_dtypes=[dtype],
        )[0]
        if getattr(conv.conv, "bias", None) is not None:
            mixed = mixed + conv.conv.bias
        if block.gamma is not None:
            mixed = mixed * block.gamma
        x = x + mixed
        normed = block.ffn_norm(x)
    else:
        x, normed = pair
    fused = try_fused_ffn(block, normed, x)
    if fused is not None:
        return fused
    ffn = block.ffn(normed)
    if block.ffn_gamma is not None:
        ffn = ffn * block.ffn_gamma
    return x + ffn


def _encode_branch(encoder, connector, x):
    for downsample, stage in zip(encoder.downsample_layers, encoder.stages):
        x = downsample(x)
        mx.eval(x)
        # Keep at most two dependent blocks submitted within this stage.
        # For long or batched inputs preserve the original live-memory bound.
        queue_blocks = x.shape[0] == 1 and x.nbytes <= 192 * 1024**2
        previous = None
        for block in stage:
            x = _block(block, x)
            if queue_blocks:
                mx.async_eval(x)
                if previous is not None:
                    mx.eval(previous)
                previous = x
            else:
                mx.eval(x)
        if previous is not None:
            mx.eval(previous)
    if encoder.norm is not None:
        x = encoder.norm(x.transpose(0, 2, 1)).transpose(0, 2, 1)
    x = encoder.head(x)
    mx.eval(x)
    x = connector(x)
    mx.eval(x)
    return x


def encode_branches(model, waveform):
    """Return projected acoustic and semantic embeddings for the same waveform."""
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    if waveform.ndim == 2:
        waveform = waveform[:, :, None]
    elif waveform.ndim == 3 and waveform.shape[1] == 1:
        waveform = waveform.transpose(0, 2, 1)
    acoustic = _encode_branch(model.acoustic_tokenizer.encoder, model.acoustic_connector, waveform)
    semantic = _encode_branch(model.semantic_tokenizer.encoder, model.semantic_connector, waveform)
    return acoustic, semantic


def encode_speech(model, waveform):
    """Return the combined speech features consumed by Murmur prefill."""
    if mx.default_device() != mx.Device(mx.gpu) or not hasattr(model, "acoustic_tokenizer"):
        return model.encode_speech(waveform)
    acoustic, semantic = encode_branches(model, waveform)
    return acoustic + semantic
