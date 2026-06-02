"""ASR-only subset of the upstream tokenizer module.

Drops the TTS-flavored `VibeVoiceTextTokenizer` / `VibeVoiceTextTokenizerFast`
(which reuse `<|vision_*|>` special tokens for speech diffusion). The ASR
processor only needs `VibeVoiceASRTextTokenizerFast`.
"""
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast


class VibeVoiceASRTextTokenizerFast(Qwen2TokenizerFast):
    """Fast Qwen2 tokenizer with VibeVoice-ASR-specific speech special tokens."""

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        self._add_vibevoice_special_tokens()
        # https://github.com/QwenLM/Qwen2.5-VL/blob/d2240f11656bfe404b9ba56db4e51cd09f522ff1/qwen-vl-finetune/qwenvl/data/data_qwen_packed.py#L57
        self.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    def _add_vibevoice_special_tokens(self):
        special_tokens = {
            "additional_special_tokens": [
                "<|object_ref_start|>",
                "<|object_ref_end|>",
                "<|box_start|>",
            ]
        }
        num_added = self.add_special_tokens(special_tokens)
        self._speech_start_id = self.convert_tokens_to_ids("<|object_ref_start|>")
        self._speech_end_id = self.convert_tokens_to_ids("<|object_ref_end|>")
        self._speech_pad_id = self.convert_tokens_to_ids("<|box_start|>")
        self._eos_id = self.eos_token_id
        self._pad_id = self.convert_tokens_to_ids("<|image_pad|>")
        return num_added

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def speech_start_id(self) -> int:
        return self._speech_start_id

    @property
    def speech_end_id(self) -> int:
        return self._speech_end_id

    @property
    def speech_pad_id(self) -> int:
        return self._speech_pad_id

    @property
    def pad_id(self) -> int:
        return self._pad_id


__all__ = ["VibeVoiceASRTextTokenizerFast"]
