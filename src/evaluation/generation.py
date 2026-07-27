#!/usr/bin/env python3
"""
src/evaluation/generation.py
Any new evaluation entry script should call `load_model_and_tokenizer` here
rather than re-implementing model loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from evaluation.text_normalization import clean_generation


def get_cuda_info() -> Dict[str, Any]:
    cuda_info: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        cuda_info["device_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    return cuda_info


def load_model_and_tokenizer(
    model_id: str,
    *,
    hf_cache_dir: Optional[str] = None,
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
    attn_implementation: Optional[str] = None,
    adapter_path: Optional[str] = None,
):
    """Load a causal LM + tokenizer for inference, optionally attaching a LoRA adapter.

    Shared by zero-shot, few-shot, and fine-tuned evaluation so quantization,
    trust_remote_code, and cache-dir behavior can never silently diverge
    between them. When `adapter_path` is given, the tokenizer is loaded from
    the adapter directory (03_finetune.py saves it there alongside the
    adapter) rather than the base model, since that is guaranteed to be the
    exact tokenizer used during training.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_kwargs: Dict[str, Any] = {}
    model_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}

    if hf_cache_dir:
        cache_dir = str(Path(hf_cache_dir).expanduser().resolve())
        tokenizer_kwargs["cache_dir"] = cache_dir
        model_kwargs["cache_dir"] = cache_dir

    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    tokenizer_source = adapter_path or model_id
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=trust_remote_code,
        **tokenizer_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = None

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "BitsAndBytesConfig not available. Install compatible transformers/bitsandbytes."
            ) from e

        compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                "peft is required to load a LoRA adapter but is not installed. "
                "Install it in the active environment (pip install peft)."
            ) from e
        model = PeftModel.from_pretrained(model, adapter_path)

    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()

    return model, tokenizer


def batched(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_input_length: int,
    max_new_tokens: int,
) -> List[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )

    target_device = model.device if hasattr(model, "device") else next(model.parameters()).device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    decoded_outputs: List[str] = []
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    for i, output_ids in enumerate(outputs):
        prompt_len = int(input_lengths[i])
        generated_ids = output_ids[prompt_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        decoded_outputs.append(clean_generation(text))
    return decoded_outputs
