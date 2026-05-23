"""Model aliases and vLLM defaults used across inference scripts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from cot_icl.paths import MODELS_ROOT


@dataclass
class ModelSpec:
    alias: str
    path: str
    max_model_len: int = 131_000
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.95
    enable_thinking: bool = False


# Short names passed via `-m` / `--model` in legacy scripts.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "llama": ModelSpec("llama", str(MODELS_ROOT / "Llama-3.1-8B-Instruct"), max_model_len=51_000),
    "llama70": ModelSpec(
        "llama70",
        str(MODELS_ROOT / "Llama-3.3-70B-Instruct"),
        max_model_len=131_000,
        gpu_memory_utilization=0.9,
    ),
    "qwen": ModelSpec("qwen", str(MODELS_ROOT / "Qwen2.5-7B-Instruct"), max_model_len=131_000),
    "qwen14": ModelSpec("qwen14", str(MODELS_ROOT / "Qwen2.5-14B-Instruct"), max_model_len=131_000),
    "qwen72": ModelSpec("qwen72", str(MODELS_ROOT / "Qwen2.5-72B-Instruct"), max_model_len=131_000),
    "qwen3": ModelSpec(
        "qwen3",
        str(MODELS_ROOT / "Qwen3-8B"),
        max_model_len=131_000,
        enable_thinking=True,
    ),
    "qwen314": ModelSpec(
        "qwen314",
        str(MODELS_ROOT / "Qwen3-14B"),
        max_model_len=131_000,
        enable_thinking=True,
    ),
    "qwen332": ModelSpec(
        "qwen332",
        str(MODELS_ROOT / "Qwen3-32B"),
        max_model_len=131_000,
        tensor_parallel_size=2,
        enable_thinking=True,
    ),
    "qwq": ModelSpec(
        "qwq",
        str(MODELS_ROOT / "QwQ-32B"),
        max_model_len=131_000,
        tensor_parallel_size=2,
        enable_thinking=True,
    ),
}


def get_model_spec(alias: str) -> ModelSpec:
    if alias not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model alias '{alias}'. Supported: {supported}")
    spec = MODEL_REGISTRY[alias]
    # Allow per-model override: COT_ICL_MODEL_QWEN3=/path/to/model
    env_key = f"COT_ICL_MODEL_{alias.upper()}"
    path = os.environ.get(env_key, spec.path)
    return ModelSpec(
        alias=spec.alias,
        path=path,
        max_model_len=spec.max_model_len,
        tensor_parallel_size=spec.tensor_parallel_size,
        gpu_memory_utilization=spec.gpu_memory_utilization,
        enable_thinking=spec.enable_thinking,
    )


def load_vllm(alias: str, **overrides):
    """Load vLLM + tokenizer for a registered model alias."""
    from transformers import AutoTokenizer
    from vllm import LLM

    spec = get_model_spec(alias)
    kwargs = {
        "tensor_parallel_size": spec.tensor_parallel_size,
        "max_model_len": spec.max_model_len,
        "gpu_memory_utilization": spec.gpu_memory_utilization,
        "max_seq_len_to_capture": spec.max_model_len,
        "dtype": "float16",
        "enable_prefix_caching": True,
        "enable_chunked_prefill": False,
        "enforce_eager": True,
    }
    kwargs.update(overrides)

    if alias == "llama70":
        llm = LLM(
            spec.path,
            tensor_parallel_size=1,
            max_model_len=spec.max_model_len,
            gpu_memory_utilization=0.9,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
        )
    else:
        llm = LLM(spec.path, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(spec.path, trust_remote_code=True)
    return llm, tokenizer, spec
