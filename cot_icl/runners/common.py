"""Shared vLLM / chat helpers for experiment runners."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from vllm import SamplingParams


def apply_user_chat(
    tokenizer,
    content: str,
    *,
    enable_thinking: bool = False,
) -> str:
    chat = [{"role": "user", "content": content}]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking:
        kwargs["enable_thinking"] = True
    return tokenizer.apply_chat_template(chat, **kwargs)


def batch_generate(
    llm,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_tokens: int = 8172,
    temperature: float = 0.0,
    repetition_penalty: float | None = None,
    enable_thinking: bool = False,
) -> list[str]:
    formatted = [
        apply_user_chat(tokenizer, p, enable_thinking=enable_thinking) for p in prompts
    ]
    sp_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stop_token_ids": [tokenizer.eos_token_id],
    }
    if repetition_penalty is not None:
        sp_kwargs["repetition_penalty"] = repetition_penalty
    outputs = llm.generate(formatted, sampling_params=SamplingParams(**sp_kwargs))
    return [o.outputs[0].text for o in outputs]


def write_json_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(record, fp)
    print(f"Wrote {path}")


def api_base_url(user: str | None) -> str | None:
    if user is None:
        return None
    template = os.environ.get("COT_ICL_API_BASE", "http://21.0.198.55/{user}/v1")
    return template.format(user=user)


def drop_null_reinforce_entries(ridata: dict) -> dict:
    """Remove demo rows whose model response is None."""
    remove_idx = [i for i, r in enumerate(ridata["complete_response"]) if r is None]
    for key in ridata:
        if isinstance(ridata[key], list) and len(ridata[key]) == len(ridata["complete_response"]):
            for i in reversed(remove_idx):
                del ridata[key][i]
    return ridata


def load_reinforce_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    return drop_null_reinforce_entries(data)
