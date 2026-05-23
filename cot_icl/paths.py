"""Project paths (override via environment variables for your machine)."""

from __future__ import annotations

import os
from pathlib import Path

# Repository root (directory that contains cot_icl/, scripts/, data/, …)
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("COT_ICL_DATA", ROOT / "data"))
MODELS_ROOT = Path(os.environ.get("COT_ICL_MODELS", ROOT.parent))


def gsm8k_dir() -> Path:
    return DATA_ROOT / "gsm8k"


def math_subtask_dir(subtask: str) -> Path:
    return DATA_ROOT / "hendrycks_math" / subtask


def detectiveqa_demo_dir() -> Path:
    return DATA_ROOT / "DetectiveQA" / "anno_data_en" / "AIsup_anno"


def detectiveqa_test_dir() -> Path:
    return DATA_ROOT / "DetectiveQA" / "anno_data_en" / "human_anno"


def infer_res_dir(*parts: str) -> Path:
    path = DATA_ROOT / "infer_res"
    for p in parts:
        path = path / p
    path.mkdir(parents=True, exist_ok=True)
    return path


def gsm8k_reinforce_dir() -> Path:
    path = DATA_ROOT / "gsm8k_r"
    path.mkdir(parents=True, exist_ok=True)
    return path


def math_reinforce_dir() -> Path:
    path = DATA_ROOT / "math"
    path.mkdir(parents=True, exist_ok=True)
    return path


def detectiveqa_reinforce_dir() -> Path:
    path = DATA_ROOT / "detectiveqa"
    path.mkdir(parents=True, exist_ok=True)
    return path


def math_reinforce_json(
    source_model: str,
    subtask: str,
    *,
    variant: str = "correctans",
    token_suffix: str = "",
) -> Path:
    """Self-generated MATH demo pool JSON (reinforce experiments)."""
    name = f"{variant}_{source_model}_math_{subtask}_all{token_suffix}.jsonl"
    return math_reinforce_dir() / name


def gsm8k_reinforce_json(
    source_model: str,
    *,
    variant: str = "default",
    r: int | None = None,
) -> Path:
    """Self-generated GSM8K demo pool JSON."""
    if variant == "wrongans":
        return gsm8k_reinforce_dir() / f"wrongans_{source_model}_gsm8k_9.jsonl"
    if variant == "firstans":
        return gsm8k_reinforce_dir() / f"firstans_{source_model}_gsm8k_all.jsonl"
    if r is not None:
        return gsm8k_reinforce_dir() / f"{source_model}_gsm8k_{r}.jsonl"
    return gsm8k_reinforce_dir() / f"{source_model}_gsm8k_9.jsonl"


def detectiveqa_reinforce_json(source_model: str, token_suffix: str = "") -> Path:
    return detectiveqa_reinforce_dir() / f"first_{source_model}_detectiveqa_None_{token_suffix}.jsonl"


def embedding_model_path(name: str) -> str:
    """Resolve embedding model path from env or built-in defaults."""
    env_key = f"COT_ICL_EMBED_{name.upper().replace('-', '_')}"
    if env_key in os.environ:
        return os.environ[env_key]
    defaults = {
        "bge-m3": "BAAI/bge-m3",
    }
    return defaults.get(name, name)
