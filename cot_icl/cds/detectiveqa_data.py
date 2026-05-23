"""DetectiveQA demonstration loading and embeddings for CDS."""

from __future__ import annotations

import json
import os
from typing import List

import torch

from cot_icl.cds.embeddings import get_embeddings
from cot_icl.paths import detectiveqa_demo_dir

# Train/test overlap IDs to exclude from demonstrations (paper setup).
OVERLAPPING_IDS = {
    "118", "124", "79", "140", "219", "209", "27", "134", "145", "149",
    "144", "150", "241", "137", "128", "252", "151", "198", "117", "142",
    "121", "126", "136", "203",
}

DEMO_TEMPLATE = (
    "Question: {question}\n"
    "Context: {context}\n"
    "Options:\n"
    "A. {optA}\n"
    "B. {optB}\n"
    "C. {optC}\n"
    "D. {optD}\n"
    "Answer:\n{answer_reasoning}\n"
    "The answer is {answer}."
)


def load_detectiveqa_demos(demo_path: str | None = None) -> List[str]:
    demo_path = demo_path or str(detectiveqa_demo_dir())
    demos: List[str] = []
    for root, _, files in os.walk(demo_path):
        for file in sorted(files):
            if not file.endswith(".json"):
                continue
            file_id = os.path.splitext(file)[0]
            if file_id in OVERLAPPING_IDS:
                continue
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                data = json.load(f)
            for q in data[0]["questions"]:
                context = " ".join(
                    reasoning
                    for i, reasoning in enumerate(q["reasoning"])
                    if q["clue_position"][i] != -1
                )
                answer_reasoning = " ".join(
                    reasoning
                    for i, reasoning in enumerate(q["reasoning"])
                    if q["clue_position"][i] == -1
                ).strip()
                demos.append(
                    DEMO_TEMPLATE.format(
                        question=q["question"],
                        context=context,
                        optA=q["options"]["A"],
                        optB=q["options"]["B"],
                        optC=q["options"]["C"],
                        optD=q["options"]["D"],
                        answer_reasoning=answer_reasoning,
                        answer=q["answer"],
                    )
                )
    return demos


def get_detectiveqa_embeddings(r: int, embed_model: str, **kwargs) -> torch.Tensor:
    demos = load_detectiveqa_demos()
    n = int(2**r)
    if n > len(demos):
        raise ValueError(f"Requested 2^{r}={n} demos but only {len(demos)} available")
    return get_embeddings(demos[:n], embed_model, **kwargs)
