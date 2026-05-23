"""MATH demonstration embeddings for CDS."""

from __future__ import annotations

from typing import List

import datasets
import torch

from cot_icl.cds.embeddings import get_embeddings
from cot_icl.demos import shuffle_demos
from cot_icl.grading.math import process_docs
from cot_icl.paths import math_subtask_dir


def build_math_demo_strings(subtask: str, r: int, seed: int) -> List[str]:
    data_dir = math_subtask_dir(subtask)
    train_doc = process_docs(datasets.load_from_disk(str(data_dir))["train"])
    n = int(2**r)
    demo_doc = train_doc.select(range(0, n))
    pairs = [[demo_doc["problem"][i], demo_doc["solution"][i]] for i in range(n)]
    pairs, _ = shuffle_demos(pairs, seed)
    return [f"Problem:\n{p}\nSolution:\n{s}" for p, s in pairs]


def get_math_embeddings(subtask: str, r: int, seed: int, embed_model: str, **kwargs) -> torch.Tensor:
    texts = build_math_demo_strings(subtask, r, seed)
    return get_embeddings(texts, embed_model, **kwargs)
