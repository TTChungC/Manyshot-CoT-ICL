"""Demonstration sampling and ordering helpers."""

from __future__ import annotations

import datetime
from typing import List, Optional, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


def shuffle_demos(items: List[T], seed: Optional[int] = None) -> Tuple[List[T], int]:
    """Shuffle a list in place and return (shuffled_list, seed_used)."""
    if seed is None:
        now = datetime.datetime.now()
        seed = now.minute + now.second
    rng = np.random.default_rng(seed)
    out = list(items)
    rng.shuffle(out)
    return out, int(seed)

