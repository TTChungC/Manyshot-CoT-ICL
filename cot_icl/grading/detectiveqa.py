"""DetectiveQA multiple-choice answer extraction."""

from __future__ import annotations

import re


def extract_answer(text: str) -> str | None:
    match = re.search(r"The answer is ([A-D])\.", text)
    return match.group(1) if match else None
