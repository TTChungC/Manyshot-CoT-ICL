"""Shared demonstration prompt construction (MATH / reasoning)."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Sequence

from cot_icl.demos import shuffle_demos


class DemoMode(str, Enum):
    ORI = "ori"
    SHUFFLE = "shuffle"


MATH_DEMO_TEMPLATE = "Problem:\n{problem}\nSolution:\n{solution}"


def _format_math_demo(problem: str, solution: str) -> str:
    return MATH_DEMO_TEMPLATE.format(problem=problem, solution=solution)


def build_math_demo_prompt(
    problems: Sequence[str],
    solutions: Sequence[str],
    mode: DemoMode | str,
    seed: Optional[int] = None,
) -> tuple[str, Optional[int]]:
    """
    Build concatenated MATH demonstration block.

    Returns (demo_prompt, seed_used).
    """
    mode = DemoMode(mode)
    demos = [_format_math_demo(problems[i], solutions[i]) for i in range(len(problems))]

    if mode == DemoMode.SHUFFLE:
        pairs = list(zip(problems, solutions))
        pairs, used_seed = shuffle_demos(pairs, seed)
        demos = [_format_math_demo(p, s) for p, s in pairs]
        return "\n\n".join(demos) + "\n\n", used_seed

    return "\n\n".join(demos) + "\n\n", seed
