"""GSM8K demonstration prompt formatting."""

from __future__ import annotations

from typing import Sequence


def format_gsm8k_pairs(data) -> list[list[str]]:
    return [[data["question"][i], data["answer"][i]] for i in range(len(data["question"]))]


def build_gsm8k_demo_prompt(demo_pairs: Sequence[Sequence[str]], *, use_cot: bool = True) -> str:
    demo = ""
    for question, answer in demo_pairs:
        if use_cot:
            demo += "Q: " + question + "\nA: " + answer.split("####")[0] + "\n"
        else:
            demo += "Q: " + question + "\nA: " + answer.split("####")[1] + "\n\n"
    return demo


def build_gsm8k_query_prompt(demo_prompt: str, question: str) -> str:
    return (
        "In the end of the response, add a summary `The answer is [answer].`\n\n"
        + demo_prompt
        + "### Q: "
        + question
        + "\n### A: "
    )
