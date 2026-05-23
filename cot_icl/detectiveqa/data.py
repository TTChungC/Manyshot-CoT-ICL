"""Load and format DetectiveQA demonstrations and test items."""

from __future__ import annotations

import json
import os
from pathlib import Path

from cot_icl.paths import detectiveqa_demo_dir, detectiveqa_test_dir

DETECTIVEQA_OVERLAPPING = frozenset(
    {
        "118",
        "124",
        "79",
        "140",
        "219",
        "209",
        "27",
        "134",
        "145",
        "149",
        "144",
        "150",
        "241",
        "137",
        "128",
        "252",
        "151",
        "198",
        "117",
        "142",
        "121",
        "126",
        "136",
        "203",
    }
)

DEMO_TEMPLATE = (
    "Question: {question}\n"
    "Context: {context}\n"
    "Options:\n"
    "A. {options[A]}\n"
    "B. {options[B]}\n"
    "C. {options[C]}\n"
    "D. {options[D]}\n"
    "Answer:\n{answer_reasoning}\n"
    "The answer is {answer}."
)

TEST_TEMPLATE = (
    "Question: {question}\n"
    "Context: {context}\n"
    "Options:\n"
    "A. {options[A]}\n"
    "B. {options[B]}\n"
    "C. {options[C]}\n"
    "D. {options[D]}\n"
    "Answer:\n"
)

INSTRUCT_PROMPT = (
    "Below is an instruction that describes a task.\n"
    " Select the correct option from A/B/C/D. Answer with 'The answer is {A/B/C/D}.' "
    "in the end of your response.\n\n"
)


def _split_reasoning(q: dict) -> tuple[str, str]:
    context = " ".join(
        reasoning for i, reasoning in enumerate(q["reasoning"]) if q["clue_position"][i] != -1
    )
    answer_reasoning = " ".join(
        reasoning for i, reasoning in enumerate(q["reasoning"]) if q["clue_position"][i] == -1
    ).strip()
    return context, answer_reasoning


def load_demo_examples(
    demo_root: Path | None = None,
    *,
    exclude_overlapping: bool = True,
) -> list[str]:
    demo_path = demo_root or detectiveqa_demo_dir()
    demos: list[str] = []
    for root, _, files in os.walk(demo_path):
        for file in files:
            if not file.endswith(".json"):
                continue
            rel_path = os.path.relpath(os.path.join(root, file), demo_path)
            if exclude_overlapping and rel_path in DETECTIVEQA_OVERLAPPING:
                continue
            with open(os.path.join(root, file), encoding="utf-8") as fp:
                data = json.load(fp)
            for q in data[0]["questions"]:
                context, answer_reasoning = _split_reasoning(q)
                demos.append(
                    DEMO_TEMPLATE.format(
                        question=q["question"],
                        context=context,
                        options=q["options"],
                        answer_reasoning=answer_reasoning,
                        answer=q["answer"],
                    )
                )
    return demos


def load_pool_examples(demo_root: Path | None = None) -> list[tuple[str, str, str]]:
    """Demo-pool items formatted like test questions (for first-pass generation)."""
    demo_path = demo_root or detectiveqa_demo_dir()
    items: list[tuple[str, str, str]] = []
    for root, _, files in os.walk(demo_path):
        for file in files:
            if not file.endswith(".json"):
                continue
            with open(os.path.join(root, file), encoding="utf-8") as fp:
                data = json.load(fp)
            for q in data[0]["questions"]:
                context, answer_reasoning = _split_reasoning(q)
                options = {k: v.strip() for k, v in q["options"].items()}
                prompt_block = TEST_TEMPLATE.format(
                    question=q["question"],
                    context=context,
                    options=options,
                )
                items.append((prompt_block, q["answer"], answer_reasoning))
    return items


def load_test_examples(test_root: Path | None = None) -> list[tuple[str, str, str]]:
    """Return list of (prompt_block, gold_answer, answer_reasoning)."""
    test_path = test_root or detectiveqa_test_dir()
    test: list[tuple[str, str, str]] = []
    for root, _, files in os.walk(test_path):
        for file in files:
            if not file.endswith(".json"):
                continue
            with open(os.path.join(root, file), encoding="utf-8") as fp:
                data = json.load(fp)
            for q in data[0]["questions"]:
                context, answer_reasoning = _split_reasoning(q)
                options = {k: v.strip() for k, v in q["options"].items()}
                prompt_block = TEST_TEMPLATE.format(
                    question=q["question"],
                    context=context,
                    options=options,
                )
                test.append((prompt_block, q["answer"], answer_reasoning))
    return test
