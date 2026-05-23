"""GSM8K answer parsing (CoT setting)."""

from __future__ import annotations

import re

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
ANSWER_TRIGGER = "The answer is"


def extract_answer_from_output(completion: str) -> str:
    match = ANS_RE.search(completion)
    if match:
        return match.group(1).strip().replace(",", "")
    return INVALID_ANS


def clean_answer(model_pred: str) -> str:
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = len(preds) > 1
    pred = preds[1] if answer_flag else preds[-1]
    pred = pred.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", pred)
    if not numbers:
        return INVALID_ANS
    pred = numbers[0] if answer_flag else numbers[-1]
    if pred.endswith("."):
        pred = pred[:-1]
    return pred


def is_correct_gsm8k(model_answer: str, reference: str) -> bool:
    gt = extract_answer_from_output(reference)
    assert gt != INVALID_ANS
    return model_answer == gt
