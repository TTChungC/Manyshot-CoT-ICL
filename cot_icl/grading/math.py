"""MATH grading utilities (from EleutherAI lm-evaluation-harness / Minerva)."""

from __future__ import annotations

import logging
import re
import signal
from typing import Dict, List, Optional

import datasets

try:
    import sympy
    from sympy.parsing.latex import parse_latex
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "sympy is required for MATH grading. Install with: pip install sympy"
    ) from exc

eval_logger = logging.getLogger(__name__)

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "ft", "hours", "km",
    "units", "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents",
    "degrees", "cm", "gm", "pounds", "meters", "meals", "edges", "students",
    "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
    r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]
    left = "\\boxed{"
    if s[: len(left)] != left:
        return s
    assert s[-1] == "}"
    return s[len(left) : -1]


def normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        return {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": normalize_final_answer(
                remove_boxed(last_boxed_only_string(doc["solution"]))
            ),
        }

    return dataset.map(_process_doc)


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                return False
            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                return False
            try:
                return sympy.simplify(diff) == 0
            except ValueError:
                return False
    except TimeoutError:
        return False
    except Exception:
        return False


def grade_boxed_response(response: str, gold: str) -> bool:
    boxed = last_boxed_only_string(response)
    if boxed is None:
        return False
    pred = normalize_final_answer(remove_boxed(boxed))
    return is_equiv(pred, gold)
