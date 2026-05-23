"""Generate MATH self-generated demonstration pools (first-answer / best-of-n)."""

from __future__ import annotations

import argparse

import datasets
from vllm import SamplingParams

from cot_icl.config import load_vllm
from cot_icl.grading.math import (
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
    process_docs,
    remove_boxed,
)
from cot_icl.paths import math_reinforce_dir, math_subtask_dir
from cot_icl.runners.common import apply_user_chat, write_json_record

POOL_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request and wrap the final answer inside \\boxed{{{{}}}}. "
    "Keep your EXPLANATION IN SHORT.\n\n"
    "### Problem:\n{problem}\n\n### Solution:\n"
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MATH first-pass demo pool generation")
    parser.add_argument("-r", "--round", type=int, required=True, help="unused; kept for CLI compat")
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-s", "--subtask", type=str, default="counting_and_probability")
    parser.add_argument("-t", "--token", type=str, default="", help="e.g. _8k caps generation length")
    args = parser.parse_args(argv)

    math_path = math_subtask_dir(args.subtask)
    train_doc = process_docs(datasets.load_from_disk(str(math_path))["train"])
    demo_doc = train_doc

    llm, tokenizer, spec = load_vllm(args.model, distributed_executor_backend="mp")
    max_tk = int(args.token[1:-1]) if args.token.startswith("_") and args.token.endswith("k") else 4096

    record = {
        "idx": [],
        "problem": [],
        "solution": [],
        "complete_response": [],
        "extracted_response": [],
        "label": [],
        "token_size": 0,
        "seed": None,
        "score": None,
    }
    results: list[bool] = []

    for tdx, test in enumerate(demo_doc):
        record["idx"].append(tdx)
        record["problem"].append(test["problem"])
        record["solution"].append(test["solution"])
        prev, prevlen, prev_has_box = None, 10_000, False
        found = False
        prompt = POOL_PROMPT.format(problem=test["problem"])

        for _attempt in range(13):
            cur_prompt = apply_user_chat(
                tokenizer, prompt, enable_thinking=spec.enable_thinking
            )
            output = llm.generate(
                cur_prompt,
                sampling_params=SamplingParams(
                    temperature=1.0,
                    max_tokens=4096,
                    stop_token_ids=[tokenizer.eos_token_id],
                ),
            )
            response = output[0].outputs[0].text.split("</think>")[-1]
            boxed = last_boxed_only_string(response)
            curlen = len(tokenizer(response).input_ids)
            if curlen > (max_tk + 40) or boxed is None:
                if curlen < prevlen and ((boxed is None and not prev_has_box) or boxed is not None):
                    prev = response
                    prevlen = curlen
                    if boxed is not None:
                        prev_has_box = True
                continue

            found = True
            pred = normalize_final_answer(remove_boxed(boxed))
            record["complete_response"].append(response)
            record["extracted_response"].append(pred)
            results.append(is_equiv(pred, test["answer"]))
            break

        if not found:
            record["complete_response"].append(prev)
            record["extracted_response"].append("")
            results.append(False)

    record["score"] = sum(results) / len(results) if results else 0.0
    out_name = f"allinc_noicl_inshort_firstans_{args.model}_math_{args.subtask}_all{args.token}.jsonl"
    write_json_record(math_reinforce_dir() / out_name, record)
