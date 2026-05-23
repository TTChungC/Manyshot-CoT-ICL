"""MATH many-shot CoT-ICL inference (gold demonstrations)."""

from __future__ import annotations

import argparse
import time

import datasets
from openai import OpenAI
from vllm import SamplingParams

from cot_icl.config import get_model_spec, load_vllm
from cot_icl.demos import shuffle_demos
from cot_icl.grading.math import (
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
    process_docs,
    remove_boxed,
)
from cot_icl.paths import infer_res_dir, math_subtask_dir
from cot_icl.prompts.demo_modes import DemoMode
from cot_icl.runners.common import api_base_url, apply_user_chat, batch_generate, write_json_record

PROBLEM_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request and wrap the final answer inside \\boxed{{}}.\n\n"
    "{demo_prompt}"
    "### Problem:\n{problem}\n\n### Solution: Let's think step by step."
)

DEMO_TEMPLATE = "Problem:\n{problem}\nSolution:\n{solution}"


def build_demo_prompt(demo_doc, n_demos: int, mode: str, seed: int | None) -> tuple[str, int | None]:
    mode = DemoMode(mode)
    if mode == DemoMode.SHUFFLE:
        pairs = [[demo_doc["problem"][i], demo_doc["solution"][i]] for i in range(n_demos)]
        pairs, used_seed = shuffle_demos(pairs, seed)
        block = "\n\n".join(
            DEMO_TEMPLATE.format(problem=p, solution=s) for p, s in pairs
        )
        return block + "\n\n", used_seed

    block = "\n\n".join(
        DEMO_TEMPLATE.format(problem=demo_doc["problem"][i], solution=demo_doc["solution"][i])
        for i in range(n_demos)
    )
    return block + "\n\n", seed


def grade_response(response: str, gold: str) -> tuple[bool, str | None]:
    boxed = last_boxed_only_string(response)
    if boxed is None:
        return False, None
    pred = normalize_final_answer(remove_boxed(boxed))
    return is_equiv(pred, gold), pred


def run_local_batch(llm, tokenizer, test_doc, demo_prompt: str, spec, record: dict) -> list[bool]:
    all_prompts = []
    all_tests = []
    for tdx, test in enumerate(test_doc):
        if tdx < record["lasttdx"]:
            continue
        test = dict(test)
        test["demo_prompt"] = demo_prompt
        prompt = PROBLEM_PROMPT.format(**test)
        all_prompts.append(
            apply_user_chat(tokenizer, prompt, enable_thinking=spec.enable_thinking)
        )
        all_tests.append((tdx, test))

    print(f"[batch inference] total prompts: {len(all_prompts)}")
    outputs = llm.generate(
        all_prompts,
        sampling_params=SamplingParams(
            temperature=0.0,
            top_p=1,
            max_tokens=8172,
            repetition_penalty=1.1,
            stop_token_ids=[tokenizer.eos_token_id],
        ),
    )

    results: list[bool] = []
    for (tdx, test), output in zip(all_tests, outputs):
        response = output.outputs[0].text
        print("response:", response)
        record["complete_response"].append(response)
        correct, pred = grade_response(response, test["answer"])
        if pred is None:
            results.append(False)
        else:
            record["extracted_response"].append(pred)
            record["label"].append(test["answer"])
            results.append(correct)
        print("correct:", correct, "tdx:", tdx, "acc:", sum(results) / len(results), "\n")
    return results


def run_api_user(test_doc, demo_prompt: str, user: str, model_path: str, record: dict) -> list[bool]:
    base = api_base_url(user)
    client = OpenAI(api_key="EMPTY", base_url=base)
    results: list[bool] = []
    savepath = record.get("_savepath")

    for tdx, test in enumerate(test_doc):
        if tdx < record["lasttdx"]:
            continue
        test = dict(test)
        test["demo_prompt"] = demo_prompt
        prompt = PROBLEM_PROMPT.format(**test)
        response_text = None
        for runcount in range(11):
            try:
                chat_response = client.chat.completions.create(
                    model=model_path.replace("../", ""),
                    messages=[
                        {
                            "role": "system",
                            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                response_text = chat_response.choices[0].message.content
                break
            except Exception:
                record["lasttdx"] = tdx
                if savepath:
                    write_json_record(savepath, record)
                time.sleep(1)
        if not response_text:
            results.append(False)
            continue
        record["complete_response"].append(response_text)
        correct, pred = grade_response(response_text, test["answer"])
        if pred is None:
            results.append(False)
        else:
            record["extracted_response"].append(pred)
            record["label"].append(test["answer"])
            results.append(correct)
    return results


def output_path(model: str, subtask: str, r: float, demo_mode: str, seed: int | None):
    if demo_mode == DemoMode.SHUFFLE.value:
        return infer_res_dir("new", "shuffle") / f"{model}_math_{subtask}_{r}_seed{seed}.jsonl"
    return infer_res_dir("new") / f"{model}_math_{subtask}_{r}.jsonl"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MATH many-shot CoT-ICL inference")
    parser.add_argument("-r", "--round", type=float, required=True, help="log2(#demos)")
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-t", "--subtask", type=str, default="counting_and_probability")
    parser.add_argument("-s", "--seed", type=int, default=71)
    parser.add_argument("-b", "--beginning", type=int, default=0)
    parser.add_argument("-u", "--user", type=str, default=None, help="remote API user slot")
    parser.add_argument(
        "-d",
        "--demo_mode",
        type=str,
        default="ori",
        choices=[DemoMode.ORI.value, DemoMode.SHUFFLE.value],
    )
    args = parser.parse_args(argv)

    n_demos = int(2 ** args.round)
    math_path = math_subtask_dir(args.subtask)
    train_doc = process_docs(datasets.load_from_disk(str(math_path))["train"])
    test_doc = process_docs(datasets.load_from_disk(str(math_path))["test"])
    demo_doc = train_doc.select(range(args.beginning, args.beginning + n_demos))

    demo_prompt, seed = build_demo_prompt(demo_doc, n_demos, args.demo_mode, args.seed)
    print(f"[demo_mode] {args.demo_mode}")

    savepath = output_path(args.model, args.subtask, args.round, args.demo_mode, seed)
    record = {
        "complete_response": [],
        "extracted_response": [],
        "label": [],
        "lasttdx": 0,
        "token_size": 0,
        "seed": seed,
        "score": None,
        "_savepath": savepath,
    }
    print(savepath)

    if args.user is None:
        llm, tokenizer, spec = load_vllm(args.model)
        res = run_local_batch(llm, tokenizer, test_doc, demo_prompt, spec, record)
    else:
        spec = get_model_spec(args.model)
        res = run_api_user(test_doc, demo_prompt, args.user, spec.path, record)

    record.pop("_savepath", None)
    record["score"] = sum(res) / len(res) if res else 0.0
    print("acc:", record["score"])
    write_json_record(savepath, record)
