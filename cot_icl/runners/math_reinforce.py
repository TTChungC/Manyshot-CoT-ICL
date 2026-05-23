"""MATH inference with self-generated chain-of-thought demonstrations."""

from __future__ import annotations

import argparse

import datasets

from cot_icl.config import load_vllm
from cot_icl.demos import shuffle_demos
from cot_icl.grading.math import (
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
    process_docs,
    remove_boxed,
)
from cot_icl.paths import infer_res_dir, math_reinforce_json, math_subtask_dir
from cot_icl.prompts.demo_modes import DemoMode
from cot_icl.runners.common import apply_user_chat, load_reinforce_json, write_json_record
from cot_icl.runners.math import PROBLEM_PROMPT, DEMO_TEMPLATE
from vllm import SamplingParams


def build_reinforce_demo_prompt(
    ridata: dict,
    n_demos: int,
    mode: str,
    seed: int | None,
) -> tuple[str, int | None]:
    mode = DemoMode(mode)
    if mode == DemoMode.SHUFFLE:
        pairs = [
            [ridata["problem"][i], ridata["complete_response"][i]] for i in range(n_demos)
        ]
        pairs, used_seed = shuffle_demos(pairs, seed)
        block = "\n\n".join(
            DEMO_TEMPLATE.format(problem=p, solution=s) for p, s in pairs
        )
        return block + "\n\n", used_seed

    block = "\n\n".join(
        DEMO_TEMPLATE.format(problem=ridata["problem"][i], solution=ridata["complete_response"][i])
        for i in range(n_demos)
    )
    return block + "\n\n", seed


def output_path(model: str, subtask: str, r: float, demo_mode: str, seed: int | None, suffix: str):
    if demo_mode == DemoMode.SHUFFLE.value:
        return (
            infer_res_dir("shuffle")
            / f"{model}_math_{subtask}_{r}_seed{seed}{suffix}.jsonl"
        )
    return infer_res_dir() / f"{model}_math_{subtask}_{r}{suffix}.jsonl"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MATH reinforce (self-generated demos)")
    parser.add_argument("-r", "--round", type=float, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-t", "--subtask", type=str, default="counting_and_probability")
    parser.add_argument("-j", "--json", type=str, default=None, help="source model for demo JSON")
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument(
        "-d",
        "--demo_mode",
        type=str,
        default="ori",
        choices=[DemoMode.ORI.value, DemoMode.SHUFFLE.value],
    )
    parser.add_argument(
        "--pool-variant",
        type=str,
        default="correctans",
        choices=["correctans", "wrongans", "firstans"],
        help="which self-gen pool file under data/math/",
    )
    parser.add_argument("--token-suffix", type=str, default="", help="filename suffix for firstans pool")
    args = parser.parse_args(argv)

    source = args.json or args.model
    n_demos = int(2 ** args.round)
    json_path = math_reinforce_json(
        source,
        args.subtask,
        variant=args.pool_variant,
        token_suffix=args.token_suffix,
    )
    ridata = load_reinforce_json(json_path)

    math_path = math_subtask_dir(args.subtask)
    test_doc = process_docs(datasets.load_from_disk(str(math_path))["test"])
    demo_prompt, seed = build_reinforce_demo_prompt(ridata, n_demos, args.demo_mode, args.seed)

    llm, tokenizer, spec = load_vllm(
        args.model,
        distributed_executor_backend="mp",
    )

    all_prompts = []
    all_tests = []
    for tdx, test in enumerate(test_doc):
        test = dict(test)
        test["demo_prompt"] = demo_prompt
        prompt = PROBLEM_PROMPT.format(**test)
        all_prompts.append(apply_user_chat(tokenizer, prompt, enable_thinking=spec.enable_thinking))
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

    res = []
    record = {
        "complete_response": [],
        "extracted_response": [],
        "label": [],
        "token_size": 0,
        "seed": seed,
        "score": None,
    }
    suffix = "_r" if args.pool_variant == "correctans" else f"_{args.pool_variant}"
    savepath = output_path(args.model, args.subtask, args.round, args.demo_mode, seed, suffix)

    for (tdx, test), output in zip(all_tests, outputs):
        response = output.outputs[0].text
        record["complete_response"].append(response)
        boxed = last_boxed_only_string(response)
        if boxed is None:
            res.append(False)
        else:
            pred = normalize_final_answer(remove_boxed(boxed))
            record["extracted_response"].append(pred)
            record["label"].append(test["answer"])
            res.append(is_equiv(pred, test["answer"]))
        print("tdx:", tdx, "acc:", sum(res) / len(res))

    record["score"] = sum(res) / len(res) if res else 0.0
    write_json_record(savepath, record)
