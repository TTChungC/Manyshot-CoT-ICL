"""DetectiveQA inference with self-generated first-pass demonstrations."""

from __future__ import annotations

import argparse
import json

from cot_icl.config import load_vllm
from cot_icl.demos import shuffle_demos
from cot_icl.detectiveqa.data import INSTRUCT_PROMPT, load_test_examples
from cot_icl.grading.detectiveqa import extract_answer
from cot_icl.paths import detectiveqa_reinforce_json, infer_res_dir
from cot_icl.prompts.demo_modes import DemoMode
from cot_icl.runners.common import apply_user_chat, write_json_record
from vllm import SamplingParams


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DetectiveQA reinforce (first-pass demos)")
    parser.add_argument("-r", "--round", type=float, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-j", "--json", type=str, default=None)
    parser.add_argument("-s", "--seed", type=int, default=71)
    parser.add_argument("-t", "--token", type=str, default="")
    parser.add_argument(
        "-d",
        "--demo_mode",
        type=str,
        default="ori",
        choices=[DemoMode.ORI.value, DemoMode.SHUFFLE.value],
    )
    args = parser.parse_args(argv)

    source = args.json or args.model
    n_demos = int(2 ** args.round)
    pool_path = detectiveqa_reinforce_json(source, args.token)
    print(pool_path)
    with open(pool_path, encoding="utf-8") as fp:
        pool = json.load(fp)

    demo_items = [pool["tq"][i] + pool["complete_response"][i] for i in range(n_demos)]
    if args.demo_mode == DemoMode.SHUFFLE.value:
        demo_items, seed = shuffle_demos(demo_items, args.seed)
    else:
        seed = args.seed
    demo_prompt = "\n\n".join(demo_items) + "\n\n"

    test = load_test_examples()
    llm, tokenizer, spec = load_vllm(args.model)

    all_prompts = []
    all_tests = []
    for tq, ta, tr in test:
        prompt = INSTRUCT_PROMPT + demo_prompt + "\n\n" + tq
        all_prompts.append(apply_user_chat(tokenizer, prompt, enable_thinking=spec.enable_thinking))
        all_tests.append((tq, ta, tr, prompt))

    outputs = llm.generate(
        all_prompts,
        sampling_params=SamplingParams(
            temperature=0.0,
            top_p=1,
            max_tokens=8172,
            stop_token_ids=[tokenizer.eos_token_id],
        ),
    )

    corr = 0
    record = {
        "tq": [],
        "ta": [],
        "tr": [],
        "complete_response": [],
        "refined_response": [],
        "token_size": None,
        "score": None,
        "seed": seed,
    }
    for (tq, ta, tr, prompt), output in zip(all_tests, outputs):
        response = output.outputs[0].text
        pred = extract_answer(response)
        record["tq"].append(tq)
        record["ta"].append(ta)
        record["tr"].append(tr)
        record["complete_response"].append(response)
        record["refined_response"].append(pred)
        if record["token_size"] is None:
            record["token_size"] = len(tokenizer.encode(prompt))
        if pred == ta:
            corr += 1

    record["score"] = corr / len(test) if test else 0.0
    out_path = infer_res_dir() / f"{args.model}_detectiveqa_reinforce_{args.round}{args.token}.jsonl"
    write_json_record(out_path, record)
