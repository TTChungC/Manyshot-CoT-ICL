"""DetectiveQA many-shot CoT-ICL inference."""

from __future__ import annotations

import argparse

from cot_icl.config import load_vllm
from cot_icl.demos import shuffle_demos
from cot_icl.detectiveqa.data import INSTRUCT_PROMPT, load_demo_examples, load_test_examples
from cot_icl.grading.detectiveqa import extract_answer
from cot_icl.paths import infer_res_dir
from cot_icl.prompts.demo_modes import DemoMode
from cot_icl.runners.common import apply_user_chat, write_json_record
from vllm import SamplingParams


def build_demo_block(demos: list[str], n_demos: int, mode: str, seed: int | None) -> tuple[str, int | None]:
    demos = demos[:n_demos]
    mode = DemoMode(mode)
    if mode == DemoMode.SHUFFLE:
        shuffled, used_seed = shuffle_demos(list(demos), seed)
        return "\n\n".join(shuffled) + "\n\n", used_seed
    return "\n\n".join(demos) + "\n\n", seed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DetectiveQA many-shot CoT-ICL inference")
    parser.add_argument("-r", "--round", type=float, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-s", "--seed", type=int, default=71)
    parser.add_argument(
        "-d",
        "--demo_mode",
        type=str,
        default="ori",
        choices=[DemoMode.ORI.value, DemoMode.SHUFFLE.value],
    )
    args = parser.parse_args(argv)

    n_demos = int(2 ** args.round)
    demos = load_demo_examples()
    demo_prompt, seed = build_demo_block(demos, n_demos, args.demo_mode, args.seed)
    test = load_test_examples()

    llm, tokenizer, spec = load_vllm(args.model, max_model_len=101_000)

    all_prompts = []
    all_tests = []
    for tq, ta, tr in test:
        prompt = INSTRUCT_PROMPT + demo_prompt + "\n\n" + tq
        all_prompts.append(apply_user_chat(tokenizer, prompt, enable_thinking=spec.enable_thinking))
        all_tests.append((tq, ta, tr, prompt))

    print(f"[batch inference] total prompts: {len(all_prompts)}")
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
        print(corr / len(record["ta"]))

    record["score"] = corr / len(test) if test else 0.0
    if args.demo_mode == DemoMode.SHUFFLE.value:
        out_path = infer_res_dir("shuffle") / f"{args.model}_detectiveqa{args.round}_seed{seed}.jsonl"
    else:
        out_path = infer_res_dir() / f"{args.model}_detectiveqa_{args.round}.jsonl"
    write_json_record(out_path, record)
