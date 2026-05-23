"""Generate DetectiveQA self-generated demonstration pools."""

from __future__ import annotations

import argparse

from vllm import SamplingParams

from cot_icl.config import load_vllm
from cot_icl.detectiveqa.data import load_pool_examples
from cot_icl.grading.detectiveqa import extract_answer
from cot_icl.paths import detectiveqa_reinforce_dir
from cot_icl.runners.common import apply_user_chat, write_json_record

POOL_INSTRUCT = (
    "Below is an instruction that describes a task.\n"
    " Select the correct option from A/B/C/D. Answer with 'The answer is {A/B/C/D}.' "
    "in the end of your response after the explanation. Keep your explanation in short.\n\n"
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DetectiveQA first-pass demo pool generation")
    parser.add_argument("-r", "--round", type=float, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-t", "--token", type=str, default="")
    parser.add_argument("-b", "--beginning", type=int, default=0)
    args = parser.parse_args(argv)

    demos = load_pool_examples()[: int(2 ** args.round)]
    llm, tokenizer, spec = load_vllm(args.model, max_model_len=40_000, distributed_executor_backend="mp")

    max_tk = int(args.token[1:-1]) if args.token.startswith("_") and args.token.endswith("k") else 4096
    record = {
        "tq": [],
        "ta": [],
        "tr": [],
        "complete_response": [],
        "refined_response": [],
        "token_size": None,
        "score": None,
    }
    corr = 0

    for tq, ta, tr in demos:
        prompt = POOL_INSTRUCT + "\n\n" + tq
        chat_prompt = apply_user_chat(tokenizer, prompt, enable_thinking=False)
        prev, prevlen, prev_ok = None, 10_000, False
        found = False

        for _ in range(13):
            output = llm.generate(
                chat_prompt,
                sampling_params=SamplingParams(
                    temperature=1.0,
                    max_tokens=4096,
                    stop_token_ids=[tokenizer.eos_token_id],
                ),
            )
            response = output[0].outputs[0].text.split("</think>")[-1]
            pred = extract_answer(response)
            curlen = len(tokenizer(response).input_ids)

            if curlen <= (max_tk + 20) and pred is not None:
                found = True
                record["tq"].append(tq)
                record["ta"].append(ta)
                record["tr"].append(tr)
                record["complete_response"].append(response)
                record["refined_response"].append(pred)
                if pred == ta:
                    corr += 1
                break

            if curlen < prevlen and ((pred is None and not prev_ok) or pred is not None):
                prev = response
                prevlen = curlen
                if pred is not None:
                    prev_ok = True

        if not found:
            record["tq"].append(tq)
            record["ta"].append(ta)
            record["tr"].append(tr)
            record["complete_response"].append(prev)
            pred = extract_answer(prev or "")
            record["refined_response"].append(pred)
            if pred == ta:
                corr += 1

    record["score"] = corr / len(demos) if demos else 0.0
    out_path = (
        detectiveqa_reinforce_dir()
        / f"first_{args.model}_detectiveqa_{args.round}_{args.token}.jsonl"
    )
    write_json_record(out_path, record)
