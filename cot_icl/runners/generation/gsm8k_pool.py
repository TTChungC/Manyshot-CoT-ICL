"""Generate GSM8K self-generated demonstration pools."""

from __future__ import annotations

import argparse

import datasets
from vllm import SamplingParams

from cot_icl.config import load_vllm
from cot_icl.grading.gsm8k import INVALID_ANS, clean_answer
from cot_icl.paths import gsm8k_dir, gsm8k_reinforce_dir
from cot_icl.prompts.gsm8k import format_gsm8k_pairs
from cot_icl.runners.common import apply_user_chat, write_json_record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GSM8K first-pass demo pool generation")
    parser.add_argument("-r", "--round", type=int, default=9, help="kept for CLI compat")
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-t", "--token", type=str, default="")
    args = parser.parse_args(argv)

    dataset = datasets.load_from_disk(str(gsm8k_dir()))
    demo_data = format_gsm8k_pairs(dataset["train"])
    llm, tokenizer, spec = load_vllm(args.model, max_model_len=4096, distributed_executor_backend="mp")

    max_tk = int(args.token[1:-1]) if args.token.startswith("_") and args.token.endswith("k") else 4096
    record = {
        "idx": [],
        "question": [],
        "complete_response": [],
        "extracted_response": [],
        "label": [],
        "token_size": 0,
        "seed": None,
        "score": None,
    }

    for edx, (question, answer) in enumerate(demo_data[:1000]):
        label = answer.lower()
        cur_prompt = (
            "In the end of the response, add a summary `The answer is [answer].`\n\n"
            f"Q: {question}\nA:"
        )
        chat_prompt = apply_user_chat(tokenizer, cur_prompt, enable_thinking=spec.enable_thinking)
        prev, prevlen, prev_ok = None, 10_000, False

        for _ in range(10):
            output = llm.generate(
                chat_prompt,
                sampling_params=SamplingParams(
                    temperature=1.0,
                    top_p=1,
                    max_tokens=4096,
                    stop_token_ids=[tokenizer.eos_token_id],
                ),
            )
            response = output[0].outputs[0].text.split("</think>")[-1]
            refined = response.lower().replace("`", "").strip()
            if "</s>" in refined:
                refined = refined[: refined.index("</s>")].strip()
            refined = clean_answer(refined).strip()
            curlen = len(tokenizer(response).input_ids)

            if curlen < (max_tk + 40) and refined != INVALID_ANS:
                record["idx"].append(edx)
                record["label"].append(label)
                record["question"].append(question)
                record["complete_response"].append(response)
                record["extracted_response"].append(refined)
                break

            if curlen < prevlen and (
                (refined == INVALID_ANS and not prev_ok) or refined != INVALID_ANS
            ):
                prev = response
                prevlen = curlen
                if refined != INVALID_ANS:
                    prev_ok = True

    out_path = gsm8k_reinforce_dir() / f"firstans_{args.model}_gsm8k_all{args.token}.jsonl"
    write_json_record(out_path, record)
