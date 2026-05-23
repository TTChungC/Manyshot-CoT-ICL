"""GSM8K many-shot CoT-ICL inference (gold demonstrations)."""

from __future__ import annotations

import argparse
import json

import datasets
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from cot_icl.config import get_model_spec
from cot_icl.demos import shuffle_demos
from cot_icl.grading.gsm8k import clean_answer, is_correct_gsm8k
from cot_icl.paths import gsm8k_dir, infer_res_dir
from cot_icl.prompts.gsm8k import (
    build_gsm8k_demo_prompt,
    build_gsm8k_query_prompt,
    format_gsm8k_pairs,
)
from cot_icl.runners.common import apply_user_chat, write_json_record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GSM8K many-shot CoT-ICL inference")
    parser.add_argument("-r", "--round", type=float, required=True, help="log2(#demonstrations)")
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-s", "--seed", type=int, default=None)
    args = parser.parse_args(argv)

    spec = get_model_spec(args.model)
    max_model_len = min(spec.max_model_len, 51_000)
    llm = LLM(
        spec.path,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.9,
        max_seq_len_to_capture=max_model_len,
    )
    tokenizer = AutoTokenizer.from_pretrained(spec.path, trust_remote_code=True)

    dataset = datasets.load_from_disk(str(gsm8k_dir()))
    demo_pairs = format_gsm8k_pairs(dataset["train"])[: int(2 ** args.round)]
    demo_pairs, seed = shuffle_demos(demo_pairs, args.seed)
    demo_prompt = build_gsm8k_demo_prompt(demo_pairs)
    eva_data = format_gsm8k_pairs(dataset["test"])

    record = {
        "complete_response": [],
        "extracted_response": [],
        "label": [],
        "token_size": 0,
        "seed": seed,
        "score": None,
    }
    correct = 0
    total = 0

    for question, answer in eva_data:
        cur_prompt = build_gsm8k_query_prompt(demo_prompt, question)
        chat_prompt = apply_user_chat(tokenizer, cur_prompt)
        output = llm.generate(
            chat_prompt,
            sampling_params=SamplingParams(
                temperature=0.0,
                top_p=1,
                max_tokens=4096,
                stop_token_ids=[tokenizer.eos_token_id],
            ),
        )
        response = output[0].outputs[0].text
        record["complete_response"].append(response)
        response = response.lower().replace("`", "").strip()
        if "</s>" in response:
            response = response[: response.index("</s>")].strip()
        response = clean_answer(response).strip()
        record["extracted_response"].append(response)
        label = answer.lower()
        record["label"].append(label)
        if record["token_size"] == 0:
            record["token_size"] = len(tokenizer(cur_prompt)[0])
        if is_correct_gsm8k(response, label):
            correct += 1
        total += 1
        print("accuracy:", correct / total)

    record["score"] = correct / total if total else 0.0
    out_path = infer_res_dir() / f"{args.model}_gsm8k_CoT_{args.round}_round_seed{seed}.jsonl"
    write_json_record(out_path, record)
