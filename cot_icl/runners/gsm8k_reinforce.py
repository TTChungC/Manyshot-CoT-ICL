"""GSM8K inference with self-generated demonstrations."""

from __future__ import annotations

import argparse

import datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from cot_icl.config import get_model_spec
from cot_icl.grading.gsm8k import clean_answer, is_correct_gsm8k
from cot_icl.paths import gsm8k_dir, gsm8k_reinforce_json, infer_res_dir
from cot_icl.prompts.gsm8k import build_gsm8k_query_prompt, format_gsm8k_pairs
from cot_icl.runners.common import apply_user_chat, load_reinforce_json, write_json_record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GSM8K reinforce inference")
    parser.add_argument("-r", "--round", type=float, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-j", "--json", type=str, default=None, help="source model for demo JSON")
    parser.add_argument(
        "--pool-variant",
        type=str,
        default="default",
        choices=["default", "wrongans", "firstans"],
    )
    parser.add_argument("--token-suffix", type=str, default="")
    args = parser.parse_args(argv)

    source = args.json or args.model
    spec = get_model_spec(args.model)
    max_model_len = 30_000
    gpu_mem = 0.95 if args.model == "qwen14" else 0.4
    llm = LLM(
        spec.path,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        max_seq_len_to_capture=max_model_len,
        distributed_executor_backend="mp",
        dtype="float16",
    )
    tokenizer = AutoTokenizer.from_pretrained(spec.path, trust_remote_code=True)

    json_path = gsm8k_reinforce_json(source, variant=args.pool_variant)
    if args.pool_variant == "firstans" and args.token_suffix:
        json_path = gsm8k_reinforce_json(source, variant="firstans").with_name(
            f"firstans_{source}_gsm8k_all{args.token_suffix}.jsonl"
        )
    ridata = load_reinforce_json(json_path)

    n_demos = int(2 ** args.round)
    demo_prompt = ""
    for i in range(n_demos):
        demo_prompt += (
            "Q: "
            + ridata["question"][i]
            + "\nA: "
            + ridata["complete_response"][i].strip()
            + "\n\n"
        )

    dataset = datasets.load_from_disk(str(gsm8k_dir()))
    eva_data = format_gsm8k_pairs(dataset["test"])
    record = {
        "complete_response": [],
        "extracted_response": [],
        "label": [],
        "token_size": 0,
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
    suffix = args.token_suffix if args.pool_variant == "firstans" else ""
    out_path = infer_res_dir() / f"{args.model}_gsm8k_reinforce_{args.round}{suffix}.jsonl"
    write_json_record(out_path, record)
