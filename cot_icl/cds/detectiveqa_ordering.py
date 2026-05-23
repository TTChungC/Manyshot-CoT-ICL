"""CDS ordering for DetectiveQA."""

from __future__ import annotations

import argparse
import json

from cot_icl.cds.detectiveqa_data import get_detectiveqa_embeddings
from cot_icl.cds.tsp import get_optimal_order
from cot_icl.paths import infer_res_dir


def main():
    parser = argparse.ArgumentParser(description="CDS for DetectiveQA")
    parser.add_argument("--r", type=int, required=True)
    parser.add_argument("--strategy", type=str, default="cds", choices=["cds", "high_curvature"])
    parser.add_argument("--embed-model", type=str, default="bge-m3", choices=["vllm", "bge-m3"])
    parser.add_argument("--n-starts", type=int, default=10)
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model-name", type=str, default="Qwen3-Embedding-4B")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    kwargs = {}
    if args.embed_model == "vllm":
        kwargs["base_url"] = args.base_url
        kwargs["model_name"] = args.model_name

    E = get_detectiveqa_embeddings(args.r, args.embed_model, **kwargs)
    best_order, best_score = get_optimal_order(
        E.numpy(),
        strategy=args.strategy,
        n_starts=args.n_starts,
    )
    print(f"\nBest order ({len(best_order)} items):")
    print(best_order)
    print(f"\nScore: {best_score:.4f}")

    out_path = args.output or (
        infer_res_dir("cds_orders")
        / f"detectiveqa_r{args.r}_{args.strategy}_{args.embed_model}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump({"order": best_order, "score": best_score, "args": vars(args)}, fp, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
