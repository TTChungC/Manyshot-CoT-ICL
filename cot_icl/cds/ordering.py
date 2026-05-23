"""
CDS ordering for MATH subtasks.

Preferred entry point:
    python scripts/cds_order_math.py --subtask geometry --r 7 --seed 71 --strategy cds
"""

from __future__ import annotations

import argparse
import json

from cot_icl.cds.math_data import get_math_embeddings
from cot_icl.cds.tsp import get_optimal_order
from cot_icl.paths import infer_res_dir


def main():
    parser = argparse.ArgumentParser(
        description="Curvilinear Demonstration Selection (CDS) for MATH"
    )
    parser.add_argument("--subtask", type=str, required=True)
    parser.add_argument("--r", type=int, required=True, help="log2(#demos), e.g. 7 -> 128")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["cds", "high_curvature"],
    )
    parser.add_argument("--embed-model", type=str, default="bge-m3", choices=["vllm", "bge-m3"])
    parser.add_argument("--n-starts", type=int, default=10)
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model-name", type=str, default="Qwen3-Embedding-4B")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    kwargs = {}
    if args.embed_model == "vllm":
        kwargs["base_url"] = args.base_url
        kwargs["model_name"] = args.model_name

    print(f"Computing embeddings with {args.embed_model}...")
    E = get_math_embeddings(args.subtask, args.r, args.seed, args.embed_model, **kwargs)
    print(f"Embedding shape: {E.shape}")

    best_order, best_score = get_optimal_order(
        E.numpy(),
        strategy=args.strategy,
        n_starts=args.n_starts,
    )

    print(f"\nBest order ({len(best_order)} items):")
    print(best_order)
    print(f"\nScore: {best_score:.4f}")

    if args.output:
        out_path = args.output
    else:
        out_path = (
            infer_res_dir("cds_orders")
            / f"{args.subtask}_r{args.r}_seed{args.seed}_{args.strategy}_{args.embed_model}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump({"order": best_order, "score": best_score, "args": vars(args)}, fp, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
