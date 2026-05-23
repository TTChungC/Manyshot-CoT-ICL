# Manyshot CoT-ICL

[![arXiv](https://img.shields.io/badge/arXiv-2605.13511-b31b1b.svg?style=plastic)](https://arxiv.org/pdf/2605.13511)


This is the official release accompanying our **Findings of ICML 2026** paper:

> **Many-Shot CoT-ICL: Making In-Context Learning Truly Learn**  
> [Tsz Ting Chung](https://ttchungc.github.io/), [Lemao Liu](https://lemaoliu.github.io/), [Mo Yu](https://sites.google.com/site/moyunlp/), [Dit-Yan Yeung](https://sites.google.com/view/dyyeung)

We reframe many-shot chain-of-thought ICL as *in-context test-time learning*, study scaling / retrieval / ordering on reasoning vs. non-reasoning setups, and propose **Curvilinear Demonstration Selection (CDS)** for demonstration ordering.

## Repository layout

Clone root (`Manyshot-CoT-ICL/`):

```
├── cot_icl/                 # Library: paths, models, grading, CDS, runners
│   ├── cds/                 # CDS ordering (TSP heuristic)
│   ├── detectiveqa/         # DetectiveQA data loading
│   ├── grading/             # Answer extraction & exact match
│   ├── prompts/             # Demo prompt builders
│   ├── runners/             # Inference & pool-generation logic
│   │   └── generation/      # First-pass / best-of-n demo pools
│   ├── config.py            # Model aliases → HuggingFace paths
│   └── paths.py             # Data / output directories
├── scripts/                 # CDS CLI entry points
├── data/                    # Datasets & outputs (not shipped; see data/README.md)
└── *_infer*.py              # Thin wrappers → cot_icl.runners (same CLI as before)
```

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `COT_ICL_DATA` | `./data` | Root for datasets and `infer_res/` |
| `COT_ICL_MODELS` | `../` (parent of repo root) | Directory containing LLM weights |
| `COT_ICL_MODEL_<ALIAS>` | — | Override path for a model alias (e.g. `COT_ICL_MODEL_QWEN3`) |
| `COT_ICL_EMBED_BGE_M3` | `BAAI/bge-m3` | Sentence-transformers path for CDS |
| `COT_ICL_API_BASE` | `http://21.0.198.55/{user}/v1` | Remote OpenAI-compatible API for `math_infer.py -u` |

For Qwen long-context models you may need:

```bash
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn  # if multiprocessing errors occur
```

## Data preparation

See [data/README.md](data/README.md). You need:

- **GSM8K** — many-shot CoT scaling on arithmetic reasoning  
- **MATH** (Hendrycks) — subtasks such as `geometry`, `number_theory`  
- **DetectiveQA** — narrative multiple-choice reasoning  

## Quick start

### 1. CDS: compute demonstration order

Ordering strategies:

| `--strategy` | Description |
|--------------|-------------|
| `cds` | Curvilinear Demonstration Selection — TSP path favoring **smooth** progression (maximize curvature score) |
| `high_curvature` | Ablation — same TSP setup but favors **high-curvature** transitions (minimize curvature score) |

**MATH** (requires `data/hendrycks_math/<subtask>/`):

CDS (default paper method):

```bash
python scripts/cds_order_math.py \
  --subtask geometry \
  --r 7 \
  --seed 71 \
  --strategy cds \
  --embed-model bge-m3
```

High-curvature ablation:

```bash
python scripts/cds_order_math.py \
  --subtask geometry \
  --r 7 \
  --seed 71 \
  --strategy high_curvature \
  --embed-model bge-m3
```

**DetectiveQA**:

CDS:

```bash
python scripts/cds_order_detectiveqa.py \
  --r 4 \
  --strategy cds \
  --embed-model bge-m3
```

High-curvature ablation:

```bash
python scripts/cds_order_detectiveqa.py \
  --r 4 \
  --strategy high_curvature \
  --embed-model bge-m3
```

**Embedding backends for CDS**

| `--embed-model` | How embeddings are computed | `--base-url` / `--model-name` |
|-----------------|----------------------------|-------------------------------|
| `bge-m3` (default) | Loaded locally via `sentence-transformers` (`COT_ICL_EMBED_BGE_M3` or `BAAI/bge-m3`) | Ignored |
| `vllm` | HTTP call to an OpenAI-compatible **embedding** server (e.g. vLLM) | **Required** — see below |

With `--embed-model vllm`, CDS does not load an embedder on your machine. It uses the OpenAI Python client against a remote server:

- **`--base-url`** — API root, e.g. `http://localhost:8000/v1` (must expose `/embeddings`).
- **`--model-name`** — model ID on that server, e.g. `Qwen3-Embedding-4B`.

Example:

```bash
python scripts/cds_order_math.py \
  --subtask geometry --r 7 --seed 71 --strategy cds \
  --embed-model vllm \
  --base-url http://localhost:8002/v1 \
  --model-name Qwen3-Embedding-4B
```

Orders are saved under `data/infer_res/cds_orders/*.json` unless `--output` is set.

### 2. Many-shot CoT inference

**GSM8K** — random demonstration order, CoT in context:

```bash
python gsm8k_infer.py -m qwen3 -r 7 -s 71
```

`-r` is \(\log_2 n\): `-r 7` → 128 demonstrations.

**MATH** — supports demonstration modes via `-d`:

| `-d` | Meaning |
|------|---------|
| `ori` | Fixed train order (no shuffle) |
| `shuffle` | Random permutation (seed `-s`) |

```bash
python math_infer.py -m qwen314 -t geometry -r 7 -s 71 -d shuffle
```

**DetectiveQA**:

```bash
python detectiveqa_infer.py -m qwen3 -r 7 -s 71 -d shuffle
```

### 3. Self-generated demonstrations (“reinforce”)

Some experiments replace gold CoT in demonstrations with model-generated chains from an earlier pass:

1. Run a **first-pass** script (e.g. `gsm8k_best-of-n.py`, `math_best-of-n-firstans.py`) to write JSON under `data/gsm8k_r/` or `data/math/`.
2. Run **reinforce** inference (`*_infer_reinforce*.py`) with `-j <generator_model>` pointing at that JSON.

Example (MATH):

```bash
python math_infer_reinforce.py -m qwen3 -t geometry -r 7 -j qwen3 -d ori
```

## Script reference

| Script | Role |
|--------|------|
| `scripts/cds_order_math.py` | CDS ordering for MATH subtasks |
| `scripts/cds_order_detectiveqa.py` | CDS ordering for DetectiveQA |
| `gsm8k_infer.py` | GSM8K evaluation (gold CoT demos) |
| `gsm8k_best-of-n.py` | GSM8K best-of-n / long CoT generation for reinforce pool |
| `gsm8k_infer_reinforce.py` | GSM8K eval with self-generated demos |
| `gsm8k_infer_reinforce_first.py` | GSM8K first-pass generation (reinforce pipeline) |
| `gsm8k_infer_reinforce_wr.py` | GSM8K reinforce variant (wrong-reasoning setup) |
| `math_infer.py` | MATH evaluation (gold CoT, batch vLLM) |
| `math_best-of-n-firstans.py` | MATH first-pass / demo pool generation |
| `math_infer_reinforce.py` | MATH eval with self-generated demos |
| `math_infer_reinforce_first.py` | MATH first-pass with extended demo modes |
| `math_infer_reinforce_wr.py` | MATH reinforce (wrong-reasoning variant) |
| `detectiveqa_infer.py` | DetectiveQA evaluation |
| `detectiveqa_infer_first.py` | DetectiveQA first-pass / reinforce pool |
| `detectiveqa_best_of_n_first.py` | DetectiveQA best-of-n first pass |

### Model aliases (`-m`)

Registered in `cot_icl/config.py`: `llama`, `llama70`, `qwen`, `qwen14`, `qwen3`, `qwen314`, `qwen332`, `qwq`, …  
Place weights under `COT_ICL_MODELS` or set per-model env overrides.

## Using the Python API

```python
from cot_icl.cds import get_optimal_order
from cot_icl.cds.math_data import get_math_embeddings

E = get_math_embeddings("geometry", r=7, seed=71, embed_model="bge-m3")
order, score = get_optimal_order(E.numpy(), strategy="cds")
```

```python
from cot_icl.prompts import DemoMode, build_math_demo_prompt

demo_block, seed = build_math_demo_prompt(
    problems, solutions, mode=DemoMode.SHUFFLE, seed=71
)
```

## Citation

```bibtex
@inproceedings{chung2026manyshot,
  title={Many-Shot {CoT-ICL}: Making In-Context Learning Truly Learn},
  author={Chung, Tsz Ting and Liu, Lemao and Yu, Mo and Yeung, Dit-Yan},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

## License

Code is released for research use. Dataset licenses follow GSM8K, MATH, and DetectiveQA respectively. Add a `LICENSE` file before public release if your institution requires one.

## Contact

Tsz Ting Chung — ttchungac@connect.ust.hk
