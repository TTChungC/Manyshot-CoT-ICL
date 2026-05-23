# Data layout

Place datasets under this directory (or set `COT_ICL_DATA` to another root).

```
data/
├── gsm8k/                          # HuggingFace datasets saved to disk
├── hendrycks_math/
│   ├── algebra/
│   ├── geometry/
│   ├── number_theory/
│   └── ...                         # one folder per MATH subtask
├── DetectiveQA/
│   └── anno_data_en/
│       ├── AIsup_anno/             # demonstration pool
│       └── human_anno/             # test set
├── gsm8k_r/                        # self-generated CoT for reinforce experiments
├── math/                           # self-generated CoT for MATH reinforce
└── infer_res/                      # inference outputs (created automatically)
```

## Preparation

- **GSM8K**: `datasets.load_dataset("gsm8k", "main")` then `save_to_disk("data/gsm8k")`.
- **MATH**: follow [hendrycks/math](https://github.com/hendrycks/math) and save each subtask under `hendrycks_math/<subtask>/`.
- **DetectiveQA**: obtain annotations from the [DetectiveQA](https://github.com/James-Yip/DetectiveQA) release and mirror the folder structure above.
