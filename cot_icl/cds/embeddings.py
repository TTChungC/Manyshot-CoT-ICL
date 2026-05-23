"""Embedding backends for CDS."""

from __future__ import annotations

from typing import List

import torch

from cot_icl.paths import embedding_model_path

_model_cache = {}


def get_embeddings(texts: List[str], embed_model: str, **kwargs) -> torch.Tensor:
    if embed_model == "vllm":
        from openai import OpenAI

        client = OpenAI(api_key="EMPTY", base_url=kwargs["base_url"])
        res = client.embeddings.create(model=kwargs["model_name"], input=texts)
        return torch.tensor([res.data[i].embedding for i in range(len(texts))])

    if embed_model == "bge-m3":
        if "bge-m3" not in _model_cache:
            from sentence_transformers import SentenceTransformer

            model_path = embedding_model_path("bge-m3")
            _model_cache["bge-m3"] = SentenceTransformer(
                model_path,
                trust_remote_code=True,
                device=kwargs.get("device", "cuda"),
            )
        model = _model_cache["bge-m3"]
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return torch.tensor(embs)

    raise ValueError(
        f"Unknown embed model: {embed_model}. Choose from: vllm, bge-m3"
    )
