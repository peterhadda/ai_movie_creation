from __future__ import annotations

import hashlib

import torch


def encode_text(text_input: str, embedding_dim: int = 64) -> torch.Tensor:
    digest = hashlib.sha256(text_input.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    generator = torch.Generator().manual_seed(seed)
    text_embedding = torch.randn(embedding_dim, generator=generator)
    text_embedding = text_embedding / text_embedding.norm().clamp_min(1e-8)
    return text_embedding
