from __future__ import annotations

import torch


def align_modalities(embedding_1: torch.Tensor, embedding_2: torch.Tensor) -> float:
    normalized_1 = embedding_1 / embedding_1.norm().clamp_min(1e-8)
    normalized_2 = embedding_2 / embedding_2.norm().clamp_min(1e-8)
    return float(torch.dot(normalized_1, normalized_2).item())


def combine_embeddings(embeddings_list: list[torch.Tensor]) -> torch.Tensor:
    if not embeddings_list:
        raise ValueError("embeddings_list cannot be empty")
    stacked_embeddings = torch.stack(embeddings_list)
    shared_embedding = stacked_embeddings.mean(dim=0)
    shared_embedding = shared_embedding / shared_embedding.norm().clamp_min(1e-8)
    return shared_embedding
