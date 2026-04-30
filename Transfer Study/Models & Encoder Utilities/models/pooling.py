import torch


def masked_mean_pool(sequence_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    mask = attention_mask.float().unsqueeze(-1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (sequence_embeddings * mask).sum(dim=1) / denom


def last_valid_token_pool(sequence_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    last_idx = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    batch_idx = torch.arange(sequence_embeddings.size(0), device=sequence_embeddings.device)
    return sequence_embeddings[batch_idx, last_idx]


def cls_pool(sequence_embeddings: torch.Tensor) -> torch.Tensor:
    return sequence_embeddings[:, 0]


def get_pooling_fn(name: str):
    name = name.lower()
    if name in {"mean", "masked_mean", "avg"}:
        return masked_mean_pool
    if name in {"last", "last_valid", "sentence_last"}:
        return last_valid_token_pool
    if name in {"cls", "first"}:
        return cls_pool
    raise ValueError(f"Unsupported pooling strategy: {name}")
