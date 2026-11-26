"""Independent deterministic capacity-pruned router, without FlashMoE imports."""

from __future__ import annotations

import torch
from torch import nn

from E8_mega_kernel_h100.cases.flashmoe.adapter import (
    CASE_ID, environment_profile, make_inputs, profile, validate_inputs,
)


def reference_route(tokens: torch.Tensor, router_weights: torch.Tensor, capacity: int) -> dict:
    """Unfused FP32 scores, softmax/top-1, and token-index prefix capacity."""
    x = tokens.detach().to(device="cpu", dtype=torch.float32)
    weights = router_weights.detach().to(device="cpu", dtype=torch.float32)
    scores = torch.mm(x, weights)
    probabilities = torch.softmax(scores, dim=1)
    selected = torch.argsort(probabilities, dim=1, descending=True, stable=True)[:, :1]
    selected_probability = torch.gather(probabilities, 1, selected)
    combine_probability = selected_probability / selected_probability.sum(dim=1, keepdim=True)

    s, e = scores.shape
    assignments = (selected == torch.arange(e)[None, :]).to(torch.int32)
    prefix_rank = assignments.cumsum(dim=0)
    survival = assignments * (prefix_rank <= capacity).to(torch.int32)
    demand = assignments.sum(dim=0, dtype=torch.int32)
    token_indices = torch.full((e, capacity), -1, dtype=torch.int64)
    kept_probabilities = torch.zeros((e, capacity), dtype=torch.float32)
    for expert in range(e):
        kept = torch.nonzero(survival[:, expert], as_tuple=False).flatten()
        token_indices[expert, :kept.numel()] = kept
        kept_probabilities[expert, :kept.numel()] = combine_probability[kept, 0]
    return {
        "expert_demand": demand.to(tokens.device),
        "token_indices": token_indices.to(tokens.device),
        "probabilities": kept_probabilities.to(tokens.device),
        "survival": survival.to(tokens.device),
    }


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tokens: torch.Tensor, router_weights: torch.Tensor) -> dict:
        p = environment_profile()
        validate_inputs(tokens, router_weights, p)
        return reference_route(tokens, router_weights, p.capacity)


def get_init_inputs() -> list:
    return []


def get_inputs() -> list[torch.Tensor]:
    p = environment_profile()
    return list(make_inputs(CASE_ID, "V0", seed=0, smoke=p == profile(smoke=True)))
