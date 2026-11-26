"""Standalone publication shim for the names that ../../../base.py imports.

This is not the RESOLVE case adapter. It is written for this repository from the
published input contract in reference/flashmoe/input-domain.txt (profiles, the
RESOLVE_MEGA_* environment selection, and the input checks), so base.py runs
unmodified. RESOLVE's input generator (test categories V0-V7) is not published;
make_inputs here is this repository's own spread-expert generator and accepts but
ignores the category.
"""

from __future__ import annotations

import hashlib
import os
from typing import NamedTuple

CASE_ID = "flashmoe_capacity_drop"
TOKEN_BOUND, WEIGHT_BOUND = 32, 4


class Profile(NamedTuple):
    tokens: int
    hidden: int
    ffn: int
    experts: int
    capacity: int
    top_k: int = 1
    tile_m: int = 128


SMOKE = Profile(tokens=256, hidden=512, ffn=1024, experts=8, capacity=32)
FULL = Profile(tokens=1024, hidden=5120, ffn=8192, experts=16, capacity=64)


def profile(smoke: bool = False) -> Profile:
    return SMOKE if smoke else FULL


def environment_profile() -> Profile:
    if os.environ.get("RESOLVE_MEGA_CASE_ID", CASE_ID) != CASE_ID:
        raise ValueError("this shim only serves the FlashMoE capacity-drop case")
    choice = os.environ.get("RESOLVE_MEGA_SMOKE", "0")
    if choice not in ("0", "1"):
        raise ValueError("RESOLVE_MEGA_SMOKE must be 0 or 1")
    return profile(choice == "1")


def validate_inputs(tokens, router_weights, p: Profile) -> None:
    import torch

    checks = (("tokens", tokens, (p.tokens, p.hidden), TOKEN_BOUND),
              ("router_weights", router_weights, (p.hidden, p.experts), WEIGHT_BOUND))
    for label, tensor, shape, bound in checks:
        if not torch.is_tensor(tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"{label}: expected a tensor of shape {shape}")
        if tensor.dtype is not torch.bfloat16 or not tensor.is_contiguous():
            raise ValueError(f"{label}: expected contiguous bfloat16")
        if not torch.isfinite(tensor).all() or tensor.abs().max() > bound:
            raise ValueError(f"{label}: values must be finite with magnitude at most {bound}")
    if tokens.device != router_weights.device:
        raise ValueError("tokens and router_weights must share a device")


def make_inputs(case_id: str, category: str = "", seed: int = 0, smoke: bool = False) -> tuple:
    """Dyadic BF16 tokens/weights whose FP32 dot products are exact; experts vary per token."""
    import torch

    if case_id != CASE_ID:
        raise ValueError("this shim only serves the FlashMoE capacity-drop case")
    p = profile(smoke)
    digest = hashlib.sha256(f"flashmoe-check|{seed}".encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
    x = torch.randint(-8, 9, (p.tokens, p.hidden), generator=generator).float() / 8
    w = torch.randint(-8, 9, (p.hidden, p.experts), generator=generator).float() / 64
    return tuple(tensor.to(torch.bfloat16).contiguous() for tensor in (x, w))
