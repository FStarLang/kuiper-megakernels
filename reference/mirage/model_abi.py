"""Checkpoint ABI and observation contract, independent of either implementation."""

import hashlib
import importlib.metadata
import json
from math import prod
from pathlib import Path


CASE_ID = "mirage_repaired_explicit_weights"
SOURCE_REVISION = "c9d83035d2e79cf85ae6a1e5dcbb507c4f6bbe61"
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
CHECKPOINT_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
REFERENCE_SHA256 = "4b95c371fd26d40c69083dab36ac1eafd8cf82b415a0bb827275097c5ad2305b"
LAYERS, HIDDEN, VOCAB, PROMPT, SEQUENCE = 28, 1024, 151936, 7, 8


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def weight_shapes():
    result = {"model.embed_tokens.weight": (VOCAB, HIDDEN), "model.norm.weight": (HIDDEN,)}
    for layer in range(LAYERS):
        for name, shape in (
            ("input_layernorm.weight", (HIDDEN,)),
            ("post_attention_layernorm.weight", (HIDDEN,)),
            ("self_attn.q_proj.weight", (2048, HIDDEN)),
            ("self_attn.k_proj.weight", (1024, HIDDEN)),
            ("self_attn.v_proj.weight", (1024, HIDDEN)),
            ("self_attn.o_proj.weight", (HIDDEN, 2048)),
            ("self_attn.q_norm.weight", (128,)),
            ("self_attn.k_norm.weight", (128,)),
            ("mlp.gate_proj.weight", (3072, HIDDEN)),
            ("mlp.up_proj.weight", (3072, HIDDEN)),
            ("mlp.down_proj.weight", (HIDDEN, 3072)),
        ):
            result[f"model.layers.{layer}.{name}"] = shape
    return result


WEIGHT_LAYOUT, WEIGHT_ELEMENTS = [], 0
for _name, _shape in weight_shapes().items():
    _count = prod(_shape)
    WEIGHT_LAYOUT.append({
        "name": _name, "shape": list(_shape), "offset": WEIGHT_ELEMENTS, "elements": _count,
    })
    WEIGHT_ELEMENTS += _count


def pack_weights(weights):
    import torch

    expected = set(weight_shapes())
    if set(weights) not in (expected, expected | {"lm_head.weight"}):
        raise ValueError("Checkpoint parameter inventory differs from the full 28-layer model")
    for name, shape in weight_shapes().items():
        if tuple(weights[name].shape) != shape or weights[name].dtype != torch.bfloat16:
            raise ValueError(f"Checkpoint shape/dtype mismatch: {name}")
    if "lm_head.weight" in weights:
        head, embedding = weights["lm_head.weight"], weights["model.embed_tokens.weight"]
        if (head.dtype != embedding.dtype or head.shape != embedding.shape
                or not torch.equal(head.view(torch.uint8), embedding.view(torch.uint8))):
            raise ValueError("The checkpoint LM head is not bitwise tied to its embedding")
    return torch.cat([weights[item["name"]].reshape(-1) for item in WEIGHT_LAYOUT])


def unpack_weights(packed_weights):
    import torch

    if (packed_weights.dtype != torch.bfloat16 or packed_weights.ndim != 1
            or packed_weights.numel() != WEIGHT_ELEMENTS or not packed_weights.is_contiguous()):
        raise ValueError(f"Expected contiguous BF16 packed_weights[{WEIGHT_ELEMENTS}]")
    result = {
        item["name"]: packed_weights.narrow(0, item["offset"], item["elements"]).view(item["shape"])
        for item in WEIGHT_LAYOUT
    }
    result["lm_head.weight"] = result["model.embed_tokens.weight"]
    return result


def validate_inputs(packed_weights, prompt_tokens, decode_steps):
    import torch

    weights = unpack_weights(packed_weights)
    if (prompt_tokens.dtype != torch.int64 or tuple(prompt_tokens.shape) != (PROMPT,)
            or not prompt_tokens.is_contiguous() or type(decode_steps) is not int or decode_steps != 1):
        raise ValueError("Expected contiguous int64 prompt_tokens[7] and decode_steps=1")
    if ((prompt_tokens < 0) | (prompt_tokens >= VOCAB)).any().item():
        raise ValueError("Prompt contains an invalid vocabulary ID")
    if prompt_tokens.device != packed_weights.device or prompt_tokens.device != torch.device("cuda:0"):
        raise ValueError("Weights and prompt must reside on the selected logical cuda:0")
    return weights


def qwen_module():
    if importlib.metadata.version("transformers") != "4.57.1":
        raise RuntimeError("This workload requires the pinned Transformers 4.57.1")
    from transformers.models.qwen3 import modeling_qwen3

    if sha256(modeling_qwen3.__file__) != REFERENCE_SHA256:
        raise RuntimeError("The installed Qwen3 implementation differs from the frozen source")
    return modeling_qwen3


def model_config():
    document = json.loads((Path(__file__).parent / "model_config.json").read_text())
    if document["num_hidden_layers"] != LAYERS or document["hidden_size"] != HIDDEN:
        raise ValueError("Truncated or reshaped Qwen3 models are not supported")
    config = qwen_module().Qwen3Config(**document)
    config._attn_implementation = "eager"
    return config
