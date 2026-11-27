# Exact trusted-reference declarations; this is a source excerpt, not a runnable module.
# The base role (HF reference) entry point and its decode driver, from the RESOLVE E13 harness.
# The model math itself is modeling_qwen3_5.py (pinned HF source, excerpted in this directory).

# ==== experiments/E13_luce_reduction/base.py (complete file) ====
# Original source SHA-256: 9cbc9d05acddc0af835e890d5c29785e5a3f328d7cff6183c20db74249c12c1e
"""Independent HF reference with exactly the same explicit checkpoint input."""

import hashlib
import json
import os
from pathlib import Path
import sys
import types

import torch

if not __package__:
    _root = Path(__file__).resolve().parent
    __package__ = "_e13_runtime_" + hashlib.sha256(str(_root).encode()).hexdigest()[:16]
    if __package__ not in sys.modules:
        _package = types.ModuleType(__package__)
        _package.__path__ = [str(_root)]
        sys.modules[__package__] = _package

from .common import require_cuda0, require_version, sha256, validate_inputs
from .layout import unpack_weights
from .reference import (
    REFERENCE_SOURCE_SHA256, TRANSFORMERS_VERSION,
    construct_reference_model, decode_reference,
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        require_version("transformers", TRANSFORMERS_VERSION)
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

        if sha256(Path(qwen.__file__)) != REFERENCE_SOURCE_SHA256:
            raise RuntimeError("The installed HF Qwen3.5 reference differs from the pinned source")
        document = json.loads((Path(__file__).parent / "model_config.json").read_text())
        self.config = qwen.Qwen3_5TextConfig(**document["text_config"])
        self.config._attn_implementation = "eager"
        self.config.dtype = torch.bfloat16
        self.qwen = qwen

    def forward(self, packed_weights, prompt_tokens, decode_steps: int):
        smoke = os.environ["RESOLVE_MEGA_SMOKE"] == "1"
        prompt = validate_inputs(
            prompt_tokens, decode_steps, smoke=False,
            max_prompt_len=8 if smoke else 520,
            decode_step_choices=(1, 5) if smoke else (1, 5, 30, 128),
        )
        require_cuda0(prompt_tokens)
        if packed_weights.device != prompt_tokens.device:
            raise ValueError("Weights and tokens must be on the same selected CUDA device")
        with torch.inference_mode():
            model = construct_reference_model(
                self.qwen, self.config, unpack_weights(packed_weights),
            ).to(prompt_tokens.device)
            cache = self.qwen.Qwen3_5DynamicCache(config=self.config)
            return decode_reference(model, cache, prompt, decode_steps)

# ==== reference.py (E13 base dependency) ====
# Original source SHA-256: 61e71ad0363b08644d7f27ea2de6a7c689601cb54e6d5228856ed777787e5ca8

# ---- constants, eager_construction, construct_reference_model, decode_reference, lines 24-116 (verbatim) ----
TRANSFORMERS_VERSION = "5.3.0"
REFERENCE_SOURCE_SHA256 = "0e7410f5251e5c324a5628c5340ee3551f7a6a505ce398569fcc3b5f3ac07f7a"
REFERENCE_SOURCE_URL = (
    "https://github.com/huggingface/transformers/blob/v5.3.0/"
    "src/transformers/models/qwen3_5/modeling_qwen3_5.py"
)


@contextlib.contextmanager
def eager_construction(module):
    overrides = {
        "FusedRMSNormGated": None,
        "causal_conv1d_fn": None,
        "causal_conv1d_update": module.torch_causal_conv1d_update,
        # Each call has sequence length one, including the first prompt token.
        "chunk_gated_delta_rule": module.torch_recurrent_gated_delta_rule,
        "fused_recurrent_gated_delta_rule": module.torch_recurrent_gated_delta_rule,
    }
    previous = {name: getattr(module, name) for name in overrides}
    for name, value in overrides.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(module, name, value)


def construct_reference_model(qwen, config, weights):
    import torch

    with eager_construction(qwen), torch.device("meta"):
        model = qwen.Qwen3_5ForCausalLM(config)
    model.load_state_dict(weights, strict=True, assign=True)
    model.tie_weights()
    # Nonpersistent RoPE buffers are absent from the checkpoint, so assign=True
    # cannot materialize them. Use HF's own constructor, not uninitialized to_empty.
    model.model.rotary_emb = qwen.Qwen3_5TextRotaryEmbedding(config=config, device="cpu")
    pending = [
        name for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.is_meta
    ]
    if pending:
        raise CaseUnavailable(
            f"Reference still has unmaterialized meta tensors: {pending}",
            kind="reference_unavailable",
        )
    return model.eval().requires_grad_(False)


def decode_reference(model, cache, prompt: list[int], decode_steps: int, *, device="cuda:0"):
    import torch

    captured = None
    position = 0

    def capture_hidden(_module, arguments):
        nonlocal captured
        captured = arguments[0][0, -1, :].detach().clone()

    def step(token_id):
        nonlocal position, cache, captured
        captured = None
        output = model(
            input_ids=torch.tensor([[token_id]], dtype=torch.int64, device=device),
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([position], dtype=torch.int64, device=device),
            logits_to_keep=1,
            return_dict=True,
        )
        position += 1
        cache = output.past_key_values
        if captured is None:
            raise CaseUnavailable(
                "HF final RMSNorm pre-hook did not observe the layer-23 residual stream",
                kind="reference_unavailable",
            )
        return int(output.logits[0, -1].argmax().item()), captured

    hook = model.model.norm.register_forward_pre_hook(capture_hidden)
    try:
        for token in prompt[:-1]:
            step(token)
        token = prompt[-1]
        tokens, hidden_states = [], []
        for _ in range(decode_steps):
            token, hidden = step(token)
            tokens.append(token)
            hidden_states.append(hidden)
        return torch.tensor(tokens, dtype=torch.int64, device=device), torch.stack(hidden_states)
    finally:
        hook.remove()

# ==== common.py (E13 base dependency) ====
# Original source SHA-256: e54a7c8ca442756f6dace9eed3a3f5d6e804df0b8a9bfd9de851badf12202b6c

# ---- model constants, lines 25-51 (verbatim) ----
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_FILES = {
    "config.json": {
        "size": 2907,
        "sha256": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    },
    "model.safetensors.index.json": {
        "size": 50900,
        "sha256": "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
    },
    "model.safetensors-00001-of-00001.safetensors": {
        "size": 1746942600,
        "sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    },
}
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3584
VOCAB_SIZE = 248320
MAX_SEQ_LEN = 2048
MAX_PROMPT_LEN = 8
LAYER_TYPES = ("linear_attention",) * 3 + ("full_attention",)
LAYER_TYPES *= 6
HISTORICAL_PROMPT = (1234, 5678, 9012)
DECODE_STEPS = (1, 5, 30)
TORCH_VERSION = "2.14.0+cu130"
SAFETENSORS_VERSION = "0.7.0"

# ---- validate_inputs, lines 137-167 (verbatim) ----
def validate_inputs(
    prompt_tokens: Any, decode_steps: Any, *, smoke: bool,
    max_prompt_len: int = MAX_PROMPT_LEN,
    decode_step_choices: tuple[int, ...] = DECODE_STEPS,
) -> list[int]:
    import torch

    if (
        not isinstance(prompt_tokens, torch.Tensor)
        or prompt_tokens.dtype != torch.int64
        or prompt_tokens.ndim != 1
        or not prompt_tokens.is_contiguous()
        or not 1 <= prompt_tokens.numel() <= max_prompt_len
    ):
        raise CaseUnavailable(
            f"prompt_tokens must be a contiguous int64 tensor of shape [P], 1 <= P <= {max_prompt_len}",
            kind="invalid_workload",
        )
    allowed = (1,) if smoke else decode_step_choices
    if type(decode_steps) is not int or decode_steps not in allowed:
        raise CaseUnavailable(
            f"decode_steps must be a Python int in {allowed}", kind="invalid_workload"
        )
    tokens = prompt_tokens.detach().cpu().tolist()
    if any(token < 0 or token >= VOCAB_SIZE for token in tokens):
        raise CaseUnavailable(
            f"Token IDs must satisfy 0 <= token < {VOCAB_SIZE}", kind="invalid_workload"
        )
    if len(tokens) + decode_steps - 1 > MAX_SEQ_LEN:
        raise CaseUnavailable("Decode exceeds the 2048-position cache", kind="invalid_workload")
    return tokens
