"""Independent pinned Hugging Face Qwen3 computation with explicit weights."""

import hashlib
from pathlib import Path
import sys
import types

import torch

if not __package__:
    _root = Path(__file__).resolve().parent
    __package__ = "_e14_" + hashlib.sha256(str(_root).encode()).hexdigest()[:16]
    if __package__ not in sys.modules:
        _package = types.ModuleType(__package__)
        _package.__path__ = [str(_root)]
        sys.modules[__package__] = _package

from .model_abi import model_config, qwen_module, validate_inputs


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen = qwen_module()
        self.config = model_config()

    def forward(self, packed_weights, prompt_tokens, decode_steps):
        weights = validate_inputs(packed_weights, prompt_tokens, decode_steps)
        with torch.inference_mode():
            with torch.device("meta"):
                model = self.qwen.Qwen3ForCausalLM(self.config)
            model.load_state_dict(weights, strict=True, assign=True)
            model.model.rotary_emb = self.qwen.Qwen3RotaryEmbedding(
                self.config, device=prompt_tokens.device,
            )
            model.eval()
            positions = torch.arange(7, device=prompt_tokens.device).unsqueeze(0)
            response = model(
                input_ids=prompt_tokens.unsqueeze(0), position_ids=positions,
                attention_mask=torch.ones_like(positions), use_cache=True,
                output_hidden_states=True, return_dict=True,
            )
            generated = response.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            return (
                torch.cat((prompt_tokens.unsqueeze(0), generated), dim=1),
                response.hidden_states[-1].squeeze(0).contiguous().clone(),
            )
