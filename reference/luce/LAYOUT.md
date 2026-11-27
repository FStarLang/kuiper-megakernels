# Packed weight layout and program contract

Derived from `experiments/E13_luce_reduction/layout.py` (`WEIGHT_LAYOUT`, built from `assets.expected_weight_shapes()` in insertion order) and checked against the frozen `weight_layout.json` of this run (sha256 f22255415c1365f8a3c72067ff52bdc5c0292aaf9d6e95c87fb63e7bdf7f86fe).

## Contract (both roles)

`forward(packed_weights, prompt_tokens, decode_steps) -> (tokens, hidden_states)`

- `packed_weights`: contiguous 1-D BF16 tensor of 752393024 elements: the concatenation, in the order below, of every Qwen3.5-0.8B text parameter flattened row-major (C order). Element `offset + flat_index` is `param[multi_index]`. `lm_head.weight` is not stored: it is tied to `model.embed_tokens.weight` (logits = final-normed hidden @ embed^T).
- `prompt_tokens`: contiguous int64 `[P]`, 1 <= P <= 8, each ID in [0, 248320).
- `decode_steps`: Python int in {1, 5} (a scalar program input, not a tensor).
- Outputs: `tokens` int64 `[decode_steps]` and `hidden_states` BF16 `[decode_steps, 1024]`.
  Step s output: run the model on one token at a time with fresh state (zero KV cache, zero DeltaNet recurrent state, zero convolution state; positions 0, 1, 2, ...). Feed `prompt[0..P-2]` (outputs discarded), then feed `prompt[P-1]`; each subsequent step feeds the previously generated token. `tokens[s]` is the greedy argmax (first maximal index, as `torch.argmax`) of the logits after feeding the step-s token, and `hidden_states[s]` is the BF16 residual stream after layer 23, before the final RMSNorm, at that same position. No EOS stop, no sampling, repetition penalty 1.0.
- The reduced CUDA programs (`inputs/<role>/program.cu`) take the same packed buffer and index it with absolute element offsets from this table (e.g. `weights[base + k]` where `base` is a layer's start offset).

## Global parameters

| name | shape | offset | elements |
|---|---|---:|---:|
| model.embed_tokens.weight | [248320, 1024] | 0 | 254279680 |
| model.norm.weight | [1024] | 254279680 | 1024 |

## Per-layer fields, linear_attention (Gated DeltaNet) layer; layers 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22

Offsets are relative to the layer start (layer 0 starts at 254280704; size 21555360 elements).

| field (model.layers.i.*) | shape | relative offset | elements |
|---|---|---:|---:|
| input_layernorm.weight | [1024] | 0 | 1024 |
| post_attention_layernorm.weight | [1024] | 1024 | 1024 |
| mlp.gate_proj.weight | [3584, 1024] | 2048 | 3670016 |
| mlp.up_proj.weight | [3584, 1024] | 3672064 | 3670016 |
| mlp.down_proj.weight | [1024, 3584] | 7342080 | 3670016 |
| linear_attn.in_proj_qkv.weight | [6144, 1024] | 11012096 | 6291456 |
| linear_attn.in_proj_z.weight | [2048, 1024] | 17303552 | 2097152 |
| linear_attn.in_proj_b.weight | [16, 1024] | 19400704 | 16384 |
| linear_attn.in_proj_a.weight | [16, 1024] | 19417088 | 16384 |
| linear_attn.conv1d.weight | [6144, 1, 4] | 19433472 | 24576 |
| linear_attn.A_log | [16] | 19458048 | 16 |
| linear_attn.dt_bias | [16] | 19458064 | 16 |
| linear_attn.norm.weight | [128] | 19458080 | 128 |
| linear_attn.out_proj.weight | [1024, 2048] | 19458208 | 2097152 |

## Per-layer fields, full_attention layer; layers 3,7,11,15,19,23

Offsets are relative to the layer start (layer 3 starts at 318946784; size 18352640 elements).

| field (model.layers.i.*) | shape | relative offset | elements |
|---|---|---:|---:|
| input_layernorm.weight | [1024] | 0 | 1024 |
| post_attention_layernorm.weight | [1024] | 1024 | 1024 |
| mlp.gate_proj.weight | [3584, 1024] | 2048 | 3670016 |
| mlp.up_proj.weight | [3584, 1024] | 3672064 | 3670016 |
| mlp.down_proj.weight | [1024, 3584] | 7342080 | 3670016 |
| self_attn.q_proj.weight | [4096, 1024] | 11012096 | 4194304 |
| self_attn.k_proj.weight | [512, 1024] | 15206400 | 524288 |
| self_attn.v_proj.weight | [512, 1024] | 15730688 | 524288 |
| self_attn.q_norm.weight | [256] | 16254976 | 256 |
| self_attn.k_norm.weight | [256] | 16255232 | 256 |
| self_attn.o_proj.weight | [1024, 2048] | 16255488 | 2097152 |

## Layer start offsets

| layer | type | start offset |
|---:|---|---:|
| 0 | linear_attention | 254280704 |
| 1 | linear_attention | 275836064 |
| 2 | linear_attention | 297391424 |
| 3 | full_attention | 318946784 |
| 4 | linear_attention | 337299424 |
| 5 | linear_attention | 358854784 |
| 6 | linear_attention | 380410144 |
| 7 | full_attention | 401965504 |
| 8 | linear_attention | 420318144 |
| 9 | linear_attention | 441873504 |
| 10 | linear_attention | 463428864 |
| 11 | full_attention | 484984224 |
| 12 | linear_attention | 503336864 |
| 13 | linear_attention | 524892224 |
| 14 | linear_attention | 546447584 |
| 15 | full_attention | 568002944 |
| 16 | linear_attention | 586355584 |
| 17 | linear_attention | 607910944 |
| 18 | linear_attention | 629466304 |
| 19 | full_attention | 651021664 |
| 20 | linear_attention | 669374304 |
| 21 | linear_attention | 690929664 |
| 22 | linear_attention | 712485024 |
| 23 | full_attention | 734040384 |

Notes: `linear_attn.conv1d.weight` `[6144, 1, 4]` is the depthwise causal conv kernel (channel c, tap k at relative offset c*4+k). `linear_attn.A_log` and `linear_attn.dt_bias` are stored in BF16 here (the packed buffer is all BF16). Qwen3.5 RMSNorm weights (`*_layernorm`, `model.norm`, `q_norm`, `k_norm`) are applied as `(1 + weight)`; `linear_attn.norm` (gated RMSNorm) is applied as `weight` (see modeling_qwen3_5.py).
