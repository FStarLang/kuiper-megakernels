module KPlay.Spec

(* Typed specification of greedy BF16 decode of Qwen3.5-0.8B (LUCE).

   This is the human-readable statement of what both roles compute. It is a
   typed mathematical model of the pinned reference, not a description of
   buffers: named dimensions, records of model weights (fields named as in
   the checkpoint), one real-valued equation per reference operation, %~
   liftings, and the program postcondition over typed inputs/outputs.

   Reference sources (all under reference/luce/):
     M = modeling_qwen3_5.py (pinned HF transformers 5.3.0 excerpt; line
         numbers below are lines of that file),
     B = base.py (decode driver),
     input-domain.txt and LAYOUT.md (profile, input domain, outputs).
   Two reference details are fixed by that driver: every model call has
   sequence length one (B:136-147 feeds one token per call, B:151-157), and
   all states start at zero, so the first DeltaNet call (M:586-597, initial
   state None, zero-padded conv state) is the recurrent update (M:599-609)
   from all-zero states.

   The math is over ideal reals. Floating results are related to it only by
   the abstract approximation %~ (Kuiper.Approximates); the specification
   gives no error bound. The model constants are pinned (numerics_ok) to the
   float32 values PyTorch computes in the reference: the f32 roundings of
   1e-6, 1/sqrt(128) and the RoPE inverse frequencies (M:217, float32). *)

open FStar.Real
open Kuiper
open Kuiper.Approximates
module S = FStar.Seq
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32
module F = Kuiper.Floating
module Exp = FStar.Math.Exp
module Sqrt = FStar.Math.Sqrt
module Trig = Kuiper.Real.Trigonometry
module Lit = Kuiper.Float32.Literals

#set-options "--z3rlimit 20"

(*** S1. Model *)

(* Dimensions (model_config.json.txt text_config; input-domain.txt) *)
let hidden_size : nat = 1024
let intermediate_size : nat = 3584
let vocab_size : nat = 248320
let num_hidden_layers : nat = 24       (* pattern: 3 linear_attention + 1 full_attention, x 6 *)
let linear_layer_count : nat = 18
let full_layer_count : nat = 6
let linear_num_heads : nat = 16        (* linear_num_key_heads = linear_num_value_heads *)
let linear_head_dim : nat = 128        (* linear_key_head_dim = linear_value_head_dim *)
let linear_key_dim : nat = 2048        (* 16 x 128; also value_dim *)
let conv_dim : nat = 6144              (* key_dim * 2 + value_dim, M:463 *)
let conv_kernel_size : nat = 4         (* linear_conv_kernel_dim *)
let num_attention_heads : nat = 8
let num_key_value_heads : nat = 2      (* GQA: query head h reads kv head h / 4 *)
let head_dim : nat = 256
let q_proj_dim : nat = 4096            (* per head [query 256 | gate 256], M:722 *)
let kv_proj_dim : nat = 512            (* 2 kv heads x 256 *)
let attn_out_dim : nat = 2048          (* 8 heads x 256 *)
let rotary_pairs : nat = 32            (* rotary dim 64 = 256 x partial_rotary_factor 0.25 *)
let max_prompt_length : nat = 8
let max_decode_steps : nat = 5

type index (n:nat) = i:nat{i < n}
type vector (a:Type0) (n:nat) = index n -> GTot a
type matrix (a:Type0) (rows cols:nat) = index rows -> GTot (vector a cols)
type tensor3 (a:Type0) (d0 d1 d2:nat) = index d0 -> GTot (matrix a d1 d2)

(* Parameters of one layer, named as model.layers.i.* in the checkpoint
   (LAYOUT.md). Matrices are [out_features, in_features] as in nn.Linear. *)
noeq type mlp_params (a:Type0) = {
  gate_proj : matrix a intermediate_size hidden_size;
  up_proj   : matrix a intermediate_size hidden_size;
  down_proj : matrix a hidden_size intermediate_size;
}

noeq type linear_attn_params (a:Type0) = {
  in_proj_qkv : matrix a conv_dim hidden_size;
  in_proj_z   : matrix a linear_key_dim hidden_size;
  in_proj_b   : matrix a linear_num_heads hidden_size;
  in_proj_a   : matrix a linear_num_heads hidden_size;
  conv1d      : matrix a conv_dim conv_kernel_size;     (* conv1d.weight [6144, 1, 4] *)
  a_log       : vector a linear_num_heads;              (* A_log *)
  dt_bias     : vector a linear_num_heads;
  norm        : vector a linear_head_dim;               (* gated RMSNorm weight *)
  out_proj    : matrix a hidden_size linear_key_dim;
}

noeq type self_attn_params (a:Type0) = {
  q_proj : matrix a q_proj_dim hidden_size;
  k_proj : matrix a kv_proj_dim hidden_size;
  v_proj : matrix a kv_proj_dim hidden_size;
  q_norm : vector a head_dim;
  k_norm : vector a head_dim;
  o_proj : matrix a hidden_size attn_out_dim;
}

(* token_mixer is linear_attn (Gated DeltaNet) or self_attn (full attention) *)
noeq type decoder_layer (mixer:Type0 -> Type0) (a:Type0) = {
  input_layernorm          : vector a hidden_size;
  post_attention_layernorm : vector a hidden_size;
  mlp                      : mlp_params a;
  token_mixer              : mixer a;
}

(* Model layer l is full_layers (l / 4) if l % 4 = 3, otherwise
   linear_layers (3 * (l / 4) + l % 4). *)
noeq type model_weights (a:Type0) = {
  embed_tokens  : matrix a vocab_size hidden_size;   (* also lm_head.weight (tied), M:989 *)
  final_norm    : vector a hidden_size;              (* model.norm.weight *)
  linear_layers : index linear_layer_count -> GTot (decoder_layer linear_attn_params a);
  full_layers   : index full_layer_count -> GTot (decoder_layer self_attn_params a);
}

(* Model constants, pinned by numerics_ok. *)
noeq type numerics = {
  rms_norm_eps       : real;                    (* 1e-6, also the l2norm epsilon *)
  linear_query_scale : real;                    (* 1 / sqrt(128), M:412 *)
  inv_freq           : vector real rotary_pairs; (* RoPE inverse frequencies, M:217 *)
}

(* Per-position decode state (Qwen3_5DynamicCache, M:78-158). *)
type conv_state = matrix real conv_dim conv_kernel_size     (* [channel][tap], tap 3 newest *)
type recurrent_state = tensor3 real linear_num_heads linear_head_dim linear_head_dim  (* [head][k][v] *)
type kv_entry = matrix real num_key_value_heads head_dim    (* [kv head][dim] at one position *)

noeq type cache_state = {
  conv_states      : index linear_layer_count -> GTot conv_state;
  recurrent_states : index linear_layer_count -> GTot recurrent_state;
  key_cache        : index full_layer_count -> GTot (nat -> GTot kv_entry);   (* by position *)
  value_cache      : index full_layer_count -> GTot (nat -> GTot kv_entry);
}

noeq type layer_state = {
  cache  : cache_state;
  hidden : vector real hidden_size;   (* residual stream *)
}

(* Program inputs and outputs (LAYOUT.md "Contract"). *)
noeq type inputs = {
  weights       : model_weights BF.t;             (* packed_weights *)
  prompt_length : nat;
  prompt        : vector int prompt_length;       (* prompt_tokens *)
  decode_steps  : int;
}

noeq type outputs = {
  generated_count : nat;
  tokens          : vector int generated_count;
  hidden_states   : vector (vector BF.t hidden_size) generated_count;
}

(*** S2. Reference equations (real-valued) *)

(* Totalized real helpers: division by zero and rsqrt of a non-positive value
   give 0 (never reached when eps > 0 except for the softmax denominator,
   which is positive). *)
let rdiv (x y:real) : real = if t2b (y =!= 0.0R) then x /. y else 0.0R
let rsqrt_pos (x:real) : real = if t2b (x >. 0.0R) then Sqrt.rsqrt x else 0.0R

let sum (n:nat) (f:index n -> GTot real) : GTot real = rsum (S.init_ghost n f)

let dot (#n:nat) (x y:vector real n) : GTot real = sum n (fun k -> x k *. y k)

(* nn.Linear without bias: y = W x *)
let linear (#rows #cols:nat) (w:matrix real rows cols) (x:vector real cols) : GTot (vector real rows) =
  fun r -> dot (w r) x

let vadd (#n:nat) (x y:vector real n) : GTot (vector real n) = fun i -> x i +. y i

let sum_of_squares (#n:nat) (x:vector real n) : GTot real = sum n (fun k -> x k *. x k)

let sigmoid (x:real) : real =
  Exp.exp_positive (0.0R -. x);
  1.0R /. (1.0R +. Exp.exp (0.0R -. x))

let silu (x:real) : real = x *. sigmoid x

(* F.softplus with its default threshold 20 (M:582) *)
let softplus (r:real) : real =
  if t2b (r >. 20.0R) then r
  else (Exp.exp_positive r; Exp.log (1.0R +. Exp.exp r))

(* Qwen3_5RMSNorm, M:809-817: x * rsqrt(mean(x^2) + eps) * (1 + w) *)
let rms_norm (#n:nat) (eps:real) (x w:vector real n) : GTot (vector real n) =
  let inv = rsqrt_pos (rdiv (sum_of_squares x) (of_int n) +. eps) in
  fun i -> (x i *. inv) *. (1.0R +. w i)

(* Qwen3_5RMSNormGated, M:266-275: w * (x * rsqrt(mean(x^2) + eps)) * silu(gate) *)
let rms_norm_gated (#n:nat) (eps:real) (x w gate:vector real n) : GTot (vector real n) =
  let inv = rsqrt_pos (rdiv (sum_of_squares x) (of_int n) +. eps) in
  fun i -> (w i *. (x i *. inv)) *. silu (gate i)

(* l2norm, M:313-316 *)
let l2norm (#n:nat) (eps:real) (x:vector real n) : GTot (vector real n) =
  let inv = rsqrt_pos (sum_of_squares x +. eps) in
  fun i -> x i *. inv

(* Qwen3_5MLP, M:798-800: down(silu(gate(x)) * up(x)) *)
let swiglu (#n:nat) (gate up:vector real n) : GTot (vector real n) = fun i -> silu (gate i) *. up i

let swiglu_mlp (p:mlp_params real) (x:vector real hidden_size) : GTot (vector real hidden_size) =
  linear p.down_proj (swiglu (linear p.gate_proj x) (linear p.up_proj x))

(* Qwen3_5DecoderLayer, M:846-877: residual + token mixer, then residual +
   mlp(post_attention_layernorm(.)). `mixed` is the token mixer output on
   input_layernorm(x). *)
let decoder_output (#m:Type0 -> Type0) (eps:real) (p:decoder_layer m real)
                   (x mixed:vector real hidden_size) : GTot (vector real hidden_size) =
  let h = vadd x mixed in
  vadd h (swiglu_mlp p.mlp (rms_norm eps h p.post_attention_layernorm))

(** Gated DeltaNet (linear_attention), Qwen3_5GatedDeltaNet.forward M:507-622,
    on the normalized input xn of one token. *)

(* torch_causal_conv1d_update, M:295-311: the state window shifts by one and
   takes the new projected input; output = silu(depthwise conv). *)
let conv_window (state:conv_state) (mixed_qkv:vector real conv_dim) : GTot conv_state =
  fun c k -> if k < 3 then state c (k + 1) else mixed_qkv c

let linear_attn_conv_state (p:linear_attn_params real) (xn:vector real hidden_size) (conv:conv_state)
  : GTot conv_state =
  conv_window conv (linear p.in_proj_qkv xn)

(* depthwise conv over the window, then silu (M:307-308) *)
let causal_conv1d_output (window:conv_state) (weight:matrix real conv_dim conv_kernel_size)
  : GTot (vector real conv_dim) =
  fun c -> silu (sum conv_kernel_size (fun k -> window c k *. weight c k))

let linear_attn_conv_out (p:linear_attn_params real) (xn:vector real hidden_size) (conv:conv_state)
  : GTot (vector real conv_dim) =
  causal_conv1d_output (linear_attn_conv_state p xn conv) p.conv1d

(* split into query / key / value heads, M:561-575; l2norm in the kernel
   (use_qk_l2norm_in_kernel, M:405-407) and query scale (M:412-413) *)
let linear_attn_query (nm:numerics) (p:linear_attn_params real) (xn:vector real hidden_size)
                      (conv:conv_state) : GTot (matrix real linear_num_heads linear_head_dim) =
  let co = linear_attn_conv_out p xn conv in
  fun h d -> l2norm nm.rms_norm_eps (fun e -> co (h * 128 + e)) d *. nm.linear_query_scale

let linear_attn_key (nm:numerics) (p:linear_attn_params real) (xn:vector real hidden_size)
                    (conv:conv_state) : GTot (matrix real linear_num_heads linear_head_dim) =
  let co = linear_attn_conv_out p xn conv in
  fun h d -> l2norm nm.rms_norm_eps (fun e -> co (2048 + h * 128 + e)) d

let linear_attn_value (p:linear_attn_params real) (xn:vector real hidden_size)
                      (conv:conv_state) : GTot (matrix real linear_num_heads linear_head_dim) =
  let co = linear_attn_conv_out p xn conv in
  fun h d -> co (4096 + h * 128 + d)

(* beta = sigmoid(b), M:580 *)
let linear_attn_beta (p:linear_attn_params real) (xn:vector real hidden_size) : GTot (vector real linear_num_heads) =
  fun h -> sigmoid (linear p.in_proj_b xn h)

(* exp(g), g = -exp(A_log) * softplus(a + dt_bias), M:582 and M:428 *)
let linear_attn_decay (p:linear_attn_params real) (xn:vector real hidden_size) : GTot (vector real linear_num_heads) =
  fun h -> Exp.exp ((0.0R -. Exp.exp (p.a_log h)) *. softplus (linear p.in_proj_a xn h +. p.dt_bias h))

(* torch_recurrent_gated_delta_rule, one step, M:429-432 *)
let delta_rule_correction (state:recurrent_state) (k v:matrix real linear_num_heads linear_head_dim)
                          (decay beta:vector real linear_num_heads)
  : GTot (matrix real linear_num_heads linear_head_dim) =
  fun h j ->
    (v h j -. sum linear_head_dim (fun i -> (state h i j *. decay h) *. k h i)) *. beta h

let delta_rule_state (state:recurrent_state) (k v:matrix real linear_num_heads linear_head_dim)
                     (decay beta:vector real linear_num_heads) : GTot recurrent_state =
  let delta = delta_rule_correction state k v decay beta in
  fun h i j -> state h i j *. decay h +. k h i *. delta h j

let linear_attn_recurrent_state (nm:numerics) (p:linear_attn_params real) (xn:vector real hidden_size)
                                (conv:conv_state) (state:recurrent_state) : GTot recurrent_state =
  delta_rule_state state (linear_attn_key nm p xn conv) (linear_attn_value p xn conv)
    (linear_attn_decay p xn) (linear_attn_beta p xn)

(* core_attn_out = state^T q with the updated state, M:433 *)
let recurrent_readout (state:recurrent_state) (q:matrix real linear_num_heads linear_head_dim)
  : GTot (matrix real linear_num_heads linear_head_dim) =
  fun h j -> sum linear_head_dim (fun i -> state h i j *. q h i)

let linear_attn_core (nm:numerics) (p:linear_attn_params real) (xn:vector real hidden_size)
                     (conv:conv_state) (state:recurrent_state)
  : GTot (matrix real linear_num_heads linear_head_dim) =
  recurrent_readout (linear_attn_recurrent_state nm p xn conv state) (linear_attn_query nm p xn conv)

(* gated RMSNorm per head, gate z = in_proj_z(x) (M:534, M:616-618) *)
let gated_head_norm (eps:real) (w:vector real linear_head_dim)
                    (core:matrix real linear_num_heads linear_head_dim) (z:vector real linear_key_dim)
  : GTot (matrix real linear_num_heads linear_head_dim) =
  fun h -> rms_norm_gated eps (core h) w (fun j -> z (h * 128 + j))

(* heads flattened to 2048, then out_proj (M:620-621) *)
let linear_attn_output (nm:numerics) (p:linear_attn_params real) (xn:vector real hidden_size)
                       (conv:conv_state) (state:recurrent_state) : GTot (vector real hidden_size) =
  let gated = gated_head_norm nm.rms_norm_eps p.norm (linear_attn_core nm p xn conv state)
                (linear p.in_proj_z xn) in
  linear p.out_proj (fun c -> gated (c / 128) (c % 128))

(** Gated full attention (self_attn), Qwen3_5Attention.forward M:736-784. *)

(* RoPE tables at position t (M:217-239): angle t * inv_freq, duplicated over
   the 64 rotary dims. *)
let rotary_cos (nm:numerics) (t:int) : GTot (vector real rotary_pairs) =
  fun i -> Trig.cos (of_int t *. nm.inv_freq i)

let rotary_sin (nm:numerics) (t:int) : GTot (vector real rotary_pairs) =
  fun i -> Trig.sin (of_int t *. nm.inv_freq i)

(* apply_rotary_pos_emb with rotate_half on the first 64 dims, M:625-668 *)
let apply_rotary (u:vector real head_dim) (cos sin:vector real rotary_pairs) : GTot (vector real head_dim) =
  fun i ->
    if i < 32 then u i *. cos i -. u (i + 32) *. sin i
    else if i < 64 then u i *. cos (i - 32) +. u (i - 32) *. sin (i - 32)
    else u i

(* query_states (q_norm then RoPE, M:748-753, M:760) and the output gate *)
let attn_query (nm:numerics) (p:self_attn_params real) (xn:vector real hidden_size)
               (cos sin:vector real rotary_pairs) : GTot (matrix real num_attention_heads head_dim) =
  let q = linear p.q_proj xn in
  fun h -> apply_rotary (rms_norm nm.rms_norm_eps (fun e -> q (h * 512 + e)) p.q_norm) cos sin

let attn_gate (p:self_attn_params real) (xn:vector real hidden_size)
  : GTot (matrix real num_attention_heads head_dim) =
  let q = linear p.q_proj xn in
  fun h e -> q (h * 512 + 256 + e)

(* key_states (k_norm then RoPE) and value_states, M:754-755, M:760 *)
let attn_key (nm:numerics) (p:self_attn_params real) (xn:vector real hidden_size)
             (cos sin:vector real rotary_pairs) : GTot kv_entry =
  let k = linear p.k_proj xn in
  fun h -> apply_rotary (rms_norm nm.rms_norm_eps (fun e -> k (h * 256 + e)) p.k_norm) cos sin

let attn_value (p:self_attn_params real) (xn:vector real hidden_size) : GTot kv_entry =
  let v = linear p.v_proj xn in
  fun h e -> v (h * 256 + e)

(* eager_attention_forward, M:683-705: scores q.k * head_dim^-0.5 (M:718),
   softmax over positions 0..t (causal), weighted values; GQA repeat_kv. *)
let attn_score (q k:vector real head_dim) : GTot real = dot q k *. 0.0625R

let attn_weight (t:nat) (q:vector real head_dim) (keys:nat -> GTot (vector real head_dim)) (pos:nat)
  : GTot real =
  rdiv (Exp.exp (attn_score q (keys pos)))
       (sum (t + 1) (fun p' -> Exp.exp (attn_score q (keys p'))))

let attn_head (t:nat) (q:vector real head_dim) (keys values:nat -> GTot (vector real head_dim))
  : GTot (vector real head_dim) =
  fun e -> sum (t + 1) (fun pos -> attn_weight t q keys pos *. values pos e)

(* attention over the cache including position t, gated by sigmoid(gate)
   (M:781) and projected by o_proj (M:783) *)
(* all query heads; query head h reads kv head h / 4 (repeat_kv, M:671-680) *)
let attention_heads (t:nat) (q:matrix real num_attention_heads head_dim)
                    (keys values:nat -> GTot kv_entry) : GTot (matrix real num_attention_heads head_dim) =
  fun h -> attn_head t (q h) (fun pos -> keys pos (h / 4)) (fun pos -> values pos (h / 4))

let self_attention (nm:numerics) (p:self_attn_params real) (xn:vector real hidden_size)
                   (cos sin:vector real rotary_pairs) (t:nat)
                   (keys values:nat -> GTot kv_entry) : GTot (vector real hidden_size) =
  let heads = attention_heads t (attn_query nm p xn cos sin) keys values in
  let gate = attn_gate p xn in
  linear p.o_proj (fun c -> heads (c / 256) (c % 256) *. sigmoid (gate (c / 256) (c % 256)))

(** Decoder stack and decode loop *)

let update (#a:Type0) (#n:nat) (f:index n -> GTot a) (i:index n) (v:a) : index n -> GTot a =
  fun j -> if j = i then v else f j

(* past_key_values.update (M:763): write position t, then attend over 0..t *)
let append_position (history:nat -> GTot kv_entry) (t:nat) (entry:kv_entry) : nat -> GTot kv_entry =
  fun pos -> if pos = t then entry else history pos

let linear_layer_step (nm:numerics) (p:decoder_layer linear_attn_params real)
                      (d:index linear_layer_count) (st:layer_state) : GTot layer_state =
  let x = st.hidden in
  let xn = rms_norm nm.rms_norm_eps x p.input_layernorm in
  let conv = st.cache.conv_states d in
  let state = st.cache.recurrent_states d in
  { cache = { st.cache with
      conv_states = update st.cache.conv_states d (linear_attn_conv_state p.token_mixer xn conv);
      recurrent_states = update st.cache.recurrent_states d
                           (linear_attn_recurrent_state nm p.token_mixer xn conv state) };
    hidden = decoder_output nm.rms_norm_eps p x (linear_attn_output nm p.token_mixer xn conv state) }

let full_layer_step (nm:numerics) (p:decoder_layer self_attn_params real)
                    (g:index full_layer_count) (t:nat) (cos sin:vector real rotary_pairs)
                    (st:layer_state) : GTot layer_state =
  let x = st.hidden in
  let xn = rms_norm nm.rms_norm_eps x p.input_layernorm in
  let keys = append_position (st.cache.key_cache g) t (attn_key nm p.token_mixer xn cos sin) in
  let values = append_position (st.cache.value_cache g) t (attn_value p.token_mixer xn) in
  { cache = { st.cache with
      key_cache = update st.cache.key_cache g keys;
      value_cache = update st.cache.value_cache g values };
    hidden = decoder_output nm.rms_norm_eps p x (self_attention nm p.token_mixer xn cos sin t keys values) }

(* Layers l .. 23 at position t (Qwen3_5TextModel.forward, M:952-963). *)
let rec run_layers (nm:numerics) (w:model_weights real) (t:nat) (cos sin:vector real rotary_pairs)
                   (st:layer_state) (l:nat)
  : GTot layer_state (decreases (if l < 24 then 24 - l else 0)) =
  if l >= 24 then st
  else if l % 4 = 3 then
    run_layers nm w t cos sin (full_layer_step nm (w.full_layers (l / 4)) (l / 4) t cos sin st) (l + 1)
  else
    let d : index linear_layer_count = 3 * (l / 4) + l % 4 in
    run_layers nm w t cos sin (linear_layer_step nm (w.linear_layers d) d st) (l + 1)

(* embed_tokens (M:914); invalid IDs are outside the input domain *)
let token_embedding (w:model_weights real) (tok:int) : GTot (vector real hidden_size) =
  fun i -> if 0 <= tok && tok < 248320 then w.embed_tokens tok i else 0.0R

let empty_cache : cache_state = {
  conv_states = (fun _ _ _ -> 0.0R);
  recurrent_states = (fun _ _ _ _ -> 0.0R);
  key_cache = (fun _ _ _ _ -> 0.0R);
  value_cache = (fun _ _ _ _ -> 0.0R);
}

(* One model call on token tok at position t (B:136-147). *)
let decode_step (nm:numerics) (w:model_weights real) (tok:int) (t:nat) (c:cache_state) : GTot layer_state =
  run_layers nm w t (rotary_cos nm t) (rotary_sin nm t)
    { cache = c; hidden = token_embedding w tok } 0

(* Positions 0 .. n-1 from fresh state; token at position p is toks p. *)
let rec decode (nm:numerics) (w:model_weights real) (toks:nat -> GTot int) (n:nat)
  : GTot layer_state (decreases n) =
  if n = 0 then { cache = empty_cache; hidden = (fun _ -> 0.0R) }
  else decode_step nm w (toks (n - 1)) (n - 1) (decode nm w toks (n - 1)).cache

(* Residual stream after layer 23, before the final norm, at position p
   (the hook in B:128-130). *)
let hidden_at (nm:numerics) (w:model_weights real) (toks:nat -> GTot int) (p:nat)
  : GTot (vector real hidden_size) =
  (decode nm w toks (p + 1)).hidden

(* Final RMSNorm (M:964) and tied LM head (M:1055) *)
let logits (nm:numerics) (w:model_weights real) (h:vector real hidden_size) : GTot (vector real vocab_size) =
  let hn = rms_norm nm.rms_norm_eps h w.final_norm in
  fun m -> dot (w.embed_tokens m) hn

(** Model constants *)

(* f32 roundings of 1e-6 and 1 / sqrt(128), as PyTorch uses them *)
let eps_f32 : real = Lit.value "0.0000009999999974752427078783512115478515625"
let query_scale_f32 : real = Lit.value "0.088388346135616302490234375"

(* PyTorch's float32 inv_freq = 1 / theta^(2i / 64), theta = 1e7 (M:217) *)
let inv_freq_f32_literal (i:nat) : string =
  match i with
  | 0 -> "1.0"
  | 1 -> "0.604296386241912841796875"
  | 2 -> "0.3651741445064544677734375"
  | 3 -> "0.22067339718341827392578125"
  | 4 -> "0.13335214555263519287109375"
  | 5 -> "0.08058421313762664794921875"
  | 6 -> "0.0486967526376247406005859375"
  | 7 -> "0.02942727319896221160888671875"
  | 8 -> "0.01778279431164264678955078125"
  | 9 -> "0.01074607856571674346923828125"
  | 10 -> "0.0064938166178762912750244140625"
  | 11 -> "0.0039241896010935306549072265625"
  | 12 -> "0.00237137381918728351593017578125"
  | 13 -> "0.001433012657798826694488525390625"
  | 14 -> "0.000865964335389435291290283203125"
  | 15 -> "0.000523299095220863819122314453125"
  | 16 -> "0.000316227786242961883544921875"
  | 17 -> "0.000191095299669541418552398681640625"
  | 18 -> "0.000115478193038143217563629150390625"
  | 19 -> "0.000069783054641447961330413818359375"
  | 20 -> "0.0000421696531702764332294464111328125"
  | 21 -> "0.000025482966520939953625202178955078125"
  | 22 -> "0.000015399264157167635858058929443359375"
  | 23 -> "0.00000930572059587575495243072509765625"
  | 24 -> "0.00000562341347176698036491870880126953125"
  | 25 -> "0.00000339820826411596499383449554443359375"
  | 26 -> "0.00000205352489501819945871829986572265625"
  | 27 -> "0.00000124093776321387849748134613037109375"
  | 28 -> "0.00000074989424092564149759709835052490234375"
  | 29 -> "0.000000453158378377338522113859653472900390625"
  | 30 -> "0.00000027384197665014653466641902923583984375"
  | 31 -> "0.00000016548170833630138076841831207275390625"
  | _ -> "0.0"

let inv_freq_f32 : vector real rotary_pairs = fun i -> Lit.value (inv_freq_f32_literal i)

let numerics_ok (nm:numerics) : prop =
  nm.rms_norm_eps == eps_f32 /\
  nm.linear_query_scale == query_scale_f32 /\
  (forall (i:index rotary_pairs). nm.inv_freq i == inv_freq_f32 i)

(*** S3. Liftings of %~ to the model types *)

let vector_approximates (#a:Type0) {| Kuiper.Scalars.scalar a, real_like a |} (#n:nat)
    (x:vector a n) (r:vector real n) : prop =
  forall (i:index n). x i %~ r i

let matrix_approximates (#a:Type0) {| Kuiper.Scalars.scalar a, real_like a |} (#rows #cols:nat)
    (x:matrix a rows cols) (r:matrix real rows cols) : prop =
  forall (i:index rows) (j:index cols). x i j %~ r i j

let mlp_approximates (x:mlp_params BF.t) (r:mlp_params real) : prop =
  matrix_approximates x.gate_proj r.gate_proj /\
  matrix_approximates x.up_proj r.up_proj /\
  matrix_approximates x.down_proj r.down_proj

let linear_attn_approximates (x:linear_attn_params BF.t) (r:linear_attn_params real) : prop =
  matrix_approximates x.in_proj_qkv r.in_proj_qkv /\
  matrix_approximates x.in_proj_z r.in_proj_z /\
  matrix_approximates x.in_proj_b r.in_proj_b /\
  matrix_approximates x.in_proj_a r.in_proj_a /\
  matrix_approximates x.conv1d r.conv1d /\
  vector_approximates x.a_log r.a_log /\
  vector_approximates x.dt_bias r.dt_bias /\
  vector_approximates x.norm r.norm /\
  matrix_approximates x.out_proj r.out_proj

let self_attn_approximates (x:self_attn_params BF.t) (r:self_attn_params real) : prop =
  matrix_approximates x.q_proj r.q_proj /\
  matrix_approximates x.k_proj r.k_proj /\
  matrix_approximates x.v_proj r.v_proj /\
  vector_approximates x.q_norm r.q_norm /\
  vector_approximates x.k_norm r.k_norm /\
  matrix_approximates x.o_proj r.o_proj

let layer_approximates (#m:Type0 -> Type0) (mixer_approximates:m BF.t -> m real -> prop)
    (x:decoder_layer m BF.t) (r:decoder_layer m real) : prop =
  vector_approximates x.input_layernorm r.input_layernorm /\
  vector_approximates x.post_attention_layernorm r.post_attention_layernorm /\
  mlp_approximates x.mlp r.mlp /\
  mixer_approximates x.token_mixer r.token_mixer

let weights_approximates (x:model_weights BF.t) (r:model_weights real) : prop =
  matrix_approximates x.embed_tokens r.embed_tokens /\
  vector_approximates x.final_norm r.final_norm /\
  (forall (d:index linear_layer_count).
     layer_approximates linear_attn_approximates (x.linear_layers d) (r.linear_layers d)) /\
  (forall (g:index full_layer_count).
     layer_approximates self_attn_approximates (x.full_layers g) (r.full_layers g))

(*** S4. Operation relations (role-independent) *)

let valid_token (x:int) : bool = 0 <= x && x < 248320

let is_nan (#t:Type) {| F.floating t |} (x:t) : prop = NaN? (F.kind x)

(* Greedy selection (torch.argmax, B:147) over the score row actually
   consumed: if the row has a NaN the chosen element is NaN, otherwise no
   element is strictly greater. Ties may be broken either way. *)
let allowed_choice (#t:Type) {| F.floating t |} (#n:nat) (row:vector t n) (sel:int) : prop =
  0 <= sel /\ sel < n /\
  ((exists (k:index n). is_nan (row k)) ==> is_nan (row sel)) /\
  ((forall (k:index n). ~(is_nan (row k))) ==> (forall (k:index n). ~(lt (row sel) (row k))))

(* The decode operation: an output hidden row approximates the reference
   residual stream at position p (consumed tokens toks). *)
let hidden_state_relation (nm:numerics) (w:model_weights real) (toks:nat -> GTot int) (p:nat)
                          (h:vector BF.t hidden_size) : prop =
  vector_approximates h (hidden_at nm w toks p)

(* The logits/argmax operation: sel is an allowed choice over a BF16 or F32
   score row that approximates the reference logits of hidden state h. *)
let greedy_choice_relation (nm:numerics) (w:model_weights real) (h:vector real hidden_size) (sel:int)
  : prop =
  (exists (row:vector BF.t vocab_size). vector_approximates row (logits nm w h) /\ allowed_choice row sel) \/
  (exists (row:vector F32.t vocab_size). vector_approximates row (logits nm w h) /\ allowed_choice row sel)

(*** S5. Typed program postcondition *)

(* input-domain.txt; decode_steps in {1, 5} is relaxed to 1..5 *)
let input_domain (x:inputs) : prop =
  1 <= x.prompt_length /\ x.prompt_length <= max_prompt_length /\
  1 <= x.decode_steps /\ x.decode_steps <= max_decode_steps /\
  (forall (i:index x.prompt_length). valid_token (x.prompt i))

(* Token consumed at decode position p: the prompt, then the generated tokens
   (B:151-157). *)
let consumed (x:inputs) (y:outputs) (p:nat) : GTot int =
  if p < x.prompt_length then x.prompt p
  else if p - x.prompt_length < y.generated_count then y.tokens (p - x.prompt_length)
  else 0

(* Position of the last prompt token (= prompt_length - 1 on the domain). *)
let last_prompt_position (x:inputs) : nat = if x.prompt_length >= 1 then x.prompt_length - 1 else 0

(* Structural: shape and token validity. *)
let structural (x:inputs) (y:outputs) : prop =
  y.generated_count == x.decode_steps /\
  (forall (j:index y.generated_count). valid_token (y.tokens j))

(* Numerical: hidden state j approximates the reference residual stream at
   position P - 1 + j. *)
let numerical (nm:numerics) (x:inputs) (y:outputs) (w:model_weights real) : prop =
  forall (j:index y.generated_count).
    hidden_state_relation nm w (consumed x y) (last_prompt_position x + j) (y.hidden_states j)

(* Selection: token j is a greedy choice over logits at that position. *)
let selection (nm:numerics) (x:inputs) (y:outputs) (w:model_weights real) : prop =
  forall (j:index y.generated_count).
    greedy_choice_relation nm w (hidden_at nm w (consumed x y) (last_prompt_position x + j)) (y.tokens j)

let postcondition (input:inputs) (output:outputs) : prop =
  input_domain input ==>
  (structural input output /\
   (exists (nm:numerics). numerics_ok nm /\
     (forall (w:model_weights real). weights_approximates input.weights w ==>
        numerical nm input output w /\ selection nm input output w)))
