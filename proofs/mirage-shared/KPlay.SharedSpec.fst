module KPlay.SharedSpec

(* Shared specification of the Qwen3-0.6B seven-token prefill + one greedy
   decode step (reference: inputs/reference/base.py over modeling_qwen3.py).

   (1) Reference math over reals.  The ideal (%~) semantics of Kuiper floats
       follows exact real arithmetic and ignores BF16/FP32 rounding (casts keep
       the ideal), so the reference is stated as exact real math: one decoder
       layer is split at the implementation's cut into `layer_residual`
       (residual stream after attention + MLP, given the residual and its
       already normalized form) and `rmsnorm_rows` (the following RMSNorm).
   (2) `program_postcondition`: role independent, structural + numerical +
       selection (G3), with an existential for the consumed score row.
   (3) One relation per role and family (G1): explicit domain premises (the
       kernels' checks) ==> output shapes, exact copies, and for ALL reals the
       inputs approximate, outputs approximate the reference of those reals.
   (4) Structural helper lemmas for the composition proofs (G4). *)

open FStar.Real
open Kuiper.Approximates
module S = FStar.Seq
module BF = Kuiper.BFloat16
module I64 = FStar.Int64
module F = Kuiper.Floating.Base
module Exp = FStar.Math.Exp
module Sqrt = FStar.Math.Sqrt
module Pow = FStar.Math.Pow
module Trig = Kuiper.Real.Trigonometry

(* ------------------------------------------------------------------ *)
(* Total helpers                                                       *)

(* Flat slice [o, o+n), empty when out of range. Same as the wiring `view`. *)
let window (#a:Type) (s:S.seq a) (o n:int) : S.seq a =
  if 0 <= o && 0 <= n && o + n <= S.length s then S.slice s o (o + n) else S.empty

(* Total element access on real tensors (0 outside the tensor). *)
let at (s:S.seq real) (i:int) : real =
  if 0 <= i && i < S.length s then S.index s i else 0.0R

(* Strided dot product: sum_{k<n} a[ao + k*sa] * b[bo + k*sb], in k order. *)
let rec sdot (n:nat) (a:S.seq real) (ao sa:int) (b:S.seq real) (bo sb:int)
  : Tot real (decreases n)
  = if n = 0 then 0.0R
    else sdot (n - 1) a ao sa b bo sb +. at a (ao + (n - 1) * sa) *. at b (bo + (n - 1) * sb)

(* sum_{k<n} a[o+k]^2, nonnegative by construction. *)
let rec sumsq (n:nat) (a:S.seq real) (o:int) : Tot (r:real{r >=. 0.0R}) (decreases n)
  = if n = 0 then 0.0R else sumsq (n - 1) a o +. at a (o + (n - 1)) *. at a (o + (n - 1))

(* sum_{k<n} exp(a[o+k]), positive for n > 0. *)
let rec sumexp (n:nat) (a:S.seq real) (o:int)
  : Tot (r:real{r >=. 0.0R /\ (n > 0 ==> r >. 0.0R)}) (decreases n)
  = if n = 0 then 0.0R
    else (Exp.exp_positive (at a (o + (n - 1)));
          sumexp (n - 1) a o +. Exp.exp (at a (o + (n - 1))))

(* ------------------------------------------------------------------ *)
(* Model constants (model_config.json, model_abi.py)                   *)

unfold let hidden_size : nat = 1024
unfold let vocab_size : nat = 151936
unfold let prompt_length : nat = 7
unfold let num_layers : nat = 28
unfold let embed_elements : nat = 155582464      (* 151936 * 1024 *)
unfold let final_norm_offset : nat = 155582464
unfold let layers_offset : nat = 155583488
unfold let layer_elements : nat = 15730944
unfold let packed_elements : nat = 596049920

(* Packed element offset of decoder layer l (model_abi.WEIGHT_LAYOUT). *)
let layer_offset (l:nat) : nat = layers_offset + l * layer_elements

(* rms_norm_eps = 1e-6 *)
let rms_eps : real = 1.0R /. 1000000.0R

(* scaling = head_dim ** -0.5 *)
let attn_scale : real = 1.0R /. Sqrt.sqrt 128.0R

(* torch.finfo(bfloat16).min, the additive causal mask of masking_utils
   (= -(2 - 2^-7) * 2^127 = -255 * 2^120). *)
let mask_min : real = 0.0R -. 338953138925153547590470800371487866880.0R

(* rope_theta = 1e6, head_dim = 128: inv_freq[j] = 1 / theta^(2j/128). *)
let inv_freq (j:int) : real = 1.0R /. Pow.pow 1000000.0R (of_int j /. 64.0R)
let rope_angle (p j:int) : real = of_int p *. inv_freq j

(* ------------------------------------------------------------------ *)
(* Reference building blocks over reals (row-major flat tensors)       *)

(* nn.Linear without bias: y[r, o] = sum_k w[o, k] * x[r, k]. *)
[@@"opaque_to_smt"]
let linear (rows outs inner:nat) (w x:S.seq real) : s:S.seq real{S.length s == rows * outs} =
  S.init (rows * outs) (fun (i:nat{i < rows * outs}) ->
    sdot inner w ((i % outs) * inner) 1 x ((i / outs) * inner) 1)

(* Qwen3RMSNorm row inverse: rsqrt(mean(x^2) + eps) over x[o, o+d). *)
let rms_inv (d:pos) (x:S.seq real) (o:int) : real =
  Sqrt.rsqrt (sumsq d x o /. of_int d +. rms_eps)

(* Qwen3RMSNorm over each of `rows` rows of width d: w[c] * (x[r,c] * rms_inv). *)
[@@"opaque_to_smt"]
let rmsnorm_rows (rows:nat) (d:pos) (x w:S.seq real) : s:S.seq real{S.length s == rows * d} =
  S.init (rows * d) (fun (i:nat{i < rows * d}) ->
    at w (i % d) *. (at x i *. rms_inv d x ((i / d) * d)))

(* apply_rotary_pos_emb on [7, nh, 128]: x*cos + rotate_half(x)*sin, with
   cos/sin given as [7, 128] tables of the seven positions. *)
[@@"opaque_to_smt"]
let rope (nh:pos) (x cos sin:S.seq real) : s:S.seq real{S.length s == 7 * nh * 128} =
  S.init (7 * nh * 128) (fun (i:nat{i < 7 * nh * 128}) ->
    let t = i / (nh * 128) in
    let c = i % 128 in
    let other = if c < 64 then 0.0R -. at x (i + 64) else at x (i - 64) in
    at x i *. at cos (t * 128 + c) +. other *. at sin (t * 128 + c))

let silu (x:real) : real =
  Exp.exp_positive (0.0R -. x);
  x /. (1.0R +. Exp.exp (0.0R -. x))

(* Weights of one decoder layer (input_layernorm excluded: the layer receives
   its normalized input explicitly). *)
noeq type layer_weights = {
  post_norm : S.seq real;  (* [1024] post_attention_layernorm *)
  wq : S.seq real;         (* [2048, 1024] *)
  wk : S.seq real;         (* [1024, 1024] *)
  wv : S.seq real;         (* [1024, 1024] *)
  wo : S.seq real;         (* [1024, 2048] *)
  q_norm : S.seq real;     (* [128] *)
  k_norm : S.seq real;     (* [128] *)
  w_gate : S.seq real;     (* [3072, 1024] *)
  w_up : S.seq real;       (* [3072, 1024] *)
  w_down : S.seq real;     (* [1024, 3072] *)
}

(* Layer weights inside the packed tensor at element offset off. *)
let packed_layer (rw:S.seq real) (off:int) : layer_weights = {
  post_norm = window rw (off + 1024) 1024;
  wq = window rw (off + 2048) 2097152;
  wk = window rw (off + 2099200) 1048576;
  wv = window rw (off + 3147776) 1048576;
  wo = window rw (off + 4196352) 2097152;
  q_norm = window rw (off + 6293504) 128;
  k_norm = window rw (off + 6293632) 128;
  w_gate = window rw (off + 6293760) 3145728;
  w_up = window rw (off + 9439488) 3145728;
  w_down = window rw (off + 12585216) 3145728;
}

(* Layer weights from separate slices: stacked QKV [4096,1024] (q;k;v rows),
   q_norm;k_norm [256], o_proj, post norm, stacked gate;up [6144,1024], down. *)
let split_layer (qkv qk_norm wo post gate_up down:S.seq real) : layer_weights = {
  post_norm = post;
  wq = window qkv 0 2097152;
  wk = window qkv 2097152 1048576;
  wv = window qkv 3145728 1048576;
  wo = wo;
  q_norm = window qk_norm 0 128;
  k_norm = window qk_norm 128 128;
  w_gate = window gate_up 0 3145728;
  w_up = window gate_up 3145728 3145728;
  w_down = down;
}

(* Rotated, normalized queries [7,16,128] and keys [7,8,128]. *)
let queries (lw:layer_weights) (cos sin n:S.seq real) : S.seq real =
  rope 16 (rmsnorm_rows 112 128 (linear 7 2048 1024 lw.wq n) lw.q_norm) cos sin

let keys (lw:layer_weights) (cos sin n:S.seq real) : S.seq real =
  rope 8 (rmsnorm_rows 56 128 (linear 7 1024 1024 lw.wk n) lw.k_norm) cos sin

(* Masked scaled scores [7 query, 16 head, 7 key]; key j attends query t iff j <= t,
   otherwise the finite additive mask finfo(bf16).min is added (as the reference). *)
[@@"opaque_to_smt"]
let attn_scores (q k:S.seq real) : s:S.seq real{S.length s == 784} =
  S.init 784 (fun (i:nat{i < 784}) ->
    let t = i / 112 in
    let h = (i / 7) % 16 in
    let j = i % 7 in
    sdot 128 q ((t * 16 + h) * 128) 1 k ((j * 8 + h / 2) * 128) 1 *. attn_scale +.
    (if j > t then mask_min else 0.0R))

(* softmax over the 7 keys of each (query, head) row (shift free; a
   max-subtracted evaluation has the same real value). *)
[@@"opaque_to_smt"]
let softmax7 (s:S.seq real) : p:S.seq real{S.length p == S.length s} =
  S.init (S.length s) (fun (i:nat{i < S.length s}) ->
    Exp.exp (S.index s i) /. sumexp 7 s ((i / 7) * 7))

(* Attention output [7, 2048]: head h uses kv head h/2 (repeat_kv, 2 groups). *)
[@@"opaque_to_smt"]
let attn_values (p v:S.seq real) : s:S.seq real{S.length s == 14336} =
  S.init 14336 (fun (i:nat{i < 14336}) ->
    let t = i / 2048 in
    let h = (i % 2048) / 128 in
    let c = i % 128 in
    sdot 7 p ((t * 16 + h) * 7) 1 v ((h / 2) * 128 + c) 1024)

let vadd (a b:S.seq real) : s:S.seq real{S.length s == S.length a} =
  S.init (S.length a) (fun (i:nat{i < S.length a}) -> S.index a i +. at b i)

[@@"opaque_to_smt"]
let silu_product (g u:S.seq real) : s:S.seq real{S.length s == S.length g} =
  S.init (S.length g) (fun (i:nat{i < S.length g}) -> silu (S.index g i) *. at u i)

(* Self-attention block output (after o_proj) of the 7 positions. *)
let attention (lw:layer_weights) (cos sin n:S.seq real) : S.seq real =
  let q = queries lw cos sin n in
  let k = keys lw cos sin n in
  let v = linear 7 1024 1024 lw.wv n in
  linear 7 1024 2048 lw.wo (attn_values (softmax7 (attn_scores q k)) v)

(* Qwen3MLP of the post-attention normalized stream. *)
let mlp (lw:layer_weights) (x:S.seq real) : S.seq real =
  linear 7 1024 3072 lw.w_down
    (silu_product (linear 7 3072 1024 lw.w_gate x) (linear 7 3072 1024 lw.w_up x))

(* Qwen3DecoderLayer residual output, given residual r [7,1024] and its
   input-layernormed form n [7,1024]; cos/sin are [7,128] tables. *)
let layer_residual (lw:layer_weights) (cos sin r n:S.seq real) : S.seq real =
  let h = vadd r (attention lw cos sin n) in
  vadd h (mlp lw (rmsnorm_rows 7 1024 h lw.post_norm))

(* The reference layer: input_layernorm computed from the residual. *)
let qwen_layer (lw:layer_weights) (input_norm cos sin h:S.seq real) : S.seq real =
  layer_residual lw cos sin h (rmsnorm_rows 7 1024 h input_norm)

(* Rotary tables: rows are positions p0 .. p0+n-1, columns 0..127 use
   frequency index c mod 64 (cat(freqs, freqs)). *)
[@@"opaque_to_smt"]
let rope_cos_table (p0:int) (n:nat) : s:S.seq real{S.length s == n * 128} =
  S.init (n * 128) (fun (i:nat{i < n * 128}) -> Trig.cos (rope_angle (p0 + i / 128) ((i % 128) % 64)))

[@@"opaque_to_smt"]
let rope_sin_table (p0:int) (n:nat) : s:S.seq real{S.length s == n * 128} =
  S.init (n * 128) (fun (i:nat{i < n * 128}) -> Trig.sin (rope_angle (p0 + i / 128) ((i % 128) % 64)))

(* [2, 7, 128]: cos then sin for positions p0 .. p0+6. *)
let rope_pair (p0:int) : S.seq real = S.append (rope_cos_table p0 7) (rope_sin_table p0 7)

(* Token id of position t; positions past the prompt read token 0. *)
let tok_at (tokens:S.seq I64.t) (t:int) : int =
  if 0 <= t && t < S.length tokens then I64.v (S.index tokens t) else 0

let tokens_ok (tokens:S.seq I64.t) : prop =
  forall (i:nat). i < S.length tokens ==>
    0 <= I64.v (S.index tokens i) /\ I64.v (S.index tokens i) < vocab_size

(* embed_tokens: rows of the [151936,1024] table selected by the tokens. *)
[@@"opaque_to_smt"]
let embed_rows (rows:nat) (we:S.seq real) (tokens:S.seq I64.t)
  : s:S.seq real{S.length s == rows * 1024} =
  S.init (rows * 1024) (fun (i:nat{i < rows * 1024}) ->
    at we (tok_at tokens (i / 1024) * 1024 + i % 1024))

(* Tied LM head on one hidden row x [1024]: cols >= 151936 are zero padding. *)
[@@"opaque_to_smt"]
let logits_row (cols:nat) (we x:S.seq real) : s:S.seq real{S.length s == cols} =
  S.init cols (fun (v:nat{v < cols}) ->
    if v < vocab_size then sdot 1024 we (v * 1024) 1 x 0 1 else 0.0R)

(* Residual stream after l decoder layers of the seven-token prompt. *)
let rec hidden_after (rw:S.seq real) (tokens:S.seq I64.t) (l:nat)
  : Tot (S.seq real) (decreases l)
  = if l = 0 then embed_rows 7 (window rw 0 embed_elements) tokens
    else
      let off = layer_offset (l - 1) in
      qwen_layer (packed_layer rw off) (window rw off 1024)
        (rope_cos_table 0 7) (rope_sin_table 0 7) (hidden_after rw tokens (l - 1))

(* hidden_states[-1] = model.norm(last layer output), [7,1024]. *)
let final_hidden (rw:S.seq real) (tokens:S.seq I64.t) : S.seq real =
  rmsnorm_rows 7 1024 (hidden_after rw tokens num_layers) (window rw final_norm_offset 1024)

(* logits[:, -1, :] of the tied LM head. *)
let next_token_logits (rw:S.seq real) (tokens:S.seq I64.t) : S.seq real =
  logits_row vocab_size (window rw 0 embed_elements) (window (final_hidden rw tokens) 6144 1024)

(* ------------------------------------------------------------------ *)
(* Selection over a native BF16 score row (comparisons are native, L1)  *)

let is_nan (x:BF.t) : prop = F.NaN? (F.kind x)

let has_nan (s:S.seq BF.t) : prop = exists (i:nat). i < S.length s /\ is_nan (S.index s i)

(* Allowed choices of argmax (torch.argmax semantics up to tie order):
   if the row has a NaN, the first NaN; otherwise any index whose value is
   >= every value (native order; +0 and -0 tie). *)
let argmax_choice (s:S.seq BF.t) (j:int) : prop =
  0 <= j /\ j < S.length s /\
  (has_nan s ==> is_nan (S.index s j) /\ (forall (i:nat). i < j ==> ~(is_nan (S.index s i)))) /\
  (~(has_nan s) ==> (forall (i:nat). i < S.length s ==> F.lte (S.index s i) (S.index s j)))

(* Base select_token: the first maximum (earlier entries strictly smaller). *)
let first_argmax (s:S.seq BF.t) (j:int) : prop =
  argmax_choice s j /\
  (~(has_nan s) ==> (forall (i:nat). i < j ==> F.lt (S.index s i) (S.index s j)))

(* Optimized select_tokens tie order. Index j = 1600*c + 128*m + t (chunk c,
   lane t, stride step m). Every pairwise tree step keeps the lower slot on
   ties, so the winner among equal maxima is the one whose path prefers the
   lower slot at the latest steps: chunk and lane ranks read their slot bits
   from least significant (last step) to most significant (first step):
   chunk c = 32*w + l (8 final warps), lane t = 32*w + l (4 warps). *)
let rec rev_bits (n:nat) (x:nat) : Tot nat (decreases n) =
  if n = 0 then 0 else (x % 2) * pow2 (n - 1) + rev_bits (n - 1) (x / 2)

let opt_key (j:nat) : nat =
  let c = j / 1600 in
  let i = j % 1600 in
  let t = i % 128 in
  let m = i / 128 in
  ((rev_bits 3 (c / 32) * 32 + rev_bits 5 (c % 32)) * 128 +
   (rev_bits 2 (t / 32) * 32 + rev_bits 5 (t % 32))) * 13 + m

let tree_argmax (s:S.seq BF.t) (j:int) : prop =
  argmax_choice s j /\
  (~(has_nan s) ==>
    (forall (i:nat). i < S.length s /\ opt_key i < opt_key j ==> F.lt (S.index s i) (S.index s j)))

(* ------------------------------------------------------------------ *)
(* Program postcondition (role independent)                            *)

let program_postcondition (input0:S.seq BF.t) (input1:S.seq I64.t) (input2:int)
                          (output0:S.seq I64.t) (output1:S.seq BF.t) : prop =
  (S.length input0 == packed_elements /\ S.length input1 == prompt_length /\
   tokens_ok input1 /\ input2 == 1) ==>
  (S.length output0 == 8 /\ S.length output1 == 7168 /\
   (* structural: the prompt is copied in front of the generated token *)
   (forall (i:nat). i < 7 ==> S.index output0 i == S.index input1 i) /\
   (* numerical: final normalized hidden states *)
   (forall (rw:S.seq real). input0 %~ rw ==> output1 %~ final_hidden rw input1) /\
   (* selection over the score row actually consumed *)
   (exists (scores:S.seq BF.t).
      S.length scores == vocab_size /\
      (forall (rw:S.seq real). input0 %~ rw ==> scores %~ next_token_logits rw input1) /\
      argmax_choice scores (I64.v (S.index output0 7))))

(* ------------------------------------------------------------------ *)
(* Family relations: base                                              *)

let int_max : int = 2147483647

let base_embedding_relation (a0:S.seq BF.t) (a1:S.seq I64.t) (a2 a3 a4:int)
                            (o0 o1 o2:S.seq BF.t) : prop =
  (S.length a1 == 7 /\ tokens_ok a1 /\
   0 <= a2 /\ a2 <= int_max - embed_elements /\ a2 + embed_elements <= S.length a0 /\
   0 <= a3 /\ a3 <= int_max - 1024 /\ a3 + 1024 <= S.length a0 /\
   0 <= a4 /\ a4 <= int_max - 6) ==>
  (S.length o0 == 7168 /\ S.length o1 == 7168 /\ S.length o2 == 1792 /\
   (* residual rows are exact copies of the selected embedding rows *)
   (forall (t:nat). t < 7 ==> window o0 (t * 1024) 1024 == window a0 (a2 + tok_at a1 t * 1024) 1024) /\
   o2 %~ rope_pair a4 /\
   (forall (rw:S.seq real). a0 %~ rw ==>
      o0 %~ embed_rows 7 (window rw a2 embed_elements) a1 /\
      o1 %~ rmsnorm_rows 7 1024 (embed_rows 7 (window rw a2 embed_elements) a1) (window rw a3 1024)))

let base_decoder_relation (a0 a1 a2 a3:S.seq BF.t) (a4 a5:int) (o0 o1:S.seq BF.t) : prop =
  (0 <= a4 /\ a4 <= int_max - layer_elements /\ a4 + layer_elements <= S.length a0 /\
   0 <= a5 /\ a5 <= int_max - 1024 /\ a5 + 1024 <= S.length a0 /\
   S.length a1 == 7168 /\ S.length a2 == 7168 /\ S.length a3 == 1792) ==>
  (S.length o0 == 7168 /\ S.length o1 == 7168 /\
   (forall (rw rr rn rrot:S.seq real).
      a0 %~ rw /\ a1 %~ rr /\ a2 %~ rn /\ a3 %~ rrot ==>
      (let res = layer_residual (packed_layer rw a4) (window rrot 0 896) (window rrot 896 896) rr rn in
       o0 %~ res /\ o1 %~ rmsnorm_rows 7 1024 res (window rw a5 1024))))

let base_projection_relation (a0 a1:S.seq BF.t) (a2:int) (o0:S.seq BF.t) : prop =
  (0 <= a2 /\ a2 <= int_max - embed_elements /\ a2 + embed_elements <= S.length a0 /\
   S.length a1 == 7168) ==>
  (S.length o0 == 7 * vocab_size /\
   (forall (rw rh:S.seq real). a0 %~ rw /\ a1 %~ rh ==>
      (forall (t:nat). t < 7 ==>
         window o0 (t * vocab_size) vocab_size %~
         logits_row vocab_size (window rw a2 embed_elements) (window rh (t * 1024) 1024))))

let base_argmax_relation (a0:S.seq I64.t) (a1:S.seq BF.t) (a2:int) (o0 o1:S.seq I64.t) : prop =
  (S.length a0 <= int_max /\ 0 < S.length a1 /\ S.length a1 <= int_max /\ a2 == 1) ==>
  (S.length o0 == S.length a0 + 1 /\ S.length o1 == 1 /\
   (forall (i:nat). i < S.length a0 ==> S.index o0 i == S.index a0 i) /\
   S.index o0 (S.length a0) == S.index o1 0 /\
   first_argmax a1 (I64.v (S.index o1 0)))

(* ------------------------------------------------------------------ *)
(* Family relations: optimized                                         *)

let optimized_embedding_relation (a0:S.seq BF.t) (a1:S.seq I64.t) (a2:S.seq BF.t)
                                 (o0 o1 o2 o3:S.seq BF.t) : prop =
  (S.length a0 == embed_elements /\ S.length a1 == 7 /\ tokens_ok a1 /\ S.length a2 == 1024) ==>
  (S.length o0 == 8192 /\ S.length o1 == 8192 /\ S.length o2 == 524288 /\ S.length o3 == 524288 /\
   (* rows 0..6 copy the prompt embeddings; padded row 7 copies token 0 *)
   (forall (t:nat). t < 8 ==> window o0 (t * 1024) 1024 == window a0 (tok_at a1 t * 1024) 1024) /\
   o2 %~ rope_cos_table 0 4096 /\ o3 %~ rope_sin_table 0 4096 /\
   (forall (r0 r2:S.seq real). a0 %~ r0 /\ a2 %~ r2 ==>
      window o0 0 7168 %~ embed_rows 7 r0 a1 /\
      window o0 7168 1024 %~ window r0 0 1024 /\
      window o1 0 7168 %~ rmsnorm_rows 7 1024 (embed_rows 7 r0 a1) r2 /\
      window o1 7168 1024 %~ rmsnorm_rows 1 1024 (window r0 0 1024) r2))

let optimized_decoder_relation (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10:S.seq BF.t) (a11:int)
                               (o0 o1:S.seq BF.t) : prop =
  (S.length a0 == 4194304 /\ S.length a1 == 256 /\ S.length a2 == 2097152 /\
   S.length a3 == 1024 /\ S.length a4 == 6291456 /\ S.length a5 == 3145728 /\
   S.length a6 == 1024 /\ S.length a7 == 8192 /\ S.length a8 == 8192 /\
   S.length a9 == 524288 /\ S.length a10 == 524288 /\ 0 <= a11 /\ a11 <= 4088) ==>
  (S.length o0 == 8192 /\ S.length o1 == 8192 /\
   (forall (r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10:S.seq real).
      a0 %~ r0 /\ a1 %~ r1 /\ a2 %~ r2 /\ a3 %~ r3 /\ a4 %~ r4 /\ a5 %~ r5 /\
      a6 %~ r6 /\ a7 %~ r7 /\ a8 %~ r8 /\ a9 %~ r9 /\ a10 %~ r10 ==>
      (let res = layer_residual (split_layer r0 r1 r2 r3 r4 r5)
                   (window r9 (a11 * 128) 896) (window r10 (a11 * 128) 896)
                   (window r7 0 7168) (window r8 0 7168) in
       let pad = window r7 7168 1024 in
       (* the seven prompt rows *)
       window o0 0 7168 %~ res /\
       window o1 0 7168 %~ rmsnorm_rows 7 1024 res r6 /\
       (* the padded row 7 receives zero attention and MLP updates *)
       window o0 7168 1024 %~ pad /\
       window o1 7168 1024 %~ rmsnorm_rows 1 1024 pad r6)))

let optimized_projection_relation (a0 a1:S.seq BF.t) (o0 o1:S.seq BF.t) : prop =
  (S.length a0 == embed_elements /\ S.length a1 == 8192) ==>
  (S.length o0 == 8 * 153600 /\ o1 == window a1 0 7168 /\
   (forall (r0 r1:S.seq real). a0 %~ r0 /\ a1 %~ r1 ==>
      (forall (t:nat). t < 8 ==>
         window o0 (t * 153600) 153600 %~ logits_row 153600 r0 (window r1 (t * 1024) 1024))))

let optimized_argmax_relation (a0:S.seq BF.t) (a1:S.seq I64.t) (o0 o1:S.seq I64.t) : prop =
  (S.length a0 == vocab_size /\ S.length a1 == 7) ==>
  (S.length o0 == 1 /\ S.length o1 == 8 /\
   (forall (i:nat). i < 7 ==> S.index o1 i == S.index a1 i) /\
   S.index o1 7 == S.index o0 0 /\
   tree_argmax a0 (I64.v (S.index o0 0)))

(* ------------------------------------------------------------------ *)
(* Structural helper lemmas for the composition proofs                 *)

let window_window (#a:Type) (s:S.seq a) (o n o' n':int)
  : Lemma (requires 0 <= o /\ 0 <= n /\ o + n <= S.length s /\ 0 <= o' /\ 0 <= n' /\ o' + n' <= n)
          (ensures window (window s o n) o' n' == window s (o + o') n')
  = S.lemma_eq_intro (window (window s o n) o' n') (window s (o + o') n')

let window_full (#a:Type) (s:S.seq a)
  : Lemma (window s 0 (S.length s) == s)
  = S.lemma_eq_intro (window s 0 (S.length s)) s

(* The base decoder's cos/sin views of the [2,7,128] rotary output. *)
let rope_pair_windows (p0:int)
  : Lemma (window (rope_pair p0) 0 896 == rope_cos_table p0 7 /\
           window (rope_pair p0) 896 896 == rope_sin_table p0 7)
  = S.lemma_eq_intro (window (rope_pair p0) 0 896) (rope_cos_table p0 7);
    S.lemma_eq_intro (window (rope_pair p0) 896 896) (rope_sin_table p0 7)

(* The optimized decoder's 7-position views of the 4096-position tables. *)
let rope_table_windows (n:nat{7 <= n})
  : Lemma (window (rope_cos_table 0 n) 0 896 == rope_cos_table 0 7 /\
           window (rope_sin_table 0 n) 0 896 == rope_sin_table 0 7)
  = reveal_opaque (`%rope_cos_table) rope_cos_table;
    reveal_opaque (`%rope_sin_table) rope_sin_table;
    S.lemma_eq_intro (window (rope_cos_table 0 n) 0 896) (rope_cos_table 0 7);
    S.lemma_eq_intro (window (rope_sin_table 0 n) 0 896) (rope_sin_table 0 7)

(* Every allowed selection of either kernel is an allowed program choice. *)
let first_argmax_choice (s:S.seq BF.t) (j:int)
  : Lemma (requires first_argmax s j) (ensures argmax_choice s j) = ()

let tree_argmax_choice (s:S.seq BF.t) (j:int)
  : Lemma (requires tree_argmax s j) (ensures argmax_choice s j) = ()
