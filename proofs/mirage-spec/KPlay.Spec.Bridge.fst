module KPlay.Spec.Bridge

open FStar.Real
open Kuiper.Approximates
open KPlay.Spec
module S = FStar.Seq
module BF = Kuiper.BFloat16
module I64 = FStar.Int64
module M = FStar.Math.Lemmas
module SS = KPlay.SharedSpec
module Exp = FStar.Math.Exp
module Sqrt = FStar.Math.Sqrt
module Pow = FStar.Math.Pow
module Trig = Kuiper.Real.Trigonometry
module N = KPlay.Numbers

(* Index arithmetic *)

let split (r c:int) (d:pos)
  : Lemma (requires 0 <= c /\ c < d) (ensures (r * d + c) / d == r /\ (r * d + c) % d == c)
  = M.lemma_div_plus c r d; M.small_div c d; M.lemma_mod_plus c r d; M.small_mod c d

let bound (r c:nat) (rows cols:nat)
  : Lemma (requires r < rows /\ c < cols) (ensures r * cols + c < rows * cols)
  = M.lemma_mult_le_right cols (r + 1) rows; M.distributivity_add_left r 1 cols

(* Pointwise relations between flat real tensors and typed values *)

let vec_eq (s:S.seq real) (o:int) (#n:nat) (x:vector real n) : prop =
  forall (i:index n). SS.at s (o + i) == x i

let mat_eq (s:S.seq real) (rows cols:nat) (x:matrix real rows cols) : prop =
  S.length s == rows * cols /\
  (forall (r:index rows) (c:index cols). SS.at s (r * cols + c) == x r c)

(* Sums *)

let rec sdot_prefix (k:nat) (a:S.seq real) (ao sa:int) (b:S.seq real) (bo sb:int)
    (#n:nat) (f:vector real n)
  : Lemma (requires k <= n /\
                    (forall (i:index n). i < k ==> f i == SS.at a (ao + i * sa) *. SS.at b (bo + i * sb)))
          (ensures SS.sdot k a ao sa b bo sb == sum_prefix f k)
          (decreases k)
  = if k > 0 then sdot_prefix (k - 1) a ao sa b bo sb f

let sdot_dot (n:nat) (a:S.seq real) (ao sa:int) (b:S.seq real) (bo sb:int) (x y:vector real n)
  : Lemma (requires forall (i:index n). SS.at a (ao + i * sa) == x i /\ SS.at b (bo + i * sb) == y i)
          (ensures SS.sdot n a ao sa b bo sb == dot x y)
  = sdot_prefix n a ao sa b bo sb (fun (i:index n) -> x i *. y i)

(* Operations *)

let linear_sim (rows outs inner:nat) (w x:S.seq real) (wm:matrix real outs inner) (xm:matrix real rows inner)
  : Lemma (requires mat_eq w outs inner wm /\ mat_eq x rows inner xm)
          (ensures mat_eq (SS.linear rows outs inner w x) rows outs (linear_rows wm xm))
  = reveal_opaque (`%SS.linear) (SS.linear rows outs inner w x);
    let aux (r:index rows) (o:index outs)
      : Lemma (SS.at (SS.linear rows outs inner w x) (r * outs + o) == linear_rows wm xm r o)
      = bound r o rows outs; split r o outs;
        sdot_dot inner w (o * inner) 1 x (r * inner) 1 (wm o) (xm r)
    in
    Classical.forall_intro_2 aux

let rec sumsq_prefix (k:nat) (x:S.seq real) (o:int) (#n:nat) (f:vector real n)
  : Lemma (requires k <= n /\ (forall (i:index n). i < k ==> f i == SS.at x (o + i) *. SS.at x (o + i)))
          (ensures SS.sumsq k x o == sum_prefix f k)
          (decreases k)
  = if k > 0 then sumsq_prefix (k - 1) x o f

let rec sumexp_prefix (k:nat) (x:S.seq real) (o:int) (#n:nat) (f:vector real n)
  : Lemma (requires k <= n /\ (forall (i:index n). i < k ==> f i == Exp.exp (SS.at x (o + i))))
          (ensures SS.sumexp k x o == sum_prefix f k)
          (decreases k)
  = if k > 0 then sumexp_prefix (k - 1) x o f

let rms_row (d:pos) (x w:S.seq real) (o c:int) (xr wv:vector real d)
  : Lemma (requires 0 <= c /\ c < d /\ (forall (i:index d). SS.at x (o + i) == xr i) /\ vec_eq w 0 wv)
          (ensures SS.at w c *. (SS.at x (o + c) *. SS.rms_inv d x o) == rms_norm wv xr c)
  = sumsq_prefix d x o (fun (i:index d) -> xr i *. xr i)

let rmsnorm_hidden (x w:S.seq real) (xm:hidden real) (wv:vector real hidden_size)
  : Lemma (requires mat_eq x 7 1024 xm /\ vec_eq w 0 wv)
          (ensures mat_eq (SS.rmsnorm_rows 7 1024 x w) 7 1024 (normalize_hidden wv xm))
  = reveal_opaque (`%SS.rmsnorm_rows) (SS.rmsnorm_rows 7 1024 x w);
    let aux (r:index 7) (c:index 1024)
      : Lemma (SS.at (SS.rmsnorm_rows 7 1024 x w) (r * 1024 + c) == normalize_hidden wv xm r c)
      = bound r c 7 1024; split r c 1024;
        rms_row 1024 x w (r * 1024) c (xm r) wv
    in
    Classical.forall_intro_2 aux

let heads_eq (s:S.seq real) (count:nat) (x:heads real count) : prop =
  S.length s == 7 * count * 128 /\
  (forall (t:index 7) (h:index count) (c:index 128). SS.at s ((t * count + h) * 128 + c) == x t h c)

let heads_index (t h c:nat) (count:pos)
  : Lemma (requires t < 7 /\ h < count /\ c < 128)
          (ensures (t * count + h) * 128 + c == t * (count * 128) + (h * 128 + c) /\
                   h * 128 + c < count * 128 /\
                   (t * count + h) * 128 + c < 7 * count * 128 /\
                   t * count + h < 7 * count)
  = M.distributivity_add_left (t * count) h 128; M.paren_mul_right t count 128;
    bound h c count 128; bound t h 7 count; bound (t * count + h) c (7 * count) 128

let split_heads_eq (s:S.seq real) (count:pos) (m:matrix real 7 (count * 128))
  : Lemma (requires mat_eq s 7 (count * 128) m)
          (ensures heads_eq s count (split_heads m))
  = M.paren_mul_right 7 count 128;
    let aux (t:index 7) (h:index count) (c:index 128)
      : Lemma (SS.at s ((t * count + h) * 128 + c) == split_heads m t h c)
      = heads_index t h c count
    in
    Classical.forall_intro_3 aux

let rmsnorm_heads (count:pos) (x w:S.seq real) (xh:heads real count) (wv:vector real head_dim)
  : Lemma (requires heads_eq x count xh /\ vec_eq w 0 wv)
          (ensures heads_eq (SS.rmsnorm_rows (7 * count) 128 x w) count (normalize_heads wv xh))
  = reveal_opaque (`%SS.rmsnorm_rows) (SS.rmsnorm_rows (7 * count) 128 x w);
    M.paren_mul_right 7 count 128;
    let aux (t:index 7) (h:index count) (c:index 128)
      : Lemma (SS.at (SS.rmsnorm_rows (7 * count) 128 x w) ((t * count + h) * 128 + c) ==
               normalize_heads wv xh t h c)
      = heads_index t h c count;
        split (t * count + h) c 128;
        rms_row 128 x w ((t * count + h) * 128) c (xh t h) wv
    in
    Classical.forall_intro_3 aux

let rope_sim (nh:pos) (x cos sin:S.seq real) (xh:heads real nh) (rt:rotary_tables real)
  : Lemma (requires heads_eq x nh xh /\ mat_eq cos 7 128 rt.cos /\ mat_eq sin 7 128 rt.sin)
          (ensures heads_eq (SS.rope nh x cos sin) nh (apply_rotary_pos_emb rt xh))
  = reveal_opaque (`%SS.rope) (SS.rope nh x cos sin);
    let aux (t:index 7) (h:index nh) (c:index 128)
      : Lemma (SS.at (SS.rope nh x cos sin) ((t * nh + h) * 128 + c) == apply_rotary_pos_emb rt xh t h c)
      = heads_index t h c nh;
        split t (h * 128 + c) (nh * 128);
        split (t * nh + h) c 128;
        if c < 64 then heads_index t h (c + 64) nh else heads_index t h (c - 64) nh
    in
    Classical.forall_intro_3 aux

let scores_eq (s:S.seq real) (a:attention_map) : prop =
  S.length s == 784 /\
  (forall (t:index 7) (h:index 16) (j:index 7). SS.at s ((t * 16 + h) * 7 + j) == a t h j)

let score_index (t h j:nat)
  : Lemma (requires t < 7 /\ h < 16 /\ j < 7)
          (ensures (let i = (t * 16 + h) * 7 + j in
                    i < 784 /\ i / 112 == t /\ (i / 7) % 16 == h /\ i % 7 == j /\ (i / 7) * 7 == (t * 16 + h) * 7))
  = bound h j 16 7; split t (h * 7 + j) 112; split (t * 16 + h) j 7; split t h 16

let scores_sim (q k:S.seq real) (qh:heads real 16) (kh:heads real 8)
  : Lemma (requires heads_eq q 16 qh /\ heads_eq k 8 kh)
          (ensures scores_eq (SS.attn_scores q k) (attention_scores qh (repeat_kv kh)))
  = reveal_opaque (`%SS.attn_scores) (SS.attn_scores q k);
    let aux (t:index 7) (h:index 16) (j:index 7)
      : Lemma (SS.at (SS.attn_scores q k) ((t * 16 + h) * 7 + j) == attention_scores qh (repeat_kv kh) t h j)
      = score_index t h j;
        sdot_dot 128 q ((t * 16 + h) * 128) 1 k ((j * 8 + h / 2) * 128) 1 (qh t h) (repeat_kv kh j h)
    in
    Classical.forall_intro_3 aux

let softmax_row (n:pos) (s:S.seq real) (o:int) (j:index n) (xr:vector real n)
  : Lemma (requires forall (i:index n). SS.at s (o + i) == xr i)
          (ensures Exp.exp (SS.at s (o + j)) /. SS.sumexp n s o == softmax xr j)
  = sumexp_prefix n s o (fun (i:index n) -> Exp.exp (xr i))

let weights_sim (q k:S.seq real) (qh:heads real 16) (kh:heads real 8)
  : Lemma (requires heads_eq q 16 qh /\ heads_eq k 8 kh)
          (ensures scores_eq (SS.softmax7 (SS.attn_scores q k)) (attention_weights qh (repeat_kv kh)))
  = scores_sim q k qh kh;
    let s = SS.attn_scores q k in
    reveal_opaque (`%SS.softmax7) (SS.softmax7 s);
    let aux (t:index 7) (h:index 16) (j:index 7)
      : Lemma (SS.at (SS.softmax7 s) ((t * 16 + h) * 7 + j) == attention_weights qh (repeat_kv kh) t h j)
      = score_index t h j;
        softmax_row 7 s ((t * 16 + h) * 7) j (attention_scores qh (repeat_kv kh) t h)
    in
    Classical.forall_intro_3 aux

let values_entry (p v:S.seq real) (w:attention_map) (vh:heads real 16) (t:index 7) (h:index 16) (c:index 128)
  : Lemma (requires scores_eq p w /\
                    (forall (j:index 7). SS.at v ((h / 2) * 128 + c + j * 1024) == vh j h c))
          (ensures SS.sdot 7 p ((t * 16 + h) * 7) 1 v ((h / 2) * 128 + c) 1024 == attention_output w vh t h c)
  = sdot_prefix 7 p ((t * 16 + h) * 7) 1 v ((h / 2) * 128 + c) 1024
      (fun (j:index prompt_length) -> w t h j *. vh j h c)

let values_sim (p v:S.seq real) (w:attention_map) (vm:matrix real 7 1024)
  : Lemma (requires scores_eq p w /\ mat_eq v 7 1024 vm)
          (ensures mat_eq (SS.attn_values p v) 7 2048
                     (merge_heads (attention_output w (repeat_kv (split_heads #real #8 vm)))))
  = reveal_opaque (`%SS.attn_values) (SS.attn_values p v);
    let vh = repeat_kv (split_heads #real #8 vm) in
    let aux (t:index 7) (ch:index 2048)
      : Lemma (SS.at (SS.attn_values p v) (t * 2048 + ch) == merge_heads (attention_output w vh) t ch)
      = bound t ch 7 2048; split t ch 2048; M.lemma_mod_plus ch (t * 16) 128;
        let h : index 16 = ch / 128 in
        let c : index 128 = ch % 128 in
        let req (j:index 7) : Lemma (SS.at v ((h / 2) * 128 + c + j * 1024) == vh j h c)
          = bound j ((h / 2) * 128 + c) 7 1024
        in
        Classical.forall_intro req;
        values_entry p v w vh t h c
    in
    Classical.forall_intro_2 aux

let vadd_sim (a b:S.seq real) (x y:hidden real)
  : Lemma (requires mat_eq a 7 1024 x /\ mat_eq b 7 1024 y)
          (ensures mat_eq (SS.vadd a b) 7 1024 (residual_add x y))
  = let aux (t:index 7) (c:index 1024)
      : Lemma (SS.at (SS.vadd a b) (t * 1024 + c) == residual_add x y t c)
      = bound t c 7 1024
    in
    Classical.forall_intro_2 aux

let mlp_tail (gs us wd:S.seq real) (g u:matrix real 7 3072) (pd:matrix real 1024 3072)
  : Lemma (requires mat_eq gs 7 3072 g /\ mat_eq us 7 3072 u /\ mat_eq wd 1024 3072 pd)
          (ensures mat_eq (SS.linear 7 1024 3072 wd (SS.silu_product gs us)) 7 1024
                     (linear_rows pd (fun (t:index prompt_length) (i:index intermediate_size) ->
                                        silu (g t i) *. u t i)))
  = reveal_opaque (`%SS.silu_product) (SS.silu_product gs us);
    let aux (t:index 7) (i:index 3072)
      : Lemma (SS.at (SS.silu_product gs us) (t * 3072 + i) == silu (g t i) *. u t i)
      = bound t i 7 3072
    in
    Classical.forall_intro_2 aux;
    linear_sim 7 1024 3072 wd (SS.silu_product gs us) pd
      (fun (t:index prompt_length) (i:index intermediate_size) -> silu (g t i) *. u t i)

let mlp_sim (lw:SS.layer_weights) (p:parameters real) (x:S.seq real) (xm:hidden real)
  : Lemma (requires mat_eq lw.SS.w_gate 3072 1024 p.gate_proj /\ mat_eq lw.SS.w_up 3072 1024 p.up_proj /\
                    mat_eq lw.SS.w_down 1024 3072 p.down_proj /\ mat_eq x 7 1024 xm)
          (ensures mat_eq (SS.mlp lw x) 7 1024 (mlp p xm))
  = linear_sim 7 3072 1024 lw.SS.w_gate x p.gate_proj xm;
    linear_sim 7 3072 1024 lw.SS.w_up x p.up_proj xm;
    mlp_tail (SS.linear 7 3072 1024 lw.SS.w_gate x) (SS.linear 7 3072 1024 lw.SS.w_up x) lw.SS.w_down
      (linear_rows p.gate_proj xm) (linear_rows p.up_proj xm) p.down_proj

let rope_eq (cos sin:S.seq real) (rt:rotary_tables real) : prop =
  mat_eq cos 7 128 rt.cos /\ mat_eq sin 7 128 rt.sin

let attention_sim (lw:SS.layer_weights) (p:parameters real) (cos sin n:S.seq real)
    (rt:rotary_tables real) (nm:hidden real)
  : Lemma (requires mat_eq lw.SS.wq 2048 1024 p.q_proj /\ mat_eq lw.SS.wk 1024 1024 p.k_proj /\
                    mat_eq lw.SS.wv 1024 1024 p.v_proj /\ mat_eq lw.SS.wo 1024 2048 p.o_proj /\
                    vec_eq lw.SS.q_norm 0 p.q_norm /\ vec_eq lw.SS.k_norm 0 p.k_norm /\
                    rope_eq cos sin rt /\ mat_eq n 7 1024 nm)
          (ensures mat_eq (SS.attention lw cos sin n) 7 1024 (self_attn p rt nm))
  = let ql = SS.linear 7 2048 1024 lw.SS.wq n in
    let kl = SS.linear 7 1024 1024 lw.SS.wk n in
    let v = SS.linear 7 1024 1024 lw.SS.wv n in
    let qm = linear_rows p.q_proj nm in
    let km = linear_rows p.k_proj nm in
    let vm = linear_rows p.v_proj nm in
    linear_sim 7 2048 1024 lw.SS.wq n p.q_proj nm;
    linear_sim 7 1024 1024 lw.SS.wk n p.k_proj nm;
    linear_sim 7 1024 1024 lw.SS.wv n p.v_proj nm;
    split_heads_eq ql 16 qm;
    split_heads_eq kl 8 km;
    let qn = normalize_heads p.q_norm (split_heads #real #num_attention_heads qm) in
    let kn = normalize_heads p.k_norm (split_heads #real #num_key_value_heads km) in
    rmsnorm_heads 16 ql lw.SS.q_norm (split_heads #real #num_attention_heads qm) p.q_norm;
    rmsnorm_heads 8 kl lw.SS.k_norm (split_heads #real #num_key_value_heads km) p.k_norm;
    let q = SS.rope 16 (SS.rmsnorm_rows 112 128 ql lw.SS.q_norm) cos sin in
    let k = SS.rope 8 (SS.rmsnorm_rows 56 128 kl lw.SS.k_norm) cos sin in
    rope_sim 16 (SS.rmsnorm_rows 112 128 ql lw.SS.q_norm) cos sin qn rt;
    rope_sim 8 (SS.rmsnorm_rows 56 128 kl lw.SS.k_norm) cos sin kn rt;
    let qh = apply_rotary_pos_emb rt qn in
    let kh = apply_rotary_pos_emb rt kn in
    weights_sim q k qh kh;
    let pw = SS.softmax7 (SS.attn_scores q k) in
    values_sim pw v (attention_weights qh (repeat_kv kh)) vm;
    linear_sim 7 1024 2048 lw.SS.wo (SS.attn_values pw v) p.o_proj
      (merge_heads (attention_output (attention_weights qh (repeat_kv kh))
                                     (repeat_kv (split_heads #real #num_key_value_heads vm))))

let layer_eq (lw:SS.layer_weights) (inorm:S.seq real) (p:parameters real) : prop =
  vec_eq inorm 0 p.input_layernorm /\
  vec_eq lw.SS.post_norm 0 p.post_attention_layernorm /\
  mat_eq lw.SS.wq 2048 1024 p.q_proj /\
  mat_eq lw.SS.wk 1024 1024 p.k_proj /\
  mat_eq lw.SS.wv 1024 1024 p.v_proj /\
  mat_eq lw.SS.wo 1024 2048 p.o_proj /\
  vec_eq lw.SS.q_norm 0 p.q_norm /\
  vec_eq lw.SS.k_norm 0 p.k_norm /\
  mat_eq lw.SS.w_gate 3072 1024 p.gate_proj /\
  mat_eq lw.SS.w_up 3072 1024 p.up_proj /\
  mat_eq lw.SS.w_down 1024 3072 p.down_proj

let layer_sim (lw:SS.layer_weights) (inorm:S.seq real) (p:parameters real) (cos sin:S.seq real)
    (rt:rotary_tables real) (h:S.seq real) (x:hidden real)
  : Lemma (requires layer_eq lw inorm p /\ rope_eq cos sin rt /\ mat_eq h 7 1024 x)
          (ensures mat_eq (SS.qwen_layer lw inorm cos sin h) 7 1024 (decoder_layer p rt x))
  = let n = SS.rmsnorm_rows 7 1024 h inorm in
    let xn = normalize_hidden p.input_layernorm x in
    rmsnorm_hidden h inorm x p.input_layernorm;
    attention_sim lw p cos sin n rt xn;
    let a = SS.attention lw cos sin n in
    vadd_sim h a x (self_attn p rt xn);
    let hh = SS.vadd h a in
    let after = residual_add x (self_attn p rt xn) in
    rmsnorm_hidden hh lw.SS.post_norm after p.post_attention_layernorm;
    let n2 = SS.rmsnorm_rows 7 1024 hh lw.SS.post_norm in
    mlp_sim lw p n2 (normalize_hidden p.post_attention_layernorm after);
    vadd_sim hh (SS.mlp lw n2) after (mlp p (normalize_hidden p.post_attention_layernorm after))

(* ------------------------------------------------------------------ *)
(* Views: the typed model values of the flat program buffers (layout only) *)

let elem (#a:Type) (d:a) (s:S.seq a) (i:int) : a =
  if 0 <= i && i < S.length s then S.index s i else d

let vector_view (#a:Type0) (d:a) (s:S.seq a) (o:int) (n:nat) : vector a n =
  fun (i:index n) -> elem d s (o + i)

let matrix_view (#a:Type0) (d:a) (s:S.seq a) (o:int) (rows cols:nat) : matrix a rows cols =
  fun (r:index rows) (c:index cols) -> elem d s (o + (r * cols + c))

(* model_abi.py:24-41 and WEIGHT_LAYOUT: per-layer tensors in state-dict order. *)
let parameters_view (#a:Type0) (d:a) (s:S.seq a) (o:int) : parameters a = {
  input_layernorm = vector_view d s o 1024;
  post_attention_layernorm = vector_view d s (o + 1024) 1024;
  q_proj = matrix_view d s (o + 2048) 2048 1024;
  k_proj = matrix_view d s (o + 2099200) 1024 1024;
  v_proj = matrix_view d s (o + 3147776) 1024 1024;
  o_proj = matrix_view d s (o + 4196352) 1024 2048;
  q_norm = vector_view d s (o + 6293504) 128;
  k_norm = vector_view d s (o + 6293632) 128;
  gate_proj = matrix_view d s (o + 6293760) 3072 1024;
  up_proj = matrix_view d s (o + 9439488) 3072 1024;
  down_proj = matrix_view d s (o + 12585216) 1024 3072;
}

(* embed_tokens [151936, 1024], norm [1024], then the 28 layers. *)
let weights_view (#a:Type0) (d:a) (s:S.seq a) : model_weights a = {
  embed_tokens = matrix_view d s 0 vocab_size hidden_size;
  norm = vector_view d s SS.final_norm_offset hidden_size;
  layers = (fun (l:index num_hidden_layers) -> parameters_view d s (SS.layer_offset l));
}

let bf_zero : BF.t = Kuiper.Scalars.Base.zero

let inputs_of (packed:S.seq BF.t) (prompt:S.seq I64.t) (decode_steps:int) : inputs = {
  weights = weights_view bf_zero packed;
  prompt_tokens = (fun (t:index prompt_length) -> SS.tok_at prompt t);
  decode_steps = decode_steps;
}

let outputs_of (tokens:S.seq I64.t) (hidden_rows:S.seq BF.t) : outputs = {
  tokens = (fun (t:index sequence_length) -> SS.tok_at tokens t);
  hidden_states = matrix_view bf_zero hidden_rows 0 prompt_length hidden_size;
}

(* ------------------------------------------------------------------ *)
(* Flat real weights correspond to typed real weights                  *)

let params_eq (a b:parameters real) : prop =
  (forall (i:index 1024). a.input_layernorm i == b.input_layernorm i) /\
  (forall (i:index 1024). a.post_attention_layernorm i == b.post_attention_layernorm i) /\
  (forall (r:index 2048) (c:index 1024). a.q_proj r c == b.q_proj r c) /\
  (forall (r:index 1024) (c:index 1024). a.k_proj r c == b.k_proj r c) /\
  (forall (r:index 1024) (c:index 1024). a.v_proj r c == b.v_proj r c) /\
  (forall (r:index 1024) (c:index 2048). a.o_proj r c == b.o_proj r c) /\
  (forall (i:index 128). a.q_norm i == b.q_norm i) /\
  (forall (i:index 128). a.k_norm i == b.k_norm i) /\
  (forall (r:index 3072) (c:index 1024). a.gate_proj r c == b.gate_proj r c) /\
  (forall (r:index 3072) (c:index 1024). a.up_proj r c == b.up_proj r c) /\
  (forall (r:index 1024) (c:index 3072). a.down_proj r c == b.down_proj r c)

let corresponds (rw:S.seq real) (w:model_weights real) : prop =
  let v = weights_view 0.0R rw in
  S.length rw == SS.packed_elements /\
  (forall (t:index vocab_size) (c:index hidden_size). v.embed_tokens t c == w.embed_tokens t c) /\
  (forall (c:index hidden_size). v.norm c == w.norm c) /\
  (forall (l:index num_hidden_layers). params_eq (v.layers l) (w.layers l))

let window_mat (s:S.seq real) (o:nat) (rows cols:nat) (m:matrix real rows cols)
  : Lemma (requires o + rows * cols <= S.length s /\
                    (forall (r:index rows) (c:index cols). matrix_view 0.0R s o rows cols r c == m r c))
          (ensures mat_eq (SS.window s o (rows * cols)) rows cols m)
  = let aux (r:index rows) (c:index cols)
      : Lemma (SS.at (SS.window s o (rows * cols)) (r * cols + c) == m r c)
      = bound r c rows cols
    in
    Classical.forall_intro_2 aux

let window_vec (s:S.seq real) (o:nat) (n:nat) (x:vector real n)
  : Lemma (requires o + n <= S.length s /\ (forall (i:index n). vector_view 0.0R s o n i == x i))
          (ensures vec_eq (SS.window s o n) 0 x)
  = ()

let layer_of (rw:S.seq real) (w:model_weights real) (l:index num_hidden_layers)
  : Lemma (requires corresponds rw w)
          (ensures layer_eq (SS.packed_layer rw (SS.layer_offset l)) (SS.window rw (SS.layer_offset l) 1024) (w.layers l))
  = let o = SS.layer_offset l in
    let p = w.layers l in
    assert (params_eq (parameters_view 0.0R rw o) p);
    window_vec rw o 1024 p.input_layernorm;
    window_vec rw (o + 1024) 1024 p.post_attention_layernorm;
    window_mat rw (o + 2048) 2048 1024 p.q_proj;
    window_mat rw (o + 2099200) 1024 1024 p.k_proj;
    window_mat rw (o + 3147776) 1024 1024 p.v_proj;
    window_mat rw (o + 4196352) 1024 2048 p.o_proj;
    window_vec rw (o + 6293504) 128 p.q_norm;
    window_vec rw (o + 6293632) 128 p.k_norm;
    window_mat rw (o + 6293760) 3072 1024 p.gate_proj;
    window_mat rw (o + 9439488) 3072 1024 p.up_proj;
    window_mat rw (o + 12585216) 1024 3072 p.down_proj

let prompt_eq (tokens:S.seq I64.t) (prompt:vector token_id prompt_length) : prop =
  forall (t:index prompt_length). SS.tok_at tokens t == prompt t

let embed_sim (rw:S.seq real) (w:model_weights real) (tokens:S.seq I64.t) (prompt:vector token_id prompt_length)
  : Lemma (requires corresponds rw w /\ prompt_eq tokens prompt)
          (ensures mat_eq (SS.embed_rows 7 (SS.window rw 0 SS.embed_elements) tokens) 7 1024 (embedding w prompt))
  = reveal_opaque (`%SS.embed_rows) (SS.embed_rows 7 (SS.window rw 0 SS.embed_elements) tokens);
    let aux (t:index 7) (c:index 1024)
      : Lemma (SS.at (SS.embed_rows 7 (SS.window rw 0 SS.embed_elements) tokens) (t * 1024 + c) == embedding w prompt t c)
      = bound t c 7 1024; split t c 1024; bound (prompt t) c vocab_size 1024
    in
    Classical.forall_intro_2 aux

let rope_tables_sim (_:unit)
  : Lemma (rope_eq (SS.rope_cos_table 0 7) (SS.rope_sin_table 0 7) (rotary_embedding position_ids))
  = reveal_opaque (`%SS.rope_cos_table) (SS.rope_cos_table 0 7);
    reveal_opaque (`%SS.rope_sin_table) (SS.rope_sin_table 0 7);
    let freq (j:index 64) : Lemma (SS.inv_freq j == inv_freq j)
      = assert (of_int (2 * j) /. of_int head_dim == of_int j /. 64.0R)
    in
    let aux (t:index 7) (c:index 128)
      : Lemma (SS.at (SS.rope_cos_table 0 7) (t * 128 + c) == (rotary_embedding position_ids).cos t c /\
               SS.at (SS.rope_sin_table 0 7) (t * 128 + c) == (rotary_embedding position_ids).sin t c)
      = bound t c 7 128; split t c 128; freq (c % 64)
    in
    Classical.forall_intro_2 aux

let rec hidden_sim (rw:S.seq real) (w:model_weights real) (tokens:S.seq I64.t)
    (prompt:vector token_id prompt_length) (l:nat{l <= num_hidden_layers})
  : Lemma (requires corresponds rw w /\ prompt_eq tokens prompt)
          (ensures mat_eq (SS.hidden_after rw tokens l) 7 1024 (hidden_after_layers w prompt l))
          (decreases l)
  = if l = 0 then embed_sim rw w tokens prompt
    else begin
      hidden_sim rw w tokens prompt (l - 1);
      layer_of rw w (l - 1);
      rope_tables_sim ();
      let o = SS.layer_offset (l - 1) in
      layer_sim (SS.packed_layer rw o) (SS.window rw o 1024) (w.layers (l - 1))
        (SS.rope_cos_table 0 7) (SS.rope_sin_table 0 7) (rotary_embedding position_ids)
        (SS.hidden_after rw tokens (l - 1)) (hidden_after_layers w prompt (l - 1))
    end

let final_sim (rw:S.seq real) (w:model_weights real) (tokens:S.seq I64.t) (prompt:vector token_id prompt_length)
  : Lemma (requires corresponds rw w /\ prompt_eq tokens prompt)
          (ensures mat_eq (SS.final_hidden rw tokens) 7 1024 (last_hidden_state w prompt))
  = hidden_sim rw w tokens prompt num_hidden_layers;
    window_vec rw SS.final_norm_offset 1024 w.norm;
    rmsnorm_hidden (SS.hidden_after rw tokens num_hidden_layers) (SS.window rw SS.final_norm_offset 1024)
      (hidden_after_layers w prompt num_hidden_layers) w.norm

let logits_sim (rw:S.seq real) (w:model_weights real) (tokens:S.seq I64.t) (prompt:vector token_id prompt_length)
  : Lemma (requires corresponds rw w /\ prompt_eq tokens prompt)
          (ensures S.length (SS.next_token_logits rw tokens) == vocab_size /\
                   (forall (v:index vocab_size). SS.at (SS.next_token_logits rw tokens) v == next_token_logits w prompt v))
  = final_sim rw w tokens prompt;
    let we = SS.window rw 0 SS.embed_elements in
    let x = SS.window (SS.final_hidden rw tokens) 6144 1024 in
    reveal_opaque (`%SS.logits_row) (SS.logits_row vocab_size we x);
    let aux (v:index vocab_size)
      : Lemma (SS.at (SS.logits_row vocab_size we x) v == next_token_logits w prompt v)
      = let row (i:index 1024)
          : Lemma (SS.at we (v * 1024 + i) == w.embed_tokens v i /\
                   SS.at x (0 + i) == last_hidden_state w prompt 6 i)
          = bound v i vocab_size 1024
        in
        Classical.forall_intro row;
        sdot_dot 1024 we (v * 1024) 1 x 0 1 (w.embed_tokens v) (last_hidden_state w prompt 6)
    in
    Classical.forall_intro aux

(* The view of any flat real tensor corresponds to it. *)
let view_corresponds (rw:S.seq real)
  : Lemma (requires S.length rw == SS.packed_elements) (ensures corresponds rw (weights_view 0.0R rw))
  = ()

(* Packing typed real weights into the flat layout (the inverse of the view). *)

let div_bound (k:nat) (rows:nat) (cols:pos)
  : Lemma (requires k < rows * cols) (ensures k / cols < rows)
  = M.euclidean_division_definition k cols;
    if k / cols >= rows then M.lemma_mult_le_right cols rows (k / cols)

let mat_at (#rows:nat) (#cols:pos) (m:matrix real rows cols) (k:nat{k < rows * cols}) : GTot real =
  div_bound k rows cols;
  m (k / cols) (k % cols)

let param_at (p:parameters real) (j:nat{j < SS.layer_elements}) : GTot real =
  if j < 1024 then p.input_layernorm j
  else if j < 2048 then p.post_attention_layernorm (j - 1024)
  else if j < 2099200 then mat_at p.q_proj (j - 2048)
  else if j < 3147776 then mat_at p.k_proj (j - 2099200)
  else if j < 4196352 then mat_at p.v_proj (j - 3147776)
  else if j < 6293504 then mat_at p.o_proj (j - 4196352)
  else if j < 6293632 then p.q_norm (j - 6293504)
  else if j < 6293760 then p.k_norm (j - 6293632)
  else if j < 9439488 then mat_at p.gate_proj (j - 6293760)
  else if j < 12585216 then mat_at p.up_proj (j - 9439488)
  else mat_at p.down_proj (j - 12585216)

let pack_at (w:model_weights real) (i:nat{i < SS.packed_elements}) : GTot real =
  if i < SS.embed_elements then mat_at #vocab_size #hidden_size w.embed_tokens i
  else if i < SS.layers_offset then w.norm (i - SS.final_norm_offset)
  else begin
    div_bound (i - SS.layers_offset) num_hidden_layers SS.layer_elements;
    param_at (w.layers ((i - SS.layers_offset) / SS.layer_elements)) ((i - SS.layers_offset) % SS.layer_elements)
  end

let pack (w:model_weights real) : GTot (S.seq real) =
  S.init_ghost SS.packed_elements (pack_at w)

let mat_at_index (#rows:nat) (#cols:pos) (m:matrix real rows cols) (r:index rows) (c:index cols)
  : Lemma (r * cols + c < rows * cols /\ mat_at m (r * cols + c) == m r c)
  = bound r c rows cols; split r c cols

let layer_pack (w:model_weights real) (l:index num_hidden_layers) (j:nat{j < SS.layer_elements})
  : Lemma (SS.layer_offset l + j < SS.packed_elements /\
           S.index (pack w) (SS.layer_offset l + j) == param_at (w.layers l) j)
  = split l j SS.layer_elements

let slot_mat (w:model_weights real) (l:index num_hidden_layers) (base:nat) (rows cols:pos) (m:matrix real rows cols)
  : Lemma (requires base + rows * cols <= SS.layer_elements /\
                    (forall (k:nat{k < rows * cols}). param_at (w.layers l) (base + k) == mat_at m k))
          (ensures forall (r:index rows) (c:index cols).
                     elem 0.0R (pack w) (SS.layer_offset l + base + (r * cols + c)) == m r c)
  = let aux (r:index rows) (c:index cols)
      : Lemma (elem 0.0R (pack w) (SS.layer_offset l + base + (r * cols + c)) == m r c)
      = mat_at_index m r c; layer_pack w l (base + (r * cols + c))
    in
    Classical.forall_intro_2 aux

let slot_vec (w:model_weights real) (l:index num_hidden_layers) (base:nat) (n:nat) (x:vector real n)
  : Lemma (requires base + n <= SS.layer_elements /\
                    (forall (k:index n). param_at (w.layers l) (base + k) == x k))
          (ensures forall (i:index n). elem 0.0R (pack w) (SS.layer_offset l + base + i) == x i)
  = let aux (i:index n) : Lemma (elem 0.0R (pack w) (SS.layer_offset l + base + i) == x i)
      = layer_pack w l (base + i)
    in
    Classical.forall_intro aux

let pack_layer (w:model_weights real) (l:index num_hidden_layers)
  : Lemma (params_eq (parameters_view 0.0R (pack w) (SS.layer_offset l)) (w.layers l))
  = let p = w.layers l in
    slot_vec w l 0 1024 p.input_layernorm;
    slot_vec w l 1024 1024 p.post_attention_layernorm;
    slot_mat w l 2048 2048 1024 p.q_proj;
    slot_mat w l 2099200 1024 1024 p.k_proj;
    slot_mat w l 3147776 1024 1024 p.v_proj;
    slot_mat w l 4196352 1024 2048 p.o_proj;
    slot_vec w l 6293504 128 p.q_norm;
    slot_vec w l 6293632 128 p.k_norm;
    slot_mat w l 6293760 3072 1024 p.gate_proj;
    slot_mat w l 9439488 3072 1024 p.up_proj;
    slot_mat w l 12585216 1024 3072 p.down_proj

let pack_corresponds (w:model_weights real)
  : Lemma (corresponds (pack w) w)
  = let embed (t:index vocab_size) (c:index hidden_size)
      : Lemma (elem 0.0R (pack w) (0 + (t * hidden_size + c)) == w.embed_tokens t c)
      = mat_at_index #vocab_size #hidden_size w.embed_tokens t c
    in
    Classical.forall_intro_2 embed;
    Classical.forall_intro (pack_layer w)

(* ------------------------------------------------------------------ *)
(* %~ through the views                                                *)

let view_approx (s:S.seq BF.t) (r:S.seq real)
  : Lemma (requires S.length s == SS.packed_elements /\ s %~ r)
          (ensures weights_approximates (weights_view bf_zero s) (weights_view 0.0R r))
  = let mat (o:nat) (rows cols:nat)
      : Lemma (requires o + rows * cols <= SS.packed_elements)
              (ensures matrix_approximates (matrix_view bf_zero s o rows cols) (matrix_view 0.0R r o rows cols))
      = let aux (i:index rows) (j:index cols)
          : Lemma (elem bf_zero s (o + (i * cols + j)) %~ elem 0.0R r (o + (i * cols + j)))
          = bound i j rows cols
        in
        Classical.forall_intro_2 aux
    in
    let vec (o:nat) (n:nat)
      : Lemma (requires o + n <= SS.packed_elements)
              (ensures vector_approximates (vector_view bf_zero s o n) (vector_view 0.0R r o n))
      = ()
    in
    mat 0 vocab_size hidden_size;
    vec SS.final_norm_offset hidden_size;
    let layer (l:index num_hidden_layers)
      : Lemma (parameters_approximates (parameters_view bf_zero s (SS.layer_offset l))
                                       (parameters_view 0.0R r (SS.layer_offset l)))
      = let o = SS.layer_offset l in
        vec o 1024; vec (o + 1024) 1024;
        mat (o + 2048) 2048 1024; mat (o + 2099200) 1024 1024;
        mat (o + 3147776) 1024 1024; mat (o + 4196352) 1024 2048;
        vec (o + 6293504) 128; vec (o + 6293632) 128;
        mat (o + 6293760) 3072 1024; mat (o + 9439488) 3072 1024;
        mat (o + 12585216) 1024 3072
    in
    Classical.forall_intro layer

let mat_at_approx (s:S.seq BF.t) (o:int) (#rows:nat) (#cols:pos) (m:matrix real rows cols) (k:nat{k < rows * cols})
  : Lemma (requires matrix_approximates (matrix_view bf_zero s o rows cols) m)
          (ensures elem bf_zero s (o + k) %~ mat_at m k)
  = div_bound k rows cols;
    M.euclidean_division_definition k cols;
    assert (vector_approximates (matrix_view bf_zero s o rows cols (k / cols)) (m (k / cols)))

let param_approx (s:S.seq BF.t) (o:int) (p:parameters real) (j:nat{j < SS.layer_elements})
  : Lemma (requires parameters_approximates (parameters_view bf_zero s o) p)
          (ensures elem bf_zero s (o + j) %~ param_at p j)
  = if j < 2048 then ()
    else if j < 2099200 then mat_at_approx s (o + 2048) p.q_proj (j - 2048)
    else if j < 3147776 then mat_at_approx s (o + 2099200) p.k_proj (j - 2099200)
    else if j < 4196352 then mat_at_approx s (o + 3147776) p.v_proj (j - 3147776)
    else if j < 6293504 then mat_at_approx s (o + 4196352) p.o_proj (j - 4196352)
    else if j < 6293760 then ()
    else if j < 9439488 then mat_at_approx s (o + 6293760) p.gate_proj (j - 6293760)
    else if j < 12585216 then mat_at_approx s (o + 9439488) p.up_proj (j - 9439488)
    else mat_at_approx s (o + 12585216) p.down_proj (j - 12585216)

let pack_approx (s:S.seq BF.t) (w:model_weights real)
  : Lemma (requires S.length s == SS.packed_elements /\ weights_approximates (weights_view bf_zero s) w)
          (ensures s %~ pack w)
  = let aux (i:nat) : Lemma (i < S.length s ==> S.index s i %~ S.index (pack w) i)
      = if i < SS.embed_elements then mat_at_approx s 0 #vocab_size #hidden_size w.embed_tokens i
        else if i < SS.layers_offset then ()
        else if i < SS.packed_elements then begin
          let k = i - SS.layers_offset in
          div_bound k num_hidden_layers SS.layer_elements;
          M.euclidean_division_definition k SS.layer_elements;
          let l : index num_hidden_layers = k / SS.layer_elements in
          param_approx s (SS.layer_offset l) (w.layers l) (k % SS.layer_elements)
        end
    in
    Classical.forall_intro aux

let hidden_to_typed (o1:S.seq BF.t) (f:S.seq real) (x:hidden real)
  : Lemma (requires S.length o1 == 7168 /\ o1 %~ f /\ mat_eq f 7 1024 x)
          (ensures matrix_approximates (matrix_view bf_zero o1 0 prompt_length hidden_size) x)
  = let aux (t:index 7) (c:index 1024) : Lemma (elem bf_zero o1 (0 + (t * 1024 + c)) %~ x t c)
      = bound t c 7 1024
    in
    Classical.forall_intro_2 aux

let hidden_to_flat (o1:S.seq BF.t) (f:S.seq real) (x:hidden real)
  : Lemma (requires S.length o1 == 7168 /\ mat_eq f 7 1024 x /\
                    matrix_approximates (matrix_view bf_zero o1 0 prompt_length hidden_size) x)
          (ensures o1 %~ f)
  = let aux (i:nat) : Lemma (i < S.length o1 ==> S.index o1 i %~ S.index f i)
      = if i < 7168 then begin
          div_bound i 7 1024;
          M.euclidean_division_definition i 1024;
          assert (vector_approximates (matrix_view bf_zero o1 0 prompt_length hidden_size (i / 1024)) (x (i / 1024)))
        end
    in
    Classical.forall_intro aux

(* ------------------------------------------------------------------ *)
(* Conjunct by conjunct                                                *)

let numerical_forward (input0:S.seq BF.t) (input1:S.seq I64.t) (output1:S.seq BF.t)
    (prompt:vector token_id prompt_length)
  : Lemma (requires S.length input0 == SS.packed_elements /\ S.length output1 == 7168 /\ prompt_eq input1 prompt /\
                    (forall (rw:S.seq real). input0 %~ rw ==> output1 %~ SS.final_hidden rw input1))
          (ensures forall (w:model_weights real).
                     weights_approximates (weights_view bf_zero input0) w ==>
                     matrix_approximates (matrix_view bf_zero output1 0 prompt_length hidden_size) (last_hidden_state w prompt))
  = let aux (w:model_weights real)
      : Lemma (requires weights_approximates (weights_view bf_zero input0) w)
              (ensures matrix_approximates (matrix_view bf_zero output1 0 prompt_length hidden_size) (last_hidden_state w prompt))
      = pack_approx input0 w;
        pack_corresponds w;
        final_sim (pack w) w input1 prompt;
        hidden_to_typed output1 (SS.final_hidden (pack w) input1) (last_hidden_state w prompt)
    in
    Classical.forall_intro (Classical.move_requires aux)

let numerical_converse (input0:S.seq BF.t) (input1:S.seq I64.t) (output1:S.seq BF.t)
    (prompt:vector token_id prompt_length)
  : Lemma (requires S.length input0 == SS.packed_elements /\ S.length output1 == 7168 /\ prompt_eq input1 prompt /\
                    (forall (w:model_weights real).
                       weights_approximates (weights_view bf_zero input0) w ==>
                       matrix_approximates (matrix_view bf_zero output1 0 prompt_length hidden_size) (last_hidden_state w prompt)))
          (ensures forall (rw:S.seq real). input0 %~ rw ==> output1 %~ SS.final_hidden rw input1)
  = let aux (rw:S.seq real)
      : Lemma (requires input0 %~ rw) (ensures output1 %~ SS.final_hidden rw input1)
      = view_approx input0 rw;
        view_corresponds rw;
        final_sim rw (weights_view 0.0R rw) input1 prompt;
        hidden_to_flat output1 (SS.final_hidden rw input1) (last_hidden_state (weights_view 0.0R rw) prompt)
    in
    Classical.forall_intro (Classical.move_requires aux)

let row_of (s:S.seq BF.t{S.length s == vocab_size}) : vector BF.t vocab_size =
  fun (i:index vocab_size) -> S.index s i

let scores_forward (input0:S.seq BF.t) (input1:S.seq I64.t) (prompt:vector token_id prompt_length)
    (scores:S.seq BF.t{S.length scores == vocab_size})
  : Lemma (requires S.length input0 == SS.packed_elements /\ prompt_eq input1 prompt /\
                    (forall (rw:S.seq real). input0 %~ rw ==> scores %~ SS.next_token_logits rw input1))
          (ensures forall (w:model_weights real).
                     weights_approximates (weights_view bf_zero input0) w ==>
                     vector_approximates (row_of scores) (next_token_logits w prompt))
  = let aux (w:model_weights real)
      : Lemma (requires weights_approximates (weights_view bf_zero input0) w)
              (ensures vector_approximates (row_of scores) (next_token_logits w prompt))
      = pack_approx input0 w;
        pack_corresponds w;
        logits_sim (pack w) w input1 prompt
    in
    Classical.forall_intro (Classical.move_requires aux)

let scores_converse (input0:S.seq BF.t) (input1:S.seq I64.t) (prompt:vector token_id prompt_length)
    (scores:vector BF.t vocab_size)
  : Lemma (requires S.length input0 == SS.packed_elements /\ prompt_eq input1 prompt /\
                    (forall (w:model_weights real).
                       weights_approximates (weights_view bf_zero input0) w ==>
                       vector_approximates scores (next_token_logits w prompt)))
          (ensures forall (rw:S.seq real). input0 %~ rw ==> S.init_ghost vocab_size scores %~ SS.next_token_logits rw input1)
  = let aux (rw:S.seq real)
      : Lemma (requires input0 %~ rw) (ensures S.init_ghost vocab_size scores %~ SS.next_token_logits rw input1)
      = view_approx input0 rw;
        view_corresponds rw;
        logits_sim rw (weights_view 0.0R rw) input1 prompt
    in
    Classical.forall_intro (Classical.move_requires aux)

let argmax_forward (s:S.seq BF.t{S.length s == vocab_size}) (j:int)
  : Lemma (requires SS.argmax_choice s j) (ensures argmax_choice (row_of s) j)
  = ()

let argmax_converse (scores:vector BF.t vocab_size) (j:int)
  : Lemma (requires argmax_choice scores j) (ensures SS.argmax_choice (S.init_ghost vocab_size scores) j)
  = ()

let domain_iff (input0:S.seq BF.t) (input1:S.seq I64.t) (input2:int)
  : Lemma (requires S.length input1 == SS.prompt_length)
          (ensures input_domain (inputs_of input0 input1 input2) <==> SS.tokens_ok input1 /\ input2 == 1)
  = let input = inputs_of input0 input1 input2 in
    let fwd (i:nat)
      : Lemma (requires input_domain input) (ensures i < S.length input1 ==>
                 0 <= I64.v (S.index input1 i) /\ I64.v (S.index input1 i) < vocab_size)
      = if i < S.length input1 then assert (input.prompt_tokens i == I64.v (S.index input1 i))
    in
    let bwd (t:index prompt_length)
      : Lemma (requires SS.tokens_ok input1)
              (ensures 0 <= input.prompt_tokens t /\ input.prompt_tokens t < vocab_size)
      = ()
    in
    Classical.forall_intro (Classical.move_requires fwd);
    Classical.forall_intro (Classical.move_requires bwd)

let typed_of_flat (input0:S.seq BF.t) (input1:S.seq I64.t) (input2:int)
    (output0:S.seq I64.t) (output1:S.seq BF.t) (_:unit)
  : Lemma (requires S.length input0 == SS.packed_elements /\ S.length input1 == SS.prompt_length /\
                    SS.program_postcondition input0 input1 input2 output0 output1)
          (ensures postcondition (inputs_of input0 input1 input2) (outputs_of output0 output1))
  = let input = inputs_of input0 input1 input2 in
    let output = outputs_of output0 output1 in
    domain_iff input0 input1 input2;
    introduce input_domain input ==>
              (structural_postcondition input output /\
               numerical_postcondition N.bf16 input output /\
               selection_postcondition N.bf16 input output)
    with begin
      let prompt = prompt_ids input in
      numerical_forward input0 input1 output1 prompt;
      eliminate exists (scores:S.seq BF.t).
          S.length scores == vocab_size /\
          (forall (rw:S.seq real). input0 %~ rw ==> scores %~ SS.next_token_logits rw input1) /\
          SS.argmax_choice scores (I64.v (S.index output0 7))
      with begin
        scores_forward input0 input1 prompt scores;
        argmax_forward scores (I64.v (S.index output0 7));
        introduce exists (row:vector BF.t vocab_size).
            (forall (w:model_weights real).
               weights_approximates input.weights w ==>
               vector_approximates row (next_token_logits w prompt)) /\
            argmax_choice row (output.tokens prompt_length)
        with (row_of scores) and ()
      end
    end

let flat_of_typed (input0:S.seq BF.t) (input1:S.seq I64.t) (input2:int)
    (output0:S.seq I64.t) (output1:S.seq BF.t) (_:unit)
  : Lemma (requires S.length input0 == SS.packed_elements /\ S.length input1 == SS.prompt_length /\
                    S.length output0 == 8 /\ S.length output1 == 7168 /\
                    SS.tokens_ok input1 /\ input2 == 1 /\
                    postcondition (inputs_of input0 input1 input2) (outputs_of output0 output1))
          (ensures SS.program_postcondition input0 input1 input2 output0 output1)
  = let input = inputs_of input0 input1 input2 in
    let output = outputs_of output0 output1 in
    domain_iff input0 input1 input2;
    let prompt = prompt_ids input in
    let copy (i:nat) : Lemma (i < 7 ==> S.index output0 i == S.index input1 i)
      = if i < 7 then begin
          assert (output.tokens i == input.prompt_tokens i);
          assert (SS.tok_at output0 i == I64.v (S.index output0 i));
          assert (SS.tok_at input1 i == I64.v (S.index input1 i));
          I64.v_inj (S.index output0 i) (S.index input1 i)
        end
    in
    Classical.forall_intro copy;
    numerical_converse input0 input1 output1 prompt;
    eliminate exists (row:vector BF.t vocab_size).
        (forall (w:model_weights real).
           weights_approximates input.weights w ==>
           vector_approximates row (next_token_logits w prompt)) /\
        argmax_choice row (output.tokens prompt_length)
    with begin
      scores_converse input0 input1 prompt row;
      argmax_converse row (I64.v (S.index output0 7));
      introduce exists (scores:S.seq BF.t).
          S.length scores == vocab_size /\
          (forall (rw:S.seq real). input0 %~ rw ==> scores %~ SS.next_token_logits rw input1) /\
          SS.argmax_choice scores (I64.v (S.index output0 7))
      with (S.init_ghost vocab_size row) and ()
    end

(* ------------------------------------------------------------------ *)
(* Bridge theorems                                                     *)

(* Flat => typed, for packed weights and prompt of the program's lengths. *)
let program_postcondition_typed (input0:S.seq BF.t) (input1:S.seq I64.t) (input2:int)
    (output0:S.seq I64.t) (output1:S.seq BF.t)
  : Lemma (requires S.length input0 == SS.packed_elements /\ S.length input1 == SS.prompt_length)
          (ensures SS.program_postcondition input0 input1 input2 output0 output1 ==>
                   postcondition (inputs_of input0 input1 input2) (outputs_of output0 output1))
  = Classical.move_requires (typed_of_flat input0 input1 input2 output0 output1) ()

(* Typed => flat, when the returned buffers also have the program's lengths. *)
let program_postcondition_flat (input0:S.seq BF.t) (input1:S.seq I64.t) (input2:int)
    (output0:S.seq I64.t) (output1:S.seq BF.t)
  : Lemma (requires S.length input0 == SS.packed_elements /\ S.length input1 == SS.prompt_length /\
                    S.length output0 == 8 /\ S.length output1 == 7168)
          (ensures postcondition (inputs_of input0 input1 input2) (outputs_of output0 output1) ==>
                   SS.program_postcondition input0 input1 input2 output0 output1)
  = Classical.move_requires (flat_of_typed input0 input1 input2 output0 output1) ()

(* Exact equivalence: the flat postcondition is the typed one plus the
   output lengths (which the typed model cannot state) on the domain. *)
let program_postcondition_iff (input0:S.seq BF.t) (input1:S.seq I64.t) (input2:int)
    (output0:S.seq I64.t) (output1:S.seq BF.t)
  : Lemma (requires S.length input0 == SS.packed_elements /\ S.length input1 == SS.prompt_length)
          (ensures SS.program_postcondition input0 input1 input2 output0 output1 <==>
                   ((input_domain (inputs_of input0 input1 input2) ==>
                       S.length output0 == 8 /\ S.length output1 == 7168) /\
                    postcondition (inputs_of input0 input1 input2) (outputs_of output0 output1)))
  = domain_iff input0 input1 input2;
    program_postcondition_typed input0 input1 input2 output0 output1;
    Classical.move_requires (flat_of_typed input0 input1 input2 output0 output1) ()
