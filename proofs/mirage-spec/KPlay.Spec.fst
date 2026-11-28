module KPlay.Spec

(* Typed specification of the Mirage program: Qwen3-0.6B, one forward pass
   over a seven-token prompt followed by one greedy (argmax) token.

   Reference (reference/mirage/): base.py:27 `Model.forward(packed_weights,
   prompt_tokens, decode_steps)` runs the pinned Transformers 4.57.1
   modeling_qwen3.py with model_config.json and eager attention
   (model_abi.py:113), and returns (base.py:44-47)
     ( cat(prompt_tokens, logits[:, -1, :].argmax(-1)),  hidden_states[-1] ).

   Read top to bottom:
     S1  model: dimensions, index/vector/matrix, weight/input/output records;
     S2  reference equations over reals, one per reference operation;
     S3  liftings of the approximation relation %~ to the model types;
     S4  role-independent operation relations and their composition;
     S5  the program postcondition = structural /\ numerical /\ selection.

   The equations are exact real arithmetic. Kuiper's `x %~ r` relates a
   machine value x (here BF16) to a real r; it is an abstract relation, not an
   error bound, so BF16/FP32 rounding and the reference's casts are abstracted
   by it. Selection compares native BF16 scores. This file speaks about model
   values only; KPlay.Spec.Bridge relates it to the flat program buffers. *)

open FStar.Real
open Kuiper.Approximates
module F = Kuiper.Floating.Base
module BF = Kuiper.BFloat16
module Exp = FStar.Math.Exp
module Sqrt = FStar.Math.Sqrt
module Pow = FStar.Math.Pow
module Trig = Kuiper.Real.Trigonometry
module N = KPlay.Numbers

(* ------------------------------------------------------------------ *)
(* S1. Model                                                           *)

(* model_config.json; model_abi.py:16 (LAYERS, HIDDEN, VOCAB, PROMPT, SEQUENCE). *)
unfold let hidden_size : pos = 1024
unfold let intermediate_size : pos = 3072
unfold let vocab_size : pos = 151936
unfold let num_hidden_layers : pos = 28
unfold let num_attention_heads : pos = 16
unfold let num_key_value_heads : pos = 8
unfold let head_dim : pos = 128
unfold let prompt_length : pos = 7
unfold let sequence_length : pos = 8   (* prompt + decode_steps = 1 generated token *)

type index (n:nat) = i:nat{i < n}
type vector (a:Type0) (n:nat) = N.row a n
type matrix (a:Type0) (rows cols:nat) = N.row (vector a cols) rows

(* [prompt_length, hidden_size]: residual / hidden stream of the prompt. *)
type hidden (a:Type0) = matrix a prompt_length hidden_size

(* [prompt_length, count, head_dim]: per-position heads of a projection
   (modeling_qwen3.py:198-202, view(hidden_shape) before transpose(1, 2)). *)
type heads (a:Type0) (count:nat) = index prompt_length -> GTot (matrix a count head_dim)

(* Attention scores / weights [query position, head, key position]. *)
type attention_map = index prompt_length -> GTot (matrix real num_attention_heads prompt_length)

type token_id = index vocab_size

(* Weights of one Qwen3DecoderLayer, named as the state-dict entries
   model.layers.{l}.* (model_abi.py:24-41). *)
noeq type parameters (a:Type0) = {
  input_layernorm : vector a hidden_size;                (* [1024] *)
  post_attention_layernorm : vector a hidden_size;       (* [1024] *)
  q_proj : matrix a 2048 hidden_size;                    (* self_attn.q_proj [2048, 1024] *)
  k_proj : matrix a 1024 hidden_size;                    (* self_attn.k_proj [1024, 1024] *)
  v_proj : matrix a 1024 hidden_size;                    (* self_attn.v_proj [1024, 1024] *)
  o_proj : matrix a hidden_size 2048;                    (* self_attn.o_proj [1024, 2048] *)
  q_norm : vector a head_dim;                            (* self_attn.q_norm [128] *)
  k_norm : vector a head_dim;                            (* self_attn.k_norm [128] *)
  gate_proj : matrix a intermediate_size hidden_size;    (* mlp.gate_proj [3072, 1024] *)
  up_proj : matrix a intermediate_size hidden_size;      (* mlp.up_proj [3072, 1024] *)
  down_proj : matrix a hidden_size intermediate_size;    (* mlp.down_proj [1024, 3072] *)
}

(* The unpacked checkpoint (model_abi.py:70-81 unpack_weights); lm_head.weight
   is tied to embed_tokens (model_abi.py:80). *)
noeq type model_weights (a:Type0) = {
  embed_tokens : matrix a vocab_size hidden_size;        (* model.embed_tokens.weight *)
  norm : vector a hidden_size;                           (* model.norm.weight *)
  layers : index num_hidden_layers -> GTot (parameters a);
}

(* position_embeddings = (cos, sin), each [prompt_length, head_dim]. *)
noeq type rotary_tables (a:Type0) = {
  cos : matrix a prompt_length head_dim;
  sin : matrix a prompt_length head_dim;
}

(* base.py:27 forward(packed_weights, prompt_tokens, decode_steps); `weights`
   is packed_weights unpacked (base.py:28). Token ids are the given int64
   values; their range is a domain premise (S5). *)
noeq type inputs_t (a:Type0) = {
  weights : model_weights a;
  prompt_tokens : vector int prompt_length;
  decode_steps : int;
}

(* base.py:44-47: the returned token sequence [8] and hidden_states[-1]. *)
noeq type outputs_t (a:Type0) = {
  tokens : vector int sequence_length;
  hidden_states : matrix a prompt_length hidden_size;
}

(* The program's number type: BF16 (packed_weights and hidden_states). *)
type inputs = inputs_t BF.t
type outputs = outputs_t BF.t

(* ------------------------------------------------------------------ *)
(* S2. Reference equations over reals                                  *)

(* Left-to-right sum f 0 + ... + f (m - 1) of the first m entries. *)
let rec sum_prefix (#n:nat) (f:vector real n) (m:nat{m <= n}) : GTot real (decreases m) =
  if m = 0 then 0.0R else sum_prefix f (m - 1) +. f (m - 1)

let sum (#n:nat) (f:vector real n) : GTot real = sum_prefix f n

let rec sum_prefix_nonnegative (#n:nat) (f:vector real n) (m:nat{m <= n})
  : Lemma (requires forall (i:index n). f i >=. 0.0R)
          (ensures sum_prefix f m >=. 0.0R)
          (decreases m)
  = if m > 0 then sum_prefix_nonnegative f (m - 1)

let rec sum_prefix_positive (#n:nat) (f:vector real n) (m:pos{m <= n})
  : Lemma (requires forall (i:index n). f i >. 0.0R)
          (ensures sum_prefix f m >. 0.0R)
          (decreases m)
  = if m > 1 then sum_prefix_positive f (m - 1)

let dot (#n:nat) (x y:vector real n) : GTot real = sum (fun (i:index n) -> x i *. y i)

(* rms_norm_eps = 1e-06 (model_config.json). *)
let rms_norm_eps : real = 1.0R /. 1000000.0R

(* Qwen3RMSNorm.forward, modeling_qwen3.py:59-64:
   variance = mean(x^2);  weight * (x * rsqrt(variance + eps)). *)
let rms_norm (#n:pos) (weight x:vector real n) : GTot (vector real n) =
  let squares : vector real n = fun (i:index n) -> x i *. x i in
  sum_prefix_nonnegative squares n;
  let variance = sum squares /. of_int n in
  fun (i:index n) -> weight i *. (x i *. Sqrt.rsqrt (variance +. rms_norm_eps))

(* nn.Linear(bias=False): y[o] = sum_k weight[o, k] * x[k]. *)
let linear (#src #dst:nat) (weight:matrix real dst src) (x:vector real src)
  : GTot (vector real dst) =
  fun (o:index dst) -> dot (weight o) x

let linear_rows (#rows #src #dst:nat) (weight:matrix real dst src) (x:matrix real rows src)
  : GTot (matrix real rows dst) =
  fun (r:index rows) -> linear weight (x r)

(* Each position's row normalized (input_layernorm, post_attention_layernorm,
   model.norm). *)
let normalize_hidden (weight:vector real hidden_size) (x:hidden real) : GTot (hidden real) =
  fun (t:index prompt_length) -> rms_norm weight (x t)

(* view(input_shape + (-1, head_dim)) (modeling_qwen3.py:198-202). *)
let split_heads (#a:Type0) (#count:pos) (x:matrix a prompt_length (count * head_dim))
  : GTot (heads a count) =
  fun (t:index prompt_length) (h:index count) (c:index head_dim) -> x t (h * head_dim + c)

(* q_norm / k_norm normalize each head (modeling_qwen3.py:183-184, 200-201). *)
let normalize_heads (#count:pos) (weight:vector real head_dim) (x:heads real count)
  : GTot (heads real count) =
  fun (t:index prompt_length) (h:index count) -> rms_norm weight (x t h)

(* Default rope type (modeling_rope_utils.py:92-133): base = rope_theta,
   inv_freq[j] = 1 / base^(2j / head_dim), attention_factor = 1.0. *)
let rope_theta : real = 1000000.0R
let attention_scaling : real = 1.0R

let inv_freq (j:index 64) : real =
  1.0R /. Pow.pow rope_theta (of_int (2 * j) /. of_int head_dim)

(* base.py:37: positions = arange(7). *)
let position_ids : vector int prompt_length = fun (t:index prompt_length) -> t

(* Qwen3RotaryEmbedding.forward, modeling_qwen3.py:321-332:
   freqs = inv_freq * position, emb = cat(freqs, freqs),
   cos = emb.cos() * attention_scaling, sin = emb.sin() * attention_scaling. *)
let rotary_embedding (positions:vector int prompt_length) : GTot (rotary_tables real) = {
  cos = (fun (t:index prompt_length) (c:index head_dim) ->
           Trig.cos (of_int (positions t) *. inv_freq (c % 64)) *. attention_scaling);
  sin = (fun (t:index prompt_length) (c:index head_dim) ->
           Trig.sin (of_int (positions t) *. inv_freq (c % 64)) *. attention_scaling);
}

(* rotate_half, modeling_qwen3.py:86-90: cat(-x2, x1). *)
let rotate_half (#count:nat) (x:heads real count) : GTot (heads real count) =
  fun (t:index prompt_length) (h:index count) (c:index head_dim) ->
    if c < 64 then 0.0R -. x t h (c + 64) else x t h (c - 64)

(* apply_rotary_pos_emb, modeling_qwen3.py:115-116: x * cos + rotate_half(x) * sin. *)
let apply_rotary_pos_emb (#count:nat) (rope:rotary_tables real) (x:heads real count)
  : GTot (heads real count) =
  fun (t:index prompt_length) (h:index count) (c:index head_dim) ->
    x t h c *. rope.cos t c +. rotate_half x t h c *. rope.sin t c

(* repeat_kv with n_rep = 16 / 8 = 2 (modeling_qwen3.py:120-129): attention
   head h reads key/value head h / 2. *)
let repeat_kv (#a:Type0) (x:heads a num_key_value_heads) : GTot (heads a num_attention_heads) =
  fun (t:index prompt_length) (h:index num_attention_heads) -> x t (h / 2)

(* scaling = head_dim ** -0.5 (modeling_qwen3.py:167). *)
let scaling : real = 1.0R /. Sqrt.sqrt 128.0R

(* torch.finfo(bfloat16).min = -(2 - 2^-7) * 2^127 (masking_utils.py:519). *)
let min_dtype : real = 0.0R -. 338953138925153547590470800371487866880.0R

(* Additive causal mask: key kv is visible to query q iff kv <= q
   (masking_utils.py:74-78 causal_mask_function; 519-521: 0 where visible,
   min_dtype elsewhere). As in the reference it is finite and added. *)
let causal_mask (q kv:index prompt_length) : real =
  if kv <= q then 0.0R else min_dtype

(* eager_attention_forward, modeling_qwen3.py:145-148:
   matmul(query, key^T) * scaling + causal_mask. *)
let attention_scores (query key:heads real num_attention_heads) : GTot attention_map =
  fun (t:index prompt_length) (h:index num_attention_heads) (j:index prompt_length) ->
    dot (query t h) (key j h) *. scaling +. causal_mask t j

(* softmax(dim=-1) (modeling_qwen3.py:150). *)
let softmax (#n:pos) (x:vector real n) : GTot (vector real n) =
  let numerator : vector real n = fun (i:index n) -> Exp.exp (x i) in
  Classical.forall_intro (fun (i:index n) -> Exp.exp_positive (x i));
  sum_prefix_positive numerator n;
  fun (i:index n) -> numerator i /. sum numerator

let attention_weights (query key:heads real num_attention_heads) : GTot attention_map =
  fun (t:index prompt_length) (h:index num_attention_heads) -> softmax (attention_scores query key t h)

(* matmul(attn_weights, value_states) (modeling_qwen3.py:152). *)
let attention_output (weights:attention_map) (value:heads real num_attention_heads)
  : GTot (heads real num_attention_heads) =
  fun (t:index prompt_length) (h:index num_attention_heads) (c:index head_dim) ->
    sum (fun (j:index prompt_length) -> weights t h j *. value j h c)

(* attn_output.reshape(input_shape + (-1,)) (modeling_qwen3.py:228). *)
let merge_heads (x:heads real num_attention_heads) : GTot (matrix real prompt_length 2048) =
  fun (t:index prompt_length) (ch:index 2048) -> x t (ch / head_dim) (ch % head_dim)

(* Qwen3Attention.forward, modeling_qwen3.py:188-230, on the normalized
   stream x. Prefill from an empty cache: the keys/values are this layer's. *)
let self_attn (p:parameters real) (rope:rotary_tables real) (x:hidden real) : GTot (hidden real) =
  let query = apply_rotary_pos_emb rope
                (normalize_heads p.q_norm (split_heads #real #num_attention_heads (linear_rows p.q_proj x))) in
  let key = apply_rotary_pos_emb rope
                (normalize_heads p.k_norm (split_heads #real #num_key_value_heads (linear_rows p.k_proj x))) in
  let value = split_heads #real #num_key_value_heads (linear_rows p.v_proj x) in
  linear_rows p.o_proj
    (merge_heads (attention_output (attention_weights query (repeat_kv key)) (repeat_kv value)))

(* SiLU, x * sigmoid(x) (hidden_act = silu; activations.py:99). *)
let silu (x:real) : real =
  Exp.exp_positive (0.0R -. x);
  x /. (1.0R +. Exp.exp (0.0R -. x))

(* Qwen3MLP.forward, modeling_qwen3.py:81-83:
   down_proj(act_fn(gate_proj(x)) * up_proj(x)). *)
let mlp (p:parameters real) (x:hidden real) : GTot (hidden real) =
  let gate = linear_rows p.gate_proj x in
  let up = linear_rows p.up_proj x in
  linear_rows p.down_proj
    (fun (t:index prompt_length) (i:index intermediate_size) -> silu (gate t i) *. up t i)

(* residual + hidden_states (modeling_qwen3.py:270, 276). *)
let residual_add (x y:hidden real) : GTot (hidden real) =
  fun (t:index prompt_length) (c:index hidden_size) -> x t c +. y t c

(* Qwen3DecoderLayer.forward, modeling_qwen3.py:246-277. *)
let decoder_layer (p:parameters real) (rope:rotary_tables real) (x:hidden real) : GTot (hidden real) =
  let after_attention = residual_add x (self_attn p rope (normalize_hidden p.input_layernorm x)) in
  residual_add after_attention (mlp p (normalize_hidden p.post_attention_layernorm after_attention))

(* embed_tokens(input_ids), modeling_qwen3.py:371. *)
let embedding (#a:Type0) (w:model_weights a) (prompt:vector token_id prompt_length) : GTot (hidden a) =
  fun (t:index prompt_length) -> w.embed_tokens (prompt t)

(* The residual stream after the first l decoder layers (modeling_qwen3.py:
   407-418: one rotary table for all layers). *)
let rec hidden_after_layers (w:model_weights real) (prompt:vector token_id prompt_length)
    (l:nat{l <= num_hidden_layers})
  : GTot (hidden real) (decreases l)
  = if l = 0 then embedding w prompt
    else decoder_layer (w.layers (l - 1)) (rotary_embedding position_ids)
           (hidden_after_layers w prompt (l - 1))

(* hidden_states[-1] = self.norm(hidden_states) (modeling_qwen3.py:421). *)
let last_hidden_state (w:model_weights real) (prompt:vector token_id prompt_length) : GTot (hidden real) =
  normalize_hidden w.norm (hidden_after_layers w prompt num_hidden_layers)

(* logits[:, -1, :] of the tied lm_head (modeling_qwen3.py:494; base.py:43). *)
let next_token_logits (w:model_weights real) (prompt:vector token_id prompt_length)
  : GTot (vector real vocab_size) =
  linear w.embed_tokens (last_hidden_state w prompt (prompt_length - 1))

(* argmax(dim=-1) (base.py:43) over a native BF16 score row: N.bf16_argmax
   (the first NaN if any, otherwise any maximum). *)
let argmax_choice (#n:nat) (scores:vector BF.t n) (j:int) : prop = N.bf16_argmax n scores j

(* ------------------------------------------------------------------ *)
(* S3. Liftings of %~ to the model types                               *)

let vector_approx (#a:Type0) (num:N.number a) (#n:nat) (x:vector a n) (r:vector real n) : prop =
  forall (i:index n). num.N.approx (x i) (r i)

let matrix_approx (#a:Type0) (num:N.number a) (#rows #cols:nat) (x:matrix a rows cols) (r:matrix real rows cols) : prop =
  forall (i:index rows). vector_approx num (x i) (r i)

let parameters_approx (#a:Type0) (num:N.number a) (p:parameters a) (r:parameters real) : prop =
  vector_approx num p.input_layernorm r.input_layernorm /\
  vector_approx num p.post_attention_layernorm r.post_attention_layernorm /\
  matrix_approx num p.q_proj r.q_proj /\
  matrix_approx num p.k_proj r.k_proj /\
  matrix_approx num p.v_proj r.v_proj /\
  matrix_approx num p.o_proj r.o_proj /\
  vector_approx num p.q_norm r.q_norm /\
  vector_approx num p.k_norm r.k_norm /\
  matrix_approx num p.gate_proj r.gate_proj /\
  matrix_approx num p.up_proj r.up_proj /\
  matrix_approx num p.down_proj r.down_proj

let weights_approx (#a:Type0) (num:N.number a) (w:model_weights a) (r:model_weights real) : prop =
  matrix_approx num w.embed_tokens r.embed_tokens /\
  vector_approx num w.norm r.norm /\
  (forall (l:index num_hidden_layers). parameters_approx num (w.layers l) (r.layers l))

(* The BF16 instances, used by the relations below and the proofs. *)
let vector_approximates (#n:nat) (x:vector BF.t n) (r:vector real n) : prop = vector_approx N.bf16 x r
let matrix_approximates (#rows #cols:nat) (x:matrix BF.t rows cols) (r:matrix real rows cols) : prop =
  matrix_approx N.bf16 x r
let parameters_approximates (p:parameters BF.t) (r:parameters real) : prop = parameters_approx N.bf16 p r
let weights_approximates (w:model_weights BF.t) (r:model_weights real) : prop = weights_approx N.bf16 w r

(* ------------------------------------------------------------------ *)
(* S4. Operation relations (role independent)                          *)

(* Each: for ALL reals that the inputs approximate, the output approximates
   the reference operation applied to those reals. *)
let embedding_relation (w:model_weights BF.t) (prompt:vector token_id prompt_length) (y:hidden BF.t) : prop =
  forall (rw:model_weights real).
    weights_approximates w rw ==> matrix_approximates y (embedding rw prompt)

let decoder_layer_relation (p:parameters BF.t) (x y:hidden BF.t) : prop =
  forall (rp:parameters real) (rx:hidden real).
    parameters_approximates p rp /\ matrix_approximates x rx ==>
    matrix_approximates y (decoder_layer rp (rotary_embedding position_ids) rx)

let final_norm_relation (weight:vector BF.t hidden_size) (x y:hidden BF.t) : prop =
  forall (rn:vector real hidden_size) (rx:hidden real).
    vector_approximates weight rn /\ matrix_approximates x rx ==>
    matrix_approximates y (normalize_hidden rn rx)

let lm_head_relation (embed:matrix BF.t vocab_size hidden_size) (x:hidden BF.t)
    (scores:vector BF.t vocab_size) : prop =
  forall (re:matrix real vocab_size hidden_size) (rx:hidden real).
    matrix_approximates embed re /\ matrix_approximates x rx ==>
    vector_approximates scores (linear re (rx (prompt_length - 1)))

(* Boundary values of one run (G2): the embedding output, each decoder
   layer's output, the normalized hidden rows and the consumed score row. *)
noeq type execution = {
  embedded : hidden BF.t;
  layer_output : index num_hidden_layers -> GTot (hidden BF.t);
  normalized : hidden BF.t;
  scores : vector BF.t vocab_size;
}

let layer_input (run:execution) (l:index num_hidden_layers) : GTot (hidden BF.t) =
  if l = 0 then run.embedded else run.layer_output (l - 1)

let numerical_execution (w:model_weights BF.t) (prompt:vector token_id prompt_length) (run:execution) : prop =
  embedding_relation w prompt run.embedded /\
  (forall (l:index num_hidden_layers).
     decoder_layer_relation (w.layers l) (layer_input run l) (run.layer_output l)) /\
  final_norm_relation w.norm (run.layer_output (num_hidden_layers - 1)) run.normalized /\
  lm_head_relation w.embed_tokens run.normalized run.scores

(* ------------------------------------------------------------------ *)
(* S5. Program postcondition                                           *)

(* Domain premises checked by validate_inputs (model_abi.py:84-92). *)
let input_domain (#a:Type0) (input:inputs_t a) : prop =
  (forall (t:index prompt_length). 0 <= input.prompt_tokens t /\ input.prompt_tokens t < vocab_size) /\
  input.decode_steps == 1

let prompt_ids (#a:Type0) (input:inputs_t a{input_domain input}) : vector token_id prompt_length =
  fun (t:index prompt_length) -> input.prompt_tokens t

(* torch.cat((prompt_tokens, generated)) (base.py:45): the prompt in front. *)
let structural_postcondition (#a:Type0) (input:inputs_t a) (output:outputs_t a) : prop =
  forall (t:index prompt_length). output.tokens t == input.prompt_tokens t

(* hidden_states[-1] approximates the reference for every real model that
   the weights approximate. *)
let numerical_postcondition (#a:Type0) (num:N.number a) (input:inputs_t a{input_domain input}) (output:outputs_t a) : prop =
  forall (rw:model_weights real).
    weights_approx num input.weights rw ==>
    matrix_approx num output.hidden_states (last_hidden_state rw (prompt_ids input))

(* The generated token is an allowed argmax of the score row actually
   consumed, and that row approximates the reference next-token logits. *)
let selection_postcondition (#a:Type0) (num:N.number a) (input:inputs_t a{input_domain input}) (output:outputs_t a) : prop =
  exists (scores:vector a vocab_size).
    (forall (rw:model_weights real).
       weights_approx num input.weights rw ==>
       vector_approx num scores (next_token_logits rw (prompt_ids input))) /\
    num.N.argmax vocab_size scores (output.tokens prompt_length)

let spec (#a:Type0) (num:N.number a) (input:inputs_t a) (output:outputs_t a) : prop =
  input_domain input ==>
  (structural_postcondition input output /\
   numerical_postcondition num input output /\
   selection_postcondition num input output)

let postcondition (input:inputs) (output:outputs) : prop = spec N.bf16 input output

(* ------------------------------------------------------------------ *)
(* Composition (G4): the S4 relations along one run imply S5.          *)

let rec chain (w:model_weights BF.t) (prompt:vector token_id prompt_length) (run:execution)
    (rw:model_weights real) (l:nat{l <= num_hidden_layers})
  : Lemma (requires numerical_execution w prompt run /\ weights_approximates w rw)
          (ensures matrix_approximates (if l = 0 then run.embedded else run.layer_output (l - 1))
                                       (hidden_after_layers rw prompt l))
          (decreases l)
  = if l > 0 then begin
      chain w prompt run rw (l - 1);
      assert (parameters_approximates (w.layers (l - 1)) (rw.layers (l - 1)));
      assert (decoder_layer_relation (w.layers (l - 1)) (layer_input run (l - 1)) (run.layer_output (l - 1)))
    end

let execution_postcondition (input:inputs) (run:execution) (output:outputs)
  : Lemma (requires input_domain input /\
                    numerical_execution input.weights (prompt_ids input) run /\
                    structural_postcondition input output /\
                    (forall (t:index prompt_length) (c:index hidden_size).
                       output.hidden_states t c == run.normalized t c) /\
                    argmax_choice run.scores (output.tokens prompt_length))
          (ensures postcondition input output)
  = let prompt = prompt_ids input in
    let aux (rw:model_weights real)
      : Lemma (weights_approximates input.weights rw ==>
               matrix_approximates output.hidden_states (last_hidden_state rw prompt) /\
               vector_approximates run.scores (next_token_logits rw prompt))
      = introduce weights_approximates input.weights rw ==>
                  (matrix_approximates output.hidden_states (last_hidden_state rw prompt) /\
                   vector_approximates run.scores (next_token_logits rw prompt))
        with chain input.weights prompt run rw num_hidden_layers
    in
    Classical.forall_intro aux
