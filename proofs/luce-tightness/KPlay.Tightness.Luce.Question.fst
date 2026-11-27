module KPlay.Tightness.Luce.Question
(* What the LUCE spec (KPlay.Spec) should pin down. Review this file: every
   output must be listed, each target must be the reference math, and
   `allowed` is the only freedom. *)
open FStar.Real
module K = KPlay.Spec
module A = KPlay.Tightness.Answer
module S = FStar.Seq
module BF = Kuiper.BFloat16

(* Inputs: weights, prompt (1..8 valid tokens), decode steps (1..5). *)
let input = K.inputs
let domain (x:input) : prop = K.input_domain x
let z : BF.t = Kuiper.Floating.zero #BF.t
let example : input = {
  K.weights = {
    K.embed_tokens = (fun _ _ -> z);
    K.final_norm = (fun _ -> z);
    K.linear_layers = (fun _ -> {
      K.input_layernorm = (fun _ -> z);
      K.post_attention_layernorm = (fun _ -> z);
      K.mlp = { K.gate_proj = (fun _ _ -> z); K.up_proj = (fun _ _ -> z); K.down_proj = (fun _ _ -> z) };
      K.token_mixer = {
        K.in_proj_qkv = (fun _ _ -> z); K.in_proj_z = (fun _ _ -> z);
        K.in_proj_b = (fun _ _ -> z); K.in_proj_a = (fun _ _ -> z);
        K.conv1d = (fun _ _ -> z); K.a_log = (fun _ -> z); K.dt_bias = (fun _ -> z);
        K.norm = (fun _ -> z); K.out_proj = (fun _ _ -> z) } });
    K.full_layers = (fun _ -> {
      K.input_layernorm = (fun _ -> z);
      K.post_attention_layernorm = (fun _ -> z);
      K.mlp = { K.gate_proj = (fun _ _ -> z); K.up_proj = (fun _ _ -> z); K.down_proj = (fun _ _ -> z) };
      K.token_mixer = {
        K.q_proj = (fun _ _ -> z); K.k_proj = (fun _ _ -> z); K.v_proj = (fun _ _ -> z);
        K.q_norm = (fun _ -> z); K.k_norm = (fun _ -> z); K.o_proj = (fun _ _ -> z) } }) };
  K.prompt_length = 1;
  K.prompt = (fun _ -> 0);
  K.decode_steps = 1 }

(* The complete answer: the generated tokens (exact), then one 1024-value
   hidden row per generated token. generated_count is the number of tokens. *)
let exact = S.seq int
let number = BF.t
let count (e:exact) : nat = S.length e * K.hidden_size
let view (y:A.answer exact number count) : GTot K.outputs = {
  K.generated_count = S.length y.A.exact;
  K.tokens = (fun j -> S.index y.A.exact j);
  K.hidden_states = (fun j k -> S.index y.A.numbers (j * K.hidden_size + k)) }
let spec (x:input) (y:A.answer exact number count) : prop = K.postcondition x (view y)

(* The real models the BF16 weights stand for. *)
let model = K.model_weights real
let models (x:input) (m:model) : prop = K.weights_approximates x.K.weights m

(* Tokens consumed at each position: the prompt, then the generated tokens. *)
let toks (x:input) (c:exact) (p:nat) : GTot int =
  if p < x.K.prompt_length then x.K.prompt p
  else if p - x.K.prompt_length < S.length c then S.index c (p - x.K.prompt_length)
  else 0

(* Reference residual stream (with PyTorch's float32 constants) at the
   position that produces generated token j. *)
let hidden (x:input) (m:model) (c:exact) (j:nat) : GTot (K.vector real K.hidden_size) =
  K.hidden_at K.model_numerics m (toks x c) (K.last_prompt_position x + j)

(* The only freedom: each generated token may be any argmax of a BF16 or F32
   score row that approximates the reference logits at its position (ties,
   rounding); there are exactly decode_steps tokens. *)
let choice (x:input) = exact
let allowed (x:input{domain x}) (c:choice x) : prop =
  S.length c == x.K.decode_steps /\
  (forall (j:nat{j < S.length c}) (m:model). models x m ==>
     K.greedy_choice_relation K.model_numerics m (hidden x m c j) (S.index c j))

(* Targets: the chosen tokens; hidden row j is the reference residual stream
   at position prompt_length - 1 + j. *)
let exact_target (x:input{domain x}) (c:choice x) : GTot exact = c
let number_target (x:input{domain x}) (m:model) (c:choice x)
  : GTot (s:S.seq real{S.length s == count (exact_target x c)}) =
  S.init_ghost (count c) (fun i -> hidden x m c (i / K.hidden_size) (i % K.hidden_size))
