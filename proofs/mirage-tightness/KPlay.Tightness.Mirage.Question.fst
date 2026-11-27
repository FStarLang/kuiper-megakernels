module KPlay.Tightness.Mirage.Question
(* What the Mirage spec (KPlay.Spec) should pin down. Review this file: every
   output must be listed, each target must be the reference math, and
   `allowed` is the only freedom. *)
open FStar.Real
module K = KPlay.Spec
module A = KPlay.Tightness.Answer
module S = FStar.Seq
module BF = Kuiper.BFloat16

(* Inputs: the spec's input record (weights, 7 prompt tokens, decode steps). *)
let input = K.inputs
let domain (x:input) : prop = K.input_domain x
let example : input = {
  K.weights = { K.embed_tokens = (fun _ _ -> Kuiper.Floating.zero #BF.t);
                K.norm = (fun _ -> Kuiper.Floating.zero #BF.t);
                K.layers = (fun _ -> {
                  K.input_layernorm = (fun _ -> Kuiper.Floating.zero #BF.t);
                  K.post_attention_layernorm = (fun _ -> Kuiper.Floating.zero #BF.t);
                  K.q_proj = (fun _ _ -> Kuiper.Floating.zero #BF.t);
                  K.k_proj = (fun _ _ -> Kuiper.Floating.zero #BF.t);
                  K.v_proj = (fun _ _ -> Kuiper.Floating.zero #BF.t);
                  K.o_proj = (fun _ _ -> Kuiper.Floating.zero #BF.t);
                  K.q_norm = (fun _ -> Kuiper.Floating.zero #BF.t);
                  K.k_norm = (fun _ -> Kuiper.Floating.zero #BF.t);
                  K.gate_proj = (fun _ _ -> Kuiper.Floating.zero #BF.t);
                  K.up_proj = (fun _ _ -> Kuiper.Floating.zero #BF.t);
                  K.down_proj = (fun _ _ -> Kuiper.Floating.zero #BF.t) }) };
  K.prompt_tokens = (fun _ -> 0);
  K.decode_steps = 1 }

(* The complete answer: the 8 returned tokens (exact), then the 7 x 1024
   final hidden values, row by row. *)
let exact = s:S.seq int{S.length s == K.sequence_length}
let number = BF.t
let cells : nat = K.prompt_length * K.hidden_size
let count (_:exact) : nat = cells
let view (y:A.answer exact number count) : GTot K.outputs = {
  K.tokens = (fun t -> S.index y.A.exact t);
  K.hidden_states = (fun r c -> S.index y.A.numbers (r * K.hidden_size + c)) }
let spec (x:input) (y:A.answer exact number count) : prop = K.postcondition x (view y)

(* The real models the BF16 weights stand for. *)
let model = K.model_weights real
let models (x:input) (m:model) : prop = K.weights_approximates x.K.weights m

(* The only freedom: the generated token may be any argmax of a BF16 score
   row that approximates the reference next-token logits (ties, rounding). *)
let choice (x:input) = K.token_id
let allowed (x:input{domain x}) (c:choice x) : prop =
  exists (scores:K.vector BF.t K.vocab_size).
    (forall (m:model). models x m ==>
       K.vector_approximates scores (K.next_token_logits m (K.prompt_ids x))) /\
    K.argmax_choice scores c

(* Targets: the prompt followed by the chosen token; the reference final
   hidden state hidden_states[-1]. *)
let exact_target (x:input{domain x}) (c:choice x) : GTot exact =
  S.append (S.init_ghost K.prompt_length (fun t -> x.K.prompt_tokens t)) (S.create 1 c)
let number_target (x:input{domain x}) (m:model) (c:choice x) : GTot (s:S.seq real{S.length s == count (exact_target x c)}) =
  S.init_ghost cells (fun i -> K.last_hidden_state m (K.prompt_ids x) (i / K.hidden_size) (i % K.hidden_size))
