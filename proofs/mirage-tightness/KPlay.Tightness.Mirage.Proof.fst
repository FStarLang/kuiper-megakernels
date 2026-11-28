module KPlay.Tightness.Mirage.Proof
(* Proof of the generated theorem KPlay.Tightness.Mirage.Statement.tight. *)
open FStar.Real
open KPlay.Spec
module T = KPlay.Tightness.Mirage.Statement
module N = KPlay.Numbers
module Br = KPlay.Spec.Bridge
module SS = KPlay.SharedSpec
module S = FStar.Seq
module I64 = FStar.Int64

#set-options "--z3rlimit 50"

(* First maximum of a real row. *)
let self (w:model_weights real) : Lemma (weights_approx N.exact w w) = ()

(* Weights that stand for each other give the same reference results. *)
let tokens_of (p:vector token_id prompt_length) : GTot (S.seq I64.t) =
  S.init_ghost prompt_length (fun (t:index prompt_length) -> I64.int_to_t (p t))

let same_pack (w rw:model_weights real)
  : Lemma (requires weights_approx N.exact w rw) (ensures Br.pack w == Br.pack rw)
  = let aux (i:nat{i < SS.packed_elements}) : Lemma (Br.pack_at w i == Br.pack_at rw i)
      = if i < SS.embed_elements then ()
        else if i < SS.layers_offset then ()
        else begin
          Br.div_bound (i - SS.layers_offset) num_hidden_layers SS.layer_elements;
          let l : index num_hidden_layers = (i - SS.layers_offset) / SS.layer_elements in
          assert (parameters_approx N.exact (w.layers l) (rw.layers l))
        end
    in
    Classical.forall_intro aux;
    S.lemma_eq_intro (Br.pack w) (Br.pack rw)

let congruent (w rw:model_weights real) (p:vector token_id prompt_length)
  : Lemma (requires weights_approx N.exact w rw)
          (ensures N.meq (last_hidden_state w p) (last_hidden_state rw p) /\
                   N.veq (next_token_logits w p) (next_token_logits rw p))
  = let tk = tokens_of p in
    assert (Br.prompt_eq tk p);
    same_pack w rw;
    Br.pack_corresponds w;
    Br.pack_corresponds rw;
    Br.final_sim (Br.pack w) w tk p;
    Br.final_sim (Br.pack rw) rw tk p;
    Br.logits_sim (Br.pack w) w tk p;
    Br.logits_sim (Br.pack rw) rw tk p;
    let aux (t:index prompt_length) (c:index hidden_size)
      : Lemma (last_hidden_state w p t c == last_hidden_state rw p t c)
      = Br.bound t c 7 1024
    in
    Classical.forall_intro_2 aux

let float_instance () : Lemma T.float_instance = ()

let nonempty () : Lemma T.nonempty =
  let w : model_weights real = {
    embed_tokens = (fun _ _ -> 0.0R); norm = (fun _ -> 0.0R);
    layers = (fun _ -> {
      input_layernorm = (fun _ -> 0.0R); post_attention_layernorm = (fun _ -> 0.0R);
      q_proj = (fun _ _ -> 0.0R); k_proj = (fun _ _ -> 0.0R); v_proj = (fun _ _ -> 0.0R);
      o_proj = (fun _ _ -> 0.0R); q_norm = (fun _ -> 0.0R); k_norm = (fun _ -> 0.0R);
      gate_proj = (fun _ _ -> 0.0R); up_proj = (fun _ _ -> 0.0R); down_proj = (fun _ _ -> 0.0R) }) } in
  let x : inputs_t real = { weights = w; prompt_tokens = (fun _ -> 0); decode_steps = 1 } in
  assert (input_domain x)

let answer (x:inputs_t real{input_domain x}) : GTot (outputs_t real) =
  let p = prompt_ids x in
  let j = N.first_max (next_token_logits x.weights p) vocab_size in
  { tokens = (fun (t:index sequence_length) -> if t < prompt_length then x.prompt_tokens t else j);
    hidden_states = last_hidden_state x.weights p }

let answer_ok (x:inputs_t real{input_domain x}) : Lemma (spec N.exact x (answer x)) =
  let p = prompt_ids x in
  let y = answer x in
  let num (rw:model_weights real)
    : Lemma (weights_approx N.exact x.weights rw ==>
             matrix_approx N.exact y.hidden_states (last_hidden_state rw p) /\
             vector_approx N.exact (next_token_logits x.weights p) (next_token_logits rw p))
    = introduce weights_approx N.exact x.weights rw ==>
                (matrix_approx N.exact y.hidden_states (last_hidden_state rw p) /\
                 vector_approx N.exact (next_token_logits x.weights p) (next_token_logits rw p))
      with congruent x.weights rw p
  in
  Classical.forall_intro num;
  assert (N.exact_argmax vocab_size (next_token_logits x.weights p) (y.tokens prompt_length));
  assert (selection_postcondition N.exact x y)

let answers () : Lemma T.answers =
  let aux (x:inputs_t real)
    : Lemma (input_domain x ==> (exists (y:outputs_t real). spec N.exact x y))
    = introduce input_domain x ==> (exists (y:outputs_t real). spec N.exact x y)
      with answer_ok x
  in
  Classical.forall_intro aux

let unique_one (x:inputs_t real) (y1 y2:outputs_t real)
  : Lemma (requires input_domain x /\ spec N.exact x y1 /\ spec N.exact x y2)
          (ensures N.veq y1.tokens y2.tokens /\ N.meq y1.hidden_states y2.hidden_states)
  = let p = prompt_ids x in
    self x.weights;
    assert (matrix_approx N.exact y1.hidden_states (last_hidden_state x.weights p));
    assert (matrix_approx N.exact y2.hidden_states (last_hidden_state x.weights p));
    eliminate exists (s1:vector real vocab_size).
        (forall (rw:model_weights real). weights_approx N.exact x.weights rw ==>
           vector_approx N.exact s1 (next_token_logits rw p)) /\
        N.exact_argmax vocab_size s1 (y1.tokens prompt_length)
    with eliminate exists (s2:vector real vocab_size).
        (forall (rw:model_weights real). weights_approx N.exact x.weights rw ==>
           vector_approx N.exact s2 (next_token_logits rw p)) /\
        N.exact_argmax vocab_size s2 (y2.tokens prompt_length)
    with begin
      assert (vector_approx N.exact s1 (next_token_logits x.weights p));
      assert (vector_approx N.exact s2 (next_token_logits x.weights p));
      N.argmax_unique s1 s2 (y1.tokens prompt_length) (y2.tokens prompt_length)
    end

let unique () : Lemma T.unique =
  let aux (x:inputs_t real) (y1 y2:outputs_t real)
    : Lemma (input_domain x /\ spec N.exact x y1 /\ spec N.exact x y2 ==>
             N.veq y1.tokens y2.tokens /\ N.meq y1.hidden_states y2.hidden_states)
    = Classical.move_requires (unique_one x y1) y2
  in
  Classical.forall_intro_3 aux

let proof () : Lemma T.tight =
  float_instance (); nonempty (); answers (); unique ()
