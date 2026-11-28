module KPlay.Tightness.Luce.Proof
(* Proof of the generated theorem KPlay.Tightness.Luce.Statement.tight. *)
open FStar.Real
open KPlay.Spec
module T = KPlay.Tightness.Luce.Statement
module N = KPlay.Numbers
module Br = KPlay.Spec.Bridge
module SS = KPlay.SharedSpec
module S = FStar.Seq

#set-options "--z3rlimit 50 --fuel 1 --ifuel 1"

(*** Congruence: equal weights, constants and earlier tokens give equal
     hidden states and logits. Both sides are compared with the same flat
     reference run (KPlay.Spec.Bridge). *)

(* The exact constants. *)
let exact_numerics : numerics = {
  rms_norm_eps = eps_exact;
  linear_query_scale = query_scale_exact;
  inv_freq = inv_freq_exact;
}

let exact_flat () : GTot SS.numerics = Br.nm_of exact_numerics

let numerics_flat (tn:numerics)
  : Lemma (requires numerics_ok N.exact tn) (ensures Br.nm_ok tn (exact_flat ()))
  = ()

(* Weights equal entry by entry agree with the same flat weights. *)
let common_tr (#m:Type0 -> Type0) (mx:m real -> m real -> prop)
    (p q:decoder_layer m real) (s:S.seq real) (o:int)
  : Lemma (requires layer_approx N.exact mx p q /\ Br.common_ok p s o) (ensures Br.common_ok q s o)
  = ()

let linear_tr (p q:decoder_layer linear_attn_params real) (s:S.seq real) (o:int)
  : Lemma (requires layer_approx N.exact (linear_attn_approx N.exact) p q /\ Br.linear_layer_ok p s o)
          (ensures Br.linear_layer_ok q s o)
  = common_tr (linear_attn_approx N.exact) p q s o

let full_tr (p q:decoder_layer self_attn_params real) (s:S.seq real) (o:int)
  : Lemma (requires layer_approx N.exact (self_attn_approx N.exact) p q /\ Br.full_layer_ok p s o)
          (ensures Br.full_layer_ok q s o)
  = common_tr (self_attn_approx N.exact) p q s o

let weights_tr (tw tw':model_weights real) (s:S.seq real)
  : Lemma (requires weights_approx N.exact tw tw' /\ Br.weights_ok tw s) (ensures Br.weights_ok tw' s)
  = let lin (d:index linear_layer_count)
      : Lemma (Br.linear_layer_ok (tw'.linear_layers d) s (SS.layer_off (4 * (d / 3) + d % 3)))
      = linear_tr (tw.linear_layers d) (tw'.linear_layers d) s (SS.layer_off (4 * (d / 3) + d % 3))
    in
    let full (g:index full_layer_count)
      : Lemma (Br.full_layer_ok (tw'.full_layers g) s (SS.layer_off (4 * g + 3)))
      = full_tr (tw.full_layers g) (tw'.full_layers g) s (SS.layer_off (4 * g + 3))
    in
    Classical.forall_intro lin;
    Classical.forall_intro full

(* A token function agrees with a total one on positions 0..p. *)
let extend (toks:nat -> GTot int) (p:pos) (f:nat -> int) (c:int)
  : Lemma (requires (forall (q:nat). q <= p - 1 ==> f q == toks q) /\ c == toks p)
          (ensures exists (g:nat -> int). forall (q:nat). q <= p ==> g q == toks q)
  = let g (q:nat) : int = if q = p then c else f q in
    assert (forall (q:nat). q <= p ==> g q == toks q)

let rec total_prefix (toks:nat -> GTot int) (p:nat)
  : Lemma (ensures exists (f:nat -> int). forall (q:nat). q <= p ==> f q == toks q) (decreases p)
  = assert (exists (c:int). c == toks p);
    if p = 0 then
      eliminate exists (c:int). c == toks p
      with (let g (q:nat) : int = c in
            assert (forall (q:nat). q <= p ==> g q == toks q))
    else begin
      total_prefix toks (p - 1);
      eliminate exists (f:nat -> int). forall (q:nat). q <= p - 1 ==> f q == toks q
      with eliminate exists (c:int). c == toks p
      with extend toks p f c
    end

let tokens_agree (t1 t2:nat -> GTot int) (p:nat) : prop =
  forall (q:nat). q <= p ==> t1 q == t2 q /\ valid_token (t1 q)

let congruent (tn1 tn2:numerics) (tw1 tw2:model_weights real) (s:S.seq real)
              (t1 t2:nat -> GTot int) (p:nat)
  : Lemma (requires Br.nm_ok tn1 (exact_flat ()) /\ Br.nm_ok tn2 (exact_flat ()) /\
                    Br.weights_ok tw1 s /\ Br.weights_ok tw2 s /\ tokens_agree t1 t2 p)
          (ensures N.veq (hidden_at tn1 tw1 t1 p) (hidden_at tn2 tw2 t2 p) /\
                   N.veq (logits tn1 tw1 (hidden_at tn1 tw1 t1 p)) (logits tn2 tw2 (hidden_at tn2 tw2 t2 p)))
  = total_prefix t1 p;
    eliminate exists (f:nat -> int). forall (q:nat). q <= p ==> f q == t1 q
    with begin
      let hs = SS.ref_hidden (exact_flat ()) s f p in
      Br.hidden_eq tn1 (exact_flat ()) tw1 s t1 f p;
      Br.hidden_eq tn2 (exact_flat ()) tw2 s t2 f p;
      Br.logits_eq tn1 (exact_flat ()) tw1 s (hidden_at tn1 tw1 t1 p) hs;
      Br.logits_eq tn2 (exact_flat ()) tw2 s (hidden_at tn2 tw2 t2 p) hs
    end

(*** The statement *)

let float_instance () : Lemma T.float_instance = ()

let zero_mlp : mlp_params real =
  { gate_proj = (fun _ _ -> 0.0R); up_proj = (fun _ _ -> 0.0R); down_proj = (fun _ _ -> 0.0R) }

let zero_weights : model_weights real = {
  embed_tokens = (fun _ _ -> 0.0R);
  final_norm = (fun _ -> 0.0R);
  linear_layers = (fun _ -> {
    input_layernorm = (fun _ -> 0.0R); post_attention_layernorm = (fun _ -> 0.0R); mlp = zero_mlp;
    token_mixer = {
      in_proj_qkv = (fun _ _ -> 0.0R); in_proj_z = (fun _ _ -> 0.0R); in_proj_b = (fun _ _ -> 0.0R);
      in_proj_a = (fun _ _ -> 0.0R); conv1d = (fun _ _ -> 0.0R); a_log = (fun _ -> 0.0R);
      dt_bias = (fun _ -> 0.0R); norm = (fun _ -> 0.0R); out_proj = (fun _ _ -> 0.0R) } });
  full_layers = (fun _ -> {
    input_layernorm = (fun _ -> 0.0R); post_attention_layernorm = (fun _ -> 0.0R); mlp = zero_mlp;
    token_mixer = {
      q_proj = (fun _ _ -> 0.0R); k_proj = (fun _ _ -> 0.0R); v_proj = (fun _ _ -> 0.0R);
      q_norm = (fun _ -> 0.0R); k_norm = (fun _ -> 0.0R); o_proj = (fun _ _ -> 0.0R) } });
}

let nonempty () : Lemma T.nonempty =
  let x : inputs_t real = { weights = zero_weights; prompt_length = 1; prompt = (fun _ -> 0); decode_steps = 1 } in
  assert (input_domain x)

(* The answer: each generated token is the first maximum of the logits at
   its position, given the tokens before it. *)
let domain_input = x:inputs_t real{input_domain x}

(* The tokens consumed when the generated tokens so far are s. *)
let toks_of (x:domain_input) (s:S.seq int) (p:nat) : GTot int =
  if p < x.prompt_length then x.prompt p
  else if p - x.prompt_length < S.length s then S.index s (p - x.prompt_length)
  else 0

let next_logits (x:domain_input) (s:S.seq int) : GTot (vector real vocab_size) =
  logits exact_numerics x.weights
    (hidden_at exact_numerics x.weights (toks_of x s) (last_prompt_position x + S.length s))

let next (x:domain_input) (s:S.seq int)
  : GTot (t:int{valid_token t /\ N.exact_argmax vocab_size (next_logits x s) t}) =
  N.first_max (next_logits x s) vocab_size

let rec gen (x:domain_input) (j:nat)
  : GTot (s:S.seq int{S.length s == j /\ (forall (i:nat). i < j ==> valid_token (S.index s i))})
    (decreases j) =
  if j = 0 then S.empty
  else let s = gen x (j - 1) in S.snoc s (next x s)

let rec gen_prefix (x:domain_input) (i:nat) (k:nat{i < k})
  : Lemma (ensures S.index (gen x k) i == next x (gen x i)) (decreases k)
  = if k > i + 1 then gen_prefix x i (k - 1)

let steps (x:domain_input) : nat = x.decode_steps

let answer (x:domain_input) : GTot (outputs_t real) = {
  generated_count = steps x;
  tokens = (fun (j:index (steps x)) -> S.index (gen x (steps x)) j);
  hidden_states = (fun (j:index (steps x)) ->
    hidden_at exact_numerics x.weights (toks_of x (gen x (steps x))) (last_prompt_position x + j));
}

let answer_tokens (x:domain_input) (j:index (steps x))
  : Lemma (tokens_agree (toks_of x (gen x j)) (consumed x (answer x)) (last_prompt_position x + j) /\
           tokens_agree (toks_of x (gen x (steps x))) (consumed x (answer x)) (last_prompt_position x + j) /\
           S.index (gen x (steps x)) j == next x (gen x j))
  = let aux (i:nat{i < j}) : Lemma (S.index (gen x j) i == S.index (gen x (steps x)) i)
      = gen_prefix x i j; gen_prefix x i (steps x)
    in
    Classical.forall_intro aux;
    gen_prefix x j (steps x)

let exact_argmax_tr (#n:nat) (s1 s2:vector real n) (j:int)
  : Lemma (requires N.veq s1 s2 /\ N.exact_argmax n s1 j) (ensures N.exact_argmax n s2 j)
  = ()

let answer_at (x:domain_input) (w:model_weights real) (j:index (steps x))
  : Lemma (requires weights_approx N.exact x.weights w)
          (ensures hidden_state_relation N.exact exact_numerics w (consumed x (answer x))
                     (last_prompt_position x + j) ((answer x).hidden_states j) /\
                   greedy_choice_relation N.exact N.exact exact_numerics w
                     (hidden_at exact_numerics w (consumed x (answer x)) (last_prompt_position x + j))
                     ((answer x).tokens j))
  = let y = answer x in
    let p = last_prompt_position x + j in
    let s = Br.flat_of x.weights in
    Br.flat_of_ok x.weights;
    weights_tr x.weights w s;
    numerics_flat exact_numerics;
    answer_tokens x j;
    congruent exact_numerics exact_numerics x.weights w s (toks_of x (gen x (steps x))) (consumed x y) p;
    congruent exact_numerics exact_numerics x.weights w s (toks_of x (gen x j)) (consumed x y) p;
    let row = logits exact_numerics w (hidden_at exact_numerics w (consumed x y) p) in
    exact_argmax_tr (next_logits x (gen x j)) row (y.tokens j);
    assert (N.exact_argmax vocab_size row (y.tokens j));
    assert (vector_approx N.exact row row);
    assert (greedy_choice_relation N.exact N.exact exact_numerics w
              (hidden_at exact_numerics w (consumed x y) p) (y.tokens j));
    assert (hidden_state_relation N.exact exact_numerics w (consumed x y) p (y.hidden_states j))

let answer_ok (x:domain_input) : Lemma (spec N.exact N.exact x (answer x)) =
  let y = answer x in
  let run (w:model_weights real)
    : Lemma (weights_approx N.exact x.weights w ==>
             numerical N.exact exact_numerics x y w /\ selection N.exact N.exact exact_numerics x y w)
    = introduce weights_approx N.exact x.weights w ==>
                (numerical N.exact exact_numerics x y w /\ selection N.exact N.exact exact_numerics x y w)
      with Classical.forall_intro (Classical.move_requires (answer_at x w))
  in
  Classical.forall_intro run;
  assert (numerics_ok N.exact exact_numerics);
  assert (structural x y)

let answers () : Lemma T.answers =
  let aux (x:inputs_t real)
    : Lemma (input_domain x ==> (exists (y:outputs_t real). spec N.exact N.exact x y))
    = introduce input_domain x ==> (exists (y:outputs_t real). spec N.exact N.exact x y)
      with answer_ok x
  in
  Classical.forall_intro aux

(* Uniqueness: by induction on j, token j and hidden state j of two answers
   are equal, given the tokens before j. *)
let self (w:model_weights real) : Lemma (weights_approx N.exact w w) = ()

let reads (x:inputs_t real) (y:outputs_t real) (nm:numerics) (j:index y.generated_count) : prop =
  N.veq (y.hidden_states j) (hidden_at nm x.weights (consumed x y) (last_prompt_position x + j)) /\
  (exists (row:vector real vocab_size).
     N.veq row (logits nm x.weights (hidden_at nm x.weights (consumed x y) (last_prompt_position x + j))) /\
     N.exact_argmax vocab_size row (y.tokens j))

let spec_reads (x:inputs_t real) (y:outputs_t real)
  : Lemma (requires input_domain x /\ spec N.exact N.exact x y)
          (ensures structural x y /\
                   (exists (nm:numerics). Br.nm_ok nm (exact_flat ()) /\
                      (forall (j:index y.generated_count). reads x y nm j)))
  = eliminate exists (nm:numerics). numerics_ok N.exact nm /\
      (forall (w:model_weights real). weights_approx N.exact x.weights w ==>
         numerical N.exact nm x y w /\ selection N.exact N.exact nm x y w)
    with begin
      self x.weights;
      numerics_flat nm;
      assert (forall (j:index y.generated_count). reads x y nm j)
    end

let same_before (x:inputs_t real) (y1 y2:outputs_t real) (j:nat) : prop =
  y1.generated_count == y2.generated_count /\
  (forall (i:nat). i < j /\ i < y1.generated_count ==> y1.tokens i == y2.tokens i)

let step (x:inputs_t real) (y1 y2:outputs_t real) (nm1 nm2:numerics) (j:index y1.generated_count)
  : Lemma (requires input_domain x /\ structural x y1 /\ structural x y2 /\
                    Br.nm_ok nm1 (exact_flat ()) /\ Br.nm_ok nm2 (exact_flat ()) /\
                    reads x y1 nm1 j /\ reads x y2 nm2 j /\ same_before x y1 y2 j)
          (ensures y1.tokens j == y2.tokens j /\ N.veq (y1.hidden_states j) (y2.hidden_states j))
  = let s = Br.flat_of x.weights in
    let p = last_prompt_position x + j in
    Br.flat_of_ok x.weights;
    assert (tokens_agree (consumed x y1) (consumed x y2) p);
    congruent nm1 nm2 x.weights x.weights s (consumed x y1) (consumed x y2) p;
    eliminate exists (r1:vector real vocab_size).
        N.veq r1 (logits nm1 x.weights (hidden_at nm1 x.weights (consumed x y1) p)) /\
        N.exact_argmax vocab_size r1 (y1.tokens j)
    with eliminate exists (r2:vector real vocab_size).
        N.veq r2 (logits nm2 x.weights (hidden_at nm2 x.weights (consumed x y2) p)) /\
        N.exact_argmax vocab_size r2 (y2.tokens j)
    with N.argmax_unique r1 r2 (y1.tokens j) (y2.tokens j)

let rec tokens_equal (x:inputs_t real) (y1 y2:outputs_t real) (nm1 nm2:numerics) (j:nat)
  : Lemma (requires input_domain x /\ structural x y1 /\ structural x y2 /\
                    Br.nm_ok nm1 (exact_flat ()) /\ Br.nm_ok nm2 (exact_flat ()) /\
                    (forall (i:index y1.generated_count). reads x y1 nm1 i) /\
                    (forall (i:index y2.generated_count). reads x y2 nm2 i))
          (ensures same_before x y1 y2 j) (decreases j)
  = if j > 0 then begin
      tokens_equal x y1 y2 nm1 nm2 (j - 1);
      if j - 1 < y1.generated_count then step x y1 y2 nm1 nm2 (j - 1)
    end

let unique_one (x:inputs_t real) (y1 y2:outputs_t real)
  : Lemma (requires input_domain x /\ spec N.exact N.exact x y1 /\ spec N.exact N.exact x y2)
          (ensures y1.generated_count == y2.generated_count /\
                   N.veq y1.tokens y2.tokens /\ N.meq y1.hidden_states y2.hidden_states)
  = spec_reads x y1;
    spec_reads x y2;
    eliminate exists (nm1:numerics). Br.nm_ok nm1 (exact_flat ()) /\
      (forall (j:index y1.generated_count). reads x y1 nm1 j)
    with eliminate exists (nm2:numerics). Br.nm_ok nm2 (exact_flat ()) /\
      (forall (j:index y2.generated_count). reads x y2 nm2 j)
    with begin
      tokens_equal x y1 y2 nm1 nm2 y1.generated_count;
      let hid (j:index y1.generated_count)
        : Lemma (N.veq (y1.hidden_states j) (y2.hidden_states j))
        = step x y1 y2 nm1 nm2 j
      in
      Classical.forall_intro hid
    end

let unique () : Lemma T.unique =
  let aux (x:inputs_t real) (y1 y2:outputs_t real)
    : Lemma (input_domain x /\ spec N.exact N.exact x y1 /\ spec N.exact N.exact x y2 ==>
             y1.generated_count == y2.generated_count /\
             N.veq y1.tokens y2.tokens /\ N.meq y1.hidden_states y2.hidden_states)
    = Classical.move_requires (unique_one x y1) y2
  in
  Classical.forall_intro_3 aux

let proof () : Lemma T.tight =
  float_instance (); nonempty (); answers (); unique ()
