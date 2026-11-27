module KPlay.Tightness.Mirage.Proof
(* Proof of the generated theorem KPlay.Tightness.Mirage.Statement.tight. *)
open FStar.Real
open Kuiper.Approximates
module K = KPlay.Spec
module Q = KPlay.Tightness.Mirage.Question
module T = KPlay.Tightness.Mirage.Statement
module A = KPlay.Tightness.Answer
module S = FStar.Seq
module BF = Kuiper.BFloat16

#set-options "--z3rlimit 50"

let cell (i:nat{i < Q.count})
  : Lemma (i / K.hidden_size < K.prompt_length /\
           (i / K.hidden_size) * K.hidden_size + i % K.hidden_size == i)
  = FStar.Math.Lemmas.euclidean_division_definition i K.hidden_size

let one (x:Q.input{Q.domain x}) (y:T.answer)
  : Lemma (requires Q.spec x y) (ensures exists (c:Q.choice x). Q.allowed x c /\ T.pinned x c y)
  = let o = Q.view y in
    let t : int = S.index y.A.exact K.prompt_length in
    assert (K.selection_postcondition x o);
    eliminate exists (scores:K.vector BF.t K.vocab_size).
        (forall (m:Q.model). Q.models x m ==>
           K.vector_approximates scores (K.next_token_logits m (K.prompt_ids x))) /\
        K.argmax_choice scores t
    with begin
      let c : Q.choice x = t in
      assert (Q.allowed x c);
      S.lemma_eq_intro y.A.exact (Q.exact_target x c);
      let num (m:Q.model) (i:nat{i < Q.count})
        : Lemma (Q.models x m ==> S.index y.A.numbers i %~ S.index (Q.number_target x m c) i)
        = cell i;
          assert (K.numerical_postcondition x o)
      in
      Classical.forall_intro_2 num;
      assert (T.pinned x c y)
    end

let proof () : Lemma T.tight =
  assert (Q.domain Q.example);
  let aux (x:Q.input{Q.domain x}) (y:T.answer)
    : Lemma (Q.spec x y ==> (exists (c:Q.choice x). Q.allowed x c /\ T.pinned x c y))
    = introduce Q.spec x y ==> (exists (c:Q.choice x). Q.allowed x c /\ T.pinned x c y)
      with one x y
  in
  Classical.forall_intro_2 aux
