module KPlay.Tightness.Luce.Proof
(* Proof of the generated theorem KPlay.Tightness.Luce.Statement.tight. *)
open FStar.Real
open Kuiper.Approximates
module K = KPlay.Spec
module Q = KPlay.Tightness.Luce.Question
module T = KPlay.Tightness.Luce.Statement
module A = KPlay.Tightness.Answer
module S = FStar.Seq

#set-options "--z3rlimit 50 --fuel 1 --ifuel 1"

(* decode reads the token function only at positions below n. *)
let rec decode_congr (nm:K.numerics) (w:Q.model) (f g:nat -> GTot int) (n:nat)
  : Lemma (requires forall (p:nat). p < n ==> f p == g p)
          (ensures K.decode nm w f n == K.decode nm w g n)
          (decreases n)
  = if n > 0 then decode_congr nm w f g (n - 1)

let hidden_eq (x:Q.input{Q.domain x}) (y:T.answer) (m:Q.model) (j:nat)
  : Lemma (K.hidden_at K.model_numerics m (K.consumed x (Q.view y)) (K.last_prompt_position x + j)
           == Q.hidden x m y.A.exact j)
  = decode_congr K.model_numerics m (K.consumed x (Q.view y)) (Q.toks x y.A.exact)
      (K.last_prompt_position x + j + 1)

let cell (i:nat) (n:pos)
  : Lemma ((i / n) * n + i % n == i /\ i % n < n)
  = FStar.Math.Lemmas.euclidean_division_definition i n

let cell_lt (i:nat) (n:pos) (k:nat)
  : Lemma (requires i < k * n) (ensures i / n < k)
  = cell i n

let one (x:Q.input{Q.domain x}) (y:T.answer)
  : Lemma (requires Q.spec x y) (ensures exists (c:Q.choice x). Q.allowed x c /\ T.pinned x c y)
  = let o = Q.view y in
    let c : Q.choice x = y.A.exact in
    assert (K.structural x o);
    let sel (j:nat{j < S.length c}) (m:Q.model)
      : Lemma (Q.models x m ==>
               K.greedy_choice_relation K.model_numerics m (Q.hidden x m c j) (S.index c j))
      = hidden_eq x y m j;
        assert (Q.models x m ==> K.selection K.model_numerics x o m)
    in
    Classical.forall_intro_2 sel;
    assert (Q.allowed x c);
    let num (m:Q.model) (i:nat{i < Q.count c})
      : Lemma (Q.models x m ==> S.index y.A.numbers i %~ S.index (Q.number_target x m c) i)
      = cell i K.hidden_size; cell_lt i K.hidden_size (S.length c);
        let j = i / K.hidden_size in
        hidden_eq x y m j;
        assert (Q.models x m ==> K.numerical K.model_numerics x o m);
        assert (Q.models x m ==> K.hidden_state_relation K.model_numerics m (K.consumed x o)
                                   (K.last_prompt_position x + j) (o.K.hidden_states j))
    in
    Classical.forall_intro_2 num;
    assert (T.pinned x c y)

let proof () : Lemma T.tight =
  assert (Q.domain Q.example);
  let aux (x:Q.input{Q.domain x}) (y:T.answer)
    : Lemma (Q.spec x y ==> (exists (c:Q.choice x). Q.allowed x c /\ T.pinned x c y))
    = introduce Q.spec x y ==> (exists (c:Q.choice x). Q.allowed x c /\ T.pinned x c y)
      with one x y
  in
  Classical.forall_intro_2 aux
