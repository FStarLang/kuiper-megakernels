module KPlay.Tightness.Flashmoe.Proof
(* Proof of the generated theorem KPlay.Tightness.Flashmoe.Statement.tight. *)
open FStar.Real
open Kuiper.Approximates
module K = KPlay.Spec
module Q = KPlay.Tightness.Flashmoe.Question
module T = KPlay.Tightness.Flashmoe.Statement
module A = KPlay.Tightness.Answer
module S = FStar.Seq
module F32 = Kuiper.Float32
module I32 = FStar.Int32
module I64 = FStar.Int64

#set-options "--z3rlimit 50 --fuel 1 --ifuel 1"

let cell (i:nat) (n:pos)
  : Lemma ((i / n) * n + i % n == i /\ i % n < n)
  = FStar.Math.Lemmas.euclidean_division_definition i n

let cell_lt (i:nat) (n:pos) (k:nat)
  : Lemma (requires i < k * n) (ensures i / n < k)
  = cell i n

(* A kept token survives for its expert. *)
let rec kept_survives (c:K.vector (K.index K.expert_count) K.token_count)
  (e:K.index K.expert_count) (s:K.index K.capacity) (n:nat{n <= K.token_count})
  : Lemma (ensures (let t = K.kept_token c e s n in
                    t >= 0 ==> K.survival_of c t e = 1))
          (decreases n)
  = if n > 0 then kept_survives c e s (n - 1)

let softmax_pos (m:K.router_inputs real) (t:K.index K.token_count) (e:K.index K.expert_count)
  : Lemma (K.router_probabilities m t e >. 0.0R)
  = let v = K.scores m.K.tokens m.K.router_weights t in
    K.sum_exps_positive v K.expert_count;
    FStar.Math.Exp.exp_positive (v e)

let one (x:Q.input{Q.domain x}) (y:T.answer)
  : Lemma (requires Q.spec x y) (ensures exists (c:Q.choice x). Q.allowed x c /\ T.pinned x c y)
  = let o = Q.view y in
    eliminate exists (run:K.routing). K.structural run o /\ K.numerical x run /\ K.selection run
    with begin
      let c : Q.choice x = run.K.selected in
      assert (Q.allowed x c);
      let tg = Q.exact_target x c in
      let d (e:nat{e < K.expert_count}) : Lemma (S.index y.A.exact.Q.demand e == S.index tg.Q.demand e)
        = Q.assigned_le c e K.token_count;
          I32.v_inj (S.index y.A.exact.Q.demand e) (S.index tg.Q.demand e) in
      Classical.forall_intro d;
      S.lemma_eq_intro y.A.exact.Q.demand tg.Q.demand;
      let ix (i:nat{i < K.expert_count * K.capacity}) : Lemma (S.index y.A.exact.Q.indices i == S.index tg.Q.indices i)
        = cell i K.capacity; cell_lt i K.capacity K.expert_count;
          I64.v_inj (S.index y.A.exact.Q.indices i) (S.index tg.Q.indices i) in
      Classical.forall_intro ix;
      S.lemma_eq_intro y.A.exact.Q.indices tg.Q.indices;
      let sv (i:nat{i < K.token_count * K.expert_count}) : Lemma (S.index y.A.exact.Q.survival i == S.index tg.Q.survival i)
        = cell i K.expert_count; cell_lt i K.expert_count K.token_count;
          I32.v_inj (S.index y.A.exact.Q.survival i) (S.index tg.Q.survival i) in
      Classical.forall_intro sv;
      S.lemma_eq_intro y.A.exact.Q.survival tg.Q.survival;
      assert (y.A.exact == tg);
      let num (m:Q.model) (i:nat{i < Q.cells})
        : Lemma (Q.models x m ==> S.index y.A.numbers i %~ S.index (Q.number_target x m c) i)
        = cell i K.capacity; cell_lt i K.capacity K.expert_count;
          let e : K.index K.expert_count = i / K.capacity in
          let s : K.index K.capacity = i % K.capacity in
          assert (S.index y.A.numbers i == o.K.probabilities e s);
          let t = K.kept_token c e s K.token_count in
          if t >= 0 then begin
            kept_survives c e s K.token_count;
            softmax_pos m t e;
            let p = K.router_probabilities m t e in
            assert (Q.models x m ==> run.K.probability_rows t e %~ p);
            assert (Q.models x m ==> run.K.combine_probability t %~ K.combine_weight p);
            assert (K.combine_weight p == 1.0R)
          end
          else Kuiper.Approximates.Scalar.a0 #F32.t
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
