module KPlay.Tightness.Flashmoe.Proof
(* Proof of the generated theorem KPlay.Tightness.Flashmoe.Statement.tight. *)
open FStar.Real
open KPlay.Spec
module N = KPlay.Numbers
module T = KPlay.Tightness.Flashmoe.Statement
module I32 = FStar.Int32
module I64 = FStar.Int64

#set-options "--z3rlimit 50 --fuel 1 --ifuel 1"

type choices = vector (index expert_count) token_count

let float_instance () : Lemma T.float_instance = ()

let nonempty () : Lemma T.nonempty =
  let x : inputs_t real = { tokens = (fun _ _ -> 0.0R); router_weights = (fun _ _ -> 0.0R) } in
  assert (input_domain x)

(* Inputs that stand for each other give the same reference probabilities. *)
let rec sum_eq (#n:nat) (f g:vector real n) (k:nat{k <= n})
  : Lemma (requires N.veq f g) (ensures sum f k == sum g k) (decreases k)
  = if k > 0 then sum_eq f g (k - 1)

let probabilities_eq (r1 r2:inputs_t real) (t:index token_count) (e:index expert_count)
  : Lemma (requires inputs_approx N.exact r1 r2)
          (ensures router_probabilities r1 t e == router_probabilities r2 t e)
  = let v1 = scores r1.tokens r1.router_weights t in
    let v2 = scores r2.tokens r2.router_weights t in
    let at (e:index expert_count) : Lemma (v1 e == v2 e) =
      sum_eq (products r1.tokens r1.router_weights t e)
             (products r2.tokens r2.router_weights t e) hidden_size
    in
    Classical.forall_intro at;
    sum_eq (exps v1) (exps v2) expert_count

let softmax_pos (m:inputs_t real) (t:index token_count) (e:index expert_count)
  : Lemma (router_probabilities m t e >. 0.0R)
  = let v = scores m.tokens m.router_weights t in
    sum_exps_positive v expert_count;
    FStar.Math.Exp.exp_positive (v e)

(* Capacity routing depends only on the choices. *)
let rec assigned_eq (s1 s2:choices) (e:index expert_count) (n:nat{n <= token_count})
  : Lemma (requires N.veq s1 s2) (ensures assigned_before s1 e n == assigned_before s2 e n)
          (decreases n)
  = if n > 0 then assigned_eq s1 s2 e (n - 1)

let survival_eq (s1 s2:choices) (t:index token_count) (e:index expert_count)
  : Lemma (requires N.veq s1 s2) (ensures survival_of s1 t e == survival_of s2 t e)
  = assigned_eq s1 s2 e (t + 1)

let rec survivors_eq (s1 s2:choices) (e:index expert_count) (n:nat{n <= token_count})
  : Lemma (requires N.veq s1 s2) (ensures survivors_before s1 e n == survivors_before s2 e n)
          (decreases n)
  = if n > 0 then (survivors_eq s1 s2 e (n - 1); survival_eq s1 s2 (n - 1) e)

let rec kept_eq (s1 s2:choices) (e:index expert_count) (c:index capacity) (n:nat{n <= token_count})
  : Lemma (requires N.veq s1 s2) (ensures kept_token s1 e c n == kept_token s2 e c n)
          (decreases n)
  = if n > 0 then begin
      survival_eq s1 s2 (n - 1) e;
      survivors_eq s1 s2 e (n - 1);
      kept_eq s1 s2 e c (n - 1)
    end

let rec assigned_le (s:choices) (e:index expert_count) (n:nat{n <= token_count})
  : Lemma (ensures assigned_before s e n <= n) (decreases n)
  = if n > 0 then assigned_le s e (n - 1)

(* A kept token survives for its expert. *)
let rec kept_survives (s:choices) (e:index expert_count) (c:index capacity) (n:nat{n <= token_count})
  : Lemma (ensures (let t = kept_token s e c n in t >= 0 ==> survival_of s t e = 1))
          (decreases n)
  = if n > 0 then kept_survives s e c (n - 1)

(* The answer: the reference probabilities, their first maxima, weight 1. *)
let choose (x:inputs_t real) : choices =
  fun t -> N.first_max (router_probabilities x t) expert_count

let run_of (x:inputs_t real) : routing_t real = {
  probability_rows = router_probabilities x;
  selected = choose x;
  combine_probability = (fun _ -> 1.0R);
}

let answer (x:inputs_t real) : outputs_t real =
  let s = choose x in {
    expert_demand = (fun e -> assigned_le s e token_count; I32.int_to_t (demand s e));
    token_indices = (fun e c -> I64.int_to_t (kept_token s e c token_count));
    probabilities = (fun e c -> kept_probability N.exact s (fun _ -> 1.0R) e c);
    survival = (fun t e -> I32.int_to_t (survival_of s t e));
  }

let answer_structural (x:inputs_t real) : Lemma (structural N.exact (run_of x) (answer x)) =
  let s = choose x in
  let y = answer x in
  let d (e:index expert_count) : Lemma (I32.v (y.expert_demand e) == demand s e) =
    assigned_le s e token_count in
  let ix (e:index expert_count) (c:index capacity)
    : Lemma (I64.v (y.token_indices e c) == kept_token s e c token_count) = () in
  let pr (e:index expert_count) (c:index capacity)
    : Lemma (y.probabilities e c == kept_probability N.exact s (fun _ -> 1.0R) e c) = () in
  let sv (t:index token_count) (e:index expert_count)
    : Lemma (I32.v (y.survival t e) == survival_of s t e) = () in
  Classical.forall_intro d;
  Classical.forall_intro_2 ix;
  Classical.forall_intro_2 pr;
  Classical.forall_intro_2 sv;
  assert (capacity_routing_relation N.exact s (fun _ -> 1.0R) y)

let answer_ok (x:inputs_t real) : Lemma (spec N.exact N.exact x (answer x)) =
  let run = run_of x in
  answer_structural x;
  let rel (r:inputs_t real)
    : Lemma (inputs_approx N.exact x r ==>
             matrix_approx N.exact run.probability_rows (router_probabilities r))
    = introduce inputs_approx N.exact x r ==>
                matrix_approx N.exact run.probability_rows (router_probabilities r)
      with Classical.forall_intro_2 (Classical.move_requires_2 (probabilities_eq x r))
  in
  Classical.forall_intro rel;
  assert (structural N.exact run (answer x));
  assert (numerical N.exact N.exact x run);
  assert (selection N.exact run)

let answers () : Lemma T.answers =
  Classical.forall_intro answer_ok

(* Two answers agree. *)
let same_runs (x:inputs_t real) (y1 y2:outputs_t real) (r1 r2:routing_t real)
  : Lemma (requires structural N.exact r1 y1 /\ numerical N.exact N.exact x r1 /\
                    selection N.exact r1 /\
                    structural N.exact r2 y2 /\ numerical N.exact N.exact x r2 /\
                    selection N.exact r2)
          (ensures N.veq y1.expert_demand y2.expert_demand /\
                   N.meq y1.token_indices y2.token_indices /\
                   N.meq y1.probabilities y2.probabilities /\
                   N.meq y1.survival y2.survival)
  = assert (inputs_approx N.exact x x);
    let row (t:index token_count) (e:index expert_count)
      : Lemma (r1.probability_rows t e == router_probabilities x t e /\
               r2.probability_rows t e == router_probabilities x t e)
      = ()
    in
    Classical.forall_intro_2 row;
    let pick (t:index token_count) : Lemma (r1.selected t == r2.selected t) =
      N.argmax_unique (r1.probability_rows t) (r2.probability_rows t) (r1.selected t) (r2.selected t)
    in
    Classical.forall_intro pick;
    let s1 = r1.selected in
    let s2 = r2.selected in
    let d (e:index expert_count) : Lemma (y1.expert_demand e == y2.expert_demand e) =
      assigned_eq s1 s2 e token_count;
      I32.v_inj (y1.expert_demand e) (y2.expert_demand e)
    in
    Classical.forall_intro d;
    let ix (e:index expert_count) (c:index capacity)
      : Lemma (y1.token_indices e c == y2.token_indices e c)
      = kept_eq s1 s2 e c token_count;
        I64.v_inj (y1.token_indices e c) (y2.token_indices e c)
    in
    Classical.forall_intro_2 ix;
    let sv (t:index token_count) (e:index expert_count)
      : Lemma (y1.survival t e == y2.survival t e)
      = survival_eq s1 s2 t e;
        I32.v_inj (y1.survival t e) (y2.survival t e)
    in
    Classical.forall_intro_2 sv;
    let pr (e:index expert_count) (c:index capacity)
      : Lemma (y1.probabilities e c == y2.probabilities e c)
      = kept_eq s1 s2 e c token_count;
        let t = kept_token s1 e c token_count in
        if t >= 0 then begin
          let t : index token_count = t in
          kept_survives s1 e c token_count;
          survival_eq s1 s2 t e;
          softmax_pos x t e;
          let p = router_probabilities x t e in
          assert (r1.combine_probability t == combine_weight p);
          assert (r2.combine_probability t == combine_weight p)
        end
    in
    Classical.forall_intro_2 pr

let unique () : Lemma T.unique =
  let one (x:inputs_t real) (y1 y2:outputs_t real)
    : Lemma (input_domain x /\ spec N.exact N.exact x y1 /\ spec N.exact N.exact x y2 ==>
             N.veq y1.expert_demand y2.expert_demand /\
             N.meq y1.token_indices y2.token_indices /\
             N.meq y1.probabilities y2.probabilities /\
             N.meq y1.survival y2.survival)
    = let both (r1 r2:routing_t real)
        : Lemma ((structural N.exact r1 y1 /\ numerical N.exact N.exact x r1 /\ selection N.exact r1 /\
                  structural N.exact r2 y2 /\ numerical N.exact N.exact x r2 /\ selection N.exact r2) ==>
                 N.veq y1.expert_demand y2.expert_demand /\
                 N.meq y1.token_indices y2.token_indices /\
                 N.meq y1.probabilities y2.probabilities /\
                 N.meq y1.survival y2.survival)
        = Classical.move_requires (same_runs x y1 y2 r1) r2
      in
      Classical.forall_intro_2 both
  in
  Classical.forall_intro_3 one

let proof () : Lemma T.tight =
  float_instance (); nonempty (); answers (); unique ()
