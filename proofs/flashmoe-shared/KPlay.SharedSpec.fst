module KPlay.SharedSpec

(* Shared specification of the FlashMoE capacity-pruned top-1 router
   (reference: base.py reference_route / Model.forward).

   Shapes: tokens x [256, 512] BF16, router weights w [512, 8] BF16 (logical
   [hidden, experts] layout for both CUDA programs), 8 experts, capacity 32.
   Outputs: expert demand [8] (int32), token indices [8, 32] (int64), kept combine
   probabilities [8, 32] (float32), survival [256, 8] (int32).

   Contents:
   (1) reference math over reals (scores, softmax, top-1, combine, capacity);
   (2) the role-independent program postcondition;
   (3) one relation per role and family (G1);
   (4) helper lemmas for the composition (selection and routing chain). *)

open Kuiper
open Kuiper.Approximates
module S = FStar.Seq
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32
module I32 = FStar.Int32
module I64 = FStar.Int64
module FB = Kuiper.Floating.Base

(* ------------------------------------------------------------------ *)
(* Shapes                                                               *)

let tokens : nat = 256
let hidden : nat = 512
let experts : nat = 8
let capacity : nat = 32
let slots : nat = 256 (* experts * capacity *)

(* Row t (experts entries) of a flat row-major [_, 8] sequence. *)
let row_of (#a:Type) (s:S.seq a) (t:nat) : S.seq a =
  if 8 * t + 8 <= S.length s then S.slice s (8 * t) (8 * t + 8) else S.empty

(* ------------------------------------------------------------------ *)
(* (1) Reference math over reals                                        *)

let ridx (s:S.seq real) (i:int) : real =
  if 0 <= i && i < S.length s then S.index s i else 0.0R

(* Ascending dot product of row t of rx [_, hd] with column e of rw [hd, ed],
   over the first n hidden positions. *)
let rec dot_prefix (hd ed:nat) (rx rw:S.seq real) (t e:nat) (n:nat)
  : Tot real (decreases n)
  = if n = 0 then 0.0R
    else dot_prefix hd ed rx rw t e (n - 1) +.
         ridx rx (t * hd + (n - 1)) *. ridx rw ((n - 1) * ed + e)

(* FP32-accumulated router logits: scores = x @ w, flat [256, 8]. *)
let ref_score (rx rw:S.seq real) (t e:nat) : real =
  dot_prefix hidden experts rx rw t e hidden

let ref_scores (rx rw:S.seq real) : S.seq real =
  S.init (tokens * experts) (fun (k:nat{k < tokens * experts}) ->
    ref_score rx rw (k / experts) (k % experts))

(* softmax over one row; torch's max shift does not change the real value. *)
let exp_row (rs:S.seq real) : S.seq real =
  S.init (S.length rs) (fun (i:nat{i < S.length rs}) -> exp (S.index rs i))

let ref_softmax (rs:S.seq real) : S.seq real =
  S.init (S.length rs) (fun (i:nat{i < S.length rs}) ->
    exp (S.index rs i) /. rsum (exp_row rs))

(* Stable descending top-1 over reals: the first maximum. *)
let rec ref_top1_prefix (rs:S.seq real) (n:nat{1 <= n /\ n <= S.length rs})
  : GTot nat (decreases n)
  = if n = 1 then 0
    else let j = ref_top1_prefix rs (n - 1) in
         if t2b (S.index rs (n - 1) >. ridx rs j) then n - 1 else j

(* Selected-probability normalization for top-1: p / sum of the top-1 probs. *)
let ref_combine (p:real{p =!= 0.0R}) : real = p /. p

(* Capacity routing in input-token order, as closed forms over the selected
   experts sel[0..n) (sel[t] is the expert chosen for token t). *)
let rec count_upto (sel:S.seq int) (e:int) (n:nat{n <= S.length sel})
  : Tot nat (decreases n)
  = if n = 0 then 0
    else count_upto sel e (n - 1) + (if S.index sel (n - 1) = e then 1 else 0)

(* Zero-based position of token t in its expert's queue (cumsum - 1). *)
let rank (sel:S.seq int) (t:nat{t < S.length sel}) : nat =
  count_upto sel (S.index sel t) t

let kept (sel:S.seq int) (t:nat{t < S.length sel}) : bool =
  rank sel t < capacity

(* The token in slot s of expert e after n tokens, or -1. *)
let rec slot_token (sel:S.seq int) (e s:int) (n:nat{n <= S.length sel})
  : Tot (r:int{-1 <= r /\ r < n}) (decreases n)
  = if n = 0 then -1
    else if S.index sel (n - 1) = e && count_upto sel e (n - 1) = s && s < capacity
    then n - 1
    else slot_token sel e s (n - 1)

(* ------------------------------------------------------------------ *)
(* Native helpers                                                       *)

(* Integer views of the count dtype: int32 counts (optimized) or int64 prefix (base). *)
class count_repr (d:Type0) = { count_v : d -> int }
instance count_repr_i32 : count_repr I32.t = { count_v = I32.v }
instance count_repr_i64 : count_repr I64.t = { count_v = I64.v }

let ninf : F32.t = FB.neg (FB.infinity #F32.t)

let is_nan (x:F32.t) : prop = FB.NaN? (FB.kind x)

(* Base argmax: start from (0, row[0]); scan 1..n-1 with strict row[e] > best. *)
let rec base_scan (row:S.seq F32.t) (n:nat{1 <= n /\ n <= S.length row})
  : Tot (int & F32.t) (decreases n)
  = if n = 1 then (0, S.index row 0)
    else let p = base_scan row (n - 1) in
         if lt (snd p) (S.index row (n - 1)) then (n - 1, S.index row (n - 1)) else p

(* Optimized argmax: start from (0, -inf); scan 0..n-1 with strict row[e] > best. *)
let rec opt_scan (row:S.seq F32.t) (n:nat{n <= S.length row})
  : Tot (int & F32.t) (decreases n)
  = if n = 0 then (0, ninf)
    else let p = opt_scan row (n - 1) in
         if lt (snd p) (S.index row (n - 1)) then (n - 1, S.index row (n - 1)) else p

(* Allowed top-1 choice over the native row actually consumed: no entry is
   strictly greater (or, for the all-NaN/-inf row, expert 0 with nothing above
   -inf), and every earlier non-NaN entry is strictly below it (stated through
   a native witness v with row[i] <= v < row[j]: first-maximum tie rule). *)
let allowed_choice (row:S.seq F32.t) (j:int) : prop =
  S.length row == experts /\ 0 <= j /\ j < experts /\
  ((forall (i:nat). i < experts ==> ~(lt (S.index row j) (S.index row i))) \/
   (j == 0 /\ (forall (i:nat). i < experts ==> ~(lt ninf (S.index row i))))) /\
  (forall (i:nat). i < j ==>
     is_nan (S.index row i) \/
     (exists (v:F32.t). lte (S.index row i) v /\ lt v (S.index row j)))

(* Router projection: the scores approximate x @ w for all reals the inputs
   approximate. *)
let scores_ok (x w:S.seq BF.t) (scores:S.seq F32.t) : prop =
  forall (rx rw:S.seq real). x %~ rx /\ w %~ rw ==> scores %~ (ref_scores rx rw)

(* Router probabilities: row t of rows approximates the real softmax of row t
   of x @ w, for all reals the inputs approximate. *)
let softmax_row_ok (x w:S.seq BF.t) (rows:S.seq F32.t) (t:nat) : prop =
  forall (rx rw:S.seq real). x %~ rx /\ w %~ rw ==>
    row_of rows t %~ (ref_softmax (row_of (ref_scores rx rw) t))

let softmax_rows_ok (x w:S.seq BF.t) (rows:S.seq F32.t) : prop =
  forall (t:nat). t < tokens ==> softmax_row_ok x w rows t

(* Combine weight of a kept token: the reference p / p of every nonzero real
   the selected probability approximates (non-degenerate selection). *)
let selected_combine_ok (row:S.seq F32.t) (j:int) (c:F32.t) : prop =
  S.length row == experts /\ 0 <= j /\ j < experts /\
  (forall (rp:real).
     lt ninf (S.index row j) /\ S.index row j %~ rp /\ rp =!= 0.0R ==>
     c %~ (ref_combine rp))

(* Capacity state after the first n tokens of sel, with kept combine weights
   comb[t]. Unfilled slots hold token -1 and +0.0. *)
let routing_state (#d:Type0) {| count_repr d |}
  (sel:S.seq int) (comb:S.seq F32.t) (n:nat{n <= S.length sel /\ n <= S.length comb})
  (counts:S.seq d) (idx:S.seq I64.t) (probs:S.seq F32.t) : prop =
  S.length counts == experts /\ S.length idx == slots /\ S.length probs == slots /\
  (forall (e:nat). e < experts ==> count_v (S.index counts e) == count_upto sel e n) /\
  (forall (k:nat). k < slots ==>
     I64.v (S.index idx k) == slot_token sel (k / capacity) (k % capacity) n) /\
  (forall (k:nat). k < slots ==>
     S.index probs k ==
       (let t = slot_token sel (k / capacity) (k % capacity) n in
        if t >= 0 then S.index comb t else zero))

(* Survival rows of tokens 0..n-1 (flat [_, 8]). *)
let survival_rows (sel:S.seq int) (n:nat{n <= S.length sel}) (surv:S.seq I32.t) : prop =
  S.length surv >= experts * n /\
  (forall (t:nat) (e:nat). t < n /\ e < experts ==>
     I32.v (S.index surv (experts * t + e)) ==
       (if S.index sel t = e && kept sel t then 1 else 0))

let initial_routing_state (#d:Type0) {| count_repr d |}
  (counts:S.seq d) (idx:S.seq I64.t) (probs:S.seq F32.t) : prop =
  S.length counts == experts /\ S.length idx == slots /\ S.length probs == slots /\
  (forall (e:nat). e < experts ==> count_v (S.index counts e) == 0) /\
  (forall (k:nat). k < slots ==> I64.v (S.index idx k) == -1) /\
  (forall (k:nat). k < slots ==> S.index probs k == zero)

(* ------------------------------------------------------------------ *)
(* (2) Program postcondition (role-independent)                         *)

let program_postcondition
  (x w:S.seq BF.t) (demand:S.seq I32.t) (idx:S.seq I64.t) (probs:S.seq F32.t)
  (surv:S.seq I32.t) : prop =
  (S.length x == tokens * hidden /\ S.length w == hidden * experts) ==>
  (S.length demand == experts /\ S.length idx == slots /\
   S.length probs == slots /\ S.length surv == tokens * experts /\
   (exists (rows:S.seq F32.t) (sel:S.seq int) (comb:S.seq F32.t).
      S.length rows == tokens * experts /\
      S.length sel == tokens /\ S.length comb == tokens /\
      softmax_rows_ok x w rows /\
      (forall (t:nat). t < tokens ==> allowed_choice (row_of rows t) (S.index sel t)) /\
      (forall (t:nat). t < tokens /\ kept sel t ==>
         selected_combine_ok (row_of rows t) (S.index sel t) (S.index comb t)) /\
      routing_state sel comb tokens demand idx probs /\
      survival_rows sel tokens surv))

(* ------------------------------------------------------------------ *)
(* (3) Family relations: base                                           *)

let base_projection_softmax_relation
  (x w:S.seq BF.t) (softmax:S.seq F32.t) (prefix idx:S.seq I64.t)
  (probs:S.seq F32.t) (surv:S.seq I32.t) : prop =
  (S.length x == tokens * hidden /\ S.length w == hidden * experts) ==>
  (S.length softmax == tokens * experts /\
   softmax_rows_ok x w softmax /\
   initial_routing_state prefix idx probs /\
   S.length surv == 0)

let base_argmax_relation (row:S.seq F32.t) (sel:S.seq I32.t) (sp:S.seq F32.t) : prop =
  S.length row == experts ==>
  (S.length sel == 1 /\ S.length sp == 1 /\
   I32.v (S.index sel 0) == fst (base_scan row experts) /\
   S.index sp 0 == snd (base_scan row experts))

(* __fdiv_rn(p, p) of the selected probability. *)
let base_combine_ok (p c:F32.t) : prop =
  forall (rp:real). p %~ rp /\ rp =!= 0.0R ==> c %~ (ref_combine rp)

let base_capacity_routing_relation
  (sel:S.seq I32.t) (sp:S.seq F32.t) (prefix idx:S.seq I64.t) (probs:S.seq F32.t)
  (surv:S.seq I32.t) (t:int)
  (prefix' idx':S.seq I64.t) (probs':S.seq F32.t) (surv':S.seq I32.t) : prop =
  (S.length sel == 1 /\ S.length sp == 1 /\ S.length prefix == experts /\
   S.length idx == slots /\ S.length probs == slots /\
   0 <= t /\ t < tokens /\ S.length surv == experts * t /\
   0 <= I32.v (S.index sel 0) /\ I32.v (S.index sel 0) < experts /\
   (forall (e:nat). e < experts ==>
      0 <= I64.v (S.index prefix e) /\ I64.v (S.index prefix e) <= t)) ==>
  (let s = I32.v (S.index sel 0) in
   let r = I64.v (S.index prefix s) in
   let slot = capacity * s + r in
   S.length prefix' == experts /\ S.length idx' == slots /\ S.length probs' == slots /\
   S.length surv' == experts * t + experts /\
   (forall (e:nat). e < experts ==>
      I64.v (S.index prefix' e) == I64.v (S.index prefix e) + (if e = s then 1 else 0)) /\
   (forall (k:nat). k < slots ==>
      (if r < capacity && k = slot
       then I64.v (S.index idx' k) == t /\ base_combine_ok (S.index sp 0) (S.index probs' k)
       else S.index idx' k == S.index idx k /\ S.index probs' k == S.index probs k)) /\
   (forall (k:nat). k < experts * t ==> S.index surv' k == S.index surv k) /\
   (forall (e:nat). e < experts ==>
      I32.v (S.index surv' (experts * t + e)) == (if e = s && r < capacity then 1 else 0)))

(* Final int32_t(prefix[e]) demand conversion; exact for counts in [0, 256]. *)
let base_expert_demand_relation (prefix:S.seq I64.t) (demand:S.seq I32.t) : prop =
  (S.length prefix == experts /\
   (forall (e:nat). e < experts ==>
      0 <= I64.v (S.index prefix e) /\ I64.v (S.index prefix e) <= tokens)) ==>
  (S.length demand == experts /\
   (forall (e:nat). e < experts ==> I32.v (S.index demand e) == I64.v (S.index prefix e)))

(* ------------------------------------------------------------------ *)
(* (3) Family relations: optimized                                      *)

let optimized_projection_relation
  (x w:S.seq BF.t) (scores:S.seq F32.t) (counts:S.seq I32.t) (idx:S.seq I64.t)
  (probs:S.seq F32.t) (surv:S.seq I32.t) : prop =
  (S.length x == tokens * hidden /\ S.length w == hidden * experts) ==>
  (S.length scores == tokens * experts /\ scores_ok x w scores /\
   initial_routing_state counts idx probs /\
   S.length surv == tokens * experts /\
   (forall (k:nat). k < tokens * experts ==> I32.v (S.index surv k) == 0))

let optimized_softmax_relation (scores:S.seq F32.t) (t:int) (row:S.seq F32.t) : prop =
  (S.length scores == tokens * experts /\ 0 <= t /\ t < tokens) ==>
  (S.length row == experts /\
   (forall (rs:S.seq real). scores %~ rs ==> row %~ (ref_softmax (row_of rs t))))

let optimized_argmax_relation (row:S.seq F32.t) (sel:S.seq I32.t) (sp:S.seq F32.t) : prop =
  S.length row == experts ==>
  (S.length sel == 1 /\ S.length sp == 1 /\
   I32.v (S.index sel 0) == fst (opt_scan row experts) /\
   S.index sp 0 == snd (opt_scan row experts))

(* fdividef(row[selected], __fadd_rn(0.0f, selected_probability)). *)
let optimized_combine_ok (q p c:F32.t) : prop =
  forall (rq rp:real). q %~ rq /\ p %~ rp /\ (0.0R +. rp) =!= 0.0R ==>
    c %~ (rq /. (0.0R +. rp))

let optimized_routing_relation
  (row:S.seq F32.t) (sel:S.seq I32.t) (sp:S.seq F32.t) (counts:S.seq I32.t)
  (idx:S.seq I64.t) (probs:S.seq F32.t) (surv:S.seq I32.t) (t:int)
  (counts':S.seq I32.t) (idx':S.seq I64.t) (probs':S.seq F32.t) (surv':S.seq I32.t) : prop =
  (S.length row == experts /\ S.length sel == 1 /\ S.length sp == 1 /\
   S.length counts == experts /\ S.length idx == slots /\ S.length probs == slots /\
   S.length surv == tokens * experts /\ 0 <= t /\ t < tokens /\
   0 <= I32.v (S.index sel 0) /\ I32.v (S.index sel 0) < experts /\
   (forall (e:nat). e < experts ==>
      0 <= I32.v (S.index counts e) /\ I32.v (S.index counts e) <= t)) ==>
  (let s = I32.v (S.index sel 0) in
   let r = I32.v (S.index counts s) in
   let slot = capacity * s + r in
   S.length counts' == experts /\ S.length idx' == slots /\ S.length probs' == slots /\
   S.length surv' == tokens * experts /\
   (forall (e:nat). e < experts ==>
      I32.v (S.index counts' e) == I32.v (S.index counts e) + (if e = s then 1 else 0)) /\
   (forall (k:nat). k < slots ==>
      (if r < capacity && k = slot
       then I64.v (S.index idx' k) == t /\
            optimized_combine_ok (S.index row s) (S.index sp 0) (S.index probs' k)
       else S.index idx' k == S.index idx k /\ S.index probs' k == S.index probs k)) /\
   (forall (k:nat). k < tokens * experts ==>
      (if r < capacity && k = experts * t + s
       then I32.v (S.index surv' k) == 1
       else S.index surv' k == S.index surv k)))

(* ------------------------------------------------------------------ *)
(* (4) Helper lemma: projection then softmax gives the real-softmax row  *)

let optimized_softmax_chain
  (x w:S.seq BF.t) (scores rows row:S.seq F32.t) (t:nat{t < tokens})
  : Lemma (requires S.length scores == tokens * experts /\ scores_ok x w scores /\
                    optimized_softmax_relation scores t row /\ row_of rows t == row)
          (ensures softmax_row_ok x w rows t)
  = ()
