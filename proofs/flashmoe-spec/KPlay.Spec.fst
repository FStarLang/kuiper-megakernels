module KPlay.Spec

(* Typed specification of the FlashMoE capacity-pruned top-1 router.

   Reference: reference/flashmoe/base.py, `reference_route` (lines 13-39),
   called by `Model.forward` (lines 46-49) with the smoke profile of
   reference/flashmoe/input-domain.txt: 256 tokens, hidden size 512,
   8 experts, per-expert capacity 32, top_k 1. Router weights use the
   logical [hidden, experts] layout (reference/flashmoe/LAYOUT.md).

   Read top to bottom:
     S1  model: dimensions, index/vector/matrix, input/output records and
         the reference intermediates a run produces;
     S2  one real-valued (or integer) equation per reference operation;
     S3  liftings of %~ (approximates) to the model types;
     S4  operation relations, independent of any implementation;
     S5  the program postcondition: structural /\ numerical /\ selection.

   What the postcondition says, in words. There are router probability rows,
   a selected expert per token and a combine weight per token such that
   - numerical: every row approximates the real softmax of tokens @ weights
     for ALL reals the BF16 inputs approximate, and every token that keeps
     its slot has a combine weight approximating p / p for every nonzero
     real p its selected probability approximates;
   - selection: each token's expert is an allowed first maximum of the
     probability row actually used (the native row, not the real one);
   - structural: demand, token indices, kept probabilities and survival are
     EXACTLY the reference's token-order capacity routing of those choices.
   It is a relation, not a function: two correct implementations may select
   different experts when their rows differ within the approximation.
   No input-domain premise is needed (the guarantee holds for every input of
   the right shape, so in particular on the domain of input-domain.txt). *)

open Kuiper
open Kuiper.Approximates
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32
module I32 = FStar.Int32
module I64 = FStar.Int64
module FB = Kuiper.Floating.Base

#set-options "--fuel 1 --ifuel 1 --z3rlimit 20"

(* ------------------------------------------------------------------ *)
(* S1. Model                                                            *)

(* Smoke profile (input-domain.txt); top_k = 1 is built into the
   selection below. *)
let token_count : pos = 256
let hidden_size : pos = 512
let expert_count : pos = 8
let capacity : pos = 32

type index (n:nat) = i:nat{i < n}
type vector (a:Type0) (n:nat) = index n -> GTot a
type matrix (a:Type0) (rows cols:nat) = index rows -> index cols -> GTot a

(* The two operands of reference_route (base.py:13), over any element type:
   BF16 for the caller's tensors, real for the reference math. *)
noeq type router_inputs (a:Type0) = {
  tokens : matrix a token_count hidden_size;          (* [tokens, hidden] *)
  router_weights : matrix a hidden_size expert_count; (* [hidden, experts] *)
}

type inputs = router_inputs BF.t

(* The returned dictionary (base.py:34-39). *)
noeq type outputs = {
  expert_demand : vector I32.t expert_count;          (* int32 [experts] *)
  token_indices : matrix I64.t expert_count capacity; (* int64 [experts, capacity] *)
  probabilities : matrix F32.t expert_count capacity; (* float32 [experts, capacity] *)
  survival : matrix I32.t token_count expert_count;   (* int32 [tokens, experts] *)
}

(* The reference intermediates that one run actually produces (boundary
   values only): the softmax rows `probabilities` (base.py:18, here
   `probability_rows`), the top-1 `selected` expert (base.py:19) and the
   `combine_probability` (base.py:21). *)
noeq type routing = {
  probability_rows : matrix F32.t token_count expert_count;
  selected : vector (index expert_count) token_count;
  combine_probability : vector F32.t token_count;
}

(* ------------------------------------------------------------------ *)
(* S2. Reference equations                                              *)

(* sum v k = v 0 + ... + v (k - 1), accumulated in ascending order. *)
let rec sum (#n:nat) (v:vector real n) (k:nat{k <= n}) : GTot real (decreases k) =
  if k = 0 then 0.0R else sum v (k - 1) +. v (k - 1)

(* Products x[t, h] * w[h, e] along the hidden dimension. *)
let products (x:matrix real token_count hidden_size) (w:matrix real hidden_size expert_count)
  (t:index token_count) (e:index expert_count) : vector real hidden_size
  = fun h -> x t h *. w h e

(* base.py:15-17: scores = torch.mm(x, weights), with both operands cast to
   FP32; over reals the cast is the identity. *)
let scores (x:matrix real token_count hidden_size) (w:matrix real hidden_size expert_count)
  : matrix real token_count expert_count
  = fun t e -> sum (products x w t e) hidden_size

let exps (#n:nat) (v:vector real n) : vector real n = fun i -> exp (v i)

let rec sum_exps_positive (#n:nat) (v:vector real n) (k:nat{k <= n})
  : Lemma (ensures sum (exps v) k >=. 0.0R /\ (k > 0 ==> sum (exps v) k >. 0.0R))
          (decreases k)
  = if k > 0 then sum_exps_positive v (k - 1)

(* base.py:18: softmax over one row. torch subtracts the row maximum first;
   over reals that shift does not change the value. *)
let softmax (#n:pos) (v:vector real n) : vector real n =
  fun i -> sum_exps_positive v n; exp (v i) /. sum (exps v) n

let router_probabilities (r:router_inputs real) : matrix real token_count expert_count =
  fun t -> softmax #expert_count (scores r.tokens r.router_weights t)

(* base.py:19 selects, per row, the first index of a stable descending
   argsort, i.e. the first maximum. Implementations select over their own
   FP32 rows, so selection is stated in S4 as an allowed choice over the
   native row (top1_relation). *)

(* base.py:20-21: selected_probability / selected_probability.sum(dim=1);
   with top_k = 1 the sum is the selected probability itself. *)
let combine_weight (p:real{p =!= 0.0R}) : real = p /. p

(* base.py:24: assignments[t, e] = (selected[t] == e). *)
let assignment (selected:vector (index expert_count) token_count)
  (t:index token_count) (e:index expert_count) : GTot nat
  = if selected t = e then 1 else 0

(* Number of tokens among 0 .. n-1 assigned to expert e. *)
let rec assigned_before (selected:vector (index expert_count) token_count)
  (e:index expert_count) (n:nat{n <= token_count}) : GTot nat (decreases n)
  = if n = 0 then 0 else assigned_before selected e (n - 1) + assignment selected (n - 1) e

(* base.py:25: prefix_rank = assignments.cumsum(dim=0) (inclusive). *)
let prefix_rank (selected:vector (index expert_count) token_count)
  (t:index token_count) (e:index expert_count) : GTot nat
  = assigned_before selected e (t + 1)

(* base.py:26: survival = assignments * (prefix_rank <= capacity). *)
let survival_of (selected:vector (index expert_count) token_count)
  (t:index token_count) (e:index expert_count) : GTot nat
  = assignment selected t e * (if prefix_rank selected t e <= capacity then 1 else 0)

(* base.py:27: demand = assignments.sum(dim=0), dropped tokens included. *)
let demand (selected:vector (index expert_count) token_count) (e:index expert_count) : GTot nat =
  assigned_before selected e token_count

(* Number of tokens among 0 .. n-1 that survive for expert e. *)
let rec survivors_before (selected:vector (index expert_count) token_count)
  (e:index expert_count) (n:nat{n <= token_count}) : GTot nat (decreases n)
  = if n = 0 then 0 else survivors_before selected e (n - 1) + survival_of selected (n - 1) e

(* base.py:28,31-32: token_indices[e, c] is the c-th token (in token order)
   with survival[t, e] = 1, i.e. the surviving token with exactly c
   surviving predecessors; -1 if there are at most c survivors. Searched
   among tokens 0 .. n-1. *)
let rec kept_token (selected:vector (index expert_count) token_count)
  (e:index expert_count) (c:index capacity) (n:nat{n <= token_count})
  : GTot (r:int{-1 <= r /\ r < n}) (decreases n)
  = if n = 0 then -1
    else if survival_of selected (n - 1) e = 1 && survivors_before selected e (n - 1) = c
    then n - 1
    else kept_token selected e c (n - 1)

(* base.py:29,33: kept_probabilities[e, c] is the combine probability of
   that token, and +0.0 for an unfilled slot. *)
let kept_probability (selected:vector (index expert_count) token_count)
  (combine:vector F32.t token_count) (e:index expert_count) (c:index capacity) : GTot F32.t
  = let t = kept_token selected e c token_count in
    if t >= 0 then combine t else zero

(* ------------------------------------------------------------------ *)
(* S3. Liftings of %~                                                   *)

let matrix_approximates (#a:Type0) {| scalar a, real_like a |} (#rows #cols:nat)
  (m:matrix a rows cols) (r:matrix real rows cols) : prop =
  forall (i:index rows) (j:index cols). m i j %~ r i j

let inputs_approximates (x:inputs) (r:router_inputs real) : prop =
  matrix_approximates x.tokens r.tokens /\
  matrix_approximates x.router_weights r.router_weights

(* ------------------------------------------------------------------ *)
(* S4. Operation relations                                              *)

(* base.py:15-18, projection then softmax: the rows approximate the real
   router probabilities of every real input the BF16 inputs approximate. *)
let router_probabilities_relation (input:inputs) (rows:matrix F32.t token_count expert_count)
  : prop
  = forall (r:router_inputs real).
      inputs_approximates input r ==> matrix_approximates rows (router_probabilities r)

let neg_infinity : F32.t = FB.neg (FB.infinity #F32.t)

let is_nan (v:F32.t) : prop = FB.NaN? (FB.kind v)

(* base.py:19, top-1 as an allowed choice over the FP32 row actually used:
   no entry is strictly greater than the chosen one (or, for a row with
   nothing above -inf, the choice is expert 0), and every earlier non-NaN
   entry is strictly below it, witnessed by a native value v with
   row[i] <= v < row[j] (first-maximum tie rule). *)
let top1_relation (row:vector F32.t expert_count) (j:index expert_count) : prop =
  ((forall (i:index expert_count). ~(lt (row j) (row i))) \/
   (j == 0 /\ (forall (i:index expert_count). ~(lt neg_infinity (row i))))) /\
  (forall (i:index expert_count). i < j ==>
     is_nan (row i) \/ (exists (v:F32.t). lte (row i) v /\ lt v (row j)))

(* base.py:20-21: the combine weight approximates p / p for every nonzero
   real p the selected probability approximates. *)
let combine_relation (selected_probability combine:F32.t) : prop =
  forall (p:real).
    selected_probability %~ p /\ p =!= 0.0R ==>
    combine %~ combine_weight p

(* base.py:23-33: the four outputs are exactly the token-order capacity
   routing of the selections, with kept probabilities copied from the
   combine probabilities. *)
let capacity_routing_relation (selected:vector (index expert_count) token_count)
  (combine:vector F32.t token_count) (output:outputs) : prop =
  (forall (e:index expert_count). I32.v (output.expert_demand e) == demand selected e) /\
  (forall (e:index expert_count) (c:index capacity).
     I64.v (output.token_indices e c) == kept_token selected e c token_count) /\
  (forall (e:index expert_count) (c:index capacity).
     output.probabilities e c == kept_probability selected combine e c) /\
  (forall (t:index token_count) (e:index expert_count).
     I32.v (output.survival t e) == survival_of selected t e)

(* ------------------------------------------------------------------ *)
(* S5. Program postcondition                                            *)

let structural (run:routing) (output:outputs) : prop =
  capacity_routing_relation run.selected run.combine_probability output

(* The combine weight matters only for tokens that keep a slot: no other
   token's weight reaches an output. *)
let numerical (input:inputs) (run:routing) : prop =
  router_probabilities_relation input run.probability_rows /\
  (forall (t:index token_count).
     survival_of run.selected t (run.selected t) = 1 ==>
     combine_relation (run.probability_rows t (run.selected t)) (run.combine_probability t))

let selection (run:routing) : prop =
  forall (t:index token_count). top1_relation (run.probability_rows t) (run.selected t)

let postcondition (input:inputs) (output:outputs) : prop =
  exists (run:routing). structural run output /\ numerical input run /\ selection run
