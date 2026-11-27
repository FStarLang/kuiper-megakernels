module KPlay.Tightness.Flashmoe.Question
(* What the FlashMoE spec (KPlay.Spec) should pin down. Review this file: every
   output must be listed, each target must be the reference math, and
   `allowed` is the only freedom. *)
open FStar.Real
module K = KPlay.Spec
module A = KPlay.Tightness.Answer
module S = FStar.Seq
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32
module I32 = FStar.Int32
module I64 = FStar.Int64

(* Inputs: the router's two BF16 operands. No input premise. *)
let input = K.inputs
let domain (x:input) : prop = True
let example : input = {
  K.tokens = (fun _ _ -> Kuiper.Floating.zero #BF.t);
  K.router_weights = (fun _ _ -> Kuiper.Floating.zero #BF.t) }

(* The complete answer. Exact part: expert_demand [8], token_indices [8 x 32]
   and survival [256 x 8], row by row. Numbers: probabilities [8 x 32]. *)
noeq type exact_outputs = {
  demand : s:S.seq I32.t{S.length s == K.expert_count};
  indices : s:S.seq I64.t{S.length s == K.expert_count * K.capacity};
  survival : s:S.seq I32.t{S.length s == K.token_count * K.expert_count} }
let exact = exact_outputs
let number = F32.t
let cells : nat = K.expert_count * K.capacity
let count (_:exact) : nat = cells
let view (y:A.answer exact number count) : GTot K.outputs = {
  K.expert_demand = (fun e -> S.index y.A.exact.demand e);
  K.token_indices = (fun e c -> S.index y.A.exact.indices (e * K.capacity + c));
  K.probabilities = (fun e c -> S.index y.A.numbers (e * K.capacity + c));
  K.survival = (fun t e -> S.index y.A.exact.survival (t * K.expert_count + e)) }
let spec (x:input) (y:A.answer exact number count) : prop = K.postcondition x (view y)

(* The real inputs the BF16 inputs stand for. *)
let model = K.router_inputs real
let models (x:input) (m:model) : prop = K.inputs_approximates x m

(* The only freedom: each token's expert may be any first maximum of an FP32
   probability row that approximates the reference softmax (ties, rounding). *)
let choice (x:input) = K.vector (K.index K.expert_count) K.token_count
let allowed (x:input{domain x}) (c:choice x) : prop =
  exists (rows:K.matrix F32.t K.token_count K.expert_count).
    (forall (m:model). models x m ==>
       K.matrix_approximates rows (K.router_probabilities m)) /\
    (forall (t:K.index K.token_count). K.top1_relation (rows t) (c t))

let rec assigned_le (c:K.vector (K.index K.expert_count) K.token_count)
  (e:K.index K.expert_count) (n:nat{n <= K.token_count})
  : Lemma (ensures K.assigned_before c e n <= n) (decreases n)
  = if n > 0 then assigned_le c e (n - 1)

(* Targets: the reference's token-order capacity routing of the chosen
   experts; a filled slot holds p / p = 1 (p is a softmax value, so p > 0),
   an empty slot holds 0. *)
let exact_target (x:input{domain x}) (c:choice x) : GTot exact = {
  demand = S.init_ghost K.expert_count (fun e ->
    assigned_le c e K.token_count; I32.int_to_t (K.demand c e));
  indices = S.init_ghost (K.expert_count * K.capacity) (fun i ->
    I64.int_to_t (K.kept_token c (i / K.capacity) (i % K.capacity) K.token_count));
  survival = S.init_ghost (K.token_count * K.expert_count) (fun i ->
    I32.int_to_t (K.survival_of c (i / K.expert_count) (i % K.expert_count))) }
let number_target (x:input{domain x}) (m:model) (c:choice x) : GTot (s:S.seq real{S.length s == count (exact_target x c)}) =
  S.init_ghost cells (fun i ->
    if K.kept_token c (i / K.capacity) (i % K.capacity) K.token_count >= 0 then 1.0R else 0.0R)
