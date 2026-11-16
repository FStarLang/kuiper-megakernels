module KPlay.Review
#lang-pulse

open Kuiper
module Seq = FStar.Seq
module I32 = FStar.Int32
module I64 = FStar.Int64
module U64 = FStar.UInt64
module A = Pulse.Lib.Array
module Shared = KPlay.SharedSpec
module B = KPlay.Base.Functional
module O = KPlay.Optimized.Functional

(* Base is one row per call; optimized is a batch. Both select the same
   stable maximum on the same FP32 probabilities. Base copies routing frame
   buffers; optimized instead materializes gathered best probabilities.
   The batch-lifting lemma below exposes how repeated base calls compose. *)
noextract type implementation = | Base | Optimized
noextract type batch_size (i:implementation) =
  n:u64{0 < U64.v n /\ U64.v n <= 2147483647 /\ (i == Base ==> U64.v n == 1)}
noextract type expert_count (tokens:u64) =
  e:u64{0 < U64.v e /\ U64.v e * U64.v tokens <= 2147483647}
inline_for_extraction noextract unfold
let probabilities (n:nat) = O.probability_values n
inline_for_extraction noextract unfold
let selections (tokens:nat) (experts:pos) = O.selected_values tokens experts
inline_for_extraction noextract unfold
let device_array (a:Type0) (n:nat) = p:larray a n{is_global_array p}

inline_for_extraction noextract
let input (#tokens:nat) (#experts:pos)
  (rows:probabilities (experts * tokens)) : Shared.probabilities tokens experts =
  fun t e -> Seq.index rows (experts * t + e)
inline_for_extraction noextract
let output (#tokens:nat) (#experts:pos)
  (rows:probabilities (experts * tokens)) (selected:selections tokens experts)
  : Shared.outputs tokens experts = {
  selected_expert = (fun t -> I32.v (Seq.index selected t));
  selected_probability = (fun t -> Kuiper.Approximates.Base.to_real #f32
    (Seq.index rows (experts * t + I32.v (Seq.index selected t))))
}
inline_for_extraction noextract
let post (#tokens:nat) (#experts:pos)
  (rows:probabilities (experts * tokens)) (selected:selections tokens experts) : prop =
  Shared.post #tokens #experts
    (input #tokens #experts rows) (input #tokens #experts rows)
    (output #tokens #experts rows selected)

noextract noeq type base_buffers (experts:pos) = {
  slots: (n:u64{U64.v n < 4294967296 /\ U64.v n % experts == 0});
  history: (n:u64{U64.v n < 4294967296});
  token: (t:i64{0 <= I64.v t /\ U64.v history == (I64.v t + 1) * experts});
  frame: erased (B.routing_state experts (U64.v slots) (U64.v history));
  input_counts: device_array i64 experts;
  input_indices: device_array i64 (U64.v slots);
  input_probabilities: device_array f32 (U64.v slots);
  input_survival: device_array i32 (U64.v history);
  copied_row: device_array f32 experts;
  copied_counts: device_array i64 experts;
  copied_indices: device_array i64 (U64.v slots);
  copied_probabilities: device_array f32 (U64.v slots);
  copied_survival: device_array i32 (U64.v history)
}
inline_for_extraction noextract unfold
let auxiliary (i:implementation) (tokens:nat) (experts:pos) : Type0 =
  match i with
  | Base -> base_buffers experts
  | Optimized -> device_array f32 tokens

inline_for_extraction noextract
let frame_inputs (#experts:pos) (b:base_buffers experts) : slprop =
  on gpu_loc (
    A.pts_to b.input_counts b.frame.B.counts ** A.pts_to b.input_indices b.frame.B.indices **
    A.pts_to b.input_probabilities b.frame.B.probabilities ** A.pts_to b.input_survival b.frame.B.survival)
inline_for_extraction noextract
let before (i:implementation) (#tokens:nat) (#experts:pos)
  : (auxiliary i tokens experts -> slprop) =
  match i with
  | Base -> fun (aux:base_buffers experts) ->
    frame_inputs #experts aux **
    (exists* row counts indices probabilities survival.
      on gpu_loc (
        A.pts_to aux.copied_row row ** A.pts_to aux.copied_counts counts **
        A.pts_to aux.copied_indices indices ** A.pts_to aux.copied_probabilities probabilities **
        A.pts_to aux.copied_survival survival))
  | Optimized -> fun (aux:device_array f32 tokens) ->
    exists* best. on gpu_loc (A.pts_to aux best)
inline_for_extraction noextract
let after (i:implementation) (#tokens:nat) (#experts:pos)
  : (auxiliary i tokens experts -> probabilities (experts * tokens) ->
     selections tokens experts -> slprop) =
  match i with
  | Base -> fun (aux:base_buffers experts) rows selected ->
    frame_inputs #experts aux ** on gpu_loc (
      A.pts_to aux.copied_row rows ** A.pts_to aux.copied_counts aux.frame.B.counts **
      A.pts_to aux.copied_indices aux.frame.B.indices **
      A.pts_to aux.copied_probabilities aux.frame.B.probabilities **
      A.pts_to aux.copied_survival aux.frame.B.survival)
  | Optimized -> fun (aux:device_array f32 tokens) rows selected ->
    exists* (best:probabilities tokens).
      on gpu_loc (A.pts_to aux best) **
      pure (forall (t:Shared.index tokens).
        Seq.index best t == Seq.index rows (experts * t + I32.v (Seq.index selected t)))

val base_batch_refines :
  #tokens:nat -> #experts:pos ->
  rows:(Shared.index tokens -> probabilities experts) ->
  selected:(Shared.index tokens -> selections 1 experts) ->
  Lemma
    (requires forall (t:Shared.index tokens). post #1 #experts (rows t) (selected t))
    (ensures Shared.post #tokens #experts
      (fun t e -> Seq.index (rows t) e)
      (fun t e -> Seq.index (rows t) e)
      {Shared.selected_expert = (fun t -> I32.v (Seq.index (selected t) 0));
       Shared.selected_probability = (fun t -> Kuiper.Approximates.Base.to_real #f32
         (Seq.index (rows t) (I32.v (Seq.index (selected t) 0))))})

fn base
  (tokens:batch_size Base) (experts:expert_count tokens)
  (rows:device_array f32 (U64.v experts * U64.v tokens))
  (selected:device_array i32 (U64.v tokens))
  (aux:auxiliary Base (U64.v tokens) (U64.v experts))
  (#values:erased (probabilities (U64.v experts * U64.v tokens)))
  preserves cpu ** on gpu_loc (A.pts_to rows values)
  requires before Base #(U64.v tokens) #(U64.v experts) aux **
    (exists* old. on gpu_loc (A.pts_to selected old))
  ensures exists* (chosen:selections (U64.v tokens) (U64.v experts)).
    on gpu_loc (A.pts_to selected chosen) **
    after Base #(U64.v tokens) #(U64.v experts) aux values chosen **
    pure (post #(U64.v tokens) #(U64.v experts) values chosen)

fn optimized
  (tokens:batch_size Optimized) (experts:expert_count tokens)
  (rows:device_array f32 (U64.v experts * U64.v tokens))
  (selected:device_array i32 (U64.v tokens))
  (aux:auxiliary Optimized (U64.v tokens) (U64.v experts))
  (#values:erased (probabilities (U64.v experts * U64.v tokens)))
  preserves cpu ** on gpu_loc (A.pts_to rows values)
  requires before Optimized #(U64.v tokens) #(U64.v experts) aux **
    (exists* old. on gpu_loc (A.pts_to selected old))
  ensures exists* (chosen:selections (U64.v tokens) (U64.v experts)).
    on gpu_loc (A.pts_to selected chosen) **
    after Optimized #(U64.v tokens) #(U64.v experts) aux values chosen **
    pure (post #(U64.v tokens) #(U64.v experts) values chosen)

(*
Signature relation: both roles preserve the same supplied FP32 probability
cells and establish post, the shared stable-maximum/gather relation.
batch_size Base forces one row; Optimized permits a positive batch within
the indexed size bounds. auxiliary Base supplies input/output routing-frame
buffers and a copied probability row. auxiliary Optimized is instead an
actual gathered-best-probability buffer. after specifies each copied frame
cell or gathered cell, not merely a logical selection.

base_batch_refines proves that collected one-row guarantees imply the
batched Shared.post for the same rows. It does not run a loop and is not a
theorem that one base invocation handles the optimized batch. Selection
equality concerns identical supplied native probabilities, not probabilities
computed independently by two projection kernels. The finite [0,1]
probability domain and stable tie rule are part of this comparison;
ownership is returned without a pending epoch in these family interfaces.

Assessment: moderate integration significance, but no stable-selection task
change after row-wise composition. This is an exact functional selector on
fixed native rows: Tightness.witness/unique determine the selected IDs and
gathered denotations, and after fixes actual gathered cells. The dispatcher
does not batch, allocate, gather, or copy by itself: each branch below calls
exactly one already-adapted wrapper.
*)
fn switcher
  (i:implementation) (tokens:batch_size i) (experts:expert_count tokens)
  (rows:device_array f32 (U64.v experts * U64.v tokens))
  (selected:device_array i32 (U64.v tokens))
  (aux:auxiliary i (U64.v tokens) (U64.v experts))
  (#values:erased (probabilities (U64.v experts * U64.v tokens)))
  preserves cpu ** on gpu_loc (A.pts_to rows values)
  requires before i #(U64.v tokens) #(U64.v experts) aux **
    (exists* old. on gpu_loc (A.pts_to selected old))
  ensures exists* (chosen:selections (U64.v tokens) (U64.v experts)).
    on gpu_loc (A.pts_to selected chosen) **
    after i #(U64.v tokens) #(U64.v experts) aux values chosen **
    pure (post #(U64.v tokens) #(U64.v experts) values chosen)
{
  match i {
    Base -> { base tokens experts rows selected aux #values }
    Optimized -> { optimized tokens experts rows selected aux #values }
  }
}

