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

#push-options "--z3rlimit 10 --fuel 4 --ifuel 2"

let base_batch_refines #tokens #experts rows selected =
  let point (t:Shared.index tokens)
    : Lemma (
        I32.v (Seq.index (selected t) 0) ==
          Shared.top1_index #experts (fun e -> Seq.index (rows t) e) /\
        Kuiper.Approximates.Base.to_real #f32
          (Seq.index (rows t) (I32.v (Seq.index (selected t) 0))) ==
        Kuiper.Approximates.Base.to_real #f32
          (Seq.index (rows t) (Shared.top1_index #experts (fun e -> Seq.index (rows t) e)))) =
    assert (post #1 #experts (rows t) (selected t));
    B.prefix_extensional #experts #experts
      (input #1 #experts (rows t) 0) (fun e -> Seq.index (rows t) e) experts;
    assert (Shared.post #1 #experts
      (input #1 #experts (rows t)) (input #1 #experts (rows t))
      (output #1 #experts (rows t) (selected t)));
    assert ((output #1 #experts (rows t) (selected t)).selected_expert 0 ==
      Shared.top1_index #experts (input #1 #experts (rows t) 0));
    norm_spec [zeta; iota; primops; delta_only [
      "KPlay.Review.post"; "KPlay.Review.input"; "KPlay.Review.output";
      "KPlay.SharedSpec.post"]] (post #1 #experts (rows t) (selected t))
  in
  FStar.Classical.forall_intro point;
  assert (Shared.post #tokens #experts
    (fun t e -> Seq.index (rows t) e) (fun t e -> Seq.index (rows t) e)
    {Shared.selected_expert = (fun t -> I32.v (Seq.index (selected t) 0));
     Shared.selected_probability = (fun t -> Kuiper.Approximates.Base.to_real #f32
       (Seq.index (rows t) (I32.v (Seq.index (selected t) 0))))}) by (
    FStar.Tactics.V2.norm [zeta; iota; delta_only ["KPlay.SharedSpec.post"]];
    FStar.Tactics.V2.smt ())

inline_for_extraction noextract
let base_post (#experts:pos) (#slots #history:nat)
  (row:probabilities experts) (state:B.routing_state experts slots history)
  (selected:B.selected_values experts)
  : Lemma
      (requires B.semantic_post row state selected row state)
      (ensures post #1 #experts row selected) =
  B.prefix_extensional #experts #experts
    (input #1 #experts row 0) (B.logical_input row 0) experts;
  assert (Shared.post #1 #experts (B.logical_input row) (B.logical_input row)
    (B.logical_output selected row));
  assert (I32.v (Seq.index selected 0) ==
    Shared.top1_index #experts (input #1 #experts row 0));
  ()

inline_for_extraction noextract
let optimized_post (#tokens:nat) (#experts:pos)
  (rows:probabilities (experts * tokens))
  (selected:selections tokens experts) (best:probabilities tokens)
  : Lemma
      (requires O.semantic_post rows rows selected best)
      (ensures post rows selected) =
  assert (input rows == O.logical_input rows) by (
    FStar.Tactics.V2.norm [zeta; iota; delta_only [
      "KPlay.Review.input"; "KPlay.Optimized.Functional.logical_input"]];
    FStar.Tactics.V2.trefl ());
  assert (forall (t:Shared.index tokens).
    (output rows selected).selected_probability t ==
    (O.logical_output selected best).selected_probability t);
  ()

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
{
  unfold before Base #(U64.v tokens) #(U64.v experts) aux;
  unfold frame_inputs #(U64.v experts) aux;
  B.kplay_run experts aux.slots aux.history aux.token
    #(FStar.Ghost.hide (FStar.Ghost.reveal values))
    #aux.frame rows aux.input_counts aux.input_indices aux.input_probabilities aux.input_survival
    selected aux.copied_row aux.copied_counts aux.copied_indices
    aux.copied_probabilities aux.copied_survival;
  with chosen. assert on gpu_loc (A.pts_to selected chosen);
  let _ = base_post values aux.frame chosen;
  assert pure (post values chosen);
  fold frame_inputs #(U64.v experts) aux;
  fold after Base #(U64.v tokens) #(U64.v experts) aux values chosen;
}

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
{
  unfold before Optimized #(U64.v tokens) #(U64.v experts) aux;
  O.kplay_run tokens experts
    #(FStar.Ghost.hide (FStar.Ghost.reveal values)) rows selected aux;
  with chosen best.
    assert on gpu_loc (A.pts_to selected chosen ** A.pts_to aux best);
  let _ = optimized_post values chosen best;
  assert pure (post values chosen);
  fold after Optimized #(U64.v tokens) #(U64.v experts) aux values chosen;
}

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

#pop-options
