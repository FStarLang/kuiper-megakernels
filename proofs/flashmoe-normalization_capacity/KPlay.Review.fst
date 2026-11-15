module KPlay.Review
#lang-pulse

open Kuiper
module Seq = FStar.Seq
module I32 = FStar.Int32
module I64 = FStar.Int64
module A = Pulse.Lib.Array
module Shared = KPlay.SharedSpec
module B = KPlay.Base.Functional
module O = KPlay.Optimized.Functional

#push-options "--z3rlimit 10 --fuel 4 --ifuel 2"

inline_for_extraction noextract
let to_base (#p:Shared.model_profile) (#n:Shared.prefix_length p)
  (v:physical_state Base p n) : B.physical_state p n = {
  B.counts = v.counts; B.indices = v.indices; B.weights = v.weights; B.flags = v.flags
}
inline_for_extraction noextract
let from_base (#p:Shared.model_profile) (#n:Shared.prefix_length p)
  (v:B.physical_state p n) : physical_state Base p n = {
  counts = v.B.counts; indices = v.B.indices; weights = v.B.weights; flags = v.B.flags
}
inline_for_extraction noextract
let base_models_agree (p:Shared.model_profile) (n:Shared.prefix_length p)
  (v:B.physical_state p n) (s:Shared.routing_state p)
  : Lemma (
      represented Base p n (from_base v) s == B.state_representation p n v s /\
      (forall (out:Shared.public_outputs p).
        output_represents Base p n (from_base v) out == B.output_representation p n v out)) =
  assert (represented Base p n (from_base v) s == B.state_representation p n v s /\
    (forall (out:Shared.public_outputs p).
      output_represents Base p n (from_base v) out == B.output_representation p n v out)) by (
    FStar.Tactics.V2.norm [zeta_full; iota; primops; delta_only [
      "KPlay.Review.represented"; "KPlay.Review.output_represents";
      "KPlay.Review.count_value"; "KPlay.Review.from_base";
      "KPlay.Base.Functional.state_representation";
      "KPlay.Base.Functional.output_representation";
      "KPlay.Base.Functional.slot_offset"; "KPlay.Base.Functional.survival_offset"]];
    FStar.Tactics.V2.smt ())
inline_for_extraction noextract
let base_input_agrees (p:Shared.model_profile) (b:Shared.selected_boundary p)
  (step:call_index Base p) (aux:auxiliary Base p step)
  (s:lseq i32 1) (r:lseq f32 (Shared.expert_count p))
  : Lemma
      (requires input_represents Base p b step aux s r)
      (ensures B.input_representation p b (I64.v step) s r (to_base aux.previous)) =
  base_models_agree p (completed Base p step) (to_base aux.previous) (Shared.prefix_state b (I64.v step));
  assert_norm (from_base (to_base aux.previous) == FStar.Ghost.reveal aux.previous);
  ()
inline_for_extraction noextract
let base_post_refines (p:Shared.model_profile) (b:Shared.selected_boundary p)
  (step:call_index Base p)
  (r:lseq f32 (Shared.expert_count p))
  (before after:B.physical_state p (I64.v step + 1))
  (d:lseq i32 (demand_count (I64.v step)))
  : Lemma
      (requires B.semantic_post p b (I64.v step) r before after d)
      (ensures post Base p b step (from_base after) /\
        (I64.v step == 255 ==> (forall (e:nat{e < 8}).
          I32.v (Seq.index d e) == I64.v (Seq.index after.B.counts e)))) =
  base_models_agree p (I64.v step + 1) after (Shared.prefix_state b (I64.v step + 1));
  ()

inline_for_extraction noextract
let optimized_post_refines (p:Shared.model_profile) (b:Shared.selected_boundary p)
  (r:lseq f32 (Shared.token_count p * Shared.expert_count p))
  (s:lseq i32 (Shared.token_count p)) (best:lseq f32 (Shared.token_count p))
  (v:physical_state Optimized p (Shared.token_count p))
  : Lemma
      (requires O.semantic_post p b r s best v.counts v.indices v.weights v.flags)
      (ensures post Optimized p b () v) =
  let out = O.logical_output p v.counts v.indices v.weights v.flags in
  let vacant (e:Shared.expert_index p) (j:Shared.slot_index p)
    : Lemma (I64.v (Seq.index v.indices (e * Shared.capacity p + j)) == -1 ==>
        Seq.index v.weights (e * Shared.capacity p + j) == zero #f32) =
    assert (O.native_slot_post p r s best (Shared.token_count p) e j
      (Seq.index v.indices (e * Shared.capacity p + j))
      (Seq.index v.weights (e * Shared.capacity p + j)));
    ()
  in
  FStar.Classical.forall_intro_2 vacant;
  assert (represented Optimized p (Shared.token_count p) v (Shared.route_result b));
  assert (output_represents Optimized p (Shared.token_count p) v out);
  FStar.Classical.exists_intro
    (fun out -> output_represents Optimized p (Shared.token_count p) v out /\ Shared.route_post b out) out

let completed_refines_shared i p b v out =
  assert (Shared.state_representation (Shared.route_result b) out);
  ()

fn base
  (p:Shared.model_profile) (#b:erased (Shared.selected_boundary p))
  (step:call_index Base p)
  (selected:device_array i32 (input_rows Base p))
  (probabilities:device_array f32 (input_rows Base p * Shared.expert_count p))
  (output:buffers Base p (completed Base p step)) (aux:auxiliary Base p step)
  (#s:erased (lseq i32 (input_rows Base p)))
  (#r:erased (lseq f32 (input_rows Base p * Shared.expert_count p)))
  preserves cpu ** on gpu_loc (A.pts_to selected s ** A.pts_to probabilities r)
  requires pure (input_represents Base p b step aux s r) ** before Base p step aux **
    (exists* (old:physical_state Base p (completed Base p step)). owns Base p (completed Base p step) output old)
  ensures exists* (v:physical_state Base p (completed Base p step)).
    owns Base p (completed Base p step) output v ** after Base p step aux v ** pure (post Base p b step v)
{
  unfold before Base p step aux;
  unfold owns Base p (completed Base p step) aux.previous_buffers aux.previous;
  with old. assert (owns Base p (completed Base p step) output old);
  unfold owns Base p (completed Base p step) output old;
  with old_demand. assert on gpu_loc (A.pts_to aux.emitted_demand old_demand);
  let old_native = FStar.Ghost.hide (to_base old);
  introduce exists* (d:lseq i32 (demand_count (I64.v step))).
    on gpu_loc (
      A.pts_to output.counts_buffer old_native.B.counts **
      A.pts_to output.indices_buffer old_native.B.indices **
      A.pts_to output.weights_buffer old_native.B.weights **
      A.pts_to output.flags_buffer old_native.B.flags **
      A.pts_to aux.emitted_demand d) with old_demand;
  introduce exists* (v:B.physical_state p (I64.v step + 1)).
    exists* (d:lseq i32 (demand_count (I64.v step))).
    on gpu_loc (
      A.pts_to output.counts_buffer v.B.counts **
      A.pts_to output.indices_buffer v.B.indices **
      A.pts_to output.weights_buffer v.B.weights **
      A.pts_to output.flags_buffer v.B.flags **
      A.pts_to aux.emitted_demand d) with old_native;
  let _ = B.approved_profile_fits_native p (I64.v step);
  let _ = base_input_agrees p b step aux s r;
  let experts = (match p with | Shared.Smoke -> 8UL | Shared.Full -> 16UL);
  let capacity = (match p with | Shared.Smoke -> 32UL | Shared.Full -> 64UL);
  B.kplay_run #(FStar.Ghost.hide p) #b
    #(FStar.Ghost.hide (FStar.Ghost.reveal s)) #(FStar.Ghost.hide (FStar.Ghost.reveal r))
    experts capacity selected probabilities
    aux.previous_buffers.counts_buffer aux.previous_buffers.indices_buffer aux.previous_buffers.weights_buffer
    step aux.previous_buffers.flags_buffer #(FStar.Ghost.hide (to_base aux.previous))
    output.counts_buffer output.indices_buffer output.weights_buffer output.flags_buffer aux.emitted_demand;
  with native demand.
    assert on gpu_loc (
      A.pts_to output.counts_buffer native.B.counts ** A.pts_to output.indices_buffer native.B.indices **
      A.pts_to output.weights_buffer native.B.weights ** A.pts_to output.flags_buffer native.B.flags **
      A.pts_to aux.emitted_demand demand);
  let _ = base_post_refines p b step r (to_base aux.previous) native demand;
  let v = FStar.Ghost.hide (from_base native);
  assert pure (post Base p b step v);
  fold owns Base p (completed Base p step) output v;
  fold owns Base p (completed Base p step) aux.previous_buffers aux.previous;
  fold after Base p step aux v;
}

fn optimized
  (p:Shared.model_profile) (#b:erased (Shared.selected_boundary p))
  (step:call_index Optimized p)
  (selected:device_array i32 (input_rows Optimized p))
  (probabilities:device_array f32 (input_rows Optimized p * Shared.expert_count p))
  (output:buffers Optimized p (completed Optimized p step)) (aux:auxiliary Optimized p step)
  (#s:erased (lseq i32 (input_rows Optimized p)))
  (#r:erased (lseq f32 (input_rows Optimized p * Shared.expert_count p)))
  preserves cpu ** on gpu_loc (A.pts_to selected s ** A.pts_to probabilities r)
  requires pure (input_represents Optimized p b step aux s r) ** before Optimized p step aux **
    (exists* (old:physical_state Optimized p (completed Optimized p step)). owns Optimized p (completed Optimized p step) output old)
  ensures exists* (v:physical_state Optimized p (completed Optimized p step)).
    owns Optimized p (completed Optimized p step) output v ** after Optimized p step aux v ** pure (post Optimized p b step v)
{
  unfold before Optimized p step aux;
  with old. assert (owns Optimized p (completed Optimized p step) output old);
  unfold owns Optimized p (completed Optimized p step) output old;
  let _ = O.profile_bounds p;
  let tokens = (match p with | Shared.Smoke -> 256sz | Shared.Full -> 1024sz);
  let experts = (match p with | Shared.Smoke -> 8sz | Shared.Full -> 16sz);
  let capacity = (match p with | Shared.Smoke -> 32L | Shared.Full -> 64L);
  O.kplay_run #(FStar.Ghost.hide p) #b tokens experts capacity probabilities selected aux.best_buffer
    output.counts_buffer output.indices_buffer output.weights_buffer output.flags_buffer
    #_ #_ #_
    #(FStar.Ghost.hide (FStar.Ghost.reveal r)) #(FStar.Ghost.hide (FStar.Ghost.reveal s))
    #(FStar.Ghost.hide (FStar.Ghost.reveal aux.best));
  with counts indices weights flags.
    assert on gpu_loc (
      A.pts_to output.counts_buffer counts ** A.pts_to output.indices_buffer indices **
      A.pts_to output.weights_buffer weights ** A.pts_to output.flags_buffer flags);
  let v = FStar.Ghost.hide ({
    counts = counts; indices = indices; weights = weights; flags = flags
  } <: physical_state Optimized p (Shared.token_count p));
  let _ = optimized_post_refines p b r s aux.best v;
  assert pure (post Optimized p b step v);
  fold owns Optimized p (completed Optimized p step) output v;
  fold after Optimized p step aux v;
}

fn switcher
  (i:implementation) (p:Shared.model_profile) (#b:erased (Shared.selected_boundary p))
  (step:call_index i p)
  (selected:device_array i32 (input_rows i p))
  (probabilities:device_array f32 (input_rows i p * Shared.expert_count p))
  (output:buffers i p (completed i p step)) (aux:auxiliary i p step)
  (#s:erased (lseq i32 (input_rows i p)))
  (#r:erased (lseq f32 (input_rows i p * Shared.expert_count p)))
  preserves cpu ** on gpu_loc (A.pts_to selected s ** A.pts_to probabilities r)
  requires pure (input_represents i p b step aux s r) ** before i p step aux **
    (exists* (old:physical_state i p (completed i p step)). owns i p (completed i p step) output old)
  ensures exists* (v:physical_state i p (completed i p step)).
    owns i p (completed i p step) output v ** after i p step aux v ** pure (post i p b step v)
{
  match i {
    Base -> { base p #b step selected probabilities output aux #s #r }
    Optimized -> { optimized p #b step selected probabilities output aux #s #r }
  }
}

#pop-options
