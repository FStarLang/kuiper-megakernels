module KPlay.Review
#lang-pulse

open Kuiper
open Pulse.Lib.Send
module Seq = FStar.Seq
module I64 = FStar.Int64
module A = Pulse.Lib.Array
module Shared = KPlay.SharedSpec
module B = KPlay.Base.Functional
module O = KPlay.Optimized.Functional

#push-options "--z3rlimit 10 --fuel 4 --ifuel 2"

inline_for_extraction noextract
let base_input (input:logical_inputs Base) : B.logical_inputs = {
  B.call = input.call; B.packed_weights = input.packed_weights;
  B.input_hidden = input.input_hidden; B.old_keys = input.old_keys;
  B.old_values = input.old_values; B.frame = input.frame
}
inline_for_extraction noextract
let optimized_input (input:logical_inputs Optimized) : O.logical_inputs =
  Shared.full_profile_position_bound input.call;
  {
    O.call = input.call; O.packed_weights = input.packed_weights;
    O.input_hidden = input.input_hidden; O.old_keys = input.old_keys;
    O.old_values = input.old_values; O.frame = input.frame
  }
inline_for_extraction noextract
let base_models_agree (input:logical_inputs Base)
  : Lemma (
      before Base input == B.logical_before (base_input input) /\
      (forall (h:Shared.vector 1024)
        (k v:Shared.vector (output_count Base input.call.position)).
        result Base input h k v == B.result_view input.call input.frame h k v)) =
  assert (before Base input == B.logical_before (base_input input) /\
    (forall (h:Shared.vector 1024) (k v:Shared.vector (output_count Base input.call.position)).
      result Base input h k v == B.result_view input.call input.frame h k v)) by (
    FStar.Tactics.V2.norm [zeta_full; iota; primops; delta_only [
      "KPlay.Review.before"; "KPlay.Review.result"; "KPlay.Review.history";
      "KPlay.Review.cache_index"; "KPlay.Review.base_input";
      "KPlay.Review.previous_count"; "KPlay.Review.output_count";
      "KPlay.Base.Functional.logical_before"; "KPlay.Base.Functional.cache_view";
      "KPlay.Base.Functional.old_history_view"; "KPlay.Base.Functional.history_view";
      "KPlay.Base.Functional.kv_view"; "KPlay.Base.Functional.result_view";
      "KPlay.Base.Functional.prefix_count"; "KPlay.Base.Functional.next_count";
      "KPlay.Base.Functional.kv_width"]];
    FStar.Tactics.V2.smt ())
inline_for_extraction noextract
let optimized_models_agree (input:logical_inputs Optimized)
  : Lemma (
      before Optimized input == O.logical_before (optimized_input input) /\
      B.weights_view input.packed_weights == O.weights_view input.packed_weights /\
      (forall (h:Shared.vector 1024)
        (k v:Shared.vector (output_count Optimized input.call.position)).
        result Optimized input h k v == O.result_view input.call input.frame h k v)) =
  Shared.full_profile_position_bound input.call;
  assert_norm (B.weights_view input.packed_weights == O.weights_view input.packed_weights);
  assert (before Optimized input == O.logical_before (optimized_input input) /\
    (forall (h:Shared.vector 1024) (k v:Shared.vector (output_count Optimized input.call.position)).
      result Optimized input h k v == O.result_view input.call input.frame h k v)) by (
    FStar.Tactics.V2.norm [zeta_full; iota; primops; delta_only [
      "KPlay.Review.before"; "KPlay.Review.result"; "KPlay.Review.history";
      "KPlay.Review.cache_index"; "KPlay.Review.optimized_input";
      "KPlay.Review.previous_count"; "KPlay.Review.output_count";
      "KPlay.Optimized.Functional.logical_before"; "KPlay.Optimized.Functional.cache_view";
      "KPlay.Optimized.Functional.old_history_view"; "KPlay.Optimized.Functional.history_view";
      "KPlay.Optimized.Functional.kv_view"; "KPlay.Optimized.Functional.result_view";
      "KPlay.Optimized.Functional.cache_index"; "KPlay.Optimized.Functional.previous_elements";
      "KPlay.Optimized.Functional.cache_elements"; "KPlay.Optimized.Functional.cache_head_elements";
      "KPlay.Optimized.Functional.cache_positions"; "KPlay.Optimized.Functional.head_width";
      "KPlay.Optimized.Functional.kv_heads"]];
    FStar.Tactics.V2.smt ())

inline_for_extraction noextract
let base_representation_agrees (input:logical_inputs Base) (aux:rotary_buffer)
  (w:lseq bf16 weight_count) (h:lseq bf16 1024)
  (k v:lseq bf16 (previous_count Base input.call.position))
  : Lemma
      (requires represented Base input w h k v /\ auxiliary_represents Base input.call.position aux)
      (ensures B.input_representation (base_input input) w h aux.values k v) =
  ()
inline_for_extraction noextract
let optimized_representation_agrees (input:logical_inputs Optimized)
  (w:lseq bf16 weight_count) (h:lseq bf16 1024)
  (k v:lseq bf16 (previous_count Optimized input.call.position))
  : Lemma
      (requires represented Optimized input w h k v)
      (ensures O.input_representation (optimized_input input) w h k v) =
  Shared.full_profile_position_bound input.call

inline_for_extraction noextract
let base_post_refines (input:logical_inputs Base)
  (old_k old_v:lseq bf16 (previous_count Base input.call.position))
  (h:lseq bf16 1024) (k v:lseq bf16 (output_count Base input.call.position))
  : Lemma
      (requires B.semantic_postcondition (base_input input) old_k old_v h k v)
      (ensures post Base input old_k old_v h k v ()) =
  base_models_agree input;
  assert (forall (n:nat) (physical:lseq bf16 n) (logical:Shared.vector n).
    represents #n physical logical == B.represents n physical logical);
  ()
inline_for_extraction noextract
let optimized_post_refines (input:logical_inputs Optimized)
  (old_k old_v:lseq bf16 (previous_count Optimized input.call.position))
  (h:lseq bf16 1024) (k v:lseq bf16 (output_count Optimized input.call.position))
  (scores:score_observation Optimized input.call.position)
  : Lemma
      (requires O.source_score_post (optimized_input input) old_k old_v h k v scores)
      (ensures post Optimized input old_k old_v h k v scores) =
  optimized_models_agree input;
  assert (forall (n:nat) (physical:lseq bf16 n) (logical:Shared.vector n).
    represents #n physical logical == O.represents n physical logical);
  assert (numerical_premise Optimized input.call.position scores ==
    O.bilateral_live_scores_finite input.call.position scores);
  ()

ghost
fn state_to_base (position:nat) (ptr:state_buffer Base position) (fraction:perm)
  (values:lseq bf16 (previous_count Base position))
  requires state_owns Base position ptr fraction values
  ensures B.prefix_ownership position ptr fraction values
{
  unfold state_owns Base position ptr fraction values;
  if (position = 0) {
    rewrite (pure (ptr == A.null)) as B.prefix_ownership position ptr fraction values;
  } else {
    rewrite (pure (is_global_array ptr /\ A.length ptr == B.prefix_count position) **
      on gpu_loc (A.pts_to ptr #fraction values)) as B.prefix_ownership position ptr fraction values;
  }
}
ghost
fn state_from_base (position:nat) (ptr:state_buffer Base position) (fraction:perm)
  (values:lseq bf16 (previous_count Base position))
  requires B.prefix_ownership position ptr fraction values
  ensures state_owns Base position ptr fraction values
{
  unfold B.prefix_ownership position ptr fraction values;
  if (position = 0) {
    rewrite emp as state_owns Base position ptr fraction values;
  } else {
    rewrite on gpu_loc (A.pts_to ptr #fraction values) as state_owns Base position ptr fraction values;
  }
}
ghost
fn state_to_optimized (position:nat{position < 2048})
  (ptr:state_buffer Optimized position) (fraction:perm)
  (values:lseq bf16 (previous_count Optimized position))
  requires state_owns Optimized position ptr fraction values
  ensures on gpu_loc (O.state_ownership ptr position fraction values)
{
  unfold state_owns Optimized position ptr fraction values;
  if (position = 0) {
    on_emp_intro gpu_loc;
    rewrite on gpu_loc emp as on gpu_loc (O.state_ownership ptr position fraction values);
  } else {
    on_pure_intro gpu_loc (is_global_array ptr);
    rewrite on gpu_loc (pure (is_global_array ptr) ** A.pts_to ptr #fraction values)
      as on gpu_loc (O.state_ownership ptr position fraction values);
  }
}
ghost
fn state_from_optimized (position:nat{position < 2048})
  (ptr:state_buffer Optimized position) (fraction:perm)
  (values:lseq bf16 (previous_count Optimized position))
  requires on gpu_loc (O.state_ownership ptr position fraction values)
  ensures state_owns Optimized position ptr fraction values
{
  if (position = 0) {
    rewrite on gpu_loc (O.state_ownership ptr position fraction values) as on gpu_loc emp;
    on_emp_elim gpu_loc;
    rewrite emp as state_owns Optimized position ptr fraction values;
  } else {
    rewrite on gpu_loc (O.state_ownership ptr position fraction values)
      as on gpu_loc (pure (is_global_array ptr) ** A.pts_to ptr #fraction values);
    rewrite on gpu_loc (A.pts_to ptr #fraction values) as state_owns Optimized position ptr fraction values;
  }
}

fn base
  (#input:erased (logical_inputs Base)) (position:i64{I64.v position == input.call.position})
  (weights:device_array bf16 weight_count) (hidden:device_array bf16 1024)
  (old_keys old_values:state_buffer Base input.call.position)
  (aux:auxiliary Base)
  (out_hidden:device_array bf16 1024)
  (new_keys new_values:device_array bf16 (output_count Base input.call.position))
  (#w:erased (lseq bf16 weight_count)) (#h:erased (lseq bf16 1024))
  (#k #v:erased (lseq bf16 (previous_count Base input.call.position)))
  (#pw #ph #pk #pv #pr:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w ** A.pts_to hidden #ph h) **
    state_owns Base input.call.position old_keys pk k **
    state_owns Base input.call.position old_values pv v ** auxiliary_owns Base pr aux
  requires pure (represented Base input w h k v /\ auxiliary_represents Base input.call.position aux) **
    on gpu_loc (A.pts_to_uninit out_hidden 1024 **
      A.pts_to_uninit new_keys (output_count Base input.call.position) **
      A.pts_to_uninit new_values (output_count Base input.call.position))
  returns scores:erased (score_observation Base input.call.position)
  ensures exists* (nh:lseq bf16 1024) (nk nv:lseq bf16 (output_count Base input.call.position)).
    on gpu_loc (A.pts_to out_hidden nh ** A.pts_to new_keys nk ** A.pts_to new_values nv) **
    pure (post Base input k v nh nk nv scores)
{
  let _ = base_representation_agrees input aux w h k v;
  state_to_base input.call.position old_keys pk k;
  state_to_base input.call.position old_values pv v;
  unfold auxiliary_owns Base pr aux;
  B.kplay_run #(FStar.Ghost.hide (base_input input))
    #(FStar.Ghost.hide (FStar.Ghost.reveal w)) #(FStar.Ghost.hide (FStar.Ghost.reveal h))
    #(FStar.Ghost.hide (FStar.Ghost.reveal aux.values))
    #(FStar.Ghost.hide (FStar.Ghost.reveal k)) #(FStar.Ghost.hide (FStar.Ghost.reveal v))
    #pw #ph #pr #pk #pv weights hidden aux.pointer old_keys old_values position out_hidden new_keys new_values;
  with nh nk nv.
    assert on gpu_loc (A.pts_to out_hidden nh ** A.pts_to new_keys nk ** A.pts_to new_values nv);
  state_from_base input.call.position old_keys pk k;
  state_from_base input.call.position old_values pv v;
  fold auxiliary_owns Base pr aux;
  let _ = base_post_refines input k v nh nk nv;
  assert pure (post Base input k v nh nk nv ());
  FStar.Ghost.hide ()
}

fn optimized
  (#input:erased (logical_inputs Optimized)) (position:i64{I64.v position == input.call.position})
  (weights:device_array bf16 weight_count) (hidden:device_array bf16 1024)
  (old_keys old_values:state_buffer Optimized input.call.position)
  (aux:auxiliary Optimized)
  (out_hidden:device_array bf16 1024)
  (new_keys new_values:device_array bf16 (output_count Optimized input.call.position))
  (#w:erased (lseq bf16 weight_count)) (#h:erased (lseq bf16 1024))
  (#k #v:erased (lseq bf16 (previous_count Optimized input.call.position)))
  (#pw #ph #pk #pv #pr:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w ** A.pts_to hidden #ph h) **
    state_owns Optimized input.call.position old_keys pk k **
    state_owns Optimized input.call.position old_values pv v ** auxiliary_owns Optimized pr aux
  requires pure (represented Optimized input w h k v /\ auxiliary_represents Optimized input.call.position aux) **
    on gpu_loc (A.pts_to_uninit out_hidden 1024 **
      A.pts_to_uninit new_keys (output_count Optimized input.call.position) **
      A.pts_to_uninit new_values (output_count Optimized input.call.position))
  returns scores:erased (score_observation Optimized input.call.position)
  ensures exists* (nh:lseq bf16 1024) (nk nv:lseq bf16 (output_count Optimized input.call.position)).
    on gpu_loc (A.pts_to out_hidden nh ** A.pts_to new_keys nk ** A.pts_to new_values nv) **
    pure (post Optimized input k v nh nk nv scores)
{
  let _ = Shared.full_profile_position_bound input.call;
  let _ = optimized_representation_agrees input w h k v;
  state_to_optimized input.call.position old_keys pk k;
  state_to_optimized input.call.position old_values pv v;
  rewrite on gpu_loc (O.state_ownership old_keys input.call.position pk k)
    as on gpu_loc (O.state_ownership old_keys (I64.v position) pk k);
  rewrite on gpu_loc (O.state_ownership old_values input.call.position pv v)
    as on gpu_loc (O.state_ownership old_values (I64.v position) pv v);
  let scores = O.kplay_run #(FStar.Ghost.hide (optimized_input input))
    weights hidden old_keys old_values position out_hidden new_keys new_values
    #(FStar.Ghost.hide (FStar.Ghost.reveal w)) #(FStar.Ghost.hide (FStar.Ghost.reveal h))
    #(FStar.Ghost.hide (FStar.Ghost.reveal k)) #(FStar.Ghost.hide (FStar.Ghost.reveal v))
    #pw #ph #pk #pv;
  with nh nk nv.
    assert on gpu_loc (A.pts_to out_hidden nh ** A.pts_to new_keys nk ** A.pts_to new_values nv);
  rewrite on gpu_loc (O.state_ownership old_keys (I64.v position) pk k)
    as on gpu_loc (O.state_ownership old_keys input.call.position pk k);
  rewrite on gpu_loc (O.state_ownership old_values (I64.v position) pv v)
    as on gpu_loc (O.state_ownership old_values input.call.position pv v);
  state_from_optimized input.call.position old_keys pk k;
  state_from_optimized input.call.position old_values pv v;
  let _ = optimized_post_refines input k v nh nk nv scores;
  assert pure (post Optimized input k v nh nk nv scores);
  scores
}

#pop-options
