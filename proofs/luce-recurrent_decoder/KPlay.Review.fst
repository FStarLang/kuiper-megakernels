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
let base_representation_agreement ()
  : Lemma (
      hidden_representation == B.bf16_vector_representation #hidden_count /\
      recurrent_representation Base == B.recurrent_representation /\
      convolution_representation Base == B.convolution_representation /\
      call_postcondition Base == B.call_postcondition) =
  assert (
      hidden_representation == B.bf16_vector_representation #hidden_count /\
      recurrent_representation Base == B.recurrent_representation /\
      convolution_representation Base == B.convolution_representation /\
      call_postcondition Base == B.call_postcondition) by (
    FStar.Tactics.V2.norm [zeta; iota; primops; delta_only [
      "KPlay.Review.hidden_representation"; "KPlay.Review.recurrent_representation";
      "KPlay.Review.convolution_representation"; "KPlay.Review.recurrent_index";
      "KPlay.Review.represents_cell"; "KPlay.Review.convolution_type";
      "KPlay.Review.call_postcondition"; "KPlay.Review.output_representation";
      "KPlay.Review.hidden_count"; "KPlay.Review.recurrent_count";
      "KPlay.Review.convolution_count";
      "KPlay.Base.Functional.bf16_vector_representation";
      "KPlay.Base.Functional.recurrent_representation";
      "KPlay.Base.Functional.convolution_representation";
      "KPlay.Base.Functional.call_postcondition"; "KPlay.Base.Functional.output_representation";
      "KPlay.Base.Functional.logical_result"; "KPlay.Base.Functional.row_major";
      "KPlay.Base.Functional.recurrent_index"; "KPlay.Base.Functional.convolution_index";
      "KPlay.Base.Functional.hidden_width"; "KPlay.Base.Functional.heads";
      "KPlay.Base.Functional.head_width"; "KPlay.Base.Functional.conv_width";
      "KPlay.Base.Functional.mixer_width"; "KPlay.Base.Functional.qkv_width";
      "KPlay.Base.Functional.recurrent_count"; "KPlay.Base.Functional.convolution_count"]];
    FStar.Tactics.V2.smt ())

inline_for_extraction noextract
let base_post_refines (position:position_t)
  (w:lseq bf16 weight_count) (h:lseq bf16 hidden_count)
  (r:lseq f32 recurrent_count) (c:lseq bf16 convolution_count)
  (nh:lseq bf16 hidden_count) (nr:lseq f32 recurrent_count) (nc:lseq bf16 convolution_count)
  : Lemma
      (requires B.semantic_postcondition w h r c nh nr nc)
      (ensures post Base position w h r c nh nr nc) =
  B.physical_extents ();
  base_representation_agreement ();
  ()

inline_for_extraction noextract
let optimized_representation_agreement ()
  : Lemma (
      hidden_representation == O.bf16_vector_representation #hidden_count /\
      recurrent_representation Optimized == O.recurrent_representation /\
      convolution_representation Optimized == O.convolution_representation /\
      call_postcondition Optimized == O.call_postcondition) =
  assert (
      hidden_representation == O.bf16_vector_representation #hidden_count /\
      recurrent_representation Optimized == O.recurrent_representation /\
      convolution_representation Optimized == O.convolution_representation /\
      call_postcondition Optimized == O.call_postcondition) by (
    FStar.Tactics.V2.norm [zeta; iota; primops; delta_only [
      "KPlay.Review.hidden_representation"; "KPlay.Review.recurrent_representation";
      "KPlay.Review.convolution_representation"; "KPlay.Review.recurrent_index";
      "KPlay.Review.represents_cell"; "KPlay.Review.convolution_type";
      "KPlay.Review.call_postcondition"; "KPlay.Review.output_representation";
      "KPlay.Review.hidden_count"; "KPlay.Review.recurrent_count";
      "KPlay.Review.convolution_count";
      "KPlay.Optimized.Functional.bf16_vector_representation";
      "KPlay.Optimized.Functional.recurrent_representation";
      "KPlay.Optimized.Functional.convolution_representation";
      "KPlay.Optimized.Functional.call_postcondition"; "KPlay.Optimized.Functional.output_representation";
      "KPlay.Optimized.Functional.logical_result"; "KPlay.Optimized.Functional.row_major";
      "KPlay.Optimized.Functional.recurrent_index"; "KPlay.Optimized.Functional.convolution_index";
      "KPlay.Optimized.Functional.hidden_width"; "KPlay.Optimized.Functional.heads";
      "KPlay.Optimized.Functional.head_width"; "KPlay.Optimized.Functional.conv_width";
      "KPlay.Optimized.Functional.mixer_width"; "KPlay.Optimized.Functional.qkv_width";
      "KPlay.Optimized.Functional.recurrent_count"; "KPlay.Optimized.Functional.convolution_count"]];
    FStar.Tactics.V2.smt ())

inline_for_extraction noextract
let optimized_post_refines (position:position_t)
  (w:lseq bf16 weight_count) (h:lseq bf16 hidden_count)
  (r:lseq f32 (previous_count Optimized position recurrent_count))
  (c:lseq f32 (previous_count Optimized position convolution_count))
  (nh:lseq bf16 hidden_count) (nr:lseq f32 recurrent_count) (nc:lseq f32 convolution_count)
  : Lemma
      (requires O.semantic_postcondition w h (I64.v position) r c nh nr nc)
      (ensures post Optimized position w h r c nh nr nc) =
  B.physical_extents ();
  O.physical_extents ();
  assert_norm (B.weights_representation == O.weights_representation);
  assert_norm (B.old_recurrent == O.old_recurrent /\ B.old_convolution == O.old_convolution);
  optimized_representation_agreement ();
  ()

ghost
fn state_to_optimized (position:position_t) (count:pos)
  (ptr:state_buffer f32 (previous_count Optimized position count)) (fraction:perm)
  (values:lseq f32 (previous_count Optimized position count))
  requires state_owns ptr fraction values
  ensures on gpu_loc (O.state_ownership ptr (I64.v position) count fraction values)
{
  if (I64.v position = 0) {
    rewrite state_owns ptr fraction values as emp;
    on_emp_intro gpu_loc;
    rewrite on gpu_loc emp as on gpu_loc (O.state_ownership ptr (I64.v position) count fraction values);
  } else {
    rewrite state_owns ptr fraction values as on gpu_loc (A.pts_to ptr #fraction values);
    on_pure_intro gpu_loc (is_global_array ptr);
    rewrite on gpu_loc (pure (is_global_array ptr) ** A.pts_to ptr #fraction values)
      as on gpu_loc (O.state_ownership ptr (I64.v position) count fraction values);
  }
}

ghost
fn state_from_optimized (position:position_t) (count:pos)
  (ptr:state_buffer f32 (previous_count Optimized position count)) (fraction:perm)
  (values:lseq f32 (previous_count Optimized position count))
  requires on gpu_loc (O.state_ownership ptr (I64.v position) count fraction values)
  ensures state_owns ptr fraction values
{
  if (I64.v position = 0) {
    rewrite on gpu_loc (O.state_ownership ptr (I64.v position) count fraction values) as on gpu_loc emp;
    on_emp_elim gpu_loc;
    rewrite emp as state_owns ptr fraction values;
  } else {
    rewrite on gpu_loc (O.state_ownership ptr (I64.v position) count fraction values)
      as on gpu_loc (pure (is_global_array ptr) ** A.pts_to ptr #fraction values);
    rewrite on gpu_loc (A.pts_to ptr #fraction values) as state_owns ptr fraction values;
  }
}

fn base
  (position:position_t)
  (weights:larray bf16 weight_count{is_global_array weights})
  (hidden:larray bf16 hidden_count{is_global_array hidden})
  (recurrent:state_buffer f32 (previous_count Base position recurrent_count))
  (convolution:state_buffer bf16 (previous_count Base position convolution_count))
  (out_hidden:larray bf16 hidden_count{is_global_array out_hidden})
  (out_recurrent:larray f32 recurrent_count{is_global_array out_recurrent})
  (out_convolution:larray bf16 convolution_count{is_global_array out_convolution})
  (#w:erased (lseq bf16 weight_count)) (#h:erased (lseq bf16 hidden_count))
  (#r:erased (lseq f32 (previous_count Base position recurrent_count)))
  (#c:erased (lseq bf16 (previous_count Base position convolution_count)))
  (#pw #ph #pr #pc:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w ** A.pts_to hidden #ph h) **
    state_owns recurrent pr r ** state_owns convolution pc c
  requires exists* (nh:lseq bf16 hidden_count) (nr:lseq f32 recurrent_count)
    (nc:lseq bf16 convolution_count).
    on gpu_loc (A.pts_to out_hidden nh ** A.pts_to out_recurrent nr ** A.pts_to out_convolution nc)
  ensures exists* (nh:lseq bf16 hidden_count) (nr:lseq f32 recurrent_count)
    (nc:lseq bf16 convolution_count).
    on gpu_loc (A.pts_to out_hidden nh ** A.pts_to out_recurrent nr ** A.pts_to out_convolution nc) **
    pure (post Base position w h r c nh nr nc)
{
  let _ = B.physical_extents ();
  rewrite state_owns recurrent pr r as on gpu_loc (A.pts_to recurrent #pr r);
  rewrite state_owns convolution pc c as on gpu_loc (A.pts_to convolution #pc c);
  B.kplay_run
    #(FStar.Ghost.hide (FStar.Ghost.reveal w))
    #(FStar.Ghost.hide (FStar.Ghost.reveal h))
    #(FStar.Ghost.hide (FStar.Ghost.reveal r))
    #(FStar.Ghost.hide (FStar.Ghost.reveal c))
    #pw #ph #pr #pc weights hidden recurrent convolution out_hidden out_recurrent out_convolution;
  with nh nr nc.
    assert on gpu_loc (A.pts_to out_hidden nh ** A.pts_to out_recurrent nr ** A.pts_to out_convolution nc);
  rewrite on gpu_loc (A.pts_to recurrent #pr r) as state_owns recurrent pr r;
  rewrite on gpu_loc (A.pts_to convolution #pc c) as state_owns convolution pc c;
  let _ = base_post_refines position w h r c nh nr nc;
  assert pure (post Base position w h r c nh nr nc);
}

fn optimized
  (position:position_t)
  (weights:larray bf16 weight_count{is_global_array weights})
  (hidden:larray bf16 hidden_count{is_global_array hidden})
  (recurrent:state_buffer f32 (previous_count Optimized position recurrent_count))
  (convolution:state_buffer f32 (previous_count Optimized position convolution_count))
  (out_hidden:larray bf16 hidden_count{is_global_array out_hidden})
  (out_recurrent:larray f32 recurrent_count{is_global_array out_recurrent})
  (out_convolution:larray f32 convolution_count{is_global_array out_convolution})
  (#w:erased (lseq bf16 weight_count)) (#h:erased (lseq bf16 hidden_count))
  (#r:erased (lseq f32 (previous_count Optimized position recurrent_count)))
  (#c:erased (lseq f32 (previous_count Optimized position convolution_count)))
  (#pw #ph #pr #pc:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w ** A.pts_to hidden #ph h) **
    state_owns recurrent pr r ** state_owns convolution pc c
  requires exists* (nh:lseq bf16 hidden_count) (nr:lseq f32 recurrent_count)
    (nc:lseq f32 convolution_count).
    on gpu_loc (A.pts_to out_hidden nh ** A.pts_to out_recurrent nr ** A.pts_to out_convolution nc)
  ensures exists* (nh:lseq bf16 hidden_count) (nr:lseq f32 recurrent_count)
    (nc:lseq f32 convolution_count).
    on gpu_loc (A.pts_to out_hidden nh ** A.pts_to out_recurrent nr ** A.pts_to out_convolution nc) **
    pure (post Optimized position w h r c nh nr nc)
{
  let _ = O.physical_extents ();
  state_to_optimized position recurrent_count recurrent pr r;
  state_to_optimized position convolution_count convolution pc c;
  O.kplay_run weights hidden recurrent convolution position out_hidden out_recurrent out_convolution
    #(FStar.Ghost.hide (FStar.Ghost.reveal w))
    #(FStar.Ghost.hide (FStar.Ghost.reveal h))
    #(FStar.Ghost.hide (FStar.Ghost.reveal r))
    #(FStar.Ghost.hide (FStar.Ghost.reveal c))
    #pw #ph #pr #pc;
  with nh nr nc.
    assert on gpu_loc (A.pts_to out_hidden nh ** A.pts_to out_recurrent nr ** A.pts_to out_convolution nc);
  state_from_optimized position recurrent_count recurrent pr r;
  state_from_optimized position convolution_count convolution pc c;
  let _ = optimized_post_refines position w h r c nh nr nc;
  assert pure (post Optimized position w h r c nh nr nc);
}

fn switcher
  (i:implementation) (position:position_t)
  (weights:larray bf16 weight_count{is_global_array weights})
  (hidden:larray bf16 hidden_count{is_global_array hidden})
  (recurrent:state_buffer f32 (previous_count i position recurrent_count))
  (convolution:state_buffer (convolution_type i) (previous_count i position convolution_count))
  (out_hidden:larray bf16 hidden_count{is_global_array out_hidden})
  (out_recurrent:larray f32 recurrent_count{is_global_array out_recurrent})
  (out_convolution:larray (convolution_type i) convolution_count{is_global_array out_convolution})
  (#w:erased (lseq bf16 weight_count)) (#h:erased (lseq bf16 hidden_count))
  (#r:erased (lseq f32 (previous_count i position recurrent_count)))
  (#c:erased (lseq (convolution_type i) (previous_count i position convolution_count)))
  (#pw #ph #pr #pc:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w ** A.pts_to hidden #ph h) **
    state_owns recurrent pr r ** state_owns convolution pc c
  requires exists* (nh:lseq bf16 hidden_count) (nr:lseq f32 recurrent_count)
    (nc:lseq (convolution_type i) convolution_count).
    on gpu_loc (A.pts_to out_hidden nh ** A.pts_to out_recurrent nr ** A.pts_to out_convolution nc)
  ensures exists* (nh:lseq bf16 hidden_count) (nr:lseq f32 recurrent_count)
    (nc:lseq (convolution_type i) convolution_count).
    on gpu_loc (A.pts_to out_hidden nh ** A.pts_to out_recurrent nr ** A.pts_to out_convolution nc) **
    pure (post i position w h r c nh nr nc)
{
  match i {
    Base -> {
      base position weights hidden recurrent convolution out_hidden out_recurrent out_convolution
        #w #h #r #c #pw #ph #pr #pc
    }
    Optimized -> {
      optimized position weights hidden recurrent convolution out_hidden out_recurrent out_convolution
        #w #h #r #c #pw #ph #pr #pc
    }
  }
}

#pop-options
