module KPlay.Review
#lang-pulse

open Kuiper
open Pulse.Lib.Send
module S = FStar.Seq
module I64 = FStar.Int64
module A = Pulse.Lib.Array
module Shared = KPlay.SharedSpec
module B = KPlay.Base.Functional
module O = KPlay.Optimized.Functional

#push-options "--z3rlimit 10 --fuel 4 --ifuel 2"

let base_post_refines_optimized
  (position:position_t) (prompt:prompt_length_t)
  (logical:Shared.vec Shared.score vocabulary)
  (previous:lseq i64 (previous_count position prompt))
  (selected:lseq i64 1) (emitted:lseq i64 (output_count position prompt))
  : Lemma
      (requires post Base position prompt logical previous selected emitted)
      (ensures post Optimized position prompt logical previous selected emitted)
  =
  let winner : Shared.token vocabulary = I64.v (S.index selected 0) in
  Shared.first_maximum_refines logical winner

inline_for_extraction noextract
let base_post
  (position:position_t) (prompt:prompt_length_t)
  (xs:lseq bf16 vocabulary) (ps:lseq i64 (previous_count position prompt))
  (logical:Shared.vec Shared.score vocabulary)
  (selected:lseq i64 1) (emitted:lseq i64 (output_count position prompt))
  : Lemma
      (requires represents_scores Base xs logical /\
        B.functional_post #B.vocabulary (B.prompt_size prompt) (B.position_index position)
          xs ps selected emitted)
      (ensures post Base position prompt logical ps selected emitted)
  =
  assert (represents_scores Base xs logical == B.scores_represent #vocabulary xs logical)
    by (FStar.Tactics.V2.norm [zeta; iota; delta_only [
      "KPlay.Review.represents_scores"; "KPlay.Review.widened";
      "KPlay.Review.score_class"; "KPlay.Review.priority_le";
      "KPlay.Review.positive_infinity"; "KPlay.Review.vocabulary";
      "KPlay.SharedSpec.token"; "KPlay.Base.Functional.scores_represent";
      "KPlay.Base.Functional.widened_scores"; "KPlay.Base.Functional.widen";
      "KPlay.Base.Functional.score_class"; "KPlay.Base.Functional.priority_le";
      "KPlay.Base.Functional.positive_infinity"; "KPlay.Base.Functional.negative_infinity"]];
      FStar.Tactics.V2.trefl ());
  B.selected_representation #vocabulary (B.prompt_size prompt) (B.position_index position)
    xs ps selected emitted logical;
  Shared.argmax_correct logical

inline_for_extraction noextract
let optimized_post
  (position:position_t) (prompt:prompt_length_t)
  (xs:lseq f32 vocabulary) (ps:lseq i64 (previous_count position prompt))
  (logical:Shared.vec Shared.score vocabulary)
  (selected:lseq i64 1) (emitted:lseq i64 (output_count position prompt))
  : Lemma
      (requires represents_scores Optimized xs logical /\
        O.semantic_post xs ps position prompt selected emitted)
      (ensures post Optimized position prompt logical ps selected emitted)
  =
  assert (represents_scores Optimized xs logical == O.scores_represent #vocabulary xs logical)
    by (FStar.Tactics.V2.norm [zeta; iota; delta_only [
      "KPlay.Review.represents_scores"; "KPlay.Review.widened";
      "KPlay.Review.score_class"; "KPlay.Review.priority_le";
      "KPlay.Review.positive_infinity"; "KPlay.Review.vocabulary";
      "KPlay.SharedSpec.token"; "KPlay.Optimized.Functional.scores_represent";
      "KPlay.Optimized.Functional.score_class"; "KPlay.Optimized.Functional.priority_le";
      "KPlay.Optimized.Functional.positive_infinity"; "KPlay.Optimized.Functional.negative_infinity"]];
      FStar.Tactics.V2.trefl ());
  O.semantic_post_refines position prompt xs ps selected emitted;
  let winner : Shared.token vocabulary = I64.v (S.index selected 0) in
  O.selected_maximum #vocabulary (I64.v prompt) (I64.v position) xs ps selected emitted logical winner

ghost
fn history_to_base (ptr:array i64) (values:seq i64)
  requires history ptr values
  ensures on gpu_loc (B.active_contents ptr #1.0R values)
{
  unfold history ptr values;
  if (S.length values = 0) {
    on_emp_intro gpu_loc;
    rewrite on gpu_loc emp as on gpu_loc (B.active_contents ptr #1.0R values);
  } else {
    rewrite on gpu_loc (A.pts_to ptr values) as
      on gpu_loc (B.active_contents ptr #1.0R values);
  }
}

ghost
fn history_from_base (ptr:array i64) (values:seq i64)
  requires on gpu_loc (B.active_contents ptr #1.0R values) **
    pure (active_array (S.length values) ptr)
  ensures history ptr values
{
  if (S.length values = 0) {
    rewrite on gpu_loc (B.active_contents ptr #1.0R values) as on gpu_loc emp;
    on_emp_elim gpu_loc;
    rewrite emp as history ptr values;
  } else {
    rewrite on gpu_loc (B.active_contents ptr #1.0R values) as
      on gpu_loc (A.pts_to ptr values);
    assert pure (is_global_array ptr);
    rewrite (on gpu_loc (A.pts_to ptr values) ** pure (is_global_array ptr))
      as history ptr values;
  }
}

fn base
  (position:position_t) (prompt:prompt_length_t)
  (scores:larray bf16 vocabulary{is_global_array scores})
  (previous:array i64{active_array (previous_count position prompt) previous})
  (selected:larray i64 1{is_global_array selected})
  (emitted:array i64{active_array (output_count position prompt) emitted})
  (#xs:erased (lseq bf16 vocabulary))
  (#ps:erased (lseq i64 (previous_count position prompt)))
  (#logical:erased (Shared.vec Shared.score vocabulary))
  preserves cpu ** on gpu_loc (A.pts_to scores xs) ** history previous ps
  requires pure (represents_scores Base xs logical)
  requires exists* (s:lseq i64 1) (e:lseq i64 (output_count position prompt)).
    on gpu_loc (A.pts_to selected s) ** history emitted e
  ensures exists* (s:lseq i64 1) (e:lseq i64 (output_count position prompt)).
    on gpu_loc (A.pts_to selected s) ** history emitted e **
    pure (post Base position prompt logical ps s e /\
          post Optimized position prompt logical ps s e)
{
  with old_selected old_emitted.
    assert on gpu_loc (A.pts_to selected old_selected) ** history emitted old_emitted;
  history_to_base previous ps;
  history_to_base emitted old_emitted;
  B.kplay_run position prompt #1.0R #1.0R
    #(FStar.Ghost.hide (FStar.Ghost.reveal xs))
    #(FStar.Ghost.hide (FStar.Ghost.reveal ps))
    scores previous selected emitted;
  with next_selected next_emitted.
    assert on gpu_loc (A.pts_to selected next_selected **
      B.active_contents emitted #1.0R next_emitted);
  history_from_base previous ps;
  history_from_base emitted next_emitted;
  let _ = base_post position prompt xs ps logical next_selected next_emitted;
  assert pure (post Base position prompt logical ps next_selected next_emitted);
  let _ = base_post_refines_optimized position prompt logical ps next_selected next_emitted;
  assert pure (post Optimized position prompt logical ps next_selected next_emitted);
}

fn optimized
  (position:position_t) (prompt:prompt_length_t)
  (scores:larray f32 vocabulary{is_global_array scores})
  (previous:array i64{active_array (previous_count position prompt) previous})
  (selected:larray i64 1{is_global_array selected})
  (emitted:array i64{active_array (output_count position prompt) emitted})
  (#xs:erased (lseq f32 vocabulary))
  (#ps:erased (lseq i64 (previous_count position prompt)))
  (#logical:erased (Shared.vec Shared.score vocabulary))
  preserves cpu ** on gpu_loc (A.pts_to scores xs) ** history previous ps
  requires pure (represents_scores Optimized xs logical)
  requires exists* (s:lseq i64 1) (e:lseq i64 (output_count position prompt)).
    on gpu_loc (A.pts_to selected s) ** history emitted e
  ensures exists* (s:lseq i64 1) (e:lseq i64 (output_count position prompt)).
    on gpu_loc (A.pts_to selected s) ** history emitted e **
    pure (post Optimized position prompt logical ps s e)
{
  with old_selected old_emitted.
    assert on gpu_loc (A.pts_to selected old_selected) ** history emitted old_emitted;
  rewrite history previous ps as O.history_ownership previous ps;
  rewrite history emitted old_emitted as O.history_ownership emitted old_emitted;
  O.kplay_run scores previous position prompt selected emitted
    #(FStar.Ghost.hide (FStar.Ghost.reveal xs))
    #(FStar.Ghost.hide (FStar.Ghost.reveal ps));
  with next_selected next_emitted.
    assert on gpu_loc (A.pts_to selected next_selected) **
      O.history_ownership emitted next_emitted;
  rewrite O.history_ownership previous ps as history previous ps;
  rewrite O.history_ownership emitted next_emitted as history emitted next_emitted;
  let _ = optimized_post position prompt xs ps logical next_selected next_emitted;
  assert pure (post Optimized position prompt logical ps next_selected next_emitted);
}

#pop-options
