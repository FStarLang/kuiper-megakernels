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

inline_for_extraction noextract
let base_post
  (position:position_t) (prompt:prompt_length_t)
  (w:lseq bf16 weight_length) (h:lseq bf16 width)
  (p:lseq bf16 (previous_rows position prompt * width))
  (s:lseq bf16 vocabulary) (e:lseq bf16 (output_rows position prompt * width))
  : Lemma
      (requires B.semantic_postcondition #unit position prompt w h p () () s e)
      (ensures post Base position prompt w h p s e)
  =
  assert_norm (targets w h ==
    Shared.logical_logits #width #vocabulary Shared.reference_epsilon
      (B.vector_view h) (B.norm_weight_view w) (B.embedding_view w))

inline_for_extraction noextract
let optimized_post
  (position:position_t) (prompt:prompt_length_t)
  (w:lseq bf16 weight_length) (h:lseq bf16 width)
  (p:lseq bf16 (previous_rows position prompt * width))
  (s:lseq f32 vocabulary) (e:lseq bf16 (output_rows position prompt * width))
  : Lemma
      (requires O.semantic_postcondition #unit w h position prompt p () () s e)
      (ensures post Optimized position prompt w h p s e)
  =
  assert_norm (targets w h ==
    Shared.logical_logits #width #vocabulary Shared.reference_epsilon
      (O.vector_view h) (O.norm_weight_view w) (O.embedding_view w))

ghost
fn history_to_optimized
  (rows:nat) (ptr:history_buffer (rows * width))
  (fraction:perm) (values:lseq bf16 (rows * width))
  requires history_owns ptr fraction values
  ensures on gpu_loc (O.history_ownership ptr rows fraction values)
{
  if (rows = 0) {
    rewrite history_owns ptr fraction values as emp;
    on_emp_intro gpu_loc;
    rewrite on gpu_loc emp as on gpu_loc (O.history_ownership ptr rows fraction values);
  } else {
    rewrite history_owns ptr fraction values as on gpu_loc (A.pts_to ptr #fraction values);
    on_pure_intro gpu_loc (is_global_array ptr);
    rewrite on gpu_loc (pure (is_global_array ptr) ** A.pts_to ptr #fraction values)
      as on gpu_loc (O.history_ownership ptr rows fraction values);
  }
}

ghost
fn history_from_optimized
  (rows:nat) (ptr:history_buffer (rows * width))
  (fraction:perm) (values:lseq bf16 (rows * width))
  requires on gpu_loc (O.history_ownership ptr rows fraction values)
  ensures history_owns ptr fraction values
{
  if (rows = 0) {
    rewrite on gpu_loc (O.history_ownership ptr rows fraction values) as on gpu_loc emp;
    on_emp_elim gpu_loc;
    rewrite emp as history_owns ptr fraction values;
  } else {
    rewrite on gpu_loc (O.history_ownership ptr rows fraction values)
      as on gpu_loc (pure (is_global_array ptr) ** A.pts_to ptr #fraction values);
    rewrite on gpu_loc (A.pts_to ptr #fraction values) as history_owns ptr fraction values;
  }
}

fn base
  (position:position_t) (prompt:prompt_length_t)
  (weights:larray bf16 weight_length{is_global_array weights})
  (hidden:larray bf16 width{is_global_array hidden})
  (previous:history_buffer (previous_rows position prompt * width))
  (scores:larray bf16 vocabulary{is_global_array scores})
  (history:history_buffer (output_rows position prompt * width))
  (#w:erased (lseq bf16 weight_length)) (#h:erased (lseq bf16 width))
  (#p:erased (lseq bf16 (previous_rows position prompt * width)))
  (#pw #ph #pp:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w ** A.pts_to hidden #ph h) **
    history_owns previous pp p
  requires exists* (s:lseq bf16 vocabulary)
    (e:lseq bf16 (output_rows position prompt * width)).
    on gpu_loc (A.pts_to scores s) ** history_owns history 1.0R e
  ensures exists* (s:lseq bf16 vocabulary)
    (e:lseq bf16 (output_rows position prompt * width)).
    on gpu_loc (A.pts_to scores s) ** history_owns history 1.0R e **
    pure (post Base position prompt w h p s e)
{
  with old_scores old_history.
    assert on gpu_loc (A.pts_to scores old_scores) ** history_owns history 1.0R old_history;
  rewrite on gpu_loc (A.pts_to weights #pw w) as B.buffer_owns #B.packed_elements weights pw w;
  rewrite on gpu_loc (A.pts_to hidden #ph h) as B.buffer_owns #B.hidden_size hidden ph h;
  rewrite history_owns previous pp p as
    B.buffer_owns #(B.previous_elements position prompt) previous pp p;
  rewrite on gpu_loc (A.pts_to scores old_scores) as
    B.buffer_owns #B.vocab_size scores 1.0R old_scores;
  rewrite history_owns history 1.0R old_history as
    B.buffer_owns #(B.emitted_elements position prompt) history 1.0R old_history;
  B.kplay_run #unit #(fun _ -> emp) #(FStar.Ghost.hide ()) #pw #ph #pp position prompt
    #(FStar.Ghost.hide (FStar.Ghost.reveal w))
    #(FStar.Ghost.hide (FStar.Ghost.reveal h))
    #(FStar.Ghost.hide (FStar.Ghost.reveal p))
    weights hidden previous scores history;
  with next_scores next_history.
    assert B.buffer_owns #B.vocab_size scores 1.0R next_scores **
      B.buffer_owns #(B.emitted_elements position prompt) history 1.0R next_history;
  rewrite B.buffer_owns #B.packed_elements weights pw w as on gpu_loc (A.pts_to weights #pw w);
  rewrite B.buffer_owns #B.hidden_size hidden ph h as on gpu_loc (A.pts_to hidden #ph h);
  rewrite B.buffer_owns #(B.previous_elements position prompt) previous pp p as
    history_owns previous pp p;
  rewrite B.buffer_owns #B.vocab_size scores 1.0R next_scores as
    on gpu_loc (A.pts_to scores next_scores);
  rewrite B.buffer_owns #(B.emitted_elements position prompt) history 1.0R next_history as
    history_owns history 1.0R next_history;
  let _ = base_post position prompt w h p next_scores next_history;
  assert pure (post Base position prompt w h p next_scores next_history);
}

fn optimized
  (position:position_t) (prompt:prompt_length_t)
  (weights:larray bf16 weight_length{is_global_array weights})
  (hidden:larray bf16 width{is_global_array hidden})
  (previous:history_buffer (previous_rows position prompt * width))
  (scores:larray f32 vocabulary{is_global_array scores})
  (history:history_buffer (output_rows position prompt * width))
  (#w:erased (lseq bf16 weight_length)) (#h:erased (lseq bf16 width))
  (#p:erased (lseq bf16 (previous_rows position prompt * width)))
  (#pw #ph #pp:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w ** A.pts_to hidden #ph h) **
    history_owns previous pp p
  requires exists* (s:lseq f32 vocabulary)
    (e:lseq bf16 (output_rows position prompt * width)).
    on gpu_loc (A.pts_to scores s) ** history_owns history 1.0R e
  ensures exists* (s:lseq f32 vocabulary)
    (e:lseq bf16 (output_rows position prompt * width)).
    on gpu_loc (A.pts_to scores s) ** history_owns history 1.0R e **
    pure (post Optimized position prompt w h p s e)
{
  with old_scores old_history.
    assert on gpu_loc (A.pts_to scores old_scores) ** history_owns history 1.0R old_history;
  history_to_optimized (previous_rows position prompt) previous pp p;
  history_to_optimized (output_rows position prompt) history 1.0R old_history;
  O.kplay_run #unit #(fun _ -> emp) #(FStar.Ghost.hide ())
    weights hidden previous position prompt scores history
    #(FStar.Ghost.hide (FStar.Ghost.reveal w))
    #(FStar.Ghost.hide (FStar.Ghost.reveal h))
    #(FStar.Ghost.hide (FStar.Ghost.reveal p))
    #(FStar.Ghost.hide (FStar.Ghost.reveal old_scores))
    #(FStar.Ghost.hide (FStar.Ghost.reveal old_history))
    #pw #ph #pp;
  with next_scores next_history.
    assert on gpu_loc (A.pts_to scores next_scores **
      O.history_ownership history (O.result_rows (I64.v position) (I64.v prompt)) 1.0R next_history);
  history_from_optimized (previous_rows position prompt) previous pp p;
  history_from_optimized (output_rows position prompt) history 1.0R next_history;
  let _ = optimized_post position prompt w h p next_scores next_history;
  assert pure (post Optimized position prompt w h p next_scores next_history);
}

fn switcher
  (i:implementation) (position:position_t) (prompt:prompt_length_t)
  (weights:larray bf16 weight_length{is_global_array weights})
  (hidden:larray bf16 width{is_global_array hidden})
  (previous:history_buffer (previous_rows position prompt * width))
  (scores:larray (score_type i) vocabulary{is_global_array scores})
  (history:history_buffer (output_rows position prompt * width))
  (#w:erased (lseq bf16 weight_length)) (#h:erased (lseq bf16 width))
  (#p:erased (lseq bf16 (previous_rows position prompt * width)))
  (#pw #ph #pp:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w ** A.pts_to hidden #ph h) **
    history_owns previous pp p
  requires exists* (s:lseq (score_type i) vocabulary)
    (e:lseq bf16 (output_rows position prompt * width)).
    on gpu_loc (A.pts_to scores s) ** history_owns history 1.0R e
  ensures exists* (s:lseq (score_type i) vocabulary)
    (e:lseq bf16 (output_rows position prompt * width)).
    on gpu_loc (A.pts_to scores s) ** history_owns history 1.0R e **
    pure (post i position prompt w h p s e)
{
  match i {
    Base -> { base position prompt weights hidden previous scores history #w #h #p #pw #ph #pp }
    Optimized -> { optimized position prompt weights hidden previous scores history #w #h #p #pw #ph #pp }
  }
}

#pop-options
