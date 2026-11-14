module KPlay.Review
#lang-pulse

open Kuiper
open Pulse.Lib.Send
module S = FStar.Seq
module I64 = FStar.Int64
module U64 = FStar.UInt64
module A = Pulse.Lib.Array
module Shared = KPlay.SharedSpec
module B = KPlay.Base.Functional
module O = KPlay.Optimized.Functional

#push-options "--z3rlimit 10 --fuel 4 --ifuel 2"

inline_for_extraction noextract
let base_post_refines (w:lseq bf16 weight_length) (t:Shared.token_id)
  (h:Shared.hidden_vector bf16)
  : Lemma (requires B.flat_lookup_post w t h) (ensures post w t h) =
  B.flat_lookup_refines_shared w t h;
  assert_norm (logical_weights w == B.logical_weights w)

inline_for_extraction noextract
let optimized_post_refines (w:lseq bf16 weight_length) (t:Shared.token_id)
  (h:Shared.hidden_vector bf16)
  : Lemma (requires O.flat_lookup_post w t h) (ensures post w t h) =
  O.flat_lookup_refines_shared w t h;
  assert_norm (logical_weights w == O.logical_weights w)

inline_for_extraction noextract
let singleton (values:lseq i64 1)
  : Lemma (values == S.create 1 (S.index values 0)) =
  S.lemma_eq_intro values (S.create 1 (S.index values 0));
  S.lemma_eq_elim values (S.create 1 (S.index values 0))

inline_for_extraction noextract
let initial_frame (position:position_t) (ptr:array i64) (values:lseq i64 1) : slprop =
  if I64.v position = 0 then on gpu_loc (A.pts_to ptr #0.5R values) else emp

ghost
fn prepare_previous (position:position_t) (ptr:array i64) (values:lseq i64 1)
  requires on gpu_loc (A.pts_to ptr #0.5R values)
  ensures on gpu_loc (B.previous_permission position ptr (S.index values 0)) **
    initial_frame position ptr values
{
  let _ = singleton values;
  if (I64.v position = 0) {
    rewrite on gpu_loc (A.pts_to ptr #0.5R values) as initial_frame position ptr values;
    on_emp_intro gpu_loc;
    rewrite on gpu_loc emp as on gpu_loc (B.previous_permission position ptr (S.index values 0));
  } else {
    rewrite on gpu_loc (A.pts_to ptr #0.5R values)
      as on gpu_loc (B.previous_permission position ptr (S.index values 0));
    rewrite emp as initial_frame position ptr values;
  }
}

ghost
fn restore_previous (position:position_t) (ptr:array i64) (values:lseq i64 1)
  requires on gpu_loc (B.previous_permission position ptr (S.index values 0)) **
    initial_frame position ptr values
  ensures on gpu_loc (A.pts_to ptr #0.5R values)
{
  let _ = singleton values;
  if (I64.v position = 0) {
    rewrite on gpu_loc (B.previous_permission position ptr (S.index values 0)) as on gpu_loc emp;
    on_emp_elim gpu_loc;
    rewrite initial_frame position ptr values as on gpu_loc (A.pts_to ptr #0.5R values);
  } else {
    rewrite initial_frame position ptr values as emp;
    rewrite on gpu_loc (B.previous_permission position ptr (S.index values 0))
      as on gpu_loc (A.pts_to ptr #0.5R values);
  }
}

fn base
  (n:length_t) (position:position_t)
  (weights:larray bf16 weight_length{is_global_array weights})
  (prompt:larray i64 (I64.v n){is_global_array prompt})
  (previous:larray i64 1{is_global_array previous})
  (hidden:larray bf16 Shared.hidden_size{is_global_array hidden})
  (extra:auxiliary Base position)
  (#w:erased (lseq bf16 weight_length)) (#q:erased (lseq i64 (I64.v n)))
  (#p:erased (lseq i64 1)) (#t:erased Shared.token_id)
  preserves cpu ** on gpu_loc (
    A.pts_to weights #0.5R w ** A.pts_to prompt #0.5R q ** A.pts_to previous #0.5R p)
  requires pure (inputs n position q p t) ** auxiliary_before Base position extra
  requires exists* (h:Shared.hidden_vector bf16). on gpu_loc (A.pts_to hidden h)
  ensures exists* (h:Shared.hidden_vector bf16).
    on gpu_loc (A.pts_to hidden h) ** auxiliary_after Base position extra ** pure (post w t h)
{
  unfold auxiliary_before Base position extra;
  prepare_previous position previous p;
  let _ = B.canonical_model_is_total w;
  let length = U64.uint_to_t (I64.v n);
  B.kplay_run length position
    #(FStar.Ghost.hide (FStar.Ghost.reveal w))
    #(FStar.Ghost.hide (FStar.Ghost.reveal q))
    #(FStar.Ghost.hide (S.index (FStar.Ghost.reveal p) 0))
    #t #(FStar.Ghost.hide (B.canonical_weights_model (FStar.Ghost.reveal w)))
    weights prompt previous hidden extra.rotary extra.state extra.convolution;
  with next_hidden r s c.
    assert on gpu_loc (A.pts_to hidden next_hidden ** A.pts_to extra.rotary r **
      A.pts_to extra.state s ** A.pts_to extra.convolution c);
  restore_previous position previous p;
  let _ = base_post_refines w t next_hidden;
  assert pure (post w t next_hidden);
  fold auxiliary_after Base position extra;
}

fn optimized
  (n:length_t) (position:position_t)
  (weights:larray bf16 weight_length{is_global_array weights})
  (prompt:larray i64 (I64.v n){is_global_array prompt})
  (previous:larray i64 1{is_global_array previous})
  (hidden:larray bf16 Shared.hidden_size{is_global_array hidden})
  (extra:auxiliary Optimized position)
  (#w:erased (lseq bf16 weight_length)) (#q:erased (lseq i64 (I64.v n)))
  (#p:erased (lseq i64 1)) (#t:erased Shared.token_id)
  preserves cpu ** on gpu_loc (
    A.pts_to weights #0.5R w ** A.pts_to prompt #0.5R q ** A.pts_to previous #0.5R p)
  requires pure (inputs n position q p t) ** auxiliary_before Optimized position extra
  requires exists* (h:Shared.hidden_vector bf16). on gpu_loc (A.pts_to hidden h)
  ensures exists* (h:Shared.hidden_vector bf16).
    on gpu_loc (A.pts_to hidden h) ** auxiliary_after Optimized position extra ** pure (post w t h)
{
  unfold auxiliary_before Optimized position extra;
  with old_hidden. assert on gpu_loc (A.pts_to hidden old_hidden);
  let _ = O.canonical_model_is_total w;
  O.kplay_run n position weights prompt previous hidden
    #(FStar.Ghost.hide (FStar.Ghost.reveal w))
    #(FStar.Ghost.hide (FStar.Ghost.reveal q))
    #(FStar.Ghost.hide (FStar.Ghost.reveal p))
    #(FStar.Ghost.hide (FStar.Ghost.reveal old_hidden))
    #t #(FStar.Ghost.hide (O.canonical_weights_model (FStar.Ghost.reveal w)))
    #0.5R #0.5R #0.5R;
  with next_hidden. assert on gpu_loc (A.pts_to hidden next_hidden);
  let _ = optimized_post_refines w t next_hidden;
  assert pure (post w t next_hidden);
  fold auxiliary_after Optimized position extra;
}

fn switcher
  (i:implementation) (n:length_t) (position:position_t)
  (weights:larray bf16 weight_length{is_global_array weights})
  (prompt:larray i64 (I64.v n){is_global_array prompt})
  (previous:larray i64 1{is_global_array previous})
  (hidden:larray bf16 Shared.hidden_size{is_global_array hidden})
  (extra:auxiliary i position)
  (#w:erased (lseq bf16 weight_length)) (#q:erased (lseq i64 (I64.v n)))
  (#p:erased (lseq i64 1)) (#t:erased Shared.token_id)
  preserves cpu ** on gpu_loc (
    A.pts_to weights #0.5R w ** A.pts_to prompt #0.5R q ** A.pts_to previous #0.5R p)
  requires pure (inputs n position q p t) ** auxiliary_before i position extra
  requires exists* (h:Shared.hidden_vector bf16). on gpu_loc (A.pts_to hidden h)
  ensures exists* (h:Shared.hidden_vector bf16).
    on gpu_loc (A.pts_to hidden h) ** auxiliary_after i position extra ** pure (post w t h)
{
  match i {
    Base -> { base n position weights prompt previous hidden extra #w #q #p #t }
    Optimized -> { optimized n position weights prompt previous hidden extra #w #q #p #t }
  }
}

#pop-options
