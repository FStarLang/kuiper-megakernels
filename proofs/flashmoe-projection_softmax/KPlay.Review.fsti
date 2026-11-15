module KPlay.Review
#lang-pulse

open Kuiper
module Seq = FStar.Seq
module I64 = FStar.Int64
module A = Pulse.Lib.Array
module C = Kuiper.Chest
module G = Kuiper.Spec.GEMM
module SM = Kuiper.Spec.Softmax
module Shared = KPlay.SharedSpec
module B = KPlay.Base.Functional

#push-options "--z3rlimit 10 --fuel 4 --ifuel 2"

(* Base computes one row and initializes/copies the routing frame; optimized
   computes the entire profile matrix. Both approximate the same row-wise
   projection/softmax target. The batch lemma connects repeated base calls
   to the complete shared task; the dispatcher does not execute that loop. *)
noextract type implementation = | Base | Optimized
inline_for_extraction noextract
let rows (i:implementation) (smoke:bool) : pos =
  match i with | Base -> 1 | Optimized -> Shared.token_count smoke
inline_for_extraction noextract
let row_major (#a:Type0) (m n:nat) (x:lseq a (m * n)) : GTot (C.chest2 a m n) =
  C.mk2 (fun t k -> Seq.index x (t * n + k))
inline_for_extraction noextract
let model_projection (#m #h:nat) (#e:pos)
  (x:C.chest2 real m h) (w:C.chest2 real e h) : C.chest2 real m e =
  G.matmul x (C.mk2 (fun k j -> C.acc2 w j k))
inline_for_extraction noextract
let model_target (#m #h:nat) (#e:pos)
  (x:C.chest2 real m h) (w:C.chest2 real e h) : C.chest2 real m e =
  let scores = model_projection x w in
  C.mk2 (fun t j -> C.acc1 (SM.softmax_real (C.chest2_row scores t)) j)
inline_for_extraction noextract
let target (m h:nat) (e:pos)
  (x:lseq bf16 (m * h)) (w:lseq bf16 (e * h)) : C.chest2 real m e =
  model_target (C.to_real_chest (row_major m h x)) (C.to_real_chest (row_major e h w))
inline_for_extraction noextract
let domain (m h:nat) (e:pos)
  (x:lseq bf16 (m * h)) (w:lseq bf16 (e * h)) : prop =
  (forall (n:nat{n < m * h}). Shared.finite_bounded_bf16 32L (Seq.index x n)) /\
  (forall (n:nat{n < e * h}). Shared.finite_bounded_bf16 4L (Seq.index w n))
inline_for_extraction noextract
let task (m h:nat) (e:pos)
  (x:lseq bf16 (m * h)) (w:lseq bf16 (e * h)) (p:lseq f32 (m * e)) : prop =
  row_major m e p %~ target m h e x w
inline_for_extraction noextract
let post (i:implementation) (smoke:bool)
  (x:lseq bf16 (rows i smoke * Shared.hidden_count smoke))
  (w:lseq bf16 (Shared.expert_count smoke * Shared.hidden_count smoke))
  (p:lseq f32 (rows i smoke * Shared.expert_count smoke)) : prop =
  task (rows i smoke) (Shared.hidden_count smoke) (Shared.expert_count smoke) x w p /\
  (i == Optimized ==>
    (forall (rx:C.chest2 real (rows i smoke) (Shared.hidden_count smoke))
      (rw:C.chest2 real (Shared.expert_count smoke) (Shared.hidden_count smoke)).
      row_major (rows i smoke) (Shared.hidden_count smoke) x %~ rx /\
      row_major (Shared.expert_count smoke) (Shared.hidden_count smoke) w %~ rw ==>
      row_major (rows i smoke) (Shared.expert_count smoke) p %~ model_target rx rw))

inline_for_extraction noextract unfold
let device_array (a:Type0) (n:nat) = p:larray a n{is_global_array p}
inline_for_extraction noextract unfold
let inactive_array (a:Type0) (n:nat) =
  p:array a{n > 0 ==> A.length p == n /\ is_global_array p}
noextract noeq type base_buffers (smoke:bool) = {
  token: (t:i64{0 <= I64.v t /\ I64.v t < Shared.token_count smoke});
  frame: erased (B.input_frame (Shared.expert_count smoke) (I64.v token));
  input_counts: inactive_array i64 (B.active_size (I64.v token) (Shared.expert_count smoke));
  input_indices: inactive_array i64 (B.active_size (I64.v token) (Shared.expert_count smoke * 32));
  input_probabilities: inactive_array f32 (B.active_size (I64.v token) (Shared.expert_count smoke * 32));
  input_survival: inactive_array i32 (I64.v token * Shared.expert_count smoke);
  output_counts: device_array i64 (Shared.expert_count smoke);
  output_indices: device_array i64 (Shared.expert_count smoke * 32);
  output_probabilities: device_array f32 (Shared.expert_count smoke * 32);
  output_survival: device_array i32 ((I64.v token + 1) * Shared.expert_count smoke)
}
inline_for_extraction noextract unfold
let auxiliary (i:implementation) (smoke:bool) : Type0 =
  match i with | Base -> base_buffers smoke | Optimized -> unit
inline_for_extraction noextract
let frame_inputs (#smoke:bool) (b:base_buffers smoke) : slprop =
  if I64.v b.token = 0 then emp
  else on gpu_loc (
    A.pts_to b.input_counts b.frame.B.previous_counts **
    A.pts_to b.input_indices b.frame.B.previous_indices **
    A.pts_to b.input_probabilities b.frame.B.previous_probabilities **
    A.pts_to b.input_survival b.frame.B.previous_survival)
inline_for_extraction noextract
let before (i:implementation) (smoke:bool) : (auxiliary i smoke -> slprop) =
  match i with
  | Base -> fun (b:base_buffers smoke) ->
    frame_inputs b ** (exists* counts indices probabilities survival.
      on gpu_loc (A.pts_to b.output_counts counts ** A.pts_to b.output_indices indices **
        A.pts_to b.output_probabilities probabilities ** A.pts_to b.output_survival survival))
  | Optimized -> fun (_:unit) -> emp
inline_for_extraction noextract
let after (i:implementation) (smoke:bool) : (auxiliary i smoke -> slprop) =
  match i with
  | Base -> fun (b:base_buffers smoke) ->
    frame_inputs b ** on gpu_loc (
      A.pts_to b.output_counts
        (if I64.v b.token = 0 then Seq.create (Shared.expert_count smoke) 0L else b.frame.B.previous_counts) **
      A.pts_to b.output_indices
        (if I64.v b.token = 0 then Seq.create (Shared.expert_count smoke * 32) (-1L) else b.frame.B.previous_indices) **
      A.pts_to b.output_probabilities
        (if I64.v b.token = 0 then Seq.create (Shared.expert_count smoke * 32) (Kuiper.Floating.zero #f32) else b.frame.B.previous_probabilities) **
      A.pts_to b.output_survival
        (Seq.append b.frame.B.previous_survival (Seq.create (Shared.expert_count smoke) 0l)))
  | Optimized -> fun (_:unit) -> emp

val base_batch_refines :
  smoke:bool -> x:Shared.token_matrix smoke -> w:Shared.weight_matrix smoke ->
  p:(t:natlt (Shared.token_count smoke) -> lseq f32 (Shared.expert_count smoke)) ->
  Lemma
    (requires Shared.input_domain smoke x w /\
      (forall (t:natlt (Shared.token_count smoke)).
        task 1 (Shared.hidden_count smoke) (Shared.expert_count smoke)
          (B.pack_token smoke x t) (B.pack_weights smoke w) (p t)))
    (ensures
      Shared.postcondition smoke x w x w (Shared.projection_softmax smoke x w) /\
      C.mk2 (fun (t:natlt (Shared.token_count smoke)) (j:natlt (Shared.expert_count smoke)) ->
        Seq.index (p t) j) %~ Shared.projection_softmax smoke x w)

fn base
  (smoke:bool)
  (tokens:device_array bf16 (rows Base smoke * Shared.hidden_count smoke))
  (weights:device_array bf16 (Shared.expert_count smoke * Shared.hidden_count smoke))
  (probabilities:device_array f32 (rows Base smoke * Shared.expert_count smoke))
  (aux:auxiliary Base smoke)
  (#x:erased (lseq bf16 (rows Base smoke * Shared.hidden_count smoke)))
  (#w:erased (lseq bf16 (Shared.expert_count smoke * Shared.hidden_count smoke)))
  preserves cpu ** on gpu_loc (A.pts_to tokens x ** A.pts_to weights w)
  requires pure (domain (rows Base smoke) (Shared.hidden_count smoke) (Shared.expert_count smoke) x w) **
    before Base smoke aux **
    (exists* (old:lseq f32 (rows Base smoke * Shared.expert_count smoke)).
      on gpu_loc (A.pts_to probabilities old))
  ensures exists* (p:lseq f32 (rows Base smoke * Shared.expert_count smoke)).
    on gpu_loc (A.pts_to probabilities p) ** after Base smoke aux ** pure (post Base smoke x w p)

fn optimized
  (smoke:bool)
  (tokens:device_array bf16 (rows Optimized smoke * Shared.hidden_count smoke))
  (weights:device_array bf16 (Shared.expert_count smoke * Shared.hidden_count smoke))
  (probabilities:device_array f32 (rows Optimized smoke * Shared.expert_count smoke))
  (aux:auxiliary Optimized smoke)
  (#x:erased (lseq bf16 (rows Optimized smoke * Shared.hidden_count smoke)))
  (#w:erased (lseq bf16 (Shared.expert_count smoke * Shared.hidden_count smoke)))
  preserves cpu ** on gpu_loc (A.pts_to tokens x ** A.pts_to weights w)
  requires pure (domain (rows Optimized smoke) (Shared.hidden_count smoke) (Shared.expert_count smoke) x w) **
    before Optimized smoke aux **
    (exists* (old:lseq f32 (rows Optimized smoke * Shared.expert_count smoke)).
      on gpu_loc (A.pts_to probabilities old))
  ensures exists* (p:lseq f32 (rows Optimized smoke * Shared.expert_count smoke)).
    on gpu_loc (A.pts_to probabilities p) ** after Optimized smoke aux ** pure (post Optimized smoke x w p)

fn switcher
  (i:implementation) (smoke:bool)
  (tokens:device_array bf16 (rows i smoke * Shared.hidden_count smoke))
  (weights:device_array bf16 (Shared.expert_count smoke * Shared.hidden_count smoke))
  (probabilities:device_array f32 (rows i smoke * Shared.expert_count smoke))
  (aux:auxiliary i smoke)
  (#x:erased (lseq bf16 (rows i smoke * Shared.hidden_count smoke)))
  (#w:erased (lseq bf16 (Shared.expert_count smoke * Shared.hidden_count smoke)))
  preserves cpu ** on gpu_loc (A.pts_to tokens x ** A.pts_to weights w)
  requires pure (domain (rows i smoke) (Shared.hidden_count smoke) (Shared.expert_count smoke) x w) **
    before i smoke aux **
    (exists* (old:lseq f32 (rows i smoke * Shared.expert_count smoke)).
      on gpu_loc (A.pts_to probabilities old))
  ensures exists* (p:lseq f32 (rows i smoke * Shared.expert_count smoke)).
    on gpu_loc (A.pts_to probabilities p) ** after i smoke aux ** pure (post i smoke x w p)

#pop-options
