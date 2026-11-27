module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open KPlay.SharedSpec
module S = FStar.Seq
module I64 = FStar.Int64
module SZ = FStar.SizeT

(* Single-thread optimized argmax over 248320 scores: a 512-block x 32-lane
   scan followed by lane and block tree reductions; the selected index is
   written to sel[0] and spliced into a copy of tokens at position d. *)
fn kplay_run
  (scores : array1 f32 (l1_forward 248320) { is_global scores })
  (n : SZ.t)
  (tokens : array1 I64.t (l1_forward (SZ.v n)) { is_global tokens })
  (d : I64.t)
  (sel : array1 I64.t (l1_forward 1) { is_global sel })
  (tout : array1 I64.t (l1_forward (SZ.v n)) { is_global tout })
  (#sp : perm)
  (#ss : chest1 f32 248320)
  (#tp : perm)
  (#ts : chest1 I64.t (SZ.v n))
  (#sel0 : chest1 I64.t 1)
  (#tout0 : chest1 I64.t (SZ.v n))
  preserves
    cpu **
    on gpu_loc (scores |-> Frac sp ss) **
    on gpu_loc (tokens |-> Frac tp ts)
  requires
    on gpu_loc (sel |-> sel0) **
    on gpu_loc (tout |-> tout0)
  ensures
    exists* (sel' : chest1 I64.t 1) (tout' : chest1 I64.t (SZ.v n)).
      on gpu_loc (sel |-> sel') **
      on gpu_loc (tout |-> tout') **
      pure (optimized_argmax_relation (chest1_to_seq ss) (chest1_to_seq ts) (I64.v d)
              (chest1_to_seq sel') (chest1_to_seq tout'))
