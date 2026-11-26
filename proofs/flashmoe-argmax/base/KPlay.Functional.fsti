module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module I32 = FStar.Int32

(* Base router argmax over one contiguous float32[8] softmax row (input0).
   Output0: rank-zero int32 selected expert; output1: rank-zero float32
   selected probability. Contents satisfy base_argmax_relation: exactly the
   serial scan from (0, row[0]) over experts 1..7 with strict > updates. *)
fn kplay_run
  (input0 : larray f32 8 { is_global input0 })
  (output0 : larray I32.t 1 { is_global output0 })
  (output1 : larray f32 1 { is_global output1 })
  (#row : erased (S.seq f32))
  preserves cpu ** on gpu_loc (input0 |-> row)
  requires exists* s0. on gpu_loc (output0 |-> s0)
  requires exists* p0. on gpu_loc (output1 |-> p0)
  ensures exists* s p.
    on gpu_loc (output0 |-> s) **
    on gpu_loc (output1 |-> p) **
    pure (base_argmax_relation row s p)
