module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module I64 = FStar.Int64

(* ABI order (target.json): input0 = scores [256, 8] float32 (flat 2048),
   input1 = token position (int64), output0 = normalized row [8] float32.
   Only ownership and buffer lengths are required. The token range and all
   numerical premises live inside optimized_softmax_relation; an out-of-range
   token is rejected at run time without launching the kernel. *)
fn kplay_run
  (scores : larray f32 2048 {is_global scores})
  (token : i64)
  (out : larray f32 8 {is_global out})
  (#s : erased (lseq f32 2048))
  preserves cpu ** on gpu_loc (scores |-> s)
  requires exists* before. on gpu_loc (out |-> before)
  ensures exists* after.
    on gpu_loc (out |-> after) **
    pure (optimized_softmax_relation s (I64.v token) after)
