module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module SZ = FStar.SizeT
module I64 = FStar.Int64

(* Physical ABI (target.json order): input0 = int64 prompt[n0], input1 = BF16 scores[n1],
   input2 = int64 decode_steps; output0 = int64 tokens[1, n0 + 1] (flat), output1 = int64
   rank-zero selected index (one element). n0/n1 are the element counts of input0/input1.
   Only ownership and lengths are required; the domain premises of the reference (nonempty
   scores, int32-sized lengths, decode_steps = 1) live inside base_argmax_relation. *)
fn kplay_run
  (n0 : sz)
  (n1 : sz)
  (input0 : larray i64 (SZ.v n0) { is_global input0 })
  (input1 : larray bf16 (SZ.v n1) { is_global input1 })
  (input2 : i64)
  (output0 : larray i64 (SZ.v n0 + 1) { is_global output0 })
  (output1 : larray i64 1 { is_global output1 })
  (#p0 #p1 : perm)
  (#s0 : erased (seq i64))
  (#s1 : erased (seq bf16))
  preserves
    cpu **
    on gpu_loc (input0 |-> Frac p0 s0) **
    on gpu_loc (input1 |-> Frac p1 s1)
  requires
    exists* (v0 v1 : seq i64).
      on gpu_loc (output0 |-> v0) ** on gpu_loc (output1 |-> v1)
  ensures
    exists* (r0 r1 : seq i64).
      on gpu_loc (output0 |-> r0) ** on gpu_loc (output1 |-> r1) **
      pure (base_argmax_relation s0 s1 (I64.v input2) r0 r1)
