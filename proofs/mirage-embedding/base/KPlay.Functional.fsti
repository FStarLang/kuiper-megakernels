module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = Kuiper.SizeT
module I64 = FStar.Int64

inline_for_extraction noextract
let unit_marker = ()

fn kplay_run
  (n : SZ.t)
  (input0 : larray bf16 (SZ.v n){is_global input0})
  (input1 : larray I64.t 7{is_global input1})
  (input2 input3 input4 : I64.t)
  (output0 : larray bf16 7168{is_global output0})
  (output1 : larray bf16 7168{is_global output1})
  (output2 : larray bf16 1792{is_global output2})
  (#f0 #f1 : perm)
  (#s0 : erased (S.seq bf16)) (#s1 : erased (S.seq I64.t))
  (#o0 #o1 #o2 : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (input0 |-> Frac f0 s0) ** on gpu_loc (input1 |-> Frac f1 s1)
  requires on gpu_loc (output0 |-> o0) ** on gpu_loc (output1 |-> o1) ** on gpu_loc (output2 |-> o2)
  ensures exists* r0 r1 r2.
    on gpu_loc (output0 |-> r0) ** on gpu_loc (output1 |-> r1) ** on gpu_loc (output2 |-> r2) **
    pure (base_embedding_relation s0 s1 (I64.v input2) (I64.v input3) (I64.v input4) r0 r1 r2)
