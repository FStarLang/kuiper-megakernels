module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module A = Pulse.Lib.Array
module F32 = Kuiper.Float32
module I32 = FStar.Int32
module I64 = FStar.Int64

inline_for_extraction noextract
let interface_marker : unit = ()

(* One serial base capacity-routing update (router_capacity_step).
   ABI order: input0 selected expert int32[] (1 element), input1 selected
   probability float32[] (1 element), input2 prefix counts int64[8], input3
   token table int64[8,32], input4 combine table float32[8,32], input5 live
   survival prefix int32[input6,8], input6 token position; then the four
   fresh outputs: output0 updated prefix int64[8], output1 token table
   int64[8,32], output2 combine table float32[8,32], output3 survival prefix
   int32[input6+1,8].
   Premises are ownership and buffer lengths only; every value premise
   (selected expert range, token range, reachable prefix counts) is an
   antecedent of base_capacity_routing_relation. *)
fn kplay_run
  (input0 : A.larray I32.t 1 { is_global input0 })
  (input1 : A.larray F32.t 1 { is_global input1 })
  (input2 : A.larray I64.t 8 { is_global input2 })
  (input3 : A.larray I64.t 256 { is_global input3 })
  (input4 : A.larray F32.t 256 { is_global input4 })
  (input5 : A.array I32.t { is_global input5 })
  (input6 : I64.t)
  (output0 : A.larray I64.t 8 { is_global output0 })
  (output1 : A.larray I64.t 256 { is_global output1 })
  (output2 : A.larray F32.t 256 { is_global output2 })
  (output3 : A.array I32.t { is_global output3 })
  (#p0 #p1 #p2 #p3 #p4 #p5 : perm)
  (#s0 : erased (S.seq I32.t))
  (#s1 : erased (S.seq F32.t))
  (#s2 : erased (S.seq I64.t))
  (#s3 : erased (S.seq I64.t))
  (#s4 : erased (S.seq F32.t))
  (#s5 : erased (S.seq I32.t))
  preserves
    cpu **
    on gpu_loc (input0 |-> Frac p0 s0) **
    on gpu_loc (input1 |-> Frac p1 s1) **
    on gpu_loc (input2 |-> Frac p2 s2) **
    on gpu_loc (input3 |-> Frac p3 s3) **
    on gpu_loc (input4 |-> Frac p4 s4) **
    on gpu_loc (input5 |-> Frac p5 s5)
  requires
    (exists* (d0 : S.seq I64.t). on gpu_loc (output0 |-> d0)) **
    (exists* (d1 : S.seq I64.t). on gpu_loc (output1 |-> d1)) **
    (exists* (d2 : S.seq F32.t). on gpu_loc (output2 |-> d2)) **
    (exists* (d3 : S.seq I32.t). on gpu_loc (output3 |-> d3)) **
    pure (A.length input5 == 8 * I64.v input6 /\
          A.length output3 == 8 * I64.v input6 + 8)
  ensures
    exists* (o0 : S.seq I64.t) (o1 : S.seq I64.t) (o2 : S.seq F32.t) (o3 : S.seq I32.t).
      on gpu_loc (output0 |-> o0) **
      on gpu_loc (output1 |-> o1) **
      on gpu_loc (output2 |-> o2) **
      on gpu_loc (output3 |-> o3) **
      pure (base_capacity_routing_relation s0 s1 s2 s3 s4 s5 (I64.v input6) o0 o1 o2 o3)
