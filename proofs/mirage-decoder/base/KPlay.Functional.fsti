module KPlay.Functional

#lang-pulse

open FStar.Real
open Kuiper
open KPlay.SharedSpec
module KC = Kuiper.Chest
module KT = Kuiper.Tensor
module LA = Kuiper.Tensor.Layout.Alg
module SZ = Kuiper.SizeT
module I64 = FStar.Int64
module S = FStar.Seq

inline_for_extraction noextract
let unit_marker = ()

fn kplay_run
  (n0 : SZ.t)
  (input0 : KT.array1 bf16 (LA.l1_forward n0) { KT.is_global input0 })
  (input1 : KT.array1 bf16 (LA.l1_forward 7168) { KT.is_global input1 })
  (input2 : KT.array1 bf16 (LA.l1_forward 7168) { KT.is_global input2 })
  (input3 : KT.array1 bf16 (LA.l1_forward 1792) { KT.is_global input3 })
  (input4 : I64.t)
  (input5 : I64.t)
  (output0 : KT.array1 bf16 (LA.l1_forward 7168) { KT.is_global output0 })
  (output1 : KT.array1 bf16 (LA.l1_forward 7168) { KT.is_global output1 })
  (#p0 #p1 #p2 #p3 : perm)
  (#s0 : KC.chest1 bf16 n0)
  (#s1 #s2 : KC.chest1 bf16 7168)
  (#s3 : KC.chest1 bf16 1792)
  (#o0 #o1 : KC.chest1 bf16 7168)
  preserves
    cpu **
    on gpu_loc KT.(input0 |-> Frac p0 s0) **
    on gpu_loc KT.(input1 |-> Frac p1 s1) **
    on gpu_loc KT.(input2 |-> Frac p2 s2) **
    on gpu_loc KT.(input3 |-> Frac p3 s3)
  requires
    on gpu_loc KT.(output0 |-> o0) **
    on gpu_loc KT.(output1 |-> o1)
  ensures
    exists* (o0' o1' : KC.chest1 bf16 7168).
      on gpu_loc KT.(output0 |-> o0') **
      on gpu_loc KT.(output1 |-> o1') **
      pure (base_decoder_relation (KC.chest1_to_seq s0) (KC.chest1_to_seq s1) (KC.chest1_to_seq s2) (KC.chest1_to_seq s3)
              (I64.v input4) (I64.v input5) (KC.chest1_to_seq o0') (KC.chest1_to_seq o1'))
