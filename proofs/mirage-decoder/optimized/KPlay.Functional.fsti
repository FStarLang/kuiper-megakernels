module KPlay.Functional
#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open KPlay.SharedSpec

inline_for_extraction let () = ()

fn kplay_run
  (input0 : array1 bf16 (l1_forward 4194304) { is_global input0 })
  (input1 : array1 bf16 (l1_forward 256) { is_global input1 })
  (input2 : array1 bf16 (l1_forward 2097152) { is_global input2 })
  (input3 : array1 bf16 (l1_forward 1024) { is_global input3 })
  (input4 : array1 bf16 (l1_forward 6291456) { is_global input4 })
  (input5 : array1 bf16 (l1_forward 3145728) { is_global input5 })
  (input6 : array1 bf16 (l1_forward 1024) { is_global input6 })
  (input7 : array1 bf16 (l1_forward 8192) { is_global input7 })
  (input8 : array1 bf16 (l1_forward 8192) { is_global input8 })
  (input9 : array1 bf16 (l1_forward 524288) { is_global input9 })
  (input10 : array1 bf16 (l1_forward 524288) { is_global input10 })
  (input11 : FStar.Int64.t)
  (output0 : array1 bf16 (l1_forward 8192) { is_global output0 })
  (output1 : array1 bf16 (l1_forward 8192) { is_global output1 })
  (#f0 #f1 #f2 #f3 #f4 #f5 #f6 #f7 #f8 #f9 #f10 : perm)
  (#s0 : chest1 bf16 4194304) (#s1 : chest1 bf16 256) (#s2 : chest1 bf16 2097152)
  (#s3 : chest1 bf16 1024) (#s4 : chest1 bf16 6291456) (#s5 : chest1 bf16 3145728)
  (#s6 : chest1 bf16 1024) (#s7 : chest1 bf16 8192) (#s8 : chest1 bf16 8192)
  (#s9 : chest1 bf16 524288) (#s10 : chest1 bf16 524288)
  (#o0 #o1 : chest1 bf16 8192)
  preserves cpu **
    on gpu_loc (input0 |-> Frac f0 s0) ** on gpu_loc (input1 |-> Frac f1 s1) **
    on gpu_loc (input2 |-> Frac f2 s2) ** on gpu_loc (input3 |-> Frac f3 s3) **
    on gpu_loc (input4 |-> Frac f4 s4) ** on gpu_loc (input5 |-> Frac f5 s5) **
    on gpu_loc (input6 |-> Frac f6 s6) ** on gpu_loc (input7 |-> Frac f7 s7) **
    on gpu_loc (input8 |-> Frac f8 s8) ** on gpu_loc (input9 |-> Frac f9 s9) **
    on gpu_loc (input10 |-> Frac f10 s10)
  requires on gpu_loc (output0 |-> o0) ** on gpu_loc (output1 |-> o1)
  ensures exists* o0' o1'. on gpu_loc (output0 |-> o0') ** on gpu_loc (output1 |-> o1') **
    pure (optimized_decoder_relation (chest1_to_seq s0) (chest1_to_seq s1) (chest1_to_seq s2)
            (chest1_to_seq s3) (chest1_to_seq s4) (chest1_to_seq s5) (chest1_to_seq s6)
            (chest1_to_seq s7) (chest1_to_seq s8) (chest1_to_seq s9) (chest1_to_seq s10)
            (FStar.Int64.v input11) (chest1_to_seq o0') (chest1_to_seq o1'))
