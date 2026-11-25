module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open KPlay.SharedSpec
module SZ = Kuiper.SizeT

(* ABI layouts: input0 = weights[151936,1024] row-major; input1 = state[8,1024]
   row-major, viewed as its transpose (1024 x 8, column-major); output0 =
   logits[8,153600] row-major; output1 = hidden[7,1024] row-major, viewed as
   its transpose (1024 x 7, column-major). *)
let lw = l2_row_major 151936 1024
let lx = l2_col_major 1024 8

inline_for_extraction noextract
let sz153600 : (x:SZ.t{SZ.v x == 153600}) = FStar.SizeT.uint32_to_sizet 153600ul

let lo0 = l2_row_major 8 (SZ.v sz153600)
let lo1 = l2_col_major 1024 7

fn kplay_run
  (input0 : array2 bf16 lw {is_global input0})
  (input1 : array2 bf16 lx {is_global input1})
  (output0 : array2 bf16 lo0 {is_global output0})
  (output1 : array2 bf16 lo1 {is_global output1})
  (#p0 #p1 : perm)
  (#sW : chest2 bf16 151936 1024) (#sX : chest2 bf16 1024 8)
  (#c0 : chest2 bf16 8 153600) (#c1 : chest2 bf16 1024 7)
  preserves cpu ** on gpu_loc (input0 |-> Frac p0 sW) ** on gpu_loc (input1 |-> Frac p1 sX)
  requires on gpu_loc (output0 |-> c0) ** on gpu_loc (output1 |-> c1)
  ensures exists* (c0' : chest2 bf16 8 153600) (c1' : chest2 bf16 1024 7).
    on gpu_loc (output0 |-> c0') ** on gpu_loc (output1 |-> c1') **
    pure (optimized_projection_relation (to_seq lw sW) (to_seq lx sX) (to_seq lo0 c0') (to_seq lo1 c1'))
