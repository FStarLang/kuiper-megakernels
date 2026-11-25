module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = Kuiper.SizeT
module I64 = FStar.Int64

inline_for_extraction let () = ()

(* Witness for the base `projection` family: the tied LM head over the seven
   prompt rows. ABI order (target.json): input0 = packed weights (n0
   elements), input1 = [7,1024] hidden rows, input2 = LM-head weight offset,
   output0 = [7,151936] logits. The postcondition is base_projection_relation
   on the actual buffer contents; only ownership, placement and lengths are
   required. *)
fn kplay_run
  (n0 : SZ.t)
  (in0 : larray bf16 (SZ.v n0))
  (in1 : larray bf16 7168)
  (in2 : I64.t)
  (out0 : larray bf16 (7 * 151936))
  (#f0 #f1 : perm)
  (#s0 #s1 #so : erased (S.seq bf16))
  preserves cpu
  preserves on gpu_loc (in0 |-> Frac f0 s0)
  preserves on gpu_loc (in1 |-> Frac f1 s1)
  requires on gpu_loc (out0 |-> so)
  requires pure (is_global in0 /\ is_global in1 /\ is_global out0)
  ensures exists* (so' : S.seq bf16).
    on gpu_loc (out0 |-> so') **
    pure (base_projection_relation s0 s1 (I64.v in2) so')
