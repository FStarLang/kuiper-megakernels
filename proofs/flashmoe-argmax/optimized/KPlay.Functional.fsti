module KPlay.Functional

#lang-pulse
open Kuiper
open KPlay.SharedSpec

inline_for_extraction let () = ()

(* Serial eight-expert argmax (optimized role). ABI order: input0 = row f32[8];
   output0 = selected (rank-zero int32, one element); output1 =
   selected_probability (rank-zero float32, one element). Only ownership and
   lengths are required; the postcondition is the frozen
   optimized_argmax_relation on the actual input and output contents. *)
fn kplay_run
  (row : larray f32 8 { is_global row })
  (sel : larray i32 1 { is_global sel })
  (sp : larray f32 1 { is_global sp })
  (#srow : erased (lseq f32 8))
  preserves cpu ** on gpu_loc (row |-> srow)
  requires (exists* (s0 : lseq i32 1). on gpu_loc (sel |-> s0))
  requires (exists* (p0 : lseq f32 1). on gpu_loc (sp |-> p0))
  ensures exists* (s1 : lseq i32 1) (p1 : lseq f32 1).
    on gpu_loc (sel |-> s1) ** on gpu_loc (sp |-> p1) **
    pure (optimized_argmax_relation srow s1 p1)
