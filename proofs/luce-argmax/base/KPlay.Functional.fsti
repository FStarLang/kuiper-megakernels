module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
module S = FStar.Seq
module I64 = FStar.Int64
module SS = KPlay.SharedSpec

(* Vocabulary row length (model-only dimension constant). *)
inline_for_extraction noextract
let kplay_vocab : szp = FStar.SizeT.uint32_to_sizet 248320ul

(* Rank-zero int64 output, stored as one element. *)
inline_for_extraction noextract
let kplay_one : szp = FStar.SizeT.uint32_to_sizet 1ul

(* ABI order: logits, prompt length P, steps S, position t; output selected.
   Only ownership and fixed shapes are required; the domain premises
   (P,S >= 1, P+S-1 <= 12, 0 <= t < 12) live inside base_argmax_relation. *)
fn kplay_run
  (input0 : array1 bf16 (l1_forward kplay_vocab) { is_global input0 })
  (input1 : I64.t)
  (input2 : I64.t)
  (input3 : I64.t)
  (output0 : array1 i64 (l1_forward kplay_one) { is_global output0 })
  (#input0_perm : perm)
  (#input0_state : chest1 bf16 kplay_vocab)
  (#output0_state : chest1 i64 kplay_one)
  preserves
    cpu **
    on gpu_loc (input0 |-> Frac input0_perm input0_state)
  requires
    on gpu_loc (output0 |-> output0_state)
  ensures
    exists* (output0_state' : chest1 i64 kplay_one).
      on gpu_loc (output0 |-> output0_state') **
      pure (SS.base_argmax_relation (chest1_to_seq input0_state)
              (I64.v input1) (I64.v input2) (I64.v input3)
              (chest1_to_seq output0_state'))
