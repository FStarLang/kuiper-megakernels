module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open KPlay.SharedSpec
module I64 = FStar.Int64

(* Model-only dimension constants. *)
inline_for_extraction noextract
let argmax_vocab : nat = 151936

inline_for_extraction noextract
let argmax_prompt : nat = 7

(* Semantic postcondition: optimized_argmax_relation over the actual
   argument and output buffer contents, in ABI order
   (input0 scores, input1 prompt, output0 selected, output1 tokens). *)
fn kplay_run
  (input0 : array1 bf16 (l1_forward 151936) { is_global input0 })
  (input1 : array1 I64.t (l1_forward 7) { is_global input1 })
  (output0 : array1 I64.t (l1_forward 1) { is_global output0 })
  (output1 : array1 I64.t (l1_forward 8) { is_global output1 })
  (#input0_perm : perm) (#input1_perm : perm)
  (#input0_state : chest1 bf16 151936) (#input1_state : chest1 I64.t 7)
  (#output0_state : chest1 I64.t 1) (#output1_state : chest1 I64.t 8)
  preserves
    cpu **
    on gpu_loc (input0 |-> Frac input0_perm input0_state) **
    on gpu_loc (input1 |-> Frac input1_perm input1_state)
  requires
    on gpu_loc (output0 |-> output0_state) ** on gpu_loc (output1 |-> output1_state)
  ensures
    exists* (o0 : chest1 I64.t 1) (o1 : chest1 I64.t 8).
      on gpu_loc (output0 |-> o0) ** on gpu_loc (output1 |-> o1) **
      pure (optimized_argmax_relation (chest1_to_seq input0_state) (chest1_to_seq input1_state)
              (chest1_to_seq o0) (chest1_to_seq o1))
