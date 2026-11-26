module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open KPlay.SharedSpec
module SZ = FStar.SizeT

(* Number of experts: the extent of both the int64 prefix input and the int32
   expert-demand output. *)
inline_for_extraction noextract
let kplay_experts : szp = 8sz

(* router_expert_demand: expert_demand[e] = int32_t(prefix[e]) for e = 0..7. *)
fn kplay_run
  (input0 : array1 i64 (l1_forward kplay_experts) { is_global input0 })
  (output0 : array1 i32 (l1_forward kplay_experts) { is_global output0 })
  (#input0_perm : perm)
  (#input0_state : chest1 i64 (SZ.v kplay_experts))
  (#output0_state : chest1 i32 (SZ.v kplay_experts))
  preserves
    cpu **
    on gpu_loc (input0 |-> Frac input0_perm input0_state)
  requires
    on gpu_loc (output0 |-> output0_state)
  ensures
    exists* (output0_state' : chest1 i32 (SZ.v kplay_experts)).
      on gpu_loc (output0 |-> output0_state') **
      pure (base_expert_demand_relation
              (chest1_to_seq input0_state) (chest1_to_seq output0_state'))
