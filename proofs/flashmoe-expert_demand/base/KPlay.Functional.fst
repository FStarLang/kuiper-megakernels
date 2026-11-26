module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open KPlay.SharedSpec
module SZ = FStar.SizeT
module S = FStar.Seq
module I32 = FStar.Int32
module I64 = FStar.Int64
module Cast = FStar.Int.Cast

(* Native meaning of int32_t(x) on an int64 value. *)
let demand_of (x : i64) : i32 = Cast.int64_to_int32 x

(* The output agrees with the elementwise conversion on the first n experts. *)
let converted_prefix (inp : chest1 i64 (SZ.v kplay_experts))
  (out : chest1 i32 (SZ.v kplay_experts)) (n : nat) : prop =
  forall (e : nat). e < n /\ e < SZ.v kplay_experts ==>
    acc1 out e == demand_of (acc1 inp e)

(* The int64 -> int32 conversion is exact on [0, 256]. *)
let demand_of_exact (x : i64)
  : Lemma (requires 0 <= I64.v x /\ I64.v x <= tokens)
          (ensures I32.v (demand_of x) == I64.v x)
  = FStar.Math.Lemmas.small_mod (I64.v x) (pow2 32)

let converted_relation (inp : chest1 i64 (SZ.v kplay_experts))
  (out : chest1 i32 (SZ.v kplay_experts))
  : Lemma (requires converted_prefix inp out (SZ.v kplay_experts))
          (ensures base_expert_demand_relation (chest1_to_seq inp) (chest1_to_seq out))
  = assert (SZ.v kplay_experts == experts);
    let si : Seq.lseq i64 experts = chest1_to_seq inp in
    let so : Seq.lseq i32 experts = chest1_to_seq out in
    let aux (e : nat{e < experts})
      : Lemma (requires 0 <= I64.v (S.index si e) /\ I64.v (S.index si e) <= tokens)
              (ensures I32.v (S.index so e) == I64.v (S.index si e))
      = demand_of_exact (acc1 inp e)
    in
    Classical.forall_intro (Classical.move_requires aux)

inline_for_extraction noextract
fn kf
  (input0 : array1 i64 (l1_forward kplay_experts) { is_global input0 })
  (output0 : array1 i32 (l1_forward kplay_experts) { is_global output0 })
  (#input0_perm : perm)
  (#input0_state : chest1 i64 (SZ.v kplay_experts))
  (#output0_state : chest1 i32 (SZ.v kplay_experts))
  preserves
    gpu **
    input0 |-> Frac input0_perm input0_state
  requires
    output0 |-> output0_state
  ensures
    exists* (output0_state' : chest1 i32 (SZ.v kplay_experts)).
      output0 |-> output0_state' **
      pure (converted_prefix input0_state output0_state' (SZ.v kplay_experts))
{
  let mut i : szle kplay_experts = 0sz;
  while (!i <^ kplay_experts)
    invariant exists* (iv : szle kplay_experts) (state : chest1 i32 (SZ.v kplay_experts)).
      i |-> iv **
      output0 |-> state **
      pure (converted_prefix input0_state state (SZ.v iv))
    decreases (SZ.v kplay_experts - SZ.v !i)
  {
    let iv = !i;
    let value =
      Kuiper.Tensor.read input0
        ((iv <: szlt kplay_experts), ())
        #input0_perm;
    Kuiper.Tensor.write output0
      ((iv <: szlt kplay_experts), ())
      (Cast.int64_to_int32 value);
    i := iv +^ 1sz;
  };
}

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
{
  launch_kernel_1
    (fun () ->
      kf input0 output0
        #input0_perm #input0_state #output0_state);
  with s. assert (on gpu_loc (output0 |-> s));
  converted_relation input0_state s;
}
