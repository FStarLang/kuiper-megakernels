module KPlay.Proof.Wiring.Base

(* GENERATED from the accepted decomposition plan (base); do not edit.
   One record field per program input and per family-call output; family_ok
   is the conjunction of the SharedSpec family relations over the exact
   arguments of each call, in host order. Tensors are flat row-major
   sequences; a view is the flat slice [offset, offset + numel). *)

open KPlay.SharedSpec
module S = FStar.Seq
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32
module I32 = FStar.Int32
module I64 = FStar.Int64

let view (#a:Type) (s:S.seq a) (offset len:nat) : S.seq a =
  if offset + len <= S.length s then S.slice s offset (offset + len) else S.empty

noeq type run = {
  input0 : S.seq BF.t;
  input1 : S.seq BF.t;
  sample_0000_projection_softmax_0 : S.seq F32.t;
  sample_0000_projection_softmax_1 : S.seq I64.t;
  sample_0000_projection_softmax_2 : S.seq I64.t;
  sample_0000_projection_softmax_3 : S.seq F32.t;
  sample_0000_projection_softmax_4 : S.seq I32.t;
  sample_0000_argmax_000_0 : S.seq I32.t;
  sample_0000_argmax_000_1 : S.seq F32.t;
  sample_0000_capacity_000_0 : S.seq I64.t;
  sample_0000_capacity_000_1 : S.seq I64.t;
  sample_0000_capacity_000_2 : S.seq F32.t;
  sample_0000_capacity_000_3 : S.seq I32.t;
  loop_sample_0000_argmax_i_0 : int -> S.seq I32.t;
  loop_sample_0000_argmax_i_1 : int -> S.seq F32.t;
  loop_sample_0000_capacity_i_0 : int -> S.seq I64.t;
  loop_sample_0000_capacity_i_1 : int -> S.seq I64.t;
  loop_sample_0000_capacity_i_2 : int -> S.seq F32.t;
  loop_sample_0000_capacity_i_3 : int -> S.seq I32.t;
  sample_0000_expert_demand_0 : S.seq I32.t;
}

let input_lengths (r:run) : prop =
  S.length r.input0 == 131072 /\ S.length r.input1 == 4096

(* The repeated host calls, iterations 0 .. loop_count - 1. Loop-carried arguments of
   iteration 0 come from the preceding fixed calls. *)
let loop_count : nat = 255

let loop_ok (r:run) (i:nat) : prop =
  (* loop_sample_0000_argmax_i: host call 3 + 2 * i *)
  base_argmax_relation (view r.sample_0000_projection_softmax_0 (8 + 8 * i) 8) (r.loop_sample_0000_argmax_i_0 i) (r.loop_sample_0000_argmax_i_1 i) /\
  (* loop_sample_0000_capacity_i: host call 4 + 2 * i *)
  base_capacity_routing_relation (r.loop_sample_0000_argmax_i_0 i) (r.loop_sample_0000_argmax_i_1 i) (if i = 0 then r.sample_0000_capacity_000_0 else r.loop_sample_0000_capacity_i_0 (i - 1)) (if i = 0 then r.sample_0000_capacity_000_1 else r.loop_sample_0000_capacity_i_1 (i - 1)) (if i = 0 then r.sample_0000_capacity_000_2 else r.loop_sample_0000_capacity_i_2 (i - 1)) (if i = 0 then r.sample_0000_capacity_000_3 else r.loop_sample_0000_capacity_i_3 (i - 1)) (1 + i) (r.loop_sample_0000_capacity_i_0 i) (r.loop_sample_0000_capacity_i_1 i) (r.loop_sample_0000_capacity_i_2 i) (r.loop_sample_0000_capacity_i_3 i)

let family_ok (r:run) : prop =
  (* sample_0000_projection_softmax *)
  base_projection_softmax_relation r.input0 r.input1 r.sample_0000_projection_softmax_0 r.sample_0000_projection_softmax_1 r.sample_0000_projection_softmax_2 r.sample_0000_projection_softmax_3 r.sample_0000_projection_softmax_4 /\
  (* sample_0000_argmax_000 *)
  base_argmax_relation (view r.sample_0000_projection_softmax_0 0 8) r.sample_0000_argmax_000_0 r.sample_0000_argmax_000_1 /\
  (* sample_0000_capacity_000 *)
  base_capacity_routing_relation r.sample_0000_argmax_000_0 r.sample_0000_argmax_000_1 r.sample_0000_projection_softmax_1 r.sample_0000_projection_softmax_2 r.sample_0000_projection_softmax_3 r.sample_0000_projection_softmax_4 0 r.sample_0000_capacity_000_0 r.sample_0000_capacity_000_1 r.sample_0000_capacity_000_2 r.sample_0000_capacity_000_3 /\
  (forall (i:nat). i < loop_count ==> loop_ok r i) /\
  (* sample_0000_expert_demand *)
  base_expert_demand_relation (r.loop_sample_0000_capacity_i_0 254) r.sample_0000_expert_demand_0

(* Program outputs, as bound by the plan. *)
let output0 (r:run) = r.sample_0000_expert_demand_0
let output1 (r:run) = (r.loop_sample_0000_capacity_i_1 254)
let output2 (r:run) = (r.loop_sample_0000_capacity_i_2 254)
let output3 (r:run) = (r.loop_sample_0000_capacity_i_3 254)
