module KPlay.Proof.Wiring.Optimized

(* GENERATED from the accepted decomposition plan (optimized); do not edit.
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
  sample_0000_projection_0 : S.seq F32.t;
  sample_0000_projection_1 : S.seq I32.t;
  sample_0000_projection_2 : S.seq I64.t;
  sample_0000_projection_3 : S.seq F32.t;
  sample_0000_projection_4 : S.seq I32.t;
  sample_0000_token_000_softmax_0 : S.seq F32.t;
  sample_0000_token_000_argmax_0 : S.seq I32.t;
  sample_0000_token_000_argmax_1 : S.seq F32.t;
  sample_0000_token_000_routing_0 : S.seq I32.t;
  sample_0000_token_000_routing_1 : S.seq I64.t;
  sample_0000_token_000_routing_2 : S.seq F32.t;
  sample_0000_token_000_routing_3 : S.seq I32.t;
  loop_sample_0000_token_i_softmax_0 : int -> S.seq F32.t;
  loop_sample_0000_token_i_argmax_0 : int -> S.seq I32.t;
  loop_sample_0000_token_i_argmax_1 : int -> S.seq F32.t;
  loop_sample_0000_token_i_routing_0 : int -> S.seq I32.t;
  loop_sample_0000_token_i_routing_1 : int -> S.seq I64.t;
  loop_sample_0000_token_i_routing_2 : int -> S.seq F32.t;
  loop_sample_0000_token_i_routing_3 : int -> S.seq I32.t;
}

let input_lengths (r:run) : prop =
  S.length r.input0 == 131072 /\ S.length r.input1 == 4096

(* The repeated host calls, iterations 0 .. loop_count - 1. Loop-carried arguments of
   iteration 0 come from the preceding fixed calls. *)
let loop_count : nat = 255

let loop_ok (r:run) (i:nat) : prop =
  (* loop_sample_0000_token_i_softmax: host call 4 + 3 * i *)
  optimized_softmax_relation r.sample_0000_projection_0 (1 + i) (r.loop_sample_0000_token_i_softmax_0 i) /\
  (* loop_sample_0000_token_i_argmax: host call 5 + 3 * i *)
  optimized_argmax_relation (r.loop_sample_0000_token_i_softmax_0 i) (r.loop_sample_0000_token_i_argmax_0 i) (r.loop_sample_0000_token_i_argmax_1 i) /\
  (* loop_sample_0000_token_i_routing: host call 6 + 3 * i *)
  optimized_routing_relation (r.loop_sample_0000_token_i_softmax_0 i) (r.loop_sample_0000_token_i_argmax_0 i) (r.loop_sample_0000_token_i_argmax_1 i) (if i = 0 then r.sample_0000_token_000_routing_0 else r.loop_sample_0000_token_i_routing_0 (i - 1)) (if i = 0 then r.sample_0000_token_000_routing_1 else r.loop_sample_0000_token_i_routing_1 (i - 1)) (if i = 0 then r.sample_0000_token_000_routing_2 else r.loop_sample_0000_token_i_routing_2 (i - 1)) (if i = 0 then r.sample_0000_token_000_routing_3 else r.loop_sample_0000_token_i_routing_3 (i - 1)) (1 + i) (r.loop_sample_0000_token_i_routing_0 i) (r.loop_sample_0000_token_i_routing_1 i) (r.loop_sample_0000_token_i_routing_2 i) (r.loop_sample_0000_token_i_routing_3 i)

let family_ok (r:run) : prop =
  (* sample_0000_projection *)
  optimized_projection_relation r.input0 r.input1 r.sample_0000_projection_0 r.sample_0000_projection_1 r.sample_0000_projection_2 r.sample_0000_projection_3 r.sample_0000_projection_4 /\
  (* sample_0000_token_000_softmax *)
  optimized_softmax_relation r.sample_0000_projection_0 0 r.sample_0000_token_000_softmax_0 /\
  (* sample_0000_token_000_argmax *)
  optimized_argmax_relation r.sample_0000_token_000_softmax_0 r.sample_0000_token_000_argmax_0 r.sample_0000_token_000_argmax_1 /\
  (* sample_0000_token_000_routing *)
  optimized_routing_relation r.sample_0000_token_000_softmax_0 r.sample_0000_token_000_argmax_0 r.sample_0000_token_000_argmax_1 r.sample_0000_projection_1 r.sample_0000_projection_2 r.sample_0000_projection_3 r.sample_0000_projection_4 0 r.sample_0000_token_000_routing_0 r.sample_0000_token_000_routing_1 r.sample_0000_token_000_routing_2 r.sample_0000_token_000_routing_3 /\
  (forall (i:nat). i < loop_count ==> loop_ok r i)

(* Program outputs, as bound by the plan. *)
let output0 (r:run) = (r.loop_sample_0000_token_i_routing_0 254)
let output1 (r:run) = (r.loop_sample_0000_token_i_routing_1 254)
let output2 (r:run) = (r.loop_sample_0000_token_i_routing_2 254)
let output3 (r:run) = (r.loop_sample_0000_token_i_routing_3 254)
