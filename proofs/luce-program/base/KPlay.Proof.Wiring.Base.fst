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
module I64 = FStar.Int64

let view (#a:Type) (s:S.seq a) (offset len:int) : S.seq a =
  if 0 <= offset && 0 <= len && offset + len <= S.length s then S.slice s offset (offset + len) else S.empty

noeq type run = {
  input0 : S.seq BF.t;
  input1 : S.seq I64.t;
  input2 : int;
  initialize_0 : S.seq BF.t;
  initialize_1 : S.seq BF.t;
  initialize_2 : S.seq F32.t;
  initialize_3 : S.seq I64.t;
  initialize_4 : S.seq BF.t;
  position_00_embedding_0 : S.seq BF.t;
  position_00_embedding_1 : S.seq BF.t;
  position_00_embedding_2 : S.seq BF.t;
  loop0_position_00_layer_i_0_0 : int -> S.seq BF.t;
  loop0_position_00_layer_i_0_1 : int -> S.seq BF.t;
  loop0_position_00_layer_i_0_2 : int -> S.seq F32.t;
  loop0_position_00_layer_i_1_0 : int -> S.seq BF.t;
  loop0_position_00_layer_i_1_1 : int -> S.seq BF.t;
  loop0_position_00_layer_i_1_2 : int -> S.seq F32.t;
  loop0_position_00_layer_i_2_0 : int -> S.seq BF.t;
  loop0_position_00_layer_i_2_1 : int -> S.seq BF.t;
  loop0_position_00_layer_i_2_2 : int -> S.seq F32.t;
  loop0_position_00_layer_i_3_0 : int -> S.seq BF.t;
  loop0_position_00_layer_i_3_1 : int -> S.seq BF.t;
  position_00_final_0 : S.seq BF.t;
  position_00_final_1 : S.seq BF.t;
  position_00_argmax_0 : S.seq I64.t;
  position_00_update_0 : S.seq I64.t;
  position_00_update_1 : S.seq BF.t;
  loop1_position_i_embedding_0 : int -> S.seq BF.t;
  loop1_position_i_embedding_1 : int -> S.seq BF.t;
  loop1_position_i_embedding_2 : int -> S.seq BF.t;
  loop1_position_i_layer_00_0 : int -> S.seq BF.t;
  loop1_position_i_layer_00_1 : int -> S.seq BF.t;
  loop1_position_i_layer_00_2 : int -> S.seq F32.t;
  loop1_position_i_layer_01_0 : int -> S.seq BF.t;
  loop1_position_i_layer_01_1 : int -> S.seq BF.t;
  loop1_position_i_layer_01_2 : int -> S.seq F32.t;
  loop1_position_i_layer_02_0 : int -> S.seq BF.t;
  loop1_position_i_layer_02_1 : int -> S.seq BF.t;
  loop1_position_i_layer_02_2 : int -> S.seq F32.t;
  loop1_position_i_layer_03_0 : int -> S.seq BF.t;
  loop1_position_i_layer_03_1 : int -> S.seq BF.t;
  loop1_position_i_layer_04_0 : int -> S.seq BF.t;
  loop1_position_i_layer_04_1 : int -> S.seq BF.t;
  loop1_position_i_layer_04_2 : int -> S.seq F32.t;
  loop1_position_i_layer_05_0 : int -> S.seq BF.t;
  loop1_position_i_layer_05_1 : int -> S.seq BF.t;
  loop1_position_i_layer_05_2 : int -> S.seq F32.t;
  loop1_position_i_layer_06_0 : int -> S.seq BF.t;
  loop1_position_i_layer_06_1 : int -> S.seq BF.t;
  loop1_position_i_layer_06_2 : int -> S.seq F32.t;
  loop1_position_i_layer_07_0 : int -> S.seq BF.t;
  loop1_position_i_layer_07_1 : int -> S.seq BF.t;
  loop1_position_i_layer_08_0 : int -> S.seq BF.t;
  loop1_position_i_layer_08_1 : int -> S.seq BF.t;
  loop1_position_i_layer_08_2 : int -> S.seq F32.t;
  loop1_position_i_layer_09_0 : int -> S.seq BF.t;
  loop1_position_i_layer_09_1 : int -> S.seq BF.t;
  loop1_position_i_layer_09_2 : int -> S.seq F32.t;
  loop1_position_i_layer_10_0 : int -> S.seq BF.t;
  loop1_position_i_layer_10_1 : int -> S.seq BF.t;
  loop1_position_i_layer_10_2 : int -> S.seq F32.t;
  loop1_position_i_layer_11_0 : int -> S.seq BF.t;
  loop1_position_i_layer_11_1 : int -> S.seq BF.t;
  loop1_position_i_layer_12_0 : int -> S.seq BF.t;
  loop1_position_i_layer_12_1 : int -> S.seq BF.t;
  loop1_position_i_layer_12_2 : int -> S.seq F32.t;
  loop1_position_i_layer_13_0 : int -> S.seq BF.t;
  loop1_position_i_layer_13_1 : int -> S.seq BF.t;
  loop1_position_i_layer_13_2 : int -> S.seq F32.t;
  loop1_position_i_layer_14_0 : int -> S.seq BF.t;
  loop1_position_i_layer_14_1 : int -> S.seq BF.t;
  loop1_position_i_layer_14_2 : int -> S.seq F32.t;
  loop1_position_i_layer_15_0 : int -> S.seq BF.t;
  loop1_position_i_layer_15_1 : int -> S.seq BF.t;
  loop1_position_i_layer_16_0 : int -> S.seq BF.t;
  loop1_position_i_layer_16_1 : int -> S.seq BF.t;
  loop1_position_i_layer_16_2 : int -> S.seq F32.t;
  loop1_position_i_layer_17_0 : int -> S.seq BF.t;
  loop1_position_i_layer_17_1 : int -> S.seq BF.t;
  loop1_position_i_layer_17_2 : int -> S.seq F32.t;
  loop1_position_i_layer_18_0 : int -> S.seq BF.t;
  loop1_position_i_layer_18_1 : int -> S.seq BF.t;
  loop1_position_i_layer_18_2 : int -> S.seq F32.t;
  loop1_position_i_layer_19_0 : int -> S.seq BF.t;
  loop1_position_i_layer_19_1 : int -> S.seq BF.t;
  loop1_position_i_layer_20_0 : int -> S.seq BF.t;
  loop1_position_i_layer_20_1 : int -> S.seq BF.t;
  loop1_position_i_layer_20_2 : int -> S.seq F32.t;
  loop1_position_i_layer_21_0 : int -> S.seq BF.t;
  loop1_position_i_layer_21_1 : int -> S.seq BF.t;
  loop1_position_i_layer_21_2 : int -> S.seq F32.t;
  loop1_position_i_layer_22_0 : int -> S.seq BF.t;
  loop1_position_i_layer_22_1 : int -> S.seq BF.t;
  loop1_position_i_layer_22_2 : int -> S.seq F32.t;
  loop1_position_i_layer_23_0 : int -> S.seq BF.t;
  loop1_position_i_layer_23_1 : int -> S.seq BF.t;
  loop1_position_i_final_0 : int -> S.seq BF.t;
  loop1_position_i_final_1 : int -> S.seq BF.t;
  loop1_position_i_argmax_0 : int -> S.seq I64.t;
  loop1_position_i_update_0 : int -> S.seq I64.t;
  loop1_position_i_update_1 : int -> S.seq BF.t;
  loop2_position_i_argmax_0 : int -> S.seq I64.t;
  loop2_position_i_update_0 : int -> S.seq I64.t;
  loop2_position_i_update_1 : int -> S.seq BF.t;
}

let input_lengths (r:run) : prop =
  S.length r.input0 == 752393024 /\ 1 <= S.length r.input1 /\ S.length r.input1 <= 8 /\ 1 <= r.input2 /\ r.input2 <= 5

(* The repeated host calls of loop 0, iterations 0 .. loop0_count - 1. Loop-carried
   arguments of iteration 0 come from the preceding calls. *)
let loop0_count (r:run) : int = 6

let loop0_ok (r:run) (i:nat) : prop =
  (* loop0_position_00_layer_i_0: host call 2 + 4 * i *)
  base_recurrent_decoder_relation (view r.input0 (254280704 + 83018720 * i) 21555360) (if i = 0 then r.position_00_embedding_0 else r.loop0_position_00_layer_i_3_0 (i - 1)) (view r.initialize_1 (0 + 73728 * i) 24576) (view r.initialize_2 (0 + 786432 * i) 262144) (r.loop0_position_00_layer_i_0_0 i) (r.loop0_position_00_layer_i_0_1 i) (r.loop0_position_00_layer_i_0_2 i) /\
  (* loop0_position_00_layer_i_1: host call 3 + 4 * i *)
  base_recurrent_decoder_relation (view r.input0 (275836064 + 83018720 * i) 21555360) (r.loop0_position_00_layer_i_0_0 i) (view r.initialize_1 (24576 + 73728 * i) 24576) (view r.initialize_2 (262144 + 786432 * i) 262144) (r.loop0_position_00_layer_i_1_0 i) (r.loop0_position_00_layer_i_1_1 i) (r.loop0_position_00_layer_i_1_2 i) /\
  (* loop0_position_00_layer_i_2: host call 4 + 4 * i *)
  base_recurrent_decoder_relation (view r.input0 (297391424 + 83018720 * i) 21555360) (r.loop0_position_00_layer_i_1_0 i) (view r.initialize_1 (49152 + 73728 * i) 24576) (view r.initialize_2 (524288 + 786432 * i) 262144) (r.loop0_position_00_layer_i_2_0 i) (r.loop0_position_00_layer_i_2_1 i) (r.loop0_position_00_layer_i_2_2 i) /\
  (* loop0_position_00_layer_i_3: host call 5 + 4 * i *)
  base_full_attention_decoder_relation (view r.input0 (318946784 + 83018720 * i) 18352640) (r.loop0_position_00_layer_i_2_0 i) (view r.initialize_1 (442368 + (1024 * S.length r.input1 + 1024 * r.input2 - 1024) * i) (1024 * S.length r.input1 + 1024 * r.input2 - 1024)) r.position_00_embedding_1 r.position_00_embedding_2 0 (r.loop0_position_00_layer_i_3_0 i) (r.loop0_position_00_layer_i_3_1 i)

(* The repeated host calls of loop 1, iterations 0 .. loop1_count - 1. Loop-carried
   arguments of iteration 0 come from the preceding calls. *)
let loop1_count (r:run) : int = (S.length r.input1 + r.input2 - 2)

let loop1_ok (r:run) (i:nat) : prop =
  (* loop1_position_i_embedding: host call 29 + 28 * i *)
  base_embedding_rope_relation (view r.input0 0 254279680) r.initialize_0 (if i = 0 then r.position_00_update_0 else r.loop1_position_i_update_0 (i - 1)) (1 + 1 * i) (r.loop1_position_i_embedding_0 i) (r.loop1_position_i_embedding_1 i) (r.loop1_position_i_embedding_2 i) /\
  (* loop1_position_i_layer_00: host call 30 + 28 * i *)
  base_recurrent_decoder_relation (view r.input0 254280704 21555360) (r.loop1_position_i_embedding_0 i) (if i = 0 then (r.loop0_position_00_layer_i_0_1 0) else r.loop1_position_i_layer_00_1 (i - 1)) (if i = 0 then (r.loop0_position_00_layer_i_0_2 0) else r.loop1_position_i_layer_00_2 (i - 1)) (r.loop1_position_i_layer_00_0 i) (r.loop1_position_i_layer_00_1 i) (r.loop1_position_i_layer_00_2 i) /\
  (* loop1_position_i_layer_01: host call 31 + 28 * i *)
  base_recurrent_decoder_relation (view r.input0 275836064 21555360) (r.loop1_position_i_layer_00_0 i) (if i = 0 then (r.loop0_position_00_layer_i_1_1 0) else r.loop1_position_i_layer_01_1 (i - 1)) (if i = 0 then (r.loop0_position_00_layer_i_1_2 0) else r.loop1_position_i_layer_01_2 (i - 1)) (r.loop1_position_i_layer_01_0 i) (r.loop1_position_i_layer_01_1 i) (r.loop1_position_i_layer_01_2 i) /\
  (* loop1_position_i_layer_02: host call 32 + 28 * i *)
  base_recurrent_decoder_relation (view r.input0 297391424 21555360) (r.loop1_position_i_layer_01_0 i) (if i = 0 then (r.loop0_position_00_layer_i_2_1 0) else r.loop1_position_i_layer_02_1 (i - 1)) (if i = 0 then (r.loop0_position_00_layer_i_2_2 0) else r.loop1_position_i_layer_02_2 (i - 1)) (r.loop1_position_i_layer_02_0 i) (r.loop1_position_i_layer_02_1 i) (r.loop1_position_i_layer_02_2 i) /\
  (* loop1_position_i_layer_03: host call 33 + 28 * i *)
  base_full_attention_decoder_relation (view r.input0 318946784 18352640) (r.loop1_position_i_layer_02_0 i) (if i = 0 then (r.loop0_position_00_layer_i_3_1 0) else r.loop1_position_i_layer_03_1 (i - 1)) (r.loop1_position_i_embedding_1 i) (r.loop1_position_i_embedding_2 i) (1 + 1 * i) (r.loop1_position_i_layer_03_0 i) (r.loop1_position_i_layer_03_1 i) /\
  (* loop1_position_i_layer_04: host call 34 + 28 * i *)
  base_recurrent_decoder_relation (view r.input0 337299424 21555360) (r.loop1_position_i_layer_03_0 i) (if i = 0 then (r.loop0_position_00_layer_i_0_1 1) else r.loop1_position_i_layer_04_1 (i - 1)) (if i = 0 then (r.loop0_position_00_layer_i_0_2 1) else r.loop1_position_i_layer_04_2 (i - 1)) (r.loop1_position_i_layer_04_0 i) (r.loop1_position_i_layer_04_1 i) (r.loop1_position_i_layer_04_2 i) /\
  (* loop1_position_i_layer_05: host call 35 + 28 * i *)
  base_recurrent_decoder_relation (view r.input0 358854784 21555360) (r.loop1_position_i_layer_04_0 i) (if i = 0 then (r.loop0_position_00_layer_i_1_1 1) else r.loop1_position_i_layer_05_1 (i - 1)) (if i = 0 then (r.loop0_position_00_layer_i_1_2 1) else r.loop1_position_i_layer_05_2 (i - 1)) (r.loop1_position_i_layer_05_0 i) (r.loop1_position_i_layer_05_1 i) (r.loop1_position_i_layer_05_2 i) /\
  (* loop1_position_i_layer_06: host call 36 + 28 * i *)
  base_recurrent_decoder_relation (view r.input0 380410144 21555360) (r.loop1_position_i_layer_05_0 i) (if i = 0 then (r.loop0_position_00_layer_i_2_1 1) else r.loop1_position_i_layer_06_1 (i - 1)) (if i = 0 then (r.loop0_position_00_layer_i_2_2 1) else r.loop1_position_i_layer_06_2 (i - 1)) (r.loop1_position_i_layer_06_0 i) (r.loop1_position_i_layer_06_1 i) (r.loop1_position_i_layer_06_2 i) /\
  (* loop1_position_i_layer_07: host call 37 + 28 * i *)
  base_full_attention_decoder_relation (view r.input0 401965504 18352640) (r.loop1_position_i_layer_06_0 i) (if i = 0 then (r.loop0_position_00_layer_i_3_1 1) else r.loop1_position_i_layer_07_1 (i - 1)) (r.loop1_position_i_embedding_1 i) (r.loop1_position_i_embedding_2 i) (1 + 1 * i) (r.loop1_position_i_layer_07_0 i) (r.loop1_position_i_layer_07_1 i) /\
  (* loop1_position_i_layer_08: host call 38 + 28 * i *)
  base_recurrent_decoder_relation (view r.input0 420318144 21555360) (r.loop1_position_i_layer_07_0 i) (if i = 0 then (r.loop0_position_00_layer_i_0_1 2) else r.loop1_position_i_layer_08_1 (i - 1)) (if i = 0 then (r.loop0_position_00_layer_i_0_2 2) else r.loop1_position_i_layer_08_2 (i - 1)) (r.loop1_position_i_layer_08_0 i) (r.loop1_position_i_layer_08_1 i) (r.loop1_position_i_layer_08_2 i) /\
  (* loop1_position_i_layer_09: host call 39 + 28 * i *)
  base_recurrent_decoder_relation (view r.input0 441873504 21555360) (r.loop1_position_i_layer_08_0 i) (if i = 0 then (r.loop0_position_00_layer_i_1_1 2) else r.loop1_position_i_layer_09_1 (i - 1)) (if i = 0 then (r.loop0_position_00_layer_i_1_2 2) else r.loop1_position_i_layer_09_2 (i - 1)) (r.loop1_position_i_layer_09_0 i) (r.loop1_position_i_layer_09_1 i) (r.loop1_position_i_layer_09_2 i) /\
  (* loop1_position_i_layer_10: host call 40 + 28 * i *)
  base_recurrent_decoder_relation (view r.input0 463428864 21555360) (r.loop1_position_i_layer_09_0 i) (if i = 0 then (r.loop0_position_00_layer_i_2_1 2) else r.loop1_position_i_layer_10_1 (i - 1)) (if i = 0 then (r.loop0_position_00_layer_i_2_2 2) else r.loop1_position_i_layer_10_2 (i - 1)) (r.loop1_position_i_layer_10_0 i) (r.loop1_position_i_layer_10_1 i) (r.loop1_position_i_layer_10_2 i) /\
  (* loop1_position_i_layer_11: host call 41 + 28 * i *)
  base_full_attention_decoder_relation (view r.input0 484984224 18352640) (r.loop1_position_i_layer_10_0 i) (if i = 0 then (r.loop0_position_00_layer_i_3_1 2) else r.loop1_position_i_layer_11_1 (i - 1)) (r.loop1_position_i_embedding_1 i) (r.loop1_position_i_embedding_2 i) (1 + 1 * i) (r.loop1_position_i_layer_11_0 i) (r.loop1_position_i_layer_11_1 i) /\
  (* loop1_position_i_layer_12: host call 42 + 28 * i *)
  base_recurrent_decoder_relation (view r.input0 503336864 21555360) (r.loop1_position_i_layer_11_0 i) (if i = 0 then (r.loop0_position_00_layer_i_0_1 3) else r.loop1_position_i_layer_12_1 (i - 1)) (if i = 0 then (r.loop0_position_00_layer_i_0_2 3) else r.loop1_position_i_layer_12_2 (i - 1)) (r.loop1_position_i_layer_12_0 i) (r.loop1_position_i_layer_12_1 i) (r.loop1_position_i_layer_12_2 i) /\
  (* loop1_position_i_layer_13: host call 43 + 28 * i *)
  base_recurrent_decoder_relation (view r.input0 524892224 21555360) (r.loop1_position_i_layer_12_0 i) (if i = 0 then (r.loop0_position_00_layer_i_1_1 3) else r.loop1_position_i_layer_13_1 (i - 1)) (if i = 0 then (r.loop0_position_00_layer_i_1_2 3) else r.loop1_position_i_layer_13_2 (i - 1)) (r.loop1_position_i_layer_13_0 i) (r.loop1_position_i_layer_13_1 i) (r.loop1_position_i_layer_13_2 i) /\
  (* loop1_position_i_layer_14: host call 44 + 28 * i *)
  base_recurrent_decoder_relation (view r.input0 546447584 21555360) (r.loop1_position_i_layer_13_0 i) (if i = 0 then (r.loop0_position_00_layer_i_2_1 3) else r.loop1_position_i_layer_14_1 (i - 1)) (if i = 0 then (r.loop0_position_00_layer_i_2_2 3) else r.loop1_position_i_layer_14_2 (i - 1)) (r.loop1_position_i_layer_14_0 i) (r.loop1_position_i_layer_14_1 i) (r.loop1_position_i_layer_14_2 i) /\
  (* loop1_position_i_layer_15: host call 45 + 28 * i *)
  base_full_attention_decoder_relation (view r.input0 568002944 18352640) (r.loop1_position_i_layer_14_0 i) (if i = 0 then (r.loop0_position_00_layer_i_3_1 3) else r.loop1_position_i_layer_15_1 (i - 1)) (r.loop1_position_i_embedding_1 i) (r.loop1_position_i_embedding_2 i) (1 + 1 * i) (r.loop1_position_i_layer_15_0 i) (r.loop1_position_i_layer_15_1 i) /\
  (* loop1_position_i_layer_16: host call 46 + 28 * i *)
  base_recurrent_decoder_relation (view r.input0 586355584 21555360) (r.loop1_position_i_layer_15_0 i) (if i = 0 then (r.loop0_position_00_layer_i_0_1 4) else r.loop1_position_i_layer_16_1 (i - 1)) (if i = 0 then (r.loop0_position_00_layer_i_0_2 4) else r.loop1_position_i_layer_16_2 (i - 1)) (r.loop1_position_i_layer_16_0 i) (r.loop1_position_i_layer_16_1 i) (r.loop1_position_i_layer_16_2 i) /\
  (* loop1_position_i_layer_17: host call 47 + 28 * i *)
  base_recurrent_decoder_relation (view r.input0 607910944 21555360) (r.loop1_position_i_layer_16_0 i) (if i = 0 then (r.loop0_position_00_layer_i_1_1 4) else r.loop1_position_i_layer_17_1 (i - 1)) (if i = 0 then (r.loop0_position_00_layer_i_1_2 4) else r.loop1_position_i_layer_17_2 (i - 1)) (r.loop1_position_i_layer_17_0 i) (r.loop1_position_i_layer_17_1 i) (r.loop1_position_i_layer_17_2 i) /\
  (* loop1_position_i_layer_18: host call 48 + 28 * i *)
  base_recurrent_decoder_relation (view r.input0 629466304 21555360) (r.loop1_position_i_layer_17_0 i) (if i = 0 then (r.loop0_position_00_layer_i_2_1 4) else r.loop1_position_i_layer_18_1 (i - 1)) (if i = 0 then (r.loop0_position_00_layer_i_2_2 4) else r.loop1_position_i_layer_18_2 (i - 1)) (r.loop1_position_i_layer_18_0 i) (r.loop1_position_i_layer_18_1 i) (r.loop1_position_i_layer_18_2 i) /\
  (* loop1_position_i_layer_19: host call 49 + 28 * i *)
  base_full_attention_decoder_relation (view r.input0 651021664 18352640) (r.loop1_position_i_layer_18_0 i) (if i = 0 then (r.loop0_position_00_layer_i_3_1 4) else r.loop1_position_i_layer_19_1 (i - 1)) (r.loop1_position_i_embedding_1 i) (r.loop1_position_i_embedding_2 i) (1 + 1 * i) (r.loop1_position_i_layer_19_0 i) (r.loop1_position_i_layer_19_1 i) /\
  (* loop1_position_i_layer_20: host call 50 + 28 * i *)
  base_recurrent_decoder_relation (view r.input0 669374304 21555360) (r.loop1_position_i_layer_19_0 i) (if i = 0 then (r.loop0_position_00_layer_i_0_1 5) else r.loop1_position_i_layer_20_1 (i - 1)) (if i = 0 then (r.loop0_position_00_layer_i_0_2 5) else r.loop1_position_i_layer_20_2 (i - 1)) (r.loop1_position_i_layer_20_0 i) (r.loop1_position_i_layer_20_1 i) (r.loop1_position_i_layer_20_2 i) /\
  (* loop1_position_i_layer_21: host call 51 + 28 * i *)
  base_recurrent_decoder_relation (view r.input0 690929664 21555360) (r.loop1_position_i_layer_20_0 i) (if i = 0 then (r.loop0_position_00_layer_i_1_1 5) else r.loop1_position_i_layer_21_1 (i - 1)) (if i = 0 then (r.loop0_position_00_layer_i_1_2 5) else r.loop1_position_i_layer_21_2 (i - 1)) (r.loop1_position_i_layer_21_0 i) (r.loop1_position_i_layer_21_1 i) (r.loop1_position_i_layer_21_2 i) /\
  (* loop1_position_i_layer_22: host call 52 + 28 * i *)
  base_recurrent_decoder_relation (view r.input0 712485024 21555360) (r.loop1_position_i_layer_21_0 i) (if i = 0 then (r.loop0_position_00_layer_i_2_1 5) else r.loop1_position_i_layer_22_1 (i - 1)) (if i = 0 then (r.loop0_position_00_layer_i_2_2 5) else r.loop1_position_i_layer_22_2 (i - 1)) (r.loop1_position_i_layer_22_0 i) (r.loop1_position_i_layer_22_1 i) (r.loop1_position_i_layer_22_2 i) /\
  (* loop1_position_i_layer_23: host call 53 + 28 * i *)
  base_full_attention_decoder_relation (view r.input0 734040384 18352640) (r.loop1_position_i_layer_22_0 i) (if i = 0 then (r.loop0_position_00_layer_i_3_1 5) else r.loop1_position_i_layer_23_1 (i - 1)) (r.loop1_position_i_embedding_1 i) (r.loop1_position_i_embedding_2 i) (1 + 1 * i) (r.loop1_position_i_layer_23_0 i) (r.loop1_position_i_layer_23_1 i) /\
  (* loop1_position_i_final: host call 54 + 28 * i *)
  base_final_norm_projection_relation (view r.input0 0 254279680) (view r.input0 254279680 1024) (r.loop1_position_i_layer_23_0 i) (r.loop1_position_i_final_0 i) (r.loop1_position_i_final_1 i) /\
  (* loop1_position_i_argmax: host call 55 + 28 * i *)
  base_argmax_relation (r.loop1_position_i_final_1 i) (S.length r.input1) r.input2 (1 + 1 * i) (r.loop1_position_i_argmax_0 i) /\
  (* loop1_position_i_update: host call 56 + 28 * i *)
  base_token_history_update_relation (if i = 0 then r.position_00_update_0 else r.loop1_position_i_update_0 (i - 1)) (if i = 0 then r.position_00_update_1 else r.loop1_position_i_update_1 (i - 1)) (r.loop1_position_i_final_0 i) (r.loop1_position_i_argmax_0 i) (S.length r.input1) (1 + 1 * i) (r.loop1_position_i_update_0 i) (r.loop1_position_i_update_1 i)

(* The repeated host calls of loop 2, iterations 0 .. loop2_count - 1. Loop-carried
   arguments of iteration 0 come from the preceding calls. *)
let loop2_count (r:run) : int = ((-1) * S.length r.input1 + (-1) * r.input2 + 13)

let loop2_ok (r:run) (i:nat) : prop =
  (* loop2_position_i_argmax: host call (28 * S.length r.input1 + 28 * r.input2 - 27) + 2 * i *)
  base_argmax_relation (if loop1_count r > 0 then (r.loop1_position_i_final_1 (loop1_count r - 1)) else r.position_00_final_1) (S.length r.input1) r.input2 ((S.length r.input1 + r.input2 - 1) + 1 * i) (r.loop2_position_i_argmax_0 i) /\
  (* loop2_position_i_update: host call (28 * S.length r.input1 + 28 * r.input2 - 26) + 2 * i *)
  base_token_history_update_relation (if i = 0 then (if loop1_count r > 0 then (r.loop1_position_i_update_0 (loop1_count r - 1)) else r.position_00_update_0) else r.loop2_position_i_update_0 (i - 1)) (if i = 0 then (if loop1_count r > 0 then (r.loop1_position_i_update_1 (loop1_count r - 1)) else r.position_00_update_1) else r.loop2_position_i_update_1 (i - 1)) (if loop1_count r > 0 then (r.loop1_position_i_final_0 (loop1_count r - 1)) else r.position_00_final_0) (r.loop2_position_i_argmax_0 i) (S.length r.input1) ((S.length r.input1 + r.input2 - 1) + 1 * i) (r.loop2_position_i_update_0 i) (r.loop2_position_i_update_1 i)

let family_ok (r:run) : prop =
  (* initialize *)
  base_initialize_relation (view r.input0 0 254279680) r.input1 r.input2 r.initialize_0 r.initialize_1 r.initialize_2 r.initialize_3 r.initialize_4 /\
  (* position_00_embedding *)
  base_embedding_rope_relation (view r.input0 0 254279680) r.initialize_0 r.initialize_3 0 r.position_00_embedding_0 r.position_00_embedding_1 r.position_00_embedding_2 /\
  (forall (i:nat). i < loop0_count r ==> loop0_ok r i) /\
  (forall (i:nat). i < loop1_count r ==> loop1_ok r i) /\
  (forall (i:nat). i < loop2_count r ==> loop2_ok r i) /\
  (* position_00_final *)
  base_final_norm_projection_relation (view r.input0 0 254279680) (view r.input0 254279680 1024) (r.loop0_position_00_layer_i_3_0 5) r.position_00_final_0 r.position_00_final_1 /\
  (* position_00_argmax *)
  base_argmax_relation r.position_00_final_1 (S.length r.input1) r.input2 0 r.position_00_argmax_0 /\
  (* position_00_update *)
  base_token_history_update_relation r.initialize_3 r.initialize_4 r.position_00_final_0 r.position_00_argmax_0 (S.length r.input1) 0 r.position_00_update_0 r.position_00_update_1

(* Program outputs, as bound by the plan. *)
let output0 (r:run) = (if loop2_count r > 0 then (r.loop2_position_i_update_0 (loop2_count r - 1)) else (r.loop1_position_i_update_0 (loop1_count r - 1)))
let output1 (r:run) = (if loop2_count r > 0 then (r.loop2_position_i_update_1 (loop2_count r - 1)) else (r.loop1_position_i_update_1 (loop1_count r - 1)))
