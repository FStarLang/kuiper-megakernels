module KPlay.Proof.Wiring.Base

(* GENERATED from the accepted decomposition plan (base); do not edit.
   One record field per program input and per family-call output; family_ok
   is the conjunction of the SharedSpec family relations over the exact
   arguments of each call, in host order. Tensors are flat row-major
   sequences; a view is the flat slice [offset, offset + numel). *)

open KPlay.SharedSpec
module S = FStar.Seq
module BF = Kuiper.BFloat16
module I64 = FStar.Int64

let view (#a:Type) (s:S.seq a) (offset len:nat) : S.seq a =
  if offset + len <= S.length s then S.slice s offset (offset + len) else S.empty

noeq type run = {
  input0 : S.seq BF.t;
  input1 : S.seq I64.t;
  input2 : int;
  embedding_setup_0 : S.seq BF.t;
  embedding_setup_1 : S.seq BF.t;
  embedding_setup_2 : S.seq BF.t;
  decoder_00_0 : S.seq BF.t;
  decoder_00_1 : S.seq BF.t;
  decoder_01_0 : S.seq BF.t;
  decoder_01_1 : S.seq BF.t;
  decoder_02_0 : S.seq BF.t;
  decoder_02_1 : S.seq BF.t;
  decoder_03_0 : S.seq BF.t;
  decoder_03_1 : S.seq BF.t;
  decoder_04_0 : S.seq BF.t;
  decoder_04_1 : S.seq BF.t;
  decoder_05_0 : S.seq BF.t;
  decoder_05_1 : S.seq BF.t;
  decoder_06_0 : S.seq BF.t;
  decoder_06_1 : S.seq BF.t;
  decoder_07_0 : S.seq BF.t;
  decoder_07_1 : S.seq BF.t;
  decoder_08_0 : S.seq BF.t;
  decoder_08_1 : S.seq BF.t;
  decoder_09_0 : S.seq BF.t;
  decoder_09_1 : S.seq BF.t;
  decoder_10_0 : S.seq BF.t;
  decoder_10_1 : S.seq BF.t;
  decoder_11_0 : S.seq BF.t;
  decoder_11_1 : S.seq BF.t;
  decoder_12_0 : S.seq BF.t;
  decoder_12_1 : S.seq BF.t;
  decoder_13_0 : S.seq BF.t;
  decoder_13_1 : S.seq BF.t;
  decoder_14_0 : S.seq BF.t;
  decoder_14_1 : S.seq BF.t;
  decoder_15_0 : S.seq BF.t;
  decoder_15_1 : S.seq BF.t;
  decoder_16_0 : S.seq BF.t;
  decoder_16_1 : S.seq BF.t;
  decoder_17_0 : S.seq BF.t;
  decoder_17_1 : S.seq BF.t;
  decoder_18_0 : S.seq BF.t;
  decoder_18_1 : S.seq BF.t;
  decoder_19_0 : S.seq BF.t;
  decoder_19_1 : S.seq BF.t;
  decoder_20_0 : S.seq BF.t;
  decoder_20_1 : S.seq BF.t;
  decoder_21_0 : S.seq BF.t;
  decoder_21_1 : S.seq BF.t;
  decoder_22_0 : S.seq BF.t;
  decoder_22_1 : S.seq BF.t;
  decoder_23_0 : S.seq BF.t;
  decoder_23_1 : S.seq BF.t;
  decoder_24_0 : S.seq BF.t;
  decoder_24_1 : S.seq BF.t;
  decoder_25_0 : S.seq BF.t;
  decoder_25_1 : S.seq BF.t;
  decoder_26_0 : S.seq BF.t;
  decoder_26_1 : S.seq BF.t;
  decoder_27_0 : S.seq BF.t;
  decoder_27_1 : S.seq BF.t;
  final_projection_0 : S.seq BF.t;
  select_next_token_0 : S.seq I64.t;
  select_next_token_1 : S.seq I64.t;
}

let input_lengths (r:run) : prop =
  S.length r.input0 == 596049920 /\ S.length r.input1 == 7

let family_ok (r:run) : prop =
  (* embedding_setup *)
  base_embedding_relation r.input0 r.input1 0 155583488 0 r.embedding_setup_0 r.embedding_setup_1 r.embedding_setup_2 /\
  (* decoder_00 *)
  base_decoder_relation r.input0 r.embedding_setup_0 r.embedding_setup_1 r.embedding_setup_2 155583488 171314432 r.decoder_00_0 r.decoder_00_1 /\
  (* decoder_01 *)
  base_decoder_relation r.input0 r.decoder_00_0 r.decoder_00_1 r.embedding_setup_2 171314432 187045376 r.decoder_01_0 r.decoder_01_1 /\
  (* decoder_02 *)
  base_decoder_relation r.input0 r.decoder_01_0 r.decoder_01_1 r.embedding_setup_2 187045376 202776320 r.decoder_02_0 r.decoder_02_1 /\
  (* decoder_03 *)
  base_decoder_relation r.input0 r.decoder_02_0 r.decoder_02_1 r.embedding_setup_2 202776320 218507264 r.decoder_03_0 r.decoder_03_1 /\
  (* decoder_04 *)
  base_decoder_relation r.input0 r.decoder_03_0 r.decoder_03_1 r.embedding_setup_2 218507264 234238208 r.decoder_04_0 r.decoder_04_1 /\
  (* decoder_05 *)
  base_decoder_relation r.input0 r.decoder_04_0 r.decoder_04_1 r.embedding_setup_2 234238208 249969152 r.decoder_05_0 r.decoder_05_1 /\
  (* decoder_06 *)
  base_decoder_relation r.input0 r.decoder_05_0 r.decoder_05_1 r.embedding_setup_2 249969152 265700096 r.decoder_06_0 r.decoder_06_1 /\
  (* decoder_07 *)
  base_decoder_relation r.input0 r.decoder_06_0 r.decoder_06_1 r.embedding_setup_2 265700096 281431040 r.decoder_07_0 r.decoder_07_1 /\
  (* decoder_08 *)
  base_decoder_relation r.input0 r.decoder_07_0 r.decoder_07_1 r.embedding_setup_2 281431040 297161984 r.decoder_08_0 r.decoder_08_1 /\
  (* decoder_09 *)
  base_decoder_relation r.input0 r.decoder_08_0 r.decoder_08_1 r.embedding_setup_2 297161984 312892928 r.decoder_09_0 r.decoder_09_1 /\
  (* decoder_10 *)
  base_decoder_relation r.input0 r.decoder_09_0 r.decoder_09_1 r.embedding_setup_2 312892928 328623872 r.decoder_10_0 r.decoder_10_1 /\
  (* decoder_11 *)
  base_decoder_relation r.input0 r.decoder_10_0 r.decoder_10_1 r.embedding_setup_2 328623872 344354816 r.decoder_11_0 r.decoder_11_1 /\
  (* decoder_12 *)
  base_decoder_relation r.input0 r.decoder_11_0 r.decoder_11_1 r.embedding_setup_2 344354816 360085760 r.decoder_12_0 r.decoder_12_1 /\
  (* decoder_13 *)
  base_decoder_relation r.input0 r.decoder_12_0 r.decoder_12_1 r.embedding_setup_2 360085760 375816704 r.decoder_13_0 r.decoder_13_1 /\
  (* decoder_14 *)
  base_decoder_relation r.input0 r.decoder_13_0 r.decoder_13_1 r.embedding_setup_2 375816704 391547648 r.decoder_14_0 r.decoder_14_1 /\
  (* decoder_15 *)
  base_decoder_relation r.input0 r.decoder_14_0 r.decoder_14_1 r.embedding_setup_2 391547648 407278592 r.decoder_15_0 r.decoder_15_1 /\
  (* decoder_16 *)
  base_decoder_relation r.input0 r.decoder_15_0 r.decoder_15_1 r.embedding_setup_2 407278592 423009536 r.decoder_16_0 r.decoder_16_1 /\
  (* decoder_17 *)
  base_decoder_relation r.input0 r.decoder_16_0 r.decoder_16_1 r.embedding_setup_2 423009536 438740480 r.decoder_17_0 r.decoder_17_1 /\
  (* decoder_18 *)
  base_decoder_relation r.input0 r.decoder_17_0 r.decoder_17_1 r.embedding_setup_2 438740480 454471424 r.decoder_18_0 r.decoder_18_1 /\
  (* decoder_19 *)
  base_decoder_relation r.input0 r.decoder_18_0 r.decoder_18_1 r.embedding_setup_2 454471424 470202368 r.decoder_19_0 r.decoder_19_1 /\
  (* decoder_20 *)
  base_decoder_relation r.input0 r.decoder_19_0 r.decoder_19_1 r.embedding_setup_2 470202368 485933312 r.decoder_20_0 r.decoder_20_1 /\
  (* decoder_21 *)
  base_decoder_relation r.input0 r.decoder_20_0 r.decoder_20_1 r.embedding_setup_2 485933312 501664256 r.decoder_21_0 r.decoder_21_1 /\
  (* decoder_22 *)
  base_decoder_relation r.input0 r.decoder_21_0 r.decoder_21_1 r.embedding_setup_2 501664256 517395200 r.decoder_22_0 r.decoder_22_1 /\
  (* decoder_23 *)
  base_decoder_relation r.input0 r.decoder_22_0 r.decoder_22_1 r.embedding_setup_2 517395200 533126144 r.decoder_23_0 r.decoder_23_1 /\
  (* decoder_24 *)
  base_decoder_relation r.input0 r.decoder_23_0 r.decoder_23_1 r.embedding_setup_2 533126144 548857088 r.decoder_24_0 r.decoder_24_1 /\
  (* decoder_25 *)
  base_decoder_relation r.input0 r.decoder_24_0 r.decoder_24_1 r.embedding_setup_2 548857088 564588032 r.decoder_25_0 r.decoder_25_1 /\
  (* decoder_26 *)
  base_decoder_relation r.input0 r.decoder_25_0 r.decoder_25_1 r.embedding_setup_2 564588032 580318976 r.decoder_26_0 r.decoder_26_1 /\
  (* decoder_27 *)
  base_decoder_relation r.input0 r.decoder_26_0 r.decoder_26_1 r.embedding_setup_2 580318976 155582464 r.decoder_27_0 r.decoder_27_1 /\
  (* final_projection *)
  base_projection_relation r.input0 r.decoder_27_1 0 r.final_projection_0 /\
  (* select_next_token *)
  base_argmax_relation r.input1 (view r.final_projection_0 911616 151936) r.input2 r.select_next_token_0 r.select_next_token_1

(* Program outputs, as bound by the plan. *)
let output0 (r:run) = r.select_next_token_0
let output1 (r:run) = r.decoder_27_1
