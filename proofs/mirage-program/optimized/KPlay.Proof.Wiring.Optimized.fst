module KPlay.Proof.Wiring.Optimized

(* GENERATED from the accepted decomposition plan (optimized); do not edit.
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
  s0000_embedding_0 : S.seq BF.t;
  s0000_embedding_1 : S.seq BF.t;
  s0000_embedding_2 : S.seq BF.t;
  s0000_embedding_3 : S.seq BF.t;
  s0000_decoder_00_0 : S.seq BF.t;
  s0000_decoder_00_1 : S.seq BF.t;
  s0000_decoder_01_0 : S.seq BF.t;
  s0000_decoder_01_1 : S.seq BF.t;
  s0000_decoder_02_0 : S.seq BF.t;
  s0000_decoder_02_1 : S.seq BF.t;
  s0000_decoder_03_0 : S.seq BF.t;
  s0000_decoder_03_1 : S.seq BF.t;
  s0000_decoder_04_0 : S.seq BF.t;
  s0000_decoder_04_1 : S.seq BF.t;
  s0000_decoder_05_0 : S.seq BF.t;
  s0000_decoder_05_1 : S.seq BF.t;
  s0000_decoder_06_0 : S.seq BF.t;
  s0000_decoder_06_1 : S.seq BF.t;
  s0000_decoder_07_0 : S.seq BF.t;
  s0000_decoder_07_1 : S.seq BF.t;
  s0000_decoder_08_0 : S.seq BF.t;
  s0000_decoder_08_1 : S.seq BF.t;
  s0000_decoder_09_0 : S.seq BF.t;
  s0000_decoder_09_1 : S.seq BF.t;
  s0000_decoder_10_0 : S.seq BF.t;
  s0000_decoder_10_1 : S.seq BF.t;
  s0000_decoder_11_0 : S.seq BF.t;
  s0000_decoder_11_1 : S.seq BF.t;
  s0000_decoder_12_0 : S.seq BF.t;
  s0000_decoder_12_1 : S.seq BF.t;
  s0000_decoder_13_0 : S.seq BF.t;
  s0000_decoder_13_1 : S.seq BF.t;
  s0000_decoder_14_0 : S.seq BF.t;
  s0000_decoder_14_1 : S.seq BF.t;
  s0000_decoder_15_0 : S.seq BF.t;
  s0000_decoder_15_1 : S.seq BF.t;
  s0000_decoder_16_0 : S.seq BF.t;
  s0000_decoder_16_1 : S.seq BF.t;
  s0000_decoder_17_0 : S.seq BF.t;
  s0000_decoder_17_1 : S.seq BF.t;
  s0000_decoder_18_0 : S.seq BF.t;
  s0000_decoder_18_1 : S.seq BF.t;
  s0000_decoder_19_0 : S.seq BF.t;
  s0000_decoder_19_1 : S.seq BF.t;
  s0000_decoder_20_0 : S.seq BF.t;
  s0000_decoder_20_1 : S.seq BF.t;
  s0000_decoder_21_0 : S.seq BF.t;
  s0000_decoder_21_1 : S.seq BF.t;
  s0000_decoder_22_0 : S.seq BF.t;
  s0000_decoder_22_1 : S.seq BF.t;
  s0000_decoder_23_0 : S.seq BF.t;
  s0000_decoder_23_1 : S.seq BF.t;
  s0000_decoder_24_0 : S.seq BF.t;
  s0000_decoder_24_1 : S.seq BF.t;
  s0000_decoder_25_0 : S.seq BF.t;
  s0000_decoder_25_1 : S.seq BF.t;
  s0000_decoder_26_0 : S.seq BF.t;
  s0000_decoder_26_1 : S.seq BF.t;
  s0000_decoder_27_0 : S.seq BF.t;
  s0000_decoder_27_1 : S.seq BF.t;
  s0000_projection_0 : S.seq BF.t;
  s0000_projection_1 : S.seq BF.t;
  s0000_argmax_0 : S.seq I64.t;
  s0000_argmax_1 : S.seq I64.t;
}

let input_lengths (r:run) : prop =
  S.length r.input0 == 596049920 /\ S.length r.input1 == 7

let family_ok (r:run) : prop =
  (* s0000_embedding *)
  optimized_embedding_relation (view r.input0 0 155582464) r.input1 (view r.input0 155583488 1024) r.s0000_embedding_0 r.s0000_embedding_1 r.s0000_embedding_2 r.s0000_embedding_3 /\
  (* s0000_decoder_00 *)
  optimized_decoder_relation (view r.input0 155585536 4194304) (view r.input0 161876992 256) (view r.input0 159779840 2097152) (view r.input0 155584512 1024) (view r.input0 161877248 6291456) (view r.input0 168168704 3145728) (view r.input0 171314432 1024) r.s0000_embedding_0 r.s0000_embedding_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_00_0 r.s0000_decoder_00_1 /\
  (* s0000_decoder_01 *)
  optimized_decoder_relation (view r.input0 171316480 4194304) (view r.input0 177607936 256) (view r.input0 175510784 2097152) (view r.input0 171315456 1024) (view r.input0 177608192 6291456) (view r.input0 183899648 3145728) (view r.input0 187045376 1024) r.s0000_decoder_00_0 r.s0000_decoder_00_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_01_0 r.s0000_decoder_01_1 /\
  (* s0000_decoder_02 *)
  optimized_decoder_relation (view r.input0 187047424 4194304) (view r.input0 193338880 256) (view r.input0 191241728 2097152) (view r.input0 187046400 1024) (view r.input0 193339136 6291456) (view r.input0 199630592 3145728) (view r.input0 202776320 1024) r.s0000_decoder_01_0 r.s0000_decoder_01_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_02_0 r.s0000_decoder_02_1 /\
  (* s0000_decoder_03 *)
  optimized_decoder_relation (view r.input0 202778368 4194304) (view r.input0 209069824 256) (view r.input0 206972672 2097152) (view r.input0 202777344 1024) (view r.input0 209070080 6291456) (view r.input0 215361536 3145728) (view r.input0 218507264 1024) r.s0000_decoder_02_0 r.s0000_decoder_02_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_03_0 r.s0000_decoder_03_1 /\
  (* s0000_decoder_04 *)
  optimized_decoder_relation (view r.input0 218509312 4194304) (view r.input0 224800768 256) (view r.input0 222703616 2097152) (view r.input0 218508288 1024) (view r.input0 224801024 6291456) (view r.input0 231092480 3145728) (view r.input0 234238208 1024) r.s0000_decoder_03_0 r.s0000_decoder_03_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_04_0 r.s0000_decoder_04_1 /\
  (* s0000_decoder_05 *)
  optimized_decoder_relation (view r.input0 234240256 4194304) (view r.input0 240531712 256) (view r.input0 238434560 2097152) (view r.input0 234239232 1024) (view r.input0 240531968 6291456) (view r.input0 246823424 3145728) (view r.input0 249969152 1024) r.s0000_decoder_04_0 r.s0000_decoder_04_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_05_0 r.s0000_decoder_05_1 /\
  (* s0000_decoder_06 *)
  optimized_decoder_relation (view r.input0 249971200 4194304) (view r.input0 256262656 256) (view r.input0 254165504 2097152) (view r.input0 249970176 1024) (view r.input0 256262912 6291456) (view r.input0 262554368 3145728) (view r.input0 265700096 1024) r.s0000_decoder_05_0 r.s0000_decoder_05_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_06_0 r.s0000_decoder_06_1 /\
  (* s0000_decoder_07 *)
  optimized_decoder_relation (view r.input0 265702144 4194304) (view r.input0 271993600 256) (view r.input0 269896448 2097152) (view r.input0 265701120 1024) (view r.input0 271993856 6291456) (view r.input0 278285312 3145728) (view r.input0 281431040 1024) r.s0000_decoder_06_0 r.s0000_decoder_06_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_07_0 r.s0000_decoder_07_1 /\
  (* s0000_decoder_08 *)
  optimized_decoder_relation (view r.input0 281433088 4194304) (view r.input0 287724544 256) (view r.input0 285627392 2097152) (view r.input0 281432064 1024) (view r.input0 287724800 6291456) (view r.input0 294016256 3145728) (view r.input0 297161984 1024) r.s0000_decoder_07_0 r.s0000_decoder_07_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_08_0 r.s0000_decoder_08_1 /\
  (* s0000_decoder_09 *)
  optimized_decoder_relation (view r.input0 297164032 4194304) (view r.input0 303455488 256) (view r.input0 301358336 2097152) (view r.input0 297163008 1024) (view r.input0 303455744 6291456) (view r.input0 309747200 3145728) (view r.input0 312892928 1024) r.s0000_decoder_08_0 r.s0000_decoder_08_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_09_0 r.s0000_decoder_09_1 /\
  (* s0000_decoder_10 *)
  optimized_decoder_relation (view r.input0 312894976 4194304) (view r.input0 319186432 256) (view r.input0 317089280 2097152) (view r.input0 312893952 1024) (view r.input0 319186688 6291456) (view r.input0 325478144 3145728) (view r.input0 328623872 1024) r.s0000_decoder_09_0 r.s0000_decoder_09_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_10_0 r.s0000_decoder_10_1 /\
  (* s0000_decoder_11 *)
  optimized_decoder_relation (view r.input0 328625920 4194304) (view r.input0 334917376 256) (view r.input0 332820224 2097152) (view r.input0 328624896 1024) (view r.input0 334917632 6291456) (view r.input0 341209088 3145728) (view r.input0 344354816 1024) r.s0000_decoder_10_0 r.s0000_decoder_10_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_11_0 r.s0000_decoder_11_1 /\
  (* s0000_decoder_12 *)
  optimized_decoder_relation (view r.input0 344356864 4194304) (view r.input0 350648320 256) (view r.input0 348551168 2097152) (view r.input0 344355840 1024) (view r.input0 350648576 6291456) (view r.input0 356940032 3145728) (view r.input0 360085760 1024) r.s0000_decoder_11_0 r.s0000_decoder_11_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_12_0 r.s0000_decoder_12_1 /\
  (* s0000_decoder_13 *)
  optimized_decoder_relation (view r.input0 360087808 4194304) (view r.input0 366379264 256) (view r.input0 364282112 2097152) (view r.input0 360086784 1024) (view r.input0 366379520 6291456) (view r.input0 372670976 3145728) (view r.input0 375816704 1024) r.s0000_decoder_12_0 r.s0000_decoder_12_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_13_0 r.s0000_decoder_13_1 /\
  (* s0000_decoder_14 *)
  optimized_decoder_relation (view r.input0 375818752 4194304) (view r.input0 382110208 256) (view r.input0 380013056 2097152) (view r.input0 375817728 1024) (view r.input0 382110464 6291456) (view r.input0 388401920 3145728) (view r.input0 391547648 1024) r.s0000_decoder_13_0 r.s0000_decoder_13_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_14_0 r.s0000_decoder_14_1 /\
  (* s0000_decoder_15 *)
  optimized_decoder_relation (view r.input0 391549696 4194304) (view r.input0 397841152 256) (view r.input0 395744000 2097152) (view r.input0 391548672 1024) (view r.input0 397841408 6291456) (view r.input0 404132864 3145728) (view r.input0 407278592 1024) r.s0000_decoder_14_0 r.s0000_decoder_14_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_15_0 r.s0000_decoder_15_1 /\
  (* s0000_decoder_16 *)
  optimized_decoder_relation (view r.input0 407280640 4194304) (view r.input0 413572096 256) (view r.input0 411474944 2097152) (view r.input0 407279616 1024) (view r.input0 413572352 6291456) (view r.input0 419863808 3145728) (view r.input0 423009536 1024) r.s0000_decoder_15_0 r.s0000_decoder_15_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_16_0 r.s0000_decoder_16_1 /\
  (* s0000_decoder_17 *)
  optimized_decoder_relation (view r.input0 423011584 4194304) (view r.input0 429303040 256) (view r.input0 427205888 2097152) (view r.input0 423010560 1024) (view r.input0 429303296 6291456) (view r.input0 435594752 3145728) (view r.input0 438740480 1024) r.s0000_decoder_16_0 r.s0000_decoder_16_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_17_0 r.s0000_decoder_17_1 /\
  (* s0000_decoder_18 *)
  optimized_decoder_relation (view r.input0 438742528 4194304) (view r.input0 445033984 256) (view r.input0 442936832 2097152) (view r.input0 438741504 1024) (view r.input0 445034240 6291456) (view r.input0 451325696 3145728) (view r.input0 454471424 1024) r.s0000_decoder_17_0 r.s0000_decoder_17_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_18_0 r.s0000_decoder_18_1 /\
  (* s0000_decoder_19 *)
  optimized_decoder_relation (view r.input0 454473472 4194304) (view r.input0 460764928 256) (view r.input0 458667776 2097152) (view r.input0 454472448 1024) (view r.input0 460765184 6291456) (view r.input0 467056640 3145728) (view r.input0 470202368 1024) r.s0000_decoder_18_0 r.s0000_decoder_18_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_19_0 r.s0000_decoder_19_1 /\
  (* s0000_decoder_20 *)
  optimized_decoder_relation (view r.input0 470204416 4194304) (view r.input0 476495872 256) (view r.input0 474398720 2097152) (view r.input0 470203392 1024) (view r.input0 476496128 6291456) (view r.input0 482787584 3145728) (view r.input0 485933312 1024) r.s0000_decoder_19_0 r.s0000_decoder_19_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_20_0 r.s0000_decoder_20_1 /\
  (* s0000_decoder_21 *)
  optimized_decoder_relation (view r.input0 485935360 4194304) (view r.input0 492226816 256) (view r.input0 490129664 2097152) (view r.input0 485934336 1024) (view r.input0 492227072 6291456) (view r.input0 498518528 3145728) (view r.input0 501664256 1024) r.s0000_decoder_20_0 r.s0000_decoder_20_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_21_0 r.s0000_decoder_21_1 /\
  (* s0000_decoder_22 *)
  optimized_decoder_relation (view r.input0 501666304 4194304) (view r.input0 507957760 256) (view r.input0 505860608 2097152) (view r.input0 501665280 1024) (view r.input0 507958016 6291456) (view r.input0 514249472 3145728) (view r.input0 517395200 1024) r.s0000_decoder_21_0 r.s0000_decoder_21_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_22_0 r.s0000_decoder_22_1 /\
  (* s0000_decoder_23 *)
  optimized_decoder_relation (view r.input0 517397248 4194304) (view r.input0 523688704 256) (view r.input0 521591552 2097152) (view r.input0 517396224 1024) (view r.input0 523688960 6291456) (view r.input0 529980416 3145728) (view r.input0 533126144 1024) r.s0000_decoder_22_0 r.s0000_decoder_22_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_23_0 r.s0000_decoder_23_1 /\
  (* s0000_decoder_24 *)
  optimized_decoder_relation (view r.input0 533128192 4194304) (view r.input0 539419648 256) (view r.input0 537322496 2097152) (view r.input0 533127168 1024) (view r.input0 539419904 6291456) (view r.input0 545711360 3145728) (view r.input0 548857088 1024) r.s0000_decoder_23_0 r.s0000_decoder_23_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_24_0 r.s0000_decoder_24_1 /\
  (* s0000_decoder_25 *)
  optimized_decoder_relation (view r.input0 548859136 4194304) (view r.input0 555150592 256) (view r.input0 553053440 2097152) (view r.input0 548858112 1024) (view r.input0 555150848 6291456) (view r.input0 561442304 3145728) (view r.input0 564588032 1024) r.s0000_decoder_24_0 r.s0000_decoder_24_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_25_0 r.s0000_decoder_25_1 /\
  (* s0000_decoder_26 *)
  optimized_decoder_relation (view r.input0 564590080 4194304) (view r.input0 570881536 256) (view r.input0 568784384 2097152) (view r.input0 564589056 1024) (view r.input0 570881792 6291456) (view r.input0 577173248 3145728) (view r.input0 580318976 1024) r.s0000_decoder_25_0 r.s0000_decoder_25_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_26_0 r.s0000_decoder_26_1 /\
  (* s0000_decoder_27 *)
  optimized_decoder_relation (view r.input0 580321024 4194304) (view r.input0 586612480 256) (view r.input0 584515328 2097152) (view r.input0 580320000 1024) (view r.input0 586612736 6291456) (view r.input0 592904192 3145728) (view r.input0 155582464 1024) r.s0000_decoder_26_0 r.s0000_decoder_26_1 r.s0000_embedding_2 r.s0000_embedding_3 0 r.s0000_decoder_27_0 r.s0000_decoder_27_1 /\
  (* s0000_projection *)
  optimized_projection_relation (view r.input0 0 155582464) r.s0000_decoder_27_1 r.s0000_projection_0 r.s0000_projection_1 /\
  (* s0000_argmax *)
  optimized_argmax_relation (view r.s0000_projection_0 921600 151936) r.input1 r.s0000_argmax_0 r.s0000_argmax_1

(* Program outputs, as bound by the plan. *)
let output0 (r:run) = r.s0000_argmax_1
let output1 (r:run) = r.s0000_projection_1
