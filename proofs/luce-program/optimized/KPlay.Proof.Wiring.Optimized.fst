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
module I64 = FStar.Int64

let view (#a:Type) (s:S.seq a) (offset len:int) : S.seq a =
  if 0 <= offset && 0 <= len && offset + len <= S.length s then S.slice s offset (offset + len) else S.empty

noeq type run = {
  input0 : S.seq BF.t;
  input1 : S.seq I64.t;
  input2 : int;
  call0_0_initialize_0 : S.seq F32.t;
  call0_0_initialize_1 : S.seq F32.t;
  call0_0_initialize_2 : S.seq BF.t;
  call0_0_initialize_3 : S.seq BF.t;
  call0_0_initialize_4 : S.seq I64.t;
  call0_0_initialize_5 : S.seq BF.t;
  loop0_s0000_pi_embedding_0 : int -> S.seq F32.t;
  loop0_s0000_pi_embedding_1 : int -> S.seq F32.t;
  loop0_s0000_pi_embedding_2 : int -> S.seq F32.t;
  loop0_s0000_pi_layer00_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer00_1 : int -> S.seq F32.t;
  loop0_s0000_pi_layer00_2 : int -> S.seq F32.t;
  loop0_s0000_pi_layer01_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer01_1 : int -> S.seq F32.t;
  loop0_s0000_pi_layer01_2 : int -> S.seq F32.t;
  loop0_s0000_pi_layer02_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer02_1 : int -> S.seq F32.t;
  loop0_s0000_pi_layer02_2 : int -> S.seq F32.t;
  loop0_s0000_pi_layer03_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer03_1 : int -> S.seq BF.t;
  loop0_s0000_pi_layer03_2 : int -> S.seq BF.t;
  loop0_s0000_pi_layer04_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer04_1 : int -> S.seq F32.t;
  loop0_s0000_pi_layer04_2 : int -> S.seq F32.t;
  loop0_s0000_pi_layer05_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer05_1 : int -> S.seq F32.t;
  loop0_s0000_pi_layer05_2 : int -> S.seq F32.t;
  loop0_s0000_pi_layer06_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer06_1 : int -> S.seq F32.t;
  loop0_s0000_pi_layer06_2 : int -> S.seq F32.t;
  loop0_s0000_pi_layer07_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer07_1 : int -> S.seq BF.t;
  loop0_s0000_pi_layer07_2 : int -> S.seq BF.t;
  loop0_s0000_pi_layer08_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer08_1 : int -> S.seq F32.t;
  loop0_s0000_pi_layer08_2 : int -> S.seq F32.t;
  loop0_s0000_pi_layer09_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer09_1 : int -> S.seq F32.t;
  loop0_s0000_pi_layer09_2 : int -> S.seq F32.t;
  loop0_s0000_pi_layer10_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer10_1 : int -> S.seq F32.t;
  loop0_s0000_pi_layer10_2 : int -> S.seq F32.t;
  loop0_s0000_pi_layer11_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer11_1 : int -> S.seq BF.t;
  loop0_s0000_pi_layer11_2 : int -> S.seq BF.t;
  loop0_s0000_pi_layer12_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer12_1 : int -> S.seq F32.t;
  loop0_s0000_pi_layer12_2 : int -> S.seq F32.t;
  loop0_s0000_pi_layer13_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer13_1 : int -> S.seq F32.t;
  loop0_s0000_pi_layer13_2 : int -> S.seq F32.t;
  loop0_s0000_pi_layer14_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer14_1 : int -> S.seq F32.t;
  loop0_s0000_pi_layer14_2 : int -> S.seq F32.t;
  loop0_s0000_pi_layer15_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer15_1 : int -> S.seq BF.t;
  loop0_s0000_pi_layer15_2 : int -> S.seq BF.t;
  loop0_s0000_pi_layer16_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer16_1 : int -> S.seq F32.t;
  loop0_s0000_pi_layer16_2 : int -> S.seq F32.t;
  loop0_s0000_pi_layer17_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer17_1 : int -> S.seq F32.t;
  loop0_s0000_pi_layer17_2 : int -> S.seq F32.t;
  loop0_s0000_pi_layer18_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer18_1 : int -> S.seq F32.t;
  loop0_s0000_pi_layer18_2 : int -> S.seq F32.t;
  loop0_s0000_pi_layer19_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer19_1 : int -> S.seq BF.t;
  loop0_s0000_pi_layer19_2 : int -> S.seq BF.t;
  loop0_s0000_pi_layer20_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer20_1 : int -> S.seq F32.t;
  loop0_s0000_pi_layer20_2 : int -> S.seq F32.t;
  loop0_s0000_pi_layer21_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer21_1 : int -> S.seq F32.t;
  loop0_s0000_pi_layer21_2 : int -> S.seq F32.t;
  loop0_s0000_pi_layer22_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer22_1 : int -> S.seq F32.t;
  loop0_s0000_pi_layer22_2 : int -> S.seq F32.t;
  loop0_s0000_pi_layer23_0 : int -> S.seq F32.t;
  loop0_s0000_pi_layer23_1 : int -> S.seq BF.t;
  loop0_s0000_pi_layer23_2 : int -> S.seq BF.t;
  loop1_s0000_pi_embedding_0 : int -> S.seq F32.t;
  loop1_s0000_pi_embedding_1 : int -> S.seq F32.t;
  loop1_s0000_pi_embedding_2 : int -> S.seq F32.t;
  loop1_s0000_pi_layer00_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer00_1 : int -> S.seq F32.t;
  loop1_s0000_pi_layer00_2 : int -> S.seq F32.t;
  loop1_s0000_pi_layer01_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer01_1 : int -> S.seq F32.t;
  loop1_s0000_pi_layer01_2 : int -> S.seq F32.t;
  loop1_s0000_pi_layer02_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer02_1 : int -> S.seq F32.t;
  loop1_s0000_pi_layer02_2 : int -> S.seq F32.t;
  loop1_s0000_pi_layer03_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer03_1 : int -> S.seq BF.t;
  loop1_s0000_pi_layer03_2 : int -> S.seq BF.t;
  loop1_s0000_pi_layer04_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer04_1 : int -> S.seq F32.t;
  loop1_s0000_pi_layer04_2 : int -> S.seq F32.t;
  loop1_s0000_pi_layer05_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer05_1 : int -> S.seq F32.t;
  loop1_s0000_pi_layer05_2 : int -> S.seq F32.t;
  loop1_s0000_pi_layer06_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer06_1 : int -> S.seq F32.t;
  loop1_s0000_pi_layer06_2 : int -> S.seq F32.t;
  loop1_s0000_pi_layer07_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer07_1 : int -> S.seq BF.t;
  loop1_s0000_pi_layer07_2 : int -> S.seq BF.t;
  loop1_s0000_pi_layer08_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer08_1 : int -> S.seq F32.t;
  loop1_s0000_pi_layer08_2 : int -> S.seq F32.t;
  loop1_s0000_pi_layer09_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer09_1 : int -> S.seq F32.t;
  loop1_s0000_pi_layer09_2 : int -> S.seq F32.t;
  loop1_s0000_pi_layer10_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer10_1 : int -> S.seq F32.t;
  loop1_s0000_pi_layer10_2 : int -> S.seq F32.t;
  loop1_s0000_pi_layer11_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer11_1 : int -> S.seq BF.t;
  loop1_s0000_pi_layer11_2 : int -> S.seq BF.t;
  loop1_s0000_pi_layer12_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer12_1 : int -> S.seq F32.t;
  loop1_s0000_pi_layer12_2 : int -> S.seq F32.t;
  loop1_s0000_pi_layer13_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer13_1 : int -> S.seq F32.t;
  loop1_s0000_pi_layer13_2 : int -> S.seq F32.t;
  loop1_s0000_pi_layer14_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer14_1 : int -> S.seq F32.t;
  loop1_s0000_pi_layer14_2 : int -> S.seq F32.t;
  loop1_s0000_pi_layer15_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer15_1 : int -> S.seq BF.t;
  loop1_s0000_pi_layer15_2 : int -> S.seq BF.t;
  loop1_s0000_pi_layer16_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer16_1 : int -> S.seq F32.t;
  loop1_s0000_pi_layer16_2 : int -> S.seq F32.t;
  loop1_s0000_pi_layer17_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer17_1 : int -> S.seq F32.t;
  loop1_s0000_pi_layer17_2 : int -> S.seq F32.t;
  loop1_s0000_pi_layer18_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer18_1 : int -> S.seq F32.t;
  loop1_s0000_pi_layer18_2 : int -> S.seq F32.t;
  loop1_s0000_pi_layer19_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer19_1 : int -> S.seq BF.t;
  loop1_s0000_pi_layer19_2 : int -> S.seq BF.t;
  loop1_s0000_pi_layer20_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer20_1 : int -> S.seq F32.t;
  loop1_s0000_pi_layer20_2 : int -> S.seq F32.t;
  loop1_s0000_pi_layer21_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer21_1 : int -> S.seq F32.t;
  loop1_s0000_pi_layer21_2 : int -> S.seq F32.t;
  loop1_s0000_pi_layer22_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer22_1 : int -> S.seq F32.t;
  loop1_s0000_pi_layer22_2 : int -> S.seq F32.t;
  loop1_s0000_pi_layer23_0 : int -> S.seq F32.t;
  loop1_s0000_pi_layer23_1 : int -> S.seq BF.t;
  loop1_s0000_pi_layer23_2 : int -> S.seq BF.t;
  loop1_s0000_pi_projection_0 : int -> S.seq F32.t;
  loop1_s0000_pi_projection_1 : int -> S.seq BF.t;
  loop1_s0000_pi_argmax_0 : int -> S.seq I64.t;
  loop1_s0000_pi_argmax_1 : int -> S.seq I64.t;
  loop2_s0000_pi_inactive_0 : int -> S.seq F32.t;
  loop2_s0000_pi_inactive_1 : int -> S.seq F32.t;
  loop2_s0000_pi_inactive_2 : int -> S.seq BF.t;
  loop2_s0000_pi_inactive_3 : int -> S.seq BF.t;
  loop2_s0000_pi_inactive_4 : int -> S.seq I64.t;
  loop2_s0000_pi_inactive_5 : int -> S.seq BF.t;
}

let input_lengths (r:run) : prop =
  S.length r.input0 == 752393024 /\ 1 <= S.length r.input1 /\ S.length r.input1 <= 8 /\ 1 <= r.input2 /\ r.input2 <= 5

(* The repeated host calls of loop 0, iterations 0 .. loop0_count - 1. Loop-carried
   arguments of iteration 0 come from the preceding calls. *)
let loop0_count (r:run) : int = (S.length r.input1 - 1)

let loop0_ok (r:run) (i:nat) : prop =
  (* loop0_s0000_pi_embedding: host call 1 + 25 * i *)
  optimized_embedding_relation (view r.input0 0 254279680) r.input1 r.call0_0_initialize_4 (0 + 1 * i) (r.loop0_s0000_pi_embedding_0 i) (r.loop0_s0000_pi_embedding_1 i) (r.loop0_s0000_pi_embedding_2 i) /\
  (* loop0_s0000_pi_layer00: host call 2 + 25 * i *)
  optimized_recurrent_decoder_relation (view r.input0 254280704 21555360) (r.loop0_s0000_pi_embedding_0 i) (if i = 0 then r.call0_0_initialize_0 else r.loop0_s0000_pi_layer22_1 (i - 1)) (if i = 0 then r.call0_0_initialize_1 else r.loop0_s0000_pi_layer22_2 (i - 1)) 0 0 (r.loop0_s0000_pi_layer00_0 i) (r.loop0_s0000_pi_layer00_1 i) (r.loop0_s0000_pi_layer00_2 i) /\
  (* loop0_s0000_pi_layer01: host call 3 + 25 * i *)
  optimized_recurrent_decoder_relation (view r.input0 275836064 21555360) (r.loop0_s0000_pi_layer00_0 i) (r.loop0_s0000_pi_layer00_1 i) (r.loop0_s0000_pi_layer00_2 i) 0 1 (r.loop0_s0000_pi_layer01_0 i) (r.loop0_s0000_pi_layer01_1 i) (r.loop0_s0000_pi_layer01_2 i) /\
  (* loop0_s0000_pi_layer02: host call 4 + 25 * i *)
  optimized_recurrent_decoder_relation (view r.input0 297391424 21555360) (r.loop0_s0000_pi_layer01_0 i) (r.loop0_s0000_pi_layer01_1 i) (r.loop0_s0000_pi_layer01_2 i) 0 2 (r.loop0_s0000_pi_layer02_0 i) (r.loop0_s0000_pi_layer02_1 i) (r.loop0_s0000_pi_layer02_2 i) /\
  (* loop0_s0000_pi_layer03: host call 5 + 25 * i *)
  optimized_attention_decoder_relation (view r.input0 318946784 18352640) (r.loop0_s0000_pi_layer02_0 i) (if i = 0 then r.call0_0_initialize_2 else r.loop0_s0000_pi_layer23_1 (i - 1)) (if i = 0 then r.call0_0_initialize_3 else r.loop0_s0000_pi_layer23_2 (i - 1)) (r.loop0_s0000_pi_embedding_1 i) (r.loop0_s0000_pi_embedding_2 i) 0 0 (0 + 1 * i) (r.loop0_s0000_pi_layer03_0 i) (r.loop0_s0000_pi_layer03_1 i) (r.loop0_s0000_pi_layer03_2 i) /\
  (* loop0_s0000_pi_layer04: host call 6 + 25 * i *)
  optimized_recurrent_decoder_relation (view r.input0 337299424 21555360) (r.loop0_s0000_pi_layer03_0 i) (r.loop0_s0000_pi_layer02_1 i) (r.loop0_s0000_pi_layer02_2 i) 0 3 (r.loop0_s0000_pi_layer04_0 i) (r.loop0_s0000_pi_layer04_1 i) (r.loop0_s0000_pi_layer04_2 i) /\
  (* loop0_s0000_pi_layer05: host call 7 + 25 * i *)
  optimized_recurrent_decoder_relation (view r.input0 358854784 21555360) (r.loop0_s0000_pi_layer04_0 i) (r.loop0_s0000_pi_layer04_1 i) (r.loop0_s0000_pi_layer04_2 i) 0 4 (r.loop0_s0000_pi_layer05_0 i) (r.loop0_s0000_pi_layer05_1 i) (r.loop0_s0000_pi_layer05_2 i) /\
  (* loop0_s0000_pi_layer06: host call 8 + 25 * i *)
  optimized_recurrent_decoder_relation (view r.input0 380410144 21555360) (r.loop0_s0000_pi_layer05_0 i) (r.loop0_s0000_pi_layer05_1 i) (r.loop0_s0000_pi_layer05_2 i) 0 5 (r.loop0_s0000_pi_layer06_0 i) (r.loop0_s0000_pi_layer06_1 i) (r.loop0_s0000_pi_layer06_2 i) /\
  (* loop0_s0000_pi_layer07: host call 9 + 25 * i *)
  optimized_attention_decoder_relation (view r.input0 401965504 18352640) (r.loop0_s0000_pi_layer06_0 i) (r.loop0_s0000_pi_layer03_1 i) (r.loop0_s0000_pi_layer03_2 i) (r.loop0_s0000_pi_embedding_1 i) (r.loop0_s0000_pi_embedding_2 i) 0 1 (0 + 1 * i) (r.loop0_s0000_pi_layer07_0 i) (r.loop0_s0000_pi_layer07_1 i) (r.loop0_s0000_pi_layer07_2 i) /\
  (* loop0_s0000_pi_layer08: host call 10 + 25 * i *)
  optimized_recurrent_decoder_relation (view r.input0 420318144 21555360) (r.loop0_s0000_pi_layer07_0 i) (r.loop0_s0000_pi_layer06_1 i) (r.loop0_s0000_pi_layer06_2 i) 0 6 (r.loop0_s0000_pi_layer08_0 i) (r.loop0_s0000_pi_layer08_1 i) (r.loop0_s0000_pi_layer08_2 i) /\
  (* loop0_s0000_pi_layer09: host call 11 + 25 * i *)
  optimized_recurrent_decoder_relation (view r.input0 441873504 21555360) (r.loop0_s0000_pi_layer08_0 i) (r.loop0_s0000_pi_layer08_1 i) (r.loop0_s0000_pi_layer08_2 i) 0 7 (r.loop0_s0000_pi_layer09_0 i) (r.loop0_s0000_pi_layer09_1 i) (r.loop0_s0000_pi_layer09_2 i) /\
  (* loop0_s0000_pi_layer10: host call 12 + 25 * i *)
  optimized_recurrent_decoder_relation (view r.input0 463428864 21555360) (r.loop0_s0000_pi_layer09_0 i) (r.loop0_s0000_pi_layer09_1 i) (r.loop0_s0000_pi_layer09_2 i) 0 8 (r.loop0_s0000_pi_layer10_0 i) (r.loop0_s0000_pi_layer10_1 i) (r.loop0_s0000_pi_layer10_2 i) /\
  (* loop0_s0000_pi_layer11: host call 13 + 25 * i *)
  optimized_attention_decoder_relation (view r.input0 484984224 18352640) (r.loop0_s0000_pi_layer10_0 i) (r.loop0_s0000_pi_layer07_1 i) (r.loop0_s0000_pi_layer07_2 i) (r.loop0_s0000_pi_embedding_1 i) (r.loop0_s0000_pi_embedding_2 i) 0 2 (0 + 1 * i) (r.loop0_s0000_pi_layer11_0 i) (r.loop0_s0000_pi_layer11_1 i) (r.loop0_s0000_pi_layer11_2 i) /\
  (* loop0_s0000_pi_layer12: host call 14 + 25 * i *)
  optimized_recurrent_decoder_relation (view r.input0 503336864 21555360) (r.loop0_s0000_pi_layer11_0 i) (r.loop0_s0000_pi_layer10_1 i) (r.loop0_s0000_pi_layer10_2 i) 0 9 (r.loop0_s0000_pi_layer12_0 i) (r.loop0_s0000_pi_layer12_1 i) (r.loop0_s0000_pi_layer12_2 i) /\
  (* loop0_s0000_pi_layer13: host call 15 + 25 * i *)
  optimized_recurrent_decoder_relation (view r.input0 524892224 21555360) (r.loop0_s0000_pi_layer12_0 i) (r.loop0_s0000_pi_layer12_1 i) (r.loop0_s0000_pi_layer12_2 i) 0 10 (r.loop0_s0000_pi_layer13_0 i) (r.loop0_s0000_pi_layer13_1 i) (r.loop0_s0000_pi_layer13_2 i) /\
  (* loop0_s0000_pi_layer14: host call 16 + 25 * i *)
  optimized_recurrent_decoder_relation (view r.input0 546447584 21555360) (r.loop0_s0000_pi_layer13_0 i) (r.loop0_s0000_pi_layer13_1 i) (r.loop0_s0000_pi_layer13_2 i) 0 11 (r.loop0_s0000_pi_layer14_0 i) (r.loop0_s0000_pi_layer14_1 i) (r.loop0_s0000_pi_layer14_2 i) /\
  (* loop0_s0000_pi_layer15: host call 17 + 25 * i *)
  optimized_attention_decoder_relation (view r.input0 568002944 18352640) (r.loop0_s0000_pi_layer14_0 i) (r.loop0_s0000_pi_layer11_1 i) (r.loop0_s0000_pi_layer11_2 i) (r.loop0_s0000_pi_embedding_1 i) (r.loop0_s0000_pi_embedding_2 i) 0 3 (0 + 1 * i) (r.loop0_s0000_pi_layer15_0 i) (r.loop0_s0000_pi_layer15_1 i) (r.loop0_s0000_pi_layer15_2 i) /\
  (* loop0_s0000_pi_layer16: host call 18 + 25 * i *)
  optimized_recurrent_decoder_relation (view r.input0 586355584 21555360) (r.loop0_s0000_pi_layer15_0 i) (r.loop0_s0000_pi_layer14_1 i) (r.loop0_s0000_pi_layer14_2 i) 0 12 (r.loop0_s0000_pi_layer16_0 i) (r.loop0_s0000_pi_layer16_1 i) (r.loop0_s0000_pi_layer16_2 i) /\
  (* loop0_s0000_pi_layer17: host call 19 + 25 * i *)
  optimized_recurrent_decoder_relation (view r.input0 607910944 21555360) (r.loop0_s0000_pi_layer16_0 i) (r.loop0_s0000_pi_layer16_1 i) (r.loop0_s0000_pi_layer16_2 i) 0 13 (r.loop0_s0000_pi_layer17_0 i) (r.loop0_s0000_pi_layer17_1 i) (r.loop0_s0000_pi_layer17_2 i) /\
  (* loop0_s0000_pi_layer18: host call 20 + 25 * i *)
  optimized_recurrent_decoder_relation (view r.input0 629466304 21555360) (r.loop0_s0000_pi_layer17_0 i) (r.loop0_s0000_pi_layer17_1 i) (r.loop0_s0000_pi_layer17_2 i) 0 14 (r.loop0_s0000_pi_layer18_0 i) (r.loop0_s0000_pi_layer18_1 i) (r.loop0_s0000_pi_layer18_2 i) /\
  (* loop0_s0000_pi_layer19: host call 21 + 25 * i *)
  optimized_attention_decoder_relation (view r.input0 651021664 18352640) (r.loop0_s0000_pi_layer18_0 i) (r.loop0_s0000_pi_layer15_1 i) (r.loop0_s0000_pi_layer15_2 i) (r.loop0_s0000_pi_embedding_1 i) (r.loop0_s0000_pi_embedding_2 i) 0 4 (0 + 1 * i) (r.loop0_s0000_pi_layer19_0 i) (r.loop0_s0000_pi_layer19_1 i) (r.loop0_s0000_pi_layer19_2 i) /\
  (* loop0_s0000_pi_layer20: host call 22 + 25 * i *)
  optimized_recurrent_decoder_relation (view r.input0 669374304 21555360) (r.loop0_s0000_pi_layer19_0 i) (r.loop0_s0000_pi_layer18_1 i) (r.loop0_s0000_pi_layer18_2 i) 0 15 (r.loop0_s0000_pi_layer20_0 i) (r.loop0_s0000_pi_layer20_1 i) (r.loop0_s0000_pi_layer20_2 i) /\
  (* loop0_s0000_pi_layer21: host call 23 + 25 * i *)
  optimized_recurrent_decoder_relation (view r.input0 690929664 21555360) (r.loop0_s0000_pi_layer20_0 i) (r.loop0_s0000_pi_layer20_1 i) (r.loop0_s0000_pi_layer20_2 i) 0 16 (r.loop0_s0000_pi_layer21_0 i) (r.loop0_s0000_pi_layer21_1 i) (r.loop0_s0000_pi_layer21_2 i) /\
  (* loop0_s0000_pi_layer22: host call 24 + 25 * i *)
  optimized_recurrent_decoder_relation (view r.input0 712485024 21555360) (r.loop0_s0000_pi_layer21_0 i) (r.loop0_s0000_pi_layer21_1 i) (r.loop0_s0000_pi_layer21_2 i) 0 17 (r.loop0_s0000_pi_layer22_0 i) (r.loop0_s0000_pi_layer22_1 i) (r.loop0_s0000_pi_layer22_2 i) /\
  (* loop0_s0000_pi_layer23: host call 25 + 25 * i *)
  optimized_attention_decoder_relation (view r.input0 734040384 18352640) (r.loop0_s0000_pi_layer22_0 i) (r.loop0_s0000_pi_layer19_1 i) (r.loop0_s0000_pi_layer19_2 i) (r.loop0_s0000_pi_embedding_1 i) (r.loop0_s0000_pi_embedding_2 i) 0 5 (0 + 1 * i) (r.loop0_s0000_pi_layer23_0 i) (r.loop0_s0000_pi_layer23_1 i) (r.loop0_s0000_pi_layer23_2 i)

(* The repeated host calls of loop 1, iterations 0 .. loop1_count - 1. Loop-carried
   arguments of iteration 0 come from the preceding calls. *)
let loop1_count (r:run) : int = (r.input2)

let loop1_ok (r:run) (i:nat) : prop =
  (* loop1_s0000_pi_embedding: host call (25 * S.length r.input1 - 24) + 27 * i *)
  optimized_embedding_relation (view r.input0 0 254279680) r.input1 (if i = 0 then r.call0_0_initialize_4 else r.loop1_s0000_pi_argmax_1 (i - 1)) ((S.length r.input1 - 1) + 1 * i) (r.loop1_s0000_pi_embedding_0 i) (r.loop1_s0000_pi_embedding_1 i) (r.loop1_s0000_pi_embedding_2 i) /\
  (* loop1_s0000_pi_layer00: host call (25 * S.length r.input1 - 23) + 27 * i *)
  optimized_recurrent_decoder_relation (view r.input0 254280704 21555360) (r.loop1_s0000_pi_embedding_0 i) (if i = 0 then (if loop0_count r > 0 then (r.loop0_s0000_pi_layer22_1 (loop0_count r - 1)) else r.call0_0_initialize_0) else r.loop1_s0000_pi_layer22_1 (i - 1)) (if i = 0 then (if loop0_count r > 0 then (r.loop0_s0000_pi_layer22_2 (loop0_count r - 1)) else r.call0_0_initialize_1) else r.loop1_s0000_pi_layer22_2 (i - 1)) 0 0 (r.loop1_s0000_pi_layer00_0 i) (r.loop1_s0000_pi_layer00_1 i) (r.loop1_s0000_pi_layer00_2 i) /\
  (* loop1_s0000_pi_layer01: host call (25 * S.length r.input1 - 22) + 27 * i *)
  optimized_recurrent_decoder_relation (view r.input0 275836064 21555360) (r.loop1_s0000_pi_layer00_0 i) (r.loop1_s0000_pi_layer00_1 i) (r.loop1_s0000_pi_layer00_2 i) 0 1 (r.loop1_s0000_pi_layer01_0 i) (r.loop1_s0000_pi_layer01_1 i) (r.loop1_s0000_pi_layer01_2 i) /\
  (* loop1_s0000_pi_layer02: host call (25 * S.length r.input1 - 21) + 27 * i *)
  optimized_recurrent_decoder_relation (view r.input0 297391424 21555360) (r.loop1_s0000_pi_layer01_0 i) (r.loop1_s0000_pi_layer01_1 i) (r.loop1_s0000_pi_layer01_2 i) 0 2 (r.loop1_s0000_pi_layer02_0 i) (r.loop1_s0000_pi_layer02_1 i) (r.loop1_s0000_pi_layer02_2 i) /\
  (* loop1_s0000_pi_layer03: host call (25 * S.length r.input1 - 20) + 27 * i *)
  optimized_attention_decoder_relation (view r.input0 318946784 18352640) (r.loop1_s0000_pi_layer02_0 i) (if i = 0 then (if loop0_count r > 0 then (r.loop0_s0000_pi_layer23_1 (loop0_count r - 1)) else r.call0_0_initialize_2) else r.loop1_s0000_pi_layer23_1 (i - 1)) (if i = 0 then (if loop0_count r > 0 then (r.loop0_s0000_pi_layer23_2 (loop0_count r - 1)) else r.call0_0_initialize_3) else r.loop1_s0000_pi_layer23_2 (i - 1)) (r.loop1_s0000_pi_embedding_1 i) (r.loop1_s0000_pi_embedding_2 i) 0 0 ((S.length r.input1 - 1) + 1 * i) (r.loop1_s0000_pi_layer03_0 i) (r.loop1_s0000_pi_layer03_1 i) (r.loop1_s0000_pi_layer03_2 i) /\
  (* loop1_s0000_pi_layer04: host call (25 * S.length r.input1 - 19) + 27 * i *)
  optimized_recurrent_decoder_relation (view r.input0 337299424 21555360) (r.loop1_s0000_pi_layer03_0 i) (r.loop1_s0000_pi_layer02_1 i) (r.loop1_s0000_pi_layer02_2 i) 0 3 (r.loop1_s0000_pi_layer04_0 i) (r.loop1_s0000_pi_layer04_1 i) (r.loop1_s0000_pi_layer04_2 i) /\
  (* loop1_s0000_pi_layer05: host call (25 * S.length r.input1 - 18) + 27 * i *)
  optimized_recurrent_decoder_relation (view r.input0 358854784 21555360) (r.loop1_s0000_pi_layer04_0 i) (r.loop1_s0000_pi_layer04_1 i) (r.loop1_s0000_pi_layer04_2 i) 0 4 (r.loop1_s0000_pi_layer05_0 i) (r.loop1_s0000_pi_layer05_1 i) (r.loop1_s0000_pi_layer05_2 i) /\
  (* loop1_s0000_pi_layer06: host call (25 * S.length r.input1 - 17) + 27 * i *)
  optimized_recurrent_decoder_relation (view r.input0 380410144 21555360) (r.loop1_s0000_pi_layer05_0 i) (r.loop1_s0000_pi_layer05_1 i) (r.loop1_s0000_pi_layer05_2 i) 0 5 (r.loop1_s0000_pi_layer06_0 i) (r.loop1_s0000_pi_layer06_1 i) (r.loop1_s0000_pi_layer06_2 i) /\
  (* loop1_s0000_pi_layer07: host call (25 * S.length r.input1 - 16) + 27 * i *)
  optimized_attention_decoder_relation (view r.input0 401965504 18352640) (r.loop1_s0000_pi_layer06_0 i) (r.loop1_s0000_pi_layer03_1 i) (r.loop1_s0000_pi_layer03_2 i) (r.loop1_s0000_pi_embedding_1 i) (r.loop1_s0000_pi_embedding_2 i) 0 1 ((S.length r.input1 - 1) + 1 * i) (r.loop1_s0000_pi_layer07_0 i) (r.loop1_s0000_pi_layer07_1 i) (r.loop1_s0000_pi_layer07_2 i) /\
  (* loop1_s0000_pi_layer08: host call (25 * S.length r.input1 - 15) + 27 * i *)
  optimized_recurrent_decoder_relation (view r.input0 420318144 21555360) (r.loop1_s0000_pi_layer07_0 i) (r.loop1_s0000_pi_layer06_1 i) (r.loop1_s0000_pi_layer06_2 i) 0 6 (r.loop1_s0000_pi_layer08_0 i) (r.loop1_s0000_pi_layer08_1 i) (r.loop1_s0000_pi_layer08_2 i) /\
  (* loop1_s0000_pi_layer09: host call (25 * S.length r.input1 - 14) + 27 * i *)
  optimized_recurrent_decoder_relation (view r.input0 441873504 21555360) (r.loop1_s0000_pi_layer08_0 i) (r.loop1_s0000_pi_layer08_1 i) (r.loop1_s0000_pi_layer08_2 i) 0 7 (r.loop1_s0000_pi_layer09_0 i) (r.loop1_s0000_pi_layer09_1 i) (r.loop1_s0000_pi_layer09_2 i) /\
  (* loop1_s0000_pi_layer10: host call (25 * S.length r.input1 - 13) + 27 * i *)
  optimized_recurrent_decoder_relation (view r.input0 463428864 21555360) (r.loop1_s0000_pi_layer09_0 i) (r.loop1_s0000_pi_layer09_1 i) (r.loop1_s0000_pi_layer09_2 i) 0 8 (r.loop1_s0000_pi_layer10_0 i) (r.loop1_s0000_pi_layer10_1 i) (r.loop1_s0000_pi_layer10_2 i) /\
  (* loop1_s0000_pi_layer11: host call (25 * S.length r.input1 - 12) + 27 * i *)
  optimized_attention_decoder_relation (view r.input0 484984224 18352640) (r.loop1_s0000_pi_layer10_0 i) (r.loop1_s0000_pi_layer07_1 i) (r.loop1_s0000_pi_layer07_2 i) (r.loop1_s0000_pi_embedding_1 i) (r.loop1_s0000_pi_embedding_2 i) 0 2 ((S.length r.input1 - 1) + 1 * i) (r.loop1_s0000_pi_layer11_0 i) (r.loop1_s0000_pi_layer11_1 i) (r.loop1_s0000_pi_layer11_2 i) /\
  (* loop1_s0000_pi_layer12: host call (25 * S.length r.input1 - 11) + 27 * i *)
  optimized_recurrent_decoder_relation (view r.input0 503336864 21555360) (r.loop1_s0000_pi_layer11_0 i) (r.loop1_s0000_pi_layer10_1 i) (r.loop1_s0000_pi_layer10_2 i) 0 9 (r.loop1_s0000_pi_layer12_0 i) (r.loop1_s0000_pi_layer12_1 i) (r.loop1_s0000_pi_layer12_2 i) /\
  (* loop1_s0000_pi_layer13: host call (25 * S.length r.input1 - 10) + 27 * i *)
  optimized_recurrent_decoder_relation (view r.input0 524892224 21555360) (r.loop1_s0000_pi_layer12_0 i) (r.loop1_s0000_pi_layer12_1 i) (r.loop1_s0000_pi_layer12_2 i) 0 10 (r.loop1_s0000_pi_layer13_0 i) (r.loop1_s0000_pi_layer13_1 i) (r.loop1_s0000_pi_layer13_2 i) /\
  (* loop1_s0000_pi_layer14: host call (25 * S.length r.input1 - 9) + 27 * i *)
  optimized_recurrent_decoder_relation (view r.input0 546447584 21555360) (r.loop1_s0000_pi_layer13_0 i) (r.loop1_s0000_pi_layer13_1 i) (r.loop1_s0000_pi_layer13_2 i) 0 11 (r.loop1_s0000_pi_layer14_0 i) (r.loop1_s0000_pi_layer14_1 i) (r.loop1_s0000_pi_layer14_2 i) /\
  (* loop1_s0000_pi_layer15: host call (25 * S.length r.input1 - 8) + 27 * i *)
  optimized_attention_decoder_relation (view r.input0 568002944 18352640) (r.loop1_s0000_pi_layer14_0 i) (r.loop1_s0000_pi_layer11_1 i) (r.loop1_s0000_pi_layer11_2 i) (r.loop1_s0000_pi_embedding_1 i) (r.loop1_s0000_pi_embedding_2 i) 0 3 ((S.length r.input1 - 1) + 1 * i) (r.loop1_s0000_pi_layer15_0 i) (r.loop1_s0000_pi_layer15_1 i) (r.loop1_s0000_pi_layer15_2 i) /\
  (* loop1_s0000_pi_layer16: host call (25 * S.length r.input1 - 7) + 27 * i *)
  optimized_recurrent_decoder_relation (view r.input0 586355584 21555360) (r.loop1_s0000_pi_layer15_0 i) (r.loop1_s0000_pi_layer14_1 i) (r.loop1_s0000_pi_layer14_2 i) 0 12 (r.loop1_s0000_pi_layer16_0 i) (r.loop1_s0000_pi_layer16_1 i) (r.loop1_s0000_pi_layer16_2 i) /\
  (* loop1_s0000_pi_layer17: host call (25 * S.length r.input1 - 6) + 27 * i *)
  optimized_recurrent_decoder_relation (view r.input0 607910944 21555360) (r.loop1_s0000_pi_layer16_0 i) (r.loop1_s0000_pi_layer16_1 i) (r.loop1_s0000_pi_layer16_2 i) 0 13 (r.loop1_s0000_pi_layer17_0 i) (r.loop1_s0000_pi_layer17_1 i) (r.loop1_s0000_pi_layer17_2 i) /\
  (* loop1_s0000_pi_layer18: host call (25 * S.length r.input1 - 5) + 27 * i *)
  optimized_recurrent_decoder_relation (view r.input0 629466304 21555360) (r.loop1_s0000_pi_layer17_0 i) (r.loop1_s0000_pi_layer17_1 i) (r.loop1_s0000_pi_layer17_2 i) 0 14 (r.loop1_s0000_pi_layer18_0 i) (r.loop1_s0000_pi_layer18_1 i) (r.loop1_s0000_pi_layer18_2 i) /\
  (* loop1_s0000_pi_layer19: host call (25 * S.length r.input1 - 4) + 27 * i *)
  optimized_attention_decoder_relation (view r.input0 651021664 18352640) (r.loop1_s0000_pi_layer18_0 i) (r.loop1_s0000_pi_layer15_1 i) (r.loop1_s0000_pi_layer15_2 i) (r.loop1_s0000_pi_embedding_1 i) (r.loop1_s0000_pi_embedding_2 i) 0 4 ((S.length r.input1 - 1) + 1 * i) (r.loop1_s0000_pi_layer19_0 i) (r.loop1_s0000_pi_layer19_1 i) (r.loop1_s0000_pi_layer19_2 i) /\
  (* loop1_s0000_pi_layer20: host call (25 * S.length r.input1 - 3) + 27 * i *)
  optimized_recurrent_decoder_relation (view r.input0 669374304 21555360) (r.loop1_s0000_pi_layer19_0 i) (r.loop1_s0000_pi_layer18_1 i) (r.loop1_s0000_pi_layer18_2 i) 0 15 (r.loop1_s0000_pi_layer20_0 i) (r.loop1_s0000_pi_layer20_1 i) (r.loop1_s0000_pi_layer20_2 i) /\
  (* loop1_s0000_pi_layer21: host call (25 * S.length r.input1 - 2) + 27 * i *)
  optimized_recurrent_decoder_relation (view r.input0 690929664 21555360) (r.loop1_s0000_pi_layer20_0 i) (r.loop1_s0000_pi_layer20_1 i) (r.loop1_s0000_pi_layer20_2 i) 0 16 (r.loop1_s0000_pi_layer21_0 i) (r.loop1_s0000_pi_layer21_1 i) (r.loop1_s0000_pi_layer21_2 i) /\
  (* loop1_s0000_pi_layer22: host call (25 * S.length r.input1 - 1) + 27 * i *)
  optimized_recurrent_decoder_relation (view r.input0 712485024 21555360) (r.loop1_s0000_pi_layer21_0 i) (r.loop1_s0000_pi_layer21_1 i) (r.loop1_s0000_pi_layer21_2 i) 0 17 (r.loop1_s0000_pi_layer22_0 i) (r.loop1_s0000_pi_layer22_1 i) (r.loop1_s0000_pi_layer22_2 i) /\
  (* loop1_s0000_pi_layer23: host call (25 * S.length r.input1) + 27 * i *)
  optimized_attention_decoder_relation (view r.input0 734040384 18352640) (r.loop1_s0000_pi_layer22_0 i) (r.loop1_s0000_pi_layer19_1 i) (r.loop1_s0000_pi_layer19_2 i) (r.loop1_s0000_pi_embedding_1 i) (r.loop1_s0000_pi_embedding_2 i) 0 5 ((S.length r.input1 - 1) + 1 * i) (r.loop1_s0000_pi_layer23_0 i) (r.loop1_s0000_pi_layer23_1 i) (r.loop1_s0000_pi_layer23_2 i) /\
  (* loop1_s0000_pi_projection: host call (25 * S.length r.input1 + 1) + 27 * i *)
  optimized_final_projection_relation (view r.input0 0 254279680) (view r.input0 254279680 1024) (r.loop1_s0000_pi_layer23_0 i) (if i = 0 then r.call0_0_initialize_5 else r.loop1_s0000_pi_projection_1 (i - 1)) (0 + 1 * i) (r.loop1_s0000_pi_projection_0 i) (r.loop1_s0000_pi_projection_1 i) /\
  (* loop1_s0000_pi_argmax: host call (25 * S.length r.input1 + 2) + 27 * i *)
  optimized_argmax_relation (r.loop1_s0000_pi_projection_0 i) (if i = 0 then r.call0_0_initialize_4 else r.loop1_s0000_pi_argmax_1 (i - 1)) (0 + 1 * i) (r.loop1_s0000_pi_argmax_0 i) (r.loop1_s0000_pi_argmax_1 i)

(* The repeated host calls of loop 2, iterations 0 .. loop2_count - 1. Loop-carried
   arguments of iteration 0 come from the preceding calls. *)
let loop2_count (r:run) : int = ((-1) * S.length r.input1 + (-1) * r.input2 + 13)

let loop2_ok (r:run) (i:nat) : prop =
  (* loop2_s0000_pi_inactive: host call (25 * S.length r.input1 + 27 * r.input2 - 24) + 1 * i *)
  optimized_inactive_decode_relation (if i = 0 then (r.loop1_s0000_pi_layer22_1 (r.input2 - 1)) else r.loop2_s0000_pi_inactive_0 (i - 1)) (if i = 0 then (r.loop1_s0000_pi_layer22_2 (r.input2 - 1)) else r.loop2_s0000_pi_inactive_1 (i - 1)) (if i = 0 then (r.loop1_s0000_pi_layer23_1 (r.input2 - 1)) else r.loop2_s0000_pi_inactive_2 (i - 1)) (if i = 0 then (r.loop1_s0000_pi_layer23_2 (r.input2 - 1)) else r.loop2_s0000_pi_inactive_3 (i - 1)) (if i = 0 then (r.loop1_s0000_pi_argmax_1 (r.input2 - 1)) else r.loop2_s0000_pi_inactive_4 (i - 1)) (if i = 0 then (r.loop1_s0000_pi_projection_1 (r.input2 - 1)) else r.loop2_s0000_pi_inactive_5 (i - 1)) ((S.length r.input1 + r.input2 - 1) + 1 * i) (S.length r.input1) (r.loop2_s0000_pi_inactive_0 i) (r.loop2_s0000_pi_inactive_1 i) (r.loop2_s0000_pi_inactive_2 i) (r.loop2_s0000_pi_inactive_3 i) (r.loop2_s0000_pi_inactive_4 i) (r.loop2_s0000_pi_inactive_5 i)

let family_ok (r:run) : prop =
  (* call0_0_initialize *)
  optimized_initialize_relation r.input1 r.input2 r.call0_0_initialize_0 r.call0_0_initialize_1 r.call0_0_initialize_2 r.call0_0_initialize_3 r.call0_0_initialize_4 r.call0_0_initialize_5 /\
  (forall (i:nat). i < loop0_count r ==> loop0_ok r i) /\
  (forall (i:nat). i < loop1_count r ==> loop1_ok r i) /\
  (forall (i:nat). i < loop2_count r ==> loop2_ok r i)

(* Program outputs, as bound by the plan. *)
let output0 (r:run) = (if loop2_count r > 0 then (r.loop2_s0000_pi_inactive_4 (loop2_count r - 1)) else (r.loop1_s0000_pi_argmax_1 (loop1_count r - 1)))
let output1 (r:run) = (if loop2_count r > 0 then (r.loop2_s0000_pi_inactive_5 (loop2_count r - 1)) else (r.loop1_s0000_pi_projection_1 (loop1_count r - 1)))
