module KPlay.Proof.Program.Base

(* Composition theorem for the base program: one size-generic chain lemma
   over the 28 decoder calls (G4), then projection and selection. *)

open FStar.Real
open Kuiper.Approximates
open KPlay.SharedSpec
open KPlay.Proof.Wiring.Base
module S = FStar.Seq
module BF = Kuiper.BFloat16
module I64 = FStar.Int64

(* Boundary values: residual and normalized stream after l decoder calls. *)
[@@"opaque_to_smt"]
let res (r:run) (l:nat{l <= 28}) : S.seq BF.t =
  match l with
  | 0 -> r.embedding_setup_0
  | 1 -> r.decoder_00_0
  | 2 -> r.decoder_01_0
  | 3 -> r.decoder_02_0
  | 4 -> r.decoder_03_0
  | 5 -> r.decoder_04_0
  | 6 -> r.decoder_05_0
  | 7 -> r.decoder_06_0
  | 8 -> r.decoder_07_0
  | 9 -> r.decoder_08_0
  | 10 -> r.decoder_09_0
  | 11 -> r.decoder_10_0
  | 12 -> r.decoder_11_0
  | 13 -> r.decoder_12_0
  | 14 -> r.decoder_13_0
  | 15 -> r.decoder_14_0
  | 16 -> r.decoder_15_0
  | 17 -> r.decoder_16_0
  | 18 -> r.decoder_17_0
  | 19 -> r.decoder_18_0
  | 20 -> r.decoder_19_0
  | 21 -> r.decoder_20_0
  | 22 -> r.decoder_21_0
  | 23 -> r.decoder_22_0
  | 24 -> r.decoder_23_0
  | 25 -> r.decoder_24_0
  | 26 -> r.decoder_25_0
  | 27 -> r.decoder_26_0
  | _ -> r.decoder_27_0

[@@"opaque_to_smt"]
let nrm (r:run) (l:nat{l <= 28}) : S.seq BF.t =
  match l with
  | 0 -> r.embedding_setup_1
  | 1 -> r.decoder_00_1
  | 2 -> r.decoder_01_1
  | 3 -> r.decoder_02_1
  | 4 -> r.decoder_03_1
  | 5 -> r.decoder_04_1
  | 6 -> r.decoder_05_1
  | 7 -> r.decoder_06_1
  | 8 -> r.decoder_07_1
  | 9 -> r.decoder_08_1
  | 10 -> r.decoder_09_1
  | 11 -> r.decoder_10_1
  | 12 -> r.decoder_11_1
  | 13 -> r.decoder_12_1
  | 14 -> r.decoder_13_1
  | 15 -> r.decoder_14_1
  | 16 -> r.decoder_15_1
  | 17 -> r.decoder_16_1
  | 18 -> r.decoder_17_1
  | 19 -> r.decoder_18_1
  | 20 -> r.decoder_19_1
  | 21 -> r.decoder_20_1
  | 22 -> r.decoder_21_1
  | 23 -> r.decoder_22_1
  | 24 -> r.decoder_23_1
  | 25 -> r.decoder_24_1
  | 26 -> r.decoder_25_1
  | 27 -> r.decoder_26_1
  | _ -> r.decoder_27_1

(* Element offset of the RMS weight applied after l layers. *)
let norm_off (l:nat) : nat = if l < 28 then layer_offset l else final_norm_offset

(* The decoder call of layer l, stated generically. *)
let dec_ok (r:run) (l:nat{l < 28}) : prop =
  base_decoder_relation r.input0 (res r l) (nrm r l) r.embedding_setup_2
    (layer_offset l) (norm_off (l + 1)) (res r (l + 1)) (nrm r (l + 1))

let dec_at (r:run) (l:nat{l < 28})
  : Lemma (requires family_ok r) (ensures dec_ok r l)
  = reveal_opaque (`%res) res;
    reveal_opaque (`%nrm) nrm;
    match l with
  | 0 -> ()
  | 1 -> ()
  | 2 -> ()
  | 3 -> ()
  | 4 -> ()
  | 5 -> ()
  | 6 -> ()
  | 7 -> ()
  | 8 -> ()
  | 9 -> ()
  | 10 -> ()
  | 11 -> ()
  | 12 -> ()
  | 13 -> ()
  | 14 -> ()
  | 15 -> ()
  | 16 -> ()
  | 17 -> ()
  | 18 -> ()
  | 19 -> ()
  | 20 -> ()
  | 21 -> ()
  | 22 -> ()
  | 23 -> ()
  | 24 -> ()
  | 25 -> ()
  | 26 -> ()
  | 27 -> ()
  | _ -> ()

let decoders_ok (r:run) : prop = forall (l:nat{l < 28}). dec_ok r l

let decoders_of_family (r:run)
  : Lemma (requires family_ok r) (ensures decoders_ok r)
  = introduce forall (l:nat{l < 28}). dec_ok r l with dec_at r l

(* Chain invariant after l layers, for one real weight vector rw. *)
let chain_inv (r:run) (rw:S.seq real) (l:nat{l <= 28}) : prop =
  S.length (res r l) == 7168 /\ S.length (nrm r l) == 7168 /\
  res r l %~ hidden_after rw r.input1 l /\
  nrm r l %~ rmsnorm_rows 7 1024 (hidden_after rw r.input1 l) (window rw (norm_off l) 1024)

let chain_pre (r:run) (rw:S.seq real) : prop =
  S.length r.input0 == packed_elements /\ r.input0 %~ rw /\
  S.length r.embedding_setup_2 == 1792 /\ r.embedding_setup_2 %~ rope_pair 0 /\
  decoders_ok r

let hidden_step (rw:S.seq real) (tokens:S.seq I64.t) (l:nat{l < 28})
  : Lemma (hidden_after rw tokens (l + 1) ==
           layer_residual (packed_layer rw (layer_offset l)) (rope_cos_table 0 7) (rope_sin_table 0 7)
             (hidden_after rw tokens l)
             (rmsnorm_rows 7 1024 (hidden_after rw tokens l) (window rw (layer_offset l) 1024)))
  = ()

#push-options "--z3rlimit 50 --fuel 0 --ifuel 0"
let step (r:run) (rw:S.seq real) (l:nat{l < 28})
  : Lemma (requires chain_pre r rw /\ chain_inv r rw l) (ensures chain_inv r rw (l + 1))
  = assert (dec_ok r l);
    let h = hidden_after rw r.input1 l in
    let n = rmsnorm_rows 7 1024 h (window rw (layer_offset l) 1024) in
    assert (norm_off l == layer_offset l);
    rope_pair_windows 0;
    hidden_step rw r.input1 l;
    assert (r.input0 %~ rw /\ res r l %~ h /\ nrm r l %~ n /\ r.embedding_setup_2 %~ rope_pair 0)
#pop-options

(* G4: the one chain lemma, size-generic, by induction over the calls. *)
let rec chain (r:run) (rw:S.seq real) (l:nat{l <= 28})
  : Lemma (requires chain_pre r rw /\ chain_inv r rw 0) (ensures chain_inv r rw l) (decreases l)
  = if l = 0 then () else (chain r rw (l - 1); step r rw (l - 1))

let res_0 (r:run) : Lemma (res r 0 == r.embedding_setup_0) = reveal_opaque (`%res) res
let nrm_0 (r:run) : Lemma (nrm r 0 == r.embedding_setup_1) = reveal_opaque (`%nrm) nrm
let nrm_28 (r:run) : Lemma (nrm r 28 == r.decoder_27_1) = reveal_opaque (`%nrm) nrm

let base_pre (r:run) : prop =
  input_lengths r /\ family_ok r /\ tokens_ok r.input1 /\ r.input2 == 1

#push-options "--z3rlimit 50 --fuel 1 --ifuel 0"
let embedding_facts (r:run)
  : Lemma (requires base_pre r)
          (ensures S.length r.embedding_setup_2 == 1792 /\ r.embedding_setup_2 %~ rope_pair 0 /\
                   (forall (rw:S.seq real). r.input0 %~ rw ==> chain_inv r rw 0))
  = res_0 r; nrm_0 r;
    assert (base_embedding_relation r.input0 r.input1 0 155583488 0
              r.embedding_setup_0 r.embedding_setup_1 r.embedding_setup_2);
    assert (norm_off 0 == 155583488)
#pop-options

(* Numerical part of the decoder stack, for every real weight vector. *)
let stack_facts (r:run) (rw:S.seq real)
  : Lemma (requires base_pre r /\ r.input0 %~ rw)
          (ensures S.length r.decoder_27_1 == 7168 /\ r.decoder_27_1 %~ final_hidden rw r.input1)
  = embedding_facts r;
    decoders_of_family r;
    chain r rw 28;
    nrm_28 r

let stack_length (r:run)
  : Lemma (requires base_pre r) (ensures S.length r.decoder_27_1 == 7168)
  = to_real_seq_is_approx r.input0;
    stack_facts r (to_real_seq r.input0)

#push-options "--z3rlimit 50 --fuel 0 --ifuel 0"
let scores_of (r:run) : S.seq BF.t = view r.final_projection_0 911616 151936

let projection_facts (r:run)
  : Lemma (requires base_pre r)
          (ensures S.length (scores_of r) == vocab_size /\
                   (forall (rw:S.seq real). r.input0 %~ rw ==> scores_of r %~ next_token_logits rw r.input1))
  = stack_length r;
    assert (base_projection_relation r.input0 r.decoder_27_1 0 r.final_projection_0);
    assert (S.length r.final_projection_0 == 7 * vocab_size);
    assert (scores_of r == window r.final_projection_0 (6 * vocab_size) vocab_size);
    introduce forall (rw:S.seq real). r.input0 %~ rw ==> scores_of r %~ next_token_logits rw r.input1
    with introduce _ ==> _ with begin
      stack_facts r rw;
      let rh = final_hidden rw r.input1 in
      assert (window r.final_projection_0 (6 * vocab_size) vocab_size %~
              logits_row vocab_size (window rw 0 embed_elements) (window rh (6 * 1024) 1024))
    end

let selection_facts (r:run)
  : Lemma (requires base_pre r)
          (ensures S.length (output0 r) == 8 /\
                   (forall (i:nat). i < 7 ==> S.index (output0 r) i == S.index r.input1 i) /\
                   argmax_choice (scores_of r) (I64.v (S.index (output0 r) 7)))
  = projection_facts r;
    assert (base_argmax_relation r.input1 (scores_of r) r.input2 r.select_next_token_0 r.select_next_token_1);
    first_argmax_choice (scores_of r) (I64.v (S.index r.select_next_token_1 0))
#pop-options

let refines_aux (r:run)
  : Lemma (requires base_pre r)
          (ensures program_postcondition r.input0 r.input1 r.input2 (output0 r) (output1 r))
  = stack_length r;
    projection_facts r;
    selection_facts r;
    introduce forall (rw:S.seq real). r.input0 %~ rw ==> output1 r %~ final_hidden rw r.input1
    with introduce _ ==> _ with stack_facts r rw;
    introduce exists (scores:S.seq BF.t).
      S.length scores == vocab_size /\
      (forall (rw:S.seq real). r.input0 %~ rw ==> scores %~ next_token_logits rw r.input1) /\
      argmax_choice scores (I64.v (S.index (output0 r) 7))
    with (scores_of r) and ()

let refines r =
  introduce tokens_ok r.input1 /\ r.input2 == 1 ==>
            program_postcondition r.input0 r.input1 r.input2 (output0 r) (output1 r)
  with refines_aux r
