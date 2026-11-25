module KPlay.Proof.Program.Optimized

(* Composition theorem for the optimized program (G2-G4): the boundary
   record of the run is indexed by layer, ONE size-generic chain lemma
   carries the decoder invariant across all layer calls by induction, and
   projection + argmax give the numerical and selection parts. *)

open FStar.Real
open Kuiper.Approximates
open KPlay.SharedSpec
open KPlay.Proof.Wiring.Optimized
open KPlay.Proof.SeqApprox
module S = FStar.Seq
module BF = Kuiper.BFloat16
module I64 = FStar.Int64

#set-options "--fuel 0 --ifuel 0 --z3rlimit 50"

(* ------------------------------------------------------------------ *)
(* Boundary record: residual / normalized stream before layer i        *)

let bnd_res (r:run) (i:nat{i <= num_layers}) : S.seq BF.t =
  match i with
  | 0 -> r.s0000_embedding_0
  | 1 -> r.s0000_decoder_00_0
  | 2 -> r.s0000_decoder_01_0
  | 3 -> r.s0000_decoder_02_0
  | 4 -> r.s0000_decoder_03_0
  | 5 -> r.s0000_decoder_04_0
  | 6 -> r.s0000_decoder_05_0
  | 7 -> r.s0000_decoder_06_0
  | 8 -> r.s0000_decoder_07_0
  | 9 -> r.s0000_decoder_08_0
  | 10 -> r.s0000_decoder_09_0
  | 11 -> r.s0000_decoder_10_0
  | 12 -> r.s0000_decoder_11_0
  | 13 -> r.s0000_decoder_12_0
  | 14 -> r.s0000_decoder_13_0
  | 15 -> r.s0000_decoder_14_0
  | 16 -> r.s0000_decoder_15_0
  | 17 -> r.s0000_decoder_16_0
  | 18 -> r.s0000_decoder_17_0
  | 19 -> r.s0000_decoder_18_0
  | 20 -> r.s0000_decoder_19_0
  | 21 -> r.s0000_decoder_20_0
  | 22 -> r.s0000_decoder_21_0
  | 23 -> r.s0000_decoder_22_0
  | 24 -> r.s0000_decoder_23_0
  | 25 -> r.s0000_decoder_24_0
  | 26 -> r.s0000_decoder_25_0
  | 27 -> r.s0000_decoder_26_0
  | _ -> r.s0000_decoder_27_0

let bnd_norm (r:run) (i:nat{i <= num_layers}) : S.seq BF.t =
  match i with
  | 0 -> r.s0000_embedding_1
  | 1 -> r.s0000_decoder_00_1
  | 2 -> r.s0000_decoder_01_1
  | 3 -> r.s0000_decoder_02_1
  | 4 -> r.s0000_decoder_03_1
  | 5 -> r.s0000_decoder_04_1
  | 6 -> r.s0000_decoder_05_1
  | 7 -> r.s0000_decoder_06_1
  | 8 -> r.s0000_decoder_07_1
  | 9 -> r.s0000_decoder_08_1
  | 10 -> r.s0000_decoder_09_1
  | 11 -> r.s0000_decoder_10_1
  | 12 -> r.s0000_decoder_11_1
  | 13 -> r.s0000_decoder_12_1
  | 14 -> r.s0000_decoder_13_1
  | 15 -> r.s0000_decoder_14_1
  | 16 -> r.s0000_decoder_15_1
  | 17 -> r.s0000_decoder_16_1
  | 18 -> r.s0000_decoder_17_1
  | 19 -> r.s0000_decoder_18_1
  | 20 -> r.s0000_decoder_19_1
  | 21 -> r.s0000_decoder_20_1
  | 22 -> r.s0000_decoder_21_1
  | 23 -> r.s0000_decoder_22_1
  | 24 -> r.s0000_decoder_23_1
  | 25 -> r.s0000_decoder_24_1
  | 26 -> r.s0000_decoder_25_1
  | 27 -> r.s0000_decoder_26_1
  | _ -> r.s0000_decoder_27_1

(* Packed offset of the RMSNorm weight applied after l layers. *)
let norm_off (l:nat) : nat = if l < num_layers then layer_offset l else final_norm_offset

let nw (rw:S.seq real) (l:nat) : S.seq real = window rw (norm_off l) 1024

(* The decoder call of layer i, stated generically in i. *)
let dec_ok (r:run) (i:nat{i < num_layers}) : prop =
  let w = r.input0 in
  let o = layer_offset i in
  optimized_decoder_relation
    (view w (o + 2048) 4194304) (view w (o + 6293504) 256) (view w (o + 4196352) 2097152)
    (view w (o + 1024) 1024) (view w (o + 6293760) 6291456) (view w (o + 12585216) 3145728)
    (view w (norm_off (i + 1)) 1024)
    (bnd_res r i) (bnd_norm r i) r.s0000_embedding_2 r.s0000_embedding_3 0
    (bnd_res r (i + 1)) (bnd_norm r (i + 1))

let all_dec_ok (r:run) : prop = forall (i:nat{i < num_layers}). dec_ok r i

(* Wiring correspondence: each generated conjunct is the generic call. *)
#push-options "--z3rlimit 100"
let wiring_dec (r:run) (i:nat{i < num_layers})
  : Lemma (requires family_ok r) (ensures dec_ok r i)
  = assert (layer_offset i == 155583488 + i * 15730944);
    assert (norm_off (i + 1) == (if i + 1 < num_layers then 155583488 + (i + 1) * 15730944
                                 else 155582464));
    match i with
  | 0 -> ()
  | 1 -> assert (layer_offset 1 == 171314432); assert (norm_off 2 == 187045376)
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
  | _ -> ()
#pop-options

let wiring_all (r:run)
  : Lemma (requires family_ok r) (ensures all_dec_ok r)
  = Classical.forall_intro (Classical.move_requires (wiring_dec r))

let dec_at (r:run) (i:nat{i < num_layers})
  : Lemma (requires all_dec_ok r) (ensures dec_ok r i)
  = ()

let scores_of (r:run) : S.seq BF.t = view r.s0000_projection_0 921600 151936

let emb_ok (r:run) : prop =
  optimized_embedding_relation (view r.input0 0 embed_elements) r.input1
    (view r.input0 (layer_offset 0) 1024)
    r.s0000_embedding_0 r.s0000_embedding_1 r.s0000_embedding_2 r.s0000_embedding_3

let proj_ok (r:run) : prop =
  optimized_projection_relation (view r.input0 0 embed_elements) (bnd_norm r num_layers)
    r.s0000_projection_0 r.s0000_projection_1

let argm_ok (r:run) : prop =
  optimized_argmax_relation (scores_of r) r.input1 r.s0000_argmax_0 r.s0000_argmax_1

let wiring_rest (r:run)
  : Lemma (requires family_ok r) (ensures emb_ok r /\ proj_ok r /\ argm_ok r)
  = assert (layer_offset 0 == 155583488)

(* ------------------------------------------------------------------ *)
(* Explicit instantiation of the family relations (L10)                *)

let dec_inst (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10:S.seq BF.t) (a11:int) (o0 o1:S.seq BF.t)
             (r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10:S.seq real)
  : Lemma (requires optimized_decoder_relation a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 o0 o1 /\
                    S.length a0 == 4194304 /\ S.length a1 == 256 /\ S.length a2 == 2097152 /\
                    S.length a3 == 1024 /\ S.length a4 == 6291456 /\ S.length a5 == 3145728 /\
                    S.length a6 == 1024 /\ S.length a7 == 8192 /\ S.length a8 == 8192 /\
                    S.length a9 == 524288 /\ S.length a10 == 524288 /\ 0 <= a11 /\ a11 <= 4088 /\
                    a0 %~ r0 /\ a1 %~ r1 /\ a2 %~ r2 /\ a3 %~ r3 /\ a4 %~ r4 /\ a5 %~ r5 /\
                    a6 %~ r6 /\ a7 %~ r7 /\ a8 %~ r8 /\ a9 %~ r9 /\ a10 %~ r10)
          (ensures S.length o0 == 8192 /\ S.length o1 == 8192 /\
                   (let res = layer_residual (split_layer r0 r1 r2 r3 r4 r5)
                                (window r9 (a11 * 128) 896) (window r10 (a11 * 128) 896)
                                (window r7 0 7168) (window r8 0 7168) in
                    window o0 0 7168 %~ res /\
                    window o1 0 7168 %~ rmsnorm_rows 7 1024 res r6))
  = (* The o1 conjunct carries the only term mentioning all eleven reals,
       i.e. the trigger of the relation's quantifier: state it first. *)
    let res = layer_residual (split_layer r0 r1 r2 r3 r4 r5)
                (window r9 (a11 * 128) 896) (window r10 (a11 * 128) 896)
                (window r7 0 7168) (window r8 0 7168) in
    assert (window o1 0 7168 %~ rmsnorm_rows 7 1024 res r6)

let emb_inst (a0:S.seq BF.t) (a1:S.seq I64.t) (a2 o0 o1 o2 o3:S.seq BF.t) (r0 r2:S.seq real)
  : Lemma (requires optimized_embedding_relation a0 a1 a2 o0 o1 o2 o3 /\
                    S.length a0 == embed_elements /\ S.length a1 == 7 /\ tokens_ok a1 /\
                    S.length a2 == 1024 /\ a0 %~ r0 /\ a2 %~ r2)
          (ensures S.length o0 == 8192 /\ S.length o1 == 8192 /\
                   S.length o2 == 524288 /\ S.length o3 == 524288 /\
                   o2 %~ rope_cos_table 0 4096 /\ o3 %~ rope_sin_table 0 4096 /\
                   window o0 0 7168 %~ embed_rows 7 r0 a1 /\
                   window o1 0 7168 %~ rmsnorm_rows 7 1024 (embed_rows 7 r0 a1) r2)
  = assert (window o1 0 7168 %~ rmsnorm_rows 7 1024 (embed_rows 7 r0 a1) r2)

let proj_at (a0 a1 o0 o1:S.seq BF.t) (r0 r1:S.seq real) (t:nat)
  : Lemma (requires optimized_projection_relation a0 a1 o0 o1 /\
                    S.length a0 == embed_elements /\ S.length a1 == 8192 /\
                    a0 %~ r0 /\ a1 %~ r1 /\ t < 8)
          (ensures window o0 (t * 153600) 153600 %~ logits_row 153600 r0 (window r1 (t * 1024) 1024))
  = ()

(* ------------------------------------------------------------------ *)
(* Reference equations                                                 *)

#push-options "--fuel 1"
let hidden_after_zero (rw:S.seq real) (tokens:S.seq I64.t)
  : Lemma (hidden_after rw tokens 0 == embed_rows 7 (window rw 0 embed_elements) tokens)
  = ()

let hidden_after_succ (rw:S.seq real) (tokens:S.seq I64.t) (l:nat)
  : Lemma (hidden_after rw tokens (l + 1) ==
           qwen_layer (packed_layer rw (layer_offset l)) (window rw (layer_offset l) 1024)
             (rope_cos_table 0 7) (rope_sin_table 0 7) (hidden_after rw tokens l))
  = ()
#pop-options

(* The decoder's split weight views are the packed layer weights. *)
let split_is_packed (rw:S.seq real) (o:nat)
  : Lemma (requires o + layer_elements <= S.length rw)
          (ensures split_layer (window rw (o + 2048) 4194304) (window rw (o + 6293504) 256)
                     (window rw (o + 4196352) 2097152) (window rw (o + 1024) 1024)
                     (window rw (o + 6293760) 6291456) (window rw (o + 12585216) 3145728)
                   == packed_layer rw o)
  = window_window rw (o + 2048) 4194304 0 2097152;
    window_window rw (o + 2048) 4194304 2097152 1048576;
    window_window rw (o + 2048) 4194304 3145728 1048576;
    window_window rw (o + 6293504) 256 0 128;
    window_window rw (o + 6293504) 256 128 128;
    window_window rw (o + 6293760) 6291456 0 3145728;
    window_window rw (o + 6293760) 6291456 3145728 3145728

(* ------------------------------------------------------------------ *)
(* The chain                                                           *)

(* Facts of the run that do not depend on the chosen reals. *)
let ctx (r:run) : prop =
  input_lengths r /\ tokens_ok r.input1 /\
  S.length r.s0000_embedding_2 == 524288 /\ S.length r.s0000_embedding_3 == 524288 /\
  r.s0000_embedding_2 %~ rope_cos_table 0 4096 /\ r.s0000_embedding_3 %~ rope_sin_table 0 4096

(* Invariant at boundary l (G3 numerical part, prompt rows). *)
let inv (rw:S.seq real) (tokens:S.seq I64.t) (l:nat) (x0 x1:S.seq BF.t) : prop =
  S.length x0 == 8192 /\ S.length x1 == 8192 /\
  window x0 0 7168 %~ hidden_after rw tokens l /\
  window x1 0 7168 %~ rmsnorm_rows 7 1024 (hidden_after rw tokens l) (nw rw l)

let emb_ctx (r:run)
  : Lemma (requires input_lengths r /\ tokens_ok r.input1 /\ emb_ok r)
          (ensures ctx r)
  = ()

let chain_base (r:run) (rw:S.seq real)
  : Lemma (requires ctx r /\ emb_ok r /\ r.input0 %~ rw)
          (ensures inv rw r.input1 0 (bnd_res r 0) (bnd_norm r 0))
  = let w = r.input0 in
    view_approx w rw 0 embed_elements;
    view_approx w rw (layer_offset 0) 1024;
    emb_inst (view w 0 embed_elements) r.input1 (view w (layer_offset 0) 1024)
      r.s0000_embedding_0 r.s0000_embedding_1 r.s0000_embedding_2 r.s0000_embedding_3
      (window rw 0 embed_elements) (window rw (layer_offset 0) 1024);
    hidden_after_zero rw r.input1

let view_len (#a:Type) (s:S.seq a) (o n:nat)
  : Lemma (requires o + n <= S.length s) (ensures S.length (view s o n) == n)
  = ()

(* Lengths of the decoder weight views of layer l. *)
let dec_views_len (w:S.seq BF.t) (l:nat{l < num_layers})
  : Lemma (requires S.length w == packed_elements)
          (ensures (let o = layer_offset l in
                    o + layer_elements <= packed_elements /\
                    S.length (view w (o + 2048) 4194304) == 4194304 /\
                    S.length (view w (o + 6293504) 256) == 256 /\
                    S.length (view w (o + 4196352) 2097152) == 2097152 /\
                    S.length (view w (o + 1024) 1024) == 1024 /\
                    S.length (view w (o + 6293760) 6291456) == 6291456 /\
                    S.length (view w (o + 12585216) 3145728) == 3145728 /\
                    S.length (view w (norm_off (l + 1)) 1024) == 1024))
  = let o = layer_offset l in
    assert (o + layer_elements <= packed_elements);
    view_len w (o + 2048) 4194304;
    view_len w (o + 6293504) 256;
    view_len w (o + 4196352) 2097152;
    view_len w (o + 1024) 1024;
    view_len w (o + 6293760) 6291456;
    view_len w (o + 12585216) 3145728;
    view_len w (norm_off (l + 1)) 1024

let chain_step (r:run) (rw:S.seq real) (l:nat{l < num_layers})
  : Lemma (requires ctx r /\ r.input0 %~ rw /\ dec_ok r l /\
                    inv rw r.input1 l (bnd_res r l) (bnd_norm r l))
          (ensures inv rw r.input1 (l + 1) (bnd_res r (l + 1)) (bnd_norm r (l + 1)))
  = let w = r.input0 in
    let o = layer_offset l in
    let x0 = bnd_res r l in
    let x1 = bnd_norm r l in
    let h = hidden_after rw r.input1 l in
    let n = rmsnorm_rows 7 1024 h (nw rw l) in
    dec_views_len w l;
    view_approx w rw (o + 2048) 4194304;
    view_approx w rw (o + 6293504) 256;
    view_approx w rw (o + 4196352) 2097152;
    view_approx w rw (o + 1024) 1024;
    view_approx w rw (o + 6293760) 6291456;
    view_approx w rw (o + 12585216) 3145728;
    view_approx w rw (norm_off (l + 1)) 1024;
    extend_approx x0 7168 h;
    extend_approx x1 7168 n;
    rope_table_windows 4096;
    split_is_packed rw o;
    hidden_after_succ rw r.input1 l;
    dec_inst
      (view w (o + 2048) 4194304) (view w (o + 6293504) 256) (view w (o + 4196352) 2097152)
      (view w (o + 1024) 1024) (view w (o + 6293760) 6291456) (view w (o + 12585216) 3145728)
      (view w (norm_off (l + 1)) 1024)
      x0 x1 r.s0000_embedding_2 r.s0000_embedding_3 0
      (bnd_res r (l + 1)) (bnd_norm r (l + 1))
      (window rw (o + 2048) 4194304) (window rw (o + 6293504) 256) (window rw (o + 4196352) 2097152)
      (window rw (o + 1024) 1024) (window rw (o + 6293760) 6291456) (window rw (o + 12585216) 3145728)
      (nw rw (l + 1))
      (extend_real x0 7168 h) (extend_real x1 7168 n)
      (rope_cos_table 0 4096) (rope_sin_table 0 4096)

(* G4: the one chain lemma, size-generic in the layer count. *)
let rec chain (r:run) (rw:S.seq real) (l:nat{l <= num_layers})
  : Lemma (requires ctx r /\ emb_ok r /\ all_dec_ok r /\ r.input0 %~ rw)
          (ensures inv rw r.input1 l (bnd_res r l) (bnd_norm r l))
          (decreases l)
  = if l = 0 then chain_base r rw
    else begin
      chain r rw (l - 1);
      dec_at r (l - 1);
      chain_step r rw (l - 1)
    end

(* ------------------------------------------------------------------ *)
(* Projection and selection                                            *)

let final_numerical (r:run) (rw:S.seq real)
  : Lemma (requires ctx r /\ emb_ok r /\ all_dec_ok r /\ r.input0 %~ rw)
          (ensures S.length (bnd_norm r num_layers) == 8192 /\
                   window (bnd_norm r num_layers) 0 7168 %~ final_hidden rw r.input1)
  = chain r rw num_layers

let final_scores (r:run) (rw:S.seq real)
  : Lemma (requires ctx r /\ emb_ok r /\ proj_ok r /\ all_dec_ok r /\ r.input0 %~ rw)
          (ensures S.length r.s0000_projection_0 == 8 * 153600 /\
                   scores_of r %~ next_token_logits rw r.input1)
  = let w = r.input0 in
    let x = bnd_norm r num_layers in
    let f = final_hidden rw r.input1 in
    let p = r.s0000_projection_0 in
    let r0 = window rw 0 embed_elements in
    final_numerical r rw;
    view_approx w rw 0 embed_elements;
    extend_approx x 7168 f;
    let r1 = extend_real x 7168 f in
    proj_at (view w 0 embed_elements) x p r.s0000_projection_1 r0 r1 6;
    window_extend x 7168 f 6144 1024;
    let row = window p 921600 153600 in
    let lr = logits_row 153600 r0 (window f 6144 1024) in
    assert (row %~ lr);
    window_approx row lr 0 151936;
    window_window p 921600 153600 0 151936;
    logits_row_prefix 151936 153600 r0 (window f 6144 1024)

let refines_core (r:run)
  : Lemma (requires input_lengths r /\ emb_ok r /\ proj_ok r /\ argm_ok r /\ all_dec_ok r)
          (ensures program_postcondition r.input0 r.input1 r.input2 (output0 r) (output1 r))
  =
  let w = r.input0 in
  let tok = r.input1 in
  introduce (S.length w == packed_elements /\ S.length tok == prompt_length /\
             tokens_ok tok /\ r.input2 == 1) ==>
            (S.length (output0 r) == 8 /\ S.length (output1 r) == 7168 /\
             (forall (i:nat). i < 7 ==> S.index (output0 r) i == S.index tok i) /\
             (forall (rw:S.seq real). w %~ rw ==> output1 r %~ final_hidden rw tok) /\
             (exists (scores:S.seq BF.t).
                S.length scores == vocab_size /\
                (forall (rw:S.seq real). w %~ rw ==> scores %~ next_token_logits rw tok) /\
                argmax_choice scores (I64.v (S.index (output0 r) 7))))
  with begin
    emb_ctx r;
    let rw0 = to_real_seq w in
    to_real_seq_is_approx w;
    final_numerical r rw0;
    final_scores r rw0;
    let num (rw:S.seq real)
      : Lemma (w %~ rw ==> output1 r %~ final_hidden rw tok)
      = Classical.move_requires (final_numerical r) rw
    in
    let sel (rw:S.seq real)
      : Lemma (w %~ rw ==> scores_of r %~ next_token_logits rw tok)
      = Classical.move_requires (final_scores r) rw
    in
    Classical.forall_intro num;
    Classical.forall_intro sel;
    let s = scores_of r in
    tree_argmax_choice s (I64.v (S.index r.s0000_argmax_0 0));
    introduce exists (scores:S.seq BF.t).
                S.length scores == vocab_size /\
                (forall (rw:S.seq real). w %~ rw ==> scores %~ next_token_logits rw tok) /\
                argmax_choice scores (I64.v (S.index (output0 r) 7))
    with s and ()
  end

let refines r =
  wiring_rest r;
  wiring_all r;
  refines_core r
