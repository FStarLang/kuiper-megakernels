module KPlay.Proof.BaseAccess

(* Accessors for the per-(position, layer) call boundaries of the base run, and
   extraction of the family relation of each call from family_ok. Position 0
   is loop 0 (iteration l / 4, member l % 4); position t >= 1 is iteration
   t - 1 of loop 1; positions >= C are loop 2 (selection/update only). *)

open KPlay.SharedSpec
open KPlay.Proof.Wiring.Base
module S = FStar.Seq
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32
module I64 = FStar.Int64

#set-options "--z3rlimit 20 --fuel 0 --ifuel 0"

(* KV capacity C = P + S - 1 *)
let cap (r:run) : int = S.length r.input1 + r.input2 - 1

(*** Layer-call boundaries *)

let lout (r:run) (t l:nat) : S.seq BF.t =
  if t = 0 then
    let g = l / 4 in let k = l % 4 in
    if k = 0 then r.loop0_position_00_layer_i_0_0 g else
    if k = 1 then r.loop0_position_00_layer_i_1_0 g else
    if k = 2 then r.loop0_position_00_layer_i_2_0 g else
    if k = 3 then r.loop0_position_00_layer_i_3_0 g else
    S.empty
  else
    let i = t - 1 in
    if l = 0 then r.loop1_position_i_layer_00_0 i else
    if l = 1 then r.loop1_position_i_layer_01_0 i else
    if l = 2 then r.loop1_position_i_layer_02_0 i else
    if l = 3 then r.loop1_position_i_layer_03_0 i else
    if l = 4 then r.loop1_position_i_layer_04_0 i else
    if l = 5 then r.loop1_position_i_layer_05_0 i else
    if l = 6 then r.loop1_position_i_layer_06_0 i else
    if l = 7 then r.loop1_position_i_layer_07_0 i else
    if l = 8 then r.loop1_position_i_layer_08_0 i else
    if l = 9 then r.loop1_position_i_layer_09_0 i else
    if l = 10 then r.loop1_position_i_layer_10_0 i else
    if l = 11 then r.loop1_position_i_layer_11_0 i else
    if l = 12 then r.loop1_position_i_layer_12_0 i else
    if l = 13 then r.loop1_position_i_layer_13_0 i else
    if l = 14 then r.loop1_position_i_layer_14_0 i else
    if l = 15 then r.loop1_position_i_layer_15_0 i else
    if l = 16 then r.loop1_position_i_layer_16_0 i else
    if l = 17 then r.loop1_position_i_layer_17_0 i else
    if l = 18 then r.loop1_position_i_layer_18_0 i else
    if l = 19 then r.loop1_position_i_layer_19_0 i else
    if l = 20 then r.loop1_position_i_layer_20_0 i else
    if l = 21 then r.loop1_position_i_layer_21_0 i else
    if l = 22 then r.loop1_position_i_layer_22_0 i else
    if l = 23 then r.loop1_position_i_layer_23_0 i else
    S.empty

let conv_out (r:run) (t l:nat) : S.seq BF.t =
  if t = 0 then
    let g = l / 4 in let k = l % 4 in
    if k = 0 then r.loop0_position_00_layer_i_0_1 g else
    if k = 1 then r.loop0_position_00_layer_i_1_1 g else
    if k = 2 then r.loop0_position_00_layer_i_2_1 g else
    S.empty
  else
    let i = t - 1 in
    if l = 0 then r.loop1_position_i_layer_00_1 i else
    if l = 1 then r.loop1_position_i_layer_01_1 i else
    if l = 2 then r.loop1_position_i_layer_02_1 i else
    if l = 4 then r.loop1_position_i_layer_04_1 i else
    if l = 5 then r.loop1_position_i_layer_05_1 i else
    if l = 6 then r.loop1_position_i_layer_06_1 i else
    if l = 8 then r.loop1_position_i_layer_08_1 i else
    if l = 9 then r.loop1_position_i_layer_09_1 i else
    if l = 10 then r.loop1_position_i_layer_10_1 i else
    if l = 12 then r.loop1_position_i_layer_12_1 i else
    if l = 13 then r.loop1_position_i_layer_13_1 i else
    if l = 14 then r.loop1_position_i_layer_14_1 i else
    if l = 16 then r.loop1_position_i_layer_16_1 i else
    if l = 17 then r.loop1_position_i_layer_17_1 i else
    if l = 18 then r.loop1_position_i_layer_18_1 i else
    if l = 20 then r.loop1_position_i_layer_20_1 i else
    if l = 21 then r.loop1_position_i_layer_21_1 i else
    if l = 22 then r.loop1_position_i_layer_22_1 i else
    S.empty

let rec_out (r:run) (t l:nat) : S.seq F32.t =
  if t = 0 then
    let g = l / 4 in let k = l % 4 in
    if k = 0 then r.loop0_position_00_layer_i_0_2 g else
    if k = 1 then r.loop0_position_00_layer_i_1_2 g else
    if k = 2 then r.loop0_position_00_layer_i_2_2 g else
    S.empty
  else
    let i = t - 1 in
    if l = 0 then r.loop1_position_i_layer_00_2 i else
    if l = 1 then r.loop1_position_i_layer_01_2 i else
    if l = 2 then r.loop1_position_i_layer_02_2 i else
    if l = 4 then r.loop1_position_i_layer_04_2 i else
    if l = 5 then r.loop1_position_i_layer_05_2 i else
    if l = 6 then r.loop1_position_i_layer_06_2 i else
    if l = 8 then r.loop1_position_i_layer_08_2 i else
    if l = 9 then r.loop1_position_i_layer_09_2 i else
    if l = 10 then r.loop1_position_i_layer_10_2 i else
    if l = 12 then r.loop1_position_i_layer_12_2 i else
    if l = 13 then r.loop1_position_i_layer_13_2 i else
    if l = 14 then r.loop1_position_i_layer_14_2 i else
    if l = 16 then r.loop1_position_i_layer_16_2 i else
    if l = 17 then r.loop1_position_i_layer_17_2 i else
    if l = 18 then r.loop1_position_i_layer_18_2 i else
    if l = 20 then r.loop1_position_i_layer_20_2 i else
    if l = 21 then r.loop1_position_i_layer_21_2 i else
    if l = 22 then r.loop1_position_i_layer_22_2 i else
    S.empty

let kv_out (r:run) (t l:nat) : S.seq BF.t =
  if t = 0 then
    let g = l / 4 in let k = l % 4 in
    if k = 3 then r.loop0_position_00_layer_i_3_1 g else
    S.empty
  else
    let i = t - 1 in
    if l = 3 then r.loop1_position_i_layer_03_1 i else
    if l = 7 then r.loop1_position_i_layer_07_1 i else
    if l = 11 then r.loop1_position_i_layer_11_1 i else
    if l = 15 then r.loop1_position_i_layer_15_1 i else
    if l = 19 then r.loop1_position_i_layer_19_1 i else
    if l = 23 then r.loop1_position_i_layer_23_1 i else
    S.empty

(* residual entering layer l (l = 24: leaving layer 23) *)
let lx (r:run) (t l:nat) : S.seq BF.t =
  if l = 0 then (if t = 0 then r.position_00_embedding_0 else r.loop1_position_i_embedding_0 (t - 1))
  else lout r t (l - 1)

let pcos (r:run) (t:nat) : S.seq BF.t =
  if t = 0 then r.position_00_embedding_1 else r.loop1_position_i_embedding_1 (t - 1)

let psin (r:run) (t:nat) : S.seq BF.t =
  if t = 0 then r.position_00_embedding_2 else r.loop1_position_i_embedding_2 (t - 1)

let kv_len (r:run) : int = 1024 * S.length r.input1 + 1024 * r.input2 - 1024

let kv_off (r:run) (l:nat) : int = 442368 + kv_len r * (l / 4)

let conv_in (r:run) (t l:nat) : S.seq BF.t =
  if t = 0 then view r.initialize_1 (24576 * dn_index l) 24576 else conv_out r (t - 1) l

let rec_in (r:run) (t l:nat) : S.seq F32.t =
  if t = 0 then view r.initialize_2 (262144 * dn_index l) 262144 else rec_out r (t - 1) l

let kv_in (r:run) (t l:nat) : S.seq BF.t =
  if t = 0 then view r.initialize_1 (kv_off r l) (kv_len r) else kv_out r (t - 1) l

let wlen (l:nat) : int = if is_attn l then attn_len else dn_len

let wv (r:run) (l:nat) : S.seq BF.t = view r.input0 (layer_off l) (wlen l)

[@@"opaque_to_smt"]
let layer_rel (r:run) (t l:nat) : prop =
  if is_attn l then
    base_full_attention_decoder_relation (wv r l) (lx r t l) (kv_in r t l) (pcos r t) (psin r t) t
      (lout r t l) (kv_out r t l)
  else
    base_recurrent_decoder_relation (wv r l) (lx r t l) (conv_in r t l) (rec_in r t l)
      (lout r t l) (conv_out r t l) (rec_out r t l)

let fam_loop0 (r:run) (i:nat)
  : Lemma (requires family_ok r /\ i < 6) (ensures loop0_ok r i)
  = ()

let fam_loop1 (r:run) (i:nat)
  : Lemma (requires family_ok r /\ i < loop1_count r) (ensures loop1_ok r i)
  = ()

let layer_rel_0_0 (r:run) (g:nat)
  : Lemma (requires loop0_ok r g /\ g < 6) (ensures layer_rel r 0 (4 * g + 0))
  = reveal_opaque (`%layer_rel) (layer_rel r 0 (4 * g + 0));
    let l = 4 * g + 0 in
    assert (l / 4 == g /\ l % 4 == 0);
    if g > 0 then assert ((l - 1) / 4 == g - 1 /\ (l - 1) % 4 == 3);
    assert (lx r 0 l == (if g = 0 then r.position_00_embedding_0 else r.loop0_position_00_layer_i_3_0 (g - 1)));
    assert (lout r 0 l == r.loop0_position_00_layer_i_0_0 g);
    assert (wv r l == view r.input0 (254280704 + 83018720 * g) 21555360);
    assert (conv_in r 0 l == view r.initialize_1 (0 + 73728 * g) 24576);
    assert (rec_in r 0 l == view r.initialize_2 (0 + 786432 * g) 262144);
    assert (conv_out r 0 l == r.loop0_position_00_layer_i_0_1 g);
    assert (rec_out r 0 l == r.loop0_position_00_layer_i_0_2 g);
    ()

let layer_rel_0_1 (r:run) (g:nat)
  : Lemma (requires loop0_ok r g /\ g < 6) (ensures layer_rel r 0 (4 * g + 1))
  = reveal_opaque (`%layer_rel) (layer_rel r 0 (4 * g + 1));
    let l = 4 * g + 1 in
    assert (l / 4 == g /\ l % 4 == 1);
    assert ((l - 1) / 4 == g /\ (l - 1) % 4 == 0);
    assert (lx r 0 l == r.loop0_position_00_layer_i_0_0 g);
    assert (lout r 0 l == r.loop0_position_00_layer_i_1_0 g);
    assert (wv r l == view r.input0 (275836064 + 83018720 * g) 21555360);
    assert (conv_in r 0 l == view r.initialize_1 (24576 + 73728 * g) 24576);
    assert (rec_in r 0 l == view r.initialize_2 (262144 + 786432 * g) 262144);
    assert (conv_out r 0 l == r.loop0_position_00_layer_i_1_1 g);
    assert (rec_out r 0 l == r.loop0_position_00_layer_i_1_2 g);
    ()

let layer_rel_0_2 (r:run) (g:nat)
  : Lemma (requires loop0_ok r g /\ g < 6) (ensures layer_rel r 0 (4 * g + 2))
  = reveal_opaque (`%layer_rel) (layer_rel r 0 (4 * g + 2));
    let l = 4 * g + 2 in
    assert (l / 4 == g /\ l % 4 == 2);
    assert ((l - 1) / 4 == g /\ (l - 1) % 4 == 1);
    assert (lx r 0 l == r.loop0_position_00_layer_i_1_0 g);
    assert (lout r 0 l == r.loop0_position_00_layer_i_2_0 g);
    assert (wv r l == view r.input0 (297391424 + 83018720 * g) 21555360);
    assert (conv_in r 0 l == view r.initialize_1 (49152 + 73728 * g) 24576);
    assert (rec_in r 0 l == view r.initialize_2 (524288 + 786432 * g) 262144);
    assert (conv_out r 0 l == r.loop0_position_00_layer_i_2_1 g);
    assert (rec_out r 0 l == r.loop0_position_00_layer_i_2_2 g);
    ()

let layer_rel_0_3 (r:run) (g:nat)
  : Lemma (requires loop0_ok r g /\ g < 6) (ensures layer_rel r 0 (4 * g + 3))
  = reveal_opaque (`%layer_rel) (layer_rel r 0 (4 * g + 3));
    let l = 4 * g + 3 in
    assert (l / 4 == g /\ l % 4 == 3);
    assert ((l - 1) / 4 == g /\ (l - 1) % 4 == 2);
    assert (lx r 0 l == r.loop0_position_00_layer_i_2_0 g);
    assert (lout r 0 l == r.loop0_position_00_layer_i_3_0 g);
    assert (wv r l == view r.input0 (318946784 + 83018720 * g) 18352640);
    assert (kv_in r 0 l == view r.initialize_1 (442368 + (1024 * S.length r.input1 + 1024 * r.input2 - 1024) * g) (1024 * S.length r.input1 + 1024 * r.input2 - 1024));
    assert (kv_out r 0 l == r.loop0_position_00_layer_i_3_1 g);
    ()

let layer_rel_0 (r:run) (g k:nat)
  : Lemma (requires loop0_ok r g /\ g < 6 /\ k < 4) (ensures layer_rel r 0 (4 * g + k))
  = if k = 0 then layer_rel_0_0 r g
    else if k = 1 then layer_rel_0_1 r g
    else if k = 2 then layer_rel_0_2 r g
    else layer_rel_0_3 r g

let layer_rel_1_0 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 0)
  = reveal_opaque (`%layer_rel) (layer_rel r t 0);
    let i = t - 1 in
    assert (lx r t 0 == r.loop1_position_i_embedding_0 i);
    assert (lout r t 0 == r.loop1_position_i_layer_00_0 i);
    assert (wv r 0 == view r.input0 254280704 21555360);
    assert (conv_in r t 0 == (if i = 0 then r.loop0_position_00_layer_i_0_1 0 else r.loop1_position_i_layer_00_1 (i - 1)));
    assert (rec_in r t 0 == (if i = 0 then r.loop0_position_00_layer_i_0_2 0 else r.loop1_position_i_layer_00_2 (i - 1)));
    assert (conv_out r t 0 == r.loop1_position_i_layer_00_1 i);
    assert (rec_out r t 0 == r.loop1_position_i_layer_00_2 i);
    ()

let layer_rel_1_1 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 1)
  = reveal_opaque (`%layer_rel) (layer_rel r t 1);
    let i = t - 1 in
    assert (lx r t 1 == r.loop1_position_i_layer_00_0 i);
    assert (lout r t 1 == r.loop1_position_i_layer_01_0 i);
    assert (wv r 1 == view r.input0 275836064 21555360);
    assert (conv_in r t 1 == (if i = 0 then r.loop0_position_00_layer_i_1_1 0 else r.loop1_position_i_layer_01_1 (i - 1)));
    assert (rec_in r t 1 == (if i = 0 then r.loop0_position_00_layer_i_1_2 0 else r.loop1_position_i_layer_01_2 (i - 1)));
    assert (conv_out r t 1 == r.loop1_position_i_layer_01_1 i);
    assert (rec_out r t 1 == r.loop1_position_i_layer_01_2 i);
    ()

let layer_rel_1_2 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 2)
  = reveal_opaque (`%layer_rel) (layer_rel r t 2);
    let i = t - 1 in
    assert (lx r t 2 == r.loop1_position_i_layer_01_0 i);
    assert (lout r t 2 == r.loop1_position_i_layer_02_0 i);
    assert (wv r 2 == view r.input0 297391424 21555360);
    assert (conv_in r t 2 == (if i = 0 then r.loop0_position_00_layer_i_2_1 0 else r.loop1_position_i_layer_02_1 (i - 1)));
    assert (rec_in r t 2 == (if i = 0 then r.loop0_position_00_layer_i_2_2 0 else r.loop1_position_i_layer_02_2 (i - 1)));
    assert (conv_out r t 2 == r.loop1_position_i_layer_02_1 i);
    assert (rec_out r t 2 == r.loop1_position_i_layer_02_2 i);
    ()

let layer_rel_1_3 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 3)
  = reveal_opaque (`%layer_rel) (layer_rel r t 3);
    let i = t - 1 in
    assert (lx r t 3 == r.loop1_position_i_layer_02_0 i);
    assert (lout r t 3 == r.loop1_position_i_layer_03_0 i);
    assert (wv r 3 == view r.input0 318946784 18352640);
    assert (kv_in r t 3 == (if i = 0 then r.loop0_position_00_layer_i_3_1 0 else r.loop1_position_i_layer_03_1 (i - 1)));
    assert (kv_out r t 3 == r.loop1_position_i_layer_03_1 i);
    assert (pcos r t == r.loop1_position_i_embedding_1 i);
    assert (psin r t == r.loop1_position_i_embedding_2 i);
    ()

let layer_rel_1_4 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 4)
  = reveal_opaque (`%layer_rel) (layer_rel r t 4);
    let i = t - 1 in
    assert (lx r t 4 == r.loop1_position_i_layer_03_0 i);
    assert (lout r t 4 == r.loop1_position_i_layer_04_0 i);
    assert (wv r 4 == view r.input0 337299424 21555360);
    assert (conv_in r t 4 == (if i = 0 then r.loop0_position_00_layer_i_0_1 1 else r.loop1_position_i_layer_04_1 (i - 1)));
    assert (rec_in r t 4 == (if i = 0 then r.loop0_position_00_layer_i_0_2 1 else r.loop1_position_i_layer_04_2 (i - 1)));
    assert (conv_out r t 4 == r.loop1_position_i_layer_04_1 i);
    assert (rec_out r t 4 == r.loop1_position_i_layer_04_2 i);
    ()

let layer_rel_1_5 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 5)
  = reveal_opaque (`%layer_rel) (layer_rel r t 5);
    let i = t - 1 in
    assert (lx r t 5 == r.loop1_position_i_layer_04_0 i);
    assert (lout r t 5 == r.loop1_position_i_layer_05_0 i);
    assert (wv r 5 == view r.input0 358854784 21555360);
    assert (conv_in r t 5 == (if i = 0 then r.loop0_position_00_layer_i_1_1 1 else r.loop1_position_i_layer_05_1 (i - 1)));
    assert (rec_in r t 5 == (if i = 0 then r.loop0_position_00_layer_i_1_2 1 else r.loop1_position_i_layer_05_2 (i - 1)));
    assert (conv_out r t 5 == r.loop1_position_i_layer_05_1 i);
    assert (rec_out r t 5 == r.loop1_position_i_layer_05_2 i);
    ()

let layer_rel_1_6 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 6)
  = reveal_opaque (`%layer_rel) (layer_rel r t 6);
    let i = t - 1 in
    assert (lx r t 6 == r.loop1_position_i_layer_05_0 i);
    assert (lout r t 6 == r.loop1_position_i_layer_06_0 i);
    assert (wv r 6 == view r.input0 380410144 21555360);
    assert (conv_in r t 6 == (if i = 0 then r.loop0_position_00_layer_i_2_1 1 else r.loop1_position_i_layer_06_1 (i - 1)));
    assert (rec_in r t 6 == (if i = 0 then r.loop0_position_00_layer_i_2_2 1 else r.loop1_position_i_layer_06_2 (i - 1)));
    assert (conv_out r t 6 == r.loop1_position_i_layer_06_1 i);
    assert (rec_out r t 6 == r.loop1_position_i_layer_06_2 i);
    ()

let layer_rel_1_7 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 7)
  = reveal_opaque (`%layer_rel) (layer_rel r t 7);
    let i = t - 1 in
    assert (lx r t 7 == r.loop1_position_i_layer_06_0 i);
    assert (lout r t 7 == r.loop1_position_i_layer_07_0 i);
    assert (wv r 7 == view r.input0 401965504 18352640);
    assert (kv_in r t 7 == (if i = 0 then r.loop0_position_00_layer_i_3_1 1 else r.loop1_position_i_layer_07_1 (i - 1)));
    assert (kv_out r t 7 == r.loop1_position_i_layer_07_1 i);
    assert (pcos r t == r.loop1_position_i_embedding_1 i);
    assert (psin r t == r.loop1_position_i_embedding_2 i);
    ()

let layer_rel_1_8 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 8)
  = reveal_opaque (`%layer_rel) (layer_rel r t 8);
    let i = t - 1 in
    assert (lx r t 8 == r.loop1_position_i_layer_07_0 i);
    assert (lout r t 8 == r.loop1_position_i_layer_08_0 i);
    assert (wv r 8 == view r.input0 420318144 21555360);
    assert (conv_in r t 8 == (if i = 0 then r.loop0_position_00_layer_i_0_1 2 else r.loop1_position_i_layer_08_1 (i - 1)));
    assert (rec_in r t 8 == (if i = 0 then r.loop0_position_00_layer_i_0_2 2 else r.loop1_position_i_layer_08_2 (i - 1)));
    assert (conv_out r t 8 == r.loop1_position_i_layer_08_1 i);
    assert (rec_out r t 8 == r.loop1_position_i_layer_08_2 i);
    ()

let layer_rel_1_9 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 9)
  = reveal_opaque (`%layer_rel) (layer_rel r t 9);
    let i = t - 1 in
    assert (lx r t 9 == r.loop1_position_i_layer_08_0 i);
    assert (lout r t 9 == r.loop1_position_i_layer_09_0 i);
    assert (wv r 9 == view r.input0 441873504 21555360);
    assert (conv_in r t 9 == (if i = 0 then r.loop0_position_00_layer_i_1_1 2 else r.loop1_position_i_layer_09_1 (i - 1)));
    assert (rec_in r t 9 == (if i = 0 then r.loop0_position_00_layer_i_1_2 2 else r.loop1_position_i_layer_09_2 (i - 1)));
    assert (conv_out r t 9 == r.loop1_position_i_layer_09_1 i);
    assert (rec_out r t 9 == r.loop1_position_i_layer_09_2 i);
    ()

let layer_rel_1_10 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 10)
  = reveal_opaque (`%layer_rel) (layer_rel r t 10);
    let i = t - 1 in
    assert (lx r t 10 == r.loop1_position_i_layer_09_0 i);
    assert (lout r t 10 == r.loop1_position_i_layer_10_0 i);
    assert (wv r 10 == view r.input0 463428864 21555360);
    assert (conv_in r t 10 == (if i = 0 then r.loop0_position_00_layer_i_2_1 2 else r.loop1_position_i_layer_10_1 (i - 1)));
    assert (rec_in r t 10 == (if i = 0 then r.loop0_position_00_layer_i_2_2 2 else r.loop1_position_i_layer_10_2 (i - 1)));
    assert (conv_out r t 10 == r.loop1_position_i_layer_10_1 i);
    assert (rec_out r t 10 == r.loop1_position_i_layer_10_2 i);
    ()

let layer_rel_1_11 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 11)
  = reveal_opaque (`%layer_rel) (layer_rel r t 11);
    let i = t - 1 in
    assert (lx r t 11 == r.loop1_position_i_layer_10_0 i);
    assert (lout r t 11 == r.loop1_position_i_layer_11_0 i);
    assert (wv r 11 == view r.input0 484984224 18352640);
    assert (kv_in r t 11 == (if i = 0 then r.loop0_position_00_layer_i_3_1 2 else r.loop1_position_i_layer_11_1 (i - 1)));
    assert (kv_out r t 11 == r.loop1_position_i_layer_11_1 i);
    assert (pcos r t == r.loop1_position_i_embedding_1 i);
    assert (psin r t == r.loop1_position_i_embedding_2 i);
    ()

let layer_rel_1_12 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 12)
  = reveal_opaque (`%layer_rel) (layer_rel r t 12);
    let i = t - 1 in
    assert (lx r t 12 == r.loop1_position_i_layer_11_0 i);
    assert (lout r t 12 == r.loop1_position_i_layer_12_0 i);
    assert (wv r 12 == view r.input0 503336864 21555360);
    assert (conv_in r t 12 == (if i = 0 then r.loop0_position_00_layer_i_0_1 3 else r.loop1_position_i_layer_12_1 (i - 1)));
    assert (rec_in r t 12 == (if i = 0 then r.loop0_position_00_layer_i_0_2 3 else r.loop1_position_i_layer_12_2 (i - 1)));
    assert (conv_out r t 12 == r.loop1_position_i_layer_12_1 i);
    assert (rec_out r t 12 == r.loop1_position_i_layer_12_2 i);
    ()

let layer_rel_1_13 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 13)
  = reveal_opaque (`%layer_rel) (layer_rel r t 13);
    let i = t - 1 in
    assert (lx r t 13 == r.loop1_position_i_layer_12_0 i);
    assert (lout r t 13 == r.loop1_position_i_layer_13_0 i);
    assert (wv r 13 == view r.input0 524892224 21555360);
    assert (conv_in r t 13 == (if i = 0 then r.loop0_position_00_layer_i_1_1 3 else r.loop1_position_i_layer_13_1 (i - 1)));
    assert (rec_in r t 13 == (if i = 0 then r.loop0_position_00_layer_i_1_2 3 else r.loop1_position_i_layer_13_2 (i - 1)));
    assert (conv_out r t 13 == r.loop1_position_i_layer_13_1 i);
    assert (rec_out r t 13 == r.loop1_position_i_layer_13_2 i);
    ()

let layer_rel_1_14 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 14)
  = reveal_opaque (`%layer_rel) (layer_rel r t 14);
    let i = t - 1 in
    assert (lx r t 14 == r.loop1_position_i_layer_13_0 i);
    assert (lout r t 14 == r.loop1_position_i_layer_14_0 i);
    assert (wv r 14 == view r.input0 546447584 21555360);
    assert (conv_in r t 14 == (if i = 0 then r.loop0_position_00_layer_i_2_1 3 else r.loop1_position_i_layer_14_1 (i - 1)));
    assert (rec_in r t 14 == (if i = 0 then r.loop0_position_00_layer_i_2_2 3 else r.loop1_position_i_layer_14_2 (i - 1)));
    assert (conv_out r t 14 == r.loop1_position_i_layer_14_1 i);
    assert (rec_out r t 14 == r.loop1_position_i_layer_14_2 i);
    ()

let layer_rel_1_15 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 15)
  = reveal_opaque (`%layer_rel) (layer_rel r t 15);
    let i = t - 1 in
    assert (lx r t 15 == r.loop1_position_i_layer_14_0 i);
    assert (lout r t 15 == r.loop1_position_i_layer_15_0 i);
    assert (wv r 15 == view r.input0 568002944 18352640);
    assert (kv_in r t 15 == (if i = 0 then r.loop0_position_00_layer_i_3_1 3 else r.loop1_position_i_layer_15_1 (i - 1)));
    assert (kv_out r t 15 == r.loop1_position_i_layer_15_1 i);
    assert (pcos r t == r.loop1_position_i_embedding_1 i);
    assert (psin r t == r.loop1_position_i_embedding_2 i);
    ()

let layer_rel_1_16 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 16)
  = reveal_opaque (`%layer_rel) (layer_rel r t 16);
    let i = t - 1 in
    assert (lx r t 16 == r.loop1_position_i_layer_15_0 i);
    assert (lout r t 16 == r.loop1_position_i_layer_16_0 i);
    assert (wv r 16 == view r.input0 586355584 21555360);
    assert (conv_in r t 16 == (if i = 0 then r.loop0_position_00_layer_i_0_1 4 else r.loop1_position_i_layer_16_1 (i - 1)));
    assert (rec_in r t 16 == (if i = 0 then r.loop0_position_00_layer_i_0_2 4 else r.loop1_position_i_layer_16_2 (i - 1)));
    assert (conv_out r t 16 == r.loop1_position_i_layer_16_1 i);
    assert (rec_out r t 16 == r.loop1_position_i_layer_16_2 i);
    ()

let layer_rel_1_17 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 17)
  = reveal_opaque (`%layer_rel) (layer_rel r t 17);
    let i = t - 1 in
    assert (lx r t 17 == r.loop1_position_i_layer_16_0 i);
    assert (lout r t 17 == r.loop1_position_i_layer_17_0 i);
    assert (wv r 17 == view r.input0 607910944 21555360);
    assert (conv_in r t 17 == (if i = 0 then r.loop0_position_00_layer_i_1_1 4 else r.loop1_position_i_layer_17_1 (i - 1)));
    assert (rec_in r t 17 == (if i = 0 then r.loop0_position_00_layer_i_1_2 4 else r.loop1_position_i_layer_17_2 (i - 1)));
    assert (conv_out r t 17 == r.loop1_position_i_layer_17_1 i);
    assert (rec_out r t 17 == r.loop1_position_i_layer_17_2 i);
    ()

let layer_rel_1_18 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 18)
  = reveal_opaque (`%layer_rel) (layer_rel r t 18);
    let i = t - 1 in
    assert (lx r t 18 == r.loop1_position_i_layer_17_0 i);
    assert (lout r t 18 == r.loop1_position_i_layer_18_0 i);
    assert (wv r 18 == view r.input0 629466304 21555360);
    assert (conv_in r t 18 == (if i = 0 then r.loop0_position_00_layer_i_2_1 4 else r.loop1_position_i_layer_18_1 (i - 1)));
    assert (rec_in r t 18 == (if i = 0 then r.loop0_position_00_layer_i_2_2 4 else r.loop1_position_i_layer_18_2 (i - 1)));
    assert (conv_out r t 18 == r.loop1_position_i_layer_18_1 i);
    assert (rec_out r t 18 == r.loop1_position_i_layer_18_2 i);
    ()

let layer_rel_1_19 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 19)
  = reveal_opaque (`%layer_rel) (layer_rel r t 19);
    let i = t - 1 in
    assert (lx r t 19 == r.loop1_position_i_layer_18_0 i);
    assert (lout r t 19 == r.loop1_position_i_layer_19_0 i);
    assert (wv r 19 == view r.input0 651021664 18352640);
    assert (kv_in r t 19 == (if i = 0 then r.loop0_position_00_layer_i_3_1 4 else r.loop1_position_i_layer_19_1 (i - 1)));
    assert (kv_out r t 19 == r.loop1_position_i_layer_19_1 i);
    assert (pcos r t == r.loop1_position_i_embedding_1 i);
    assert (psin r t == r.loop1_position_i_embedding_2 i);
    ()

let layer_rel_1_20 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 20)
  = reveal_opaque (`%layer_rel) (layer_rel r t 20);
    let i = t - 1 in
    assert (lx r t 20 == r.loop1_position_i_layer_19_0 i);
    assert (lout r t 20 == r.loop1_position_i_layer_20_0 i);
    assert (wv r 20 == view r.input0 669374304 21555360);
    assert (conv_in r t 20 == (if i = 0 then r.loop0_position_00_layer_i_0_1 5 else r.loop1_position_i_layer_20_1 (i - 1)));
    assert (rec_in r t 20 == (if i = 0 then r.loop0_position_00_layer_i_0_2 5 else r.loop1_position_i_layer_20_2 (i - 1)));
    assert (conv_out r t 20 == r.loop1_position_i_layer_20_1 i);
    assert (rec_out r t 20 == r.loop1_position_i_layer_20_2 i);
    ()

let layer_rel_1_21 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 21)
  = reveal_opaque (`%layer_rel) (layer_rel r t 21);
    let i = t - 1 in
    assert (lx r t 21 == r.loop1_position_i_layer_20_0 i);
    assert (lout r t 21 == r.loop1_position_i_layer_21_0 i);
    assert (wv r 21 == view r.input0 690929664 21555360);
    assert (conv_in r t 21 == (if i = 0 then r.loop0_position_00_layer_i_1_1 5 else r.loop1_position_i_layer_21_1 (i - 1)));
    assert (rec_in r t 21 == (if i = 0 then r.loop0_position_00_layer_i_1_2 5 else r.loop1_position_i_layer_21_2 (i - 1)));
    assert (conv_out r t 21 == r.loop1_position_i_layer_21_1 i);
    assert (rec_out r t 21 == r.loop1_position_i_layer_21_2 i);
    ()

let layer_rel_1_22 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 22)
  = reveal_opaque (`%layer_rel) (layer_rel r t 22);
    let i = t - 1 in
    assert (lx r t 22 == r.loop1_position_i_layer_21_0 i);
    assert (lout r t 22 == r.loop1_position_i_layer_22_0 i);
    assert (wv r 22 == view r.input0 712485024 21555360);
    assert (conv_in r t 22 == (if i = 0 then r.loop0_position_00_layer_i_2_1 5 else r.loop1_position_i_layer_22_1 (i - 1)));
    assert (rec_in r t 22 == (if i = 0 then r.loop0_position_00_layer_i_2_2 5 else r.loop1_position_i_layer_22_2 (i - 1)));
    assert (conv_out r t 22 == r.loop1_position_i_layer_22_1 i);
    assert (rec_out r t 22 == r.loop1_position_i_layer_22_2 i);
    ()

let layer_rel_1_23 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1)) (ensures layer_rel r t 23)
  = reveal_opaque (`%layer_rel) (layer_rel r t 23);
    let i = t - 1 in
    assert (lx r t 23 == r.loop1_position_i_layer_22_0 i);
    assert (lout r t 23 == r.loop1_position_i_layer_23_0 i);
    assert (wv r 23 == view r.input0 734040384 18352640);
    assert (kv_in r t 23 == (if i = 0 then r.loop0_position_00_layer_i_3_1 5 else r.loop1_position_i_layer_23_1 (i - 1)));
    assert (kv_out r t 23 == r.loop1_position_i_layer_23_1 i);
    assert (pcos r t == r.loop1_position_i_embedding_1 i);
    assert (psin r t == r.loop1_position_i_embedding_2 i);
    ()

let layer_rel_1 (r:run) (t l:nat)
  : Lemma (requires 1 <= t /\ loop1_ok r (t - 1) /\ l < 24) (ensures layer_rel r t l)
  = if l = 0 then layer_rel_1_0 r t
    else if l = 1 then layer_rel_1_1 r t
    else if l = 2 then layer_rel_1_2 r t
    else if l = 3 then layer_rel_1_3 r t
    else if l = 4 then layer_rel_1_4 r t
    else if l = 5 then layer_rel_1_5 r t
    else if l = 6 then layer_rel_1_6 r t
    else if l = 7 then layer_rel_1_7 r t
    else if l = 8 then layer_rel_1_8 r t
    else if l = 9 then layer_rel_1_9 r t
    else if l = 10 then layer_rel_1_10 r t
    else if l = 11 then layer_rel_1_11 r t
    else if l = 12 then layer_rel_1_12 r t
    else if l = 13 then layer_rel_1_13 r t
    else if l = 14 then layer_rel_1_14 r t
    else if l = 15 then layer_rel_1_15 r t
    else if l = 16 then layer_rel_1_16 r t
    else if l = 17 then layer_rel_1_17 r t
    else if l = 18 then layer_rel_1_18 r t
    else if l = 19 then layer_rel_1_19 r t
    else if l = 20 then layer_rel_1_20 r t
    else if l = 21 then layer_rel_1_21 r t
    else if l = 22 then layer_rel_1_22 r t
    else layer_rel_1_23 r t

(*** Position-level boundaries *)

let utok (r:run) (t:int) : S.seq I64.t =
  if t < 0 then r.initialize_3
  else if t = 0 then r.position_00_update_0
  else if t < cap r then r.loop1_position_i_update_0 (t - 1)
  else r.loop2_position_i_update_0 (t - cap r)

let uhist (r:run) (t:int) : S.seq BF.t =
  if t < 0 then r.initialize_4
  else if t = 0 then r.position_00_update_1
  else if t < cap r then r.loop1_position_i_update_1 (t - 1)
  else r.loop2_position_i_update_1 (t - cap r)

let uarg (r:run) (t:nat) : S.seq I64.t =
  if t = 0 then r.position_00_argmax_0
  else if t < cap r then r.loop1_position_i_argmax_0 (t - 1)
  else r.loop2_position_i_argmax_0 (t - cap r)

let ufin (r:run) (t:nat) : S.seq BF.t =
  if t = 0 then r.position_00_final_0
  else if t < cap r then r.loop1_position_i_final_0 (t - 1)
  else if loop1_count r > 0 then r.loop1_position_i_final_0 (loop1_count r - 1) else r.position_00_final_0

let ulog (r:run) (t:nat) : S.seq BF.t =
  if t = 0 then r.position_00_final_1
  else if t < cap r then r.loop1_position_i_final_1 (t - 1)
  else if loop1_count r > 0 then r.loop1_position_i_final_1 (loop1_count r - 1) else r.position_00_final_1

let table_view (r:run) : S.seq BF.t = view r.input0 0 254279680

let norm_view (r:run) : S.seq BF.t = view r.input0 254279680 1024

[@@"opaque_to_smt"]
let init_rel (r:run) : prop =
  base_initialize_relation (table_view r) r.input1 r.input2 r.initialize_0 r.initialize_1 r.initialize_2
    r.initialize_3 r.initialize_4

[@@"opaque_to_smt"]
let emb_rel (r:run) (t:nat) : prop =
  base_embedding_rope_relation (table_view r) r.initialize_0 (utok r (t - 1)) t (lx r t 0) (pcos r t) (psin r t)

[@@"opaque_to_smt"]
let fin_rel (r:run) (t:nat) : prop =
  base_final_norm_projection_relation (table_view r) (norm_view r) (lx r t 24) (ufin r t) (ulog r t)

[@@"opaque_to_smt"]
let arg_rel (r:run) (t:nat) : prop =
  base_argmax_relation (ulog r t) (S.length r.input1) r.input2 t (uarg r t)

[@@"opaque_to_smt"]
let upd_rel (r:run) (t:nat) : prop =
  base_token_history_update_relation (utok r (t - 1)) (uhist r (t - 1)) (ufin r t) (uarg r t)
    (S.length r.input1) t (utok r t) (uhist r t)

(*** Extraction from family_ok, behind an opaque wrapper *)

[@@"opaque_to_smt"]
let fam (r:run) : prop = input_lengths r /\ family_ok r

let fam_lengths (r:run)
  : Lemma (requires fam r)
          (ensures input_lengths r /\ 1 <= cap r /\ cap r <= 12 /\ loop1_count r == cap r - 1 /\
                   loop2_count r == 12 - cap r)
  = reveal_opaque (`%fam) (fam r)

let fam_loop2 (r:run) (i:nat)
  : Lemma (requires family_ok r /\ i < loop2_count r) (ensures loop2_ok r i)
  = ()

let fam_loop0' (r:run) (i:nat)
  : Lemma (requires fam r /\ i < 6) (ensures loop0_ok r i)
  = reveal_opaque (`%fam) (fam r); fam_loop0 r i

let fam_loop1' (r:run) (i:nat)
  : Lemma (requires fam r /\ i < cap r - 1) (ensures loop1_ok r i)
  = reveal_opaque (`%fam) (fam r); fam_loop1 r i

let layer_rel_0' (r:run) (l:nat)
  : Lemma (requires loop0_ok r (l / 4) /\ l < 24) (ensures layer_rel r 0 l)
  = layer_rel_0 r (l / 4) (l % 4);
    assert (4 * (l / 4) + l % 4 == l)

let layer_rel_ok (r:run) (t l:nat)
  : Lemma (requires fam r /\ t < cap r /\ l < 24) (ensures layer_rel r t l)
  = if t = 0 then begin
      fam_loop0' r (l / 4);
      layer_rel_0' r l
    end else begin
      fam_loop1' r (t - 1);
      layer_rel_1 r t l
    end

let fam_loop2' (r:run) (i:nat)
  : Lemma (requires fam r /\ i < 12 - cap r) (ensures loop2_ok r i)
  = reveal_opaque (`%fam) (fam r); fam_loop2 r i

let init_rel_ok (r:run)
  : Lemma (requires fam r) (ensures init_rel r)
  = reveal_opaque (`%fam) (fam r);
    reveal_opaque (`%init_rel) (init_rel r)

let emb_rel_0 (r:run)
  : Lemma (requires fam r) (ensures emb_rel r 0)
  = reveal_opaque (`%fam) (fam r);
    reveal_opaque (`%emb_rel) (emb_rel r 0)

let lx_0_24 (r:run) : Lemma (lx r 0 24 == r.loop0_position_00_layer_i_3_0 5) = ()

let fin_rel_0 (r:run)
  : Lemma (requires fam r) (ensures fin_rel r 0)
  = lx_0_24 r;
    reveal_opaque (`%fam) (fam r);
    reveal_opaque (`%fin_rel) (fin_rel r 0)

let pos_rel_0 (r:run)
  : Lemma (requires fam r) (ensures arg_rel r 0 /\ upd_rel r 0)
  = reveal_opaque (`%fam) (fam r);
    reveal_opaque (`%arg_rel) (arg_rel r 0);
    reveal_opaque (`%upd_rel) (upd_rel r 0)

let emb_rel_1 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ t < cap r /\ loop1_ok r (t - 1)) (ensures emb_rel r t)
  = reveal_opaque (`%emb_rel) (emb_rel r t);
    let i = t - 1 in
    assert (utok r (t - 1) == (if i = 0 then r.position_00_update_0 else r.loop1_position_i_update_0 (i - 1)));
    assert (lx r t 0 == r.loop1_position_i_embedding_0 i)

let emb_rel_ok (r:run) (t:nat)
  : Lemma (requires fam r /\ t < cap r) (ensures emb_rel r t)
  = if t = 0 then emb_rel_0 r
    else begin
      fam_loop1' r (t - 1);
      emb_rel_1 r t
    end

let fin_rel_1 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ t < cap r /\ loop1_ok r (t - 1)) (ensures fin_rel r t)
  = reveal_opaque (`%fin_rel) (fin_rel r t);
    assert (lx r t 24 == r.loop1_position_i_layer_23_0 (t - 1))

let fin_rel_ok (r:run) (t:nat)
  : Lemma (requires fam r /\ t < cap r) (ensures fin_rel r t)
  = if t = 0 then fin_rel_0 r
    else begin
      fam_loop1' r (t - 1);
      fin_rel_1 r t
    end

let pos_rel_1 (r:run) (t:nat)
  : Lemma (requires 1 <= t /\ t < cap r /\ loop1_ok r (t - 1)) (ensures arg_rel r t /\ upd_rel r t)
  = reveal_opaque (`%arg_rel) (arg_rel r t);
    reveal_opaque (`%upd_rel) (upd_rel r t);
    let i = t - 1 in
    assert (utok r (t - 1) == (if i = 0 then r.position_00_update_0 else r.loop1_position_i_update_0 (i - 1)));
    assert (uhist r (t - 1) == (if i = 0 then r.position_00_update_1 else r.loop1_position_i_update_1 (i - 1)))

let pos_rel_2 (r:run) (t:nat)
  : Lemma (requires cap r <= t /\ 1 <= cap r /\ loop1_count r == cap r - 1 /\ loop2_ok r (t - cap r))
          (ensures arg_rel r t /\ upd_rel r t)
  = reveal_opaque (`%arg_rel) (arg_rel r t);
    reveal_opaque (`%upd_rel) (upd_rel r t);
    let i = t - cap r in
    assert (utok r (t - 1) == (if i = 0 then (if loop1_count r > 0 then r.loop1_position_i_update_0 (loop1_count r - 1)
                                              else r.position_00_update_0)
                               else r.loop2_position_i_update_0 (i - 1)));
    assert (uhist r (t - 1) == (if i = 0 then (if loop1_count r > 0 then r.loop1_position_i_update_1 (loop1_count r - 1)
                                               else r.position_00_update_1)
                                else r.loop2_position_i_update_1 (i - 1)))

let pos_rel_ok (r:run) (t:nat)
  : Lemma (requires fam r /\ t < 12) (ensures arg_rel r t /\ upd_rel r t)
  = fam_lengths r;
    if t = 0 then pos_rel_0 r
    else if t < cap r then begin
      fam_loop1' r (t - 1);
      pos_rel_1 r t
    end else begin
      fam_loop2' r (t - cap r);
      pos_rel_2 r t
    end

let outputs_ok (r:run)
  : Lemma (requires fam r) (ensures output0 r == utok r 11 /\ output1 r == uhist r 11)
  = fam_lengths r
