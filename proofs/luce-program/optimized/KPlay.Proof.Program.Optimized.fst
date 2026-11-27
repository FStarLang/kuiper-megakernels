module KPlay.Proof.Program.Optimized

(* Composition theorem for the optimized program. Each decode position is one
   instance of the generic layer chain (KPlay.Proof.OptChain.chain); the maps
   below only name the wiring buffers at the layer boundaries. *)

open FStar.Real
open Kuiper
open Kuiper.Approximates
open KPlay.SharedSpec
open KPlay.Proof.Wiring.Optimized
open KPlay.Proof.OptWin
open KPlay.Proof.OptApprox
open KPlay.Proof.OptState
open KPlay.Proof.OptChain
module S = FStar.Seq
module M = FStar.Math.Lemmas
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32
module I64 = FStar.Int64

#set-options "--fuel 0 --ifuel 0 --z3rlimit 50"

(*** Boundary buffers *)

(* loop-carried decoder state entering loop0 / loop1 iteration i *)
let crc0 (r:run) (i:nat) : S.seq F32.t = if i = 0 then r.call0_0_initialize_0 else r.loop0_s0000_pi_layer22_1 (i - 1)
let ccv0 (r:run) (i:nat) : S.seq F32.t = if i = 0 then r.call0_0_initialize_1 else r.loop0_s0000_pi_layer22_2 (i - 1)
let ck0 (r:run) (i:nat) : S.seq BF.t = if i = 0 then r.call0_0_initialize_2 else r.loop0_s0000_pi_layer23_1 (i - 1)
let cv0 (r:run) (i:nat) : S.seq BF.t = if i = 0 then r.call0_0_initialize_3 else r.loop0_s0000_pi_layer23_2 (i - 1)

let crc1 (r:run) (i:nat) : S.seq F32.t =
  if i = 0 then (if loop0_count r > 0 then (r.loop0_s0000_pi_layer22_1 (loop0_count r - 1)) else r.call0_0_initialize_0)
  else r.loop1_s0000_pi_layer22_1 (i - 1)
let ccv1 (r:run) (i:nat) : S.seq F32.t =
  if i = 0 then (if loop0_count r > 0 then (r.loop0_s0000_pi_layer22_2 (loop0_count r - 1)) else r.call0_0_initialize_1)
  else r.loop1_s0000_pi_layer22_2 (i - 1)
let ck1 (r:run) (i:nat) : S.seq BF.t =
  if i = 0 then (if loop0_count r > 0 then (r.loop0_s0000_pi_layer23_1 (loop0_count r - 1)) else r.call0_0_initialize_2)
  else r.loop1_s0000_pi_layer23_1 (i - 1)
let cv1 (r:run) (i:nat) : S.seq BF.t =
  if i = 0 then (if loop0_count r > 0 then (r.loop0_s0000_pi_layer23_2 (loop0_count r - 1)) else r.call0_0_initialize_3)
  else r.loop1_s0000_pi_layer23_2 (i - 1)

(* tokens and history entering loop1 iteration i *)
let tk (r:run) (i:nat) : S.seq I64.t = if i = 0 then r.call0_0_initialize_4 else r.loop1_s0000_pi_argmax_1 (i - 1)
let hh (r:run) (i:nat) : S.seq BF.t = if i = 0 then r.call0_0_initialize_5 else r.loop1_s0000_pi_projection_1 (i - 1)

(* decode position of loop1 iteration i *)
let pos1 (r:run) (i:nat) : nat = if S.length r.input1 >= 1 then S.length r.input1 - 1 + i else i

let xs0 (r:run) (i:nat) (l:nat) : S.seq F32.t =
  match l with
  | 0 -> r.loop0_s0000_pi_embedding_0 i
  | 1 -> r.loop0_s0000_pi_layer00_0 i
  | 2 -> r.loop0_s0000_pi_layer01_0 i
  | 3 -> r.loop0_s0000_pi_layer02_0 i
  | 4 -> r.loop0_s0000_pi_layer03_0 i
  | 5 -> r.loop0_s0000_pi_layer04_0 i
  | 6 -> r.loop0_s0000_pi_layer05_0 i
  | 7 -> r.loop0_s0000_pi_layer06_0 i
  | 8 -> r.loop0_s0000_pi_layer07_0 i
  | 9 -> r.loop0_s0000_pi_layer08_0 i
  | 10 -> r.loop0_s0000_pi_layer09_0 i
  | 11 -> r.loop0_s0000_pi_layer10_0 i
  | 12 -> r.loop0_s0000_pi_layer11_0 i
  | 13 -> r.loop0_s0000_pi_layer12_0 i
  | 14 -> r.loop0_s0000_pi_layer13_0 i
  | 15 -> r.loop0_s0000_pi_layer14_0 i
  | 16 -> r.loop0_s0000_pi_layer15_0 i
  | 17 -> r.loop0_s0000_pi_layer16_0 i
  | 18 -> r.loop0_s0000_pi_layer17_0 i
  | 19 -> r.loop0_s0000_pi_layer18_0 i
  | 20 -> r.loop0_s0000_pi_layer19_0 i
  | 21 -> r.loop0_s0000_pi_layer20_0 i
  | 22 -> r.loop0_s0000_pi_layer21_0 i
  | 23 -> r.loop0_s0000_pi_layer22_0 i
  | 24 -> r.loop0_s0000_pi_layer23_0 i
  | _ -> S.empty

let rcs0 (r:run) (i:nat) (l:nat) : S.seq F32.t =
  match l with
  | 0 -> crc0 r i
  | 1 -> r.loop0_s0000_pi_layer00_1 i
  | 2 -> r.loop0_s0000_pi_layer01_1 i
  | 3 -> r.loop0_s0000_pi_layer02_1 i
  | 4 -> r.loop0_s0000_pi_layer02_1 i
  | 5 -> r.loop0_s0000_pi_layer04_1 i
  | 6 -> r.loop0_s0000_pi_layer05_1 i
  | 7 -> r.loop0_s0000_pi_layer06_1 i
  | 8 -> r.loop0_s0000_pi_layer06_1 i
  | 9 -> r.loop0_s0000_pi_layer08_1 i
  | 10 -> r.loop0_s0000_pi_layer09_1 i
  | 11 -> r.loop0_s0000_pi_layer10_1 i
  | 12 -> r.loop0_s0000_pi_layer10_1 i
  | 13 -> r.loop0_s0000_pi_layer12_1 i
  | 14 -> r.loop0_s0000_pi_layer13_1 i
  | 15 -> r.loop0_s0000_pi_layer14_1 i
  | 16 -> r.loop0_s0000_pi_layer14_1 i
  | 17 -> r.loop0_s0000_pi_layer16_1 i
  | 18 -> r.loop0_s0000_pi_layer17_1 i
  | 19 -> r.loop0_s0000_pi_layer18_1 i
  | 20 -> r.loop0_s0000_pi_layer18_1 i
  | 21 -> r.loop0_s0000_pi_layer20_1 i
  | 22 -> r.loop0_s0000_pi_layer21_1 i
  | 23 -> r.loop0_s0000_pi_layer22_1 i
  | 24 -> r.loop0_s0000_pi_layer22_1 i
  | _ -> S.empty

let cvs0 (r:run) (i:nat) (l:nat) : S.seq F32.t =
  match l with
  | 0 -> ccv0 r i
  | 1 -> r.loop0_s0000_pi_layer00_2 i
  | 2 -> r.loop0_s0000_pi_layer01_2 i
  | 3 -> r.loop0_s0000_pi_layer02_2 i
  | 4 -> r.loop0_s0000_pi_layer02_2 i
  | 5 -> r.loop0_s0000_pi_layer04_2 i
  | 6 -> r.loop0_s0000_pi_layer05_2 i
  | 7 -> r.loop0_s0000_pi_layer06_2 i
  | 8 -> r.loop0_s0000_pi_layer06_2 i
  | 9 -> r.loop0_s0000_pi_layer08_2 i
  | 10 -> r.loop0_s0000_pi_layer09_2 i
  | 11 -> r.loop0_s0000_pi_layer10_2 i
  | 12 -> r.loop0_s0000_pi_layer10_2 i
  | 13 -> r.loop0_s0000_pi_layer12_2 i
  | 14 -> r.loop0_s0000_pi_layer13_2 i
  | 15 -> r.loop0_s0000_pi_layer14_2 i
  | 16 -> r.loop0_s0000_pi_layer14_2 i
  | 17 -> r.loop0_s0000_pi_layer16_2 i
  | 18 -> r.loop0_s0000_pi_layer17_2 i
  | 19 -> r.loop0_s0000_pi_layer18_2 i
  | 20 -> r.loop0_s0000_pi_layer18_2 i
  | 21 -> r.loop0_s0000_pi_layer20_2 i
  | 22 -> r.loop0_s0000_pi_layer21_2 i
  | 23 -> r.loop0_s0000_pi_layer22_2 i
  | 24 -> r.loop0_s0000_pi_layer22_2 i
  | _ -> S.empty

let ks0 (r:run) (i:nat) (l:nat) : S.seq BF.t =
  match l with
  | 0 -> ck0 r i
  | 1 -> ck0 r i
  | 2 -> ck0 r i
  | 3 -> ck0 r i
  | 4 -> r.loop0_s0000_pi_layer03_1 i
  | 5 -> r.loop0_s0000_pi_layer03_1 i
  | 6 -> r.loop0_s0000_pi_layer03_1 i
  | 7 -> r.loop0_s0000_pi_layer03_1 i
  | 8 -> r.loop0_s0000_pi_layer07_1 i
  | 9 -> r.loop0_s0000_pi_layer07_1 i
  | 10 -> r.loop0_s0000_pi_layer07_1 i
  | 11 -> r.loop0_s0000_pi_layer07_1 i
  | 12 -> r.loop0_s0000_pi_layer11_1 i
  | 13 -> r.loop0_s0000_pi_layer11_1 i
  | 14 -> r.loop0_s0000_pi_layer11_1 i
  | 15 -> r.loop0_s0000_pi_layer11_1 i
  | 16 -> r.loop0_s0000_pi_layer15_1 i
  | 17 -> r.loop0_s0000_pi_layer15_1 i
  | 18 -> r.loop0_s0000_pi_layer15_1 i
  | 19 -> r.loop0_s0000_pi_layer15_1 i
  | 20 -> r.loop0_s0000_pi_layer19_1 i
  | 21 -> r.loop0_s0000_pi_layer19_1 i
  | 22 -> r.loop0_s0000_pi_layer19_1 i
  | 23 -> r.loop0_s0000_pi_layer19_1 i
  | 24 -> r.loop0_s0000_pi_layer23_1 i
  | _ -> S.empty

let vs0 (r:run) (i:nat) (l:nat) : S.seq BF.t =
  match l with
  | 0 -> cv0 r i
  | 1 -> cv0 r i
  | 2 -> cv0 r i
  | 3 -> cv0 r i
  | 4 -> r.loop0_s0000_pi_layer03_2 i
  | 5 -> r.loop0_s0000_pi_layer03_2 i
  | 6 -> r.loop0_s0000_pi_layer03_2 i
  | 7 -> r.loop0_s0000_pi_layer03_2 i
  | 8 -> r.loop0_s0000_pi_layer07_2 i
  | 9 -> r.loop0_s0000_pi_layer07_2 i
  | 10 -> r.loop0_s0000_pi_layer07_2 i
  | 11 -> r.loop0_s0000_pi_layer07_2 i
  | 12 -> r.loop0_s0000_pi_layer11_2 i
  | 13 -> r.loop0_s0000_pi_layer11_2 i
  | 14 -> r.loop0_s0000_pi_layer11_2 i
  | 15 -> r.loop0_s0000_pi_layer11_2 i
  | 16 -> r.loop0_s0000_pi_layer15_2 i
  | 17 -> r.loop0_s0000_pi_layer15_2 i
  | 18 -> r.loop0_s0000_pi_layer15_2 i
  | 19 -> r.loop0_s0000_pi_layer15_2 i
  | 20 -> r.loop0_s0000_pi_layer19_2 i
  | 21 -> r.loop0_s0000_pi_layer19_2 i
  | 22 -> r.loop0_s0000_pi_layer19_2 i
  | 23 -> r.loop0_s0000_pi_layer19_2 i
  | 24 -> r.loop0_s0000_pi_layer23_2 i
  | _ -> S.empty

let xs1 (r:run) (i:nat) (l:nat) : S.seq F32.t =
  match l with
  | 0 -> r.loop1_s0000_pi_embedding_0 i
  | 1 -> r.loop1_s0000_pi_layer00_0 i
  | 2 -> r.loop1_s0000_pi_layer01_0 i
  | 3 -> r.loop1_s0000_pi_layer02_0 i
  | 4 -> r.loop1_s0000_pi_layer03_0 i
  | 5 -> r.loop1_s0000_pi_layer04_0 i
  | 6 -> r.loop1_s0000_pi_layer05_0 i
  | 7 -> r.loop1_s0000_pi_layer06_0 i
  | 8 -> r.loop1_s0000_pi_layer07_0 i
  | 9 -> r.loop1_s0000_pi_layer08_0 i
  | 10 -> r.loop1_s0000_pi_layer09_0 i
  | 11 -> r.loop1_s0000_pi_layer10_0 i
  | 12 -> r.loop1_s0000_pi_layer11_0 i
  | 13 -> r.loop1_s0000_pi_layer12_0 i
  | 14 -> r.loop1_s0000_pi_layer13_0 i
  | 15 -> r.loop1_s0000_pi_layer14_0 i
  | 16 -> r.loop1_s0000_pi_layer15_0 i
  | 17 -> r.loop1_s0000_pi_layer16_0 i
  | 18 -> r.loop1_s0000_pi_layer17_0 i
  | 19 -> r.loop1_s0000_pi_layer18_0 i
  | 20 -> r.loop1_s0000_pi_layer19_0 i
  | 21 -> r.loop1_s0000_pi_layer20_0 i
  | 22 -> r.loop1_s0000_pi_layer21_0 i
  | 23 -> r.loop1_s0000_pi_layer22_0 i
  | 24 -> r.loop1_s0000_pi_layer23_0 i
  | _ -> S.empty

let rcs1 (r:run) (i:nat) (l:nat) : S.seq F32.t =
  match l with
  | 0 -> crc1 r i
  | 1 -> r.loop1_s0000_pi_layer00_1 i
  | 2 -> r.loop1_s0000_pi_layer01_1 i
  | 3 -> r.loop1_s0000_pi_layer02_1 i
  | 4 -> r.loop1_s0000_pi_layer02_1 i
  | 5 -> r.loop1_s0000_pi_layer04_1 i
  | 6 -> r.loop1_s0000_pi_layer05_1 i
  | 7 -> r.loop1_s0000_pi_layer06_1 i
  | 8 -> r.loop1_s0000_pi_layer06_1 i
  | 9 -> r.loop1_s0000_pi_layer08_1 i
  | 10 -> r.loop1_s0000_pi_layer09_1 i
  | 11 -> r.loop1_s0000_pi_layer10_1 i
  | 12 -> r.loop1_s0000_pi_layer10_1 i
  | 13 -> r.loop1_s0000_pi_layer12_1 i
  | 14 -> r.loop1_s0000_pi_layer13_1 i
  | 15 -> r.loop1_s0000_pi_layer14_1 i
  | 16 -> r.loop1_s0000_pi_layer14_1 i
  | 17 -> r.loop1_s0000_pi_layer16_1 i
  | 18 -> r.loop1_s0000_pi_layer17_1 i
  | 19 -> r.loop1_s0000_pi_layer18_1 i
  | 20 -> r.loop1_s0000_pi_layer18_1 i
  | 21 -> r.loop1_s0000_pi_layer20_1 i
  | 22 -> r.loop1_s0000_pi_layer21_1 i
  | 23 -> r.loop1_s0000_pi_layer22_1 i
  | 24 -> r.loop1_s0000_pi_layer22_1 i
  | _ -> S.empty

let cvs1 (r:run) (i:nat) (l:nat) : S.seq F32.t =
  match l with
  | 0 -> ccv1 r i
  | 1 -> r.loop1_s0000_pi_layer00_2 i
  | 2 -> r.loop1_s0000_pi_layer01_2 i
  | 3 -> r.loop1_s0000_pi_layer02_2 i
  | 4 -> r.loop1_s0000_pi_layer02_2 i
  | 5 -> r.loop1_s0000_pi_layer04_2 i
  | 6 -> r.loop1_s0000_pi_layer05_2 i
  | 7 -> r.loop1_s0000_pi_layer06_2 i
  | 8 -> r.loop1_s0000_pi_layer06_2 i
  | 9 -> r.loop1_s0000_pi_layer08_2 i
  | 10 -> r.loop1_s0000_pi_layer09_2 i
  | 11 -> r.loop1_s0000_pi_layer10_2 i
  | 12 -> r.loop1_s0000_pi_layer10_2 i
  | 13 -> r.loop1_s0000_pi_layer12_2 i
  | 14 -> r.loop1_s0000_pi_layer13_2 i
  | 15 -> r.loop1_s0000_pi_layer14_2 i
  | 16 -> r.loop1_s0000_pi_layer14_2 i
  | 17 -> r.loop1_s0000_pi_layer16_2 i
  | 18 -> r.loop1_s0000_pi_layer17_2 i
  | 19 -> r.loop1_s0000_pi_layer18_2 i
  | 20 -> r.loop1_s0000_pi_layer18_2 i
  | 21 -> r.loop1_s0000_pi_layer20_2 i
  | 22 -> r.loop1_s0000_pi_layer21_2 i
  | 23 -> r.loop1_s0000_pi_layer22_2 i
  | 24 -> r.loop1_s0000_pi_layer22_2 i
  | _ -> S.empty

let ks1 (r:run) (i:nat) (l:nat) : S.seq BF.t =
  match l with
  | 0 -> ck1 r i
  | 1 -> ck1 r i
  | 2 -> ck1 r i
  | 3 -> ck1 r i
  | 4 -> r.loop1_s0000_pi_layer03_1 i
  | 5 -> r.loop1_s0000_pi_layer03_1 i
  | 6 -> r.loop1_s0000_pi_layer03_1 i
  | 7 -> r.loop1_s0000_pi_layer03_1 i
  | 8 -> r.loop1_s0000_pi_layer07_1 i
  | 9 -> r.loop1_s0000_pi_layer07_1 i
  | 10 -> r.loop1_s0000_pi_layer07_1 i
  | 11 -> r.loop1_s0000_pi_layer07_1 i
  | 12 -> r.loop1_s0000_pi_layer11_1 i
  | 13 -> r.loop1_s0000_pi_layer11_1 i
  | 14 -> r.loop1_s0000_pi_layer11_1 i
  | 15 -> r.loop1_s0000_pi_layer11_1 i
  | 16 -> r.loop1_s0000_pi_layer15_1 i
  | 17 -> r.loop1_s0000_pi_layer15_1 i
  | 18 -> r.loop1_s0000_pi_layer15_1 i
  | 19 -> r.loop1_s0000_pi_layer15_1 i
  | 20 -> r.loop1_s0000_pi_layer19_1 i
  | 21 -> r.loop1_s0000_pi_layer19_1 i
  | 22 -> r.loop1_s0000_pi_layer19_1 i
  | 23 -> r.loop1_s0000_pi_layer19_1 i
  | 24 -> r.loop1_s0000_pi_layer23_1 i
  | _ -> S.empty

let vs1 (r:run) (i:nat) (l:nat) : S.seq BF.t =
  match l with
  | 0 -> cv1 r i
  | 1 -> cv1 r i
  | 2 -> cv1 r i
  | 3 -> cv1 r i
  | 4 -> r.loop1_s0000_pi_layer03_2 i
  | 5 -> r.loop1_s0000_pi_layer03_2 i
  | 6 -> r.loop1_s0000_pi_layer03_2 i
  | 7 -> r.loop1_s0000_pi_layer03_2 i
  | 8 -> r.loop1_s0000_pi_layer07_2 i
  | 9 -> r.loop1_s0000_pi_layer07_2 i
  | 10 -> r.loop1_s0000_pi_layer07_2 i
  | 11 -> r.loop1_s0000_pi_layer07_2 i
  | 12 -> r.loop1_s0000_pi_layer11_2 i
  | 13 -> r.loop1_s0000_pi_layer11_2 i
  | 14 -> r.loop1_s0000_pi_layer11_2 i
  | 15 -> r.loop1_s0000_pi_layer11_2 i
  | 16 -> r.loop1_s0000_pi_layer15_2 i
  | 17 -> r.loop1_s0000_pi_layer15_2 i
  | 18 -> r.loop1_s0000_pi_layer15_2 i
  | 19 -> r.loop1_s0000_pi_layer15_2 i
  | 20 -> r.loop1_s0000_pi_layer19_2 i
  | 21 -> r.loop1_s0000_pi_layer19_2 i
  | 22 -> r.loop1_s0000_pi_layer19_2 i
  | 23 -> r.loop1_s0000_pi_layer19_2 i
  | 24 -> r.loop1_s0000_pi_layer23_2 i
  | _ -> S.empty


let e0 (r:run) (i:nat) : pexec =
  { pw = r.input0; xs = xs0 r i; rcs = rcs0 r i; cvs = cvs0 r i; ks = ks0 r i; vs = vs0 r i;
    pcs = r.loop0_s0000_pi_embedding_1 i; psn = r.loop0_s0000_pi_embedding_2 i; pt = i }

let e1 (r:run) (i:nat) : pexec =
  { pw = r.input0; xs = xs1 r i; rcs = rcs1 r i; cvs = cvs1 r i; ks = ks1 r i; vs = vs1 r i;
    pcs = r.loop1_s0000_pi_embedding_1 i; psn = r.loop1_s0000_pi_embedding_2 i; pt = pos1 r i }

(*** Wiring bridge: the layer calls of one iteration are the chain's layer_ok *)

let attn_bridge (e:pexec) (l:nat) (w:S.seq BF.t) (x:S.seq F32.t) (k v:S.seq BF.t) (cs sn:S.seq F32.t)
                (fa pos:int) (x':S.seq F32.t) (k' v':S.seq BF.t)
  : Lemma (requires optimized_attention_decoder_relation w x k v cs sn 0 fa pos x' k' v' /\
                    is_attn l /\ w == view e.pw (layer_off l) attn_len /\ x == e.xs l /\ k == e.ks l /\ v == e.vs l /\
                    cs == e.pcs /\ sn == e.psn /\ fa == l / 4 /\ pos == e.pt /\
                    x' == e.xs (l + 1) /\ k' == e.ks (l + 1) /\ v' == e.vs (l + 1) /\
                    e.rcs (l + 1) == e.rcs l /\ e.cvs (l + 1) == e.cvs l)
          (ensures layer_ok e l)
  = reveal_opaque (`%layer_ok) (layer_ok e l)

let dn_bridge (e:pexec) (l:nat) (w:S.seq BF.t) (x rc cv:S.seq F32.t) (dn:int) (x' rc' cv':S.seq F32.t)
  : Lemma (requires optimized_recurrent_decoder_relation w x rc cv 0 dn x' rc' cv' /\
                    not (is_attn l) /\ w == view e.pw (layer_off l) dn_len /\ x == e.xs l /\ rc == e.rcs l /\ cv == e.cvs l /\
                    dn == dn_index l /\ x' == e.xs (l + 1) /\ rc' == e.rcs (l + 1) /\ cv' == e.cvs (l + 1) /\
                    e.ks (l + 1) == e.ks l /\ e.vs (l + 1) == e.vs l)
          (ensures layer_ok e l)
  = reveal_opaque (`%layer_ok) (layer_ok e l)

let bridge0 (r:run) (i:nat) (l:nat)
  : Lemma (requires loop0_ok r i /\ l < 24) (ensures layer_ok (e0 r i) l)
  = match l with
    | 0 -> dn_bridge (e0 r i) 0 (view r.input0 254280704 21555360) (r.loop0_s0000_pi_embedding_0 i) (if i = 0 then r.call0_0_initialize_0 else r.loop0_s0000_pi_layer22_1 (i - 1)) (if i = 0 then r.call0_0_initialize_1 else r.loop0_s0000_pi_layer22_2 (i - 1)) 0 (r.loop0_s0000_pi_layer00_0 i) (r.loop0_s0000_pi_layer00_1 i) (r.loop0_s0000_pi_layer00_2 i)
    | 1 -> dn_bridge (e0 r i) 1 (view r.input0 275836064 21555360) (r.loop0_s0000_pi_layer00_0 i) (r.loop0_s0000_pi_layer00_1 i) (r.loop0_s0000_pi_layer00_2 i) 1 (r.loop0_s0000_pi_layer01_0 i) (r.loop0_s0000_pi_layer01_1 i) (r.loop0_s0000_pi_layer01_2 i)
    | 2 -> dn_bridge (e0 r i) 2 (view r.input0 297391424 21555360) (r.loop0_s0000_pi_layer01_0 i) (r.loop0_s0000_pi_layer01_1 i) (r.loop0_s0000_pi_layer01_2 i) 2 (r.loop0_s0000_pi_layer02_0 i) (r.loop0_s0000_pi_layer02_1 i) (r.loop0_s0000_pi_layer02_2 i)
    | 3 -> attn_bridge (e0 r i) 3 (view r.input0 318946784 18352640) (r.loop0_s0000_pi_layer02_0 i) (if i = 0 then r.call0_0_initialize_2 else r.loop0_s0000_pi_layer23_1 (i - 1)) (if i = 0 then r.call0_0_initialize_3 else r.loop0_s0000_pi_layer23_2 (i - 1)) (r.loop0_s0000_pi_embedding_1 i) (r.loop0_s0000_pi_embedding_2 i) 0 (0 + 1 * i) (r.loop0_s0000_pi_layer03_0 i) (r.loop0_s0000_pi_layer03_1 i) (r.loop0_s0000_pi_layer03_2 i)
    | 4 -> dn_bridge (e0 r i) 4 (view r.input0 337299424 21555360) (r.loop0_s0000_pi_layer03_0 i) (r.loop0_s0000_pi_layer02_1 i) (r.loop0_s0000_pi_layer02_2 i) 3 (r.loop0_s0000_pi_layer04_0 i) (r.loop0_s0000_pi_layer04_1 i) (r.loop0_s0000_pi_layer04_2 i)
    | 5 -> dn_bridge (e0 r i) 5 (view r.input0 358854784 21555360) (r.loop0_s0000_pi_layer04_0 i) (r.loop0_s0000_pi_layer04_1 i) (r.loop0_s0000_pi_layer04_2 i) 4 (r.loop0_s0000_pi_layer05_0 i) (r.loop0_s0000_pi_layer05_1 i) (r.loop0_s0000_pi_layer05_2 i)
    | 6 -> dn_bridge (e0 r i) 6 (view r.input0 380410144 21555360) (r.loop0_s0000_pi_layer05_0 i) (r.loop0_s0000_pi_layer05_1 i) (r.loop0_s0000_pi_layer05_2 i) 5 (r.loop0_s0000_pi_layer06_0 i) (r.loop0_s0000_pi_layer06_1 i) (r.loop0_s0000_pi_layer06_2 i)
    | 7 -> attn_bridge (e0 r i) 7 (view r.input0 401965504 18352640) (r.loop0_s0000_pi_layer06_0 i) (r.loop0_s0000_pi_layer03_1 i) (r.loop0_s0000_pi_layer03_2 i) (r.loop0_s0000_pi_embedding_1 i) (r.loop0_s0000_pi_embedding_2 i) 1 (0 + 1 * i) (r.loop0_s0000_pi_layer07_0 i) (r.loop0_s0000_pi_layer07_1 i) (r.loop0_s0000_pi_layer07_2 i)
    | 8 -> dn_bridge (e0 r i) 8 (view r.input0 420318144 21555360) (r.loop0_s0000_pi_layer07_0 i) (r.loop0_s0000_pi_layer06_1 i) (r.loop0_s0000_pi_layer06_2 i) 6 (r.loop0_s0000_pi_layer08_0 i) (r.loop0_s0000_pi_layer08_1 i) (r.loop0_s0000_pi_layer08_2 i)
    | 9 -> dn_bridge (e0 r i) 9 (view r.input0 441873504 21555360) (r.loop0_s0000_pi_layer08_0 i) (r.loop0_s0000_pi_layer08_1 i) (r.loop0_s0000_pi_layer08_2 i) 7 (r.loop0_s0000_pi_layer09_0 i) (r.loop0_s0000_pi_layer09_1 i) (r.loop0_s0000_pi_layer09_2 i)
    | 10 -> dn_bridge (e0 r i) 10 (view r.input0 463428864 21555360) (r.loop0_s0000_pi_layer09_0 i) (r.loop0_s0000_pi_layer09_1 i) (r.loop0_s0000_pi_layer09_2 i) 8 (r.loop0_s0000_pi_layer10_0 i) (r.loop0_s0000_pi_layer10_1 i) (r.loop0_s0000_pi_layer10_2 i)
    | 11 -> attn_bridge (e0 r i) 11 (view r.input0 484984224 18352640) (r.loop0_s0000_pi_layer10_0 i) (r.loop0_s0000_pi_layer07_1 i) (r.loop0_s0000_pi_layer07_2 i) (r.loop0_s0000_pi_embedding_1 i) (r.loop0_s0000_pi_embedding_2 i) 2 (0 + 1 * i) (r.loop0_s0000_pi_layer11_0 i) (r.loop0_s0000_pi_layer11_1 i) (r.loop0_s0000_pi_layer11_2 i)
    | 12 -> dn_bridge (e0 r i) 12 (view r.input0 503336864 21555360) (r.loop0_s0000_pi_layer11_0 i) (r.loop0_s0000_pi_layer10_1 i) (r.loop0_s0000_pi_layer10_2 i) 9 (r.loop0_s0000_pi_layer12_0 i) (r.loop0_s0000_pi_layer12_1 i) (r.loop0_s0000_pi_layer12_2 i)
    | 13 -> dn_bridge (e0 r i) 13 (view r.input0 524892224 21555360) (r.loop0_s0000_pi_layer12_0 i) (r.loop0_s0000_pi_layer12_1 i) (r.loop0_s0000_pi_layer12_2 i) 10 (r.loop0_s0000_pi_layer13_0 i) (r.loop0_s0000_pi_layer13_1 i) (r.loop0_s0000_pi_layer13_2 i)
    | 14 -> dn_bridge (e0 r i) 14 (view r.input0 546447584 21555360) (r.loop0_s0000_pi_layer13_0 i) (r.loop0_s0000_pi_layer13_1 i) (r.loop0_s0000_pi_layer13_2 i) 11 (r.loop0_s0000_pi_layer14_0 i) (r.loop0_s0000_pi_layer14_1 i) (r.loop0_s0000_pi_layer14_2 i)
    | 15 -> attn_bridge (e0 r i) 15 (view r.input0 568002944 18352640) (r.loop0_s0000_pi_layer14_0 i) (r.loop0_s0000_pi_layer11_1 i) (r.loop0_s0000_pi_layer11_2 i) (r.loop0_s0000_pi_embedding_1 i) (r.loop0_s0000_pi_embedding_2 i) 3 (0 + 1 * i) (r.loop0_s0000_pi_layer15_0 i) (r.loop0_s0000_pi_layer15_1 i) (r.loop0_s0000_pi_layer15_2 i)
    | 16 -> dn_bridge (e0 r i) 16 (view r.input0 586355584 21555360) (r.loop0_s0000_pi_layer15_0 i) (r.loop0_s0000_pi_layer14_1 i) (r.loop0_s0000_pi_layer14_2 i) 12 (r.loop0_s0000_pi_layer16_0 i) (r.loop0_s0000_pi_layer16_1 i) (r.loop0_s0000_pi_layer16_2 i)
    | 17 -> dn_bridge (e0 r i) 17 (view r.input0 607910944 21555360) (r.loop0_s0000_pi_layer16_0 i) (r.loop0_s0000_pi_layer16_1 i) (r.loop0_s0000_pi_layer16_2 i) 13 (r.loop0_s0000_pi_layer17_0 i) (r.loop0_s0000_pi_layer17_1 i) (r.loop0_s0000_pi_layer17_2 i)
    | 18 -> dn_bridge (e0 r i) 18 (view r.input0 629466304 21555360) (r.loop0_s0000_pi_layer17_0 i) (r.loop0_s0000_pi_layer17_1 i) (r.loop0_s0000_pi_layer17_2 i) 14 (r.loop0_s0000_pi_layer18_0 i) (r.loop0_s0000_pi_layer18_1 i) (r.loop0_s0000_pi_layer18_2 i)
    | 19 -> attn_bridge (e0 r i) 19 (view r.input0 651021664 18352640) (r.loop0_s0000_pi_layer18_0 i) (r.loop0_s0000_pi_layer15_1 i) (r.loop0_s0000_pi_layer15_2 i) (r.loop0_s0000_pi_embedding_1 i) (r.loop0_s0000_pi_embedding_2 i) 4 (0 + 1 * i) (r.loop0_s0000_pi_layer19_0 i) (r.loop0_s0000_pi_layer19_1 i) (r.loop0_s0000_pi_layer19_2 i)
    | 20 -> dn_bridge (e0 r i) 20 (view r.input0 669374304 21555360) (r.loop0_s0000_pi_layer19_0 i) (r.loop0_s0000_pi_layer18_1 i) (r.loop0_s0000_pi_layer18_2 i) 15 (r.loop0_s0000_pi_layer20_0 i) (r.loop0_s0000_pi_layer20_1 i) (r.loop0_s0000_pi_layer20_2 i)
    | 21 -> dn_bridge (e0 r i) 21 (view r.input0 690929664 21555360) (r.loop0_s0000_pi_layer20_0 i) (r.loop0_s0000_pi_layer20_1 i) (r.loop0_s0000_pi_layer20_2 i) 16 (r.loop0_s0000_pi_layer21_0 i) (r.loop0_s0000_pi_layer21_1 i) (r.loop0_s0000_pi_layer21_2 i)
    | 22 -> dn_bridge (e0 r i) 22 (view r.input0 712485024 21555360) (r.loop0_s0000_pi_layer21_0 i) (r.loop0_s0000_pi_layer21_1 i) (r.loop0_s0000_pi_layer21_2 i) 17 (r.loop0_s0000_pi_layer22_0 i) (r.loop0_s0000_pi_layer22_1 i) (r.loop0_s0000_pi_layer22_2 i)
    | 23 -> attn_bridge (e0 r i) 23 (view r.input0 734040384 18352640) (r.loop0_s0000_pi_layer22_0 i) (r.loop0_s0000_pi_layer19_1 i) (r.loop0_s0000_pi_layer19_2 i) (r.loop0_s0000_pi_embedding_1 i) (r.loop0_s0000_pi_embedding_2 i) 5 (0 + 1 * i) (r.loop0_s0000_pi_layer23_0 i) (r.loop0_s0000_pi_layer23_1 i) (r.loop0_s0000_pi_layer23_2 i)
    | _ -> ()

let bridge1 (r:run) (i:nat) (l:nat)
  : Lemma (requires 1 <= S.length r.input1 /\ loop1_ok r i /\ l < 24) (ensures layer_ok (e1 r i) l)
  = match l with
    | 0 -> dn_bridge (e1 r i) 0 (view r.input0 254280704 21555360) (r.loop1_s0000_pi_embedding_0 i) (if i = 0 then (if loop0_count r > 0 then (r.loop0_s0000_pi_layer22_1 (loop0_count r - 1)) else r.call0_0_initialize_0) else r.loop1_s0000_pi_layer22_1 (i - 1)) (if i = 0 then (if loop0_count r > 0 then (r.loop0_s0000_pi_layer22_2 (loop0_count r - 1)) else r.call0_0_initialize_1) else r.loop1_s0000_pi_layer22_2 (i - 1)) 0 (r.loop1_s0000_pi_layer00_0 i) (r.loop1_s0000_pi_layer00_1 i) (r.loop1_s0000_pi_layer00_2 i)
    | 1 -> dn_bridge (e1 r i) 1 (view r.input0 275836064 21555360) (r.loop1_s0000_pi_layer00_0 i) (r.loop1_s0000_pi_layer00_1 i) (r.loop1_s0000_pi_layer00_2 i) 1 (r.loop1_s0000_pi_layer01_0 i) (r.loop1_s0000_pi_layer01_1 i) (r.loop1_s0000_pi_layer01_2 i)
    | 2 -> dn_bridge (e1 r i) 2 (view r.input0 297391424 21555360) (r.loop1_s0000_pi_layer01_0 i) (r.loop1_s0000_pi_layer01_1 i) (r.loop1_s0000_pi_layer01_2 i) 2 (r.loop1_s0000_pi_layer02_0 i) (r.loop1_s0000_pi_layer02_1 i) (r.loop1_s0000_pi_layer02_2 i)
    | 3 -> attn_bridge (e1 r i) 3 (view r.input0 318946784 18352640) (r.loop1_s0000_pi_layer02_0 i) (if i = 0 then (if loop0_count r > 0 then (r.loop0_s0000_pi_layer23_1 (loop0_count r - 1)) else r.call0_0_initialize_2) else r.loop1_s0000_pi_layer23_1 (i - 1)) (if i = 0 then (if loop0_count r > 0 then (r.loop0_s0000_pi_layer23_2 (loop0_count r - 1)) else r.call0_0_initialize_3) else r.loop1_s0000_pi_layer23_2 (i - 1)) (r.loop1_s0000_pi_embedding_1 i) (r.loop1_s0000_pi_embedding_2 i) 0 ((S.length r.input1 - 1) + 1 * i) (r.loop1_s0000_pi_layer03_0 i) (r.loop1_s0000_pi_layer03_1 i) (r.loop1_s0000_pi_layer03_2 i)
    | 4 -> dn_bridge (e1 r i) 4 (view r.input0 337299424 21555360) (r.loop1_s0000_pi_layer03_0 i) (r.loop1_s0000_pi_layer02_1 i) (r.loop1_s0000_pi_layer02_2 i) 3 (r.loop1_s0000_pi_layer04_0 i) (r.loop1_s0000_pi_layer04_1 i) (r.loop1_s0000_pi_layer04_2 i)
    | 5 -> dn_bridge (e1 r i) 5 (view r.input0 358854784 21555360) (r.loop1_s0000_pi_layer04_0 i) (r.loop1_s0000_pi_layer04_1 i) (r.loop1_s0000_pi_layer04_2 i) 4 (r.loop1_s0000_pi_layer05_0 i) (r.loop1_s0000_pi_layer05_1 i) (r.loop1_s0000_pi_layer05_2 i)
    | 6 -> dn_bridge (e1 r i) 6 (view r.input0 380410144 21555360) (r.loop1_s0000_pi_layer05_0 i) (r.loop1_s0000_pi_layer05_1 i) (r.loop1_s0000_pi_layer05_2 i) 5 (r.loop1_s0000_pi_layer06_0 i) (r.loop1_s0000_pi_layer06_1 i) (r.loop1_s0000_pi_layer06_2 i)
    | 7 -> attn_bridge (e1 r i) 7 (view r.input0 401965504 18352640) (r.loop1_s0000_pi_layer06_0 i) (r.loop1_s0000_pi_layer03_1 i) (r.loop1_s0000_pi_layer03_2 i) (r.loop1_s0000_pi_embedding_1 i) (r.loop1_s0000_pi_embedding_2 i) 1 ((S.length r.input1 - 1) + 1 * i) (r.loop1_s0000_pi_layer07_0 i) (r.loop1_s0000_pi_layer07_1 i) (r.loop1_s0000_pi_layer07_2 i)
    | 8 -> dn_bridge (e1 r i) 8 (view r.input0 420318144 21555360) (r.loop1_s0000_pi_layer07_0 i) (r.loop1_s0000_pi_layer06_1 i) (r.loop1_s0000_pi_layer06_2 i) 6 (r.loop1_s0000_pi_layer08_0 i) (r.loop1_s0000_pi_layer08_1 i) (r.loop1_s0000_pi_layer08_2 i)
    | 9 -> dn_bridge (e1 r i) 9 (view r.input0 441873504 21555360) (r.loop1_s0000_pi_layer08_0 i) (r.loop1_s0000_pi_layer08_1 i) (r.loop1_s0000_pi_layer08_2 i) 7 (r.loop1_s0000_pi_layer09_0 i) (r.loop1_s0000_pi_layer09_1 i) (r.loop1_s0000_pi_layer09_2 i)
    | 10 -> dn_bridge (e1 r i) 10 (view r.input0 463428864 21555360) (r.loop1_s0000_pi_layer09_0 i) (r.loop1_s0000_pi_layer09_1 i) (r.loop1_s0000_pi_layer09_2 i) 8 (r.loop1_s0000_pi_layer10_0 i) (r.loop1_s0000_pi_layer10_1 i) (r.loop1_s0000_pi_layer10_2 i)
    | 11 -> attn_bridge (e1 r i) 11 (view r.input0 484984224 18352640) (r.loop1_s0000_pi_layer10_0 i) (r.loop1_s0000_pi_layer07_1 i) (r.loop1_s0000_pi_layer07_2 i) (r.loop1_s0000_pi_embedding_1 i) (r.loop1_s0000_pi_embedding_2 i) 2 ((S.length r.input1 - 1) + 1 * i) (r.loop1_s0000_pi_layer11_0 i) (r.loop1_s0000_pi_layer11_1 i) (r.loop1_s0000_pi_layer11_2 i)
    | 12 -> dn_bridge (e1 r i) 12 (view r.input0 503336864 21555360) (r.loop1_s0000_pi_layer11_0 i) (r.loop1_s0000_pi_layer10_1 i) (r.loop1_s0000_pi_layer10_2 i) 9 (r.loop1_s0000_pi_layer12_0 i) (r.loop1_s0000_pi_layer12_1 i) (r.loop1_s0000_pi_layer12_2 i)
    | 13 -> dn_bridge (e1 r i) 13 (view r.input0 524892224 21555360) (r.loop1_s0000_pi_layer12_0 i) (r.loop1_s0000_pi_layer12_1 i) (r.loop1_s0000_pi_layer12_2 i) 10 (r.loop1_s0000_pi_layer13_0 i) (r.loop1_s0000_pi_layer13_1 i) (r.loop1_s0000_pi_layer13_2 i)
    | 14 -> dn_bridge (e1 r i) 14 (view r.input0 546447584 21555360) (r.loop1_s0000_pi_layer13_0 i) (r.loop1_s0000_pi_layer13_1 i) (r.loop1_s0000_pi_layer13_2 i) 11 (r.loop1_s0000_pi_layer14_0 i) (r.loop1_s0000_pi_layer14_1 i) (r.loop1_s0000_pi_layer14_2 i)
    | 15 -> attn_bridge (e1 r i) 15 (view r.input0 568002944 18352640) (r.loop1_s0000_pi_layer14_0 i) (r.loop1_s0000_pi_layer11_1 i) (r.loop1_s0000_pi_layer11_2 i) (r.loop1_s0000_pi_embedding_1 i) (r.loop1_s0000_pi_embedding_2 i) 3 ((S.length r.input1 - 1) + 1 * i) (r.loop1_s0000_pi_layer15_0 i) (r.loop1_s0000_pi_layer15_1 i) (r.loop1_s0000_pi_layer15_2 i)
    | 16 -> dn_bridge (e1 r i) 16 (view r.input0 586355584 21555360) (r.loop1_s0000_pi_layer15_0 i) (r.loop1_s0000_pi_layer14_1 i) (r.loop1_s0000_pi_layer14_2 i) 12 (r.loop1_s0000_pi_layer16_0 i) (r.loop1_s0000_pi_layer16_1 i) (r.loop1_s0000_pi_layer16_2 i)
    | 17 -> dn_bridge (e1 r i) 17 (view r.input0 607910944 21555360) (r.loop1_s0000_pi_layer16_0 i) (r.loop1_s0000_pi_layer16_1 i) (r.loop1_s0000_pi_layer16_2 i) 13 (r.loop1_s0000_pi_layer17_0 i) (r.loop1_s0000_pi_layer17_1 i) (r.loop1_s0000_pi_layer17_2 i)
    | 18 -> dn_bridge (e1 r i) 18 (view r.input0 629466304 21555360) (r.loop1_s0000_pi_layer17_0 i) (r.loop1_s0000_pi_layer17_1 i) (r.loop1_s0000_pi_layer17_2 i) 14 (r.loop1_s0000_pi_layer18_0 i) (r.loop1_s0000_pi_layer18_1 i) (r.loop1_s0000_pi_layer18_2 i)
    | 19 -> attn_bridge (e1 r i) 19 (view r.input0 651021664 18352640) (r.loop1_s0000_pi_layer18_0 i) (r.loop1_s0000_pi_layer15_1 i) (r.loop1_s0000_pi_layer15_2 i) (r.loop1_s0000_pi_embedding_1 i) (r.loop1_s0000_pi_embedding_2 i) 4 ((S.length r.input1 - 1) + 1 * i) (r.loop1_s0000_pi_layer19_0 i) (r.loop1_s0000_pi_layer19_1 i) (r.loop1_s0000_pi_layer19_2 i)
    | 20 -> dn_bridge (e1 r i) 20 (view r.input0 669374304 21555360) (r.loop1_s0000_pi_layer19_0 i) (r.loop1_s0000_pi_layer18_1 i) (r.loop1_s0000_pi_layer18_2 i) 15 (r.loop1_s0000_pi_layer20_0 i) (r.loop1_s0000_pi_layer20_1 i) (r.loop1_s0000_pi_layer20_2 i)
    | 21 -> dn_bridge (e1 r i) 21 (view r.input0 690929664 21555360) (r.loop1_s0000_pi_layer20_0 i) (r.loop1_s0000_pi_layer20_1 i) (r.loop1_s0000_pi_layer20_2 i) 16 (r.loop1_s0000_pi_layer21_0 i) (r.loop1_s0000_pi_layer21_1 i) (r.loop1_s0000_pi_layer21_2 i)
    | 22 -> dn_bridge (e1 r i) 22 (view r.input0 712485024 21555360) (r.loop1_s0000_pi_layer21_0 i) (r.loop1_s0000_pi_layer21_1 i) (r.loop1_s0000_pi_layer21_2 i) 17 (r.loop1_s0000_pi_layer22_0 i) (r.loop1_s0000_pi_layer22_1 i) (r.loop1_s0000_pi_layer22_2 i)
    | 23 -> attn_bridge (e1 r i) 23 (view r.input0 734040384 18352640) (r.loop1_s0000_pi_layer22_0 i) (r.loop1_s0000_pi_layer19_1 i) (r.loop1_s0000_pi_layer19_2 i) (r.loop1_s0000_pi_embedding_1 i) (r.loop1_s0000_pi_embedding_2 i) 5 ((S.length r.input1 - 1) + 1 * i) (r.loop1_s0000_pi_layer23_0 i) (r.loop1_s0000_pi_layer23_1 i) (r.loop1_s0000_pi_layer23_2 i)
    | _ -> ()

(*** Reference run *)

(* token consumed at decode position p: prompt, then the token written by the
   argmax of loop1 iteration p - P *)
let toks_r (r:run) (p:nat) : int =
  if p < S.length r.input1 then tok_at r.input1 p
  else tok_at (tk r (p - S.length r.input1 + 1)) (p - S.length r.input1)

let ref_s (r:run) (rw:S.seq real) (n:nat) : rstate = fst (ref_run opt_numerics rw (toks_r r) n)

#push-options "--fuel 1"
let ref_run_zero (nm:numerics) (w:S.seq real) (toks:nat -> int)
  : Lemma (ref_run nm w toks 0 == (init_state, S.empty))
  = ()

let ref_run_succ (nm:numerics) (w:S.seq real) (toks:nat -> int) (n:nat)
  : Lemma (ref_run nm w toks (n + 1) == ref_step nm w (toks n) n (fst (ref_run nm w toks n)))
  = ()

let rec ref_run_ext (nm:numerics) (w:S.seq real) (t1 t2:nat -> int) (n:nat)
  : Lemma (requires forall (p:nat). p < n ==> t1 p == t2 p)
          (ensures ref_run nm w t1 n == ref_run nm w t2 n) (decreases n)
  = if n = 0 then () else ref_run_ext nm w t1 t2 (n - 1)
#pop-options

(*** Relation eliminators (one per family, instantiated with the wiring arguments) *)

let init_elim (prompt:S.seq I64.t) (steps:int) (rc cv:S.seq F32.t) (k v:S.seq BF.t)
              (tokens:S.seq I64.t) (history:S.seq BF.t)
  : Lemma (requires optimized_initialize_relation prompt steps rc cv k v tokens history /\
                    1 <= S.length prompt /\ 1 <= steps /\ steps <= 12)
          (ensures pinv rc cv k v init_state 0 /\ S.length tokens == steps /\
                   tok_at tokens 0 == tok_at prompt 0 /\ S.length history == steps * 1024)
  = zero_approx_f32 rc; zero_approx_f32 cv; zero_approx_bf k; zero_approx_bf v;
    init_enc (); kv_init 0; kv_init 512

[@@"opaque_to_smt"]
let emb_ok (table:S.seq BF.t) (prompt tokens:S.seq I64.t) (p:int) (hidden cos sin:S.seq F32.t) : prop =
  optimized_embedding_relation table prompt tokens p hidden cos sin

[@@"opaque_to_smt"]
let proj_ok (table normw:S.seq BF.t) (x:S.seq F32.t) (history:S.seq BF.t) (d:int)
            (scores:S.seq F32.t) (history':S.seq BF.t) : prop =
  optimized_final_projection_relation table normw x history d scores history'

[@@"opaque_to_smt"]
let argmax_ok (scores:S.seq F32.t) (tokens:S.seq I64.t) (d:int) (selected tokens':S.seq I64.t) : prop =
  optimized_argmax_relation scores tokens d selected tokens'

let emb_elim (table:S.seq BF.t) (prompt tokens:S.seq I64.t) (p:int) (hidden cos sin:S.seq F32.t) (rt:S.seq real)
  : Lemma (requires emb_ok table prompt tokens p hidden cos sin /\
                    S.length table == table_len /\ 1 <= S.length prompt /\ 1 <= S.length tokens /\
                    0 <= p /\ p < 12 /\ p < S.length prompt + S.length tokens - 1 /\
                    valid_token (opt_emb_token prompt tokens p) /\ table %~ rt)
          (ensures S.length hidden == 1024 /\ S.length cos == 32 /\ S.length sin == 32 /\
                   hidden %~ embed rt (opt_emb_token prompt tokens p) /\
                   cos %~ rope_cos opt_numerics p /\ sin %~ rope_sin opt_numerics p)
  = reveal_opaque (`%emb_ok) (emb_ok table prompt tokens p hidden cos sin)

let proj_elim (table normw:S.seq BF.t) (x:S.seq F32.t) (history:S.seq BF.t) (d:int)
              (scores:S.seq F32.t) (history':S.seq BF.t) (rt rn rx:S.seq real)
  : Lemma (requires proj_ok table normw x history d scores history' /\
                    S.length table == table_len /\ S.length normw == 1024 /\ S.length x == 1024 /\
                    S.length history % 1024 == 0 /\ 0 <= d /\ d < S.length history / 1024 /\
                    table %~ rt /\ normw %~ rn /\ x %~ rx)
          (ensures S.length scores == 248320 /\ S.length history' == S.length history /\
                   (forall (k:nat). k < S.length history /\ k / 1024 <> d ==> S.index history' k == S.index history k) /\
                   scores %~ logits opt_numerics rt 0 rn 0 rx /\ bf_row history' d %~ rx)
  = reveal_opaque (`%proj_ok) (proj_ok table normw x history d scores history')

let argmax_elim (scores:S.seq F32.t) (tokens:S.seq I64.t) (d:int) (selected tokens':S.seq I64.t)
  : Lemma (requires argmax_ok scores tokens d selected tokens' /\
                    S.length scores == 248320 /\ 0 <= d /\ d < S.length tokens)
          (ensures S.length tokens' == S.length tokens /\ allowed_choice scores (tok_at tokens' d) /\
                   (forall (i:nat). i < S.length tokens /\ i <> d ==> S.index tokens' i == S.index tokens i))
  = reveal_opaque (`%argmax_ok) (argmax_ok scores tokens d selected tokens')

let inact_elim (rc cv:S.seq F32.t) (k v:S.seq BF.t) (tokens:S.seq I64.t) (history:S.seq BF.t) (p l:int)
               (rc' cv':S.seq F32.t) (k' v':S.seq BF.t) (tokens':S.seq I64.t) (history':S.seq BF.t)
  : Lemma (requires optimized_inactive_decode_relation rc cv k v tokens history p l rc' cv' k' v' tokens' history' /\
                    S.length rc == opt_rec_len /\ S.length cv == opt_conv_len /\
                    S.length k == opt_kv_len /\ S.length v == opt_kv_len /\
                    1 <= S.length tokens /\ S.length history == S.length tokens * 1024 /\
                    0 < l /\ l + S.length tokens - 1 <= p /\ p < 12)
          (ensures rc' == rc /\ cv' == cv /\ k' == k /\ v' == v /\ tokens' == tokens /\ history' == history)
  = ()

(* the embedding table and final-norm windows of the packed weights *)
let win_facts (w:S.seq BF.t) (rw:S.seq real)
  : Lemma (requires S.length w == weights_len /\ w %~ rw)
          (ensures S.length (view w 0 table_len) == table_len /\ view w 0 table_len %~ S.slice rw 0 table_len /\
                   S.length (view w final_norm_off 1024) == 1024 /\
                   view w final_norm_off 1024 %~ S.slice rw final_norm_off (final_norm_off + 1024) /\
                   wwin (S.slice rw 0 table_len) 0 rw 0 table_len /\
                   wwin (S.slice rw final_norm_off (final_norm_off + 1024)) 0 rw final_norm_off 1024)
  = view_approx w rw 0 table_len;
    view_approx w rw final_norm_off 1024;
    slice_win rw 0 table_len;
    slice_win rw final_norm_off 1024

(*** The calls of one loop iteration, with the wiring hidden behind opaque predicates *)

let wire0r (r:run) (i:nat) : prop =
  emb_ok (view r.input0 0 254279680) r.input1 r.call0_0_initialize_4 (0 + 1 * i)
    (r.loop0_s0000_pi_embedding_0 i) (r.loop0_s0000_pi_embedding_1 i) (r.loop0_s0000_pi_embedding_2 i) /\
  (forall (l:nat). l < 24 ==> layer_ok (e0 r i) l)

let wire0_intro (r:run) (i:nat)
  : Lemma (requires loop0_ok r i) (ensures wire0r r i)
  = reveal_opaque (`%emb_ok) (emb_ok (view r.input0 0 254279680) r.input1 r.call0_0_initialize_4 (0 + 1 * i)
    (r.loop0_s0000_pi_embedding_0 i) (r.loop0_s0000_pi_embedding_1 i) (r.loop0_s0000_pi_embedding_2 i));
    introduce forall (l:nat). l < 24 ==> layer_ok (e0 r i) l
    with introduce _ ==> _ with bridge0 r i l

let wire1r (r:run) (i:nat) : prop =
  emb_ok (view r.input0 0 254279680) r.input1
    (if i = 0 then r.call0_0_initialize_4 else r.loop1_s0000_pi_argmax_1 (i - 1)) ((S.length r.input1 - 1) + 1 * i)
    (r.loop1_s0000_pi_embedding_0 i) (r.loop1_s0000_pi_embedding_1 i) (r.loop1_s0000_pi_embedding_2 i) /\
  proj_ok (view r.input0 0 254279680) (view r.input0 254279680 1024) (r.loop1_s0000_pi_layer23_0 i)
    (if i = 0 then r.call0_0_initialize_5 else r.loop1_s0000_pi_projection_1 (i - 1)) (0 + 1 * i)
    (r.loop1_s0000_pi_projection_0 i) (r.loop1_s0000_pi_projection_1 i) /\
  argmax_ok (r.loop1_s0000_pi_projection_0 i)
    (if i = 0 then r.call0_0_initialize_4 else r.loop1_s0000_pi_argmax_1 (i - 1)) (0 + 1 * i)
    (r.loop1_s0000_pi_argmax_0 i) (r.loop1_s0000_pi_argmax_1 i) /\
  (forall (l:nat). l < 24 ==> layer_ok (e1 r i) l)

let wire1_intro (r:run) (i:nat)
  : Lemma (requires 1 <= S.length r.input1 /\ loop1_ok r i) (ensures wire1r r i)
  = reveal_opaque (`%emb_ok) (emb_ok (view r.input0 0 254279680) r.input1
    (if i = 0 then r.call0_0_initialize_4 else r.loop1_s0000_pi_argmax_1 (i - 1)) ((S.length r.input1 - 1) + 1 * i)
    (r.loop1_s0000_pi_embedding_0 i) (r.loop1_s0000_pi_embedding_1 i) (r.loop1_s0000_pi_embedding_2 i));
    reveal_opaque (`%proj_ok) (proj_ok (view r.input0 0 254279680) (view r.input0 254279680 1024) (r.loop1_s0000_pi_layer23_0 i)
    (if i = 0 then r.call0_0_initialize_5 else r.loop1_s0000_pi_projection_1 (i - 1)) (0 + 1 * i)
    (r.loop1_s0000_pi_projection_0 i) (r.loop1_s0000_pi_projection_1 i));
    reveal_opaque (`%argmax_ok) (argmax_ok (r.loop1_s0000_pi_projection_0 i)
    (if i = 0 then r.call0_0_initialize_4 else r.loop1_s0000_pi_argmax_1 (i - 1)) (0 + 1 * i)
    (r.loop1_s0000_pi_argmax_0 i) (r.loop1_s0000_pi_argmax_1 i));
    introduce forall (l:nat). l < 24 ==> layer_ok (e1 r i) l
    with introduce _ ==> _ with bridge1 r i l

let emb_cong (t:S.seq BF.t) (pr tk1 tk2:S.seq I64.t) (p1 p2:int) (h c sn:S.seq F32.t)
  : Lemma (requires emb_ok t pr tk1 p1 h c sn /\ tk1 == tk2 /\ p1 == p2) (ensures emb_ok t pr tk2 p2 h c sn)
  = ()

let proj_cong (t n:S.seq BF.t) (x:S.seq F32.t) (h1 h2:S.seq BF.t) (d1 d2:int) (sc:S.seq F32.t) (g1 g2:S.seq BF.t)
  : Lemma (requires proj_ok t n x h1 d1 sc g1 /\ h1 == h2 /\ d1 == d2 /\ g1 == g2) (ensures proj_ok t n x h2 d2 sc g2)
  = ()

let argmax_cong (sc:S.seq F32.t) (t1 t2:S.seq I64.t) (d1 d2:int) (sel u1 u2:S.seq I64.t)
  : Lemma (requires argmax_ok sc t1 d1 sel u1 /\ t1 == t2 /\ d1 == d2 /\ u1 == u2) (ensures argmax_ok sc t2 d2 sel u2)
  = ()

(* the same calls with the loop-carried arguments named by tk / hh / pos1 *)
let wire0 (r:run) (i:nat) : prop =
  emb_ok (view r.input0 0 254279680) r.input1 (tk r 0) i
    (r.loop0_s0000_pi_embedding_0 i) (r.loop0_s0000_pi_embedding_1 i) (r.loop0_s0000_pi_embedding_2 i) /\
  (forall (l:nat). l < 24 ==> layer_ok (e0 r i) l)

let wire0_norm (r:run) (i:nat)
  : Lemma (requires wire0r r i) (ensures wire0 r i)
  = emb_cong (view r.input0 0 254279680) r.input1 r.call0_0_initialize_4 (tk r 0) (0 + 1 * i) i
      (r.loop0_s0000_pi_embedding_0 i) (r.loop0_s0000_pi_embedding_1 i) (r.loop0_s0000_pi_embedding_2 i)

let wire1 (r:run) (i:nat) : prop =
  emb_ok (view r.input0 0 254279680) r.input1 (tk r i) (pos1 r i)
    (r.loop1_s0000_pi_embedding_0 i) (r.loop1_s0000_pi_embedding_1 i) (r.loop1_s0000_pi_embedding_2 i) /\
  proj_ok (view r.input0 0 254279680) (view r.input0 254279680 1024) (r.loop1_s0000_pi_layer23_0 i)
    (hh r i) i (r.loop1_s0000_pi_projection_0 i) (hh r (i + 1)) /\
  argmax_ok (r.loop1_s0000_pi_projection_0 i) (tk r i) i (r.loop1_s0000_pi_argmax_0 i) (tk r (i + 1)) /\
  (forall (l:nat). l < 24 ==> layer_ok (e1 r i) l)

let wire1_norm (r:run) (i:nat)
  : Lemma (requires 1 <= S.length r.input1 /\ wire1r r i) (ensures wire1 r i)
  = emb_cong (view r.input0 0 254279680) r.input1
      (if i = 0 then r.call0_0_initialize_4 else r.loop1_s0000_pi_argmax_1 (i - 1)) (tk r i)
      ((S.length r.input1 - 1) + 1 * i) (pos1 r i)
      (r.loop1_s0000_pi_embedding_0 i) (r.loop1_s0000_pi_embedding_1 i) (r.loop1_s0000_pi_embedding_2 i);
    proj_cong (view r.input0 0 254279680) (view r.input0 254279680 1024) (r.loop1_s0000_pi_layer23_0 i)
      (if i = 0 then r.call0_0_initialize_5 else r.loop1_s0000_pi_projection_1 (i - 1)) (hh r i) (0 + 1 * i) i
      (r.loop1_s0000_pi_projection_0 i) (r.loop1_s0000_pi_projection_1 i) (hh r (i + 1));
    argmax_cong (r.loop1_s0000_pi_projection_0 i)
      (if i = 0 then r.call0_0_initialize_4 else r.loop1_s0000_pi_argmax_1 (i - 1)) (tk r i) (0 + 1 * i) i
      (r.loop1_s0000_pi_argmax_0 i) (r.loop1_s0000_pi_argmax_1 i) (tk r (i + 1))

(*** Loop 0: prompt positions 0 .. P-2 *)

let init_ok (r:run) : prop =
  S.length (tk r 0) == r.input2 /\ tok_at (tk r 0) 0 == tok_at r.input1 0 /\
  S.length (hh r 0) == r.input2 * 1024 /\
  pinv (crc0 r 0) (ccv0 r 0) (ck0 r 0) (cv0 r 0) init_state 0

let pre (r:run) (rw:S.seq real) : prop =
  input_lengths r /\ valid_tokens r.input1 /\ r.input0 %~ rw /\ init_ok r

[@@"opaque_to_smt"]
let loops_ok (r:run) : prop =
  (forall (i:nat). i < loop0_count r ==> loop0_ok r i) /\
  (forall (i:nat). i < loop1_count r ==> loop1_ok r i)

let loops_intro (r:run)
  : Lemma (requires family_ok r) (ensures loops_ok r)
  = reveal_opaque (`%loops_ok) (loops_ok r)

let loops_wire0 (r:run) (i:nat)
  : Lemma (requires loops_ok r /\ i < loop0_count r) (ensures wire0 r i)
  = reveal_opaque (`%loops_ok) (loops_ok r);
    wire0_intro r i;
    wire0_norm r i

let loops_wire1 (r:run) (i:nat)
  : Lemma (requires 1 <= S.length r.input1 /\ loops_ok r /\ i < loop1_count r) (ensures wire1 r i)
  = reveal_opaque (`%loops_ok) (loops_ok r);
    wire1_intro r i;
    wire1_norm r i

let pinv0 (r:run) (rw:S.seq real) (i:nat) : prop =
  pinv (crc0 r i) (ccv0 r i) (ck0 r i) (cv0 r i) (ref_s r rw i) i

let tok0 (r:run) (i:nat)
  : Lemma (requires input_lengths r /\ valid_tokens r.input1 /\ tok_at (tk r 0) 0 == tok_at r.input1 0 /\
                    i < S.length r.input1 - 1)
          (ensures opt_emb_token r.input1 (tk r 0) i == toks_r r i /\
                   valid_token (toks_r r i))
  = ()

let emb0 (r:run) (rw:S.seq real) (i:nat)
  : Lemma (requires input_lengths r /\ r.input0 %~ rw /\ S.length (tk r 0) == r.input2 /\
                    i < S.length r.input1 - 1 /\
                    opt_emb_token r.input1 (tk r 0) i == toks_r r i /\
                    valid_token (toks_r r i) /\
                    emb_ok (view r.input0 0 254279680) r.input1 (tk r 0) i
                      (r.loop0_s0000_pi_embedding_0 i) (r.loop0_s0000_pi_embedding_1 i) (r.loop0_s0000_pi_embedding_2 i))
          (ensures pctx (e0 r i) rw /\ S.length ((e0 r i).xs 0) == 1024 /\
                   (e0 r i).xs 0 %~ embed rw (toks_r r i))
  = let rt = S.slice rw 0 table_len in
    win_facts r.input0 rw;
    embed_ext rt rw (toks_r r i);
    emb_elim (view r.input0 0 254279680) r.input1 (tk r 0) i
      (r.loop0_s0000_pi_embedding_0 i) (r.loop0_s0000_pi_embedding_1 i) (r.loop0_s0000_pi_embedding_2 i) rt

let step0 (r:run) (rw:S.seq real) (i:nat)
  : Lemma (requires pre r rw /\ i < S.length r.input1 - 1 /\ wire0 r i /\ pinv0 r rw i)
          (ensures pinv0 r rw (i + 1))
  = let tok = toks_r r i in
    tok0 r i;
    emb0 r rw i;
    position (e0 r i) rw tok (ref_s r rw i);
    ref_run_succ opt_numerics rw (toks_r r) i

let rec loop0_inv (r:run) (rw:S.seq real) (i:nat)
  : Lemma (requires pre r rw /\ loops_ok r /\ i <= S.length r.input1 - 1)
          (ensures pinv0 r rw i) (decreases i)
  = if i = 0 then ref_run_zero opt_numerics rw (toks_r r)
    else begin
      loop0_inv r rw (i - 1);
      loops_wire0 r (i - 1);
      step0 r rw (i - 1)
    end

(*** Loop 1: decode steps, positions P-1 .. P-2+steps *)

let pinv1 (r:run) (rw:S.seq real) (i:nat) : prop =
  pinv (crc1 r i) (ccv1 r i) (ck1 r i) (cv1 r i) (ref_s r rw (pos1 r i)) (pos1 r i)

let indep (r:run) (i:nat) : prop =
  S.length (tk r i) == r.input2 /\ S.length (hh r i) == r.input2 * 1024 /\
  (i > 0 ==> valid_token (tok_at (tk r i) (i - 1)))

(* iteration i writes only token i and history row i *)
[@@"opaque_to_smt"]
let stab (r:run) (i:nat) : prop =
  S.length (tk r (i + 1)) == S.length (tk r i) /\ S.length (hh r (i + 1)) == S.length (hh r i) /\
  (forall (j:int). j <> i ==> tok_at (tk r (i + 1)) j == tok_at (tk r i) j) /\
  (forall (k:nat). k < S.length (hh r i) /\ k < S.length (hh r (i + 1)) /\ k / 1024 <> i ==>
     S.index (hh r (i + 1)) k == S.index (hh r i) k)

let hcur (r:run) (rw:S.seq real) (i:nat) : S.seq real =
  ref_hidden opt_numerics rw (toks_r r) (pos1 r i)

[@@"opaque_to_smt"]
let outs (r:run) (rw:S.seq real) (i:nat) : prop =
  bf_row (hh r (i + 1)) i %~ hcur r rw i /\
  S.length (r.loop1_s0000_pi_projection_0 i) == 248320 /\
  allowed_choice (r.loop1_s0000_pi_projection_0 i) (tok_at (tk r (i + 1)) i) /\
  r.loop1_s0000_pi_projection_0 i %~ ref_logits opt_numerics rw (hcur r rw i)

let emb_tok1 (r:run) (i:nat)
  : Lemma (requires input_lengths r /\ valid_tokens r.input1 /\ init_ok r /\ indep r i)
          (ensures opt_emb_token r.input1 (tk r i) (pos1 r i) == toks_r r (pos1 r i) /\
                   valid_token (toks_r r (pos1 r i)))
  = ()

let emb1 (r:run) (rw:S.seq real) (i:nat)
  : Lemma (requires input_lengths r /\ r.input0 %~ rw /\ S.length (tk r i) == r.input2 /\ i < r.input2 /\
                    opt_emb_token r.input1 (tk r i) (pos1 r i) == toks_r r (pos1 r i) /\
                    valid_token (toks_r r (pos1 r i)) /\
                    emb_ok (view r.input0 0 254279680) r.input1 (tk r i) (pos1 r i)
                      (r.loop1_s0000_pi_embedding_0 i) (r.loop1_s0000_pi_embedding_1 i) (r.loop1_s0000_pi_embedding_2 i))
          (ensures pctx (e1 r i) rw /\ S.length ((e1 r i).xs 0) == 1024 /\
                   (e1 r i).xs 0 %~ embed rw (toks_r r (pos1 r i)))
  = let rt = S.slice rw 0 table_len in
    win_facts r.input0 rw;
    embed_ext rt rw (toks_r r (pos1 r i));
    emb_elim (view r.input0 0 254279680) r.input1 (tk r i) (pos1 r i)
      (r.loop1_s0000_pi_embedding_0 i) (r.loop1_s0000_pi_embedding_1 i) (r.loop1_s0000_pi_embedding_2 i) rt

let proj1 (r:run) (rw:S.seq real) (i:nat) (h:S.seq real)
  : Lemma (requires input_lengths r /\ r.input0 %~ rw /\ i < r.input2 /\ S.length (hh r i) == r.input2 * 1024 /\
                    S.length (r.loop1_s0000_pi_layer23_0 i) == 1024 /\ r.loop1_s0000_pi_layer23_0 i %~ h /\
                    proj_ok (view r.input0 0 254279680) (view r.input0 254279680 1024) (r.loop1_s0000_pi_layer23_0 i)
                      (hh r i) i (r.loop1_s0000_pi_projection_0 i) (hh r (i + 1)))
          (ensures S.length (r.loop1_s0000_pi_projection_0 i) == 248320 /\
                   r.loop1_s0000_pi_projection_0 i %~ ref_logits opt_numerics rw h /\
                   S.length (hh r (i + 1)) == S.length (hh r i) /\
                   bf_row (hh r (i + 1)) i %~ h /\
                   (forall (k:nat). k < S.length (hh r i) /\ k < S.length (hh r (i + 1)) /\ k / 1024 <> i ==>
                      S.index (hh r (i + 1)) k == S.index (hh r i) k))
  = let rt = S.slice rw 0 table_len in
    let rn = S.slice rw final_norm_off (final_norm_off + 1024) in
    win_facts r.input0 rw;
    M.multiple_modulo_lemma r.input2 1024;
    M.multiple_division_lemma r.input2 1024;
    proj_elim (view r.input0 0 254279680) (view r.input0 254279680 1024) (r.loop1_s0000_pi_layer23_0 i)
      (hh r i) i (r.loop1_s0000_pi_projection_0 i) (hh r (i + 1)) rt rn h;
    logits_ext opt_numerics rt 0 rn 0 rw 0 rw final_norm_off h

let argmax1 (r:run) (i:nat)
  : Lemma (requires i < r.input2 /\ S.length (tk r i) == r.input2 /\
                    S.length (r.loop1_s0000_pi_projection_0 i) == 248320 /\
                    argmax_ok (r.loop1_s0000_pi_projection_0 i) (tk r i) i (r.loop1_s0000_pi_argmax_0 i) (tk r (i + 1)))
          (ensures S.length (tk r (i + 1)) == r.input2 /\
                   allowed_choice (r.loop1_s0000_pi_projection_0 i) (tok_at (tk r (i + 1)) i) /\
                   valid_token (tok_at (tk r (i + 1)) i) /\
                   (forall (j:int). j <> i ==> tok_at (tk r (i + 1)) j == tok_at (tk r i) j))
  = argmax_elim (r.loop1_s0000_pi_projection_0 i) (tk r i) i (r.loop1_s0000_pi_argmax_0 i) (tk r (i + 1))

let outs_intro (r:run) (rw:S.seq real) (i:nat) (h:S.seq real)
  : Lemma (requires h == hcur r rw i /\ bf_row (hh r (i + 1)) i %~ h /\
                    S.length (r.loop1_s0000_pi_projection_0 i) == 248320 /\
                    allowed_choice (r.loop1_s0000_pi_projection_0 i) (tok_at (tk r (i + 1)) i) /\
                    r.loop1_s0000_pi_projection_0 i %~ ref_logits opt_numerics rw h)
          (ensures outs r rw i)
  = reveal_opaque (`%outs) (outs r rw i)

let stab_intro (r:run) (i:nat)
  : Lemma (requires S.length (tk r (i + 1)) == S.length (tk r i) /\ S.length (hh r (i + 1)) == S.length (hh r i) /\
                    (forall (j:int). j <> i ==> tok_at (tk r (i + 1)) j == tok_at (tk r i) j) /\
                    (forall (k:nat). k < S.length (hh r i) /\ k < S.length (hh r (i + 1)) /\ k / 1024 <> i ==>
                       S.index (hh r (i + 1)) k == S.index (hh r i) k))
          (ensures stab r i)
  = reveal_opaque (`%stab) (stab r i)

let outs_valid (r:run) (rw:S.seq real) (j k:nat)
  : Lemma (requires outs r rw j /\ j < k /\ S.length (tk r k) == k /\ tok_at (tk r k) j == tok_at (tk r (j + 1)) j)
          (ensures valid_token (I64.v (S.index (tk r k) j)))
  = reveal_opaque (`%outs) (outs r rw j)

let pinv1_next (r:run) (rw:S.seq real) (i:nat) (st:rstate)
  : Lemma (requires 1 <= S.length r.input1 /\ st == ref_s r rw (pos1 r i + 1) /\
                    pinv ((e1 r i).rcs 24) ((e1 r i).cvs 24) ((e1 r i).ks 24) ((e1 r i).vs 24) st (pos1 r i + 1))
          (ensures pinv1 r rw (i + 1))
  = ()

let step1 (r:run) (rw:S.seq real) (i:nat)
  : Lemma (requires pre r rw /\ i < r.input2 /\ wire1 r i /\ pinv1 r rw i /\ indep r i)
          (ensures pinv1 r rw (i + 1) /\ indep r (i + 1) /\ stab r i /\ outs r rw i)
  = let pt = pos1 r i in
    let tok = toks_r r pt in
    let s = ref_s r rw pt in
    emb_tok1 r i;
    emb1 r rw i;
    position (e1 r i) rw tok s;
    ref_run_succ opt_numerics rw (toks_r r) pt;
    let h = snd (ref_step opt_numerics rw tok pt s) in
    assert (hcur r rw i == h);
    assert (ref_s r rw (pt + 1) == fst (ref_step opt_numerics rw tok pt s));
    pinv1_next r rw i (fst (ref_step opt_numerics rw tok pt s));
    proj1 r rw i h;
    argmax1 r i;
    stab_intro r i;
    outs_intro r rw i h

[@@"opaque_to_smt"]
let done1 (r:run) (rw:S.seq real) (i:nat) : prop =
  forall (j:nat). j < i ==> stab r j /\ outs r rw j

let done1_zero (r:run) (rw:S.seq real)
  : Lemma (done1 r rw 0)
  = reveal_opaque (`%done1) (done1 r rw 0)

let done1_succ (r:run) (rw:S.seq real) (i:nat)
  : Lemma (requires done1 r rw i /\ stab r i /\ outs r rw i) (ensures done1 r rw (i + 1))
  = reveal_opaque (`%done1) (done1 r rw i);
    reveal_opaque (`%done1) (done1 r rw (i + 1))

let done1_elim (r:run) (rw:S.seq real) (i j:nat)
  : Lemma (requires done1 r rw i /\ j < i) (ensures outs r rw j)
  = reveal_opaque (`%done1) (done1 r rw i)

let done1_stab (r:run) (rw:S.seq real) (i:nat)
  : Lemma (requires done1 r rw i) (ensures forall (m:nat). m < i ==> stab r m)
  = reveal_opaque (`%done1) (done1 r rw i)

let rec loop1_all (r:run) (rw:S.seq real) (i:nat)
  : Lemma (requires pre r rw /\ loops_ok r /\ i <= r.input2)
          (ensures pinv1 r rw i /\ indep r i /\ done1 r rw i)
          (decreases i)
  = if i = 0 then begin loop0_inv r rw (S.length r.input1 - 1); done1_zero r rw end
    else begin
      loop1_all r rw (i - 1);
      loops_wire1 r (i - 1);
      step1 r rw (i - 1);
      done1_succ r rw (i - 1)
    end

(*** Later iterations do not touch earlier tokens / history rows *)

let rec tk_stable (r:run) (j k:nat)
  : Lemma (requires j < k /\ (forall (m:nat). m < k ==> stab r m))
          (ensures tok_at (tk r k) j == tok_at (tk r (j + 1)) j) (decreases k)
  = if k = j + 1 then ()
    else begin
      reveal_opaque (`%stab) (stab r (k - 1));
      tk_stable r j (k - 1)
    end

let hh_row_step (r:run) (j m:nat)
  : Lemma (requires j < r.input2 /\ j <> m /\ stab r m /\ S.length (hh r m) == r.input2 * 1024)
          (ensures bf_row (hh r (m + 1)) j == bf_row (hh r m) j)
  = reveal_opaque (`%stab) (stab r m);
    let a = hh r m in
    let b = hh r (m + 1) in
    assert (j * 1024 + 1024 <= r.input2 * 1024);
    introduce forall (q:nat). q < 1024 ==> S.index (S.slice b (j * 1024) (j * 1024 + 1024)) q ==
                                       S.index (S.slice a (j * 1024) (j * 1024 + 1024)) q
    with introduce _ ==> _ with begin
      M.lemma_div_plus q j 1024;
      M.small_div q 1024
    end;
    S.lemma_eq_intro (S.slice b (j * 1024) (j * 1024 + 1024)) (S.slice a (j * 1024) (j * 1024 + 1024))

let rec hh_stable (r:run) (j k:nat)
  : Lemma (requires j < k /\ k <= r.input2 /\ (forall (m:nat). m < k ==> stab r m) /\
                    S.length (hh r k) == r.input2 * 1024)
          (ensures bf_row (hh r k) j == bf_row (hh r (j + 1)) j) (decreases k)
  = if k = j + 1 then ()
    else begin
      reveal_opaque (`%stab) (stab r (k - 1));
      hh_row_step r j (k - 1);
      hh_stable r j (k - 1)
    end

(*** Loop 2: inactive iterations keep everything *)

let orc2 (r:run) (i:nat) : S.seq F32.t = if i = 0 then r.loop1_s0000_pi_layer22_1 (r.input2 - 1) else r.loop2_s0000_pi_inactive_0 (i - 1)
let ocv2 (r:run) (i:nat) : S.seq F32.t = if i = 0 then r.loop1_s0000_pi_layer22_2 (r.input2 - 1) else r.loop2_s0000_pi_inactive_1 (i - 1)
let ok2 (r:run) (i:nat) : S.seq BF.t = if i = 0 then r.loop1_s0000_pi_layer23_1 (r.input2 - 1) else r.loop2_s0000_pi_inactive_2 (i - 1)
let ov2 (r:run) (i:nat) : S.seq BF.t = if i = 0 then r.loop1_s0000_pi_layer23_2 (r.input2 - 1) else r.loop2_s0000_pi_inactive_3 (i - 1)
let ot2 (r:run) (i:nat) : S.seq I64.t = if i = 0 then r.loop1_s0000_pi_argmax_1 (r.input2 - 1) else r.loop2_s0000_pi_inactive_4 (i - 1)
let oh2 (r:run) (i:nat) : S.seq BF.t = if i = 0 then r.loop1_s0000_pi_projection_1 (r.input2 - 1) else r.loop2_s0000_pi_inactive_5 (i - 1)

let same2 (r:run) (i:nat) : prop =
  orc2 r i == orc2 r 0 /\ ocv2 r i == ocv2 r 0 /\ ok2 r i == ok2 r 0 /\ ov2 r i == ov2 r 0 /\
  ot2 r i == ot2 r 0 /\ oh2 r i == oh2 r 0

let lens2 (r:run) : prop =
  S.length (orc2 r 0) == opt_rec_len /\ S.length (ocv2 r 0) == opt_conv_len /\
  S.length (ok2 r 0) == opt_kv_len /\ S.length (ov2 r 0) == opt_kv_len /\
  S.length (ot2 r 0) == r.input2 /\ S.length (oh2 r 0) == r.input2 * 1024

let step2 (r:run) (i:nat)
  : Lemma (requires input_lengths r /\ i < loop2_count r /\ loop2_ok r i /\ lens2 r /\ same2 r i)
          (ensures same2 r (i + 1))
  = inact_elim (orc2 r i) (ocv2 r i) (ok2 r i) (ov2 r i) (ot2 r i) (oh2 r i)
      ((S.length r.input1 + r.input2 - 1) + 1 * i) (S.length r.input1)
      (r.loop2_s0000_pi_inactive_0 i) (r.loop2_s0000_pi_inactive_1 i) (r.loop2_s0000_pi_inactive_2 i)
      (r.loop2_s0000_pi_inactive_3 i) (r.loop2_s0000_pi_inactive_4 i) (r.loop2_s0000_pi_inactive_5 i)

let rec loop2_inv (r:run) (i:nat)
  : Lemma (requires input_lengths r /\ i <= loop2_count r /\ lens2 r /\
                    (forall (j:nat). j < loop2_count r ==> loop2_ok r j))
          (ensures same2 r i) (decreases i)
  = if i = 0 then () else begin loop2_inv r (i - 1); step2 r (i - 1) end

[@@"opaque_to_smt"]
let loops2 (r:run) : prop = forall (j:nat). j < loop2_count r ==> loop2_ok r j

let outputs_eq (r:run)
  : Lemma (requires input_lengths r /\ lens2 r /\ loops2 r)
          (ensures output0 r == tk r r.input2 /\ output1 r == hh r r.input2)
  = reveal_opaque (`%loops2) (loops2 r);
    if loop2_count r > 0 then loop2_inv r (loop2_count r) else ()

(*** The theorem *)

let sel_intro (nm:numerics) (rw h:S.seq real) (row:S.seq F32.t) (sel:int)
  : Lemma (requires row %~ ref_logits nm rw h /\ allowed_choice row sel)
          (ensures selected_ok nm rw h sel)
  = ()

let toks_agree (r:run)
  : Lemma (requires input_lengths r /\
                    (forall (j:nat). j < r.input2 ==> tok_at (tk r r.input2) j == tok_at (tk r (j + 1)) j))
          (ensures forall (p:nat). p < S.length r.input1 + r.input2 ==> prog_tok r.input1 (tk r r.input2) p == toks_r r p)
  = ()

let hid_eq (r:run) (rw:S.seq real) (j:nat)
  : Lemma (requires input_lengths r /\ j < r.input2 /\ output0 r == tk r r.input2 /\
                    (forall (p:nat). p < S.length r.input1 + r.input2 ==> prog_tok r.input1 (tk r r.input2) p == toks_r r p))
          (ensures ref_hidden opt_numerics rw (prog_tok r.input1 (output0 r)) (S.length r.input1 - 1 + j) == hcur r rw j)
  = ref_run_ext opt_numerics rw (prog_tok r.input1 (output0 r)) (toks_r r) (S.length r.input1 + j)

let per_j (r:run) (rw:S.seq real) (j:nat)
  : Lemma (requires input_lengths r /\ j < r.input2 /\ outs r rw j /\
                    output0 r == tk r r.input2 /\ output1 r == hh r r.input2 /\
                    tok_at (tk r r.input2) j == tok_at (tk r (j + 1)) j /\
                    bf_row (hh r r.input2) j == bf_row (hh r (j + 1)) j /\
                    (forall (p:nat). p < S.length r.input1 + r.input2 ==> prog_tok r.input1 (tk r r.input2) p == toks_r r p))
          (ensures bf_row (output1 r) j %~ ref_hidden opt_numerics rw (prog_tok r.input1 (output0 r)) (S.length r.input1 - 1 + j) /\
                   selected_ok opt_numerics rw (ref_hidden opt_numerics rw (prog_tok r.input1 (output0 r)) (S.length r.input1 - 1 + j))
                     (tok_at (output0 r) j))
  = reveal_opaque (`%outs) (outs r rw j);
    hid_eq r rw j;
    let hp = ref_hidden opt_numerics rw (prog_tok r.input1 (output0 r)) (S.length r.input1 - 1 + j) in
    assert (bf_row (output1 r) j == bf_row (hh r (j + 1)) j);
    assert (bf_row (output1 r) j %~ hp);
    assert (tok_at (output0 r) j == tok_at (tk r (j + 1)) j);
    sel_intro opt_numerics rw hp (r.loop1_s0000_pi_projection_0 j) (tok_at (output0 r) j)

let fam_elim (r:run)
  : Lemma (requires input_lengths r /\ family_ok r)
          (ensures init_ok r /\ loops_ok r /\ loops2 r)
  = init_elim r.input1 r.input2 r.call0_0_initialize_0 r.call0_0_initialize_1 r.call0_0_initialize_2
      r.call0_0_initialize_3 r.call0_0_initialize_4 r.call0_0_initialize_5;
    loops_intro r;
    reveal_opaque (`%loops2) (loops2 r)

let lens2_intro (r:run) (rw:S.seq real)
  : Lemma (requires 1 <= r.input2 /\ pinv1 r rw r.input2 /\ indep r r.input2) (ensures lens2 r)
  = ()

let lens2_from (r:run) (rw:S.seq real)
  : Lemma (requires pre r rw /\ loops_ok r) (ensures lens2 r)
  = loop1_all r rw r.input2;
    lens2_intro r rw

let loop1_done (r:run) (rw:S.seq real)
  : Lemma (requires pre r rw /\ loops_ok r) (ensures done1 r rw r.input2)
  = loop1_all r rw r.input2

let agree (r:run) : prop =
  input_lengths r /\ output0 r == tk r r.input2 /\ output1 r == hh r r.input2 /\
  (forall (j:nat). j < r.input2 ==> tok_at (tk r r.input2) j == tok_at (tk r (j + 1)) j) /\
  (forall (j:nat). j < r.input2 ==> bf_row (hh r r.input2) j == bf_row (hh r (j + 1)) j) /\
  (forall (p:nat). p < S.length r.input1 + r.input2 ==> prog_tok r.input1 (tk r r.input2) p == toks_r r p)

let rw_post (r:run) (rw:S.seq real) : prop =
  1 <= S.length r.input1 /\ forall (j:nat). j < r.input2 ==>
    bf_row (output1 r) j %~ ref_hidden opt_numerics rw (prog_tok r.input1 (output0 r)) (S.length r.input1 - 1 + j) /\
    selected_ok opt_numerics rw (ref_hidden opt_numerics rw (prog_tok r.input1 (output0 r)) (S.length r.input1 - 1 + j))
      (tok_at (output0 r) j)

let per_rw (r:run) (rw:S.seq real)
  : Lemma (requires pre r rw /\ loops_ok r /\ agree r) (ensures rw_post r rw)
  = loop1_done r rw;
    introduce forall (j:nat). j < r.input2 ==>
       bf_row (output1 r) j %~ ref_hidden opt_numerics rw (prog_tok r.input1 (output0 r)) (S.length r.input1 - 1 + j) /\
       selected_ok opt_numerics rw (ref_hidden opt_numerics rw (prog_tok r.input1 (output0 r)) (S.length r.input1 - 1 + j))
         (tok_at (output0 r) j)
    with introduce _ ==> _ with begin done1_elim r rw r.input2 j; per_j r rw j end

let post_intro (r:run)
  : Lemma (requires S.length (output0 r) == r.input2 /\ S.length (output1 r) == r.input2 * 1024 /\
                    valid_tokens (output0 r) /\ (forall (rw:S.seq real). r.input0 %~ rw ==> rw_post r rw))
          (ensures program_postcondition r.input0 r.input1 r.input2 (output0 r) (output1 r))
  = assert (numerics_ok opt_numerics)

let agree_intro (r:run)
  : Lemma (requires input_lengths r /\ lens2 r /\ loops2 r /\ (forall (m:nat). m < r.input2 ==> stab r m))
          (ensures agree r)
  = let sS = r.input2 in
    outputs_eq r;
    introduce forall (j:nat). j < sS ==> tok_at (tk r sS) j == tok_at (tk r (j + 1)) j
    with introduce _ ==> _ with tk_stable r j sS;
    introduce forall (j:nat). j < sS ==> bf_row (hh r sS) j == bf_row (hh r (j + 1)) j
    with introduce _ ==> _ with hh_stable r j sS;
    toks_agree r

let main_core (r:run)
  : Lemma (requires input_lengths r /\ valid_tokens r.input1 /\ init_ok r /\ loops_ok r /\ loops2 r)
          (ensures program_postcondition r.input0 r.input1 r.input2 (output0 r) (output1 r))
  = let sS = r.input2 in
    let rw0 = to_real_seq r.input0 in
    to_real_seq_is_approx r.input0;
    lens2_from r rw0;
    loop1_done r rw0;
    done1_stab r rw0 sS;
    agree_intro r;
    introduce forall (j:nat). j < sS ==> valid_token (I64.v (S.index (output0 r) j))
    with introduce _ ==> _ with begin done1_elim r rw0 sS j; outs_valid r rw0 j sS end;
    introduce forall (rw:S.seq real). r.input0 %~ rw ==> rw_post r rw
    with introduce _ ==> _ with per_rw r rw;
    post_intro r

let main (r:run)
  : Lemma (requires input_lengths r /\ family_ok r /\ program_premise r.input0 r.input1 r.input2)
          (ensures program_postcondition r.input0 r.input1 r.input2 (output0 r) (output1 r))
  = fam_elim r;
    main_core r

let refines (r:run) = FStar.Classical.move_requires main r
