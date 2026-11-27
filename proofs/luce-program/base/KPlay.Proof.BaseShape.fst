module KPlay.Proof.BaseShape

(* Phase 1 (no reals): buffer lengths along the layer chain, and the token /
   history bookkeeping over positions 0 .. 11, by one induction on t. *)

open FStar.Real
open Kuiper
open Kuiper.Approximates
open KPlay.SharedSpec
open KPlay.Proof.Wiring.Base
open KPlay.Proof.BaseAccess
open KPlay.Proof.BaseView
module S = FStar.Seq
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32
module I64 = FStar.Int64
module ML = FStar.Math.Lemmas

#set-options "--z3rlimit 20 --fuel 0 --ifuel 0"

let pP (r:run) : int = S.length r.input1

(* token selected at position t *)
let sel (r:run) (t:int) : int = if t < 0 then 0 else tok_at (uarg r t) 0

(* hidden row consumed by the update at position t *)
let hrow (r:run) (t:int) : S.seq BF.t = if t < 0 then S.empty else ufin r t

let premise (r:run) : prop = fam r /\ input_lengths r /\ valid_tokens r.input1

(*** Shapes *)

let shape_l (r:run) (t l:nat) : prop =
  (not (is_attn l) ==> S.length (conv_in r t l) == 24576 /\ S.length (rec_in r t l) == 262144) /\
  (is_attn l ==> S.length (kv_in r t l) == kv_len r)

[@@"opaque_to_smt"]
let shape_in (r:run) (t:nat) : prop = forall (l:nat). l < 24 ==> shape_l r t l

let shape_in_at (r:run) (t l:nat)
  : Lemma (requires shape_in r t /\ l < 24) (ensures shape_l r t l)
  = reveal_opaque (`%shape_in) (shape_in r t)

let emb_shape (r:run) (t:nat) : prop =
  S.length (lx r t 0) == 1024 /\ S.length (pcos r t) == 32 /\ S.length (psin r t) == 32

let kv_len_cap (r:run)
  : Lemma (requires fam r)
          (ensures kv_len r == cap r * 1024 /\ kv_len r / 1024 == cap r)
  = fam_lengths r;
    ML.cancel_mul_div (cap r) 1024

let wv_len (r:run) (l:nat)
  : Lemma (requires fam r /\ l < 24) (ensures S.length (wv r l) == wlen l)
  = fam_lengths r;
    assert (l / 4 <= 5 /\ l % 4 <= 3);
    assert (layer_off l + wlen l <= 752393024);
    view_len r.input0 (layer_off l) (wlen l)

let table_len_ok (r:run)
  : Lemma (requires fam r)
          (ensures S.length (table_view r) == table_len /\ S.length (norm_view r) == 1024)
  = fam_lengths r;
    view_len r.input0 0 254279680;
    view_len r.input0 254279680 1024

let init_facts (r:run)
  : Lemma (requires premise r)
          (ensures (let p = pP r in let s = r.input2 in
                    S.length r.initialize_0 == p * 1024 /\
                    (forall (i:nat). i < p * 1024 ==>
                       S.index r.initialize_0 i == S.index (table_view r) (tok_at r.input1 (i / 1024) * 1024 + i % 1024)) /\
                    S.length r.initialize_1 == 442368 + (p + s - 1) * 6144 /\ all_zero r.initialize_1 /\
                    S.length r.initialize_2 == opt_rec_len /\ all_zero r.initialize_2 /\
                    S.length r.initialize_3 == s /\ (forall (i:nat). i < s ==> tok_at r.initialize_3 i == 0) /\
                    S.length r.initialize_4 == s * 1024 /\ all_zero r.initialize_4))
  = fam_lengths r;
    table_len_ok r;
    init_rel_ok r;
    reveal_opaque (`%init_rel) (init_rel r)

let kv_view_bound (c g:nat)
  : Lemma (requires g <= 5) (ensures 1024 * c * g + 1024 * c <= c * 6144 /\ 0 <= 1024 * c * g)
  = ML.lemma_mult_le_left (1024 * c) g 5

let shape_0_attn (r:run) (l:nat)
  : Lemma (requires l < 24 /\ is_attn l /\ 0 <= cap r /\ kv_len r == cap r * 1024 /\
                    S.length r.initialize_1 == 442368 + cap r * 6144)
          (ensures S.length (kv_in r 0 l) == kv_len r)
  = let g : nat = l / 4 in
    let c : nat = cap r in
    kv_view_bound c g;
    assert (kv_len r * g == 1024 * c * g);
    view_len r.initialize_1 (kv_off r l) (kv_len r)

let shape_0_dn (r:run) (l:nat)
  : Lemma (requires l < 24 /\ not (is_attn l) /\ S.length r.initialize_1 >= 442368 /\
                    S.length r.initialize_2 == opt_rec_len)
          (ensures S.length (conv_in r 0 l) == 24576 /\ S.length (rec_in r 0 l) == 262144)
  = KPlay.Proof.BaseRef.dn_index_facts l l;
    view_len r.initialize_1 (24576 * dn_index l) 24576;
    view_len r.initialize_2 (262144 * dn_index l) 262144

let init_lens (r:run)
  : Lemma (requires premise r)
          (ensures S.length r.initialize_1 == 442368 + cap r * 6144 /\ S.length r.initialize_2 == opt_rec_len /\
                   1 <= cap r /\ kv_len r == cap r * 1024)
  = init_facts r;
    fam_lengths r;
    kv_len_cap r

let shape_0 (r:run)
  : Lemma (requires premise r) (ensures shape_in r 0)
  = init_lens r;
    let c : nat = cap r in
    ML.lemma_mult_le_right 6144 0 c;
    assert (S.length r.initialize_1 >= 442368);
    let aux (l:nat) : Lemma (l < 24 ==> shape_l r 0 l) =
      if l < 24 then begin
        if is_attn l then shape_0_attn r l else shape_0_dn r l
      end
    in
    Classical.forall_intro aux;
    reveal_opaque (`%shape_in) (shape_in r 0)

let layer_len (r:run) (t l:nat)
  : Lemma (requires fam r /\ t < cap r /\ l < 24 /\ shape_l r t l /\ S.length (lx r t l) == 1024 /\ emb_shape r t)
          (ensures S.length (lout r t l) == 1024 /\ shape_l r (t + 1) l)
  = layer_rel_ok r t l;
    reveal_opaque (`%layer_rel) (layer_rel r t l);
    fam_lengths r;
    wv_len r l;
    kv_len_cap r

let rec lx_len (r:run) (t:nat) (l:nat)
  : Lemma (requires fam r /\ t < cap r /\ l <= 24 /\ shape_in r t /\ emb_shape r t)
          (ensures S.length (lx r t l) == 1024)
          (decreases l)
  = if l = 0 then ()
    else begin
      lx_len r t (l - 1);
      shape_in_at r t (l - 1);
      layer_len r t (l - 1)
    end

let shape_next (r:run) (t:nat)
  : Lemma (requires fam r /\ t < cap r /\ shape_in r t /\ emb_shape r t)
          (ensures shape_in r (t + 1))
  = let aux (l:nat) : Lemma (l < 24 ==> shape_l r (t + 1) l) =
      if l < 24 then begin
        lx_len r t l;
        shape_in_at r t l;
        layer_len r t l
      end
    in
    Classical.forall_intro aux;
    reveal_opaque (`%shape_in) (shape_in r (t + 1))

let fin_len (r:run) (t:nat)
  : Lemma (requires fam r /\ t < cap r /\ shape_in r t /\ emb_shape r t)
          (ensures S.length (lx r t 24) == 1024 /\ S.length (ufin r t) == 1024 /\ S.length (ulog r t) == 248320 /\
                   (forall (i:nat). i < 1024 ==> S.index (ufin r t) i == S.index (lx r t 24) i))
  = lx_len r t 24;
    table_len_ok r;
    fin_rel_ok r t;
    reveal_opaque (`%fin_rel) (fin_rel r t)

let ufin_tail (r:run) (t:nat)
  : Lemma (requires fam r /\ 1 <= cap r /\ cap r <= t)
          (ensures ufin r t == ufin r (cap r - 1) /\ ulog r t == ulog r (cap r - 1))
  = fam_lengths r

(*** Tokens and history *)

[@@"opaque_to_smt"]
let tok_inv (r:run) (t:int) : prop =
  S.length (utok r t) == r.input2 /\ S.length (uhist r t) == r.input2 * 1024 /\
  S.length r.initialize_4 == r.input2 * 1024 /\
  (forall (i:nat). i < r.input2 ==> valid_token (tok_at (utok r t) i)) /\
  (forall (i:nat). i < r.input2 ==>
     tok_at (utok r t) i == (if pP r - 1 + i <= t then sel r (pP r - 1 + i) else 0)) /\
  (forall (k:nat). k < r.input2 * 1024 ==>
     (pP r - 1 + k / 1024 <= t ==>
        S.length (hrow r (pP r - 1 + k / 1024)) == 1024 /\
        S.index (uhist r t) k == S.index (hrow r (pP r - 1 + k / 1024)) (k % 1024)) /\
     (~(pP r - 1 + k / 1024 <= t) ==> S.index (uhist r t) k == S.index r.initialize_4 k))

let tok_inv_init_core (r:run)
  : Lemma (requires 1 <= pP r /\ S.length r.initialize_3 == r.input2 /\
                    (forall (i:nat). i < r.input2 ==> tok_at r.initialize_3 i == 0) /\
                    S.length r.initialize_4 == r.input2 * 1024)
          (ensures tok_inv r (-1))
  = reveal_opaque (`%tok_inv) (tok_inv r (-1));
    let ti (i:nat) : Lemma (i < r.input2 ==> valid_token (tok_at r.initialize_3 i) /\
                             tok_at r.initialize_3 i == (if pP r - 1 + i <= -1 then sel r (pP r - 1 + i) else 0)) = ()
    in
    Classical.forall_intro ti;
    let hi (k:nat) : Lemma (k < r.input2 * 1024 ==> ~(pP r - 1 + k / 1024 <= -1)) = () in
    Classical.forall_intro hi

let tok_inv_init (r:run)
  : Lemma (requires premise r) (ensures tok_inv r (-1))
  = init_facts r;
    tok_inv_init_core r

let argmax_facts (r:run) (t:nat)
  : Lemma (requires fam r /\ t < 12 /\ S.length (ulog r t) == 248320)
          (ensures S.length (uarg r t) == 1 /\
                   (0 <= t - pP r + 1 /\ t - pP r + 1 < r.input2 ==> valid_token (sel r t)) /\
                   (~(0 <= t - pP r + 1 /\ t - pP r + 1 < r.input2) ==> sel r t == 0) /\
                   (0 <= t - pP r + 1 /\ t - pP r + 1 < r.input2 ==> first_allowed_choice (ulog r t) (sel r t)))
  = fam_lengths r;
    pos_rel_ok r t;
    reveal_opaque (`%arg_rel) (arg_rel r t)

let tok_one (r:run) (t:nat) (i:nat)
  : Lemma (requires i < r.input2 /\ S.length (utok r t) == r.input2 /\ S.length (utok r (t - 1)) == r.input2 /\
                    S.length (uarg r t) == 1 /\
                    S.index (utok r t) i == (if i = t - pP r + 1 then S.index (uarg r t) 0 else S.index (utok r (t - 1)) i) /\
                    valid_token (tok_at (utok r (t - 1)) i) /\
                    tok_at (utok r (t - 1)) i == (if pP r - 1 + i <= t - 1 then sel r (pP r - 1 + i) else 0) /\
                    (i = t - pP r + 1 ==> valid_token (sel r t)))
          (ensures valid_token (tok_at (utok r t) i) /\
                   tok_at (utok r t) i == (if pP r - 1 + i <= t then sel r (pP r - 1 + i) else 0))
  = ()

let hist_one (r:run) (t:nat) (k:nat)
  : Lemma (requires k < r.input2 * 1024 /\ S.length (uhist r t) == r.input2 * 1024 /\
                    S.length (uhist r (t - 1)) == r.input2 * 1024 /\ S.length r.initialize_4 == r.input2 * 1024 /\
                    S.length (ufin r t) == 1024 /\
                    S.index (uhist r t) k ==
                      (if k / 1024 = t - pP r + 1 then S.index (ufin r t) (k % 1024) else S.index (uhist r (t - 1)) k) /\
                    (pP r - 1 + k / 1024 <= t - 1 ==>
                       S.length (hrow r (pP r - 1 + k / 1024)) == 1024 /\
                       S.index (uhist r (t - 1)) k == S.index (hrow r (pP r - 1 + k / 1024)) (k % 1024)) /\
                    (~(pP r - 1 + k / 1024 <= t - 1) ==> S.index (uhist r (t - 1)) k == S.index r.initialize_4 k))
          (ensures (pP r - 1 + k / 1024 <= t ==>
                      S.length (hrow r (pP r - 1 + k / 1024)) == 1024 /\
                      S.index (uhist r t) k == S.index (hrow r (pP r - 1 + k / 1024)) (k % 1024)) /\
                   (~(pP r - 1 + k / 1024 <= t) ==> S.index (uhist r t) k == S.index r.initialize_4 k))
  = ()

let upd_facts (r:run) (t:nat)
  : Lemma (requires fam r /\ t < 12 /\ tok_inv r (t - 1) /\ S.length (ufin r t) == 1024 /\
                    S.length (uarg r t) == 1)
          (ensures (let s = r.input2 in let dest = t - pP r + 1 in
                    S.length (utok r (t - 1)) == s /\ S.length (uhist r (t - 1)) == s * 1024 /\
                    S.length (utok r t) == s /\ S.length (uhist r t) == s * 1024 /\
                    (forall (i:nat). i < s ==>
                       S.index (utok r t) i == (if i = dest then S.index (uarg r t) 0 else S.index (utok r (t - 1)) i)) /\
                    (forall (k:nat). k < s * 1024 ==>
                       S.index (uhist r t) k ==
                         (if k / 1024 = dest then S.index (ufin r t) (k % 1024) else S.index (uhist r (t - 1)) k))))
  = fam_lengths r;
    reveal_opaque (`%tok_inv) (tok_inv r (t - 1));
    pos_rel_ok r t;
    reveal_opaque (`%upd_rel) (upd_rel r t)

let tok_step (r:run) (t:nat)
  : Lemma (requires fam r /\ t < 12 /\ tok_inv r (t - 1) /\ S.length (ufin r t) == 1024 /\
                    S.length (ulog r t) == 248320)
          (ensures tok_inv r t)
  = reveal_opaque (`%tok_inv) (tok_inv r (t - 1));
    reveal_opaque (`%tok_inv) (tok_inv r t);
    argmax_facts r t;
    upd_facts r t;
    let ti (i:nat) : Lemma (i < r.input2 ==>
                            (valid_token (tok_at (utok r t) i) /\
                             tok_at (utok r t) i == (if pP r - 1 + i <= t then sel r (pP r - 1 + i) else 0))) =
      if i < r.input2 then tok_one r t i
    in
    Classical.forall_intro ti;
    let hi (k:nat) : Lemma (k < r.input2 * 1024 ==>
                            (pP r - 1 + k / 1024 <= t ==>
                               S.length (hrow r (pP r - 1 + k / 1024)) == 1024 /\
                               S.index (uhist r t) k == S.index (hrow r (pP r - 1 + k / 1024)) (k % 1024)) /\
                            (~(pP r - 1 + k / 1024 <= t) ==> S.index (uhist r t) k == S.index r.initialize_4 k)) =
      if k < r.input2 * 1024 then hist_one r t k
    in
    Classical.forall_intro hi

(*** Joint induction over positions *)

[@@"opaque_to_smt"]
let jinv (r:run) (t:nat) : prop =
  tok_inv r (t - 1) /\
  (t < cap r ==> shape_in r t) /\
  (t >= cap r /\ 1 <= cap r ==> S.length (ufin r (cap r - 1)) == 1024 /\ S.length (ulog r (cap r - 1)) == 248320)

let mul_bound (a b c:nat) : Lemma (requires a < b) (ensures 0 <= a * c /\ a * c + c <= b * c)
  = ML.lemma_mult_le_right c (a + 1) b;
    ML.nat_times_nat_is_nat a c

let init_len0 (r:run)
  : Lemma (requires premise r)
          (ensures S.length r.initialize_0 == pP r * 1024 /\ 1 <= pP r /\ S.length (table_view r) == table_len)
  = init_facts r;
    fam_lengths r;
    table_len_ok r

let tok_valid_at (r:run) (t:int) (i:nat)
  : Lemma (requires tok_inv r t /\ i < r.input2)
          (ensures valid_token (tok_at (utok r t) i) /\ S.length (utok r t) == r.input2)
  = reveal_opaque (`%tok_inv) (tok_inv r t)

let emb_bounds (r:run) (t:nat)
  : Lemma (requires premise r /\ t < cap r /\ tok_inv r (t - 1))
          (ensures S.length r.initialize_0 == pP r * 1024 /\ S.length (table_view r) == table_len /\
                   1 <= pP r /\ S.length (utok r (t - 1)) == r.input2 /\
                   (t < pP r ==> t * 1024 + 1024 <= S.length r.initialize_0) /\
                   (t >= pP r ==> valid_token (tok_at (utok r (t - 1)) (t - pP r)) /\
                                  0 <= tok_at (utok r (t - 1)) (t - pP r) * 1024 /\
                                  tok_at (utok r (t - 1)) (t - pP r) * 1024 + 1024 <= S.length (table_view r)))
  = init_len0 r;
    tok_valid_at r (t - 1) 0;
    if t < pP r then mul_bound t (pP r) 1024
    else begin
      let i : nat = t - pP r in
      tok_valid_at r (t - 1) i;
      let tk : nat = tok_at (utok r (t - 1)) i in
      mul_bound tk 248320 1024
    end

let emb_ok (r:run) (t:nat)
  : Lemma (requires premise r /\ t < cap r /\ tok_inv r (t - 1))
          (ensures S.length r.initialize_0 == pP r * 1024 /\ S.length (table_view r) == table_len /\
                   (t < pP r ==> t * 1024 + 1024 <= S.length r.initialize_0) /\
                   (t >= pP r ==> valid_token (tok_at (utok r (t - 1)) (t - pP r)) /\
                                  0 <= tok_at (utok r (t - 1)) (t - pP r) * 1024 /\
                                  tok_at (utok r (t - 1)) (t - pP r) * 1024 + 1024 <= S.length (table_view r)) /\
                   emb_shape r t /\
                   (forall (i:nat). i < 1024 ==>
                      S.index (lx r t 0) i ==
                        (if t < pP r then S.index r.initialize_0 (t * 1024 + i)
                         else S.index (table_view r) (tok_at (utok r (t - 1)) (t - pP r) * 1024 + i))) /\
                   pcos r t %~ rope_cos base_numerics t /\ psin r t %~ rope_sin base_numerics t)
  = emb_bounds r t;
    fam_lengths r;
    ML.cancel_mul_div (pP r) 1024;
    emb_rel_ok r t;
    reveal_opaque (`%emb_rel) (emb_rel r t)

let emb_shape_ok (r:run) (t:nat)
  : Lemma (requires premise r /\ t < cap r /\ tok_inv r (t - 1)) (ensures emb_shape r t)
  = emb_ok r t

let jinv_step (r:run) (t:nat)
  : Lemma (requires premise r /\ t < 12 /\ jinv r t) (ensures jinv r (t + 1))
  = reveal_opaque (`%jinv) (jinv r t);
    reveal_opaque (`%jinv) (jinv r (t + 1));
    fam_lengths r;
    if t < cap r then begin
      emb_shape_ok r t;
      fin_len r t;
      tok_step r t;
      if t + 1 < cap r then shape_next r t
      else begin
        let c1 : nat = cap r - 1 in
        assert (c1 == t);
        assert (ufin r c1 == ufin r t /\ ulog r c1 == ulog r t)
      end;
      assert (tok_inv r ((t + 1) - 1))
    end else begin
      ufin_tail r t;
      tok_step r t
    end

let rec jinv_all (r:run) (t:nat)
  : Lemma (requires premise r /\ t <= 12) (ensures jinv r t) (decreases t)
  = if t = 0 then begin
      fam_lengths r;
      tok_inv_init r;
      shape_0 r;
      reveal_opaque (`%jinv) (jinv r 0)
    end else begin
      jinv_all r (t - 1);
      jinv_step r (t - 1)
    end

(* Everything Phase 2 needs at position t < C. *)
let pos_shape (r:run) (t:nat)
  : Lemma (requires premise r /\ t < cap r)
          (ensures tok_inv r (t - 1) /\ shape_in r t /\ emb_shape r t /\
                   (forall (l:nat). l <= 24 ==> S.length (lx r t l) == 1024) /\
                   S.length (ufin r t) == 1024 /\ S.length (ulog r t) == 248320 /\
                   (forall (i:nat). i < 1024 ==> S.index (ufin r t) i == S.index (lx r t 24) i))
  = fam_lengths r;
    jinv_all r t;
    reveal_opaque (`%jinv) (jinv r t);
    emb_shape_ok r t;
    fin_len r t;
    let aux (l:nat) : Lemma (l <= 24 ==> S.length (lx r t l) == 1024) =
      if l <= 24 then lx_len r t l
    in
    Classical.forall_intro aux

let final_tokens (r:run)
  : Lemma (requires premise r) (ensures tok_inv r 11)
  = jinv_all r 12;
    reveal_opaque (`%jinv) (jinv r 12)
