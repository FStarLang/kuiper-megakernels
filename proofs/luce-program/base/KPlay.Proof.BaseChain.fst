module KPlay.Proof.BaseChain

(* Phase 2: numerics. For fixed real weights rw approximated by the input
   weights, one size-generic induction over layers (chain) and one over
   positions (ninv_all) relate every call boundary to the reference. *)

open FStar.Real
open Kuiper
open Kuiper.Approximates
open KPlay.SharedSpec
open KPlay.Proof.Wiring.Base
open KPlay.Proof.BaseCongr
open KPlay.Proof.BaseRef
open KPlay.Proof.BaseView
open KPlay.Proof.BaseAccess
open KPlay.Proof.BaseShape
open KPlay.Proof.BaseKV
module S = FStar.Seq
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32
module I64 = FStar.Int64
module ML = FStar.Math.Lemmas

#set-options "--z3rlimit 20 --fuel 0 --ifuel 0"

(*** Reference objects *)

let toks (r:run) : nat -> int = prog_tok r.input1 (output0 r)

let rs (r:run) (rw:S.seq real) (t:nat) : rstate = fst (ref_run base_numerics rw (toks r) t)

let st0 (r:run) (rw:S.seq real) (t:nat) : lstate =
  { l_convs = (rs r rw t).convs; l_recs = (rs r rw t).recs; l_cur = S.create 6144 0.0R; l_x = embed rw (toks r t) }

let ff (r:run) (rw:S.seq real) (t l:nat) : lstate =
  fwd base_numerics rw t (rope_cos base_numerics t) (rope_sin base_numerics t) (rs r rw t).kvs (st0 r rw t) l

#push-options "--fuel 1"
let ref_run_succ (nm:numerics) (w:S.seq real) (tk:nat -> int) (n:nat)
  : Lemma (ref_run nm w tk (n + 1) == ref_step nm w (tk n) n (fst (ref_run nm w tk n)))
  = ()

let ref_run_zero (nm:numerics) (w:S.seq real) (tk:nat -> int)
  : Lemma (fst (ref_run nm w tk 0) == init_state)
  = ()
#pop-options

let rs_succ (r:run) (rw:S.seq real) (t:nat)
  : Lemma ((rs r rw (t + 1)).convs == (ff r rw t 24).l_convs /\
           (rs r rw (t + 1)).recs == (ff r rw t 24).l_recs /\
           (rs r rw (t + 1)).kvs == S.snoc (rs r rw t).kvs (ff r rw t 24).l_cur /\
           ref_hidden base_numerics rw (toks r) t == (ff r rw t 24).l_x)
  = ref_run_succ base_numerics rw (toks r) t;
    run_layers_fwd base_numerics rw t (rope_cos base_numerics t) (rope_sin base_numerics t) (rs r rw t).kvs (st0 r rw t)

(*** Invariant over positions *)

let ctx (r:run) (rw:S.seq real) : prop = premise r /\ r.input0 %~ rw

let state_rel (r:run) (rw:S.seq real) (t l:nat) : prop =
  (is_attn l ==> kv_in r t l %~ rkv_of (cap r) (rs r rw t).kvs (l / 4)) /\
  (not (is_attn l) ==>
     conv_in r t l %~ sget (rs r rw t).convs (dn_index l) /\ rec_in r t l %~ sget (rs r rw t).recs (dn_index l))

let rs_lens (r:run) (rw:S.seq real) (t:nat) : prop =
  S.length (rs r rw t).convs == 18 /\ S.length (rs r rw t).recs == 18 /\ S.length (rs r rw t).kvs == t

[@@"opaque_to_smt"]
let ninv (r:run) (rw:S.seq real) (t:nat) : prop =
  rs_lens r rw t /\ (forall (l:nat). l < 24 ==> state_rel r rw t l)

let ninv_at (r:run) (rw:S.seq real) (t l:nat)
  : Lemma (requires ninv r rw t /\ l < 24) (ensures rs_lens r rw t /\ state_rel r rw t l)
  = reveal_opaque (`%ninv) (ninv r rw t)

let slots0 (r:run) (rw:S.seq real) (t:nat)
  : Lemma (requires rs_lens r rw t) (ensures slots_ok (st0 r rw t))
  = ()

(*** Instances of the base relations *)

let attn_inst (w x kv cos sin:S.seq BF.t) (t:int) (x' kv':S.seq BF.t) (rw rx rkv rc rs:S.seq real) (c:int)
  : Lemma (requires base_full_attention_decoder_relation w x kv cos sin t x' kv' /\
                    c == S.length kv / 1024 /\
                    S.length w == attn_len /\ S.length x == 1024 /\ S.length kv == c * 1024 /\ 1 <= c /\ c <= 12 /\
                    0 <= t /\ t < c /\ S.length cos == 32 /\ S.length sin == 32 /\
                    w %~ rw /\ x %~ rx /\ kv %~ rkv /\ cos %~ rc /\ sin %~ rs)
          (ensures x' %~ at_out base_numerics rw 0 rx rc rs t (base_hk c rkv) (base_hv c rkv) /\
                   kv' %~ base_kv_write c t rkv (at_k base_numerics rw 0 rx rc rs) (at_v base_numerics rw 0 rx))
  = ()

let dn_inst (w x conv:S.seq BF.t) (st:S.seq F32.t) (x' conv':S.seq BF.t) (st':S.seq F32.t) (rw rx rc rs:S.seq real)
  : Lemma (requires base_recurrent_decoder_relation w x conv st x' conv' st' /\
                    S.length w == dn_len /\ S.length x == 1024 /\ S.length conv == 24576 /\ S.length st == 262144 /\
                    w %~ rw /\ x %~ rx /\ conv %~ rc /\ st %~ rs)
          (ensures x' %~ dn_out base_numerics rw 0 rx rc rs /\
                   conv' %~ dn_conv base_numerics rw 0 rx rc /\
                   st' %~ dn_state base_numerics rw 0 rx rc rs)
  = ()

let fin_inst (table normw residual hidden logits':S.seq BF.t) (rt rn rx:S.seq real)
  : Lemma (requires base_final_norm_projection_relation table normw residual hidden logits' /\
                    S.length table == table_len /\ S.length normw == 1024 /\ S.length residual == 1024 /\
                    table %~ rt /\ normw %~ rn /\ residual %~ rx)
          (ensures logits' %~ logits base_numerics rt 0 rn 0 rx)
  = ()

(*** One layer *)

let lo_bound (l:nat)
  : Lemma (requires l < 24) (ensures 0 <= layer_off l /\ layer_off l + wlen l <= 752393024)
  = assert (l / 4 <= 5 /\ l % 4 <= 3)

let rope_ok (r:run) (t:nat)
  : Lemma (requires premise r /\ t < cap r)
          (ensures pcos r t %~ rope_cos base_numerics t /\ psin r t %~ rope_sin base_numerics t)
  = pos_shape r t;
    emb_ok r t

let layer_pre (r:run) (t l:nat)
  : Lemma (requires premise r /\ t < cap r /\ l < 24)
          (ensures S.length (lx r t l) == 1024 /\ emb_shape r t /\ shape_l r t l /\
                   S.length (wv r l) == wlen l /\ kv_len r == cap r * 1024 /\ kv_len r / 1024 == cap r /\
                   1 <= cap r /\ cap r <= 12 /\ layer_rel r t l /\
                   pcos r t %~ rope_cos base_numerics t /\ psin r t %~ rope_sin base_numerics t /\
                   0 <= layer_off l /\ layer_off l + wlen l <= S.length r.input0)
  = pos_shape r t;
    shape_in_at r t l;
    fam_lengths r;
    wv_len r l;
    kv_len_cap r;
    layer_rel_ok r t l;
    rope_ok r t;
    lo_bound l

(* The reference attention step, read through the base history views. *)
let attn_step_x (w:S.seq real) (t:nat) (cs sn:S.seq real) (kvs:S.seq (S.seq real)) (st:lstate) (l:nat)
                (c:pos) (w':S.seq real)
  : Lemma (requires is_attn l /\ t <= c /\ wagree w' 0 w (layer_off l) attn_len)
          (ensures (layer_step base_numerics w t cs sn kvs st l).l_x ==
                   at_out base_numerics w' 0 st.l_x cs sn t (base_hk c (rkv_of c kvs (l / 4))) (base_hv c (rkv_of c kvs (l / 4))))
  = layer_step_attn base_numerics w t cs sn kvs st l;
    hk_agree c kvs l t;
    at_out_congr base_numerics w' 0 w (layer_off l) st.l_x cs sn t
      (base_hk c (rkv_of c kvs (l / 4))) (fun (p j:nat) -> kv_get kvs p ((l / 4) * 1024 + j))
      (base_hv c (rkv_of c kvs (l / 4))) (fun (p j:nat) -> kv_get kvs p ((l / 4) * 1024 + 512 + j))

let layer_attn (r:run) (rw:S.seq real) (t l:nat)
  : Lemma (requires ctx r rw /\ t < cap r /\ l < 24 /\ is_attn l /\ state_rel r rw t l /\
                    lx r t l %~ (ff r rw t l).l_x)
          (ensures lout r t l %~ (ff r rw t (l + 1)).l_x /\
                   kv_out r t l %~ base_kv_write (cap r) t (rkv_of (cap r) (rs r rw t).kvs (l / 4))
                      (at_k base_numerics rw (layer_off l) (ff r rw t l).l_x (rope_cos base_numerics t) (rope_sin base_numerics t))
                      (at_v base_numerics rw (layer_off l) (ff r rw t l).l_x))
  = layer_pre r t l;
    reveal_opaque (`%layer_rel) (layer_rel r t l);
    let c : pos = cap r in
    let lo = layer_off l in
    let cs = rope_cos base_numerics t in
    let sn = rope_sin base_numerics t in
    let x = (ff r rw t l).l_x in
    let w' = view rw lo attn_len in
    let rkv = rkv_of c (rs r rw t).kvs (l / 4) in
    view_approx_bf r.input0 rw lo attn_len;
    attn_inst (wv r l) (lx r t l) (kv_in r t l) (pcos r t) (psin r t) t (lout r t l) (kv_out r t l) w' x rkv cs sn c;
    view_wagree rw lo attn_len;
    at_pre_congr base_numerics w' 0 rw lo x cs sn;
    fwd_succ base_numerics rw t cs sn (rs r rw t).kvs (st0 r rw t) l;
    attn_step_x rw t cs sn (rs r rw t).kvs (ff r rw t l) l c w'

let layer_dn (r:run) (rw:S.seq real) (t l:nat)
  : Lemma (requires ctx r rw /\ t < cap r /\ l < 24 /\ not (is_attn l) /\ state_rel r rw t l /\ rs_lens r rw t /\
                    lx r t l %~ (ff r rw t l).l_x)
          (ensures lout r t l %~ (ff r rw t (l + 1)).l_x /\
                   conv_out r t l %~ dn_conv base_numerics rw (layer_off l) (ff r rw t l).l_x (sget (rs r rw t).convs (dn_index l)) /\
                   rec_out r t l %~ dn_state base_numerics rw (layer_off l) (ff r rw t l).l_x
                                      (sget (rs r rw t).convs (dn_index l)) (sget (rs r rw t).recs (dn_index l)))
  = layer_pre r t l;
    reveal_opaque (`%layer_rel) (layer_rel r t l);
    let lo = layer_off l in
    let cs = rope_cos base_numerics t in
    let sn = rope_sin base_numerics t in
    let x = (ff r rw t l).l_x in
    let w' = view rw lo dn_len in
    let d = dn_index l in
    let cv = sget (rs r rw t).convs d in
    let rc = sget (rs r rw t).recs d in
    view_approx_bf r.input0 rw lo dn_len;
    dn_inst (wv r l) (lx r t l) (conv_in r t l) (rec_in r t l) (lout r t l) (conv_out r t l) (rec_out r t l) w' x cv rc;
    view_wagree rw lo dn_len;
    dn_out_congr base_numerics w' 0 rw lo x cv rc;
    dn_pre_congr base_numerics w' 0 rw lo x cv;
    dn_state_congr base_numerics w' 0 rw lo x cv rc;
    slots0 r rw t;
    conv_fwd base_numerics rw t cs sn (rs r rw t).kvs (st0 r rw t) l l;
    fwd_succ base_numerics rw t cs sn (rs r rw t).kvs (st0 r rw t) l;
    layer_step_dn base_numerics rw t cs sn (rs r rw t).kvs (ff r rw t l) l

(*** Embedding *)

let tok_inv_at (r:run) (t:int) (i:nat)
  : Lemma (requires tok_inv r t /\ i < r.input2)
          (ensures S.length (utok r t) == r.input2 /\ valid_token (tok_at (utok r t) i) /\
                   tok_at (utok r t) i == (if pP r - 1 + i <= t then sel r (pP r - 1 + i) else 0))
  = reveal_opaque (`%tok_inv) (tok_inv r t)

let tok_gen (r:run) (t:nat)
  : Lemma (requires premise r /\ t < cap r /\ pP r <= t)
          (ensures toks r t == tok_at (utok r (t - 1)) (t - pP r))
  = fam_lengths r;
    pos_shape r t;
    final_tokens r;
    outputs_ok r;
    tok_inv_at r (t - 1) (t - pP r);
    tok_inv_at r 11 (t - pP r)

let emb_tok (r:run) (t:nat)
  : Lemma (requires premise r /\ t < cap r)
          (ensures valid_token (toks r t) /\ 0 <= toks r t * 1024 /\ toks r t * 1024 + 1024 <= table_len /\
                   S.length (table_view r) == table_len)
  = pos_shape r t;
    emb_bounds r t;
    if t < pP r then mul_bound (toks r t) 248320 1024
    else tok_gen r t

let dm1024 (t i:nat)
  : Lemma (requires i < 1024) (ensures (t * 1024 + i) / 1024 == t /\ (t * 1024 + i) % 1024 == i)
  = ML.lemma_div_plus i t 1024;
    ML.small_div i 1024;
    ML.lemma_mod_plus i t 1024;
    ML.small_mod i 1024

let emb_prompt (r:run) (t:nat) (i:nat)
  : Lemma (requires premise r /\ t < pP r /\ i < 1024 /\ toks r t * 1024 + 1024 <= table_len /\
                    0 <= toks r t * 1024 /\ S.length (table_view r) == table_len)
          (ensures S.length r.initialize_0 == pP r * 1024 /\ t * 1024 + i < S.length r.initialize_0 /\
                   S.index r.initialize_0 (t * 1024 + i) == S.index (table_view r) (toks r t * 1024 + i))
  = init_facts r;
    mul_bound t (pP r) 1024;
    dm1024 t i

let emb_entries (r:run) (t:nat)
  : Lemma (requires premise r /\ t < cap r)
          (ensures valid_token (toks r t) /\ 0 <= toks r t * 1024 /\ toks r t * 1024 + 1024 <= table_len /\
                   S.length (table_view r) == table_len /\ S.length (lx r t 0) == 1024 /\
                   (forall (i:nat). i < 1024 ==> S.index (lx r t 0) i == S.index (table_view r) (toks r t * 1024 + i)))
  = emb_tok r t;
    pos_shape r t;
    emb_ok r t;
    if t < pP r then begin
      let aux (i:nat) : Lemma (i < 1024 ==> S.index (lx r t 0) i == S.index (table_view r) (toks r t * 1024 + i)) =
        if i < 1024 then emb_prompt r t i
      in
      Classical.forall_intro aux
    end
    else tok_gen r t

let emb_base (r:run) (rw:S.seq real) (t:nat)
  : Lemma (requires ctx r rw /\ t < cap r) (ensures lx r t 0 %~ (ff r rw t 0).l_x)
  = emb_entries r t;
    fam_lengths r;
    fwd_zero base_numerics rw t (rope_cos base_numerics t) (rope_sin base_numerics t) (rs r rw t).kvs (st0 r rw t);
    let tk = toks r t in
    mul_bound tk 248320 1024;
    let aux (i:nat{i < 1024}) : Lemma (S.index (lx r t 0) i %~ S.index (embed rw tk) i) =
      view_index r.input0 0 254279680 (tk * 1024 + i)
    in
    Classical.forall_intro aux

(*** Layer chain at one position *)

let layer_x (r:run) (rw:S.seq real) (t l:nat)
  : Lemma (requires ctx r rw /\ t < cap r /\ l < 24 /\ ninv r rw t /\ lx r t l %~ (ff r rw t l).l_x)
          (ensures lx r t (l + 1) %~ (ff r rw t (l + 1)).l_x)
  = ninv_at r rw t l;
    if is_attn l then layer_attn r rw t l else layer_dn r rw t l

let rec chain (r:run) (rw:S.seq real) (t l:nat)
  : Lemma (requires ctx r rw /\ t < cap r /\ ninv r rw t /\ l <= 24)
          (ensures lx r t l %~ (ff r rw t l).l_x) (decreases l)
  = if l = 0 then emb_base r rw t
    else begin
      let k : nat = l - 1 in
      chain r rw t k;
      layer_x r rw t k
    end

(*** Next position *)

let cur_all (r:run) (rw:S.seq real) (t l:nat)
  : Lemma (requires rs_lens r rw t /\ l < 24 /\ is_attn l)
          (ensures forall (j:nat). j < 512 ==>
             at (ff r rw t 24).l_cur ((l / 4) * 1024 + j) ==
               at (at_k base_numerics rw (layer_off l) (ff r rw t l).l_x (rope_cos base_numerics t) (rope_sin base_numerics t)) j /\
             at (ff r rw t 24).l_cur ((l / 4) * 1024 + 512 + j) ==
               at (at_v base_numerics rw (layer_off l) (ff r rw t l).l_x) j)
  = slots0 r rw t;
    let aux (j:nat) : Lemma (j < 512 ==>
             at (ff r rw t 24).l_cur ((l / 4) * 1024 + j) ==
               at (at_k base_numerics rw (layer_off l) (ff r rw t l).l_x (rope_cos base_numerics t) (rope_sin base_numerics t)) j /\
             at (ff r rw t 24).l_cur ((l / 4) * 1024 + 512 + j) ==
               at (at_v base_numerics rw (layer_off l) (ff r rw t l).l_x) j) =
      if j < 512 then
        cur_fwd base_numerics rw t (rope_cos base_numerics t) (rope_sin base_numerics t) (rs r rw t).kvs (st0 r rw t) 24 l j
    in
    Classical.forall_intro aux

let in_succ (r:run) (t l:nat)
  : Lemma (kv_in r (t + 1) l == kv_out r t l /\ conv_in r (t + 1) l == conv_out r t l /\
           rec_in r (t + 1) l == rec_out r t l)
  = ()

let state_next_attn (r:run) (rw:S.seq real) (t l:nat)
  : Lemma (requires ctx r rw /\ t < cap r /\ l < 24 /\ is_attn l /\ rs_lens r rw t /\ state_rel r rw t l /\
                    lx r t l %~ (ff r rw t l).l_x)
          (ensures kv_in r (t + 1) l %~ rkv_of (cap r) (rs r rw (t + 1)).kvs (l / 4))
  = layer_attn r rw t l;
    cur_all r rw t l;
    rs_succ r rw t;
    kv_write_ok (cap r) t (rs r rw t).kvs (ff r rw t 24).l_cur
      (at_k base_numerics rw (layer_off l) (ff r rw t l).l_x (rope_cos base_numerics t) (rope_sin base_numerics t))
      (at_v base_numerics rw (layer_off l) (ff r rw t l).l_x) (l / 4);
    in_succ r t l

let state_next_dn (r:run) (rw:S.seq real) (t l:nat)
  : Lemma (requires ctx r rw /\ t < cap r /\ l < 24 /\ not (is_attn l) /\ rs_lens r rw t /\ state_rel r rw t l /\
                    lx r t l %~ (ff r rw t l).l_x)
          (ensures conv_in r (t + 1) l %~ sget (rs r rw (t + 1)).convs (dn_index l) /\
                   rec_in r (t + 1) l %~ sget (rs r rw (t + 1)).recs (dn_index l))
  = layer_dn r rw t l;
    rs_succ r rw t;
    slots0 r rw t;
    conv_fwd base_numerics rw t (rope_cos base_numerics t) (rope_sin base_numerics t) (rs r rw t).kvs (st0 r rw t) 24 l;
    in_succ r t l

let state_next (r:run) (rw:S.seq real) (t l:nat)
  : Lemma (requires ctx r rw /\ t < cap r /\ ninv r rw t /\ l < 24)
          (ensures state_rel r rw (t + 1) l)
  = chain r rw t l;
    ninv_at r rw t l;
    if is_attn l then state_next_attn r rw t l else state_next_dn r rw t l

let ninv_next (r:run) (rw:S.seq real) (t:nat)
  : Lemma (requires ctx r rw /\ t < cap r /\ ninv r rw t) (ensures ninv r rw (t + 1))
  = ninv_at r rw t 0;
    rs_succ r rw t;
    slots0 r rw t;
    slots_fwd base_numerics rw t (rope_cos base_numerics t) (rope_sin base_numerics t) (rs r rw t).kvs (st0 r rw t) 24;
    S.lemma_len_append (rs r rw t).kvs (S.create 1 (ff r rw t 24).l_cur);
    let aux (l:nat) : Lemma (l < 24 ==> state_rel r rw (t + 1) l) =
      if l < 24 then state_next r rw t l
    in
    Classical.forall_intro aux;
    reveal_opaque (`%ninv) (ninv r rw (t + 1))

(*** Position 0 *)

let view_zero_bf (s:S.seq BF.t) (o n:int)
  : Lemma (requires all_zero s /\ S.length (view s o n) == n /\ n > 0) (ensures view s o n %~ S.create n 0.0R)
  = let aux (i:nat{i < n}) : Lemma (S.index (view s o n) i %~ S.index (S.create n 0.0R) i) =
      view_index s o n i; zero_bf ()
    in
    Classical.forall_intro aux

let view_zero_f32 (s:S.seq F32.t) (o n:int)
  : Lemma (requires all_zero s /\ S.length (view s o n) == n /\ n > 0) (ensures view s o n %~ S.create n 0.0R)
  = let aux (i:nat{i < n}) : Lemma (S.index (view s o n) i %~ S.index (S.create n 0.0R) i) =
      view_index s o n i; zero_f32 ()
    in
    Classical.forall_intro aux

let zero_facts (r:run)
  : Lemma (requires premise r) (ensures all_zero r.initialize_1 /\ all_zero r.initialize_2)
  = init_facts r

#push-options "--z3rlimit 40"
let state_0 (r:run) (rw:S.seq real) (l:nat)
  : Lemma (requires ctx r rw /\ l < 24 /\ shape_in r 0 /\ rs r rw 0 == init_state /\
                    all_zero r.initialize_1 /\ all_zero r.initialize_2 /\ 1 <= cap r /\ kv_len r == cap r * 1024)
          (ensures state_rel r rw 0 l)
  = shape_in_at r 0 l;
    if is_attn l then begin
      view_zero_bf r.initialize_1 (kv_off r l) (kv_len r);
      rkv_empty (cap r) (l / 4)
    end else begin
      dn_index_facts l l;
      view_zero_bf r.initialize_1 (24576 * dn_index l) 24576;
      view_zero_f32 r.initialize_2 (262144 * dn_index l) 262144
    end

#pop-options

let ninv_0 (r:run) (rw:S.seq real)
  : Lemma (requires ctx r rw) (ensures ninv r rw 0)
  = ref_run_zero base_numerics rw (toks r);
    shape_0 r;
    zero_facts r;
    fam_lengths r;
    kv_len_cap r;
    let aux (l:nat) : Lemma (l < 24 ==> state_rel r rw 0 l) =
      if l < 24 then state_0 r rw l
    in
    Classical.forall_intro aux;
    reveal_opaque (`%ninv) (ninv r rw 0)

let rec ninv_all (r:run) (rw:S.seq real) (t:nat)
  : Lemma (requires ctx r rw /\ t < cap r) (ensures ninv r rw t) (decreases t)
  = if t = 0 then ninv_0 r rw
    else begin
      ninv_all r rw (t - 1);
      ninv_next r rw (t - 1)
    end

(*** Exports *)

let pos_ok (r:run) (rw:S.seq real) (t:nat)
  : Lemma (requires ctx r rw /\ t < cap r)
          (ensures lx r t 24 %~ ref_hidden base_numerics rw (toks r) t /\
                   ulog r t %~ ref_logits base_numerics rw (ref_hidden base_numerics rw (toks r) t))
  = ninv_all r rw t;
    chain r rw t 24;
    rs_succ r rw t;
    pos_shape r t;
    fam_lengths r;
    table_len_ok r;
    fin_rel_ok r t;
    reveal_opaque (`%fin_rel) (fin_rel r t);
    let h = (ff r rw t 24).l_x in
    let rt = view rw 0 254279680 in
    let rn = view rw 254279680 1024 in
    view_approx_bf r.input0 rw 0 254279680;
    view_approx_bf r.input0 rw 254279680 1024;
    fin_inst (table_view r) (norm_view r) (lx r t 24) (ufin r t) (ulog r t) rt rn h;
    view_wagree rw 0 254279680;
    view_wagree rw 254279680 1024;
    logits_congr base_numerics rt 0 rw 0 rn 0 rw final_norm_off h
