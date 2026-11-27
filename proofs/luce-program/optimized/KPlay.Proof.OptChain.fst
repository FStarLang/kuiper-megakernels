module KPlay.Proof.OptChain

(* One decode position of the optimized program: the 24 decoder-layer calls
   as a boundary record, one generic chain lemma over the layer index, and
   the resulting position step against ref_step. *)

open FStar.Real
open Kuiper
open Kuiper.Approximates
open KPlay.SharedSpec
open KPlay.Proof.Wiring.Optimized
open KPlay.Proof.OptWin
open KPlay.Proof.OptApprox
open KPlay.Proof.OptState
module S = FStar.Seq
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32

#set-options "--fuel 0 --ifuel 0 --z3rlimit 50"

(* Buffers at the layer boundaries of one position: xs l / rcs l / cvs l /
   ks l / vs l are the buffers consumed by layer l (l = 24: final ones). *)
noeq type pexec = {
  pw  : S.seq BF.t;
  xs  : nat -> S.seq F32.t;
  rcs : nat -> S.seq F32.t;
  cvs : nat -> S.seq F32.t;
  ks  : nat -> S.seq BF.t;
  vs  : nat -> S.seq BF.t;
  pcs : S.seq F32.t;
  psn : S.seq F32.t;
  pt  : nat;
}

[@@"opaque_to_smt"]
let layer_ok (e:pexec) (l:nat) : prop =
  if is_attn l then
    optimized_attention_decoder_relation (view e.pw (layer_off l) attn_len) (e.xs l) (e.ks l) (e.vs l)
      e.pcs e.psn 0 (l / 4) e.pt (e.xs (l + 1)) (e.ks (l + 1)) (e.vs (l + 1)) /\
    e.rcs (l + 1) == e.rcs l /\ e.cvs (l + 1) == e.cvs l
  else
    optimized_recurrent_decoder_relation (view e.pw (layer_off l) dn_len) (e.xs l) (e.rcs l) (e.cvs l)
      0 (dn_index l) (e.xs (l + 1)) (e.rcs (l + 1)) (e.cvs (l + 1)) /\
    e.ks (l + 1) == e.ks l /\ e.vs (l + 1) == e.vs l

let ctx (e:pexec) (rw:S.seq real) (kvs:S.seq (S.seq real)) : prop =
  S.length e.pw == weights_len /\ e.pw %~ rw /\
  S.length e.pcs == 32 /\ S.length e.psn == 32 /\
  e.pcs %~ rope_cos opt_numerics e.pt /\ e.psn %~ rope_sin opt_numerics e.pt /\
  S.length kvs == e.pt /\ e.pt < 2048

let inv (e:pexec) (l:nat) (kvs:S.seq (S.seq real)) (st:lstate) : prop =
  S.length (e.xs l) == 1024 /\ e.xs l %~ st.l_x /\
  S.length (e.rcs l) == opt_rec_len /\ e.rcs l %~ enc_rec st.l_recs /\
  S.length (e.cvs l) == opt_conv_len /\ e.cvs l %~ enc_conv st.l_convs /\
  shaped st.l_convs st.l_recs /\
  S.length (e.ks l) == opt_kv_len /\ e.ks l %~ enc_kv 0 kvs st.l_cur /\
  S.length (e.vs l) == opt_kv_len /\ e.vs l %~ enc_kv 512 kvs st.l_cur

let layer_bounds (l:nat)
  : Lemma (requires l < 24)
          (ensures 0 <= layer_off l /\ l / 4 < 6 /\
                   (is_attn l ==> layer_off l + attn_len <= weights_len) /\
                   (not (is_attn l) ==> layer_off l + dn_len <= weights_len /\ dn_index l < 18))
  = ()

let attn_elim (w:S.seq BF.t) (x:S.seq F32.t) (k v:S.seq BF.t) (cos sin:S.seq F32.t) (fa pos:int)
              (x':S.seq F32.t) (k' v':S.seq BF.t) (rw rx rk rv rc rs:S.seq real)
  : Lemma (requires optimized_attention_decoder_relation w x k v cos sin 0 fa pos x' k' v' /\
                    attn_len <= S.length w /\ S.length x == 1024 /\
                    S.length k == opt_kv_len /\ S.length v == opt_kv_len /\ S.length cos == 32 /\ S.length sin == 32 /\
                    0 <= fa /\ fa < 6 /\ 0 <= pos /\ pos < 2048 /\
                    w %~ rw /\ x %~ rx /\ k %~ rk /\ v %~ rv /\ cos %~ rc /\ sin %~ rs)
          (ensures S.length x' == 1024 /\ S.length k' == opt_kv_len /\ S.length v' == opt_kv_len /\
                   x' %~ at_out opt_numerics rw 0 rx rc rs pos (opt_hk fa rk) (opt_hk fa rv) /\
                   k' %~ opt_kv_write fa pos rk (at_k opt_numerics rw 0 rx rc rs) /\
                   v' %~ opt_kv_write fa pos rv (at_v opt_numerics rw 0 rx))
  = ()

let dn_elim (w:S.seq BF.t) (x rc cv:S.seq F32.t) (dn:int) (x' rc' cv':S.seq F32.t) (rw rx rr rcv:S.seq real)
  : Lemma (requires optimized_recurrent_decoder_relation w x rc cv 0 dn x' rc' cv' /\
                    dn_len <= S.length w /\ S.length x == 1024 /\
                    S.length rc == opt_rec_len /\ S.length cv == opt_conv_len /\ 0 <= dn /\ dn < 18 /\
                    w %~ rw /\ x %~ rx /\ rc %~ rr /\ cv %~ rcv)
          (ensures S.length x' == 1024 /\ S.length rc' == opt_rec_len /\ S.length cv' == opt_conv_len /\
                   x' %~ dn_out opt_numerics rw 0 rx (opt_conv_slot rcv dn) (opt_rec_slot rr dn) /\
                   cv' %~ opt_conv_put rcv dn (dn_conv opt_numerics rw 0 rx (opt_conv_slot rcv dn)) /\
                   rc' %~ opt_rec_put rr dn (dn_state opt_numerics rw 0 rx (opt_conv_slot rcv dn) (opt_rec_slot rr dn)))
  = ()

let layer_ok_attn (e:pexec) (l:nat)
  : Lemma (requires is_attn l /\ layer_ok e l)
          (ensures optimized_attention_decoder_relation (view e.pw (layer_off l) attn_len) (e.xs l) (e.ks l) (e.vs l)
                     e.pcs e.psn 0 (l / 4) e.pt (e.xs (l + 1)) (e.ks (l + 1)) (e.vs (l + 1)) /\
                   e.rcs (l + 1) == e.rcs l /\ e.cvs (l + 1) == e.cvs l)
  = reveal_opaque (`%layer_ok) (layer_ok e l)

let layer_ok_dn (e:pexec) (l:nat)
  : Lemma (requires not (is_attn l) /\ layer_ok e l)
          (ensures optimized_recurrent_decoder_relation (view e.pw (layer_off l) dn_len) (e.xs l) (e.rcs l) (e.cvs l)
                     0 (dn_index l) (e.xs (l + 1)) (e.rcs (l + 1)) (e.cvs (l + 1)) /\
                   e.ks (l + 1) == e.ks l /\ e.vs (l + 1) == e.vs l)
  = reveal_opaque (`%layer_ok) (layer_ok e l)

let dn_next (w:S.seq real) (st:lstate) (l:nat) : lstate =
  { l_convs = supd st.l_convs (dn_index l) (dn_conv opt_numerics w (layer_off l) st.l_x (sget st.l_convs (dn_index l)));
    l_recs = supd st.l_recs (dn_index l) (dn_state opt_numerics w (layer_off l) st.l_x (sget st.l_convs (dn_index l)) (sget st.l_recs (dn_index l)));
    l_cur = st.l_cur;
    l_x = dn_out opt_numerics w (layer_off l) st.l_x (sget st.l_convs (dn_index l)) (sget st.l_recs (dn_index l)) }

let attn_next (w:S.seq real) (t:nat) (cs sn:S.seq real) (kvs:S.seq (S.seq real)) (st:lstate) (l:nat) : lstate =
  { st with
    l_cur = write_kv st.l_cur (l / 4) (at_k opt_numerics w (layer_off l) st.l_x cs sn) (at_v opt_numerics w (layer_off l) st.l_x);
    l_x = at_out opt_numerics w (layer_off l) st.l_x cs sn t
            (fun (p j:nat) -> kv_get kvs p ((l / 4) * 1024 + j))
            (fun (p j:nat) -> kv_get kvs p ((l / 4) * 1024 + 512 + j)) }

#push-options "--fuel 1"
let unfold_layer (w:S.seq real) (t:nat) (cs sn:S.seq real) (kvs:S.seq (S.seq real)) (st:lstate) (l:nat)
  : Lemma (run_layers opt_numerics w t cs sn kvs st l ==
           (if l >= 24 then st
            else if is_attn l then run_layers opt_numerics w t cs sn kvs (attn_next w t cs sn kvs st l) (l + 1)
            else run_layers opt_numerics w t cs sn kvs (dn_next w st l) (l + 1)))
  = ()
#pop-options

let dn_step (e:pexec) (rw:S.seq real) (kvs:S.seq (S.seq real)) (st:lstate) (l:nat)
  : Lemma (requires l < 24 /\ not (is_attn l) /\ ctx e rw kvs /\ inv e l kvs st /\ layer_ok e l)
          (ensures inv e (l + 1) kvs (dn_next rw st l))
  = layer_ok_dn e l;
    layer_bounds l;
    let off : nat = layer_off l in
    let d = dn_index l in
    let rw' = S.slice rw off (off + dn_len) in
    view_approx e.pw rw off dn_len;
    slice_win rw off dn_len;
    let rx = st.l_x in
    let rr = enc_rec st.l_recs in
    let rcv = enc_conv st.l_convs in
    let cv = sget st.l_convs d in
    let rs = sget st.l_recs d in
    conv_slot st.l_convs st.l_recs d;
    rec_slot st.l_convs st.l_recs d;
    dn_elim (view e.pw off dn_len) (e.xs l) (e.rcs l) (e.cvs l) d (e.xs (l + 1)) (e.rcs (l + 1)) (e.cvs (l + 1)) rw' rx rr rcv;
    dn_out_ext opt_numerics rw' 0 rw off rx cv rs;
    dn_conv_ext opt_numerics rw' 0 rw off rx cv;
    dn_state_ext opt_numerics rw' 0 rw off rx cv rs;
    conv_put st.l_convs d (dn_conv opt_numerics rw off rx cv);
    rec_put st.l_recs d (dn_state opt_numerics rw off rx cv rs);
    shaped_put st.l_convs st.l_recs d (dn_conv opt_numerics rw off rx cv) (dn_state opt_numerics rw off rx cv rs)

let attn_raw (e:pexec) (rw:S.seq real) (kvs:S.seq (S.seq real)) (st:lstate) (l:nat)
  : Lemma (requires l < 24 /\ is_attn l /\ ctx e rw kvs /\ inv e l kvs st /\ layer_ok e l)
          (ensures (let off = layer_off l in
                    let rw' = view rw off attn_len in
                    let rc = rope_cos opt_numerics e.pt in
                    let rs = rope_sin opt_numerics e.pt in
                    let rk = enc_kv 0 kvs st.l_cur in
                    let rv = enc_kv 512 kvs st.l_cur in
                    S.length (e.xs (l + 1)) == 1024 /\ S.length (e.ks (l + 1)) == opt_kv_len /\
                    S.length (e.vs (l + 1)) == opt_kv_len /\
                    e.xs (l + 1) %~ at_out opt_numerics rw' 0 st.l_x rc rs e.pt (opt_hk (l / 4) rk) (opt_hk (l / 4) rv) /\
                    e.ks (l + 1) %~ opt_kv_write (l / 4) e.pt rk (at_k opt_numerics rw' 0 st.l_x rc rs) /\
                    e.vs (l + 1) %~ opt_kv_write (l / 4) e.pt rv (at_v opt_numerics rw' 0 st.l_x)))
  = layer_ok_attn e l;
    layer_bounds l;
    let off : nat = layer_off l in
    view_approx e.pw rw off attn_len;
    attn_elim (view e.pw off attn_len) (e.xs l) (e.ks l) (e.vs l) e.pcs e.psn (l / 4) e.pt
      (e.xs (l + 1)) (e.ks (l + 1)) (e.vs (l + 1))
      (view rw off attn_len) st.l_x (enc_kv 0 kvs st.l_cur) (enc_kv 512 kvs st.l_cur)
      (rope_cos opt_numerics e.pt) (rope_sin opt_numerics e.pt)

let attn_eqs (rw:S.seq real) (t:nat) (cs sn:S.seq real) (kvs:S.seq (S.seq real)) (st:lstate) (l:nat)
  : Lemma (requires l < 24 /\ is_attn l /\ S.length rw == weights_len /\ S.length kvs == t /\ t < 2048)
          (ensures (let off = layer_off l in
                    let rw' = view rw off attn_len in
                    let rk = enc_kv 0 kvs st.l_cur in
                    let rv = enc_kv 512 kvs st.l_cur in
                    let st' = attn_next rw t cs sn kvs st l in
                    at_out opt_numerics rw' 0 st.l_x cs sn t (opt_hk (l / 4) rk) (opt_hk (l / 4) rv) == st'.l_x /\
                    opt_kv_write (l / 4) t rk (at_k opt_numerics rw' 0 st.l_x cs sn) == enc_kv 0 kvs st'.l_cur /\
                    opt_kv_write (l / 4) t rv (at_v opt_numerics rw' 0 st.l_x) == enc_kv 512 kvs st'.l_cur /\
                    st'.l_convs == st.l_convs /\ st'.l_recs == st.l_recs))
  = layer_bounds l;
    let off : nat = layer_off l in
    let g = l / 4 in
    let rw' = view rw off attn_len in
    slice_win rw off attn_len;
    let rx = st.l_x in
    let rk = enc_kv 0 kvs st.l_cur in
    let rv = enc_kv 512 kvs st.l_cur in
    let hk = fun (p j:nat) -> kv_get kvs p ((l / 4) * 1024 + j) in
    let hv = fun (p j:nat) -> kv_get kvs p ((l / 4) * 1024 + 512 + j) in
    introduce forall (p j:nat). p < t /\ j < 512 ==> opt_hk g rk p j == hk p j
    with introduce _ ==> _ with kv_read 0 kvs st.l_cur g p j;
    introduce forall (p j:nat). p < t /\ j < 512 ==> opt_hk g rv p j == hv p j
    with introduce _ ==> _ with kv_read 512 kvs st.l_cur g p j;
    at_out_ext opt_numerics rw' 0 rw off rx cs sn t (opt_hk g rk) (opt_hk g rv) hk hv;
    at_qk_ext opt_numerics rw' 0 rw off rx cs sn;
    at_proj_ext opt_numerics rw' 0 rw off rx;
    let kn = at_k opt_numerics rw off rx cs sn in
    let vn = at_v opt_numerics rw off rx in
    kv_write 0 kvs st.l_cur g t kn vn kn;
    kv_write 512 kvs st.l_cur g t kn vn vn

let attn_step (e:pexec) (rw:S.seq real) (kvs:S.seq (S.seq real)) (st:lstate) (l:nat)
  : Lemma (requires l < 24 /\ is_attn l /\ ctx e rw kvs /\ inv e l kvs st /\ layer_ok e l)
          (ensures inv e (l + 1) kvs
                     (attn_next rw e.pt (rope_cos opt_numerics e.pt) (rope_sin opt_numerics e.pt) kvs st l))
  = layer_ok_attn e l;
    attn_raw e rw kvs st l;
    attn_eqs rw e.pt (rope_cos opt_numerics e.pt) (rope_sin opt_numerics e.pt) kvs st l

(* The generic chain: layers l .. 23 of one position. *)
let rec chain (e:pexec) (rw:S.seq real) (kvs:S.seq (S.seq real)) (st:lstate) (l:nat)
  : Lemma (requires l <= 24 /\ ctx e rw kvs /\ inv e l kvs st /\
                    (forall (l':nat). l <= l' /\ l' < 24 ==> layer_ok e l'))
          (ensures inv e 24 kvs
                     (run_layers opt_numerics rw e.pt (rope_cos opt_numerics e.pt) (rope_sin opt_numerics e.pt) kvs st l))
          (decreases (24 - l))
  = let cs = rope_cos opt_numerics e.pt in
    let sn = rope_sin opt_numerics e.pt in
    unfold_layer rw e.pt cs sn kvs st l;
    if l = 24 then ()
    else if is_attn l then begin
      attn_step e rw kvs st l;
      chain e rw kvs (attn_next rw e.pt cs sn kvs st l) (l + 1)
    end else begin
      dn_step e rw kvs st l;
      chain e rw kvs (dn_next rw st l) (l + 1)
    end

(*** One position *)

let pinv (rc cv:S.seq F32.t) (k v:S.seq BF.t) (s:rstate) (t:nat) : prop =
  S.length rc == opt_rec_len /\ rc %~ enc_rec s.recs /\
  S.length cv == opt_conv_len /\ cv %~ enc_conv s.convs /\
  shaped s.convs s.recs /\
  S.length k == opt_kv_len /\ k %~ enc_kv 0 s.kvs (S.create 6144 0.0R) /\
  S.length v == opt_kv_len /\ v %~ enc_kv 512 s.kvs (S.create 6144 0.0R) /\
  S.length s.kvs == t

let pctx (e:pexec) (rw:S.seq real) : prop =
  S.length e.pw == weights_len /\ e.pw %~ rw /\
  S.length e.pcs == 32 /\ S.length e.psn == 32 /\
  e.pcs %~ rope_cos opt_numerics e.pt /\ e.psn %~ rope_sin opt_numerics e.pt /\
  e.pt < 2048

let position (e:pexec) (rw:S.seq real) (tok:int) (s:rstate)
  : Lemma (requires pctx e rw /\ pinv (e.rcs 0) (e.cvs 0) (e.ks 0) (e.vs 0) s e.pt /\
                    S.length (e.xs 0) == 1024 /\ e.xs 0 %~ embed rw tok /\
                    (forall (l:nat). l < 24 ==> layer_ok e l))
          (ensures pinv (e.rcs 24) (e.cvs 24) (e.ks 24) (e.vs 24) (fst (ref_step opt_numerics rw tok e.pt s)) (e.pt + 1) /\
                   S.length (e.xs 24) == 1024 /\ e.xs 24 %~ snd (ref_step opt_numerics rw tok e.pt s))
  = let cs = rope_cos opt_numerics e.pt in
    let sn = rope_sin opt_numerics e.pt in
    let st0 = { l_convs = s.convs; l_recs = s.recs; l_cur = S.create 6144 0.0R; l_x = embed rw tok } in
    chain e rw s.kvs st0 0;
    let r = run_layers opt_numerics rw e.pt cs sn s.kvs st0 0 in
    kv_new 0 s.kvs r.l_cur;
    kv_new 512 s.kvs r.l_cur
