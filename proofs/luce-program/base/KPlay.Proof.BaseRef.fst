module KPlay.Proof.BaseRef

(* Forward view of the reference layer stack (run_layers is tail recursive),
   with size-generic frame facts for the per-layer state slots. *)

open FStar.Real
open KPlay.SharedSpec
module S = FStar.Seq

#set-options "--z3rlimit 20 --fuel 0 --ifuel 0"

(* One layer of run_layers (its body without the recursive call). *)
let layer_step (nm:numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real)
               (kvs:S.seq (S.seq real)) (st:lstate) (l:nat) : lstate =
    let wo = layer_off l in
    if is_attn l then
      let g = l / 4 in
      let hk = fun (p j:nat) -> kv_get kvs p (g * 1024 + j) in
      let hv = fun (p j:nat) -> kv_get kvs p (g * 1024 + 512 + j) in
        ({ st with
           l_cur = write_kv st.l_cur g (at_k nm w wo st.l_x cs sn) (at_v nm w wo st.l_x);
           l_x = at_out nm w wo st.l_x cs sn t hk hv })
    else
      let d = dn_index l in
      let cv = sget st.l_convs d in
      let rs = sget st.l_recs d in
        ({ l_convs = supd st.l_convs d (dn_conv nm w wo st.l_x cv);
           l_recs = supd st.l_recs d (dn_state nm w wo st.l_x cv rs);
           l_cur = st.l_cur;
           l_x = dn_out nm w wo st.l_x cv rs })

let layer_step_attn (nm:numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real)
                    (kvs:S.seq (S.seq real)) (st:lstate) (l:nat)
  : Lemma (requires is_attn l)
          (ensures (let st' = layer_step nm w t cs sn kvs st l in
                    st'.l_convs == st.l_convs /\ st'.l_recs == st.l_recs /\
                    st'.l_cur == write_kv st.l_cur (l / 4) (at_k nm w (layer_off l) st.l_x cs sn)
                                          (at_v nm w (layer_off l) st.l_x) /\
                    st'.l_x == at_out nm w (layer_off l) st.l_x cs sn t
                        (fun (p j:nat) -> kv_get kvs p ((l / 4) * 1024 + j))
                        (fun (p j:nat) -> kv_get kvs p ((l / 4) * 1024 + 512 + j))))
  = ()

let layer_step_dn (nm:numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real)
                  (kvs:S.seq (S.seq real)) (st:lstate) (l:nat)
  : Lemma (requires not (is_attn l))
          (ensures (let st' = layer_step nm w t cs sn kvs st l in
                    let d = dn_index l in
                    let cv = sget st.l_convs d in
                    let rs = sget st.l_recs d in
                    st'.l_convs == supd st.l_convs d (dn_conv nm w (layer_off l) st.l_x cv) /\
                    st'.l_recs == supd st.l_recs d (dn_state nm w (layer_off l) st.l_x cv rs) /\
                    st'.l_cur == st.l_cur /\
                    st'.l_x == dn_out nm w (layer_off l) st.l_x cv rs))
  = ()

#push-options "--fuel 1"
let run_unfold (nm:numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real)
               (kvs:S.seq (S.seq real)) (st:lstate) (l:nat{l < 24})
  : Lemma (run_layers nm w t cs sn kvs st l == run_layers nm w t cs sn kvs (layer_step nm w t cs sn kvs st l) (l + 1))
  = ()

let run_done (nm:numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real)
             (kvs:S.seq (S.seq real)) (st:lstate)
  : Lemma (run_layers nm w t cs sn kvs st 24 == st)
  = ()
#pop-options

(* Layers 0 .. l-1 applied to st0. *)
let rec fwd (nm:numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real)
            (kvs:S.seq (S.seq real)) (st0:lstate) (l:nat)
  : Tot lstate (decreases l) =
  if l = 0 then st0 else layer_step nm w t cs sn kvs (fwd nm w t cs sn kvs st0 (l - 1)) (l - 1)

#push-options "--fuel 1"
let fwd_succ (nm:numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real)
             (kvs:S.seq (S.seq real)) (st0:lstate) (l:nat)
  : Lemma (fwd nm w t cs sn kvs st0 (l + 1) == layer_step nm w t cs sn kvs (fwd nm w t cs sn kvs st0 l) l)
  = ()

let fwd_zero (nm:numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real)
             (kvs:S.seq (S.seq real)) (st0:lstate)
  : Lemma (fwd nm w t cs sn kvs st0 0 == st0)
  = ()
#pop-options

let rec run_fwd (nm:numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real)
                (kvs:S.seq (S.seq real)) (st0:lstate) (l:nat{l <= 24})
  : Lemma (ensures run_layers nm w t cs sn kvs st0 0 == run_layers nm w t cs sn kvs (fwd nm w t cs sn kvs st0 l) l)
          (decreases l)
  = if l = 0 then fwd_zero nm w t cs sn kvs st0
    else begin
      run_fwd nm w t cs sn kvs st0 (l - 1);
      run_unfold nm w t cs sn kvs (fwd nm w t cs sn kvs st0 (l - 1)) (l - 1);
      fwd_succ nm w t cs sn kvs st0 (l - 1)
    end

let run_layers_fwd (nm:numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real)
                   (kvs:S.seq (S.seq real)) (st0:lstate)
  : Lemma (run_layers nm w t cs sn kvs st0 0 == fwd nm w t cs sn kvs st0 24)
  = run_fwd nm w t cs sn kvs st0 24;
    run_done nm w t cs sn kvs (fwd nm w t cs sn kvs st0 24)

(*** Slot arithmetic *)

let dn_index_facts (l1 l2:nat)
  : Lemma (requires l1 < 24 /\ l2 < 24 /\ not (is_attn l1) /\ not (is_attn l2))
          (ensures dn_index l1 < 18 /\ (dn_index l1 == dn_index l2 <==> l1 == l2))
  = ()

let sget_supd (s:S.seq (S.seq real)) (i j:int) (v:S.seq real)
  : Lemma (requires 0 <= i /\ i < S.length s)
          (ensures S.length (supd s i v) == S.length s /\
                   sget (supd s i v) j == (if j = i then v else sget s j))
  = ()

let write_kv_at (cur:S.seq real) (g g':nat) (kn vn:S.seq real) (j:nat)
  : Lemma (requires g' < 6 /\ j < 512)
          (ensures S.length (write_kv cur g kn vn) == 6144 /\
                   at (write_kv cur g kn vn) (g' * 1024 + j) == (if g' = g then at kn j else at cur (g' * 1024 + j)) /\
                   at (write_kv cur g kn vn) (g' * 1024 + 512 + j) == (if g' = g then at vn j else at cur (g' * 1024 + 512 + j)))
  = ()

(*** Frame facts *)

let slots_ok (st:lstate) : prop =
  S.length st.l_convs == 18 /\ S.length st.l_recs == 18 /\ S.length st.l_cur == 6144

let step_conv (nm:numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real)
              (kvs:S.seq (S.seq real)) (st:lstate) (k l':nat)
  : Lemma (requires slots_ok st /\ k < 24 /\ l' < 24 /\ not (is_attn l'))
          (ensures (let st' = layer_step nm w t cs sn kvs st k in
                    let d = dn_index l' in
                    slots_ok st' /\
                    sget st'.l_convs d ==
                      (if k = l' then dn_conv nm w (layer_off l') st.l_x (sget st.l_convs d) else sget st.l_convs d) /\
                    sget st'.l_recs d ==
                      (if k = l' then dn_state nm w (layer_off l') st.l_x (sget st.l_convs d) (sget st.l_recs d)
                       else sget st.l_recs d)))
  = if is_attn k then layer_step_attn nm w t cs sn kvs st k
    else begin
      layer_step_dn nm w t cs sn kvs st k;
      dn_index_facts k l';
      let d = dn_index k in
      sget_supd st.l_convs d (dn_index l') (dn_conv nm w (layer_off k) st.l_x (sget st.l_convs d));
      sget_supd st.l_recs d (dn_index l') (dn_state nm w (layer_off k) st.l_x (sget st.l_convs d) (sget st.l_recs d))
    end

let step_cur (nm:numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real)
             (kvs:S.seq (S.seq real)) (st:lstate) (k l' j:nat)
  : Lemma (requires slots_ok st /\ k < 24 /\ l' < 24 /\ is_attn l' /\ j < 512)
          (ensures (let st' = layer_step nm w t cs sn kvs st k in
                    let g = l' / 4 in
                    at st'.l_cur (g * 1024 + j) ==
                      (if k = l' then at (at_k nm w (layer_off l') st.l_x cs sn) j else at st.l_cur (g * 1024 + j)) /\
                    at st'.l_cur (g * 1024 + 512 + j) ==
                      (if k = l' then at (at_v nm w (layer_off l') st.l_x) j else at st.l_cur (g * 1024 + 512 + j))))
  = if is_attn k then begin
      layer_step_attn nm w t cs sn kvs st k;
      write_kv_at st.l_cur (k / 4) (l' / 4) (at_k nm w (layer_off k) st.l_x cs sn) (at_v nm w (layer_off k) st.l_x) j
    end else layer_step_dn nm w t cs sn kvs st k

let conv_frame (nm:numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real)
               (kvs:S.seq (S.seq real)) (st0:lstate) (l l':nat) : prop =
  let d = dn_index l' in
  sget (fwd nm w t cs sn kvs st0 l).l_convs d ==
    (if l' < l then dn_conv nm w (layer_off l') (fwd nm w t cs sn kvs st0 l').l_x (sget st0.l_convs d)
     else sget st0.l_convs d) /\
  sget (fwd nm w t cs sn kvs st0 l).l_recs d ==
    (if l' < l then dn_state nm w (layer_off l') (fwd nm w t cs sn kvs st0 l').l_x (sget st0.l_convs d) (sget st0.l_recs d)
     else sget st0.l_recs d)

let cur_frame (nm:numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real)
              (kvs:S.seq (S.seq real)) (st0:lstate) (l l' j:nat) : prop =
  let g = l' / 4 in
  at (fwd nm w t cs sn kvs st0 l).l_cur (g * 1024 + j) ==
    (if l' < l then at (at_k nm w (layer_off l') (fwd nm w t cs sn kvs st0 l').l_x cs sn) j
     else at st0.l_cur (g * 1024 + j)) /\
  at (fwd nm w t cs sn kvs st0 l).l_cur (g * 1024 + 512 + j) ==
    (if l' < l then at (at_v nm w (layer_off l') (fwd nm w t cs sn kvs st0 l').l_x) j
     else at st0.l_cur (g * 1024 + 512 + j))

let rec slots_fwd (nm:numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real)
                  (kvs:S.seq (S.seq real)) (st0:lstate) (l:nat{l <= 24})
  : Lemma (requires slots_ok st0) (ensures slots_ok (fwd nm w t cs sn kvs st0 l)) (decreases l)
  = if l = 0 then fwd_zero nm w t cs sn kvs st0
    else begin
      let k : nat = l - 1 in
      slots_fwd nm w t cs sn kvs st0 k;
      fwd_succ nm w t cs sn kvs st0 k;
      step_conv nm w t cs sn kvs (fwd nm w t cs sn kvs st0 k) k 0
    end

let rec conv_fwd (nm:numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real)
                 (kvs:S.seq (S.seq real)) (st0:lstate) (l:nat{l <= 24}) (l':nat)
  : Lemma (requires slots_ok st0 /\ l' < 24 /\ not (is_attn l'))
          (ensures conv_frame nm w t cs sn kvs st0 l l') (decreases l)
  = if l = 0 then fwd_zero nm w t cs sn kvs st0
    else begin
      let k : nat = l - 1 in
      conv_fwd nm w t cs sn kvs st0 k l';
      slots_fwd nm w t cs sn kvs st0 k;
      fwd_succ nm w t cs sn kvs st0 k;
      step_conv nm w t cs sn kvs (fwd nm w t cs sn kvs st0 k) k l'
    end

let rec cur_fwd (nm:numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real)
                (kvs:S.seq (S.seq real)) (st0:lstate) (l:nat{l <= 24}) (l' j:nat)
  : Lemma (requires slots_ok st0 /\ l' < 24 /\ is_attn l' /\ j < 512)
          (ensures cur_frame nm w t cs sn kvs st0 l l' j) (decreases l)
  = if l = 0 then fwd_zero nm w t cs sn kvs st0
    else begin
      let k : nat = l - 1 in
      cur_fwd nm w t cs sn kvs st0 k l' j;
      slots_fwd nm w t cs sn kvs st0 k;
      fwd_succ nm w t cs sn kvs st0 k;
      step_cur nm w t cs sn kvs (fwd nm w t cs sn kvs st0 k) k l' j
    end
