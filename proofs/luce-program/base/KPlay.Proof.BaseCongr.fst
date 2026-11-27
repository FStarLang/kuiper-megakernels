module KPlay.Proof.BaseCongr

(* Weight-footprint and KV-history congruence of the reference functions:
   a reference function evaluated on a sliced weight view at offset 0 equals
   the same function on the full weights at the layer offset. Size-generic;
   no unrolling. *)

open FStar.Real
open Kuiper
open Kuiper.Approximates
open KPlay.SharedSpec
module S = FStar.Seq
module BF = Kuiper.BFloat16
module ML = FStar.Math.Lemmas

#set-options "--z3rlimit 20 --fuel 0 --ifuel 0"

(*** Footprint agreement *)

let wagree (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (n:int) : prop =
  forall (i:int). {:pattern (at w1 i)} o1 <= i /\ i < o1 + n ==> at w1 i == at w2 (o2 + (i - o1))

let wagree_sub (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (n e m:int)
  : Lemma (requires wagree w1 o1 w2 o2 n /\ 0 <= e /\ e + m <= n)
          (ensures wagree w1 (o1 + e) w2 (o2 + e) m)
  = ()

let wagree_weaken (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (n m:int)
  : Lemma (requires wagree w1 o1 w2 o2 n /\ m <= n)
          (ensures wagree w1 o1 w2 o2 m)
  = ()

let vsum_ext (n:nat) (f g:nat -> real)
  : Lemma (requires forall (k:nat). k < n ==> f k == g k) (ensures vsum n f == vsum n g)
  = S.lemma_eq_intro (S.init n (fun (k:nat{k < n}) -> f k)) (S.init n (fun (k:nat{k < n}) -> g k))

let vsum_ext_pat (n:nat) (f g:nat -> real)
  : Lemma (requires forall (k:nat). k < n ==> f k == g k) (ensures vsum n f == vsum n g)
    [SMTPat (vsum n f); SMTPat (vsum n g)]
  = vsum_ext n f g

(*** Leaves *)

let dot_congr (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (x:S.seq real) (n:nat)
  : Lemma (requires wagree w1 o1 w2 o2 n) (ensures dot w1 o1 x n == dot w2 o2 x n)
  = vsum_ext n (fun k -> at w1 (o1 + k) *. at x k) (fun k -> at w2 (o2 + k) *. at x k)

let matvec_congr (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (rows cols:nat) (x:S.seq real)
  : Lemma (requires wagree w1 o1 w2 o2 (rows * cols))
          (ensures matvec w1 o1 rows cols x == matvec w2 o2 rows cols x)
  = let aux (r:nat{r < rows}) : Lemma (dot w1 (o1 + r * cols) x cols == dot w2 (o2 + r * cols) x cols) =
      ML.lemma_mult_le_right cols (r + 1) rows;
      ML.distributivity_add_left r 1 cols;
      wagree_sub w1 o1 w2 o2 (rows * cols) (r * cols) cols;
      dot_congr w1 (o1 + r * cols) w2 (o2 + r * cols) x cols
    in
    Classical.forall_intro aux;
    S.lemma_eq_intro (matvec w1 o1 rows cols x) (matvec w2 o2 rows cols x)

let rms_congr (eps:real) (x:S.seq real) (n:nat) (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int)
  : Lemma (requires wagree w1 o1 w2 o2 n)
          (ensures rmsnorm1p eps x n w1 o1 == rmsnorm1p eps x n w2 o2)
  = S.lemma_eq_intro (rmsnorm1p eps x n w1 o1) (rmsnorm1p eps x n w2 o2)

let head_norm_congr (eps:real) (raw:S.seq real) (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int)
  : Lemma (requires wagree w1 o1 w2 o2 256)
          (ensures forall (o:int). head_norm eps raw o w1 o1 == head_norm eps raw o w2 o2)
  = let aux (o:int) : Lemma (head_norm eps raw o w1 o1 == head_norm eps raw o w2 o2) =
      S.lemma_eq_intro (head_norm eps raw o w1 o1) (head_norm eps raw o w2 o2) in
    Classical.forall_intro aux

(*** MLP (footprint 11012096) *)

let mlp_len : int = 11012096

let mlp_act_congr (nm:numerics) (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (x1:S.seq real)
  : Lemma (requires wagree w1 o1 w2 o2 mlp_len)
          (ensures mlp_in nm w1 o1 x1 == mlp_in nm w2 o2 x1 /\ mlp_act nm w1 o1 x1 == mlp_act nm w2 o2 x1)
  = wagree_sub w1 o1 w2 o2 mlp_len 1024 1024;
    rms_congr nm.eps x1 1024 w1 (o1 + 1024) w2 (o2 + 1024);
    let h = mlp_in nm w1 o1 x1 in
    wagree_sub w1 o1 w2 o2 mlp_len 2048 (3584 * 1024);
    matvec_congr w1 (o1 + 2048) w2 (o2 + 2048) 3584 1024 h;
    wagree_sub w1 o1 w2 o2 mlp_len 3672064 (3584 * 1024);
    matvec_congr w1 (o1 + 3672064) w2 (o2 + 3672064) 3584 1024 h;
    S.lemma_eq_intro (mlp_act nm w1 o1 x1) (mlp_act nm w2 o2 x1)

let mlp_out_congr (nm:numerics) (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (x1:S.seq real)
  : Lemma (requires wagree w1 o1 w2 o2 mlp_len)
          (ensures mlp_out nm w1 o1 x1 == mlp_out nm w2 o2 x1)
  = mlp_act_congr nm w1 o1 w2 o2 x1;
    wagree_sub w1 o1 w2 o2 mlp_len 7342080 (1024 * 3584);
    matvec_congr w1 (o1 + 7342080) w2 (o2 + 7342080) 1024 3584 (mlp_act nm w1 o1 x1)

(*** Gated DeltaNet layer (footprint dn_len) *)

let dn_pre_congr (nm:numerics) (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (x conv:S.seq real)
  : Lemma (requires wagree w1 o1 w2 o2 dn_len)
          (ensures dn_hin nm w1 o1 x == dn_hin nm w2 o2 x /\
                   dn_qkv nm w1 o1 x == dn_qkv nm w2 o2 x /\
                   dn_conv nm w1 o1 x conv == dn_conv nm w2 o2 x conv /\
                   dn_conv_out nm w1 o1 x conv == dn_conv_out nm w2 o2 x conv)
  = rms_congr nm.eps x 1024 w1 o1 w2 o2;
    wagree_sub w1 o1 w2 o2 dn_len 11012096 (6144 * 1024);
    matvec_congr w1 (o1 + 11012096) w2 (o2 + 11012096) 6144 1024 (dn_hin nm w1 o1 x);
    S.lemma_eq_intro (dn_conv nm w1 o1 x conv) (dn_conv nm w2 o2 x conv);
    S.lemma_eq_intro (dn_conv_out nm w1 o1 x conv) (dn_conv_out nm w2 o2 x conv)

let dn_qkv_congr (nm:numerics) (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (x conv:S.seq real)
  : Lemma (requires wagree w1 o1 w2 o2 dn_len)
          (ensures dn_q nm w1 o1 x conv == dn_q nm w2 o2 x conv /\
                   dn_k nm w1 o1 x conv == dn_k nm w2 o2 x conv /\
                   dn_v nm w1 o1 x conv == dn_v nm w2 o2 x conv)
  = dn_pre_congr nm w1 o1 w2 o2 x conv;
    S.lemma_eq_intro (dn_q nm w1 o1 x conv) (dn_q nm w2 o2 x conv);
    S.lemma_eq_intro (dn_k nm w1 o1 x conv) (dn_k nm w2 o2 x conv);
    S.lemma_eq_intro (dn_v nm w1 o1 x conv) (dn_v nm w2 o2 x conv)

let dn_gate_congr (nm:numerics) (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (x:S.seq real) (h:int)
  : Lemma (requires wagree w1 o1 w2 o2 dn_len /\ 0 <= h /\ h < 16)
          (ensures dn_beta nm w1 o1 x h == dn_beta nm w2 o2 x h /\
                   dn_decay nm w1 o1 x h == dn_decay nm w2 o2 x h)
  = rms_congr nm.eps x 1024 w1 o1 w2 o2;
    let hn = dn_hin nm w1 o1 x in
    wagree_sub w1 o1 w2 o2 dn_len (11012096 + 8388608 + h * 1024) 1024;
    dot_congr w1 (dn_mix o1 + 8388608 + h * 1024) w2 (dn_mix o2 + 8388608 + h * 1024) hn 1024;
    wagree_sub w1 o1 w2 o2 dn_len (11012096 + 8404992 + h * 1024) 1024;
    dot_congr w1 (dn_mix o1 + 8404992 + h * 1024) w2 (dn_mix o2 + 8404992 + h * 1024) hn 1024

let dn_delta_congr (nm:numerics) (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (x conv st:S.seq real)
  : Lemma (requires wagree w1 o1 w2 o2 dn_len)
          (ensures forall (h j:int). 0 <= h /\ h < 16 ==>
                     dn_delta nm w1 o1 x conv st h j == dn_delta nm w2 o2 x conv st h j)
  = dn_qkv_congr nm w1 o1 w2 o2 x conv;
    let aux (h j:int) : Lemma (0 <= h /\ h < 16 ==>
                     dn_delta nm w1 o1 x conv st h j == dn_delta nm w2 o2 x conv st h j) =
      if 0 <= h && h < 16 then dn_gate_congr nm w1 o1 w2 o2 x h in
    Classical.forall_intro_2 aux

let dn_state_congr (nm:numerics) (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (x conv st:S.seq real)
  : Lemma (requires wagree w1 o1 w2 o2 dn_len)
          (ensures dn_state nm w1 o1 x conv st == dn_state nm w2 o2 x conv st)
  = dn_qkv_congr nm w1 o1 w2 o2 x conv;
    dn_delta_congr nm w1 o1 w2 o2 x conv st;
    let aux (idx:nat{idx < 262144})
      : Lemma (S.index (dn_state nm w1 o1 x conv st) idx == S.index (dn_state nm w2 o2 x conv st) idx) =
      ML.lemma_div_le idx 262143 16384;
      dn_gate_congr nm w1 o1 w2 o2 x (idx / 16384) in
    Classical.forall_intro aux;
    S.lemma_eq_intro (dn_state nm w1 o1 x conv st) (dn_state nm w2 o2 x conv st)

let dn_out_congr (nm:numerics) (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (x conv st:S.seq real)
  : Lemma (requires wagree w1 o1 w2 o2 dn_len)
          (ensures dn_out nm w1 o1 x conv st == dn_out nm w2 o2 x conv st)
  = dn_pre_congr nm w1 o1 w2 o2 x conv;
    dn_qkv_congr nm w1 o1 w2 o2 x conv;
    dn_state_congr nm w1 o1 w2 o2 x conv st;
    S.lemma_eq_intro (dn_core nm w1 o1 x conv st) (dn_core nm w2 o2 x conv st);
    wagree_sub w1 o1 w2 o2 dn_len (11012096 + 6291456) (2048 * 1024);
    matvec_congr w1 (dn_mix o1 + 6291456) w2 (dn_mix o2 + 6291456) 2048 1024 (dn_hin nm w1 o1 x);
    S.lemma_eq_intro (dn_gated nm w1 o1 x conv st) (dn_gated nm w2 o2 x conv st);
    wagree_sub w1 o1 w2 o2 dn_len (11012096 + 8446112) (1024 * 2048);
    matvec_congr w1 (dn_mix o1 + 8446112) w2 (dn_mix o2 + 8446112) 1024 2048 (dn_gated nm w1 o1 x conv st);
    assert (dn_x1 nm w1 o1 x conv st == dn_x1 nm w2 o2 x conv st);
    wagree_weaken w1 o1 w2 o2 dn_len mlp_len;
    mlp_out_congr nm w1 o1 w2 o2 (dn_x1 nm w1 o1 x conv st)

(*** Gated full attention (footprint attn_len) and KV-history agreement *)

(* two histories agree on all positions p < t and components j < 512 *)
let hagree (h1 h2:nat -> nat -> real) (t:nat) : prop =
  forall (p j:nat). {:pattern (h1 p j)} p < t /\ j < 512 ==> h1 p j == h2 p j

let at_pre_congr (nm:numerics) (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (x cs sn:S.seq real)
  : Lemma (requires wagree w1 o1 w2 o2 attn_len)
          (ensures at_qraw nm w1 o1 x == at_qraw nm w2 o2 x /\
                   at_q nm w1 o1 x cs sn == at_q nm w2 o2 x cs sn /\
                   at_k nm w1 o1 x cs sn == at_k nm w2 o2 x cs sn /\
                   at_v nm w1 o1 x == at_v nm w2 o2 x)
  = rms_congr nm.eps x 1024 w1 o1 w2 o2;
    let hn = at_hin nm w1 o1 x in
    wagree_sub w1 o1 w2 o2 attn_len 11012096 (4096 * 1024);
    matvec_congr w1 (at_mix o1) w2 (at_mix o2) 4096 1024 hn;
    wagree_sub w1 o1 w2 o2 attn_len (11012096 + 4194304) (512 * 1024);
    matvec_congr w1 (at_mix o1 + 4194304) w2 (at_mix o2 + 4194304) 512 1024 hn;
    wagree_sub w1 o1 w2 o2 attn_len (11012096 + 4718592) (512 * 1024);
    matvec_congr w1 (at_mix o1 + 4718592) w2 (at_mix o2 + 4718592) 512 1024 hn;
    wagree_sub w1 o1 w2 o2 attn_len (11012096 + 5242880) 256;
    head_norm_congr nm.eps (at_qraw nm w1 o1 x) w1 (at_mix o1 + 5242880) w2 (at_mix o2 + 5242880);
    wagree_sub w1 o1 w2 o2 attn_len (11012096 + 5243136) 256;
    head_norm_congr nm.eps (at_kraw nm w1 o1 x) w1 (at_mix o1 + 5243136) w2 (at_mix o2 + 5243136);
    S.lemma_eq_intro (at_q nm w1 o1 x cs sn) (at_q nm w2 o2 x cs sn);
    S.lemma_eq_intro (at_k nm w1 o1 x cs sn) (at_k nm w2 o2 x cs sn)

let at_score_congr (nm:numerics) (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (x cs sn:S.seq real)
                   (t:nat) (hk1 hk2:nat -> nat -> real)
  : Lemma (requires wagree w1 o1 w2 o2 attn_len /\ hagree hk1 hk2 t)
          (ensures forall (hd p:nat). hd < 8 /\ p <= t ==>
             at_score nm w1 o1 x cs sn t hk1 hd p == at_score nm w2 o2 x cs sn t hk2 hd p)
  = at_pre_congr nm w1 o1 w2 o2 x cs sn

let at_prob_congr (nm:numerics) (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (x cs sn:S.seq real)
                  (t:nat) (hk1 hk2:nat -> nat -> real)
  : Lemma (requires wagree w1 o1 w2 o2 attn_len /\ hagree hk1 hk2 t)
          (ensures forall (hd p:nat). hd < 8 /\ p <= t ==>
             at_prob nm w1 o1 x cs sn t hk1 hd p == at_prob nm w2 o2 x cs sn t hk2 hd p)
  = at_score_congr nm w1 o1 w2 o2 x cs sn t hk1 hk2

let at_out_congr (nm:numerics) (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (x cs sn:S.seq real)
                 (t:nat) (hk1 hk2 hv1 hv2:nat -> nat -> real)
  : Lemma (requires wagree w1 o1 w2 o2 attn_len /\ hagree hk1 hk2 t /\ hagree hv1 hv2 t)
          (ensures at_out nm w1 o1 x cs sn t hk1 hv1 == at_out nm w2 o2 x cs sn t hk2 hv2)
  = at_pre_congr nm w1 o1 w2 o2 x cs sn;
    at_prob_congr nm w1 o1 w2 o2 x cs sn t hk1 hk2;
    let aux (c:nat{c < 2048})
      : Lemma (S.index (at_gated nm w1 o1 x cs sn t hk1 hv1) c == S.index (at_gated nm w2 o2 x cs sn t hk2 hv2) c) =
      ML.lemma_div_le c 2047 256;
      let hd = c / 256 in
      let j = (hd / 4) * 256 + c % 256 in
      assert (hd < 8 /\ j < 512);
      let vn = at_v nm w1 o1 x in
      let hp (p:nat) : Lemma (p <= t ==> hist_or hv1 vn t p j == hist_or hv2 vn t p j) = () in
      Classical.forall_intro hp in
    Classical.forall_intro aux;
    S.lemma_eq_intro (at_gated nm w1 o1 x cs sn t hk1 hv1) (at_gated nm w2 o2 x cs sn t hk2 hv2);
    wagree_sub w1 o1 w2 o2 attn_len (11012096 + 5243392) (1024 * 2048);
    matvec_congr w1 (at_mix o1 + 5243392) w2 (at_mix o2 + 5243392) 1024 2048 (at_gated nm w1 o1 x cs sn t hk1 hv1);
    assert (at_x1 nm w1 o1 x cs sn t hk1 hv1 == at_x1 nm w2 o2 x cs sn t hk2 hv2);
    wagree_weaken w1 o1 w2 o2 attn_len mlp_len;
    mlp_out_congr nm w1 o1 w2 o2 (at_x1 nm w1 o1 x cs sn t hk1 hv1)

(*** Final RMS layer and LM head *)

let logits_congr (nm:numerics) (t1:S.seq real) (to1:int) (t2:S.seq real) (to2:int)
                 (n1:S.seq real) (no1:int) (n2:S.seq real) (no2:int) (h:S.seq real)
  : Lemma (requires wagree t1 to1 t2 to2 table_len /\ wagree n1 no1 n2 no2 1024)
          (ensures logits nm t1 to1 n1 no1 h == logits nm t2 to2 n2 no2 h)
  = rms_congr nm.eps h 1024 n1 no1 n2 no2;
    let hn = rmsnorm1p nm.eps h 1024 n1 no1 in
    let aux (m:nat{m < 248320}) : Lemma (dot t1 (to1 + m * 1024) hn 1024 == dot t2 (to2 + m * 1024) hn 1024) =
      wagree_sub t1 to1 t2 to2 table_len (m * 1024) 1024;
      dot_congr t1 (to1 + m * 1024) t2 (to2 + m * 1024) hn 1024 in
    Classical.forall_intro aux;
    S.lemma_eq_intro (logits nm t1 to1 n1 no1 h) (logits nm t2 to2 n2 no2 h)
