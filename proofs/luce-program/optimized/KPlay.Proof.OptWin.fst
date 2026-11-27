module KPlay.Proof.OptWin

(* Weight-window extensionality: the reference layer functions read the
   weights only inside their packed window, so a layer evaluated on the
   sliced view at base 0 equals the layer evaluated on the full weights at
   the layer offset. Also KV-history extensionality for attention. *)

open FStar.Real
open KPlay.SharedSpec
module S = FStar.Seq
module M = FStar.Math.Lemmas
module Exp = FStar.Math.Exp

#set-options "--fuel 0 --ifuel 0 --z3rlimit 50"

[@@"opaque_to_smt"]
let wwin (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (n:nat) : prop =
  forall (k:nat). k < n ==> at w1 (o1 + k) == at w2 (o2 + k)

let wwin_at (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (n:nat) (k:nat)
  : Lemma (requires wwin w1 o1 w2 o2 n /\ k < n) (ensures at w1 (o1 + k) == at w2 (o2 + k))
  = reveal_opaque (`%wwin) (wwin w1 o1 w2 o2 n)

let wwin_intro (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (n:nat)
  : Lemma (requires forall (k:nat). k < n ==> at w1 (o1 + k) == at w2 (o2 + k))
          (ensures wwin w1 o1 w2 o2 n)
  = reveal_opaque (`%wwin) (wwin w1 o1 w2 o2 n)

let wwin_sub (w1:S.seq real) (o1:int) (w2:S.seq real) (o2:int) (n:nat) (a1 a2:int) (m:nat)
  : Lemma (requires wwin w1 o1 w2 o2 n /\ a1 - o1 == a2 - o2 /\ 0 <= a1 - o1 /\ a1 - o1 + m <= n)
          (ensures wwin w1 a1 w2 a2 m)
  = introduce forall (k:nat). k < m ==> at w1 (a1 + k) == at w2 (a2 + k)
    with introduce _ ==> _ with wwin_at w1 o1 w2 o2 n (a1 - o1 + k);
    wwin_intro w1 a1 w2 a2 m

let slice_win (rw:S.seq real) (off len:nat)
  : Lemma (requires off + len <= S.length rw)
          (ensures wwin (S.slice rw off (off + len)) 0 rw off len)
  = wwin_intro (S.slice rw off (off + len)) 0 rw off len

(*** Base lemmas *)

let vsum_ext (n:nat) (f g:nat -> real)
  : Lemma (requires forall (k:nat). k < n ==> f k == g k) (ensures vsum n f == vsum n g)
  = S.lemma_eq_intro (S.init n (fun (k:nat{k < n}) -> f k)) (S.init n (fun (k:nat{k < n}) -> g k))

let dot_ext (w1:S.seq real) (a1:int) (w2:S.seq real) (a2:int) (x:S.seq real) (n:nat)
  : Lemma (requires wwin w1 a1 w2 a2 n) (ensures dot w1 a1 x n == dot w2 a2 x n)
  = introduce forall (k:nat). k < n ==> at w1 (a1 + k) *. at x k == at w2 (a2 + k) *. at x k
    with introduce _ ==> _ with wwin_at w1 a1 w2 a2 n k;
    vsum_ext n (fun k -> at w1 (a1 + k) *. at x k) (fun k -> at w2 (a2 + k) *. at x k)

let mul_bound (r rows cols:nat)
  : Lemma (requires r < rows) (ensures r * cols + cols <= rows * cols)
  = M.lemma_mult_le_right cols (r + 1) rows;
    M.distributivity_add_left r 1 cols

let matvec_ext (w1:S.seq real) (a1:int) (w2:S.seq real) (a2:int) (rows cols:nat) (x:S.seq real)
  : Lemma (requires wwin w1 a1 w2 a2 (rows * cols))
          (ensures matvec w1 a1 rows cols x == matvec w2 a2 rows cols x)
  = introduce forall (r:nat). r < rows ==> dot w1 (a1 + r * cols) x cols == dot w2 (a2 + r * cols) x cols
    with introduce _ ==> _ with begin
      mul_bound r rows cols;
      wwin_sub w1 a1 w2 a2 (rows * cols) (a1 + r * cols) (a2 + r * cols) cols;
      dot_ext w1 (a1 + r * cols) w2 (a2 + r * cols) x cols
    end;
    S.lemma_eq_intro (matvec w1 a1 rows cols x) (matvec w2 a2 rows cols x)

let rmsnorm1p_ext (eps:real) (x:S.seq real) (n:nat) (w1:S.seq real) (a1:int) (w2:S.seq real) (a2:int)
  : Lemma (requires wwin w1 a1 w2 a2 n)
          (ensures rmsnorm1p eps x n w1 a1 == rmsnorm1p eps x n w2 a2)
  = introduce forall (i:nat). i < n ==> at w1 (a1 + i) == at w2 (a2 + i)
    with introduce _ ==> _ with wwin_at w1 a1 w2 a2 n i;
    S.lemma_eq_intro (rmsnorm1p eps x n w1 a1) (rmsnorm1p eps x n w2 a2)

(*** MLP (window [wo, wo + 11012096)) *)

let mlp_ext (nm:numerics) (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) (x1:S.seq real)
  : Lemma (requires wwin w1 wo1 w2 wo2 11012096)
          (ensures mlp_out nm w1 wo1 x1 == mlp_out nm w2 wo2 x1)
  = wwin_sub w1 wo1 w2 wo2 11012096 (wo1 + 1024) (wo2 + 1024) 1024;
    rmsnorm1p_ext nm.eps x1 1024 w1 (wo1 + 1024) w2 (wo2 + 1024);
    let h = mlp_in nm w1 wo1 x1 in
    wwin_sub w1 wo1 w2 wo2 11012096 (wo1 + 2048) (wo2 + 2048) (3584 * 1024);
    matvec_ext w1 (wo1 + 2048) w2 (wo2 + 2048) 3584 1024 h;
    wwin_sub w1 wo1 w2 wo2 11012096 (wo1 + 3672064) (wo2 + 3672064) (3584 * 1024);
    matvec_ext w1 (wo1 + 3672064) w2 (wo2 + 3672064) 3584 1024 h;
    S.lemma_eq_intro (mlp_act nm w1 wo1 x1) (mlp_act nm w2 wo2 x1);
    wwin_sub w1 wo1 w2 wo2 11012096 (wo1 + 7342080) (wo2 + 7342080) (1024 * 3584);
    matvec_ext w1 (wo1 + 7342080) w2 (wo2 + 7342080) 1024 3584 (mlp_act nm w1 wo1 x1)

(*** Gated DeltaNet layer (window [wo, wo + dn_len)) *)

let dwin (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) : prop = wwin w1 wo1 w2 wo2 21555360

let dn_hin_ext (nm:numerics) (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) (x:S.seq real)
  : Lemma (requires dwin w1 wo1 w2 wo2) (ensures dn_hin nm w1 wo1 x == dn_hin nm w2 wo2 x)
  = wwin_sub w1 wo1 w2 wo2 21555360 wo1 wo2 1024;
    rmsnorm1p_ext nm.eps x 1024 w1 wo1 w2 wo2

let dn_qkv_ext (nm:numerics) (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) (x:S.seq real)
  : Lemma (requires dwin w1 wo1 w2 wo2) (ensures dn_qkv nm w1 wo1 x == dn_qkv nm w2 wo2 x)
  = dn_hin_ext nm w1 wo1 w2 wo2 x;
    wwin_sub w1 wo1 w2 wo2 21555360 (dn_mix wo1) (dn_mix wo2) (6144 * 1024);
    matvec_ext w1 (dn_mix wo1) w2 (dn_mix wo2) 6144 1024 (dn_hin nm w1 wo1 x)

let dn_conv_ext (nm:numerics) (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) (x conv:S.seq real)
  : Lemma (requires dwin w1 wo1 w2 wo2) (ensures dn_conv nm w1 wo1 x conv == dn_conv nm w2 wo2 x conv)
  = dn_qkv_ext nm w1 wo1 w2 wo2 x;
    S.lemma_eq_intro (dn_conv nm w1 wo1 x conv) (dn_conv nm w2 wo2 x conv)

let dn_conv_out_ext (nm:numerics) (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) (x conv:S.seq real)
  : Lemma (requires dwin w1 wo1 w2 wo2)
          (ensures dn_conv_out nm w1 wo1 x conv == dn_conv_out nm w2 wo2 x conv)
  = dn_conv_ext nm w1 wo1 w2 wo2 x conv;
    introduce forall (c:nat). c < 6144 ==>
      vsum 4 (fun k -> at (dn_conv nm w1 wo1 x conv) (c * 4 + k) *. at w1 (dn_mix wo1 + 8421376 + c * 4 + k)) ==
      vsum 4 (fun k -> at (dn_conv nm w2 wo2 x conv) (c * 4 + k) *. at w2 (dn_mix wo2 + 8421376 + c * 4 + k))
    with introduce _ ==> _ with begin
      introduce forall (k:nat). k < 4 ==>
        at (dn_conv nm w1 wo1 x conv) (c * 4 + k) *. at w1 (dn_mix wo1 + 8421376 + c * 4 + k) ==
        at (dn_conv nm w2 wo2 x conv) (c * 4 + k) *. at w2 (dn_mix wo2 + 8421376 + c * 4 + k)
      with introduce _ ==> _ with wwin_at w1 wo1 w2 wo2 21555360 (11012096 + 8421376 + c * 4 + k);
      vsum_ext 4 (fun k -> at (dn_conv nm w1 wo1 x conv) (c * 4 + k) *. at w1 (dn_mix wo1 + 8421376 + c * 4 + k))
                 (fun k -> at (dn_conv nm w2 wo2 x conv) (c * 4 + k) *. at w2 (dn_mix wo2 + 8421376 + c * 4 + k))
    end;
    S.lemma_eq_intro (dn_conv_out nm w1 wo1 x conv) (dn_conv_out nm w2 wo2 x conv)

let dn_qkv3_ext (nm:numerics) (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) (x conv:S.seq real)
  : Lemma (requires dwin w1 wo1 w2 wo2)
          (ensures dn_q nm w1 wo1 x conv == dn_q nm w2 wo2 x conv /\
                   dn_k nm w1 wo1 x conv == dn_k nm w2 wo2 x conv /\
                   dn_v nm w1 wo1 x conv == dn_v nm w2 wo2 x conv)
  = dn_conv_out_ext nm w1 wo1 w2 wo2 x conv;
    S.lemma_eq_intro (dn_q nm w1 wo1 x conv) (dn_q nm w2 wo2 x conv);
    S.lemma_eq_intro (dn_k nm w1 wo1 x conv) (dn_k nm w2 wo2 x conv);
    S.lemma_eq_intro (dn_v nm w1 wo1 x conv) (dn_v nm w2 wo2 x conv)

let dn_beta_ext (nm:numerics) (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) (x:S.seq real) (h:nat)
  : Lemma (requires dwin w1 wo1 w2 wo2 /\ h < 16)
          (ensures dn_beta nm w1 wo1 x h == dn_beta nm w2 wo2 x h)
  = dn_hin_ext nm w1 wo1 w2 wo2 x;
    wwin_sub w1 wo1 w2 wo2 21555360 (dn_mix wo1 + 8388608 + h * 1024) (dn_mix wo2 + 8388608 + h * 1024) 1024;
    dot_ext w1 (dn_mix wo1 + 8388608 + h * 1024) w2 (dn_mix wo2 + 8388608 + h * 1024) (dn_hin nm w1 wo1 x) 1024

let dn_decay_ext (nm:numerics) (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) (x:S.seq real) (h:nat)
  : Lemma (requires dwin w1 wo1 w2 wo2 /\ h < 16)
          (ensures dn_decay nm w1 wo1 x h == dn_decay nm w2 wo2 x h)
  = dn_hin_ext nm w1 wo1 w2 wo2 x;
    wwin_sub w1 wo1 w2 wo2 21555360 (dn_mix wo1 + 8404992 + h * 1024) (dn_mix wo2 + 8404992 + h * 1024) 1024;
    dot_ext w1 (dn_mix wo1 + 8404992 + h * 1024) w2 (dn_mix wo2 + 8404992 + h * 1024) (dn_hin nm w1 wo1 x) 1024;
    wwin_at w1 wo1 w2 wo2 21555360 (11012096 + 8445968 + h);
    wwin_at w1 wo1 w2 wo2 21555360 (11012096 + 8445952 + h)

let dn_delta_ext (nm:numerics) (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) (x conv st:S.seq real) (h:nat) (j:int)
  : Lemma (requires dwin w1 wo1 w2 wo2 /\ h < 16)
          (ensures dn_delta nm w1 wo1 x conv st h j == dn_delta nm w2 wo2 x conv st h j)
  = dn_qkv3_ext nm w1 wo1 w2 wo2 x conv;
    dn_decay_ext nm w1 wo1 w2 wo2 x h;
    dn_beta_ext nm w1 wo1 w2 wo2 x h;
    vsum_ext 128 (fun i -> (at st (h * 16384 + i * 128 + j) *. dn_decay nm w1 wo1 x h) *. at (dn_k nm w1 wo1 x conv) (h * 128 + i))
                 (fun i -> (at st (h * 16384 + i * 128 + j) *. dn_decay nm w2 wo2 x h) *. at (dn_k nm w2 wo2 x conv) (h * 128 + i))

let dn_state_ext (nm:numerics) (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) (x conv st:S.seq real)
  : Lemma (requires dwin w1 wo1 w2 wo2)
          (ensures dn_state nm w1 wo1 x conv st == dn_state nm w2 wo2 x conv st)
  = dn_qkv3_ext nm w1 wo1 w2 wo2 x conv;
    introduce forall (idx:nat). idx < 262144 ==>
      dn_decay nm w1 wo1 x (idx / 16384) == dn_decay nm w2 wo2 x (idx / 16384) /\
      dn_delta nm w1 wo1 x conv st (idx / 16384) (idx % 128) == dn_delta nm w2 wo2 x conv st (idx / 16384) (idx % 128)
    with introduce _ ==> _ with begin
      dn_decay_ext nm w1 wo1 w2 wo2 x (idx / 16384);
      dn_delta_ext nm w1 wo1 w2 wo2 x conv st (idx / 16384) (idx % 128)
    end;
    S.lemma_eq_intro (dn_state nm w1 wo1 x conv st) (dn_state nm w2 wo2 x conv st)

let dn_core_ext (nm:numerics) (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) (x conv st:S.seq real)
  : Lemma (requires dwin w1 wo1 w2 wo2)
          (ensures dn_core nm w1 wo1 x conv st == dn_core nm w2 wo2 x conv st)
  = dn_qkv3_ext nm w1 wo1 w2 wo2 x conv;
    dn_state_ext nm w1 wo1 w2 wo2 x conv st;
    introduce forall (c:nat). c < 2048 ==>
      vsum 128 (fun i -> at (dn_state nm w1 wo1 x conv st) ((c / 128) * 16384 + i * 128 + c % 128) *. at (dn_q nm w1 wo1 x conv) ((c / 128) * 128 + i)) ==
      vsum 128 (fun i -> at (dn_state nm w2 wo2 x conv st) ((c / 128) * 16384 + i * 128 + c % 128) *. at (dn_q nm w2 wo2 x conv) ((c / 128) * 128 + i))
    with introduce _ ==> _ with
      vsum_ext 128 (fun i -> at (dn_state nm w1 wo1 x conv st) ((c / 128) * 16384 + i * 128 + c % 128) *. at (dn_q nm w1 wo1 x conv) ((c / 128) * 128 + i))
                   (fun i -> at (dn_state nm w2 wo2 x conv st) ((c / 128) * 16384 + i * 128 + c % 128) *. at (dn_q nm w2 wo2 x conv) ((c / 128) * 128 + i));
    S.lemma_eq_intro (dn_core nm w1 wo1 x conv st) (dn_core nm w2 wo2 x conv st)

let dn_gated_ext (nm:numerics) (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) (x conv st:S.seq real)
  : Lemma (requires dwin w1 wo1 w2 wo2)
          (ensures dn_gated nm w1 wo1 x conv st == dn_gated nm w2 wo2 x conv st)
  = dn_core_ext nm w1 wo1 w2 wo2 x conv st;
    dn_hin_ext nm w1 wo1 w2 wo2 x;
    wwin_sub w1 wo1 w2 wo2 21555360 (dn_mix wo1 + 6291456) (dn_mix wo2 + 6291456) (2048 * 1024);
    matvec_ext w1 (dn_mix wo1 + 6291456) w2 (dn_mix wo2 + 6291456) 2048 1024 (dn_hin nm w1 wo1 x);
    introduce forall (c:nat). c < 2048 ==>
      at w1 (dn_mix wo1 + 8445984 + c % 128) == at w2 (dn_mix wo2 + 8445984 + c % 128)
    with introduce _ ==> _ with wwin_at w1 wo1 w2 wo2 21555360 (11012096 + 8445984 + c % 128);
    S.lemma_eq_intro (dn_gated nm w1 wo1 x conv st) (dn_gated nm w2 wo2 x conv st)

let dn_out_ext (nm:numerics) (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) (x conv st:S.seq real)
  : Lemma (requires dwin w1 wo1 w2 wo2)
          (ensures dn_out nm w1 wo1 x conv st == dn_out nm w2 wo2 x conv st)
  = dn_gated_ext nm w1 wo1 w2 wo2 x conv st;
    wwin_sub w1 wo1 w2 wo2 21555360 (dn_mix wo1 + 8446112) (dn_mix wo2 + 8446112) (1024 * 2048);
    matvec_ext w1 (dn_mix wo1 + 8446112) w2 (dn_mix wo2 + 8446112) 1024 2048 (dn_gated nm w1 wo1 x conv st);
    assert (dn_x1 nm w1 wo1 x conv st == dn_x1 nm w2 wo2 x conv st);
    wwin_sub w1 wo1 w2 wo2 21555360 wo1 wo2 11012096;
    mlp_ext nm w1 wo1 w2 wo2 (dn_x1 nm w1 wo1 x conv st)

(*** Gated full-attention layer (window [wo, wo + attn_len)) and KV history *)

let awin (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) : prop = wwin w1 wo1 w2 wo2 18352640

(* two histories agree on the positions and components attention reads *)
let hagree (t:nat) (h1 h2:nat -> nat -> real) : prop =
  forall (p j:nat). p < t /\ j < 512 ==> h1 p j == h2 p j

let at_proj_ext (nm:numerics) (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) (x:S.seq real)
  : Lemma (requires awin w1 wo1 w2 wo2)
          (ensures at_qraw nm w1 wo1 x == at_qraw nm w2 wo2 x /\
                   at_kraw nm w1 wo1 x == at_kraw nm w2 wo2 x /\
                   at_v nm w1 wo1 x == at_v nm w2 wo2 x)
  = wwin_sub w1 wo1 w2 wo2 18352640 wo1 wo2 1024;
    rmsnorm1p_ext nm.eps x 1024 w1 wo1 w2 wo2;
    assert (at_hin nm w1 wo1 x == at_hin nm w2 wo2 x);
    let h = at_hin nm w1 wo1 x in
    wwin_sub w1 wo1 w2 wo2 18352640 (at_mix wo1) (at_mix wo2) (4096 * 1024);
    matvec_ext w1 (at_mix wo1) w2 (at_mix wo2) 4096 1024 h;
    wwin_sub w1 wo1 w2 wo2 18352640 (at_mix wo1 + 4194304) (at_mix wo2 + 4194304) (512 * 1024);
    matvec_ext w1 (at_mix wo1 + 4194304) w2 (at_mix wo2 + 4194304) 512 1024 h;
    wwin_sub w1 wo1 w2 wo2 18352640 (at_mix wo1 + 4718592) (at_mix wo2 + 4718592) (512 * 1024);
    matvec_ext w1 (at_mix wo1 + 4718592) w2 (at_mix wo2 + 4718592) 512 1024 h

let head_norm_ext (eps:real) (raw:S.seq real) (o:int) (w1:S.seq real) (a1:int) (w2:S.seq real) (a2:int)
  : Lemma (requires wwin w1 a1 w2 a2 256)
          (ensures head_norm eps raw o w1 a1 == head_norm eps raw o w2 a2)
  = introduce forall (i:nat). i < 256 ==> at w1 (a1 + i) == at w2 (a2 + i)
    with introduce _ ==> _ with wwin_at w1 a1 w2 a2 256 i;
    S.lemma_eq_intro (head_norm eps raw o w1 a1) (head_norm eps raw o w2 a2)

let at_qk_ext (nm:numerics) (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) (x cs sn:S.seq real)
  : Lemma (requires awin w1 wo1 w2 wo2)
          (ensures at_q nm w1 wo1 x cs sn == at_q nm w2 wo2 x cs sn /\
                   at_k nm w1 wo1 x cs sn == at_k nm w2 wo2 x cs sn)
  = at_proj_ext nm w1 wo1 w2 wo2 x;
    wwin_sub w1 wo1 w2 wo2 18352640 (at_mix wo1 + 5242880) (at_mix wo2 + 5242880) 256;
    wwin_sub w1 wo1 w2 wo2 18352640 (at_mix wo1 + 5243136) (at_mix wo2 + 5243136) 256;
    introduce forall (c:nat). c < 2048 ==>
      head_norm nm.eps (at_qraw nm w1 wo1 x) ((c / 256) * 512) w1 (at_mix wo1 + 5242880) ==
      head_norm nm.eps (at_qraw nm w1 wo1 x) ((c / 256) * 512) w2 (at_mix wo2 + 5242880)
    with introduce _ ==> _ with
      head_norm_ext nm.eps (at_qraw nm w1 wo1 x) ((c / 256) * 512) w1 (at_mix wo1 + 5242880) w2 (at_mix wo2 + 5242880);
    S.lemma_eq_intro (at_q nm w1 wo1 x cs sn) (at_q nm w2 wo2 x cs sn);
    introduce forall (c:nat). c < 512 ==>
      head_norm nm.eps (at_kraw nm w1 wo1 x) ((c / 256) * 256) w1 (at_mix wo1 + 5243136) ==
      head_norm nm.eps (at_kraw nm w1 wo1 x) ((c / 256) * 256) w2 (at_mix wo2 + 5243136)
    with introduce _ ==> _ with
      head_norm_ext nm.eps (at_kraw nm w1 wo1 x) ((c / 256) * 256) w1 (at_mix wo1 + 5243136) w2 (at_mix wo2 + 5243136);
    S.lemma_eq_intro (at_k nm w1 wo1 x cs sn) (at_k nm w2 wo2 x cs sn)

let at_score_ext (nm:numerics) (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) (x cs sn:S.seq real)
                 (t:nat) (hk1 hk2:nat -> nat -> real) (hd p:nat)
  : Lemma (requires awin w1 wo1 w2 wo2 /\ hagree t hk1 hk2 /\ hd < 8)
          (ensures at_score nm w1 wo1 x cs sn t hk1 hd p == at_score nm w2 wo2 x cs sn t hk2 hd p)
  = at_qk_ext nm w1 wo1 w2 wo2 x cs sn;
    vsum_ext 256 (fun i -> at (at_q nm w1 wo1 x cs sn) (hd * 256 + i) *. hist_or hk1 (at_k nm w1 wo1 x cs sn) t p ((hd / 4) * 256 + i))
                 (fun i -> at (at_q nm w2 wo2 x cs sn) (hd * 256 + i) *. hist_or hk2 (at_k nm w2 wo2 x cs sn) t p ((hd / 4) * 256 + i))

let at_prob_ext (nm:numerics) (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) (x cs sn:S.seq real)
                (t:nat) (hk1 hk2:nat -> nat -> real) (hd p:nat)
  : Lemma (requires awin w1 wo1 w2 wo2 /\ hagree t hk1 hk2 /\ hd < 8)
          (ensures at_prob nm w1 wo1 x cs sn t hk1 hd p == at_prob nm w2 wo2 x cs sn t hk2 hd p)
  = at_score_ext nm w1 wo1 w2 wo2 x cs sn t hk1 hk2 hd p;
    introduce forall (p':nat). p' < t + 1 ==>
      Exp.exp (at_score nm w1 wo1 x cs sn t hk1 hd p') == Exp.exp (at_score nm w2 wo2 x cs sn t hk2 hd p')
    with introduce _ ==> _ with at_score_ext nm w1 wo1 w2 wo2 x cs sn t hk1 hk2 hd p';
    vsum_ext (t + 1) (fun p' -> Exp.exp (at_score nm w1 wo1 x cs sn t hk1 hd p'))
                     (fun p' -> Exp.exp (at_score nm w2 wo2 x cs sn t hk2 hd p'))

let at_gated_ext (nm:numerics) (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) (x cs sn:S.seq real)
                 (t:nat) (hk1 hv1 hk2 hv2:nat -> nat -> real)
  : Lemma (requires awin w1 wo1 w2 wo2 /\ hagree t hk1 hk2 /\ hagree t hv1 hv2)
          (ensures at_gated nm w1 wo1 x cs sn t hk1 hv1 == at_gated nm w2 wo2 x cs sn t hk2 hv2)
  = at_proj_ext nm w1 wo1 w2 wo2 x;
    introduce forall (c:nat). c < 2048 ==>
      vsum (t + 1) (fun p -> at_prob nm w1 wo1 x cs sn t hk1 (c / 256) p *. hist_or hv1 (at_v nm w1 wo1 x) t p (((c / 256) / 4) * 256 + c % 256)) ==
      vsum (t + 1) (fun p -> at_prob nm w2 wo2 x cs sn t hk2 (c / 256) p *. hist_or hv2 (at_v nm w2 wo2 x) t p (((c / 256) / 4) * 256 + c % 256))
    with introduce _ ==> _ with begin
      introduce forall (p:nat). p < t + 1 ==>
        at_prob nm w1 wo1 x cs sn t hk1 (c / 256) p == at_prob nm w2 wo2 x cs sn t hk2 (c / 256) p
      with introduce _ ==> _ with at_prob_ext nm w1 wo1 w2 wo2 x cs sn t hk1 hk2 (c / 256) p;
      vsum_ext (t + 1) (fun p -> at_prob nm w1 wo1 x cs sn t hk1 (c / 256) p *. hist_or hv1 (at_v nm w1 wo1 x) t p (((c / 256) / 4) * 256 + c % 256))
                       (fun p -> at_prob nm w2 wo2 x cs sn t hk2 (c / 256) p *. hist_or hv2 (at_v nm w2 wo2 x) t p (((c / 256) / 4) * 256 + c % 256))
    end;
    S.lemma_eq_intro (at_gated nm w1 wo1 x cs sn t hk1 hv1) (at_gated nm w2 wo2 x cs sn t hk2 hv2)

let at_out_ext (nm:numerics) (w1:S.seq real) (wo1:int) (w2:S.seq real) (wo2:int) (x cs sn:S.seq real)
               (t:nat) (hk1 hv1 hk2 hv2:nat -> nat -> real)
  : Lemma (requires awin w1 wo1 w2 wo2 /\ hagree t hk1 hk2 /\ hagree t hv1 hv2)
          (ensures at_out nm w1 wo1 x cs sn t hk1 hv1 == at_out nm w2 wo2 x cs sn t hk2 hv2)
  = at_gated_ext nm w1 wo1 w2 wo2 x cs sn t hk1 hv1 hk2 hv2;
    wwin_sub w1 wo1 w2 wo2 18352640 (at_mix wo1 + 5243392) (at_mix wo2 + 5243392) (1024 * 2048);
    matvec_ext w1 (at_mix wo1 + 5243392) w2 (at_mix wo2 + 5243392) 1024 2048 (at_gated nm w1 wo1 x cs sn t hk1 hv1);
    assert (at_x1 nm w1 wo1 x cs sn t hk1 hv1 == at_x1 nm w2 wo2 x cs sn t hk2 hv2);
    wwin_sub w1 wo1 w2 wo2 18352640 wo1 wo2 11012096;
    mlp_ext nm w1 wo1 w2 wo2 (at_x1 nm w1 wo1 x cs sn t hk1 hv1)

(*** Embedding and LM head *)

let embed_ext (w1 w2:S.seq real) (tok:int)
  : Lemma (requires wwin w1 0 w2 0 254279680 /\ 0 <= tok /\ tok < 248320)
          (ensures embed w1 tok == embed w2 tok)
  = introduce forall (i:nat). i < 1024 ==> at w1 (tok * 1024 + i) == at w2 (tok * 1024 + i)
    with introduce _ ==> _ with wwin_at w1 0 w2 0 254279680 (tok * 1024 + i);
    S.lemma_eq_intro (embed w1 tok) (embed w2 tok)

let logits_ext (nm:numerics) (t1:S.seq real) (a1:int) (n1:S.seq real) (b1:int)
               (t2:S.seq real) (a2:int) (n2:S.seq real) (b2:int) (h:S.seq real)
  : Lemma (requires wwin t1 a1 t2 a2 254279680 /\ wwin n1 b1 n2 b2 1024)
          (ensures logits nm t1 a1 n1 b1 h == logits nm t2 a2 n2 b2 h)
  = rmsnorm1p_ext nm.eps h 1024 n1 b1 n2 b2;
    let hn = rmsnorm1p nm.eps h 1024 n1 b1 in
    introduce forall (m:nat). m < 248320 ==> dot t1 (a1 + m * 1024) hn 1024 == dot t2 (a2 + m * 1024) hn 1024
    with introduce _ ==> _ with begin
      wwin_sub t1 a1 t2 a2 254279680 (a1 + m * 1024) (a2 + m * 1024) 1024;
      dot_ext t1 (a1 + m * 1024) t2 (a2 + m * 1024) hn 1024
    end;
    S.lemma_eq_intro (logits nm t1 a1 n1 b1 h) (logits nm t2 a2 n2 b2 h)
