module KPlay.Spec.Bridge

(* Bridge between the flat specification KPlay.SharedSpec (program buffers,
   packed weight offsets, sequences) and the typed specification KPlay.Spec.

   Views (layout only): `weights_view`, `inputs_of`, `outputs_of` read the
   flat program buffers as typed records.

   Main result (`program_postcondition_iff`): for packed weights of the
   expected length and hidden output of one row per token,
     KPlay.SharedSpec.program_postcondition w prompt steps tokens hidden
       <==> KPlay.Spec.postcondition (inputs_of w prompt steps) (outputs_of tokens hidden).

   Proof outline: `weights_ok tw w` says the typed real weights tw agree with
   the flat real weights w at the packed offsets. Under it, each flat
   reference operation equals its typed equation pointwise, and by induction
   over layers and positions `T.hidden_at` equals `SS.ref_hidden` and
   `T.logits` equals `SS.ref_logits`. For ==> (`program_to_typed`) a typed
   real weight record tw is packed into a flat sequence (`flat_of`, built
   from the layout reader `flat_read`); for <== (`typed_to_program`) a flat
   real sequence rw is read through `weights_view`. The premises
   `S.length w == SS.weights_len` (used by ==>) and
   `S.length hidden == S.length tokens * 1024` (used by <==) fix the buffer
   sizes that the flat postcondition does not state on its own. *)

open FStar.Real
open Kuiper
open Kuiper.Approximates
module S = FStar.Seq
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32
module F = Kuiper.Floating
module I64 = FStar.Int64
module Exp = FStar.Math.Exp
module ML = FStar.Math.Lemmas
module SS = KPlay.SharedSpec
module T = KPlay.Spec

#set-options "--z3rlimit 20 --fuel 1 --ifuel 1"

(*** Index arithmetic *)

let divmod (q:int) (r:nat) (n:pos)
  : Lemma (requires r < n) (ensures (q * n + r) / n == q /\ (q * n + r) % n == r)
  = ML.lemma_div_plus r q n; ML.small_div r n; ML.lemma_mod_plus r q n; ML.small_mod r n

let split3 (h i j:nat)
  : Lemma (requires h < 16 /\ i < 128 /\ j < 128)
          (ensures (let idx = h * 16384 + i * 128 + j in
                    idx / 16384 == h /\ (idx % 16384) / 128 == i /\ idx % 128 == j))
  = divmod h (i * 128 + j) 16384; divmod i j 128; divmod (h * 128 + i) j 128

(*** Same real helpers *)

let rdiv_eq (x y:real) : Lemma (T.rdiv x y == SS.rdiv x y) [SMTPat (T.rdiv x y)] = ()
let rsqrt_eq (x:real) : Lemma (T.rsqrt_pos x == SS.rsqrt_t x) [SMTPat (T.rsqrt_pos x)] = ()
let sigmoid_eq (x:real) : Lemma (T.sigmoid x == SS.sigmoid x) [SMTPat (T.sigmoid x)] = ()
let silu_eq (x:real) : Lemma (T.silu x == SS.silu x) [SMTPat (T.silu x)] = ()
let softplus_eq (x:real) : Lemma (T.softplus x == SS.softplus x) [SMTPat (T.softplus x)] = ()

(*** Agreement of typed values with flat sequences *)

let vec_ok (#n:nat) (v:T.vector real n) (s:S.seq real) (o:int) : prop =
  forall (i:T.index n). v i == SS.at s (o + i)

let mat_ok (#r #c:nat) (m:T.matrix real r c) (s:S.seq real) (o:int) : prop =
  forall (i:T.index r) (j:T.index c). m i j == SS.at s (o + i * c + j)

let nm_ok (tn:T.numerics) (nm:SS.numerics) : prop =
  tn.T.rms_norm_eps == nm.SS.eps /\ tn.T.linear_query_scale == nm.SS.dn_qscale /\
  vec_ok tn.T.inv_freq nm.SS.inv_freq 0

(*** Sums, dot products, norms *)

let sum_vsum (n:nat) (g:T.index n -> GTot real) (f:nat -> real)
  : Lemma (requires forall (k:T.index n). g k == f k) (ensures T.sum n g == SS.vsum n f)
  = S.lemma_eq_intro (S.init_ghost n g) (S.init n (fun (k:nat{k < n}) -> f k))

let dot_vsum (#n:nat) (x y:T.vector real n) (f:nat -> real)
  : Lemma (requires forall (k:T.index n). x k *. y k == f k) (ensures T.dot x y == SS.vsum n f)
  = sum_vsum n (fun k -> x k *. y k) f

let dot_eq (#n:nat) (x y:T.vector real n) (w:S.seq real) (wo:int) (v:S.seq real)
  : Lemma (requires vec_ok x w wo /\ vec_ok y v 0) (ensures T.dot x y == SS.dot w wo v n)
  = dot_vsum x y (fun k -> SS.at w (wo + k) *. SS.at v k)

let sq_sum_eq (#n:nat) (x:T.vector real n) (s:S.seq real) (o:int)
  : Lemma (requires vec_ok x s o) (ensures T.sum_of_squares x == SS.sq_sum s o n)
  = sum_vsum n (fun k -> x k *. x k) (fun k -> SS.at s (o + k) *. SS.at s (o + k))

let linear_eq (#r #c:nat) (m:T.matrix real r c) (x:T.vector real c) (w:S.seq real) (o:int) (s:S.seq real)
  : Lemma (requires mat_ok m w o /\ vec_ok x s 0) (ensures vec_ok (T.linear m x) (SS.matvec w o r c s) 0)
  = introduce forall (i:T.index r). T.linear m x i == SS.at (SS.matvec w o r c s) (0 + i)
    with dot_eq (m i) x w (o + i * c) s

let rms_norm_eq (#n:nat) (eps:real) (x wv:T.vector real n) (xs w:S.seq real) (wo:int)
  : Lemma (requires vec_ok x xs 0 /\ vec_ok wv w wo)
          (ensures vec_ok (T.rms_norm eps x wv) (SS.rmsnorm1p eps xs n w wo) 0)
  = sq_sum_eq x xs 0

let head_norm_eq (eps:real) (x wv:T.vector real T.head_dim) (raw w:S.seq real) (o wo:int)
  : Lemma (requires vec_ok x raw o /\ vec_ok wv w wo)
          (ensures vec_ok (T.rms_norm eps x wv) (SS.head_norm eps raw o w wo) 0)
  = sq_sum_eq x raw o; assert (of_int T.head_dim == 256.0R)

(*** Per-layer weight agreement *)

let mlp_ok (p:T.mlp_params real) (w:S.seq real) (wo:int) : prop =
  mat_ok p.T.gate_proj w (wo + 2048) /\ mat_ok p.T.up_proj w (wo + 3672064) /\
  mat_ok p.T.down_proj w (wo + 7342080)

let common_ok (#m:Type0 -> Type0) (p:T.decoder_layer m real) (w:S.seq real) (wo:int) : prop =
  vec_ok p.T.input_layernorm w wo /\ vec_ok p.T.post_attention_layernorm w (wo + 1024) /\
  mlp_ok p.T.mlp w wo

let linear_attn_ok (p:T.linear_attn_params real) (w:S.seq real) (mo:int) : prop =
  mat_ok p.T.in_proj_qkv w mo /\ mat_ok p.T.in_proj_z w (mo + 6291456) /\
  mat_ok p.T.in_proj_b w (mo + 8388608) /\ mat_ok p.T.in_proj_a w (mo + 8404992) /\
  mat_ok p.T.conv1d w (mo + 8421376) /\ vec_ok p.T.a_log w (mo + 8445952) /\
  vec_ok p.T.dt_bias w (mo + 8445968) /\ vec_ok p.T.norm w (mo + 8445984) /\
  mat_ok p.T.out_proj w (mo + 8446112)

let self_attn_ok (p:T.self_attn_params real) (w:S.seq real) (mo:int) : prop =
  mat_ok p.T.q_proj w mo /\ mat_ok p.T.k_proj w (mo + 4194304) /\
  mat_ok p.T.v_proj w (mo + 4718592) /\ vec_ok p.T.q_norm w (mo + 5242880) /\
  vec_ok p.T.k_norm w (mo + 5243136) /\ mat_ok p.T.o_proj w (mo + 5243392)

let linear_layer_ok (p:T.decoder_layer T.linear_attn_params real) (w:S.seq real) (wo:int) : prop =
  common_ok p w wo /\ linear_attn_ok p.T.token_mixer w (wo + 11012096)

let full_layer_ok (p:T.decoder_layer T.self_attn_params real) (w:S.seq real) (wo:int) : prop =
  common_ok p w wo /\ self_attn_ok p.T.token_mixer w (wo + 11012096)

(*** MLP and residuals (both layer kinds) *)

let decoder_output_eq (#m:Type0 -> Type0) (nm:SS.numerics) (p:T.decoder_layer m real) (w:S.seq real) (wo:int)
                      (x mixed:T.vector real T.hidden_size) (xs ms:S.seq real)
  : Lemma (requires common_ok p w wo /\ vec_ok x xs 0 /\ vec_ok mixed ms 0)
          (ensures vec_ok (T.decoder_output nm.SS.eps p x mixed) (SS.mlp_out nm w wo (SS.vadd xs ms 1024)) 0)
  = let x1 = SS.vadd xs ms 1024 in
    let h = T.vadd x mixed in
    assert (vec_ok h x1 0);
    let hn = T.rms_norm nm.SS.eps h p.T.post_attention_layernorm in
    rms_norm_eq nm.SS.eps h p.T.post_attention_layernorm x1 w (wo + 1024);
    let fin = SS.mlp_in nm w wo x1 in
    linear_eq p.T.mlp.T.gate_proj hn w (wo + 2048) fin;
    linear_eq p.T.mlp.T.up_proj hn w (wo + 3672064) fin;
    let act = T.swiglu (T.linear p.T.mlp.T.gate_proj hn) (T.linear p.T.mlp.T.up_proj hn) in
    assert (vec_ok act (SS.mlp_act nm w wo x1) 0);
    linear_eq p.T.mlp.T.down_proj act w (wo + 7342080) (SS.mlp_act nm w wo x1)

(*** Gated DeltaNet layer *)

let conv_ok (c:T.conv_state) (s:S.seq real) : prop =
  forall (ch:T.index T.conv_dim) (k:T.index T.conv_kernel_size). c ch k == SS.at s (ch * 4 + k)

let rec_ok (r:T.recurrent_state) (s:S.seq real) : prop =
  forall (h:T.index T.linear_num_heads) (i j:T.index T.linear_head_dim).
    r h i j == SS.at s (h * 16384 + i * 128 + j)

let conv_out_eq (window:T.conv_state) (weight:T.matrix real T.conv_dim T.conv_kernel_size)
                (nm:SS.numerics) (w:S.seq real) (wo:int) (x cv:S.seq real)
  : Lemma (requires conv_ok window (SS.dn_conv nm w wo x cv) /\ mat_ok weight w (SS.dn_mix wo + 8421376))
          (ensures vec_ok (T.causal_conv1d_output window weight) (SS.dn_conv_out nm w wo x cv) 0)
  = introduce forall (c:T.index T.conv_dim).
      T.causal_conv1d_output window weight c == SS.at (SS.dn_conv_out nm w wo x cv) (0 + c)
    with sum_vsum T.conv_kernel_size (fun k -> window c k *. weight c k)
           (fun k -> SS.at (SS.dn_conv nm w wo x cv) (c * 4 + k) *. SS.at w (SS.dn_mix wo + 8421376 + c * 4 + k))

let dn_qkv_eq (tn:T.numerics) (nm:SS.numerics) (p:T.linear_attn_params real) (xn:T.vector real T.hidden_size)
              (conv:T.conv_state) (w:S.seq real) (wo:int) (x cv:S.seq real)
  : Lemma (requires nm_ok tn nm /\ vec_ok (T.linear_attn_conv_out p xn conv) (SS.dn_conv_out nm w wo x cv) 0)
          (ensures mat_ok (T.linear_attn_query tn p xn conv) (SS.dn_q nm w wo x cv) 0 /\
                   mat_ok (T.linear_attn_key tn p xn conv) (SS.dn_k nm w wo x cv) 0 /\
                   mat_ok (T.linear_attn_value p xn conv) (SS.dn_v nm w wo x cv) 0)
  = let co = SS.dn_conv_out nm w wo x cv in
    introduce forall (h:T.index T.linear_num_heads) (d:T.index T.linear_head_dim).
      T.linear_attn_query tn p xn conv h d == SS.at (SS.dn_q nm w wo x cv) (0 + h * 128 + d) /\
      T.linear_attn_key tn p xn conv h d == SS.at (SS.dn_k nm w wo x cv) (0 + h * 128 + d) /\
      T.linear_attn_value p xn conv h d == SS.at (SS.dn_v nm w wo x cv) (0 + h * 128 + d)
    with begin
      sq_sum_eq #T.linear_head_dim (fun e -> T.linear_attn_conv_out p xn conv (h * 128 + e)) co (h * 128);
      sq_sum_eq #T.linear_head_dim (fun e -> T.linear_attn_conv_out p xn conv (2048 + h * 128 + e)) co (2048 + h * 128);
      divmod h d 128
    end

let delta_eq (state:T.recurrent_state) (k v:T.matrix real T.linear_num_heads T.linear_head_dim)
             (decay beta:T.vector real T.linear_num_heads)
             (nm:SS.numerics) (w:S.seq real) (wo:int) (x cv st:S.seq real)
             (h:T.index T.linear_num_heads) (j:T.index T.linear_head_dim)
  : Lemma (requires rec_ok state st /\ mat_ok k (SS.dn_k nm w wo x cv) 0 /\ mat_ok v (SS.dn_v nm w wo x cv) 0 /\
                    decay h == SS.dn_decay nm w wo x h /\ beta h == SS.dn_beta nm w wo x h)
          (ensures T.delta_rule_correction state k v decay beta h j == SS.dn_delta nm w wo x cv st h j)
  = sum_vsum T.linear_head_dim (fun i -> (state h i j *. decay h) *. k h i)
      (fun i -> (SS.at st (h * 16384 + i * 128 + j) *. SS.dn_decay nm w wo x h) *.
                SS.at (SS.dn_k nm w wo x cv) (h * 128 + i))

let dn_state_eq (state:T.recurrent_state) (k v:T.matrix real T.linear_num_heads T.linear_head_dim)
                (decay beta:T.vector real T.linear_num_heads)
                (nm:SS.numerics) (w:S.seq real) (wo:int) (x cv st:S.seq real)
  : Lemma (requires rec_ok state st /\ mat_ok k (SS.dn_k nm w wo x cv) 0 /\ mat_ok v (SS.dn_v nm w wo x cv) 0 /\
                    (forall (h:T.index T.linear_num_heads).
                       decay h == SS.dn_decay nm w wo x h /\ beta h == SS.dn_beta nm w wo x h))
          (ensures rec_ok (T.delta_rule_state state k v decay beta) (SS.dn_state nm w wo x cv st))
  = introduce forall (h:T.index T.linear_num_heads) (i j:T.index T.linear_head_dim).
      T.delta_rule_state state k v decay beta h i j == SS.at (SS.dn_state nm w wo x cv st) (h * 16384 + i * 128 + j)
    with (split3 h i j; delta_eq state k v decay beta nm w wo x cv st h j)

let readout_eq (state:T.recurrent_state) (q:T.matrix real T.linear_num_heads T.linear_head_dim)
               (nm:SS.numerics) (w:S.seq real) (wo:int) (x cv st:S.seq real)
               (h:T.index T.linear_num_heads) (j:T.index T.linear_head_dim) (c:nat)
  : Lemma (requires c == h * 128 + j /\ rec_ok state (SS.dn_state nm w wo x cv st) /\
                    mat_ok q (SS.dn_q nm w wo x cv) 0)
          (ensures T.recurrent_readout state q h j == SS.at (SS.dn_core nm w wo x cv st) c)
  = divmod h j 128;
    sum_vsum T.linear_head_dim (fun i -> state h i j *. q h i)
      (fun i -> SS.at (SS.dn_state nm w wo x cv st) ((c / 128) * 16384 + i * 128 + c % 128) *.
                SS.at (SS.dn_q nm w wo x cv) ((c / 128) * 128 + i))

let gated_eq (eps:real) (wv:T.vector real T.linear_head_dim) (core:T.matrix real T.linear_num_heads T.linear_head_dim)
             (z:T.vector real T.linear_key_dim)
             (nm:SS.numerics) (w:S.seq real) (wo:int) (x cv st:S.seq real)
             (h:T.index T.linear_num_heads) (j:T.index T.linear_head_dim) (c:nat)
  : Lemma (requires c == h * 128 + j /\ eps == nm.SS.eps /\ vec_ok wv w (SS.dn_mix wo + 8445984) /\
                    mat_ok core (SS.dn_core nm w wo x cv st) 0 /\
                    vec_ok z (SS.matvec w (SS.dn_mix wo + 6291456) 2048 1024 (SS.dn_hin nm w wo x)) 0)
          (ensures T.gated_head_norm eps wv core z h j == SS.at (SS.dn_gated nm w wo x cv st) c)
  = divmod h j 128;
    sq_sum_eq (core h) (SS.dn_core nm w wo x cv st) (h * 128);
    assert (of_int T.linear_head_dim == 128.0R)

let dn_mixer_eq (tn:T.numerics) (nm:SS.numerics) (p:T.linear_attn_params real) (xn:T.vector real T.hidden_size)
                (conv:T.conv_state) (state:T.recurrent_state) (w:S.seq real) (wo:int) (x cv rs:S.seq real)
  : Lemma (requires nm_ok tn nm /\ linear_attn_ok p w (wo + 11012096) /\ vec_ok xn (SS.dn_hin nm w wo x) 0 /\
                    conv_ok conv cv /\ rec_ok state rs)
          (ensures conv_ok (T.linear_attn_conv_state p xn conv) (SS.dn_conv nm w wo x cv) /\
                   rec_ok (T.linear_attn_recurrent_state tn p xn conv state) (SS.dn_state nm w wo x cv rs) /\
                   vec_ok (T.linear_attn_output tn p xn conv state)
                     (SS.matvec w (SS.dn_mix wo + 8446112) 1024 2048 (SS.dn_gated nm w wo x cv rs)) 0)
  = let hin = SS.dn_hin nm w wo x in
    linear_eq p.T.in_proj_qkv xn w (SS.dn_mix wo) hin;
    let window = T.linear_attn_conv_state p xn conv in
    introduce forall (ch:T.index T.conv_dim) (k:T.index T.conv_kernel_size).
      window ch k == SS.at (SS.dn_conv nm w wo x cv) (ch * 4 + k)
    with divmod ch k 4;
    conv_out_eq window p.T.conv1d nm w wo x cv;
    dn_qkv_eq tn nm p xn conv w wo x cv;
    linear_eq p.T.in_proj_b xn w (SS.dn_mix wo + 8388608) hin;
    linear_eq p.T.in_proj_a xn w (SS.dn_mix wo + 8404992) hin;
    let key = T.linear_attn_key tn p xn conv in
    let value = T.linear_attn_value p xn conv in
    let decay = T.linear_attn_decay p xn in
    let beta = T.linear_attn_beta p xn in
    assert (forall (h:T.index T.linear_num_heads).
              decay h == SS.dn_decay nm w wo x h /\ beta h == SS.dn_beta nm w wo x h);
    dn_state_eq state key value decay beta nm w wo x cv rs;
    let state' = T.linear_attn_recurrent_state tn p xn conv state in
    let q = T.linear_attn_query tn p xn conv in
    let core = T.linear_attn_core tn p xn conv state in
    introduce forall (h:T.index T.linear_num_heads) (j:T.index T.linear_head_dim).
      core h j == SS.at (SS.dn_core nm w wo x cv rs) (0 + h * 128 + j)
    with readout_eq state' q nm w wo x cv rs h j (h * 128 + j);
    let z = T.linear p.T.in_proj_z xn in
    linear_eq p.T.in_proj_z xn w (SS.dn_mix wo + 6291456) hin;
    let flat_gated (c:T.index T.linear_key_dim) : GTot real =
      T.gated_head_norm tn.T.rms_norm_eps p.T.norm core z (c / 128) (c % 128) in
    introduce forall (c:T.index T.linear_key_dim).
      flat_gated c == SS.at (SS.dn_gated nm w wo x cv rs) (0 + c)
    with gated_eq tn.T.rms_norm_eps p.T.norm core z nm w wo x cv rs (c / 128) (c % 128) c;
    linear_eq p.T.out_proj (fun (c:T.index T.linear_key_dim) ->
        T.gated_head_norm tn.T.rms_norm_eps p.T.norm (T.linear_attn_core tn p xn conv state)
          (T.linear p.T.in_proj_z xn) (c / 128) (c % 128))
      w (SS.dn_mix wo + 8446112) (SS.dn_gated nm w wo x cv rs)

(*** Gated full-attention layer *)

let rope_eq (u:T.vector real T.head_dim) (cos sin:T.vector real T.rotary_pairs) (us cs sn:S.seq real)
  : Lemma (requires vec_ok u us 0 /\ vec_ok cos cs 0 /\ vec_ok sin sn 0)
          (ensures vec_ok (T.apply_rotary u cos sin) (SS.rope us cs sn) 0)
  = ()

let attn_qkv_eq (tn:T.numerics) (nm:SS.numerics) (p:T.self_attn_params real) (xn:T.vector real T.hidden_size)
                (cos sin:T.vector real T.rotary_pairs) (w:S.seq real) (wo:int) (x cs sn:S.seq real)
  : Lemma (requires nm_ok tn nm /\ self_attn_ok p w (wo + 11012096) /\ vec_ok xn (SS.at_hin nm w wo x) 0 /\
                    vec_ok cos cs 0 /\ vec_ok sin sn 0)
          (ensures mat_ok (T.attn_query tn p xn cos sin) (SS.at_q nm w wo x cs sn) 0 /\
                   mat_ok (T.attn_key tn p xn cos sin) (SS.at_k nm w wo x cs sn) 0 /\
                   mat_ok (T.attn_value p xn) (SS.at_v nm w wo x) 0 /\
                   vec_ok (T.linear p.T.q_proj xn) (SS.at_qraw nm w wo x) 0)
  = let hin = SS.at_hin nm w wo x in
    let raw = SS.at_qraw nm w wo x in
    let kraw = SS.at_kraw nm w wo x in
    linear_eq p.T.q_proj xn w (SS.at_mix wo) hin;
    linear_eq p.T.k_proj xn w (SS.at_mix wo + 4194304) hin;
    linear_eq p.T.v_proj xn w (SS.at_mix wo + 4718592) hin;
    introduce forall (h:T.index T.num_attention_heads) (i:T.index T.head_dim).
      T.attn_query tn p xn cos sin h i == SS.at (SS.at_q nm w wo x cs sn) (0 + h * 256 + i)
    with begin
      let u = (fun (e:T.index T.head_dim) -> T.linear p.T.q_proj xn (h * 512 + e)) in
      head_norm_eq tn.T.rms_norm_eps u p.T.q_norm raw w (h * 512) (SS.at_mix wo + 5242880);
      rope_eq (T.rms_norm tn.T.rms_norm_eps u p.T.q_norm) cos sin
        (SS.head_norm nm.SS.eps raw (h * 512) w (SS.at_mix wo + 5242880)) cs sn;
      divmod h i 256
    end;
    introduce forall (h:T.index T.num_key_value_heads) (i:T.index T.head_dim).
      T.attn_key tn p xn cos sin h i == SS.at (SS.at_k nm w wo x cs sn) (0 + h * 256 + i)
    with begin
      let u = (fun (e:T.index T.head_dim) -> T.linear p.T.k_proj xn (h * 256 + e)) in
      head_norm_eq tn.T.rms_norm_eps u p.T.k_norm kraw w (h * 256) (SS.at_mix wo + 5243136);
      rope_eq (T.rms_norm tn.T.rms_norm_eps u p.T.k_norm) cos sin
        (SS.head_norm nm.SS.eps kraw (h * 256) w (SS.at_mix wo + 5243136)) cs sn;
      divmod h i 256
    end

(* the typed cache of one attention layer over positions 0..t agrees with the
   flat history-or-current lookup *)
let keys_ok (t:nat) (keys:nat -> GTot T.kv_entry) (hk:nat -> nat -> real) (cur:S.seq real) : prop =
  forall (pos:nat) (kvh:T.index T.num_key_value_heads) (e:T.index T.head_dim).
    pos <= t ==> keys pos kvh e == SS.hist_or hk cur t pos (kvh * 256 + e)

let score_eq (q k:T.vector real T.head_dim) (nm:SS.numerics) (w:S.seq real) (wo:int) (x cs sn:S.seq real)
             (t:nat) (hk:nat -> nat -> real) (hd p:nat)
  : Lemma (requires forall (i:T.index T.head_dim).
                      q i == SS.at (SS.at_q nm w wo x cs sn) (hd * 256 + i) /\
                      k i == SS.hist_or hk (SS.at_k nm w wo x cs sn) t p ((hd / 4) * 256 + i))
          (ensures T.attn_score q k == SS.at_score nm w wo x cs sn t hk hd p)
  = dot_vsum q k (fun i -> SS.at (SS.at_q nm w wo x cs sn) (hd * 256 + i) *.
                           SS.hist_or hk (SS.at_k nm w wo x cs sn) t p ((hd / 4) * 256 + i))

let weight_eq (t:nat) (q:T.vector real T.head_dim) (keys:nat -> GTot (T.vector real T.head_dim))
              (nm:SS.numerics) (w:S.seq real) (wo:int) (x cs sn:S.seq real) (hk:nat -> nat -> real) (hd p:nat)
  : Lemma (requires p <= t /\
                    (forall (pos:nat). pos <= t ==>
                       T.attn_score q (keys pos) == SS.at_score nm w wo x cs sn t hk hd pos))
          (ensures T.attn_weight t q keys p == SS.at_prob nm w wo x cs sn t hk hd p)
  = sum_vsum (t + 1) (fun p' -> Exp.exp (T.attn_score q (keys p')))
      (fun p' -> Exp.exp (SS.at_score nm w wo x cs sn t hk hd p'))

let mul_cong (a b c d:real) : Lemma (requires a == c /\ b == d) (ensures a *. b == c *. d) = ()

let head_eq (t:nat) (q:T.vector real T.head_dim) (keys values:nat -> GTot (T.vector real T.head_dim))
            (e:T.index T.head_dim) (nm:SS.numerics) (w:S.seq real) (wo:int) (x cs sn:S.seq real)
            (hk hv:nat -> nat -> real) (c:nat)
  : Lemma (requires (forall (pos:nat). pos <= t ==>
                       T.attn_weight t q keys pos == SS.at_prob nm w wo x cs sn t hk (c / 256) pos) /\
                    (forall (pos:nat). pos <= t ==>
                       values pos e == SS.hist_or hv (SS.at_v nm w wo x) t pos (((c / 256) / 4) * 256 + c % 256)))
          (ensures T.attn_head t q keys values e ==
                   SS.vsum (t + 1) (fun p -> SS.at_prob nm w wo x cs sn t hk (c / 256) p *.
                     SS.hist_or hv (SS.at_v nm w wo x) t p (((c / 256) / 4) * 256 + c % 256)))
  = introduce forall (pos:T.index (t + 1)).
        T.attn_weight t q keys pos *. values pos e ==
        SS.at_prob nm w wo x cs sn t hk (c / 256) pos *.
        SS.hist_or hv (SS.at_v nm w wo x) t pos (((c / 256) / 4) * 256 + c % 256)
    with mul_cong (T.attn_weight t q keys pos) (values pos e)
           (SS.at_prob nm w wo x cs sn t hk (c / 256) pos)
           (SS.hist_or hv (SS.at_v nm w wo x) t pos (((c / 256) / 4) * 256 + c % 256));
    sum_vsum (t + 1) (fun pos -> T.attn_weight t q keys pos *. values pos e)
      (fun p -> SS.at_prob nm w wo x cs sn t hk (c / 256) p *.
                SS.hist_or hv (SS.at_v nm w wo x) t p (((c / 256) / 4) * 256 + c % 256))

let heads_eq (t:nat) (q:T.matrix real T.num_attention_heads T.head_dim) (keys values:nat -> GTot T.kv_entry)
             (nm:SS.numerics) (w:S.seq real) (wo:int) (x cs sn:S.seq real) (hk hv:nat -> nat -> real)
             (h:T.index T.num_attention_heads) (e:T.index T.head_dim) (c:nat)
  : Lemma (requires c == h * 256 + e /\ mat_ok q (SS.at_q nm w wo x cs sn) 0 /\
                    keys_ok t keys hk (SS.at_k nm w wo x cs sn) /\ keys_ok t values hv (SS.at_v nm w wo x))
          (ensures T.attention_heads t q keys values h e ==
                   SS.vsum (t + 1) (fun p -> SS.at_prob nm w wo x cs sn t hk (c / 256) p *.
                     SS.hist_or hv (SS.at_v nm w wo x) t p (((c / 256) / 4) * 256 + c % 256)))
  = divmod h e 256;
    let kh = (fun (pos:nat) -> keys pos (h / 4)) in
    let vh = (fun (pos:nat) -> values pos (h / 4)) in
    introduce forall (pos:nat). pos <= t ==>
        T.attn_score (q h) (kh pos) == SS.at_score nm w wo x cs sn t hk (c / 256) pos
    with introduce _ ==> _
    with score_eq (q h) (kh pos) nm w wo x cs sn t hk (c / 256) pos;
    introduce forall (pos:nat). pos <= t ==>
        T.attn_weight t (q h) kh pos == SS.at_prob nm w wo x cs sn t hk (c / 256) pos
    with introduce _ ==> _
    with weight_eq t (q h) kh nm w wo x cs sn hk (c / 256) pos;
    head_eq t (q h) kh vh e nm w wo x cs sn hk hv c

let self_attention_eq (tn:T.numerics) (nm:SS.numerics) (p:T.self_attn_params real) (xn:T.vector real T.hidden_size)
                      (cos sin:T.vector real T.rotary_pairs) (t:nat) (keys values:nat -> GTot T.kv_entry)
                      (w:S.seq real) (wo:int) (x cs sn:S.seq real) (hk hv:nat -> nat -> real)
  : Lemma (requires mat_ok (T.attn_query tn p xn cos sin) (SS.at_q nm w wo x cs sn) 0 /\
                    vec_ok (T.linear p.T.q_proj xn) (SS.at_qraw nm w wo x) 0 /\
                    mat_ok p.T.o_proj w (SS.at_mix wo + 5243392) /\
                    keys_ok t keys hk (SS.at_k nm w wo x cs sn) /\ keys_ok t values hv (SS.at_v nm w wo x))
          (ensures vec_ok (T.self_attention tn p xn cos sin t keys values)
                     (SS.matvec w (SS.at_mix wo + 5243392) 1024 2048 (SS.at_gated nm w wo x cs sn t hk hv)) 0)
  = let q = T.attn_query tn p xn cos sin in
    let heads = T.attention_heads t q keys values in
    introduce forall (c:T.index T.attn_out_dim).
        heads (c / 256) (c % 256) *. T.sigmoid (T.attn_gate p xn (c / 256) (c % 256)) ==
        SS.at (SS.at_gated nm w wo x cs sn t hk hv) (0 + c)
    with heads_eq t q keys values nm w wo x cs sn hk hv (c / 256) (c % 256) c;
    linear_eq p.T.o_proj (fun (c:T.index T.attn_out_dim) ->
        T.attention_heads t (T.attn_query tn p xn cos sin) keys values (c / 256) (c % 256) *.
        T.sigmoid (T.attn_gate p xn (c / 256) (c % 256)))
      w (SS.at_mix wo + 5243392) (SS.at_gated nm w wo x cs sn t hk hv)

(*** Decoder state agreement *)

(* DeltaNet layer d of the model is layer 4 * (d / 3) + d % 3, attention
   layer g is layer 4 * g + 3 *)
let weights_ok (tw:T.model_weights real) (w:S.seq real) : prop =
  mat_ok tw.T.embed_tokens w 0 /\ vec_ok tw.T.final_norm w SS.final_norm_off /\
  (forall (d:T.index T.linear_layer_count).
     linear_layer_ok (tw.T.linear_layers d) w (SS.layer_off (4 * (d / 3) + d % 3))) /\
  (forall (g:T.index T.full_layer_count). full_layer_ok (tw.T.full_layers g) w (SS.layer_off (4 * g + 3)))

let dn_cache_ok (c:T.cache_state) (convs recs:S.seq (S.seq real)) : prop =
  S.length convs == 18 /\ S.length recs == 18 /\
  (forall (d:T.index T.linear_layer_count). conv_ok (c.T.conv_states d) (SS.sget convs d)) /\
  (forall (d:T.index T.linear_layer_count). rec_ok (c.T.recurrent_states d) (SS.sget recs d))

(* positions before t: typed cache vs flat KV history *)
let past_ok (t:nat) (c:T.cache_state) (kvs:S.seq (S.seq real)) : prop =
  (forall (g:T.index T.full_layer_count) (pos:nat) (kvh:T.index T.num_key_value_heads) (e:T.index T.head_dim).
     pos < t ==> c.T.key_cache g pos kvh e == SS.kv_get kvs pos (g * 1024 + kvh * 256 + e)) /\
  (forall (g:T.index T.full_layer_count) (pos:nat) (kvh:T.index T.num_key_value_heads) (e:T.index T.head_dim).
     pos < t ==> c.T.value_cache g pos kvh e == SS.kv_get kvs pos (g * 1024 + 512 + kvh * 256 + e))

(* position t, attention layers already run (model layer < l) *)
let cur_ok (t l:nat) (c:T.cache_state) (cur:S.seq real) : prop =
  (forall (g:T.index T.full_layer_count) (kvh:T.index T.num_key_value_heads) (e:T.index T.head_dim).
     4 * g + 3 < l ==> c.T.key_cache g t kvh e == SS.at cur (g * 1024 + kvh * 256 + e)) /\
  (forall (g:T.index T.full_layer_count) (kvh:T.index T.num_key_value_heads) (e:T.index T.head_dim).
     4 * g + 3 < l ==> c.T.value_cache g t kvh e == SS.at cur (g * 1024 + 512 + kvh * 256 + e))

let lstate_ok (t l:nat) (ts:T.layer_state) (kvs:S.seq (S.seq real)) (fs:SS.lstate) : prop =
  dn_cache_ok ts.T.cache fs.SS.l_convs fs.SS.l_recs /\ past_ok t ts.T.cache kvs /\
  cur_ok t l ts.T.cache fs.SS.l_cur /\ vec_ok ts.T.hidden fs.SS.l_x 0

let linear_layer_index (l:nat)
  : Lemma (requires l % 4 <> 3)
          (ensures (let d = 3 * (l / 4) + l % 4 in 4 * (d / 3) + d % 3 == l /\ d == SS.dn_index l))
  = divmod (l / 4) (l % 4) 3; ML.euclidean_division_definition l 4

let sget_supd (s:S.seq (S.seq real)) (d:nat) (v:S.seq real) (d':nat)
  : Lemma (requires d < S.length s /\ d' < S.length s)
          (ensures S.length (SS.supd s d v) == S.length s /\
                   SS.sget (SS.supd s d v) d' == (if d' = d then v else SS.sget s d'))
  = ()

(* the flat layer transitions of SS.run_layers, named *)
let dn_next (nm:SS.numerics) (w:S.seq real) (st:SS.lstate) (l:nat) : SS.lstate =
  let wo = SS.layer_off l in
  let d = SS.dn_index l in
  let cv = SS.sget st.SS.l_convs d in
  let rs = SS.sget st.SS.l_recs d in
  { SS.l_convs = SS.supd st.SS.l_convs d (SS.dn_conv nm w wo st.SS.l_x cv);
    SS.l_recs = SS.supd st.SS.l_recs d (SS.dn_state nm w wo st.SS.l_x cv rs);
    SS.l_cur = st.SS.l_cur;
    SS.l_x = SS.dn_out nm w wo st.SS.l_x cv rs }

let full_next (nm:SS.numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real) (kvs:S.seq (S.seq real))
              (st:SS.lstate) (l:nat) : SS.lstate =
  let wo = SS.layer_off l in
  let g = l / 4 in
  let hk = fun (p j:nat) -> SS.kv_get kvs p (g * 1024 + j) in
  let hv = fun (p j:nat) -> SS.kv_get kvs p (g * 1024 + 512 + j) in
  { st with
    SS.l_cur = SS.write_kv st.SS.l_cur g (SS.at_k nm w wo st.SS.l_x cs sn) (SS.at_v nm w wo st.SS.l_x);
    SS.l_x = SS.at_out nm w wo st.SS.l_x cs sn t hk hv }

let run_layers_unfold (nm:SS.numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real) (kvs:S.seq (S.seq real))
                      (st:SS.lstate) (l:nat)
  : Lemma (requires l < 24)
          (ensures SS.run_layers nm w t cs sn kvs st l ==
                   SS.run_layers nm w t cs sn kvs
                     (if l % 4 = 3 then full_next nm w t cs sn kvs st l else dn_next nm w st l) (l + 1))
  = ()

let linear_step_eq (tn:T.numerics) (nm:SS.numerics) (tw:T.model_weights real) (w:S.seq real) (t:nat)
                   (kvs:S.seq (S.seq real)) (ts:T.layer_state) (fs:SS.lstate) (l:nat)
  : Lemma (requires nm_ok tn nm /\ weights_ok tw w /\ l < 24 /\ l % 4 <> 3 /\ lstate_ok t l ts kvs fs)
          (ensures (let d : T.index T.linear_layer_count = 3 * (l / 4) + l % 4 in
                    lstate_ok t (l + 1) (T.linear_layer_step tn (tw.T.linear_layers d) d ts) kvs (dn_next nm w fs l)))
  = let d : T.index T.linear_layer_count = 3 * (l / 4) + l % 4 in
    linear_layer_index l;
    let wo = SS.layer_off l in
    let p = tw.T.linear_layers d in
    assert (linear_layer_ok p w wo);
    let cv = SS.sget fs.SS.l_convs d in
    let rs = SS.sget fs.SS.l_recs d in
    let x = ts.T.hidden in
    let xn = T.rms_norm tn.T.rms_norm_eps x p.T.input_layernorm in
    rms_norm_eq tn.T.rms_norm_eps x p.T.input_layernorm fs.SS.l_x w wo;
    let conv = ts.T.cache.T.conv_states d in
    let state = ts.T.cache.T.recurrent_states d in
    dn_mixer_eq tn nm p.T.token_mixer xn conv state w wo fs.SS.l_x cv rs;
    decoder_output_eq nm p w wo x (T.linear_attn_output tn p.T.token_mixer xn conv state) fs.SS.l_x
      (SS.matvec w (SS.dn_mix wo + 8446112) 1024 2048 (SS.dn_gated nm w wo fs.SS.l_x cv rs));
    let ts' = T.linear_layer_step tn p d ts in
    let convs' = SS.supd fs.SS.l_convs d (SS.dn_conv nm w wo fs.SS.l_x cv) in
    let recs' = SS.supd fs.SS.l_recs d (SS.dn_state nm w wo fs.SS.l_x cv rs) in
    introduce forall (d':T.index T.linear_layer_count). conv_ok (ts'.T.cache.T.conv_states d') (SS.sget convs' d')
    with sget_supd fs.SS.l_convs d (SS.dn_conv nm w wo fs.SS.l_x cv) d';
    introduce forall (d':T.index T.linear_layer_count). rec_ok (ts'.T.cache.T.recurrent_states d') (SS.sget recs' d')
    with sget_supd fs.SS.l_recs d (SS.dn_state nm w wo fs.SS.l_x cv rs) d'

let write_kv_at (cur kn vn:S.seq real) (g g':T.index T.full_layer_count) (j:nat)
  : Lemma (requires j < 1024)
          (ensures SS.at (SS.write_kv cur g kn vn) (g' * 1024 + j) ==
                   (if g' = g then (if j < 512 then SS.at kn j else SS.at vn (j - 512))
                    else SS.at cur (g' * 1024 + j)))
  = divmod g' j 1024

(* the attended keys / values of attention layer l / 4 at position t *)
let attn_keys_ok (t l:nat) (hist:nat -> GTot T.kv_entry) (entry:T.kv_entry) (kvs:S.seq (S.seq real)) (cur:S.seq real)
  : Lemma (requires (forall (pos:nat) (kvh:T.index T.num_key_value_heads) (e:T.index T.head_dim).
                       pos < t ==> hist pos kvh e == SS.kv_get kvs pos ((l / 4) * 1024 + kvh * 256 + e)) /\
                    mat_ok entry cur 0)
          (ensures keys_ok t (T.append_position hist t entry) (fun (p j:nat) -> SS.kv_get kvs p ((l / 4) * 1024 + j)) cur)
  = ()

let attn_values_ok (t l:nat) (hist:nat -> GTot T.kv_entry) (entry:T.kv_entry) (kvs:S.seq (S.seq real)) (cur:S.seq real)
  : Lemma (requires (forall (pos:nat) (kvh:T.index T.num_key_value_heads) (e:T.index T.head_dim).
                       pos < t ==> hist pos kvh e == SS.kv_get kvs pos ((l / 4) * 1024 + 512 + kvh * 256 + e)) /\
                    mat_ok entry cur 0)
          (ensures keys_ok t (T.append_position hist t entry)
                     (fun (p j:nat) -> SS.kv_get kvs p ((l / 4) * 1024 + 512 + j)) cur)
  = ()

let full_step_eq (tn:T.numerics) (nm:SS.numerics) (tw:T.model_weights real) (w:S.seq real) (t:nat)
                 (cos sin:T.vector real T.rotary_pairs) (cs sn:S.seq real)
                 (kvs:S.seq (S.seq real)) (ts:T.layer_state) (fs:SS.lstate) (l:nat)
  : Lemma (requires nm_ok tn nm /\ weights_ok tw w /\ l < 24 /\ l % 4 = 3 /\ lstate_ok t l ts kvs fs /\
                    vec_ok cos cs 0 /\ vec_ok sin sn 0)
          (ensures lstate_ok t (l + 1) (T.full_layer_step tn (tw.T.full_layers (l / 4)) (l / 4) t cos sin ts) kvs
                     (full_next nm w t cs sn kvs fs l))
  = let g : T.index T.full_layer_count = l / 4 in
    ML.euclidean_division_definition l 4;
    let wo = SS.layer_off l in
    let p = tw.T.full_layers g in
    assert (full_layer_ok p w wo);
    let x = ts.T.hidden in
    let xs = fs.SS.l_x in
    let xn = T.rms_norm tn.T.rms_norm_eps x p.T.input_layernorm in
    rms_norm_eq tn.T.rms_norm_eps x p.T.input_layernorm xs w wo;
    attn_qkv_eq tn nm p.T.token_mixer xn cos sin w wo xs cs sn;
    let kn = SS.at_k nm w wo xs cs sn in
    let vn = SS.at_v nm w wo xs in
    let entry_k = T.attn_key tn p.T.token_mixer xn cos sin in
    let entry_v = T.attn_value p.T.token_mixer xn in
    let keys = T.append_position (ts.T.cache.T.key_cache g) t entry_k in
    let values = T.append_position (ts.T.cache.T.value_cache g) t entry_v in
    let hk = fun (p j:nat) -> SS.kv_get kvs p ((l / 4) * 1024 + j) in
    let hv = fun (p j:nat) -> SS.kv_get kvs p ((l / 4) * 1024 + 512 + j) in
    attn_keys_ok t l (ts.T.cache.T.key_cache g) entry_k kvs kn;
    attn_values_ok t l (ts.T.cache.T.value_cache g) entry_v kvs vn;
    self_attention_eq tn nm p.T.token_mixer xn cos sin t keys values w wo xs cs sn hk hv;
    decoder_output_eq nm p w wo x (T.self_attention tn p.T.token_mixer xn cos sin t keys values) xs
      (SS.matvec w (SS.at_mix wo + 5243392) 1024 2048 (SS.at_gated nm w wo xs cs sn t hk hv));
    let ts' = T.full_layer_step tn p g t cos sin ts in
    let cur' = SS.write_kv fs.SS.l_cur g kn vn in
    introduce forall (g':T.index T.full_layer_count) (kvh:T.index T.num_key_value_heads) (e:T.index T.head_dim).
      4 * g' + 3 < l + 1 ==> ts'.T.cache.T.key_cache g' t kvh e == SS.at cur' (g' * 1024 + kvh * 256 + e)
    with write_kv_at fs.SS.l_cur kn vn g g' (kvh * 256 + e);
    introduce forall (g':T.index T.full_layer_count) (kvh:T.index T.num_key_value_heads) (e:T.index T.head_dim).
      4 * g' + 3 < l + 1 ==> ts'.T.cache.T.value_cache g' t kvh e == SS.at cur' (g' * 1024 + 512 + kvh * 256 + e)
    with write_kv_at fs.SS.l_cur kn vn g g' (512 + kvh * 256 + e)

let rec run_layers_eq (tn:T.numerics) (nm:SS.numerics) (tw:T.model_weights real) (w:S.seq real) (t:nat)
                      (cos sin:T.vector real T.rotary_pairs) (cs sn:S.seq real)
                      (kvs:S.seq (S.seq real)) (ts:T.layer_state) (fs:SS.lstate) (l:nat)
  : Lemma (requires nm_ok tn nm /\ weights_ok tw w /\ l <= 24 /\ lstate_ok t l ts kvs fs /\
                    vec_ok cos cs 0 /\ vec_ok sin sn 0)
          (ensures lstate_ok t 24 (T.run_layers tn tw t cos sin ts l) kvs (SS.run_layers nm w t cs sn kvs fs l))
          (decreases 24 - l)
  = if l >= 24 then ()
    else begin
      run_layers_unfold nm w t cs sn kvs fs l;
      if l % 4 = 3 then begin
        full_step_eq tn nm tw w t cos sin cs sn kvs ts fs l;
        run_layers_eq tn nm tw w t cos sin cs sn kvs
          (T.full_layer_step tn (tw.T.full_layers (l / 4)) (l / 4) t cos sin ts) (full_next nm w t cs sn kvs fs l) (l + 1)
      end else begin
        let d : T.index T.linear_layer_count = 3 * (l / 4) + l % 4 in
        linear_step_eq tn nm tw w t kvs ts fs l;
        run_layers_eq tn nm tw w t cos sin cs sn kvs (T.linear_layer_step tn (tw.T.linear_layers d) d ts)
          (dn_next nm w fs l) (l + 1)
      end
    end

(*** Decode positions *)

let rstate_ok (n:nat) (c:T.cache_state) (rs:SS.rstate) : prop =
  dn_cache_ok c rs.SS.convs rs.SS.recs /\ S.length rs.SS.kvs == n /\ past_ok n c rs.SS.kvs

let kv_get_snoc (kvs:S.seq (S.seq real)) (cur:S.seq real) (pos:nat) (o:int)
  : Lemma (requires pos <= S.length kvs)
          (ensures SS.kv_get (S.snoc kvs cur) pos o == (if pos < S.length kvs then SS.kv_get kvs pos o else SS.at cur o))
  = ()

let rotary_eq (tn:T.numerics) (nm:SS.numerics) (t:int)
  : Lemma (requires nm_ok tn nm)
          (ensures vec_ok (T.rotary_cos tn t) (SS.rope_cos nm t) 0 /\ vec_ok (T.rotary_sin tn t) (SS.rope_sin nm t) 0)
  = ()

let step_eq (tn:T.numerics) (nm:SS.numerics) (tw:T.model_weights real) (w:S.seq real) (tok:int) (t:nat)
            (c:T.cache_state) (rs:SS.rstate)
  : Lemma (requires nm_ok tn nm /\ weights_ok tw w /\ T.valid_token tok /\ rstate_ok t c rs)
          (ensures (let ts = T.decode_step tn tw tok t c in
                    let r = SS.ref_step nm w tok t rs in
                    rstate_ok (t + 1) ts.T.cache (fst r) /\ vec_ok ts.T.hidden (snd r) 0))
  = let cos = T.rotary_cos tn t in
    let sin = T.rotary_sin tn t in
    let cs = SS.rope_cos nm t in
    let sn = SS.rope_sin nm t in
    rotary_eq tn nm t;
    let ts0 : T.layer_state = { T.cache = c; T.hidden = T.token_embedding tw tok } in
    let fs0 : SS.lstate = { SS.l_convs = rs.SS.convs; SS.l_recs = rs.SS.recs;
                            SS.l_cur = S.create 6144 0.0R; SS.l_x = SS.embed w tok } in
    assert (vec_ok ts0.T.hidden fs0.SS.l_x 0);
    run_layers_eq tn nm tw w t cos sin cs sn rs.SS.kvs ts0 fs0 0;
    let ts = T.run_layers tn tw t cos sin ts0 0 in
    let r = SS.run_layers nm w t cs sn rs.SS.kvs fs0 0 in
    let kvs' = S.snoc rs.SS.kvs r.SS.l_cur in
    introduce forall (g:T.index T.full_layer_count) (pos:nat) (kvh:T.index T.num_key_value_heads) (e:T.index T.head_dim).
      pos < t + 1 ==> ts.T.cache.T.key_cache g pos kvh e == SS.kv_get kvs' pos (g * 1024 + kvh * 256 + e)
    with introduce _ ==> _
    with kv_get_snoc rs.SS.kvs r.SS.l_cur pos (g * 1024 + kvh * 256 + e);
    introduce forall (g:T.index T.full_layer_count) (pos:nat) (kvh:T.index T.num_key_value_heads) (e:T.index T.head_dim).
      pos < t + 1 ==> ts.T.cache.T.value_cache g pos kvh e == SS.kv_get kvs' pos (g * 1024 + 512 + kvh * 256 + e)
    with introduce _ ==> _
    with kv_get_snoc rs.SS.kvs r.SS.l_cur pos (g * 1024 + 512 + kvh * 256 + e)

let init_state_ok () : Lemma (rstate_ok 0 T.empty_cache SS.init_state) = ()

let rec decode_eq (tn:T.numerics) (nm:SS.numerics) (tw:T.model_weights real) (w:S.seq real)
                  (toks:nat -> GTot int) (ftoks:nat -> int) (n:nat)
  : Lemma (requires nm_ok tn nm /\ weights_ok tw w /\
                    (forall (p:nat). p < n ==> toks p == ftoks p) /\
                    (forall (p:nat). p < n ==> T.valid_token (ftoks p)))
          (ensures rstate_ok n (T.decode tn tw toks n).T.cache (fst (SS.ref_run nm w ftoks n)) /\
                   (n > 0 ==> vec_ok (T.decode tn tw toks n).T.hidden (snd (SS.ref_run nm w ftoks n)) 0))
          (decreases n)
  = if n = 0 then init_state_ok ()
    else begin
      decode_eq tn nm tw w toks ftoks (n - 1);
      step_eq tn nm tw w (ftoks (n - 1)) (n - 1) (T.decode tn tw toks (n - 1)).T.cache
        (fst (SS.ref_run nm w ftoks (n - 1)))
    end

let hidden_eq (tn:T.numerics) (nm:SS.numerics) (tw:T.model_weights real) (w:S.seq real)
              (toks:nat -> GTot int) (ftoks:nat -> int) (p:nat)
  : Lemma (requires nm_ok tn nm /\ weights_ok tw w /\
                    (forall (q:nat). q <= p ==> toks q == ftoks q) /\
                    (forall (q:nat). q <= p ==> T.valid_token (ftoks q)))
          (ensures vec_ok (T.hidden_at tn tw toks p) (SS.ref_hidden nm w ftoks p) 0)
  = decode_eq tn nm tw w toks ftoks (p + 1)

let logits_eq (tn:T.numerics) (nm:SS.numerics) (tw:T.model_weights real) (w:S.seq real)
              (h:T.vector real T.hidden_size) (hs:S.seq real)
  : Lemma (requires nm_ok tn nm /\ weights_ok tw w /\ vec_ok h hs 0)
          (ensures vec_ok (T.logits tn tw h) (SS.ref_logits nm w hs) 0)
  = let hn = T.rms_norm tn.T.rms_norm_eps h tw.T.final_norm in
    rms_norm_eq tn.T.rms_norm_eps h tw.T.final_norm hs w SS.final_norm_off;
    let hns = SS.rmsnorm1p nm.SS.eps hs 1024 w SS.final_norm_off in
    introduce forall (m:T.index T.vocab_size). T.logits tn tw h m == SS.at (SS.ref_logits nm w hs) (0 + m)
    with dot_eq (tw.T.embed_tokens m) hn w (0 + m * 1024) hns

(*** Views of the flat program buffers (layout only) *)

let elem (#a:Type0) (d:a) (s:S.seq a) (i:int) : a =
  if 0 <= i && i < S.length s then S.index s i else d

let vec_view (#a:Type0) (e:int -> GTot a) (o:int) (n:nat) : T.vector a n =
  fun i -> e (o + i)

let mat_view (#a:Type0) (e:int -> GTot a) (o:int) (r c:nat) : T.matrix a r c =
  fun i j -> e (o + i * c + j)

let mlp_view (#a:Type0) (e:int -> GTot a) (wo:int) : T.mlp_params a = {
  T.gate_proj = mat_view e (wo + 2048) T.intermediate_size T.hidden_size;
  T.up_proj = mat_view e (wo + 3672064) T.intermediate_size T.hidden_size;
  T.down_proj = mat_view e (wo + 7342080) T.hidden_size T.intermediate_size;
}

let linear_attn_view (#a:Type0) (e:int -> GTot a) (mo:int) : T.linear_attn_params a = {
  T.in_proj_qkv = mat_view e mo T.conv_dim T.hidden_size;
  T.in_proj_z = mat_view e (mo + 6291456) T.linear_key_dim T.hidden_size;
  T.in_proj_b = mat_view e (mo + 8388608) T.linear_num_heads T.hidden_size;
  T.in_proj_a = mat_view e (mo + 8404992) T.linear_num_heads T.hidden_size;
  T.conv1d = mat_view e (mo + 8421376) T.conv_dim T.conv_kernel_size;
  T.a_log = vec_view e (mo + 8445952) T.linear_num_heads;
  T.dt_bias = vec_view e (mo + 8445968) T.linear_num_heads;
  T.norm = vec_view e (mo + 8445984) T.linear_head_dim;
  T.out_proj = mat_view e (mo + 8446112) T.hidden_size T.linear_key_dim;
}

let self_attn_view (#a:Type0) (e:int -> GTot a) (mo:int) : T.self_attn_params a = {
  T.q_proj = mat_view e mo T.q_proj_dim T.hidden_size;
  T.k_proj = mat_view e (mo + 4194304) T.kv_proj_dim T.hidden_size;
  T.v_proj = mat_view e (mo + 4718592) T.kv_proj_dim T.hidden_size;
  T.q_norm = vec_view e (mo + 5242880) T.head_dim;
  T.k_norm = vec_view e (mo + 5243136) T.head_dim;
  T.o_proj = mat_view e (mo + 5243392) T.hidden_size T.attn_out_dim;
}

let linear_layer_view (#a:Type0) (e:int -> GTot a) (wo:int) : T.decoder_layer T.linear_attn_params a = {
  T.input_layernorm = vec_view e wo T.hidden_size;
  T.post_attention_layernorm = vec_view e (wo + 1024) T.hidden_size;
  T.mlp = mlp_view e wo;
  T.token_mixer = linear_attn_view e (wo + 11012096);
}

let full_layer_view (#a:Type0) (e:int -> GTot a) (wo:int) : T.decoder_layer T.self_attn_params a = {
  T.input_layernorm = vec_view e wo T.hidden_size;
  T.post_attention_layernorm = vec_view e (wo + 1024) T.hidden_size;
  T.mlp = mlp_view e wo;
  T.token_mixer = self_attn_view e (wo + 11012096);
}

(* the packed weights as a typed record (LAYOUT.md offsets) *)
let weights_view (#a:Type0) (e:int -> GTot a) : T.model_weights a = {
  T.embed_tokens = mat_view e 0 T.vocab_size T.hidden_size;
  T.final_norm = vec_view e SS.final_norm_off T.hidden_size;
  T.linear_layers = (fun d -> linear_layer_view e (SS.layer_off (4 * (d / 3) + d % 3)));
  T.full_layers = (fun g -> full_layer_view e (SS.layer_off (4 * g + 3)));
}

let inputs_of (w:S.seq BF.t) (prompt:S.seq I64.t) (steps:int) : T.inputs = {
  T.weights = weights_view (elem zero w);
  T.prompt_length = S.length prompt;
  T.prompt = (fun i -> I64.v (S.index prompt i));
  T.decode_steps = steps;
}

let outputs_of (tokens:S.seq I64.t) (hidden:S.seq BF.t) : T.outputs = {
  T.generated_count = S.length tokens;
  T.tokens = (fun j -> I64.v (S.index tokens j));
  T.hidden_states = (fun j i -> elem zero hidden (j * 1024 + i));
}

(*** Packing typed weights into the flat layout *)

let vec_rd (#a:Type0) (z:a) (#n:nat) (v:T.vector a n) (k:int) : GTot a =
  if 0 <= k && k < n then v k else z

let mat_rd (#a:Type0) (z:a) (#r #c:nat) (m:T.matrix a r c) (k:int) : GTot a =
  if 0 <= k && c > 0 && k / c < r then m (k / c) (k % c) else z

let common_rd (#m:Type0 -> Type0) (#a:Type0) (z:a) (p:T.decoder_layer m a) (k:int) : GTot a =
  if k < 1024 then vec_rd z p.T.input_layernorm k
  else if k < 2048 then vec_rd z p.T.post_attention_layernorm (k - 1024)
  else if k < 3672064 then mat_rd z p.T.mlp.T.gate_proj (k - 2048)
  else if k < 7342080 then mat_rd z p.T.mlp.T.up_proj (k - 3672064)
  else mat_rd z p.T.mlp.T.down_proj (k - 7342080)

let linear_attn_rd (#a:Type0) (z:a) (p:T.linear_attn_params a) (k:int) : GTot a =
  if k < 6291456 then mat_rd z p.T.in_proj_qkv k
  else if k < 8388608 then mat_rd z p.T.in_proj_z (k - 6291456)
  else if k < 8404992 then mat_rd z p.T.in_proj_b (k - 8388608)
  else if k < 8421376 then mat_rd z p.T.in_proj_a (k - 8404992)
  else if k < 8445952 then mat_rd z p.T.conv1d (k - 8421376)
  else if k < 8445968 then vec_rd z p.T.a_log (k - 8445952)
  else if k < 8445984 then vec_rd z p.T.dt_bias (k - 8445968)
  else if k < 8446112 then vec_rd z p.T.norm (k - 8445984)
  else mat_rd z p.T.out_proj (k - 8446112)

let self_attn_rd (#a:Type0) (z:a) (p:T.self_attn_params a) (k:int) : GTot a =
  if k < 4194304 then mat_rd z p.T.q_proj k
  else if k < 4718592 then mat_rd z p.T.k_proj (k - 4194304)
  else if k < 5242880 then mat_rd z p.T.v_proj (k - 4718592)
  else if k < 5243136 then vec_rd z p.T.q_norm (k - 5242880)
  else if k < 5243392 then vec_rd z p.T.k_norm (k - 5243136)
  else mat_rd z p.T.o_proj (k - 5243392)

let linear_layer_rd (#a:Type0) (z:a) (p:T.decoder_layer T.linear_attn_params a) (k:int) : GTot a =
  if k < 11012096 then common_rd z p k else linear_attn_rd z p.T.token_mixer (k - 11012096)

let full_layer_rd (#a:Type0) (z:a) (p:T.decoder_layer T.self_attn_params a) (k:int) : GTot a =
  if k < 11012096 then common_rd z p k else self_attn_rd z p.T.token_mixer (k - 11012096)

(* element o of the packed layout of tw (z outside the layout) *)
let flat_read (#a:Type0) (z:a) (tw:T.model_weights a) (o:int) : GTot a =
  if o < SS.layers_off then
    (if o < SS.table_len then mat_rd z tw.T.embed_tokens o
     else vec_rd z tw.T.final_norm (o - SS.final_norm_off))
  else
    let r = o - SS.layers_off in
    let grp = r / SS.group_len in
    let rg = r % SS.group_len in
    if grp >= 6 then z
    else if rg < 3 * SS.dn_len then
      (let q = rg / SS.dn_len in
       if q < 3 then linear_layer_rd z (tw.T.linear_layers (3 * grp + q)) (rg % SS.dn_len) else z)
    else full_layer_rd z (tw.T.full_layers grp) (rg - 3 * SS.dn_len)

let flat_of (tw:T.model_weights real) : GTot (S.seq real) =
  S.init_ghost SS.weights_len (fun (o:nat{o < SS.weights_len}) -> flat_read 0.0R tw o)

let div_lt (k:int) (r:nat) (c:pos)
  : Lemma (requires 0 <= k /\ k < r * c) (ensures k / c < r)
  = if k / c >= r then begin
      ML.lemma_mult_le_right c r (k / c); ML.euclidean_division_definition k c; ML.lemma_mod_lt k c
    end

(** reading a view back *)

let vec_rd_view (#a:Type0) (z:a) (e:int -> GTot a) (o:int) (n:nat) (k:int)
  : Lemma (requires 0 <= k /\ k < n) (ensures vec_rd z (vec_view e o n) k == e (o + k))
  = ()

let mat_rd_view (#a:Type0) (z:a) (e:int -> GTot a) (o:int) (r c:nat) (k:int)
  : Lemma (requires 0 <= k /\ k < r * c) (ensures mat_rd z (mat_view e o r c) k == e (o + k))
  = if c = 0 then () else begin ML.euclidean_division_definition k c; div_lt k r c end

let common_rd_view (#m:Type0 -> Type0) (#a:Type0) (z:a) (p:T.decoder_layer m a) (e:int -> GTot a) (wo:int) (k:int)
  : Lemma (requires p.T.input_layernorm == vec_view e wo T.hidden_size /\
                    p.T.post_attention_layernorm == vec_view e (wo + 1024) T.hidden_size /\
                    p.T.mlp == mlp_view e wo /\ 0 <= k /\ k < 11012096)
          (ensures common_rd z p k == e (wo + k))
  = if k < 1024 then vec_rd_view z e wo T.hidden_size k
    else if k < 2048 then vec_rd_view z e (wo + 1024) T.hidden_size (k - 1024)
    else if k < 3672064 then mat_rd_view z e (wo + 2048) T.intermediate_size T.hidden_size (k - 2048)
    else if k < 7342080 then mat_rd_view z e (wo + 3672064) T.intermediate_size T.hidden_size (k - 3672064)
    else mat_rd_view z e (wo + 7342080) T.hidden_size T.intermediate_size (k - 7342080)

let linear_attn_rd_view (#a:Type0) (z:a) (e:int -> GTot a) (mo:int) (k:int)
  : Lemma (requires 0 <= k /\ k < 10543264)
          (ensures linear_attn_rd z (linear_attn_view e mo) k == e (mo + k))
  = if k < 6291456 then mat_rd_view z e mo T.conv_dim T.hidden_size k
    else if k < 8388608 then mat_rd_view z e (mo + 6291456) T.linear_key_dim T.hidden_size (k - 6291456)
    else if k < 8404992 then mat_rd_view z e (mo + 8388608) T.linear_num_heads T.hidden_size (k - 8388608)
    else if k < 8421376 then mat_rd_view z e (mo + 8404992) T.linear_num_heads T.hidden_size (k - 8404992)
    else if k < 8445952 then mat_rd_view z e (mo + 8421376) T.conv_dim T.conv_kernel_size (k - 8421376)
    else if k < 8445968 then vec_rd_view z e (mo + 8445952) T.linear_num_heads (k - 8445952)
    else if k < 8445984 then vec_rd_view z e (mo + 8445968) T.linear_num_heads (k - 8445968)
    else if k < 8446112 then vec_rd_view z e (mo + 8445984) T.linear_head_dim (k - 8445984)
    else mat_rd_view z e (mo + 8446112) T.hidden_size T.linear_key_dim (k - 8446112)

let self_attn_rd_view (#a:Type0) (z:a) (e:int -> GTot a) (mo:int) (k:int)
  : Lemma (requires 0 <= k /\ k < 7340544)
          (ensures self_attn_rd z (self_attn_view e mo) k == e (mo + k))
  = if k < 4194304 then mat_rd_view z e mo T.q_proj_dim T.hidden_size k
    else if k < 4718592 then mat_rd_view z e (mo + 4194304) T.kv_proj_dim T.hidden_size (k - 4194304)
    else if k < 5242880 then mat_rd_view z e (mo + 4718592) T.kv_proj_dim T.hidden_size (k - 4718592)
    else if k < 5243136 then vec_rd_view z e (mo + 5242880) T.head_dim (k - 5242880)
    else if k < 5243392 then vec_rd_view z e (mo + 5243136) T.head_dim (k - 5243136)
    else mat_rd_view z e (mo + 5243392) T.hidden_size T.attn_out_dim (k - 5243392)

let linear_layer_rd_view (#a:Type0) (z:a) (e:int -> GTot a) (wo:int) (k:int)
  : Lemma (requires 0 <= k /\ k < SS.dn_len)
          (ensures linear_layer_rd z (linear_layer_view e wo) k == e (wo + k))
  = if k < 11012096 then common_rd_view z (linear_layer_view e wo) e wo k
    else linear_attn_rd_view z e (wo + 11012096) (k - 11012096)

let full_layer_rd_view (#a:Type0) (z:a) (e:int -> GTot a) (wo:int) (k:int)
  : Lemma (requires 0 <= k /\ k < SS.attn_len)
          (ensures full_layer_rd z (full_layer_view e wo) k == e (wo + k))
  = if k < 11012096 then common_rd_view z (full_layer_view e wo) e wo k
    else self_attn_rd_view z e (wo + 11012096) (k - 11012096)

let flat_read_view (#a:Type0) (z:a) (e:int -> GTot a) (o:int)
  : Lemma (requires 0 <= o /\ o < SS.weights_len) (ensures flat_read z (weights_view e) o == e o)
  = if o < SS.layers_off then begin
      if o < SS.table_len then mat_rd_view z e 0 T.vocab_size T.hidden_size o
      else vec_rd_view z e SS.final_norm_off T.hidden_size (o - SS.final_norm_off)
    end else begin
      let r = o - SS.layers_off in
      let grp = r / SS.group_len in
      let rg = r % SS.group_len in
      ML.euclidean_division_definition r SS.group_len;
      ML.lemma_mod_lt r SS.group_len;
      div_lt r 6 SS.group_len;
      if rg < 3 * SS.dn_len then begin
        let q = rg / SS.dn_len in
        ML.euclidean_division_definition rg SS.dn_len;
        ML.lemma_mod_lt rg SS.dn_len;
        div_lt rg 3 SS.dn_len;
        divmod grp q 3;
        divmod grp q 4;
        linear_layer_rd_view z e (SS.layer_off (4 * grp + q)) (rg % SS.dn_len)
      end else begin
        divmod grp 3 4;
        full_layer_rd_view z e (SS.layer_off (4 * grp + 3)) (rg - 3 * SS.dn_len)
      end
    end

(** reading approximating records *)

let zero_bf () : Lemma ((zero #BF.t) %~ 0.0R) = a0 #BF.t

let vec_rd_approx (#n:nat) (x:T.vector BF.t n) (r:T.vector real n) (k:int)
  : Lemma (requires T.vector_approximates x r) (ensures vec_rd zero x k %~ vec_rd 0.0R r k)
  = zero_bf ()

let mat_rd_approx (#rows #cols:nat) (x:T.matrix BF.t rows cols) (r:T.matrix real rows cols) (k:int)
  : Lemma (requires T.matrix_approximates x r) (ensures mat_rd zero x k %~ mat_rd 0.0R r k)
  = zero_bf ()

let common_rd_approx (#m:Type0 -> Type0) (x:T.decoder_layer m BF.t) (r:T.decoder_layer m real) (k:int)
  : Lemma (requires T.vector_approximates x.T.input_layernorm r.T.input_layernorm /\
                    T.vector_approximates x.T.post_attention_layernorm r.T.post_attention_layernorm /\
                    T.mlp_approximates x.T.mlp r.T.mlp)
          (ensures common_rd zero x k %~ common_rd 0.0R r k)
  = vec_rd_approx x.T.input_layernorm r.T.input_layernorm k;
    vec_rd_approx x.T.post_attention_layernorm r.T.post_attention_layernorm (k - 1024);
    mat_rd_approx x.T.mlp.T.gate_proj r.T.mlp.T.gate_proj (k - 2048);
    mat_rd_approx x.T.mlp.T.up_proj r.T.mlp.T.up_proj (k - 3672064);
    mat_rd_approx x.T.mlp.T.down_proj r.T.mlp.T.down_proj (k - 7342080)

let linear_attn_rd_approx (x:T.linear_attn_params BF.t) (r:T.linear_attn_params real) (k:int)
  : Lemma (requires T.linear_attn_approximates x r)
          (ensures linear_attn_rd zero x k %~ linear_attn_rd 0.0R r k)
  = mat_rd_approx x.T.in_proj_qkv r.T.in_proj_qkv k;
    mat_rd_approx x.T.in_proj_z r.T.in_proj_z (k - 6291456);
    mat_rd_approx x.T.in_proj_b r.T.in_proj_b (k - 8388608);
    mat_rd_approx x.T.in_proj_a r.T.in_proj_a (k - 8404992);
    mat_rd_approx x.T.conv1d r.T.conv1d (k - 8421376);
    vec_rd_approx x.T.a_log r.T.a_log (k - 8445952);
    vec_rd_approx x.T.dt_bias r.T.dt_bias (k - 8445968);
    vec_rd_approx x.T.norm r.T.norm (k - 8445984);
    mat_rd_approx x.T.out_proj r.T.out_proj (k - 8446112)

let self_attn_rd_approx (x:T.self_attn_params BF.t) (r:T.self_attn_params real) (k:int)
  : Lemma (requires T.self_attn_approximates x r)
          (ensures self_attn_rd zero x k %~ self_attn_rd 0.0R r k)
  = mat_rd_approx x.T.q_proj r.T.q_proj k;
    mat_rd_approx x.T.k_proj r.T.k_proj (k - 4194304);
    mat_rd_approx x.T.v_proj r.T.v_proj (k - 4718592);
    vec_rd_approx x.T.q_norm r.T.q_norm (k - 5242880);
    vec_rd_approx x.T.k_norm r.T.k_norm (k - 5243136);
    mat_rd_approx x.T.o_proj r.T.o_proj (k - 5243392)

let flat_read_approx (x:T.model_weights BF.t) (r:T.model_weights real) (o:int)
  : Lemma (requires T.weights_approximates x r) (ensures flat_read zero x o %~ flat_read 0.0R r o)
  = zero_bf ();
    mat_rd_approx x.T.embed_tokens r.T.embed_tokens o;
    vec_rd_approx x.T.final_norm r.T.final_norm (o - SS.final_norm_off);
    let rr = o - SS.layers_off in
    let grp = rr / SS.group_len in
    let rg = rr % SS.group_len in
    if o >= SS.layers_off && grp < 6 then begin
      if rg < 3 * SS.dn_len then begin
        let q = rg / SS.dn_len in
        if q < 3 then begin
          let k = rg % SS.dn_len in
          let xl = x.T.linear_layers (3 * grp + q) in
          let rl = r.T.linear_layers (3 * grp + q) in
          common_rd_approx xl rl k;
          linear_attn_rd_approx xl.T.token_mixer rl.T.token_mixer (k - 11012096)
        end
      end else begin
        let k = rg - 3 * SS.dn_len in
        let xl = x.T.full_layers grp in
        let rl = r.T.full_layers grp in
        common_rd_approx xl rl k;
        self_attn_rd_approx xl.T.token_mixer rl.T.token_mixer (k - 11012096)
      end
    end

(** the packed typed weights agree with the typed weights *)

let vec_rd_at (#a:Type0) (z:a) (#n:nat) (v:T.vector a n) (i:T.index n)
  : Lemma (vec_rd z v i == v i) = ()

let mat_rd_at (#a:Type0) (z:a) (#r #c:nat) (m:T.matrix a r c) (i:T.index r) (j:T.index c)
  : Lemma (mat_rd z m (i * c + j) == m i j) = divmod i j c

let slots_ok (w:S.seq real) (wo:int) (n:nat) (rd:int -> GTot real) : prop =
  forall (k:nat). k < n ==> SS.at w (wo + k) == rd k

let mat_of_slots (#r #c:nat) (m:T.matrix real r c) (w:S.seq real) (wo:int) (n:nat) (rd:int -> GTot real)
                 (fo:nat) (o:int)
  : Lemma (requires slots_ok w wo n rd /\ fo + r * c <= n /\ o == wo + fo /\
                    (forall (k:nat). k < r * c ==> rd (fo + k) == mat_rd 0.0R m k))
          (ensures mat_ok m w o)
  = introduce forall (i:T.index r) (j:T.index c). m i j == SS.at w (o + i * c + j)
    with begin
      ML.lemma_mult_le_right c (i + 1) r;
      ML.distributivity_add_left i 1 c;
      mat_rd_at 0.0R m i j;
      assert (rd (fo + (i * c + j)) == mat_rd 0.0R m (i * c + j));
      assert (SS.at w (wo + (fo + (i * c + j))) == rd (fo + (i * c + j)))
    end

let vec_of_slots (#n':nat) (v:T.vector real n') (w:S.seq real) (wo:int) (n:nat) (rd:int -> GTot real)
                 (fo:nat) (o:int)
  : Lemma (requires slots_ok w wo n rd /\ fo + n' <= n /\ o == wo + fo /\
                    (forall (k:nat). k < n' ==> rd (fo + k) == v k))
          (ensures vec_ok v w o)
  = introduce forall (i:T.index n'). v i == SS.at w (o + i)
    with assert (SS.at w (wo + (fo + i)) == rd (fo + i))

let common_of_slots (#m:Type0 -> Type0) (p:T.decoder_layer m real) (w:S.seq real) (wo:int)
  : Lemma (requires slots_ok w wo 11012096 (common_rd 0.0R p))
          (ensures common_ok p w wo)
  = let rd = common_rd 0.0R p in
    let n = 11012096 in
    vec_of_slots p.T.input_layernorm w wo n rd 0 wo;
    vec_of_slots p.T.post_attention_layernorm w wo n rd 1024 (wo + 1024);
    mat_of_slots p.T.mlp.T.gate_proj w wo n rd 2048 (wo + 2048);
    mat_of_slots p.T.mlp.T.up_proj w wo n rd 3672064 (wo + 3672064);
    mat_of_slots p.T.mlp.T.down_proj w wo n rd 7342080 (wo + 7342080)

let linear_attn_of_slots (q:T.linear_attn_params real) (w:S.seq real) (mo:int)
  : Lemma (requires slots_ok w mo 10543264 (linear_attn_rd 0.0R q))
          (ensures linear_attn_ok q w mo)
  = let rd = linear_attn_rd 0.0R q in
    let n = 10543264 in
    mat_of_slots q.T.in_proj_qkv w mo n rd 0 mo;
    mat_of_slots q.T.in_proj_z w mo n rd 6291456 (mo + 6291456);
    mat_of_slots q.T.in_proj_b w mo n rd 8388608 (mo + 8388608);
    mat_of_slots q.T.in_proj_a w mo n rd 8404992 (mo + 8404992);
    mat_of_slots q.T.conv1d w mo n rd 8421376 (mo + 8421376);
    vec_of_slots q.T.a_log w mo n rd 8445952 (mo + 8445952);
    vec_of_slots q.T.dt_bias w mo n rd 8445968 (mo + 8445968);
    vec_of_slots q.T.norm w mo n rd 8445984 (mo + 8445984);
    mat_of_slots q.T.out_proj w mo n rd 8446112 (mo + 8446112)

let self_attn_of_slots (q:T.self_attn_params real) (w:S.seq real) (mo:int)
  : Lemma (requires slots_ok w mo 7340544 (self_attn_rd 0.0R q))
          (ensures self_attn_ok q w mo)
  = let rd = self_attn_rd 0.0R q in
    let n = 7340544 in
    mat_of_slots q.T.q_proj w mo n rd 0 mo;
    mat_of_slots q.T.k_proj w mo n rd 4194304 (mo + 4194304);
    mat_of_slots q.T.v_proj w mo n rd 4718592 (mo + 4718592);
    vec_of_slots q.T.q_norm w mo n rd 5242880 (mo + 5242880);
    vec_of_slots q.T.k_norm w mo n rd 5243136 (mo + 5243136);
    mat_of_slots q.T.o_proj w mo n rd 5243392 (mo + 5243392)

let linear_layer_of_slots (p:T.decoder_layer T.linear_attn_params real) (w:S.seq real) (wo:int)
  : Lemma (requires slots_ok w wo SS.dn_len (linear_layer_rd 0.0R p))
          (ensures linear_layer_ok p w wo)
  = introduce forall (k:nat). k < 11012096 ==> SS.at w (wo + k) == common_rd 0.0R p k
    with introduce _ ==> _ with assert (SS.at w (wo + k) == linear_layer_rd 0.0R p k);
    common_of_slots p w wo;
    introduce forall (k:nat). k < 10543264 ==>
      SS.at w ((wo + 11012096) + k) == linear_attn_rd 0.0R p.T.token_mixer k
    with introduce _ ==> _ with
      assert (SS.at w (wo + (11012096 + k)) == linear_layer_rd 0.0R p (11012096 + k));
    linear_attn_of_slots p.T.token_mixer w (wo + 11012096)

let full_layer_of_slots (p:T.decoder_layer T.self_attn_params real) (w:S.seq real) (wo:int)
  : Lemma (requires slots_ok w wo SS.attn_len (full_layer_rd 0.0R p))
          (ensures full_layer_ok p w wo)
  = introduce forall (k:nat). k < 11012096 ==> SS.at w (wo + k) == common_rd 0.0R p k
    with introduce _ ==> _ with assert (SS.at w (wo + k) == full_layer_rd 0.0R p k);
    common_of_slots p w wo;
    introduce forall (k:nat). k < 7340544 ==>
      SS.at w ((wo + 11012096) + k) == self_attn_rd 0.0R p.T.token_mixer k
    with introduce _ ==> _ with
      assert (SS.at w (wo + (11012096 + k)) == full_layer_rd 0.0R p (11012096 + k));
    self_attn_of_slots p.T.token_mixer w (wo + 11012096)

let flat_of_at (tw:T.model_weights real) (o:int)
  : Lemma (requires 0 <= o /\ o < SS.weights_len) (ensures SS.at (flat_of tw) o == flat_read 0.0R tw o)
  = ()

let linear_slot (tw:T.model_weights real) (d:T.index T.linear_layer_count) (k:nat)
  : Lemma (requires k < SS.dn_len)
          (ensures SS.at (flat_of tw) (SS.layer_off (4 * (d / 3) + d % 3) + k) ==
                   linear_layer_rd 0.0R (tw.T.linear_layers d) k)
  = let g = d / 3 in
    let q = d % 3 in
    ML.euclidean_division_definition d 3;
    divmod g q 4;
    divmod g (SS.dn_len * q + k) SS.group_len;
    divmod q k SS.dn_len;
    flat_of_at tw (SS.layer_off (4 * (d / 3) + d % 3) + k)

let full_slot (tw:T.model_weights real) (g:T.index T.full_layer_count) (k:nat)
  : Lemma (requires k < SS.attn_len)
          (ensures SS.at (flat_of tw) (SS.layer_off (4 * g + 3) + k) == full_layer_rd 0.0R (tw.T.full_layers g) k)
  = divmod g 3 4;
    divmod g (3 * SS.dn_len + k) SS.group_len;
    flat_of_at tw (SS.layer_off (4 * g + 3) + k)

let flat_of_ok (tw:T.model_weights real) : Lemma (weights_ok tw (flat_of tw))
  = let fw = flat_of tw in
    introduce forall (i:T.index T.vocab_size) (j:T.index T.hidden_size).
      tw.T.embed_tokens i j == SS.at fw (0 + i * 1024 + j)
    with (flat_of_at tw (i * 1024 + j); mat_rd_at 0.0R tw.T.embed_tokens i j);
    introduce forall (i:T.index T.hidden_size). tw.T.final_norm i == SS.at fw (SS.final_norm_off + i)
    with flat_of_at tw (SS.final_norm_off + i);
    introduce forall (d:T.index T.linear_layer_count).
      linear_layer_ok (tw.T.linear_layers d) fw (SS.layer_off (4 * (d / 3) + d % 3))
    with begin
      introduce forall (k:nat). k < SS.dn_len ==>
        SS.at fw (SS.layer_off (4 * (d / 3) + d % 3) + k) == linear_layer_rd 0.0R (tw.T.linear_layers d) k
      with introduce _ ==> _ with linear_slot tw d k;
      linear_layer_of_slots (tw.T.linear_layers d) fw (SS.layer_off (4 * (d / 3) + d % 3))
    end;
    introduce forall (g:T.index T.full_layer_count).
      full_layer_ok (tw.T.full_layers g) fw (SS.layer_off (4 * g + 3))
    with begin
      introduce forall (k:nat). k < SS.attn_len ==>
        SS.at fw (SS.layer_off (4 * g + 3) + k) == full_layer_rd 0.0R (tw.T.full_layers g) k
      with introduce _ ==> _ with full_slot tw g k;
      full_layer_of_slots (tw.T.full_layers g) fw (SS.layer_off (4 * g + 3))
    end

(** numerics *)

let tn_of (nm:SS.numerics) : T.numerics = {
  T.rms_norm_eps = nm.SS.eps;
  T.linear_query_scale = nm.SS.dn_qscale;
  T.inv_freq = (fun i -> SS.at nm.SS.inv_freq i);
}

let nm_of (tn:T.numerics) : GTot SS.numerics = {
  SS.eps = tn.T.rms_norm_eps;
  SS.dn_qscale = tn.T.linear_query_scale;
  SS.inv_freq = S.init_ghost 32 tn.T.inv_freq;
}

let literal_eq (i:nat) : Lemma (SS.base_freq_literal i == T.inv_freq_f32_literal i) = ()

let freq_f32_at (s:S.seq real) (i:T.index T.rotary_pairs)
  : Lemma (requires s == SS.inv_freq_f32) (ensures SS.at s i == T.inv_freq_f32 i) = literal_eq i

let tn_of_ok (nm:SS.numerics)
  : Lemma (requires SS.numerics_ok nm) (ensures T.numerics_ok (tn_of nm) /\ nm_ok (tn_of nm) nm)
  = let s = nm.SS.inv_freq in
    introduce forall (i:T.index T.rotary_pairs). SS.at s i == T.inv_freq_f32 i
    with freq_f32_at s i

let seq_of_freq (f:T.vector real T.rotary_pairs) (g:T.vector real T.rotary_pairs) (s:S.seq real)
  : Lemma (requires S.length s == 32 /\ (forall (i:T.index T.rotary_pairs). f i == g i) /\
                    (forall (i:T.index T.rotary_pairs). S.index s i == g i))
          (ensures S.init_ghost 32 f == s)
  = S.lemma_eq_intro (S.init_ghost 32 f) s

let nm_of_ok (tn:T.numerics)
  : Lemma (requires T.numerics_ok tn) (ensures SS.numerics_ok (nm_of tn) /\ nm_ok tn (nm_of tn))
  = let f = tn.T.inv_freq in
    introduce forall (i:T.index T.rotary_pairs). S.index SS.inv_freq_f32 i == T.inv_freq_f32 i
    with literal_eq i;
    seq_of_freq f T.inv_freq_f32 SS.inv_freq_f32

(** lengths of the flat reference values *)

let rec run_layers_len (nm:SS.numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real) (kvs:S.seq (S.seq real))
                       (st:SS.lstate) (l:nat)
  : Lemma (requires l < 24) (ensures S.length (SS.run_layers nm w t cs sn kvs st l).SS.l_x == 1024)
          (decreases 24 - l)
  = run_layers_unfold nm w t cs sn kvs st l;
    if l + 1 < 24 then
      run_layers_len nm w t cs sn kvs
        (if l % 4 = 3 then full_next nm w t cs sn kvs st l else dn_next nm w st l) (l + 1)

let ref_hidden_len (nm:SS.numerics) (w:S.seq real) (toks:nat -> int) (p:nat)
  : Lemma (S.length (SS.ref_hidden nm w toks p) == 1024)
  = let st = fst (SS.ref_run nm w toks p) in
    run_layers_len nm w p (SS.rope_cos nm p) (SS.rope_sin nm p) st.SS.kvs
      ({ SS.l_convs = st.SS.convs; SS.l_recs = st.SS.recs;
         SS.l_cur = S.create 6144 0.0R; SS.l_x = SS.embed w (toks p) }) 0

(** selection rows *)

let seq_vec (#t:Type0) (n:nat) (row:S.seq t{S.length row == n}) : T.vector t n =
  fun k -> S.index row k

let allowed_init_iff (#t:Type0) {| F.floating t |} (n:nat) (v:T.vector t n) (sel:int)
  : Lemma (SS.allowed_choice (S.init_ghost n v) sel <==> T.allowed_choice v sel)
  = ()

let seq_vec_init (#t:Type0) (n:nat) (row:S.seq t{S.length row == n})
  : Lemma (S.init_ghost n (seq_vec n row) == row)
  = S.lemma_eq_intro (S.init_ghost n (seq_vec n row)) row

let allowed_iff (#t:Type0) {| F.floating t |} (n:nat) (row:S.seq t{S.length row == n}) (sel:int)
  : Lemma (SS.allowed_choice row sel <==> T.allowed_choice (seq_vec n row) sel)
  = seq_vec_init n row; allowed_init_iff n (seq_vec n row) sel

let choice_fwd (#t:Type0) {| Kuiper.Scalars.scalar t, real_like t, F.floating t |}
    (lr:T.vector real T.vocab_size) (ls:S.seq real) (sel:int) (row:S.seq t)
  : Lemma (requires S.length ls == 248320 /\ vec_ok lr ls 0 /\ row %~ ls /\ SS.allowed_choice row sel)
          (ensures exists (v:T.vector t T.vocab_size). T.vector_approximates v lr /\ T.allowed_choice v sel)
  = allowed_iff 248320 row sel;
    assert (T.vector_approximates (seq_vec 248320 row) lr)

let choice_bwd (#t:Type0) {| Kuiper.Scalars.scalar t, real_like t, F.floating t |}
    (lr:T.vector real T.vocab_size) (ls:S.seq real) (sel:int) (v:T.vector t T.vocab_size)
  : Lemma (requires S.length ls == 248320 /\ vec_ok lr ls 0 /\ T.vector_approximates v lr /\ T.allowed_choice v sel)
          (ensures exists (row:S.seq t). row %~ ls /\ SS.allowed_choice row sel)
  = allowed_init_iff 248320 v sel;
    assert (S.init_ghost 248320 v %~ ls)

let selection_iff (tn:T.numerics) (nm:SS.numerics) (tw:T.model_weights real) (rw:S.seq real)
                  (h:T.vector real T.hidden_size) (hs:S.seq real) (sel:int)
  : Lemma (requires vec_ok (T.logits tn tw h) (SS.ref_logits nm rw hs) 0)
          (ensures SS.selected_ok nm rw hs sel <==> T.greedy_choice_relation tn tw h sel)
  = let lr = T.logits tn tw h in
    let ls = SS.ref_logits nm rw hs in
    FStar.Classical.forall_intro (FStar.Classical.move_requires (choice_fwd #BF.t lr ls sel));
    FStar.Classical.forall_intro (FStar.Classical.move_requires (choice_fwd #F32.t lr ls sel));
    FStar.Classical.forall_intro (FStar.Classical.move_requires (choice_bwd #BF.t lr ls sel));
    FStar.Classical.forall_intro (FStar.Classical.move_requires (choice_bwd #F32.t lr ls sel))

let hidden_row_iff (tokens:S.seq I64.t) (hidden:S.seq BF.t) (j:nat) (hr:T.vector real T.hidden_size)
                   (hs:S.seq real)
  : Lemma (requires j < S.length tokens /\ S.length hidden == S.length tokens * 1024 /\
                    S.length hs == 1024 /\ vec_ok hr hs 0)
          (ensures SS.bf_row hidden j %~ hs <==>
                   T.vector_approximates ((outputs_of tokens hidden).T.hidden_states j) hr)
  = let row = SS.bf_row hidden j in
    let v = (outputs_of tokens hidden).T.hidden_states j in
    assert (row == S.slice hidden (j * 1024) (j * 1024 + 1024));
    assert (S.length row == 1024);
    let at (i:T.index T.hidden_size) : Lemma (S.index row i == v i /\ S.index hs i == hr i) = () in
    FStar.Classical.forall_intro at;
    introduce row %~ hs ==> T.vector_approximates v hr
    with introduce forall (i:T.index T.hidden_size). v i %~ hr i
         with assert ((row @! i) %~ (hs @! i));
    introduce T.vector_approximates v hr ==> row %~ hs
    with introduce forall (i:nat). i < len row ==> (row @! i) %~ (hs @! i)
         with introduce _ ==> _ with assert (v i %~ hr i)

(** weights in both directions *)

let weights_fwd (w:S.seq BF.t) (tw:T.model_weights real)
  : Lemma (requires S.length w == SS.weights_len /\ T.weights_approximates (weights_view (elem zero w)) tw)
          (ensures w %~ flat_of tw /\ weights_ok tw (flat_of tw))
  = flat_of_ok tw;
    let fw = flat_of tw in
    introduce forall (o:nat). o < len w ==> (w @! o) %~ (fw @! o)
    with introduce _ ==> _ with begin
      flat_read_view zero (elem zero w) o;
      flat_read_approx (weights_view (elem zero w)) tw o;
      flat_of_at tw o
    end

let linear_view_approx (e1:int -> GTot BF.t) (e2:int -> GTot real) (wo:int)
  : Lemma (requires forall (o:int). e1 o %~ e2 o)
          (ensures T.layer_approximates T.linear_attn_approximates (linear_layer_view e1 wo) (linear_layer_view e2 wo))
  = ()

let full_view_approx (e1:int -> GTot BF.t) (e2:int -> GTot real) (wo:int)
  : Lemma (requires forall (o:int). e1 o %~ e2 o)
          (ensures T.layer_approximates T.self_attn_approximates (full_layer_view e1 wo) (full_layer_view e2 wo))
  = ()

let view_approx (e1:int -> GTot BF.t) (e2:int -> GTot real)
  : Lemma (requires forall (o:int). e1 o %~ e2 o)
          (ensures T.weights_approximates (weights_view e1) (weights_view e2))
  = introduce forall (d:T.index T.linear_layer_count).
      T.layer_approximates T.linear_attn_approximates ((weights_view e1).T.linear_layers d)
        ((weights_view e2).T.linear_layers d)
    with linear_view_approx e1 e2 (SS.layer_off (4 * (d / 3) + d % 3));
    introduce forall (g:T.index T.full_layer_count).
      T.layer_approximates T.self_attn_approximates ((weights_view e1).T.full_layers g)
        ((weights_view e2).T.full_layers g)
    with full_view_approx e1 e2 (SS.layer_off (4 * g + 3))

let linear_view_ok (rw:S.seq real) (wo:int)
  : Lemma (linear_layer_ok (linear_layer_view (SS.at rw) wo) rw wo) = ()

let full_view_ok (rw:S.seq real) (wo:int)
  : Lemma (full_layer_ok (full_layer_view (SS.at rw) wo) rw wo) = ()

let view_ok (rw:S.seq real) : Lemma (weights_ok (weights_view (SS.at rw)) rw)
  = introduce forall (d:T.index T.linear_layer_count).
      linear_layer_ok ((weights_view (SS.at rw)).T.linear_layers d) rw (SS.layer_off (4 * (d / 3) + d % 3))
    with linear_view_ok rw (SS.layer_off (4 * (d / 3) + d % 3));
    introduce forall (g:T.index T.full_layer_count).
      full_layer_ok ((weights_view (SS.at rw)).T.full_layers g) rw (SS.layer_off (4 * g + 3))
    with full_view_ok rw (SS.layer_off (4 * g + 3))

let weights_bwd (w:S.seq BF.t) (rw:S.seq real)
  : Lemma (requires w %~ rw)
          (ensures T.weights_approximates (weights_view (elem zero w)) (weights_view (SS.at rw)) /\
                   weights_ok (weights_view (SS.at rw)) rw)
  = zero_bf ();
    introduce forall (o:int). elem zero w o %~ SS.at rw o
    with (if 0 <= o && o < S.length w then assert ((w @! o) %~ (rw @! o)));
    view_approx (elem zero w) (SS.at rw);
    view_ok rw

(** one output position *)

let consumed_eq (w:S.seq BF.t) (prompt:S.seq I64.t) (steps:int) (tokens:S.seq I64.t) (hidden:S.seq BF.t) (q:nat)
  : Lemma (T.consumed (inputs_of w prompt steps) (outputs_of tokens hidden) q == SS.prog_tok prompt tokens q)
  = ()

let position_iff (tn:T.numerics) (nm:SS.numerics) (tw:T.model_weights real) (rw:S.seq real)
                 (w:S.seq BF.t) (prompt:S.seq I64.t) (steps:int) (tokens:S.seq I64.t) (hidden:S.seq BF.t) (j:nat)
  : Lemma (requires nm_ok tn nm /\ weights_ok tw rw /\ 1 <= S.length prompt /\
                    SS.valid_tokens prompt /\ SS.valid_tokens tokens /\ j < S.length tokens /\
                    S.length hidden == S.length tokens * 1024)
          (ensures (let x = inputs_of w prompt steps in
                    let y = outputs_of tokens hidden in
                    let p = S.length prompt - 1 + j in
                    let hs = SS.ref_hidden nm rw (SS.prog_tok prompt tokens) p in
                    let h = T.hidden_at tn tw (T.consumed x y) (T.last_prompt_position x + j) in
                    (SS.bf_row hidden j %~ hs <==>
                       T.hidden_state_relation tn tw (T.consumed x y) (T.last_prompt_position x + j)
                         (y.T.hidden_states j)) /\
                    (SS.selected_ok nm rw hs (SS.tok_at tokens j) <==>
                       T.greedy_choice_relation tn tw h (y.T.tokens j))))
  = let x = inputs_of w prompt steps in
    let y = outputs_of tokens hidden in
    let p = S.length prompt - 1 + j in
    let ftoks = SS.prog_tok prompt tokens in
    let hs = SS.ref_hidden nm rw ftoks p in
    let h = T.hidden_at tn tw (T.consumed x y) p in
    FStar.Classical.forall_intro (consumed_eq w prompt steps tokens hidden);
    hidden_eq tn nm tw rw (T.consumed x y) ftoks p;
    ref_hidden_len nm rw ftoks p;
    hidden_row_iff tokens hidden j h hs;
    logits_eq tn nm tw rw h hs;
    selection_iff tn nm tw rw h hs (SS.tok_at tokens j)

(** the two directions *)

let fwd_with (w:S.seq BF.t) (prompt:S.seq I64.t) (steps:int) (tokens:S.seq I64.t) (hidden:S.seq BF.t)
             (nm:SS.numerics)
  : Lemma (requires S.length w == SS.weights_len /\ 1 <= S.length prompt /\ SS.valid_tokens prompt /\
                    S.length tokens == steps /\ S.length hidden == steps * 1024 /\ SS.valid_tokens tokens /\
                    SS.numerics_ok nm /\
                    (forall (rw:S.seq real). w %~ rw ==>
                      (forall (j:nat). j < steps ==>
                         SS.bf_row hidden j %~ SS.ref_hidden nm rw (SS.prog_tok prompt tokens) (S.length prompt - 1 + j) /\
                         SS.selected_ok nm rw (SS.ref_hidden nm rw (SS.prog_tok prompt tokens) (S.length prompt - 1 + j))
                                     (SS.tok_at tokens j))))
          (ensures (let x = inputs_of w prompt steps in
                    let y = outputs_of tokens hidden in
                    exists (tn:T.numerics). T.numerics_ok tn /\
                      (forall (tw:T.model_weights real). T.weights_approximates x.T.weights tw ==>
                         T.numerical tn x y tw /\ T.selection tn x y tw)))
  = let x = inputs_of w prompt steps in
    let y = outputs_of tokens hidden in
    let tn = tn_of nm in
    tn_of_ok nm;
    introduce forall (tw:T.model_weights real). T.weights_approximates x.T.weights tw ==>
                T.numerical tn x y tw /\ T.selection tn x y tw
    with introduce _ ==> _
    with begin
      weights_fwd w tw;
      let rw = flat_of tw in
      introduce forall (j:T.index y.T.generated_count).
        T.hidden_state_relation tn tw (T.consumed x y) (T.last_prompt_position x + j) (y.T.hidden_states j) /\
        T.greedy_choice_relation tn tw (T.hidden_at tn tw (T.consumed x y) (T.last_prompt_position x + j))
          (y.T.tokens j)
      with position_iff tn nm tw rw w prompt steps tokens hidden j
    end

let bwd_with (w:S.seq BF.t) (prompt:S.seq I64.t) (steps:int) (tokens:S.seq I64.t) (hidden:S.seq BF.t)
             (tn:T.numerics)
  : Lemma (requires 1 <= S.length prompt /\ SS.valid_tokens prompt /\
                    S.length tokens == steps /\ S.length hidden == steps * 1024 /\ SS.valid_tokens tokens /\
                    T.numerics_ok tn /\
                    (let x = inputs_of w prompt steps in
                     let y = outputs_of tokens hidden in
                     forall (tw:T.model_weights real). T.weights_approximates x.T.weights tw ==>
                       T.numerical tn x y tw /\ T.selection tn x y tw))
          (ensures exists (nm:SS.numerics). SS.numerics_ok nm /\
                     (forall (rw:S.seq real). w %~ rw ==>
                       (forall (j:nat). j < steps ==>
                          SS.bf_row hidden j %~ SS.ref_hidden nm rw (SS.prog_tok prompt tokens) (S.length prompt - 1 + j) /\
                          SS.selected_ok nm rw (SS.ref_hidden nm rw (SS.prog_tok prompt tokens) (S.length prompt - 1 + j))
                                      (SS.tok_at tokens j))))
  = let nm = nm_of tn in
    nm_of_ok tn;
    introduce forall (rw:S.seq real). w %~ rw ==>
      (forall (j:nat). j < steps ==>
         SS.bf_row hidden j %~ SS.ref_hidden nm rw (SS.prog_tok prompt tokens) (S.length prompt - 1 + j) /\
         SS.selected_ok nm rw (SS.ref_hidden nm rw (SS.prog_tok prompt tokens) (S.length prompt - 1 + j))
                     (SS.tok_at tokens j))
    with introduce _ ==> _
    with begin
      weights_bwd w rw;
      let tw = weights_view (SS.at rw) in
      introduce forall (j:nat). j < steps ==>
         SS.bf_row hidden j %~ SS.ref_hidden nm rw (SS.prog_tok prompt tokens) (S.length prompt - 1 + j) /\
         SS.selected_ok nm rw (SS.ref_hidden nm rw (SS.prog_tok prompt tokens) (S.length prompt - 1 + j))
                     (SS.tok_at tokens j)
      with introduce _ ==> _
      with position_iff tn nm tw rw w prompt steps tokens hidden j
    end

(*** Main theorems *)

let program_to_typed (w:S.seq BF.t) (prompt:S.seq I64.t) (steps:int) (tokens:S.seq I64.t) (hidden:S.seq BF.t)
  : Lemma (requires S.length w == SS.weights_len /\ SS.program_postcondition w prompt steps tokens hidden)
          (ensures T.postcondition (inputs_of w prompt steps) (outputs_of tokens hidden))
  = let x = inputs_of w prompt steps in
    let y = outputs_of tokens hidden in
    introduce T.input_domain x ==>
      (T.structural x y /\
       (exists (tn:T.numerics). T.numerics_ok tn /\
         (forall (tw:T.model_weights real). T.weights_approximates x.T.weights tw ==>
            T.numerical tn x y tw /\ T.selection tn x y tw)))
    with begin
      assert (forall (i:nat). i < S.length prompt ==> T.valid_token (x.T.prompt i));
      assert (SS.program_premise w prompt steps);
      FStar.Classical.forall_intro (FStar.Classical.move_requires (fwd_with w prompt steps tokens hidden))
    end

let typed_to_program (w:S.seq BF.t) (prompt:S.seq I64.t) (steps:int) (tokens:S.seq I64.t) (hidden:S.seq BF.t)
  : Lemma (requires S.length hidden == S.length tokens * 1024 /\
                    T.postcondition (inputs_of w prompt steps) (outputs_of tokens hidden))
          (ensures SS.program_postcondition w prompt steps tokens hidden)
  = let x = inputs_of w prompt steps in
    let y = outputs_of tokens hidden in
    introduce SS.program_premise w prompt steps ==>
      (S.length tokens == steps /\ S.length hidden == steps * 1024 /\ SS.valid_tokens tokens /\
       (exists (nm:SS.numerics). SS.numerics_ok nm /\
         (forall (rw:S.seq real). w %~ rw ==>
           (forall (j:nat). j < steps ==>
              SS.bf_row hidden j %~ SS.ref_hidden nm rw (SS.prog_tok prompt tokens) (S.length prompt - 1 + j) /\
              SS.selected_ok nm rw (SS.ref_hidden nm rw (SS.prog_tok prompt tokens) (S.length prompt - 1 + j))
                          (SS.tok_at tokens j)))))
    with begin
      assert (T.input_domain x);
      assert (T.structural x y);
      introduce forall (i:nat). i < S.length tokens ==> SS.valid_token (I64.v (S.index tokens i))
      with introduce _ ==> _ with assert (T.valid_token (y.T.tokens i));
      FStar.Classical.forall_intro (FStar.Classical.move_requires (bwd_with w prompt steps tokens hidden))
    end

let program_postcondition_iff (w:S.seq BF.t) (prompt:S.seq I64.t) (steps:int)
                              (tokens:S.seq I64.t) (hidden:S.seq BF.t)
  : Lemma (requires S.length w == SS.weights_len /\ S.length hidden == S.length tokens * 1024)
          (ensures SS.program_postcondition w prompt steps tokens hidden <==>
                   T.postcondition (inputs_of w prompt steps) (outputs_of tokens hidden))
  = FStar.Classical.move_requires (program_to_typed w prompt steps tokens) hidden;
    FStar.Classical.move_requires (typed_to_program w prompt steps tokens) hidden
