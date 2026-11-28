module KPlay.SharedSpec

#set-options "--z3rlimit 20"

(* Shared specification of greedy BF16 decode of Qwen3.5-0.8B
   (24 layers: 18 Gated DeltaNet + 6 gated full attention, tied embeddings).
   Reference math is real-valued; every relation is a G1 quantified relation
   "inputs %~ reals ==> outputs %~ reference(reals)" plus exact datum facts
   (copies, frames, token bookkeeping) and allowed-choice selection. *)

open FStar.Real
open Kuiper
open Kuiper.Approximates
module S = FStar.Seq
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32
module I64 = FStar.Int64
module F = Kuiper.Floating
module Exp = FStar.Math.Exp
module Sqrt = FStar.Math.Sqrt
module Trig = Kuiper.Real.Trigonometry
module Lit = Kuiper.Float32.Literals
module RO = Kuiper.Float.Realops

(*** Model dimensions and packed weight layout *)

let hidden_size : nat = 1024
let vocab : nat = 248320
let table_len : nat = 254279680          (* vocab * hidden, tied embedding / lm head *)
let final_norm_off : nat = 254279680     (* final RMSNorm weight in the packed weights *)
let layers_off : nat = 254280704         (* first layer group *)
let group_len : nat = 83018720           (* 3 DeltaNet layers + 1 attention layer *)
let dn_len : nat = 21555360              (* one packed DeltaNet layer *)
let attn_len : nat = 18352640            (* one packed attention layer *)
let weights_len : nat = 752393024
let dn_conv_len : nat = 24576            (* 6144 channels x 4 taps *)
let dn_state_len : nat = 262144          (* 16 heads x 128 x 128 *)
let opt_rec_len : nat = 4718592          (* 18 x dn_state_len *)
let opt_conv_len : nat = 442368          (* 18 x dn_conv_len *)
let opt_kv_len : nat = 6291456           (* 6 x 2 x 2048 x 256 *)

(* Weight offset of layer l (0 <= l < 24) inside the packed weights. *)
let layer_off (l:nat) : int = layers_off + group_len * (l / 4) + dn_len * (l % 4)

(*** Real helpers (total) *)

let at (s:S.seq real) (i:int) : real =
  if 0 <= i && i < S.length s then S.index s i else 0.0R

let vsum (n:nat) (f:nat -> real) : real =
  rsum (S.init n (fun (k:nat{k < n}) -> f k))

let rdiv (x y:real) : real = if t2b (y =!= 0.0R) then x /. y else 0.0R

let rsqrt_t (x:real) : real = if t2b (x >. 0.0R) then Sqrt.rsqrt x else 0.0R

let dot (w:S.seq real) (wo:int) (x:S.seq real) (n:nat) : real =
  vsum n (fun k -> at w (wo + k) *. at x k)

(* rows x cols row-major matrix at offset wo, times x *)
let matvec (w:S.seq real) (wo:int) (rows cols:nat) (x:S.seq real) : S.seq real =
  S.init rows (fun (r:nat{r < rows}) -> dot w (wo + r * cols) x cols)

let sq_sum (x:S.seq real) (o:int) (n:nat) : real =
  vsum n (fun k -> at x (o + k) *. at x (o + k))

let vadd (x y:S.seq real) (n:nat) : S.seq real =
  S.init n (fun (i:nat{i < n}) -> at x i +. at y i)

let sigmoid (x:real) : real =
  Exp.exp_positive (0.0R -. x);
  1.0R /. (1.0R +. Exp.exp (0.0R -. x))

let silu (x:real) : real = x *. sigmoid x

(* Thresholded softplus, the same definition as Luce.Softplus.model. *)
let softplus (r:real) : real =
  if t2b (r >. 20.0R) then r
  else (Exp.exp_positive r; Exp.log (1.0R +. Exp.exp r))

(* Qwen3.5 RMSNorm with (1 + w) scaling over x[0..n), weight at w[wo..]. *)
let rmsnorm1p (eps:real) (x:S.seq real) (n:nat) (w:S.seq real) (wo:int) : S.seq real =
  let inv = rsqrt_t (rdiv (sq_sum x 0 n) (of_int n) +. eps) in
  S.init n (fun (i:nat{i < n}) -> (at x i *. inv) *. (1.0R +. at w (wo + i)))

(*** Numeric constants of the model *)

noeq type numerics = {
  eps      : real;            (* RMSNorm / l2norm epsilon (1e-6) *)
  dn_qscale: real;            (* DeltaNet query scale 1/sqrt(128) *)
  inv_freq : S.seq real;      (* 32 RoPE inverse frequencies, theta = 1e7, rotary dim 64 *)
}

let eps_exact : real = 0.000001R
let eps_f32 : real = Lit.value "0.0000009999999974752427078783512115478515625"
let qscale_exact : real = Sqrt.rsqrt 128.0R
let qscale_f32 : real = Lit.value "0.088388346135616302490234375"

let base_freq_literal (i:nat) : string =
  match i with
  | 0 -> "1.0"
  | 1 -> "0.604296386241912841796875"
  | 2 -> "0.3651741445064544677734375"
  | 3 -> "0.22067339718341827392578125"
  | 4 -> "0.13335214555263519287109375"
  | 5 -> "0.08058421313762664794921875"
  | 6 -> "0.0486967526376247406005859375"
  | 7 -> "0.02942727319896221160888671875"
  | 8 -> "0.01778279431164264678955078125"
  | 9 -> "0.01074607856571674346923828125"
  | 10 -> "0.0064938166178762912750244140625"
  | 11 -> "0.0039241896010935306549072265625"
  | 12 -> "0.00237137381918728351593017578125"
  | 13 -> "0.001433012657798826694488525390625"
  | 14 -> "0.000865964335389435291290283203125"
  | 15 -> "0.000523299095220863819122314453125"
  | 16 -> "0.000316227786242961883544921875"
  | 17 -> "0.000191095299669541418552398681640625"
  | 18 -> "0.000115478193038143217563629150390625"
  | 19 -> "0.000069783054641447961330413818359375"
  | 20 -> "0.0000421696531702764332294464111328125"
  | 21 -> "0.000025482966520939953625202178955078125"
  | 22 -> "0.000015399264157167635858058929443359375"
  | 23 -> "0.00000930572059587575495243072509765625"
  | 24 -> "0.00000562341347176698036491870880126953125"
  | 25 -> "0.00000339820826411596499383449554443359375"
  | 26 -> "0.00000205352489501819945871829986572265625"
  | 27 -> "0.00000124093776321387849748134613037109375"
  | 28 -> "0.00000074989424092564149759709835052490234375"
  | 29 -> "0.000000453158378377338522113859653472900390625"
  | 30 -> "0.00000027384197665014653466641902923583984375"
  | 31 -> "0.00000016548170833630138076841831207275390625"
  | _ -> "0.0"

(* f32-rounded inverse frequencies (the base kernel's constants) *)
let inv_freq_f32 : S.seq real =
  S.init 32 (fun (i:nat{i < 32}) -> Lit.value (base_freq_literal i))

(* The exact inverse frequencies 1 / 1e7^(2i/64), written as
   1 / 2^((2i/64) * log2(1e7)), the optimized kernel's form (the library has
   no law linking Pow.pow to exp/log, so one form is used throughout). *)
let inv_freq_exp2 : S.seq real =
  S.init 32 (fun (i:nat{i < 32}) ->
    1.0R /. RO.r_exp2 ((of_int (2 * i) /. 64.0R) *. RO.r_log2 10000000.0R))

(* Admissible spellings of the model constants: the exact real value or its
   f32 rounding. *)
let numerics_ok (nm:numerics) : prop =
  (nm.eps == eps_exact \/ nm.eps == eps_f32) /\
  (nm.dn_qscale == qscale_exact \/ nm.dn_qscale == qscale_f32) /\
  (nm.inv_freq == inv_freq_exp2 \/ nm.inv_freq == inv_freq_f32)

let base_numerics : numerics = { eps = eps_f32; dn_qscale = qscale_f32; inv_freq = inv_freq_f32 }
let opt_numerics : numerics = { eps = eps_f32; dn_qscale = qscale_f32; inv_freq = inv_freq_exp2 }

let rope_cos (nm:numerics) (t:int) : S.seq real =
  S.init 32 (fun (i:nat{i < 32}) -> Trig.cos (of_int t *. at nm.inv_freq i))

let rope_sin (nm:numerics) (t:int) : S.seq real =
  S.init 32 (fun (i:nat{i < 32}) -> Trig.sin (of_int t *. at nm.inv_freq i))

(* Token embedding row (table at offset 0 of w). *)
let embed (w:S.seq real) (tok:int) : S.seq real =
  S.init 1024 (fun (i:nat{i < 1024}) -> at w (tok * 1024 + i))

(*** SwiGLU MLP with post-attention norm (both layer kinds) *)
(* Layer-relative offsets: input norm 0, post norm 1024, gate 2048,
   up 3672064, down 7342080 (1024 x 3584), mixer at 11012096. *)

let mlp_in (nm:numerics) (w:S.seq real) (wo:int) (x1:S.seq real) : S.seq real =
  rmsnorm1p nm.eps x1 1024 w (wo + 1024)

let mlp_act (nm:numerics) (w:S.seq real) (wo:int) (x1:S.seq real) : S.seq real =
  let h = mlp_in nm w wo x1 in
  let g = matvec w (wo + 2048) 3584 1024 h in
  let u = matvec w (wo + 3672064) 3584 1024 h in
  S.init 3584 (fun (i:nat{i < 3584}) -> silu (at g i) *. at u i)

let mlp_out (nm:numerics) (w:S.seq real) (wo:int) (x1:S.seq real) : S.seq real =
  vadd x1 (matvec w (wo + 7342080) 1024 3584 (mlp_act nm w wo x1)) 1024

(*** Gated DeltaNet layer *)
(* Mixer-relative offsets (mixer at wo + 11012096): qkv 0 (6144 x 1024),
   z 6291456 (2048 x 1024), b 8388608 (16 x 1024), a 8404992 (16 x 1024),
   conv1d 8421376 (6144 x 4), A_log 8445952, dt_bias 8445968,
   gated norm 8445984 (128), out_proj 8446112 (1024 x 2048).
   Conv state per layer: [c*4 + k], window of the last 4 inputs (k = 3 newest).
   Recurrent state per layer: [h*16384 + k*128 + v]. *)

let dn_mix (wo:int) : int = wo + 11012096

let dn_hin (nm:numerics) (w:S.seq real) (wo:int) (x:S.seq real) : S.seq real =
  rmsnorm1p nm.eps x 1024 w wo

let dn_qkv (nm:numerics) (w:S.seq real) (wo:int) (x:S.seq real) : S.seq real =
  matvec w (dn_mix wo) 6144 1024 (dn_hin nm w wo x)

let dn_conv (nm:numerics) (w:S.seq real) (wo:int) (x conv:S.seq real) : S.seq real =
  let m = dn_qkv nm w wo x in
  S.init 24576 (fun (j:nat{j < 24576}) -> if j % 4 < 3 then at conv (j + 1) else at m (j / 4))

let dn_conv_out (nm:numerics) (w:S.seq real) (wo:int) (x conv:S.seq real) : S.seq real =
  let c' = dn_conv nm w wo x conv in
  S.init 6144 (fun (c:nat{c < 6144}) ->
    silu (vsum 4 (fun k -> at c' (c * 4 + k) *. at w (dn_mix wo + 8421376 + c * 4 + k))))

(* l2-normalized, scaled query (2048 = 16 heads x 128) *)
let dn_q (nm:numerics) (w:S.seq real) (wo:int) (x conv:S.seq real) : S.seq real =
  let co = dn_conv_out nm w wo x conv in
  S.init 2048 (fun (i:nat{i < 2048}) ->
    (at co i *. rsqrt_t (sq_sum co ((i / 128) * 128) 128 +. nm.eps)) *. nm.dn_qscale)

let dn_k (nm:numerics) (w:S.seq real) (wo:int) (x conv:S.seq real) : S.seq real =
  let co = dn_conv_out nm w wo x conv in
  S.init 2048 (fun (i:nat{i < 2048}) ->
    at co (2048 + i) *. rsqrt_t (sq_sum co (2048 + (i / 128) * 128) 128 +. nm.eps))

let dn_v (nm:numerics) (w:S.seq real) (wo:int) (x conv:S.seq real) : S.seq real =
  let co = dn_conv_out nm w wo x conv in
  S.init 2048 (fun (i:nat{i < 2048}) -> at co (4096 + i))

let dn_beta (nm:numerics) (w:S.seq real) (wo:int) (x:S.seq real) (h:int) : real =
  sigmoid (dot w (dn_mix wo + 8388608 + h * 1024) (dn_hin nm w wo x) 1024)

let dn_decay (nm:numerics) (w:S.seq real) (wo:int) (x:S.seq real) (h:int) : real =
  let a = dot w (dn_mix wo + 8404992 + h * 1024) (dn_hin nm w wo x) 1024
          +. at w (dn_mix wo + 8445968 + h) in
  Exp.exp ((0.0R -. Exp.exp (at w (dn_mix wo + 8445952 + h))) *. softplus a)

(* delta-rule correction for head h, value column j *)
let dn_delta (nm:numerics) (w:S.seq real) (wo:int) (x conv st:S.seq real) (h j:int) : real =
  let k = dn_k nm w wo x conv in
  let v = dn_v nm w wo x conv in
  let dec = dn_decay nm w wo x h in
  let mem = vsum 128 (fun i -> (at st (h * 16384 + i * 128 + j) *. dec) *. at k (h * 128 + i)) in
  (at v (h * 128 + j) -. mem) *. dn_beta nm w wo x h

let dn_state (nm:numerics) (w:S.seq real) (wo:int) (x conv st:S.seq real) : S.seq real =
  let k = dn_k nm w wo x conv in
  S.init 262144 (fun (idx:nat{idx < 262144}) ->
    let h = idx / 16384 in
    let i = (idx % 16384) / 128 in
    let j = idx % 128 in
    at st idx *. dn_decay nm w wo x h +. at k (h * 128 + i) *. dn_delta nm w wo x conv st h j)

let dn_core (nm:numerics) (w:S.seq real) (wo:int) (x conv st:S.seq real) : S.seq real =
  let st' = dn_state nm w wo x conv st in
  let q = dn_q nm w wo x conv in
  S.init 2048 (fun (c:nat{c < 2048}) ->
    let h = c / 128 in
    vsum 128 (fun i -> at st' (h * 16384 + i * 128 + c % 128) *. at q (h * 128 + i)))

let dn_gated (nm:numerics) (w:S.seq real) (wo:int) (x conv st:S.seq real) : S.seq real =
  let core = dn_core nm w wo x conv st in
  let z = matvec w (dn_mix wo + 6291456) 2048 1024 (dn_hin nm w wo x) in
  S.init 2048 (fun (c:nat{c < 2048}) ->
    let inv = rsqrt_t (rdiv (sq_sum core ((c / 128) * 128) 128) 128.0R +. nm.eps) in
    (at w (dn_mix wo + 8445984 + c % 128) *. (at core c *. inv)) *. silu (at z c))

let dn_x1 (nm:numerics) (w:S.seq real) (wo:int) (x conv st:S.seq real) : S.seq real =
  vadd x (matvec w (dn_mix wo + 8446112) 1024 2048 (dn_gated nm w wo x conv st)) 1024

let dn_out (nm:numerics) (w:S.seq real) (wo:int) (x conv st:S.seq real) : S.seq real =
  mlp_out nm w wo (dn_x1 nm w wo x conv st)

(*** Gated full-attention layer *)
(* Mixer-relative offsets: q 0 (4096 x 1024, per head [query 256, gate 256]),
   k 4194304 (512 x 1024), v 4718592 (512 x 1024), q_norm 5242880 (256),
   k_norm 5243136 (256), o_proj 5243392 (1024 x 2048).
   8 query heads, 2 KV heads (GQA, kv head = h / 4), head dim 256,
   partial RoPE on the first 64 dims (rotate-half pairs (i, i + 32)).
   History: hk p j / hv p j = key / value component j (kvh*256 + e) of past
   position p < t. *)

let at_mix (wo:int) : int = wo + 11012096

let at_hin (nm:numerics) (w:S.seq real) (wo:int) (x:S.seq real) : S.seq real =
  rmsnorm1p nm.eps x 1024 w wo

let at_qraw (nm:numerics) (w:S.seq real) (wo:int) (x:S.seq real) : S.seq real =
  matvec w (at_mix wo) 4096 1024 (at_hin nm w wo x)

let at_kraw (nm:numerics) (w:S.seq real) (wo:int) (x:S.seq real) : S.seq real =
  matvec w (at_mix wo + 4194304) 512 1024 (at_hin nm w wo x)

let at_v (nm:numerics) (w:S.seq real) (wo:int) (x:S.seq real) : S.seq real =
  matvec w (at_mix wo + 4718592) 512 1024 (at_hin nm w wo x)

(* per-head RMSNorm (1 + w) over raw[o..o+256) *)
let head_norm (eps:real) (raw:S.seq real) (o:int) (w:S.seq real) (wo:int) : S.seq real =
  let inv = rsqrt_t (rdiv (sq_sum raw o 256) 256.0R +. eps) in
  S.init 256 (fun (i:nat{i < 256}) -> (at raw (o + i) *. inv) *. (1.0R +. at w (wo + i)))

let rope (u cs sn:S.seq real) : S.seq real =
  S.init 256 (fun (i:nat{i < 256}) ->
    if i < 32 then at u i *. at cs i -. at u (i + 32) *. at sn i
    else if i < 64 then at u i *. at cs (i - 32) +. at u (i - 32) *. at sn (i - 32)
    else at u i)

let at_q (nm:numerics) (w:S.seq real) (wo:int) (x cs sn:S.seq real) : S.seq real =
  let raw = at_qraw nm w wo x in
  S.init 2048 (fun (c:nat{c < 2048}) ->
    at (rope (head_norm nm.eps raw ((c / 256) * 512) w (at_mix wo + 5242880)) cs sn) (c % 256))

let at_k (nm:numerics) (w:S.seq real) (wo:int) (x cs sn:S.seq real) : S.seq real =
  let raw = at_kraw nm w wo x in
  S.init 512 (fun (c:nat{c < 512}) ->
    at (rope (head_norm nm.eps raw ((c / 256) * 256) w (at_mix wo + 5243136)) cs sn) (c % 256))

let hist_or (h:nat -> nat -> real) (cur:S.seq real) (t:nat) (p j:nat) : real =
  if p < t then h p j else at cur j

let at_score (nm:numerics) (w:S.seq real) (wo:int) (x cs sn:S.seq real) (t:nat)
             (hk:nat -> nat -> real) (hd p:nat) : real =
  let q = at_q nm w wo x cs sn in
  let kn = at_k nm w wo x cs sn in
  vsum 256 (fun i -> at q (hd * 256 + i) *. hist_or hk kn t p ((hd / 4) * 256 + i)) *. 0.0625R

(* softmax over positions 0..t (inclusive) *)
let at_prob (nm:numerics) (w:S.seq real) (wo:int) (x cs sn:S.seq real) (t:nat)
            (hk:nat -> nat -> real) (hd p:nat) : real =
  rdiv (Exp.exp (at_score nm w wo x cs sn t hk hd p))
       (vsum (t + 1) (fun p' -> Exp.exp (at_score nm w wo x cs sn t hk hd p')))

let at_gated (nm:numerics) (w:S.seq real) (wo:int) (x cs sn:S.seq real) (t:nat)
             (hk hv:nat -> nat -> real) : S.seq real =
  let vn = at_v nm w wo x in
  let raw = at_qraw nm w wo x in
  S.init 2048 (fun (c:nat{c < 2048}) ->
    let hd = c / 256 in
    let e = c % 256 in
    vsum (t + 1) (fun p -> at_prob nm w wo x cs sn t hk hd p *. hist_or hv vn t p ((hd / 4) * 256 + e))
    *. sigmoid (at raw (hd * 512 + 256 + e)))

let at_x1 (nm:numerics) (w:S.seq real) (wo:int) (x cs sn:S.seq real) (t:nat)
          (hk hv:nat -> nat -> real) : S.seq real =
  vadd x (matvec w (at_mix wo + 5243392) 1024 2048 (at_gated nm w wo x cs sn t hk hv)) 1024

let at_out (nm:numerics) (w:S.seq real) (wo:int) (x cs sn:S.seq real) (t:nat)
           (hk hv:nat -> nat -> real) : S.seq real =
  mlp_out nm w wo (at_x1 nm w wo x cs sn t hk hv)

(*** Final norm and tied LM head *)

let logits (nm:numerics) (tab:S.seq real) (tabo:int) (nw:S.seq real) (nwo:int) (h:S.seq real)
  : S.seq real =
  let hn = rmsnorm1p nm.eps h 1024 nw nwo in
  S.init 248320 (fun (m:nat{m < 248320}) -> dot tab (tabo + m * 1024) hn 1024)

(*** Reference decode *)
(* Decoder state: per DeltaNet layer d (0..17) the conv window and recurrent
   state (reference layouts above); per processed position the KV entry of
   all 6 attention layers (layer g: key at g*1024 + j, value at g*1024 + 512 + j). *)

noeq type rstate = {
  convs : S.seq (S.seq real);
  recs  : S.seq (S.seq real);
  kvs   : S.seq (S.seq real);
}

let init_state : rstate = {
  convs = S.create 18 (S.create 24576 0.0R);
  recs  = S.create 18 (S.create 262144 0.0R);
  kvs   = S.empty;
}

let sget (s:S.seq (S.seq real)) (i:int) : S.seq real =
  if 0 <= i && i < S.length s then S.index s i else S.empty

let supd (s:S.seq (S.seq real)) (i:int) (v:S.seq real) : S.seq (S.seq real) =
  if 0 <= i && i < S.length s then S.upd s i v else s

let kv_get (kvs:S.seq (S.seq real)) (p:nat) (o:int) : real = at (sget kvs p) o

(* Attention layer g = l/4 for l % 4 == 3, otherwise DeltaNet layer 3*(l/4) + l%4. *)
let is_attn (l:nat) : bool = l % 4 = 3
let dn_index (l:nat) : nat = 3 * (l / 4) + l % 4

let write_kv (cur:S.seq real) (g:nat) (kn vn:S.seq real) : S.seq real =
  S.init 6144 (fun (j:nat{j < 6144}) ->
    if j / 1024 = g then (if j % 1024 < 512 then at kn (j % 1024) else at vn (j % 1024 - 512))
    else at cur j)

noeq type lstate = {
  l_convs : S.seq (S.seq real);
  l_recs  : S.seq (S.seq real);
  l_cur   : S.seq real;        (* KV entry being built for the current position *)
  l_x     : S.seq real;        (* residual stream *)
}

(* Layers l .. 23 at position t. *)
let rec run_layers (nm:numerics) (w:S.seq real) (t:nat) (cs sn:S.seq real)
                   (kvs:S.seq (S.seq real)) (st:lstate) (l:nat)
  : Tot lstate (decreases (if l < 24 then 24 - l else 0)) =
  if l >= 24 then st
  else
    let wo = layer_off l in
    if is_attn l then
      let g = l / 4 in
      let hk = fun (p j:nat) -> kv_get kvs p (g * 1024 + j) in
      let hv = fun (p j:nat) -> kv_get kvs p (g * 1024 + 512 + j) in
      run_layers nm w t cs sn kvs
        ({ st with
           l_cur = write_kv st.l_cur g (at_k nm w wo st.l_x cs sn) (at_v nm w wo st.l_x);
           l_x = at_out nm w wo st.l_x cs sn t hk hv }) (l + 1)
    else
      let d = dn_index l in
      let cv = sget st.l_convs d in
      let rs = sget st.l_recs d in
      run_layers nm w t cs sn kvs
        ({ l_convs = supd st.l_convs d (dn_conv nm w wo st.l_x cv);
           l_recs = supd st.l_recs d (dn_state nm w wo st.l_x cv rs);
           l_cur = st.l_cur;
           l_x = dn_out nm w wo st.l_x cv rs }) (l + 1)

(* One decode position t with input token tok. Returns the new state and the
   final residual (pre final norm). *)
let ref_step (nm:numerics) (w:S.seq real) (tok:int) (t:nat) (st:rstate) : rstate & S.seq real =
  let r = run_layers nm w t (rope_cos nm t) (rope_sin nm t) st.kvs
            ({ l_convs = st.convs; l_recs = st.recs; l_cur = S.create 6144 0.0R; l_x = embed w tok }) 0 in
  ({ convs = r.l_convs; recs = r.l_recs; kvs = S.snoc st.kvs r.l_cur }, r.l_x)

(* Positions 0 .. n-1, token at position p is toks p. *)
let rec ref_run (nm:numerics) (w:S.seq real) (toks:nat -> int) (n:nat)
  : Tot (rstate & S.seq real) (decreases n) =
  if n = 0 then (init_state, S.empty)
  else
    let prev = ref_run nm w toks (n - 1) in
    ref_step nm w (toks (n - 1)) (n - 1) (fst prev)

(* Final residual (pre-norm hidden) at position p. *)
let ref_hidden (nm:numerics) (w:S.seq real) (toks:nat -> int) (p:nat) : S.seq real =
  snd (ref_run nm w toks (p + 1))

(* Logits of the packed model for hidden h. *)
let ref_logits (nm:numerics) (w:S.seq real) (h:S.seq real) : S.seq real =
  logits nm w 0 w final_norm_off h

(*** Selection *)

let is_nan (#t:Type) {| F.floating t |} (x:t) : prop = NaN? (F.kind x)

(* sel is an allowed greedy choice over the native row: if the row has a NaN
   the chosen element is NaN, otherwise no element is strictly greater. *)
let allowed_choice (#t:Type) {| F.floating t |} (row:S.seq t) (sel:int) : prop =
  0 <= sel /\ sel < S.length row /\
  ((exists (k:nat). k < S.length row /\ is_nan (S.index row k)) ==> is_nan (S.index row sel)) /\
  ((forall (k:nat). k < S.length row ==> ~(is_nan (S.index row k))) ==>
     (forall (k:nat). k < S.length row ==> ~(lt (S.index row sel) (S.index row k))))

(* The first allowed choice (first NaN, else first maximum). *)
let first_allowed_choice (#t:Type) {| F.floating t |} (row:S.seq t) (sel:int) : prop =
  allowed_choice row sel /\ (forall (k:nat). k < sel ==> ~(allowed_choice row k))

(* Exact model of the optimized two-level argmax tree.
   A candidate is (value, index); index -1 is an empty lane whose value is
   never consulted. *)
let better (v:F32.t) (i:int) (best:F32.t) (bi:int) : GTot bool =
  i >= 0 && (bi < 0 || lt best v || (F.NaN? (F.kind v) && not (F.NaN? (F.kind best))))

let pick (acc ch:F32.t & int) : GTot (F32.t & int) =
  if better (fst ch) (snd ch) (fst acc) (snd acc) then ch else acc

let f32_at (s:S.seq F32.t) (i:int) : F32.t =
  if 0 <= i && i < S.length s then S.index s i else zero

let rec lane_scan (sc:S.seq F32.t) (m stop:int) (acc:F32.t & int)
  : GTot (F32.t & int) (decreases (if m < stop then stop - m else 0)) =
  if m >= stop then acc else lane_scan sc (m + 8) stop (pick acc (f32_at sc m, m))

let lane_cand (sc:S.seq F32.t) (b w:nat) : GTot (F32.t & int) =
  if w < 8 then lane_scan sc (b * 485 + w) ((b + 1) * 485) (zero, -1) else (zero, -1)

(* tree reduction with strides d, d/2, .., 1; element i absorbs element i + d *)
let rec tree_at (f:nat -> GTot (F32.t & int)) (d:nat) (i:nat) : GTot (F32.t & int) (decreases d) =
  if d = 0 then f i
  else tree_at (fun (k:nat) -> if k < d then pick (f k) (f (k + d)) else f k) (d / 2) i

let block_cand (sc:S.seq F32.t) (b:nat) : GTot (F32.t & int) =
  tree_at (fun (w:nat) -> lane_cand sc b w) 16 0

let tree_argmax (sc:S.seq F32.t) : GTot int =
  snd (tree_at (fun (b:nat) -> block_cand sc b) 256 0)

(*** Program postcondition (role-independent) *)

let valid_token (x:int) : bool = 0 <= x && x < 248320

let tok_at (s:S.seq I64.t) (i:int) : int =
  if 0 <= i && i < S.length s then I64.v (S.index s i) else 0

let valid_tokens (s:S.seq I64.t) : prop =
  forall (i:nat). i < S.length s ==> valid_token (I64.v (S.index s i))

(* token at decode position p: prompt, then generated tokens *)
let prog_tok (prompt tokens:S.seq I64.t) (p:nat) : int =
  if p < S.length prompt then tok_at prompt p else tok_at tokens (p - S.length prompt)

let bf_row (s:S.seq BF.t) (j:int) : S.seq BF.t =
  if 0 <= j && j * 1024 + 1024 <= S.length s then S.slice s (j * 1024) (j * 1024 + 1024) else S.empty

(* greedy choice over a consumed score row that approximates the reference
   logits: the first allowed choice over a BF16 row, an allowed choice over
   an F32 row *)
let selected_ok (nm:numerics) (rw:S.seq real) (h:S.seq real) (sel:int) : prop =
  (exists (row:S.seq BF.t). row %~ ref_logits nm rw h /\ first_allowed_choice row sel) \/
  (exists (row:S.seq F32.t). row %~ ref_logits nm rw h /\ allowed_choice row sel)

let program_premise (weights:S.seq BF.t) (prompt:S.seq I64.t) (steps:int) : prop =
  S.length weights == weights_len /\
  1 <= S.length prompt /\ S.length prompt <= 8 /\ 1 <= steps /\ steps <= 5 /\
  valid_tokens prompt

(* Greedy decode: output token j is chosen from the logits of position
   P - 1 + j, output hidden row j is the final (pre-norm) residual there,
   positions >= P consume the previously generated tokens. *)
let program_postcondition (weights:S.seq BF.t) (prompt:S.seq I64.t) (steps:int)
                          (tokens:S.seq I64.t) (hidden:S.seq BF.t) : prop =
  program_premise weights prompt steps ==>
  (S.length tokens == steps /\ S.length hidden == steps * 1024 /\ valid_tokens tokens /\
   (exists (nm:numerics). numerics_ok nm /\
     (forall (rw:S.seq real). weights %~ rw ==>
       (forall (j:nat). j < steps ==>
          bf_row hidden j %~ ref_hidden nm rw (prog_tok prompt tokens) (S.length prompt - 1 + j) /\
          selected_ok nm rw (ref_hidden nm rw (prog_tok prompt tokens) (S.length prompt - 1 + j))
                      (tok_at tokens j)))))

(*** Layout helpers for the relations *)

(* Base KV cache of one attention layer, capacity C:
   K at (kvh*C + p)*256 + e, V at C*512 + (kvh*C + p)*256 + e. *)
let base_kv_index (c:int) (p j:nat) : int = ((j / 256) * c + p) * 256 + j % 256

let base_hk (c:int) (rkv:S.seq real) : nat -> nat -> real =
  fun (p j:nat) -> at rkv (base_kv_index c p j)

let base_hv (c:int) (rkv:S.seq real) : nat -> nat -> real =
  fun (p j:nat) -> at rkv (c * 512 + base_kv_index c p j)

let base_kv_pos (c:int) (i:int) : int =
  if c > 0 && i >= 0 then ((i % (c * 512)) % (c * 256)) / 256 else (-1)

let base_kv_comp (c:int) (i:int) : int =
  if c > 0 && i >= 0 then ((i % (c * 512)) / (c * 256)) * 256 + i % 256 else 0

let base_kv_write (c:int) (t:int) (rkv kn vn:S.seq real) : S.seq real =
  S.init (S.length rkv) (fun (i:nat{i < S.length rkv}) ->
    if base_kv_pos c i = t then (if i < c * 512 then at kn (base_kv_comp c i) else at vn (base_kv_comp c i))
    else at rkv i)

(* Optimized KV caches [fa][kvh][2048][256]. *)
let opt_kv_index (fa:int) (p j:nat) : int = fa * 1048576 + (j / 256) * 524288 + p * 256 + j % 256

let opt_hk (fa:int) (rk:S.seq real) : nat -> nat -> real =
  fun (p j:nat) -> at rk (opt_kv_index fa p j)

let opt_kv_hit (fa pos:int) (i:nat) : bool = i / 1048576 = fa && (i % 524288) / 256 = pos

let opt_kv_comp (i:nat) : nat = ((i % 1048576) / 524288) * 256 + i % 256

let opt_kv_write (fa pos:int) (rk kn:S.seq real) : S.seq real =
  S.init 6291456 (fun (i:nat{i < 6291456}) -> if opt_kv_hit fa pos i then at kn (opt_kv_comp i) else at rk i)

(* Optimized recurrent state slot layout [h][v][k] versus reference [h][k][v]
   (the index map is an involution). *)
let rec_perm (idx:nat) : nat = (idx / 16384) * 16384 + (idx % 128) * 128 + (idx % 16384) / 128

let opt_conv_slot (rc:S.seq real) (dn:int) : S.seq real =
  S.init 24576 (fun (j:nat{j < 24576}) -> at rc (dn * 24576 + j))

let opt_rec_slot (rr:S.seq real) (dn:int) : S.seq real =
  S.init 262144 (fun (idx:nat{idx < 262144}) -> at rr (dn * 262144 + rec_perm idx))

let opt_conv_put (rc:S.seq real) (dn:int) (c':S.seq real) : S.seq real =
  S.init 442368 (fun (i:nat{i < 442368}) -> if i / 24576 = dn then at c' (i % 24576) else at rc i)

let opt_rec_put (rr:S.seq real) (dn:int) (s':S.seq real) : S.seq real =
  S.init 4718592 (fun (i:nat{i < 4718592}) -> if i / 262144 = dn then at s' (rec_perm (i % 262144)) else at rr i)

let opt_emb_token (prompt tokens:S.seq I64.t) (p:int) : int =
  if p = 0 then tok_at tokens 0
  else if p < S.length prompt then tok_at prompt p
  else tok_at tokens (p - S.length prompt)

let all_zero (#t:Type) {| F.floating t |} (s:S.seq t) : prop =
  forall (i:nat). i < S.length s ==> S.index s i == zero

(*** Base relations *)

let base_argmax_relation (logits:S.seq BF.t) (pP sS t:int) (selected:S.seq I64.t) : prop =
  (S.length logits == 248320 /\ 1 <= pP /\ 1 <= sS /\ pP + sS - 1 <= 12 /\ 0 <= t /\ t < 12) ==>
  (S.length selected == 1 /\
   (0 <= t - pP + 1 /\ t - pP + 1 < sS ==> first_allowed_choice logits (tok_at selected 0)) /\
   (~(0 <= t - pP + 1 /\ t - pP + 1 < sS) ==> tok_at selected 0 == 0))

let base_embedding_rope_relation (table emb:S.seq BF.t) (tokens:S.seq I64.t) (t:int)
                                 (residual cos sin:S.seq BF.t) : prop =
  let pP = S.length emb / 1024 in
  (S.length table == table_len /\ S.length emb == pP * 1024 /\ 1 <= pP /\
   1 <= S.length tokens /\ 0 <= t /\ t < pP + S.length tokens - 1 /\ t < 12 /\
   (t >= pP ==> valid_token (tok_at tokens (t - pP)))) ==>
  (S.length residual == 1024 /\ S.length cos == 32 /\ S.length sin == 32 /\
   (forall (i:nat). i < 1024 ==>
      S.index residual i ==
        (if t < pP then S.index emb (t * 1024 + i) else S.index table (tok_at tokens (t - pP) * 1024 + i))) /\
   cos %~ rope_cos base_numerics t /\ sin %~ rope_sin base_numerics t)

let base_final_norm_projection_relation (table normw residual:S.seq BF.t)
                                        (hidden logits':S.seq BF.t) : prop =
  (S.length table == table_len /\ S.length normw == 1024 /\ S.length residual == 1024) ==>
  (S.length hidden == 1024 /\ S.length logits' == 248320 /\
   (forall (i:nat). i < 1024 ==> S.index hidden i == S.index residual i) /\
   (forall (rt rn rx:S.seq real). table %~ rt /\ normw %~ rn /\ residual %~ rx ==>
      logits' %~ logits base_numerics rt 0 rn 0 rx))

let base_full_attention_decoder_relation (w x kv cos sin:S.seq BF.t) (t:int) (x' kv':S.seq BF.t) : prop =
  let c = S.length kv / 1024 in
  (S.length w == attn_len /\ S.length x == 1024 /\ S.length kv == c * 1024 /\ 1 <= c /\ c <= 12 /\
   0 <= t /\ t < c /\ S.length cos == 32 /\ S.length sin == 32) ==>
  (S.length x' == 1024 /\ S.length kv' == S.length kv /\
   (forall (i:nat). i < S.length kv /\ base_kv_pos c i <> t ==> S.index kv' i == S.index kv i) /\
   (forall (rw rx rkv rc rs:S.seq real).
      w %~ rw /\ x %~ rx /\ kv %~ rkv /\ cos %~ rc /\ sin %~ rs ==>
      x' %~ at_out base_numerics rw 0 rx rc rs t (base_hk c rkv) (base_hv c rkv) /\
      kv' %~ base_kv_write c t rkv (at_k base_numerics rw 0 rx rc rs) (at_v base_numerics rw 0 rx)))

let base_initialize_relation (table:S.seq BF.t) (prompt:S.seq I64.t) (steps:int)
                             (emb cache:S.seq BF.t) (recurrent:S.seq F32.t)
                             (tokens:S.seq I64.t) (history:S.seq BF.t) : prop =
  let pP = S.length prompt in
  (S.length table == table_len /\ 1 <= pP /\ 1 <= steps /\ pP + steps - 1 <= 12 /\ valid_tokens prompt) ==>
  (S.length emb == pP * 1024 /\
   (forall (i:nat). i < pP * 1024 ==> S.index emb i == S.index table (tok_at prompt (i / 1024) * 1024 + i % 1024)) /\
   S.length cache == 442368 + (pP + steps - 1) * 6144 /\ all_zero cache /\
   S.length recurrent == opt_rec_len /\ all_zero recurrent /\
   S.length tokens == steps /\ (forall (i:nat). i < steps ==> tok_at tokens i == 0) /\
   S.length history == steps * 1024 /\ all_zero history)

let base_recurrent_decoder_relation (w x conv:S.seq BF.t) (st:S.seq F32.t)
                                    (x' conv':S.seq BF.t) (st':S.seq F32.t) : prop =
  (S.length w == dn_len /\ S.length x == 1024 /\ S.length conv == 24576 /\ S.length st == 262144) ==>
  (S.length x' == 1024 /\ S.length conv' == 24576 /\ S.length st' == 262144 /\
   (forall (j:nat). j + 1 < 24576 /\ j % 4 < 3 ==> S.index conv' j == S.index conv (j + 1)) /\
   (forall (rw rx rc rs:S.seq real).
      w %~ rw /\ x %~ rx /\ conv %~ rc /\ st %~ rs ==>
      x' %~ dn_out base_numerics rw 0 rx rc rs /\
      conv' %~ dn_conv base_numerics rw 0 rx rc /\
      st' %~ dn_state base_numerics rw 0 rx rc rs))

let base_token_history_update_relation (old_tokens:S.seq I64.t) (old_history hidden:S.seq BF.t)
                                       (selected:S.seq I64.t) (pP t:int)
                                       (tokens:S.seq I64.t) (history:S.seq BF.t) : prop =
  let sS = S.length old_tokens in
  let dest = t - pP + 1 in
  (1 <= sS /\ S.length old_history == sS * 1024 /\ S.length hidden == 1024 /\ S.length selected == 1 /\
   1 <= pP /\ pP + sS - 1 <= 12 /\ 0 <= t /\ t < 12) ==>
  (S.length tokens == sS /\ S.length history == sS * 1024 /\
   (forall (i:nat). i < sS ==>
      S.index tokens i == (if i = dest then S.index selected 0 else S.index old_tokens i)) /\
   (forall (k:nat). k < sS * 1024 ==>
      S.index history k == (if k / 1024 = dest then S.index hidden (k % 1024) else S.index old_history k)))

(*** Optimized relations *)

let optimized_argmax_relation (scores:S.seq F32.t) (tokens:S.seq I64.t) (d:int)
                              (selected tokens':S.seq I64.t) : prop =
  (S.length scores == 248320 /\ 0 <= d /\ d < S.length tokens) ==>
  (S.length selected == 1 /\ tok_at selected 0 == tree_argmax scores /\
   allowed_choice scores (tok_at selected 0) /\
   S.length tokens' == S.length tokens /\
   (forall (i:nat). i < S.length tokens ==>
      S.index tokens' i == (if i = d then S.index selected 0 else S.index tokens i)))

let optimized_attention_decoder_relation (w:S.seq BF.t) (x:S.seq F32.t) (k v:S.seq BF.t)
                                         (cos sin:S.seq F32.t) (base fa pos:int)
                                         (x':S.seq F32.t) (k' v':S.seq BF.t) : prop =
  (0 <= base /\ base + attn_len <= S.length w /\ S.length x == 1024 /\
   S.length k == opt_kv_len /\ S.length v == opt_kv_len /\ S.length cos == 32 /\ S.length sin == 32 /\
   0 <= fa /\ fa < 6 /\ 0 <= pos /\ pos < 2048) ==>
  (S.length x' == 1024 /\ S.length k' == opt_kv_len /\ S.length v' == opt_kv_len /\
   (forall (i:nat). i < opt_kv_len /\ not (opt_kv_hit fa pos i) ==>
      S.index k' i == S.index k i /\ S.index v' i == S.index v i) /\
   (forall (rw rx rk rv rc rs:S.seq real).
      w %~ rw /\ x %~ rx /\ k %~ rk /\ v %~ rv /\ cos %~ rc /\ sin %~ rs ==>
      x' %~ at_out opt_numerics rw base rx rc rs pos (opt_hk fa rk) (opt_hk fa rv) /\
      k' %~ opt_kv_write fa pos rk (at_k opt_numerics rw base rx rc rs) /\
      v' %~ opt_kv_write fa pos rv (at_v opt_numerics rw base rx)))

let optimized_embedding_relation (table:S.seq BF.t) (prompt tokens:S.seq I64.t) (p:int)
                                 (hidden cos sin:S.seq F32.t) : prop =
  (S.length table == table_len /\ 1 <= S.length prompt /\ 1 <= S.length tokens /\
   0 <= p /\ p < 12 /\ p < S.length prompt + S.length tokens - 1 /\
   valid_token (opt_emb_token prompt tokens p)) ==>
  (S.length hidden == 1024 /\ S.length cos == 32 /\ S.length sin == 32 /\
   (forall (rt:S.seq real). table %~ rt ==> hidden %~ embed rt (opt_emb_token prompt tokens p)) /\
   cos %~ rope_cos opt_numerics p /\ sin %~ rope_sin opt_numerics p)

let optimized_final_projection_relation (table normw:S.seq BF.t) (x:S.seq F32.t) (history:S.seq BF.t)
                                        (d:int) (scores:S.seq F32.t) (history':S.seq BF.t) : prop =
  (S.length table == table_len /\ S.length normw == 1024 /\ S.length x == 1024 /\
   S.length history % 1024 == 0 /\ 0 <= d /\ d < S.length history / 1024) ==>
  (S.length scores == 248320 /\ S.length history' == S.length history /\
   (forall (k:nat). k < S.length history /\ k / 1024 <> d ==> S.index history' k == S.index history k) /\
   (forall (rt rn rx:S.seq real). table %~ rt /\ normw %~ rn /\ x %~ rx ==>
      scores %~ logits opt_numerics rt 0 rn 0 rx /\ bf_row history' d %~ rx))

let optimized_inactive_decode_relation (rc cv:S.seq F32.t) (k v:S.seq BF.t) (tokens:S.seq I64.t)
                                       (history:S.seq BF.t) (p l:int)
                                       (rc' cv':S.seq F32.t) (k' v':S.seq BF.t)
                                       (tokens':S.seq I64.t) (history':S.seq BF.t) : prop =
  (S.length rc == opt_rec_len /\ S.length cv == opt_conv_len /\
   S.length k == opt_kv_len /\ S.length v == opt_kv_len /\
   1 <= S.length tokens /\ S.length history == S.length tokens * 1024 /\
   0 < l /\ l + S.length tokens - 1 <= p /\ p < 12) ==>
  (rc' == rc /\ cv' == cv /\ k' == k /\ v' == v /\ tokens' == tokens /\ history' == history)

let optimized_initialize_relation (prompt:S.seq I64.t) (steps:int) (rc cv:S.seq F32.t)
                                  (k v:S.seq BF.t) (tokens:S.seq I64.t) (history:S.seq BF.t) : prop =
  (1 <= S.length prompt /\ 1 <= steps /\ steps <= 12) ==>
  (S.length rc == opt_rec_len /\ all_zero rc /\ S.length cv == opt_conv_len /\ all_zero cv /\
   S.length k == opt_kv_len /\ all_zero k /\ S.length v == opt_kv_len /\ all_zero v /\
   S.length tokens == steps /\ tok_at tokens 0 == tok_at prompt 0 /\
   (forall (i:nat). 0 < i /\ i < steps ==> tok_at tokens i == 0) /\
   S.length history == steps * 1024 /\ all_zero history)

let optimized_recurrent_decoder_relation (w:S.seq BF.t) (x rc cv:S.seq F32.t) (base dn:int)
                                         (x' rc' cv':S.seq F32.t) : prop =
  (0 <= base /\ base + dn_len <= S.length w /\ S.length x == 1024 /\
   S.length rc == opt_rec_len /\ S.length cv == opt_conv_len /\ 0 <= dn /\ dn < 18) ==>
  (S.length x' == 1024 /\ S.length rc' == opt_rec_len /\ S.length cv' == opt_conv_len /\
   (forall (i:nat). i < opt_rec_len /\ i / 262144 <> dn ==> S.index rc' i == S.index rc i) /\
   (forall (i:nat). i < opt_conv_len /\ i / 24576 <> dn ==> S.index cv' i == S.index cv i) /\
   (forall (j:nat). j + 1 < 24576 /\ j % 4 < 3 ==> S.index cv' (dn * 24576 + j) == S.index cv (dn * 24576 + j + 1)) /\
   (forall (rw rx rr rcv:S.seq real).
      w %~ rw /\ x %~ rx /\ rc %~ rr /\ cv %~ rcv ==>
      x' %~ dn_out opt_numerics rw base rx (opt_conv_slot rcv dn) (opt_rec_slot rr dn) /\
      cv' %~ opt_conv_put rcv dn (dn_conv opt_numerics rw base rx (opt_conv_slot rcv dn)) /\
      rc' %~ opt_rec_put rr dn (dn_state opt_numerics rw base rx (opt_conv_slot rcv dn) (opt_rec_slot rr dn))))
