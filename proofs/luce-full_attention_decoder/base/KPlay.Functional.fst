module KPlay.Functional

#lang-pulse

open Kuiper.TensorCore
open Kuiper.Tensor
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as row_major, l2_col_major as col_major }
open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = Kuiper.SizeT
module I64 = FStar.Int64
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32
module CB = Kuiper.Float.Casts.Base
module FB = Kuiper.Floating.Base
module FM = Kuiper.Float32.FastMath
module MM = Kuiper.TensorCore.MMA

instance is_send_global_pts_to (#a:Type0) (x:array a { is_global x }) (#f:perm) (v:S.seq a)
  : is_send_across gpu_of (pts_to x #f v)
  = let i : is_send_across (visibility_of x) (pts_to x #f v) = solve in i

inline_for_extraction noextract
let fb (x : bf16) : f32 = CB.cast_bf16_to_f32 x
inline_for_extraction noextract
let tb (x : f32) : bf16 = CB.cast_f32_to_bf16 x

inline_for_extraction noextract
let f_eps : f32 = FB.of_literal #f32 "0.0000009999999974752427078783512115478515625"
inline_for_extraction noextract
let f_inv1024 : f32 = FB.of_literal #f32 "0.0009765625"
inline_for_extraction noextract
let f_inv256 : f32 = FB.of_literal #f32 "0.00390625"
inline_for_extraction noextract
let f_sixteenth : f32 = FB.of_literal #f32 "0.0625"

inline_for_extraction noextract
let sigm (x : f32) : f32 = FB.div one (add one (FB.fexp (FB.neg x)))
inline_for_extraction noextract
let silu_f (x : f32) : f32 = FB.div x (add one (FB.fexp (FB.neg x)))

(*** Native models: exact f32/bf16 semantics of every kernel *)

noextract
let fat (s:S.seq f32) (i:int) : f32 = if 0 <= i && i < S.length s then S.index s i else zero
noextract
let bat (s:S.seq bf16) (i:int) : bf16 = if 0 <= i && i < S.length s then S.index s i else zero

(* sequential sum from zero: ((0 + f 0) + f 1) + ... *)
noextract
let rec fsum (n:nat) (f:nat -> f32) : f32 = if n = 0 then zero else add (fsum (n - 1) f) (f (n - 1))

noextract
let halve (s:S.seq f32) (off:nat) : S.seq f32 =
  S.init (S.length s) (fun (i:nat{i < S.length s}) ->
    if i < off && i + off < S.length s then add (S.index s i) (S.index s (i + off)) else S.index s i)

(* in-place halving tree with offsets off, off/2, ..., 1 *)
noextract
let rec tree (s:S.seq f32) (off:nat) : Tot (S.seq f32) (decreases off) =
  if off = 0 then s else tree (halve s off) (off / 2)

(* tree reduction of f 0 .. f (cnt-1), offsets cnt/2 .. 1 *)
noextract
let tsum (cnt:nat) (f:nat -> f32) : f32 =
  fat (tree (S.init cnt (fun (i:nat{i < cnt}) -> f i)) (cnt / 2)) 0

noextract
let sqf (x:bf16) : f32 = mul (fb x) (fb x)

(* sum of squares of src[base .. base + 4*cnt) as cnt sequential groups of 4 *)
noextract
let sq4 (src:S.seq bf16) (base:int) (g:nat) : f32 = fsum 4 (fun j -> sqf (bat src (base + 4 * g + j)))

noextract
let rms_inv (src:S.seq bf16) (base:int) (cnt:nat) (c:f32) : f32 =
  FB.rsqrt (add (mul (tsum cnt (sq4 src base)) c) f_eps)

noextract
let rms_out (src w:S.seq bf16) (base woff:int) (inv:f32) (i:int) : bf16 =
  tb (mul (mul (fb (bat src (base + i))) inv) (add one (fb (bat w (woff + i)))))

(* rms_k: dst[i] = rms1024 src w woff i *)
noextract
let rms1024 (src w:S.seq bf16) (woff:int) (i:int) : bf16 =
  rms_out src w 0 woff (rms_inv src 0 256 f_inv1024) i

(* hn_k, one head starting at src[base]: per-head norm then RoPE on dims < 64 *)
noextract
let hnorm (src w:S.seq bf16) (base woff:int) (i:int) : bf16 =
  rms_out src w base woff (rms_inv src base 64 f_inv256) i

noextract
let hn_m (src w cs sn:S.seq bf16) (base woff:int) (i:nat) : bf16 =
  if i < 32 then
    (let a = fb (hnorm src w base woff i) in
     let b = fb (hnorm src w base woff (i + 32)) in
     let c = fb (bat cs i) in
     let s = fb (bat sn i) in
     tb (add (fb (tb (mul a c))) (fb (tb (mul (FB.neg b) s)))))
  else if i < 64 then
    (let r = i - 32 in
     let a = fb (hnorm src w base woff r) in
     let b = fb (hnorm src w base woff i) in
     let c = fb (bat cs r) in
     let s = fb (bat sn r) in
     tb (add (fb (tb (mul b c))) (fb (tb (mul a s)))))
  else hnorm src w base woff i

noextract
let resid_m (a b:S.seq bf16) (i:int) : bf16 = tb (add (fb (bat a i)) (fb (bat b i)))

noextract
let kvw_m (c t:int) (kv kn pv:S.seq bf16) (i:int) : bf16 =
  if base_kv_pos c i = t then (if i < c * 512 then bat kn (base_kv_comp c i) else bat pv (base_kv_comp c i))
  else bat kv i

(* score_k: q . key of head h, position p (key from the written cache kvo) *)
noextract
let dot16 (qn kvo:S.seq bf16) (c h p:int) (g:nat) : f32 =
  fsum 16 (fun j -> mul (fb (bat qn (h * 256 + 16 * g + j))) (fb (bat kvo (((h / 4) * c + p) * 256 + 16 * g + j))))

noextract
let score_m (qn kvo:S.seq bf16) (c h p:int) : bf16 = tb (tsum 16 (dot16 qn kvo c h p))

(* softmax_k *)
noextract
let sm_s (dots:S.seq bf16) (h p:int) : f32 = fb (tb (mul (fb (bat dots (h * 16 + p))) f_sixteenth))

noextract
let rec sm_max (dots:S.seq bf16) (h:int) (n:nat) : f32 =
  if n = 0 then sm_s dots h 0
  else (let m = sm_max dots h (n - 1) in if lt m (sm_s dots h n) then sm_s dots h n else m)

noextract
let sm_e (dots:S.seq bf16) (t:nat) (h p:int) : f32 = FB.fexp (FB.sub (sm_s dots h p) (sm_max dots h t))

noextract
let sm_width (t:nat) : nat = if t < 1 then 1 else if t < 2 then 2 else if t < 4 then 4 else if t < 8 then 8 else 16

noextract
let sm_sum (dots:S.seq bf16) (t:nat) (h:int) : f32 =
  tsum (sm_width t) (fun p -> if p <= t then sm_e dots t h p else zero)

noextract
let prob_m (dots:S.seq bf16) (t:nat) (h p:int) : bf16 =
  if 0 <= p && p <= t then tb (FB.div (sm_e dots t h p) (sm_sum dots t h)) else zero

(* vals_k *)
noextract
let vv (kvo:S.seq bf16) (c h p i:int) : f32 = fb (bat kvo (c * 512 + ((h / 4) * c + p) * 256 + i))
noextract
let pr (probs:S.seq bf16) (h p:int) : f32 = fb (bat probs (h * 16 + p))

(* real model of the fma partials: group g after positions 0 .. n-1 *)
noextract
let rec rgrp (rp rv:nat -> real) (g:int) (n:nat) : real =
  if n = 0 then 0.0R
  else (let prev = rgrp rp rv g (n - 1) in if (n - 1) / 4 = g then rp (n - 1) *. rv (n - 1) +. prev else prev)

(* vals_k post (fma_rn only has an approximation contract) *)
noextract
let vals_ok (probs kvo:S.seq bf16) (c t:nat) (vals:S.seq f32) : prop =
  S.length vals == 2048 /\
  (forall (h i:nat) (rp rv:nat -> real).
    h < 8 /\ i < 256 /\ (forall (p:nat). p <= t ==> pr probs h p %~ rp p /\ vv kvo c h p i %~ rv p) ==>
    S.index vals (h * 256 + i) %~ vsum (t + 1) (fun p -> rp p *. rv p))

noextract
let gate_m (vals:S.seq f32) (pq:S.seq bf16) (i:int) : bf16 =
  tb (mul (fb (tb (fat vals i))) (fb (tb (sigm (fb (bat pq ((i / 256) * 512 + 256 + i % 256)))))))

noextract
let act_m (g u:S.seq bf16) (i:int) : bf16 = tb (mul (fb (tb (silu_f (fb (bat g i))))) (fb (bat u i)))

(* t == 7 operand matrices *)
noextract
let keys_m (kvo:S.seq bf16) (c h i:int) : bf16 =
  if i / 256 < 8 then bat kvo (((h / 4) * c + i / 256) * 256 + i % 256) else zero

noextract
let valm_m (kvo:S.seq bf16) (c h i:int) : bf16 =
  if i % 16 < 8 then bat kvo (c * 512 + ((h / 4) * c + i % 16) * 256 + i / 16) else zero

(* tensor-core projection: tile r of C after k-tiles j0 .. j1-1, accumulator from zero *)
noextract
let rec mma_tile (#n #k:pos) (eA:chest2 bf16 n k) (eB:chest2 bf16 k 8)
  (#_:squash (16 /? n /\ 16 /? k)) (r:natlt (n / 16)) (j0:nat) (j1:nat{j1 <= k / 16})
  : Tot (chest2 f32 16 8) (decreases j1) =
  if j1 <= j0 then const (16 @| 8 @| INil) zero
  else MM.emma (mma_tile eA eB r j0 (j1 - 1)) (ematrix_subtile eA 16 16 r (j1 - 1)) (ematrix_subtile eB 16 8 (j1 - 1) 0)

noextract
let mma_model (#n #k:pos) (eA:chest2 bf16 n k) (eB:chest2 bf16 k 8)
  (#_:squash (16 /? n /\ 16 /? k)) (j0:nat) (j1:nat{j1 <= k / 16}) : chest2 f32 n 8 =
  ematrix_from_tiles 16 8 (fun (r:natlt (n / 16)) (_:natlt (8 / 8)) -> mma_tile eA eB r j0 j1)
module Lit = Kuiper.Float32.Literals
module CF = Kuiper.Float32.ConstantFold
module Cert = Kuiper.Float32.ConstantFold.Certificates
let lit_value_gen (s:string) : Lemma (Lit.value s == (match Lit.integral (FStar.String.list_of_string s) 0 false with Some (n,d) -> FStar.Real.of_int n /. FStar.Real.of_int d | None -> 0.0R)) = ()
let lit_cert_gen (s:string) (m:nat) (e:int) (n:nat) (d:pos) (un ud:pos) : Lemma
  (requires Lit.integral (FStar.String.list_of_string s) 0 false == Some (n, d) /\
            Cert.unit_fraction e == (un, ud) /\ Cert.normal_encoding m e /\ n * ud = m * un * d)
  (ensures Cert.exact_normal s m e) = ()
let lit_inv1024_chars () : Lemma (FStar.String.list_of_string "0.0009765625" == ['0';'.';'0';'0';'0';'9';'7';'6';'5';'6';'2';'5']) = ()
#push-options "--fuel 64 --ifuel 2 --z3rlimit 400"
let lit_inv1024_int () : Lemma (Lit.integral ['0';'.';'0';'0';'0';'9';'7';'6';'5';'6';'2';'5'] 0 false == Some (9765625, 10000000000)) = ()
let lit_inv1024_uf () : Lemma (Cert.unit_fraction (-33) == (1, 8589934592)) = ()
#pop-options
let lit_inv1024_value () : Lemma (Lit.value "0.0009765625" == FStar.Real.of_int 9765625 /. FStar.Real.of_int 10000000000) =
  lit_value_gen "0.0009765625"; lit_inv1024_chars (); lit_inv1024_int ()
let lit_inv1024_cert () : Lemma (Cert.exact_normal "0.0009765625" 8388608 (-33)) =
  lit_inv1024_chars (); lit_inv1024_int (); lit_inv1024_uf ();
  lit_cert_gen "0.0009765625" 8388608 (-33) 9765625 10000000000 1 8589934592

let lit_inv256_chars () : Lemma (FStar.String.list_of_string "0.00390625" == ['0';'.';'0';'0';'3';'9';'0';'6';'2';'5']) = ()
#push-options "--fuel 64 --ifuel 2 --z3rlimit 400"
let lit_inv256_int () : Lemma (Lit.integral ['0';'.';'0';'0';'3';'9';'0';'6';'2';'5'] 0 false == Some (390625, 100000000)) = ()
let lit_inv256_uf () : Lemma (Cert.unit_fraction (-31) == (1, 2147483648)) = ()
#pop-options
let lit_inv256_value () : Lemma (Lit.value "0.00390625" == FStar.Real.of_int 390625 /. FStar.Real.of_int 100000000) =
  lit_value_gen "0.00390625"; lit_inv256_chars (); lit_inv256_int ()
let lit_inv256_cert () : Lemma (Cert.exact_normal "0.00390625" 8388608 (-31)) =
  lit_inv256_chars (); lit_inv256_int (); lit_inv256_uf ();
  lit_cert_gen "0.00390625" 8388608 (-31) 390625 100000000 1 2147483648

let lit_eps_chars () : Lemma (FStar.String.list_of_string "0.0000009999999974752427078783512115478515625" == ['0';'.';'0';'0';'0';'0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5']) = ()
#push-options "--fuel 2 --ifuel 1 --z3rlimit 100"
let lit_fr_step (c:FStar.Char.char) (r:list FStar.Char.char) (n:nat) (d:pos) (v:nat) (n1:nat) (d1:pos) : Lemma (requires Lit.digit c == Some v /\ n1 == 10 * n + v /\ d1 == 10 * d) (ensures Lit.fraction (c::r) n d == Lit.fraction r n1 d1) = ()
let lit_eps_fr43 () : Lemma (Lit.fraction [] 9999999974752427078783512115478515625 10000000000000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = ()
let lit_eps_fr42 () : Lemma (Lit.fraction ['5'] 999999997475242707878351211547851562 1000000000000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '5' [] 999999997475242707878351211547851562 1000000000000000000000000000000000000000000 5 9999999974752427078783512115478515625 10000000000000000000000000000000000000000000; lit_eps_fr43 ()
let lit_eps_fr41 () : Lemma (Lit.fraction ['2';'5'] 99999999747524270787835121154785156 100000000000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '2' ['5'] 99999999747524270787835121154785156 100000000000000000000000000000000000000000 2 999999997475242707878351211547851562 1000000000000000000000000000000000000000000; lit_eps_fr42 ()
let lit_eps_fr40 () : Lemma (Lit.fraction ['6';'2';'5'] 9999999974752427078783512115478515 10000000000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '6' ['2';'5'] 9999999974752427078783512115478515 10000000000000000000000000000000000000000 6 99999999747524270787835121154785156 100000000000000000000000000000000000000000; lit_eps_fr41 ()
let lit_eps_fr39 () : Lemma (Lit.fraction ['5';'6';'2';'5'] 999999997475242707878351211547851 1000000000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '5' ['6';'2';'5'] 999999997475242707878351211547851 1000000000000000000000000000000000000000 5 9999999974752427078783512115478515 10000000000000000000000000000000000000000; lit_eps_fr40 ()
let lit_eps_fr38 () : Lemma (Lit.fraction ['1';'5';'6';'2';'5'] 99999999747524270787835121154785 100000000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '1' ['5';'6';'2';'5'] 99999999747524270787835121154785 100000000000000000000000000000000000000 1 999999997475242707878351211547851 1000000000000000000000000000000000000000; lit_eps_fr39 ()
let lit_eps_fr37 () : Lemma (Lit.fraction ['5';'1';'5';'6';'2';'5'] 9999999974752427078783512115478 10000000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '5' ['1';'5';'6';'2';'5'] 9999999974752427078783512115478 10000000000000000000000000000000000000 5 99999999747524270787835121154785 100000000000000000000000000000000000000; lit_eps_fr38 ()
let lit_eps_fr36 () : Lemma (Lit.fraction ['8';'5';'1';'5';'6';'2';'5'] 999999997475242707878351211547 1000000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '8' ['5';'1';'5';'6';'2';'5'] 999999997475242707878351211547 1000000000000000000000000000000000000 8 9999999974752427078783512115478 10000000000000000000000000000000000000; lit_eps_fr37 ()
let lit_eps_fr35 () : Lemma (Lit.fraction ['7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524270787835121154 100000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '7' ['8';'5';'1';'5';'6';'2';'5'] 99999999747524270787835121154 100000000000000000000000000000000000 7 999999997475242707878351211547 1000000000000000000000000000000000000; lit_eps_fr36 ()
let lit_eps_fr34 () : Lemma (Lit.fraction ['4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427078783512115 10000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '4' ['7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427078783512115 10000000000000000000000000000000000 4 99999999747524270787835121154 100000000000000000000000000000000000; lit_eps_fr35 ()
let lit_eps_fr33 () : Lemma (Lit.fraction ['5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242707878351211 1000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '5' ['4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242707878351211 1000000000000000000000000000000000 5 9999999974752427078783512115 10000000000000000000000000000000000; lit_eps_fr34 ()
let lit_eps_fr32 () : Lemma (Lit.fraction ['1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524270787835121 100000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '1' ['5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524270787835121 100000000000000000000000000000000 1 999999997475242707878351211 1000000000000000000000000000000000; lit_eps_fr33 ()
let lit_eps_fr31 () : Lemma (Lit.fraction ['1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427078783512 10000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '1' ['1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427078783512 10000000000000000000000000000000 1 99999999747524270787835121 100000000000000000000000000000000; lit_eps_fr32 ()
let lit_eps_fr30 () : Lemma (Lit.fraction ['2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242707878351 1000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '2' ['1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242707878351 1000000000000000000000000000000 2 9999999974752427078783512 10000000000000000000000000000000; lit_eps_fr31 ()
let lit_eps_fr29 () : Lemma (Lit.fraction ['1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524270787835 100000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '1' ['2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524270787835 100000000000000000000000000000 1 999999997475242707878351 1000000000000000000000000000000; lit_eps_fr30 ()
let lit_eps_fr28 () : Lemma (Lit.fraction ['5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427078783 10000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '5' ['1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427078783 10000000000000000000000000000 5 99999999747524270787835 100000000000000000000000000000; lit_eps_fr29 ()
let lit_eps_fr27 () : Lemma (Lit.fraction ['3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242707878 1000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '3' ['5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242707878 1000000000000000000000000000 3 9999999974752427078783 10000000000000000000000000000; lit_eps_fr28 ()
let lit_eps_fr26 () : Lemma (Lit.fraction ['8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524270787 100000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '8' ['3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524270787 100000000000000000000000000 8 999999997475242707878 1000000000000000000000000000; lit_eps_fr27 ()
let lit_eps_fr25 () : Lemma (Lit.fraction ['7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427078 10000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '7' ['8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427078 10000000000000000000000000 7 99999999747524270787 100000000000000000000000000; lit_eps_fr26 ()
let lit_eps_fr24 () : Lemma (Lit.fraction ['8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242707 1000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '8' ['7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242707 1000000000000000000000000 8 9999999974752427078 10000000000000000000000000; lit_eps_fr25 ()
let lit_eps_fr23 () : Lemma (Lit.fraction ['7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524270 100000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '7' ['8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524270 100000000000000000000000 7 999999997475242707 1000000000000000000000000; lit_eps_fr24 ()
let lit_eps_fr22 () : Lemma (Lit.fraction ['0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427 10000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '0' ['7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427 10000000000000000000000 0 99999999747524270 100000000000000000000000; lit_eps_fr23 ()
let lit_eps_fr21 () : Lemma (Lit.fraction ['7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242 1000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '7' ['0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242 1000000000000000000000 7 9999999974752427 10000000000000000000000; lit_eps_fr22 ()
let lit_eps_fr20 () : Lemma (Lit.fraction ['2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524 100000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '2' ['7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524 100000000000000000000 2 999999997475242 1000000000000000000000; lit_eps_fr21 ()
let lit_eps_fr19 () : Lemma (Lit.fraction ['4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752 10000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '4' ['2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752 10000000000000000000 4 99999999747524 100000000000000000000; lit_eps_fr20 ()
let lit_eps_fr18 () : Lemma (Lit.fraction ['2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475 1000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '2' ['4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475 1000000000000000000 2 9999999974752 10000000000000000000; lit_eps_fr19 ()
let lit_eps_fr17 () : Lemma (Lit.fraction ['5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747 100000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '5' ['2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747 100000000000000000 5 999999997475 1000000000000000000; lit_eps_fr18 ()
let lit_eps_fr16 () : Lemma (Lit.fraction ['7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974 10000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '7' ['5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974 10000000000000000 7 99999999747 100000000000000000; lit_eps_fr17 ()
let lit_eps_fr15 () : Lemma (Lit.fraction ['4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997 1000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '4' ['7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997 1000000000000000 4 9999999974 10000000000000000; lit_eps_fr16 ()
let lit_eps_fr14 () : Lemma (Lit.fraction ['7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999 100000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '7' ['4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999 100000000000000 7 999999997 1000000000000000; lit_eps_fr15 ()
let lit_eps_fr13 () : Lemma (Lit.fraction ['9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999 10000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '9' ['7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999 10000000000000 9 99999999 100000000000000; lit_eps_fr14 ()
let lit_eps_fr12 () : Lemma (Lit.fraction ['9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999 1000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '9' ['9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999 1000000000000 9 9999999 10000000000000; lit_eps_fr13 ()
let lit_eps_fr11 () : Lemma (Lit.fraction ['9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999 100000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '9' ['9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999 100000000000 9 999999 1000000000000; lit_eps_fr12 ()
let lit_eps_fr10 () : Lemma (Lit.fraction ['9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999 10000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '9' ['9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999 10000000000 9 99999 100000000000; lit_eps_fr11 ()
let lit_eps_fr9 () : Lemma (Lit.fraction ['9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999 1000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '9' ['9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999 1000000000 9 9999 10000000000; lit_eps_fr10 ()
let lit_eps_fr8 () : Lemma (Lit.fraction ['9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99 100000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '9' ['9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99 100000000 9 999 1000000000; lit_eps_fr9 ()
let lit_eps_fr7 () : Lemma (Lit.fraction ['9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9 10000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '9' ['9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9 10000000 9 99 100000000; lit_eps_fr8 ()
let lit_eps_fr6 () : Lemma (Lit.fraction ['9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 1000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '9' ['9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 1000000 9 9 10000000; lit_eps_fr7 ()
let lit_eps_fr5 () : Lemma (Lit.fraction ['0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 100000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '0' ['9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 100000 0 0 1000000; lit_eps_fr6 ()
let lit_eps_fr4 () : Lemma (Lit.fraction ['0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 10000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '0' ['0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 10000 0 0 100000; lit_eps_fr5 ()
let lit_eps_fr3 () : Lemma (Lit.fraction ['0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 1000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '0' ['0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 1000 0 0 10000; lit_eps_fr4 ()
let lit_eps_fr2 () : Lemma (Lit.fraction ['0';'0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 100 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '0' ['0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 100 0 0 1000; lit_eps_fr3 ()
let lit_eps_fr1 () : Lemma (Lit.fraction ['0';'0';'0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 10 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '0' ['0';'0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 10 0 0 100; lit_eps_fr2 ()
let lit_eps_fr0 () : Lemma (Lit.fraction ['0';'0';'0';'0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 1 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_fr_step '0' ['0';'0';'0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 1 0 0 10; lit_eps_fr1 ()
let lit_int_step (r:list FStar.Char.char{Cons? r}) : Lemma (Lit.integral ('0'::'.'::r) 0 false == Lit.fraction r 0 1) = ()
let lit_eps_int () : Lemma (Lit.integral ['0';'.';'0';'0';'0';'0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 false == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) = lit_int_step (FStar.List.Tot.tl (FStar.List.Tot.tl ['0';'.';'0';'0';'0';'0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'])); lit_eps_fr0 ()
#pop-options
#push-options "--fuel 64 --ifuel 2 --z3rlimit 400"
let lit_eps_uf () : Lemma (Cert.unit_fraction (-43) == (1, 8796093022208)) = ()
#pop-options
let lit_eps_value () : Lemma (Lit.value "0.0000009999999974752427078783512115478515625" == FStar.Real.of_int 9999999974752427078783512115478515625 /. FStar.Real.of_int 10000000000000000000000000000000000000000000) =
  lit_value_gen "0.0000009999999974752427078783512115478515625"; lit_eps_chars (); lit_eps_int ()
let lit_eps_cert () : Lemma (Cert.exact_normal "0.0000009999999974752427078783512115478515625" 8796093 (-43)) =
  lit_eps_chars (); lit_eps_int (); lit_eps_uf ();
  lit_cert_gen "0.0000009999999974752427078783512115478515625" 8796093 (-43) 9999999974752427078783512115478515625 10000000000000000000000000000000000000000000 1 8796093022208

let lit_sixteenth_chars () : Lemma (FStar.String.list_of_string "0.0625" == ['0';'.';'0';'6';'2';'5']) = ()
#push-options "--fuel 64 --ifuel 2 --z3rlimit 400"
let lit_sixteenth_int () : Lemma (Lit.integral ['0';'.';'0';'6';'2';'5'] 0 false == Some (625, 10000)) = ()
let lit_sixteenth_uf () : Lemma (Cert.unit_fraction (-27) == (1, 134217728)) = ()
#pop-options
let lit_sixteenth_value () : Lemma (Lit.value "0.0625" == FStar.Real.of_int 625 /. FStar.Real.of_int 10000) =
  lit_value_gen "0.0625"; lit_sixteenth_chars (); lit_sixteenth_int ()
let lit_sixteenth_cert () : Lemma (Cert.exact_normal "0.0625" 8388608 (-27)) =
  lit_sixteenth_chars (); lit_sixteenth_int (); lit_sixteenth_uf ();
  lit_cert_gen "0.0625" 8388608 (-27) 625 10000 1 134217728
let f_eps_approx () : Lemma (f_eps %~ eps_f32 /\ eps_f32 >. 0.0R) =
  lit_eps_cert (); lit_eps_value ();
  CF.literal_approx "0.0000009999999974752427078783512115478515625" 8796093 (-43)
let f_inv1024_approx () : Lemma (f_inv1024 %~ (1.0R /. 1024.0R)) =
  lit_inv1024_cert (); lit_inv1024_value ();
  CF.literal_approx "0.0009765625" 8388608 (-33)
let f_inv256_approx () : Lemma (f_inv256 %~ (1.0R /. 256.0R)) =
  lit_inv256_cert (); lit_inv256_value ();
  CF.literal_approx "0.00390625" 8388608 (-31)
let f_sixteenth_approx () : Lemma (f_sixteenth %~ 0.0625R) =
  lit_sixteenth_cert (); lit_sixteenth_value ();
  CF.literal_approx "0.0625" 8388608 (-27)

(*** Sums: real vsum algebra, float sum/tree approximation, rms and per-head rms stages *)

noextract
let rec is_pow2 (n:nat) : bool = n = 1 || (n >= 2 && n % 2 = 0 && is_pow2 (n / 2))

#push-options "--fuel 1 --ifuel 0 --z3rlimit 20"
let su_pow2_double (n:nat) : Lemma (requires is_pow2 n) (ensures is_pow2 (2 * n)) = ()
#pop-options

let pow2_facts () : Lemma (is_pow2 1 /\ is_pow2 2 /\ is_pow2 4 /\ is_pow2 8 /\ is_pow2 16 /\ is_pow2 64 /\ is_pow2 256) =
  su_pow2_double 1; su_pow2_double 2; su_pow2_double 4; su_pow2_double 8;
  su_pow2_double 16; su_pow2_double 32; su_pow2_double 64; su_pow2_double 128

#push-options "--fuel 2 --ifuel 1 --z3rlimit 40"
let su_rsum_empty () : Lemma (rsum (S.empty #real) == 0.0R) = ()

let su_rsum_single (x:real) : Lemma (rsum (S.create 1 x) == x) =
  assert (S.equal (S.create 1 x) (S.cons x (S.empty #real)));
  Kuiper.Seq.Common.lemma_view_seq_cons x (S.empty #real);
  assert (S.length (S.tail (S.create 1 x)) == 0);
  assert (S.equal (S.tail (S.create 1 x)) (S.empty #real))
#pop-options

noextract
let su_seq (n:nat) (f:nat -> real) : S.seq real = S.init n (fun (k:nat{k < n}) -> f k)

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let su_vsum_def (n:nat) (f:nat -> real) : Lemma (vsum n f == rsum (su_seq n f)) = ()

let su_vsum_zero (f:nat -> real) : Lemma (vsum 0 f == 0.0R) =
  su_vsum_def 0 f;
  assert (S.equal (su_seq 0 f) (S.empty #real));
  su_rsum_empty ()

let vsum_ext (n:nat) (f g:nat -> real)
  : Lemma (requires forall (i:nat). i < n ==> f i == g i) (ensures vsum n f == vsum n g) =
  su_vsum_def n f; su_vsum_def n g;
  assert (S.equal (su_seq n f) (su_seq n g))

let vsum_split (n m:nat) (f:nat -> real)
  : Lemma (vsum (n + m) f == vsum n f +. vsum m (fun i -> f (n + i))) =
  let g = (fun (i:nat) -> f (n + i)) in
  su_vsum_def (n + m) f; su_vsum_def n f; su_vsum_def m g;
  assert (S.equal (su_seq (n + m) f) (S.append (su_seq n f) (su_seq m g)));
  Kuiper.Real.rsum_append (su_seq n f) (su_seq m g)

let vsum_succ (n:nat) (f:nat -> real) : Lemma (vsum (n + 1) f == vsum n f +. f n) =
  let g = (fun (i:nat) -> f (n + i)) in
  vsum_split n 1 f;
  su_vsum_def 1 g;
  assert (S.equal (su_seq 1 g) (S.create 1 (f n)));
  su_rsum_single (f n)

let vsum_one (f:nat -> real) : Lemma (vsum 1 f == f 0) =
  vsum_succ 0 f; su_vsum_zero f
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let rec su_vsum_zeros (m:nat) (f:nat -> real)
  : Lemma (requires forall (i:nat). i < m ==> f i == 0.0R) (ensures vsum m f == 0.0R) (decreases m) =
  if m = 0 then su_vsum_zero f
  else (su_vsum_zeros (m - 1) f; vsum_succ (m - 1) f)

let vsum_zero_tail (n m:nat) (f:nat -> real)
  : Lemma (requires n <= m /\ (forall (i:nat). n <= i /\ i < m ==> f i == 0.0R)) (ensures vsum m f == vsum n f) =
  vsum_split n (m - n) f;
  su_vsum_zeros (m - n) (fun i -> f (n + i))

let rec vsum_block (n m:nat) (f:nat -> real)
  : Lemma (ensures vsum (n * m) f == vsum n (fun g -> vsum m (fun j -> f (g * m + j)))) (decreases n) =
  if n = 0 then (su_vsum_zero f; su_vsum_zero (fun g -> vsum m (fun j -> f (g * m + j))))
  else begin
    vsum_block (n - 1) m f;
    vsum_split ((n - 1) * m) m f;
    assert ((n - 1) * m + m == n * m);
    vsum_succ (n - 1) (fun g -> vsum m (fun j -> f (g * m + j)));
    vsum_ext m (fun i -> f ((n - 1) * m + i)) (fun j -> f ((n - 1) * m + j))
  end

let rec su_vsum_add (n:nat) (h f g:nat -> real)
  : Lemma (requires forall (i:nat). i < n ==> h i == f i +. g i)
          (ensures vsum n h == vsum n f +. vsum n g) (decreases n) =
  if n = 0 then (su_vsum_zero f; su_vsum_zero g; su_vsum_zero h)
  else (su_vsum_add (n - 1) h f g; vsum_succ (n - 1) f; vsum_succ (n - 1) g; vsum_succ (n - 1) h)

let su_vsum_split (n m:nat) (f g:nat -> real)
  : Lemma (requires forall (i:nat). i < m ==> g i == f (n + i))
          (ensures vsum (n + m) f == vsum n f +. vsum m g) =
  vsum_split n m f;
  vsum_ext m (fun i -> f (n + i)) g

let rec su_vsum_nonneg (n:nat) (f:nat -> real)
  : Lemma (requires forall (i:nat). i < n ==> f i >=. 0.0R) (ensures vsum n f >=. 0.0R) (decreases n) =
  if n = 0 then su_vsum_zero f
  else (su_vsum_nonneg (n - 1) f; vsum_succ (n - 1) f)

let rec su_vsum_pos (n:nat) (f:nat -> real)
  : Lemma (requires n >= 1 /\ (forall (i:nat). i < n ==> f i >. 0.0R)) (ensures vsum n f >. 0.0R) (decreases n) =
  if n = 1 then vsum_one f
  else (su_vsum_pos (n - 1) f; vsum_succ (n - 1) f)
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let su_sq_nonneg (a:real) : Lemma (a *. a >=. 0.0R) = ()

let sq_sum_nonneg (x:S.seq real) (o:int) (n:nat) : Lemma (sq_sum x o n >=. 0.0R) =
  let f = (fun (k:nat) -> at x (o + k) *. at x (o + k)) in
  FStar.Classical.forall_intro (fun (k:nat) -> su_sq_nonneg (at x (o + k)) <: Lemma (f k >=. 0.0R));
  su_vsum_nonneg n f

let vsum_exp_pos (n:nat) (f:nat -> real)
  : Lemma (requires n >= 1) (ensures vsum n (fun p -> FStar.Math.Exp.exp (f p)) >. 0.0R) =
  let g = (fun (p:nat) -> FStar.Math.Exp.exp (f p)) in
  FStar.Classical.forall_intro (fun (p:nat) -> FStar.Math.Exp.exp_positive (f p) <: Lemma (g p >. 0.0R));
  su_vsum_pos n g
#pop-options

(*** float sums *)

#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
let su_zero_f32 () : Lemma ((zero #f32) %~ 0.0R) = a0 #f32
let su_one_f32 () : Lemma ((one #f32) %~ 1.0R) = a1 #f32
#pop-options

#push-options "--fuel 1 --ifuel 0 --z3rlimit 40"
let rec fsum_approx (n:nat) (f:nat -> f32) (r:nat -> real)
  : Lemma (requires forall (i:nat). i < n ==> f i %~ r i) (ensures fsum n f %~ vsum n r) (decreases n) =
  if n = 0 then (su_vsum_zero r; su_zero_f32 ())
  else begin
    fsum_approx (n - 1) f r;
    vsum_succ (n - 1) r;
    a_add (fsum (n - 1) f) (f (n - 1)) (vsum (n - 1) r) (r (n - 1))
  end
#pop-options

#push-options "--fuel 1 --ifuel 0 --z3rlimit 40"
let rec su_tree_agree (s1 s2:S.seq f32) (k off:nat)
  : Lemma (requires 2 * off <= k /\ 1 <= k /\ k <= S.length s1 /\ k <= S.length s2 /\
                    (forall (i:nat). i < k ==> S.index s1 i == S.index s2 i))
          (ensures fat (tree s1 off) 0 == fat (tree s2 off) 0) (decreases off) =
  if off = 0 then ()
  else begin
    let h1 = halve s1 off in
    let h2 = halve s2 off in
    assert (forall (i:nat). i < k ==> S.index h1 i == S.index h2 i);
    su_tree_agree h1 h2 k (off / 2)
  end
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let tree_prefix (s:S.seq f32) (cnt:nat) (f:nat -> f32)
  : Lemma (requires 1 <= cnt /\ cnt <= S.length s /\ (forall (i:nat). i < cnt ==> S.index s i == f i))
          (ensures fat (tree s (cnt / 2)) 0 == tsum cnt f) =
  su_tree_agree s (S.init cnt (fun (i:nat{i < cnt}) -> f i)) cnt (cnt / 2)
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let su_halve_approx (s:S.seq f32) (off:nat) (r:nat -> real)
  : Lemma (requires 2 * off <= S.length s /\ (forall (i:nat). i < 2 * off ==> S.index s i %~ r i))
          (ensures S.length (halve s off) == S.length s /\
                   (forall (i:nat). i < off ==> S.index (halve s off) i %~ (r i +. r (i + off)))) =
  let aux (i:nat) : Lemma (i < off ==> S.index (halve s off) i %~ (r i +. r (i + off))) =
    if i < off then a_add (S.index s i) (S.index s (i + off)) (r i) (r (i + off))
  in
  FStar.Classical.forall_intro aux
#pop-options

#push-options "--fuel 1 --ifuel 0 --z3rlimit 40"
let rec su_tree_approx (k:nat) (s:S.seq f32) (r:nat -> real)
  : Lemma (requires is_pow2 k /\ k <= S.length s /\ (forall (i:nat). i < k ==> S.index s i %~ r i))
          (ensures fat (tree s (k / 2)) 0 %~ vsum k r) (decreases k) =
  if k = 1 then vsum_one r
  else begin
    let off = k / 2 in
    let r' = (fun (i:nat) -> r i +. r (i + off)) in
    su_halve_approx s off r;
    su_tree_approx off (halve s off) r';
    su_vsum_split off off r (fun (i:nat) -> r (off + i));
    su_vsum_add off r' r (fun (i:nat) -> r (off + i))
  end
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let tsum_approx (cnt:nat) (f:nat -> f32) (r:nat -> real)
  : Lemma (requires is_pow2 cnt /\ (forall (i:nat). i < cnt ==> f i %~ r i)) (ensures tsum cnt f %~ vsum cnt r) =
  su_tree_approx cnt (S.init cnt (fun (i:nat{i < cnt}) -> f i)) r
#pop-options

(*** stage lemmas *)

noextract
let su_sqf_r (rx:S.seq real) (base:int) (k:nat) : real = at rx (base + k) *. at rx (base + k)
noextract
let su_shift (rx:S.seq real) (base:int) (g:nat) (j:nat) : real = su_sqf_r rx base (g * 4 + j)
noextract
let su_grp_r (rx:S.seq real) (base:int) (g:nat) : real = vsum 4 (su_shift rx base g)

#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
let su_sq_sum_def (x:S.seq real) (o:int) (n:nat)
  : Lemma (sq_sum x o n == vsum n (fun (k:nat) -> at x (o + k) *. at x (o + k))) = ()

let su_sq_sum_eq (x:S.seq real) (o:int) (n:nat) : Lemma (sq_sum x o n == vsum n (su_sqf_r x o)) =
  su_sq_sum_def x o n;
  vsum_ext n (fun (k:nat) -> at x (o + k) *. at x (o + k)) (su_sqf_r x o)
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let rec su_grp_sum (rx:S.seq real) (base:int) (cnt:nat)
  : Lemma (ensures vsum (4 * cnt) (su_sqf_r rx base) == vsum cnt (su_grp_r rx base)) (decreases cnt) =
  if cnt = 0 then (su_vsum_zero (su_sqf_r rx base); su_vsum_zero (su_grp_r rx base))
  else begin
    let c = cnt - 1 in
    su_grp_sum rx base c;
    su_vsum_split (4 * c) 4 (su_sqf_r rx base) (su_shift rx base c);
    assert (4 * c + 4 == 4 * cnt);
    vsum_succ c (su_grp_r rx base)
  end
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let su_fb (x:bf16) (a:real) : Lemma (requires x %~ a) (ensures fb x %~ a) = CB.cast_bf16_to_f32_ok x a
let su_tb (x:f32) (a:real) : Lemma (requires x %~ a) (ensures tb x %~ a) = CB.cast_f32_to_bf16_ok x a

let su_sqf_approx (x:bf16) (a:real) : Lemma (requires x %~ a) (ensures sqf x %~ (a *. a)) =
  su_fb x a;
  a_mul (fb x) (fb x) a a

let su_sq4_approx (src:S.seq bf16) (rx:S.seq real) (base:int) (g:nat)
  : Lemma (requires forall (j:nat). j < 4 ==> bat src (base + (g * 4 + j)) %~ at rx (base + (g * 4 + j)))
          (ensures sq4 src base g %~ su_grp_r rx base g) =
  let aux (j:nat) : Lemma (j < 4 ==> sqf (bat src (base + 4 * g + j)) %~ su_shift rx base g j) =
    if j < 4 then su_sqf_approx (bat src (base + (g * 4 + j))) (at rx (base + (g * 4 + j)))
  in
  FStar.Classical.forall_intro aux;
  fsum_approx 4 (fun (j:nat) -> sqf (bat src (base + 4 * g + j))) (su_shift rx base g)

let su_sumsq_approx (src:S.seq bf16) (rx:S.seq real) (base:int) (cnt:nat)
  : Lemma (requires is_pow2 cnt /\ (forall (k:nat). k < 4 * cnt ==> bat src (base + k) %~ at rx (base + k)))
          (ensures tsum cnt (sq4 src base) %~ sq_sum rx base (4 * cnt)) =
  let aux (g:nat) : Lemma (g < cnt ==> sq4 src base g %~ su_grp_r rx base g) =
    if g < cnt then su_sq4_approx src rx base g
  in
  FStar.Classical.forall_intro aux;
  tsum_approx cnt (sq4 src base) (su_grp_r rx base);
  su_grp_sum rx base cnt;
  su_sq_sum_eq rx base (4 * cnt)
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let su_rsqrt_approx (x:f32) (y:real)
  : Lemma (requires x %~ y /\ y >. 0.0R) (ensures FB.rsqrt x %~ rsqrt_t y) =
  Kuiper.Approximates.Base.rsqrt_models_pat f32;
  Kuiper.Approximates.Core.approx_apply_dom #f32 #real #f32 #real #(fun (r:real) -> r >. 0.0R)
    (FB.rsqrt #f32) Kuiper.Float.Realops.r_rsqrt x y

let su_inv_approx (src:S.seq bf16) (rx:S.seq real) (base:int) (cnt:nat) (c:f32) (rc:real)
  : Lemma (requires is_pow2 cnt /\ (forall (k:nat). k < 4 * cnt ==> bat src (base + k) %~ at rx (base + k)) /\
                    c %~ rc /\ rc >. 0.0R)
          (ensures rms_inv src base cnt c %~ rsqrt_t (sq_sum rx base (4 * cnt) *. rc +. eps_f32)) =
  su_sumsq_approx src rx base cnt;
  sq_sum_nonneg rx base (4 * cnt);
  f_eps_approx ();
  let t = tsum cnt (sq4 src base) in
  let rs = sq_sum rx base (4 * cnt) in
  a_mul t c rs rc;
  a_add (mul t c) f_eps (rs *. rc) eps_f32;
  su_rsqrt_approx (add (mul t c) f_eps) (rs *. rc +. eps_f32)

let su_rms_out (src w:S.seq bf16) (rx rw:S.seq real) (base woff:int) (inv:f32) (rinv:real) (i:int)
  : Lemma (requires inv %~ rinv /\ bat src (base + i) %~ at rx (base + i) /\ bat w (woff + i) %~ at rw (woff + i))
          (ensures rms_out src w base woff inv i %~ ((at rx (base + i) *. rinv) *. (1.0R +. at rw (woff + i)))) =
  su_one_f32 ();
  let a = fb (bat src (base + i)) in
  let b = fb (bat w (woff + i)) in
  su_fb (bat src (base + i)) (at rx (base + i));
  su_fb (bat w (woff + i)) (at rw (woff + i));
  a_mul a inv (at rx (base + i)) rinv;
  a_add one b 1.0R (at rw (woff + i));
  a_mul (mul a inv) (add one b) (at rx (base + i) *. rinv) (1.0R +. at rw (woff + i));
  su_tb (mul (mul a inv) (add one b)) ((at rx (base + i) *. rinv) *. (1.0R +. at rw (woff + i)))

let su_bat_approx (s:S.seq bf16) (rs:S.seq real) (k:int)
  : Lemma (requires s %~ rs /\ 0 <= k /\ k < S.length s) (ensures bat s k %~ at rs k) = ()
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let su_src_prefix (src:S.seq bf16) (rx:S.seq real) (base:nat) (n:nat)
  : Lemma (requires src %~ rx /\ base + n <= S.length src)
          (ensures forall (k:nat). k < n ==> bat src (base + k) %~ at rx (base + k)) =
  let aux (k:nat) : Lemma (k < n ==> bat src (base + k) %~ at rx (base + k)) =
    if k < n then su_bat_approx src rx (base + k)
  in
  FStar.Classical.forall_intro aux

let su_rdiv_eq (x:real) (n:real) : Lemma (requires n >. 0.0R) (ensures rdiv x n == x *. (1.0R /. n)) = ()

let rms1024_approx (src w:S.seq bf16) (woff:nat) (rx rw:S.seq real)
  : Lemma (requires S.length src == 1024 /\ src %~ rx /\ w %~ rw /\ woff + 1024 <= S.length w)
          (ensures forall (i:nat). i < 1024 ==> rms1024 src w woff i %~ at (rmsnorm1p eps_f32 rx 1024 rw woff) i) =
  f_inv1024_approx (); f_eps_approx (); pow2_facts ();
  su_src_prefix src rx 0 (4 * 256);
  su_inv_approx src rx 0 256 f_inv1024 (1.0R /. 1024.0R);
  let ssum = sq_sum rx 0 1024 in
  su_rdiv_eq ssum (of_int 1024);
  let inv = rms_inv src 0 256 f_inv1024 in
  let rinv = rsqrt_t (rdiv ssum (of_int 1024) +. eps_f32) in
  assert (inv %~ rinv);
  let aux (i:nat) : Lemma (i < 1024 ==> rms1024 src w woff i %~ at (rmsnorm1p eps_f32 rx 1024 rw woff) i) =
    if i < 1024 then begin
      su_bat_approx src rx (0 + i);
      su_bat_approx w rw (woff + i);
      su_rms_out src w rx rw 0 woff inv rinv i
    end
  in
  FStar.Classical.forall_intro aux
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let su_hnorm_approx (src w:S.seq bf16) (base woff:nat) (rsrc rw:S.seq real)
  : Lemma (requires src %~ rsrc /\ base + 256 <= S.length src /\ w %~ rw /\ woff + 256 <= S.length w)
          (ensures forall (j:nat). j < 256 ==> hnorm src w base woff j %~ at (head_norm eps_f32 rsrc base rw woff) j) =
  f_inv256_approx (); f_eps_approx (); pow2_facts ();
  su_src_prefix src rsrc base (4 * 64);
  su_inv_approx src rsrc base 64 f_inv256 (1.0R /. 256.0R);
  let ssum = sq_sum rsrc base 256 in
  su_rdiv_eq ssum 256.0R;
  let inv = rms_inv src base 64 f_inv256 in
  let rinv = rsqrt_t (rdiv ssum 256.0R +. eps_f32) in
  assert (inv %~ rinv);
  let aux (j:nat) : Lemma (j < 256 ==> hnorm src w base woff j %~ at (head_norm eps_f32 rsrc base rw woff) j) =
    if j < 256 then begin
      su_bat_approx src rsrc (base + j);
      su_bat_approx w rw (woff + j);
      su_rms_out src w rsrc rw base woff inv rinv j
    end
  in
  FStar.Classical.forall_intro aux

let su_ring_minus (a b c d:real) : Lemma (a *. c +. (0.0R -. b) *. d == a *. c -. b *. d) = ()

let su_rot_minus (ha hb c s:bf16) (ra rb rc rs:real)
  : Lemma (requires ha %~ ra /\ hb %~ rb /\ c %~ rc /\ s %~ rs)
          (ensures tb (add (fb (tb (mul (fb ha) (fb c)))) (fb (tb (mul (FB.neg (fb hb)) (fb s))))) %~
                   (ra *. rc -. rb *. rs)) =
  su_fb ha ra; su_fb hb rb; su_fb c rc; su_fb s rs;
  a_mul (fb ha) (fb c) ra rc;
  su_tb (mul (fb ha) (fb c)) (ra *. rc);
  su_fb (tb (mul (fb ha) (fb c))) (ra *. rc);
  Kuiper.Approximates.Base.neg_models_pat f32;
  Kuiper.Approximates.Core.approx_apply (FB.neg #f32) Kuiper.Float.Realops.r_neg (fb hb) rb;
  a_mul (FB.neg (fb hb)) (fb s) (0.0R -. rb) rs;
  su_tb (mul (FB.neg (fb hb)) (fb s)) ((0.0R -. rb) *. rs);
  su_fb (tb (mul (FB.neg (fb hb)) (fb s))) ((0.0R -. rb) *. rs);
  a_add (fb (tb (mul (fb ha) (fb c)))) (fb (tb (mul (FB.neg (fb hb)) (fb s)))) (ra *. rc) ((0.0R -. rb) *. rs);
  su_ring_minus ra rb rc rs;
  su_tb (add (fb (tb (mul (fb ha) (fb c)))) (fb (tb (mul (FB.neg (fb hb)) (fb s))))) (ra *. rc -. rb *. rs)

let su_rot_plus (ha hb c s:bf16) (ra rb rc rs:real)
  : Lemma (requires ha %~ ra /\ hb %~ rb /\ c %~ rc /\ s %~ rs)
          (ensures tb (add (fb (tb (mul (fb hb) (fb c)))) (fb (tb (mul (fb ha) (fb s))))) %~
                   (rb *. rc +. ra *. rs)) =
  su_fb ha ra; su_fb hb rb; su_fb c rc; su_fb s rs;
  a_mul (fb hb) (fb c) rb rc;
  su_tb (mul (fb hb) (fb c)) (rb *. rc);
  su_fb (tb (mul (fb hb) (fb c))) (rb *. rc);
  a_mul (fb ha) (fb s) ra rs;
  su_tb (mul (fb ha) (fb s)) (ra *. rs);
  su_fb (tb (mul (fb ha) (fb s))) (ra *. rs);
  a_add (fb (tb (mul (fb hb) (fb c)))) (fb (tb (mul (fb ha) (fb s)))) (rb *. rc) (ra *. rs);
  su_tb (add (fb (tb (mul (fb hb) (fb c)))) (fb (tb (mul (fb ha) (fb s))))) (rb *. rc +. ra *. rs)

let su_hn_i (src w cs sn:S.seq bf16) (base woff:nat) (u rc rs:S.seq real) (i:nat)
  : Lemma (requires (forall (j:nat). j < 256 ==> hnorm src w base woff j %~ at u j) /\
                    cs %~ rc /\ sn %~ rs /\ S.length cs == 32 /\ S.length sn == 32 /\ i < 256)
          (ensures hn_m src w cs sn base woff i %~ at (rope u rc rs) i) =
  if i < 32 then begin
    su_bat_approx cs rc i; su_bat_approx sn rs i;
    su_rot_minus (hnorm src w base woff i) (hnorm src w base woff (i + 32)) (bat cs i) (bat sn i)
                 (at u i) (at u (i + 32)) (at rc i) (at rs i)
  end
  else if i < 64 then begin
    let r = i - 32 in
    su_bat_approx cs rc r; su_bat_approx sn rs r;
    su_rot_plus (hnorm src w base woff r) (hnorm src w base woff i) (bat cs r) (bat sn r)
                (at u r) (at u i) (at rc r) (at rs r)
  end
  else ()

let hn_approx (src w cs sn:S.seq bf16) (base woff:nat) (rsrc rw rc rs:S.seq real)
  : Lemma (requires src %~ rsrc /\ base + 256 <= S.length src /\ w %~ rw /\ woff + 256 <= S.length w /\
                    cs %~ rc /\ sn %~ rs /\ S.length cs == 32 /\ S.length sn == 32)
          (ensures forall (i:nat). i < 256 ==> hn_m src w cs sn base woff i %~ at (rope (head_norm eps_f32 rsrc base rw woff) rc rs) i) =
  su_hnorm_approx src w base woff rsrc rw;
  let u = head_norm eps_f32 rsrc base rw woff in
  let aux (i:nat) : Lemma (i < 256 ==> hn_m src w cs sn base woff i %~ at (rope u rc rs) i) =
    if i < 256 then su_hn_i src w cs sn base woff u rc rs i
  in
  FStar.Classical.forall_intro aux
#pop-options

(*** Attention lemmas (lem_attn) *)

#push-options "--z3rlimit 10"

let resid_approx (a b:S.seq bf16) (i:nat) (ra rb:real) : Lemma
  (requires bat a i %~ ra /\ bat b i %~ rb) (ensures resid_m a b i %~ (ra +. rb))
  = CB.cast_bf16_to_f32_ok (bat a i) ra; CB.cast_bf16_to_f32_ok (bat b i) rb;
    a_add (fb (bat a i)) (fb (bat b i)) ra rb;
    CB.cast_f32_to_bf16_ok (add (fb (bat a i)) (fb (bat b i))) (ra +. rb)

let at_h_den (x:f32) (r:real) : Lemma (requires x %~ r)
  (ensures add one (FB.fexp (FB.neg x)) %~ (1.0R +. FStar.Math.Exp.exp (0.0R -. r)) /\
           1.0R +. FStar.Math.Exp.exp (0.0R -. r) =!= 0.0R)
  = assert (has_type Kuiper.Float32.is_floating_real_like (Kuiper.Approximates.Base.floating_real_like f32));
    let nx = FB.neg #f32 x in
    assert (nx %~ (0.0R -. r));
    let e = FB.fexp #f32 nx in
    assert (e %~ FStar.Math.Exp.exp (0.0R -. r));
    a_add (one #f32) e 1.0R (FStar.Math.Exp.exp (0.0R -. r));
    FStar.Math.Exp.exp_positive (0.0R -. r)

let at_h_sigm (x:f32) (r:real) : Lemma (requires x %~ r) (ensures sigm x %~ sigmoid r)
  = assert (has_type Kuiper.Float32.is_floating_real_like (Kuiper.Approximates.Base.floating_real_like f32));
    at_h_den x r;
    let d = add one (FB.fexp (FB.neg x)) in
    assert (FB.div #f32 one d %~ (1.0R /. (1.0R +. FStar.Math.Exp.exp (0.0R -. r))))

let gate_approx (vals:S.seq f32) (pq:S.seq bf16) (i:nat) (rv rg:real) : Lemma
  (requires fat vals i %~ rv /\ bat pq ((i / 256) * 512 + 256 + i % 256) %~ rg)
  (ensures gate_m vals pq i %~ (rv *. sigmoid rg))
  = let x = fb (bat pq ((i / 256) * 512 + 256 + i % 256)) in
    CB.cast_bf16_to_f32_ok (bat pq ((i / 256) * 512 + 256 + i % 256)) rg;
    at_h_sigm x rg;
    CB.cast_f32_to_bf16_ok (sigm x) (sigmoid rg);
    CB.cast_bf16_to_f32_ok (tb (sigm x)) (sigmoid rg);
    CB.cast_f32_to_bf16_ok (fat vals i) rv;
    CB.cast_bf16_to_f32_ok (tb (fat vals i)) rv;
    a_mul (fb (tb (fat vals i))) (fb (tb (sigm x))) rv (sigmoid rg);
    CB.cast_f32_to_bf16_ok (mul (fb (tb (fat vals i))) (fb (tb (sigm x)))) (rv *. sigmoid rg)

let at_h_silu (x:f32) (r:real) : Lemma (requires x %~ r) (ensures silu_f x %~ silu r)
  = assert (has_type Kuiper.Float32.is_floating_real_like (Kuiper.Approximates.Base.floating_real_like f32));
    at_h_den x r;
    let d = add one (FB.fexp (FB.neg x)) in
    assert (FB.div #f32 x d %~ (r /. (1.0R +. FStar.Math.Exp.exp (0.0R -. r))));
    assert (r /. (1.0R +. FStar.Math.Exp.exp (0.0R -. r)) == silu r)

let act_approx (g u:S.seq bf16) (i:nat) (rg ru:real) : Lemma
  (requires bat g i %~ rg /\ bat u i %~ ru) (ensures act_m g u i %~ (silu rg *. ru))
  = let x = fb (bat g i) in
    CB.cast_bf16_to_f32_ok (bat g i) rg;
    at_h_silu x rg;
    CB.cast_f32_to_bf16_ok (silu_f x) (silu rg);
    CB.cast_bf16_to_f32_ok (tb (silu_f x)) (silu rg);
    CB.cast_bf16_to_f32_ok (bat u i) ru;
    a_mul (fb (tb (silu_f x))) (fb (bat u i)) (silu rg) ru;
    CB.cast_f32_to_bf16_ok (mul (fb (tb (silu_f x))) (fb (bat u i))) (silu rg *. ru)

(* generic real sums; every lambda over a sequence index has the shape of the one in vsum *)

#push-options "--fuel 2 --ifuel 1"
let at_h_rsum1 (x:real) : Lemma (rsum (S.create 1 x) == 0.0R +. x)
  = assert (S.equal (S.create 1 x) (S.cons x S.empty))
#pop-options

let at_h_vsum_split (nm n m:nat) (f g:nat -> real) : Lemma
  (requires nm == n + m /\ (forall (i:nat). g i == f (n + i)))
  (ensures vsum nm f == vsum n f +. vsum m g)
  = assert (S.equal (S.init nm (fun (k:nat{k < nm}) -> f k))
                    (S.append (S.init n (fun (k:nat{k < n}) -> f k)) (S.init m (fun (k:nat{k < m}) -> g k))))

let at_h_vsum_one (o:nat) (f:nat -> real) : Lemma (requires o == 1) (ensures vsum o f == f 0)
  = assert (S.equal (S.init o (fun (k:nat{k < o}) -> f k)) (S.create 1 (f 0)));
    at_h_rsum1 (f 0)

let at_h_vsum_zero (o:nat) (f:nat -> real) : Lemma (requires o == 0) (ensures vsum o f == 0.0R)
  = assert (S.equal (S.init o (fun (k:nat{k < o}) -> f k)) S.empty)

let at_h_vsum_ext (n:nat) (f g:nat -> real) : Lemma
  (requires forall (i:nat). i < n ==> f i == g i) (ensures vsum n f == vsum n g)
  = assert (S.equal (S.init n (fun (k:nat{k < n}) -> f k)) (S.init n (fun (k:nat{k < n}) -> g k)))

let at_h_vsum_succ (n:nat) (f:nat -> real) : Lemma (vsum (n + 1) f == vsum n f +. f n)
  = at_h_vsum_split (n + 1) n 1 f (fun i -> f (n + i));
    at_h_vsum_one 1 (fun i -> f (n + i))

#push-options "--fuel 0 --ifuel 0"
let rec at_h_vsum_add (n:nat) (f g fg:nat -> real) : Lemma
  (requires forall (i:nat). fg i == f i +. g i)
  (ensures vsum n fg == vsum n f +. vsum n g) (decreases n)
  = if n = 0 then (at_h_vsum_zero n f; at_h_vsum_zero n g; at_h_vsum_zero n fg)
    else (at_h_vsum_add (n - 1) f g fg;
          at_h_vsum_succ (n - 1) f; at_h_vsum_succ (n - 1) g; at_h_vsum_succ (n - 1) fg)

let rec at_h_vsum_zeros (n:nat) (f:nat -> real) : Lemma
  (requires forall (i:nat). i < n ==> f i == 0.0R) (ensures vsum n f == 0.0R) (decreases n)
  = if n = 0 then at_h_vsum_zero n f
    else (at_h_vsum_zeros (n - 1) f; at_h_vsum_succ (n - 1) f)

let at_h_vsum_prefix (n w:nat) (f:nat -> real) : Lemma
  (requires n <= w /\ (forall (i:nat). n <= i /\ i < w ==> f i == 0.0R))
  (ensures vsum w f == vsum n f)
  = at_h_vsum_split w n (w - n) f (fun i -> f (n + i));
    at_h_vsum_zeros (w - n) (fun i -> f (n + i))

let rec at_h_vsum_block (k m:nat) (f:nat -> real) (fg:nat -> nat -> real) (gs:nat -> real) : Lemma
  (requires (forall (g j:nat). fg g j == f (m * g + j)) /\ (forall (g:nat). gs g == vsum m (fg g)))
  (ensures vsum (k * m) f == vsum k gs) (decreases k)
  = if k = 0 then (at_h_vsum_zero (k * m) f; at_h_vsum_zero k gs)
    else (let k1 = k - 1 in
          let b = k1 * m in
          at_h_vsum_block k1 m f fg gs;
          FStar.Math.Lemmas.distributivity_add_left k1 1 m;
          FStar.Math.Lemmas.swap_mul m k1;
          assert (k * m == b + m);
          assert (m * k1 == b);
          at_h_vsum_split (k * m) b m f (fun i -> f (b + i));
          at_h_vsum_ext m (fun i -> f (b + i)) (fg k1);
          assert (gs k1 == vsum m (fg k1));
          at_h_vsum_succ k1 gs)
#pop-options

let rvals_one (rp rv:nat -> real) : Lemma (vsum 1 (fun p -> rp p *. rv p) == rp 0 *. rv 0)
  = at_h_vsum_one 1 (fun p -> rp p *. rv p)

#push-options "--fuel 1 --ifuel 0"
let at_h_rgrp_step (rp rv:nat -> real) (g:int) (n:pos) : Lemma
  (rgrp rp rv g n == (if (n - 1) / 4 = g then rp (n - 1) *. rv (n - 1) +. rgrp rp rv g (n - 1) else rgrp rp rv g (n - 1)))
  = ()
let at_h_rgrp0 (rp rv:nat -> real) (g:int) : Lemma (rgrp rp rv g 0 == 0.0R) = ()
#pop-options

#push-options "--fuel 0 --ifuel 0"
let rec at_h_rgrp (rp rv:nat -> real) (n:nat) : Lemma (requires n <= 16)
  (ensures (rgrp rp rv 0 n +. rgrp rp rv 2 n) +. (rgrp rp rv 1 n +. rgrp rp rv 3 n) == vsum n (fun p -> rp p *. rv p))
  (decreases n)
  = if n = 0 then (at_h_vsum_zero n (fun p -> rp p *. rv p);
                   at_h_rgrp0 rp rv 0; at_h_rgrp0 rp rv 1; at_h_rgrp0 rp rv 2; at_h_rgrp0 rp rv 3)
    else (let n1 = n - 1 in
          at_h_rgrp rp rv n1; at_h_vsum_succ n1 (fun p -> rp p *. rv p);
          let x = rp n1 *. rv n1 in
          at_h_rgrp_step rp rv 0 n; at_h_rgrp_step rp rv 1 n; at_h_rgrp_step rp rv 2 n; at_h_rgrp_step rp rv 3 n;
          assert (n1 / 4 == 0 \/ n1 / 4 == 1 \/ n1 / 4 == 2 \/ n1 / 4 == 3))

#pop-options

let rgrp_sum (rp rv:nat -> real) (t:nat) : Lemma (requires t < 16)
  (ensures (rgrp rp rv 0 (t + 1) +. rgrp rp rv 2 (t + 1)) +. (rgrp rp rv 1 (t + 1) +. rgrp rp rv 3 (t + 1)) == vsum (t + 1) (fun p -> rp p *. rv p))
  = at_h_rgrp rp rv (t + 1)

(* float sums *)

let at_h_zero () : Lemma (zero #f32 %~ 0.0R)
  = let pf : squash (v_approximates (zero #f32) Kuiper.Float.Realops.r_zero) = Kuiper.Approximates.Scalar.a0 #f32 in ()

#push-options "--fuel 1 --ifuel 0"
let rec at_h_fsum (n:nat) (f:nat -> f32) (r:nat -> real) : Lemma
  (requires forall (i:nat). i < n ==> f i %~ r i)
  (ensures fsum n f %~ vsum n r) (decreases n)
  = if n = 0 then (at_h_zero (); at_h_vsum_zero n r)
    else (at_h_fsum (n - 1) f r; at_h_vsum_succ (n - 1) r;
          a_add (fsum (n - 1) f) (f (n - 1)) (vsum (n - 1) r) (r (n - 1)))
#pop-options

noextract
let at_h_shift (r:nat -> real) (h:nat) (i:nat) : real = r (h + i)
noextract
let at_h_pairf (r:nat -> real) (h:nat) (i:nat) : real = r i +. r (h + i)

let at_h_vsum_pair (h:nat) (r:nat -> real) : Lemma (vsum h (at_h_pairf r h) == vsum (h + h) r)
  = at_h_vsum_add h r (at_h_shift r h) (at_h_pairf r h);
    at_h_vsum_split (h + h) h h r (at_h_shift r h)

let at_h_halve (s:S.seq f32) (h:pos) (r:nat -> real) : Lemma
  (requires h + h <= S.length s /\ (forall (i:nat). i < h + h ==> S.index s i %~ r i))
  (ensures S.length (halve s h) == S.length s /\
           (forall (i:nat). i < h ==> S.index (halve s h) i %~ at_h_pairf r h i))
  = introduce forall (i:nat). i < h ==> S.index (halve s h) i %~ at_h_pairf r h i
    with introduce _ ==> _
    with a_add (S.index s i) (S.index s (i + h)) (r i) (r (h + i))

#push-options "--fuel 1 --ifuel 0"
let at_h_tree_unfold (s:S.seq f32) (off:pos) : Lemma (tree s off == tree (halve s off) (off / 2)) = ()
let at_h_tree0 (s:S.seq f32) : Lemma (tree s 0 == s) = ()
let at_h_pow2_0 (_:unit) : Lemma (pow2 0 == 1) = ()
let at_h_pow2_succ (k:pos) : Lemma (pow2 k == pow2 (k - 1) + pow2 (k - 1)) = ()
#pop-options

#push-options "--fuel 0 --ifuel 0"
let rec at_h_tree (k:nat) (s:S.seq f32) (r:nat -> real) : Lemma
  (requires pow2 k <= S.length s /\ (forall (i:nat). i < pow2 k ==> S.index s i %~ r i))
  (ensures fat (tree s (pow2 k / 2)) 0 %~ vsum (pow2 k) r) (decreases k)
  = if k = 0 then (at_h_pow2_0 (); at_h_tree0 s; at_h_vsum_one (pow2 k) r)
    else (let h = pow2 (k - 1) in
          at_h_pow2_succ k;
          at_h_tree_unfold s h;
          at_h_halve s h r;
          at_h_tree (k - 1) (halve s h) (at_h_pairf r h);
          at_h_vsum_pair h r)
#pop-options

let at_h_tsum (k:nat) (cnt:nat) (f:nat -> f32) (r:nat -> real) : Lemma
  (requires cnt == pow2 k /\ (forall (i:nat). i < cnt ==> f i %~ r i))
  (ensures tsum cnt f %~ vsum cnt r)
  = at_h_tree k (S.init cnt (fun (i:nat{i < cnt}) -> f i)) r

(* attention scores *)

noextract
let at_h_sc (rq:S.seq real) (h:nat) (rk:nat -> real) (i:nat) : real = at rq (h * 256 + i) *. rk i
noextract
let at_h_scg (rq:S.seq real) (h:nat) (rk:nat -> real) (g j:nat) : real = at_h_sc rq h rk (16 * g + j)
noextract
let at_h_scb (rq:S.seq real) (h:nat) (rk:nat -> real) (g:nat) : real = vsum 16 (at_h_scg rq h rk g)

let at_h_dot_term (qn kvo:S.seq bf16) (c h p:nat) (rq:S.seq real) (rk:nat -> real) (g j:nat) : Lemma
  (requires qn %~ rq /\ S.length qn == 2048 /\ h < 8 /\ g < 16 /\ j < 16 /\
            (forall (j:nat). j < 256 ==> bat kvo (((h / 4) * c + p) * 256 + j) %~ rk j))
  (ensures mul (fb (bat qn (h * 256 + 16 * g + j))) (fb (bat kvo (((h / 4) * c + p) * 256 + 16 * g + j)))
           %~ at_h_scg rq h rk g j)
  = let iq = h * 256 + 16 * g + j in
    let jj = 16 * g + j in
    assert (iq < 2048);
    assert (bat qn iq %~ at rq iq);
    assert (bat kvo (((h / 4) * c + p) * 256 + jj) %~ rk jj);
    assert (((h / 4) * c + p) * 256 + 16 * g + j == ((h / 4) * c + p) * 256 + jj);
    CB.cast_bf16_to_f32_ok (bat qn iq) (at rq iq);
    CB.cast_bf16_to_f32_ok (bat kvo (((h / 4) * c + p) * 256 + jj)) (rk jj);
    a_mul (fb (bat qn iq)) (fb (bat kvo (((h / 4) * c + p) * 256 + jj))) (at rq iq) (rk jj)

let at_h_dot16 (qn kvo:S.seq bf16) (c h p:nat) (rq:S.seq real) (rk:nat -> real) (g:nat) : Lemma
  (requires qn %~ rq /\ S.length qn == 2048 /\ h < 8 /\ g < 16 /\
            (forall (j:nat). j < 256 ==> bat kvo (((h / 4) * c + p) * 256 + j) %~ rk j))
  (ensures dot16 qn kvo c h p g %~ at_h_scb rq h rk g)
  = introduce forall (j:nat). j < 16 ==>
      mul (fb (bat qn (h * 256 + 16 * g + j))) (fb (bat kvo (((h / 4) * c + p) * 256 + 16 * g + j)))
        %~ at_h_scg rq h rk g j
    with introduce _ ==> _
    with at_h_dot_term qn kvo c h p rq rk g j;
    at_h_fsum 16 (fun j -> mul (fb (bat qn (h * 256 + 16 * g + j))) (fb (bat kvo (((h / 4) * c + p) * 256 + 16 * g + j))))
      (at_h_scg rq h rk g)

let at_h_pow2_16 () : Lemma (pow2 4 == 16)
  = at_h_pow2_0 (); at_h_pow2_succ 1; at_h_pow2_succ 2; at_h_pow2_succ 3; at_h_pow2_succ 4

let score_approx (qn kvo:S.seq bf16) (c h p:nat) (rq:S.seq real) (rk:nat -> real) : Lemma
  (requires qn %~ rq /\ S.length qn == 2048 /\ h < 8 /\
            (forall (j:nat). j < 256 ==> bat kvo (((h / 4) * c + p) * 256 + j) %~ rk j))
  (ensures score_m qn kvo c h p %~ vsum 256 (fun i -> at rq (h * 256 + i) *. rk i))
  = introduce forall (g:nat). g < 16 ==> dot16 qn kvo c h p g %~ at_h_scb rq h rk g
    with introduce _ ==> _
    with at_h_dot16 qn kvo c h p rq rk g;
    at_h_pow2_16 ();
    at_h_tsum 4 16 (dot16 qn kvo c h p) (at_h_scb rq h rk);
    at_h_vsum_block 16 16 (at_h_sc rq h rk) (at_h_scg rq h rk) (at_h_scb rq h rk);
    at_h_vsum_ext 256 (fun i -> at rq (h * 256 + i) *. rk i) (at_h_sc rq h rk);
    CB.cast_f32_to_bf16_ok (tsum 16 (dot16 qn kvo c h p)) (vsum 256 (at_h_sc rq h rk))

(* softmax *)

let at_h_distr (a b c:real) : Lemma ((a +. b) *. c == a *. c +. b *. c) = ()

#push-options "--fuel 0 --ifuel 0"
let rec at_h_vsum_scale (n:nat) (f g:nat -> real) (c:real) : Lemma
  (requires forall (i:nat). g i == f i *. c)
  (ensures vsum n g == vsum n f *. c) (decreases n)
  = if n = 0 then (at_h_vsum_zero n f; at_h_vsum_zero n g)
    else (at_h_vsum_scale (n - 1) f g c; at_h_vsum_succ (n - 1) f; at_h_vsum_succ (n - 1) g;
          at_h_distr (vsum (n - 1) f) (f (n - 1)) c)

#pop-options

let rec at_h_vsum_pos (n:pos) (f:nat -> real) : Lemma
  (requires forall (i:nat). i < n ==> f i >. 0.0R)
  (ensures vsum n f >. 0.0R) (decreases n)
  = if n = 1 then at_h_vsum_one n f
    else (at_h_vsum_pos (n - 1) f; at_h_vsum_succ (n - 1) f)

noextract
let at_h_X (rd:nat -> real) (p:nat) : real = FStar.Math.Exp.exp (rd p *. 0.0625R)
noextract
let at_h_E (rd:nat -> real) (mm:real) (p:nat) : real = FStar.Math.Exp.exp (rd p *. 0.0625R -. mm)
noextract
let at_h_EP (rd:nat -> real) (mm:real) (t:nat) (p:nat) : real = if p <= t then at_h_E rd mm p else 0.0R

let at_h_div_mul (a e:real) : Lemma (requires e >. 0.0R) (ensures a /. e == a *. (1.0R /. e)) = ()

let at_h_E_X (rd:nat -> real) (mm:real) (i:nat) : Lemma
  (at_h_E rd mm i == at_h_X rd i *. (1.0R /. FStar.Math.Exp.exp mm))
  = FStar.Math.Exp.exp_sub (rd i *. 0.0625R) mm;
    FStar.Math.Exp.exp_positive mm;
    at_h_div_mul (FStar.Math.Exp.exp (rd i *. 0.0625R)) (FStar.Math.Exp.exp mm)

let at_h_sms (dots:S.seq bf16) (h p:nat) (r:real) : Lemma
  (requires bat dots (h * 16 + p) %~ r) (ensures sm_s dots h p %~ (r *. 0.0625R))
  = f_sixteenth_approx ();
    CB.cast_bf16_to_f32_ok (bat dots (h * 16 + p)) r;
    a_mul (fb (bat dots (h * 16 + p))) f_sixteenth r 0.0625R;
    CB.cast_f32_to_bf16_ok (mul (fb (bat dots (h * 16 + p))) f_sixteenth) (r *. 0.0625R);
    CB.cast_bf16_to_f32_ok (tb (mul (fb (bat dots (h * 16 + p))) f_sixteenth)) (r *. 0.0625R)

let at_h_sme (dots:S.seq bf16) (t h p:nat) (rd:nat -> real) (mm:real) : Lemma
  (requires sm_max dots h t %~ mm /\ bat dots (h * 16 + p) %~ rd p)
  (ensures sm_e dots t h p %~ at_h_E rd mm p)
  = assert (has_type Kuiper.Float32.is_floating_real_like (Kuiper.Approximates.Base.floating_real_like f32));
    at_h_sms dots h p (rd p);
    let d = FB.sub #f32 (sm_s dots h p) (sm_max dots h t) in
    assert (d %~ (rd p *. 0.0625R -. mm));
    assert (FB.fexp #f32 d %~ FStar.Math.Exp.exp (rd p *. 0.0625R -. mm))

noextract
let at_h_wk (t:nat) : nat = if t < 1 then 0 else if t < 2 then 1 else if t < 4 then 2 else if t < 8 then 3 else 4

let at_h_width (t:nat) : Lemma (requires t < 16) (ensures sm_width t == pow2 (at_h_wk t) /\ t + 1 <= sm_width t)
  = at_h_pow2_0 (); at_h_pow2_succ 1; at_h_pow2_succ 2; at_h_pow2_succ 3; at_h_pow2_succ 4

let at_h_smsum (dots:S.seq bf16) (t h:nat) (rd:nat -> real) (mm:real) : Lemma
  (requires t < 16 /\ sm_max dots h t %~ mm /\ (forall (p:nat). p <= t ==> bat dots (h * 16 + p) %~ rd p))
  (ensures sm_sum dots t h %~ vsum (t + 1) (at_h_E rd mm))
  = introduce forall (p:nat). p < sm_width t ==>
      (if p <= t then sm_e dots t h p else zero) %~ at_h_EP rd mm t p
    with introduce _ ==> _
    with (if p <= t then at_h_sme dots t h p rd mm else at_h_zero ());
    at_h_width t;
    at_h_tsum (at_h_wk t) (sm_width t) (fun p -> if p <= t then sm_e dots t h p else zero) (at_h_EP rd mm t);
    at_h_vsum_prefix (t + 1) (sm_width t) (at_h_EP rd mm t);
    at_h_vsum_ext (t + 1) (at_h_EP rd mm t) (at_h_E rd mm)

let at_h_ratio (a b c:real) : Lemma (requires b >. 0.0R /\ c >. 0.0R)
  (ensures (a *. c) /. (b *. c) == a /. b)
  = ()

let at_h_prob1 (dots:S.seq bf16) (t h:nat) (rd:nat -> real) (p:nat) : Lemma
  (requires t < 16 /\ p <= t /\ (forall (p:nat). p <= t ==> bat dots (h * 16 + p) %~ rd p))
  (ensures prob_m dots t h p %~ rdiv (FStar.Math.Exp.exp (rd p *. 0.0625R)) (vsum (t + 1) (fun p' -> FStar.Math.Exp.exp (rd p' *. 0.0625R))))
  = assert (has_type Kuiper.Float32.is_floating_real_like (Kuiper.Approximates.Base.floating_real_like f32));
    let m = sm_max dots h t in
    let mm = Kuiper.Approximates.Scalar.to_real m in
    Kuiper.Approximates.Scalar.to_real_ok m;
    at_h_sme dots t h p rd mm;
    at_h_smsum dots t h rd mm;
    let em = FStar.Math.Exp.exp mm in
    FStar.Math.Exp.exp_positive mm;
    let c = 1.0R /. em in
    introduce forall (i:nat). at_h_E rd mm i == at_h_X rd i *. c
    with at_h_E_X rd mm i;
    at_h_vsum_scale (t + 1) (at_h_X rd) (at_h_E rd mm) c;
    at_h_vsum_pos (t + 1) (at_h_X rd);
    let sx = vsum (t + 1) (at_h_X rd) in
    let se = vsum (t + 1) (at_h_E rd mm) in
    assert (se == sx *. c);
    assert (se >. 0.0R);
    let q = FB.div #f32 (sm_e dots t h p) (sm_sum dots t h) in
    assert (q %~ (at_h_E rd mm p /. se));
    at_h_ratio (at_h_X rd p) sx c;
    at_h_vsum_ext (t + 1) (fun p' -> FStar.Math.Exp.exp (rd p' *. 0.0625R)) (at_h_X rd);
    CB.cast_f32_to_bf16_ok q (at_h_E rd mm p /. se)

let prob_approx (dots:S.seq bf16) (t h:nat) (rd:nat -> real) : Lemma
  (requires t < 16 /\ (forall (p:nat). p <= t ==> bat dots (h * 16 + p) %~ rd p))
  (ensures forall (p:nat). p <= t ==>
     prob_m dots t h p %~ rdiv (FStar.Math.Exp.exp (rd p *. 0.0625R)) (vsum (t + 1) (fun p' -> FStar.Math.Exp.exp (rd p' *. 0.0625R))))
  = introduce forall (p:nat). p <= t ==>
      prob_m dots t h p %~ rdiv (FStar.Math.Exp.exp (rd p *. 0.0625R)) (vsum (t + 1) (fun p' -> FStar.Math.Exp.exp (rd p' *. 0.0625R)))
    with introduce _ ==> _
    with at_h_prob1 dots t h rd p

#pop-options

module GM = Kuiper.Spec.GEMM
module ET = Kuiper.EMatrix.Tiling

(*** MMA projection: column 0 of mma_model approximates a real dot product *)

#push-options "--z3rlimit 20"

(* generic vsum facts *)
noextract
let mm_seq (n:nat) (f:nat -> real) : S.seq real = S.init n (fun (k:nat{k < n}) -> f k)

let mm_seq_index (n:nat) (f:nat -> real) (i:nat{i < n}) : Lemma (S.index (mm_seq n f) i == f i) = ()

let mm_vsum_def (n:nat) (f:nat -> real) : Lemma (vsum n f == rsum (mm_seq n f)) = ()

let mm_vsum_ext (n:nat) (f g:nat -> real) : Lemma
  (requires forall (i:nat). i < n ==> f i == g i)
  (ensures vsum n f == vsum n g)
  = mm_vsum_def n f; mm_vsum_def n g;
    assert (S.equal (mm_seq n f) (mm_seq n g))

let mm_rsum_one (x:real) : Lemma (rsum (S.create 1 x) == x)
  = assert (S.equal (S.tail (S.create 1 x)) S.empty)

let mm_vsum_step (n:nat) (f:nat -> real) : Lemma (vsum (n + 1) f == vsum n f +. f n)
  = mm_vsum_def (n + 1) f; mm_vsum_def n f;
    let s2 = S.create 1 (f n) in
    let s = S.append (mm_seq n f) s2 in
    introduce forall (i:nat{i < n + 1}). S.index (mm_seq (n + 1) f) i == S.index s i
    with (mm_seq_index (n + 1) f i; if i < n then mm_seq_index n f i else ());
    assert (S.equal (mm_seq (n + 1) f) s);
    rsum_append (mm_seq n f) s2;
    mm_rsum_one (f n)

let mm_vsum_zero (f:nat -> real) : Lemma (vsum 0 f == 0.0R)
  = mm_vsum_def 0 f;
    assert (S.equal (mm_seq 0 f) S.empty)

let rec mm_vsum_split (m p:nat) (f:nat -> real) : Lemma
  (ensures vsum (m + p) f == vsum m f +. vsum p (fun b -> f (m + b)))
  (decreases p)
  = if p = 0 then mm_vsum_zero (fun b -> f (m + b))
    else begin
      mm_vsum_split m (p - 1) f;
      mm_vsum_step (m + p - 1) f;
      mm_vsum_step (p - 1) (fun b -> f (m + b))
    end

(* layouts *)
#push-options "--fuel 2 --ifuel 2 --z3rlimit 80"
let mm_rm_imap (m n:nat) (i:natlt m) (j:natlt n) : Lemma ((row_major m n).imap.f (i, (j, ())) == i * n + j) = ()
let mm_cm_imap (m n:nat) (i:natlt m) (j:natlt n) : Lemma ((col_major m n).imap.f (i, (j, ())) == j * m + i) = ()
let mm_rm_full (m n:nat) : Lemma (is_full (row_major m n)) = ()
let mm_cm_full (m n:nat) : Lemma (is_full (col_major m n)) = ()
#pop-options

let mm_bound (x y n k:nat) : Lemma (requires x < n /\ y < k) (ensures x * k + y < n * k)
  = FStar.Math.Lemmas.lemma_mult_le_right k (x + 1) n

let mm_from_rm (#et:Type) (n k:nat) (s:S.seq et{S.length s == n * k}) (i:natlt n) (j:natlt k)
  : Lemma (acc2 (from_seq (row_major n k) s) i j == S.index s (i * k + j))
  = mm_rm_imap n k i j

let mm_from_cm (#et:Type) (k m:nat) (s:S.seq et{S.length s == k * m}) (i:natlt k) (j:natlt m)
  : Lemma (acc2 (from_seq (col_major k m) s) i j == S.index s (j * k + i))
  = mm_cm_imap k m i j

let mm_to_rm (#et:Type) (n m:nat) (c:chest2 et n m) (i:natlt n) (j:natlt m)
  : Lemma (i * m + j < n * m /\ S.index (to_seq (row_major n m) c) (i * m + j) == acc2 c i j)
  = mm_bound i j n m;
    let l = row_major n m in
    mm_rm_imap n m i j;
    Kuiper.Injection.lem_pat l.imap (i, (j, ()))

(* real tiles *)
noextract
let mm_ra_tile (ra:nat -> real) (k r j:nat) : chest2 real 16 16 =
  mk2 (fun (a:natlt 16) (b:natlt 16) -> ra ((16 * r + a) * k + 16 * j + b))

noextract
let mm_rb_tile (rb:nat -> real) (j:nat) : chest2 real 16 8 =
  mk2 (fun (b:natlt 16) (c:natlt 8) -> rb (16 * j + b))

noextract
let rec mm_rtile (ra rb:nat -> real) (k r j0 j1:nat) : Tot (chest2 real 16 8) (decreases j1) =
  if j1 <= j0 then const (16 @| 8 @| INil) 0.0R
  else GM.matplus (mm_rtile ra rb k r j0 (j1 - 1)) (GM.matmul (mm_ra_tile ra k r (j1 - 1)) (mm_rb_tile rb (j1 - 1)))

let mm_tileA_approx (n k:pos) (sa:S.seq bf16) (ra:nat -> real) (r:nat) (j:nat) : Lemma
  (requires 16 /? n /\ 16 /? k /\ r < n / 16 /\ j < k / 16 /\ S.length sa == n * k /\
            (forall (idx:nat). idx < n * k ==> S.index sa idx %~ ra idx))
  (ensures ematrix_subtile (from_seq (row_major n k) sa) 16 16 r j %~ mm_ra_tile ra k r j)
  = let eA = from_seq (row_major n k) sa in
    introduce forall (a:natlt 16) (b:natlt 16).
      acc2 (ematrix_subtile eA 16 16 r j) a b %~ acc2 (mm_ra_tile ra k r j) a b
    with begin
      subtile_acc2 eA 16 16 r j a b;
      mm_from_rm n k sa (r * 16 + a) (j * 16 + b);
      mm_bound (r * 16 + a) (j * 16 + b) n k
    end

let mm_tileB_approx (k:pos) (sb:S.seq bf16) (rb:nat -> real) (j:nat) : Lemma
  (requires 16 /? k /\ j < k / 16 /\ S.length sb == k * 8 /\
            (forall (kk c:nat). kk < k /\ c < 8 ==> S.index sb (c * k + kk) %~ rb kk))
  (ensures ematrix_subtile (from_seq (col_major k 8) sb) 16 8 j 0 %~ mm_rb_tile rb j)
  = let eB = from_seq (col_major k 8) sb in
    introduce forall (b:natlt 16) (c:natlt 8).
      acc2 (ematrix_subtile eB 16 8 j 0) b c %~ acc2 (mm_rb_tile rb j) b c
    with begin
      subtile_acc2 eB 16 8 j 0 b c;
      mm_from_cm k 8 sb (j * 16 + b) c
    end

let mm_zero_tile_approx () : Lemma
  (const (16 @| 8 @| INil) (zero #f32) %~ const (16 @| 8 @| INil) 0.0R)
  = let z : squash (v_approximates (zero #f32) 0.0R) = Kuiper.Approximates.Scalar.a0 #f32 in
    introduce forall (a:natlt 16) (b:natlt 8).
      acc2 (const (16 @| 8 @| INil) (zero #f32)) a b %~ acc2 (const (16 @| 8 @| INil) 0.0R) a b
    with ()

let rec mm_tile_approx (n k:pos) (sa sb:S.seq bf16) (ra rb:nat -> real) (r:nat) (j0 j1:nat) : Lemma
  (requires 16 /? n /\ 16 /? k /\ r < n / 16 /\ j1 <= k / 16 /\ S.length sa == n * k /\ S.length sb == k * 8 /\
            (forall (idx:nat). idx < n * k ==> S.index sa idx %~ ra idx) /\
            (forall (kk c:nat). kk < k /\ c < 8 ==> S.index sb (c * k + kk) %~ rb kk))
  (ensures mma_tile (from_seq (row_major n k) sa) (from_seq (col_major k 8) sb) r j0 j1 %~ mm_rtile ra rb k r j0 j1)
  (decreases j1)
  = if j1 <= j0 then mm_zero_tile_approx ()
    else begin
      let eA = from_seq (row_major n k) sa in
      let eB = from_seq (col_major k 8) sb in
      mm_tile_approx n k sa sb ra rb r j0 (j1 - 1);
      mm_tileA_approx n k sa ra r (j1 - 1);
      mm_tileB_approx k sb rb (j1 - 1);
      MM.emma_approx_lemma (mma_tile eA eB r j0 (j1 - 1)) (ematrix_subtile eA 16 16 r (j1 - 1)) (ematrix_subtile eB 16 8 (j1 - 1) 0)
        (mm_rtile ra rb k r j0 (j1 - 1)) (mm_ra_tile ra k r (j1 - 1)) (mm_rb_tile rb (j1 - 1))
    end

(* real arithmetic of the tiles *)
let mm_vsum_split' (m p:nat) (f g:nat -> real) : Lemma
  (requires forall (b:nat). b < p ==> g b == f (m + b))
  (ensures vsum (m + p) f == vsum m f +. vsum p g)
  = mm_vsum_split m p f;
    mm_vsum_ext p (fun b -> f (m + b)) g

let rec mm_matmul_col (#rows #shared #cols:nat) (ma:chest2 real rows shared) (mb:chest2 real shared cols)
  (a:natlt rows) (c:natlt cols) (t:nat{t <= shared}) (g:nat -> real) : Lemma
  (requires forall (b:nat). b < t ==> g b == acc2 ma a b *. acc2 mb b c)
  (ensures GM.__matmul_single ma mb a c t == vsum t g)
  (decreases t)
  = if t = 0 then mm_vsum_zero g
    else begin
      mm_matmul_col ma mb a c (t - 1) g;
      GM.matmul_single_lemma ma mb a c t;
      mm_vsum_step (t - 1) g
    end

noextract
let mm_dot (ra rb:nat -> real) (oa ob:nat) (kk:nat) : real = ra (oa + kk) *. rb (ob + kk)

let mm_tile_step (ra rb:nat -> real) (k r j:nat) (a:natlt 16) : Lemma
  (acc2 (GM.matmul (mm_ra_tile ra k r j) (mm_rb_tile rb j)) a 0 ==
   vsum 16 (mm_dot ra rb ((16 * r + a) * k + 16 * j) (16 * j)))
  = let ma = mm_ra_tile ra k r j in
    let mb = mm_rb_tile rb j in
    GM.lemma_matmul_index ma mb a 0;
    mm_matmul_col ma mb a 0 16 (mm_dot ra rb ((16 * r + a) * k + 16 * j) (16 * j))

let mm_dot_shift (ra rb:nat -> real) (x j0 j:nat) (b:nat) : Lemma
  (requires j0 <= j)
  (ensures mm_dot ra rb (x + 16 * j) (16 * j) b == mm_dot ra rb (x + 16 * j0) (16 * j0) (16 * (j - j0) + b))
  = assert (x + 16 * j + b == x + 16 * j0 + (16 * (j - j0) + b));
    assert (16 * j + b == 16 * j0 + (16 * (j - j0) + b))

let mm_rtile_unfold (ra rb:nat -> real) (k r j0 j1:nat) (a:natlt 16) : Lemma
  (requires j0 < j1)
  (ensures acc2 (mm_rtile ra rb k r j0 j1) a 0 ==
           acc2 (mm_rtile ra rb k r j0 (j1 - 1)) a 0 +.
           acc2 (GM.matmul (mm_ra_tile ra k r (j1 - 1)) (mm_rb_tile rb (j1 - 1))) a 0)
  = let p = mm_rtile ra rb k r j0 (j1 - 1) in
    let q = GM.matmul (mm_ra_tile ra k r (j1 - 1)) (mm_rb_tile rb (j1 - 1)) in
    Kuiper.EMatrix.chest_comb_acc2 (add #real) p q a 0

let mm_arith16 (j0 j:nat) : Lemma (requires j0 <= j) (ensures 16 * (j + 1 - j0) == 16 * (j - j0) + 16) = ()

let rec mm_rtile_entry (ra rb:nat -> real) (k r j0 j1:nat) (a:natlt 16) : Lemma
  (requires j0 <= j1)
  (ensures acc2 (mm_rtile ra rb k r j0 j1) a 0 == vsum (16 * (j1 - j0)) (mm_dot ra rb ((16 * r + a) * k + 16 * j0) (16 * j0)))
  (decreases j1)
  = let x = (16 * r + a) * k in
    let f = mm_dot ra rb (x + 16 * j0) (16 * j0) in
    if j1 = j0 then mm_vsum_zero f
    else begin
      let j = j1 - 1 in
      let g = mm_dot ra rb (x + 16 * j) (16 * j) in
      mm_rtile_entry ra rb k r j0 j a;
      mm_rtile_unfold ra rb k r j0 j1 a;
      mm_tile_step ra rb k r j a;
      introduce forall (b:nat). b < 16 ==> g b == f (16 * (j - j0) + b)
      with introduce _ ==> _ with mm_dot_shift ra rb x j0 j b;
      mm_arith16 j0 j;
      mm_vsum_split' (16 * (j - j0)) 16 f g
    end

(* final assembly *)
let mm_approx_elim (m1:chest2 f32 16 8) (m2:chest2 real 16 8) (a:natlt 16) (b:natlt 8) : Lemma
  (requires m1 %~ m2) (ensures acc2 m1 a b %~ acc2 m2 a b) = ()

let mm_divmod16 (i:nat) : Lemma (16 * (i / 16) + i % 16 == i) = ()

let mm_model_entry (#n #k:pos) (eA:chest2 bf16 n k) (eB:chest2 bf16 k 8) (j0:nat) (j1:nat) (i:nat) : Lemma
  (requires 16 /? n /\ 16 /? k /\ j1 <= k / 16 /\ i < n)
  (ensures acc2 (mma_model eA eB j0 j1) i 0 == acc2 (mma_tile eA eB (i / 16) j0 j1) (i % 16) 0)
  = ()

let mma_col0_approx (n k:pos) (j0:nat) (j1:nat) (sa sb:S.seq bf16) (ra rb:nat -> real) : Lemma
  (requires 16 /? n /\ 16 /? k /\ j0 <= j1 /\ j1 <= k / 16 /\ S.length sa == n * k /\ S.length sb == k * 8 /\
            (forall (idx:nat). idx < n * k ==> S.index sa idx %~ ra idx) /\
            (forall (kk c:nat). kk < k /\ c < 8 ==> S.index sb (c * k + kk) %~ rb kk))
  (ensures forall (i:nat). i < n ==>
     S.index (to_seq (row_major n 8) (mma_model (from_seq (row_major n k) sa) (from_seq (col_major k 8) sb) j0 j1)) (i * 8)
       %~ vsum (16 * (j1 - j0)) (fun kk -> ra (i * k + 16 * j0 + kk) *. rb (16 * j0 + kk)))
  = let eA = from_seq (row_major n k) sa in
    let eB = from_seq (col_major k 8) sb in
    let m = mma_model eA eB j0 j1 in
    introduce forall (i:nat). i < n ==>
       S.index (to_seq (row_major n 8) m) (i * 8)
         %~ vsum (16 * (j1 - j0)) (fun kk -> ra (i * k + 16 * j0 + kk) *. rb (16 * j0 + kk))
    with introduce _ ==> _ with begin
      let r : natlt (n / 16) = i / 16 in
      let a : natlt 16 = i % 16 in
      mm_to_rm n 8 m i 0;
      mm_model_entry eA eB j0 j1 i;
      assert (S.index (to_seq (row_major n 8) m) (i * 8) == acc2 (mma_tile eA eB r j0 j1) a 0);
      mm_tile_approx n k sa sb ra rb r j0 j1;
      mm_approx_elim (mma_tile eA eB r j0 j1) (mm_rtile ra rb k r j0 j1) a 0;
      mm_rtile_entry ra rb k r j0 j1 a;
      assert (S.index (to_seq (row_major n 8) m) (i * 8) %~ vsum (16 * (j1 - j0)) (mm_dot ra rb ((16 * r + a) * k + 16 * j0) (16 * j0)));
      mm_divmod16 i;
      mm_vsum_ext (16 * (j1 - j0)) (mm_dot ra rb ((16 * r + a) * k + 16 * j0) (16 * j0))
        (fun kk -> ra (i * k + 16 * j0 + kk) *. rb (16 * j0 + kk))
    end

#pop-options

(*** Pure pipeline: exact native value of every intermediate buffer of run_main *)

noextract
let mmr (#n #k:nat) (eA:chest2 bf16 n k) (eB:chest2 bf16 k 8) (j0 j1:nat) : chest2 f32 n 8 =
  if n > 0 && k > 0 && n % 16 = 0 && k % 16 = 0 && j1 <= k / 16 then mma_model #n #k eA eB j0 j1
  else const (n @| 8 @| INil) zero

(* tensor-core product of row-major A (n x k) and column-major B (k x 8), as a row-major seq *)
noextract
let mmo (n k:nat) (sa sb:S.seq bf16) (j0 j1:nat) : GTot (S.seq f32) =
  if S.length sa = n * k && S.length sb = k * 8 then
    to_seq (row_major n 8) (mmr (from_seq (row_major n k) sa) (from_seq (col_major k 8) sb) j0 j1)
  else S.empty

(* weight slice *)
noextract
let wsl (sw:S.seq bf16) (off n:nat) : S.seq bf16 =
  if off + n <= S.length sw then S.slice sw off (off + n) else S.empty

(* B operand: 8 copies of src[soff .. soff + k) *)
noextract
let ubs (k:nat) (src:S.seq bf16) (soff:int) : S.seq bf16 =
  S.init (k * 8) (fun (j:nat{j < k * 8}) -> bat src (soff + j % k))

(* column 0 of an n x 8 product, rounded to bf16 *)
noextract
let col0 (n:nat) (c:S.seq f32) : S.seq bf16 =
  S.init n (fun (j:nat{j < n}) -> tb (fat c (j * 8)))

noextract
let rms_s (src w:S.seq bf16) (woff:int) : S.seq bf16 =
  S.init 1024 (fun (j:nat{j < 1024}) -> rms1024 src w woff j)

noextract
let hn_s (heads stride:nat) (src w cs sn:S.seq bf16) (woff:int) : S.seq bf16 =
  S.init (heads * 256) (fun (j:nat{j < heads * 256}) -> hn_m src w cs sn ((j / 256) * stride) woff (j % 256))

(* front: attention input norm, q/k/v projections, head norms + rope, KV write *)
noextract
let p_hin (sw sx:S.seq bf16) : S.seq bf16 = rms_s sx sw 0
noextract
let p_ub1 (sw sx:S.seq bf16) : S.seq bf16 = ubs 1024 (p_hin sw sx) 0
noextract
let p_cq (sw sx:S.seq bf16) : GTot (S.seq f32) = mmo 4096 1024 (wsl sw 11012096 4194304) (p_ub1 sw sx) 0 64
noextract
let p_pq (sw sx:S.seq bf16) : GTot (S.seq bf16) = col0 4096 (p_cq sw sx)
noextract
let p_ck (sw sx:S.seq bf16) : GTot (S.seq f32) = mmo 512 1024 (wsl sw 15206400 524288) (p_ub1 sw sx) 0 64
noextract
let p_pk (sw sx:S.seq bf16) : GTot (S.seq bf16) = col0 512 (p_ck sw sx)
noextract
let p_cv (sw sx:S.seq bf16) : GTot (S.seq f32) = mmo 512 1024 (wsl sw 15730688 524288) (p_ub1 sw sx) 0 64
noextract
let p_pv (sw sx:S.seq bf16) : GTot (S.seq bf16) = col0 512 (p_cv sw sx)
noextract
let p_qn (sw sx sc ss:S.seq bf16) : GTot (S.seq bf16) = hn_s 8 512 (p_pq sw sx) sw sc ss 16254976
noextract
let p_kn (sw sx sc ss:S.seq bf16) : GTot (S.seq bf16) = hn_s 2 256 (p_pk sw sx) sw sc ss 16255232
noextract
let kvw_s (c t:int) (kv kn pv:S.seq bf16) : S.seq bf16 =
  S.init (S.length kv) (fun (j:nat{j < S.length kv}) -> kvw_m c t kv kn pv j)
noextract
let p_kvo (sw sx skv sc ss:S.seq bf16) (c t:int) : GTot (S.seq bf16) =
  kvw_s c t skv (p_kn sw sx sc ss) (p_pv sw sx)

(* attention scores (dots, 8 heads x 16 positions) *)
noextract
let keys_s (kvo:S.seq bf16) (c h:int) : S.seq bf16 = S.init 4096 (fun (i:nat{i < 4096}) -> keys_m kvo c h i)
noextract
let valm_s (kvo:S.seq bf16) (c h:int) : S.seq bf16 = S.init 4096 (fun (i:nat{i < 4096}) -> valm_m kvo c h i)

(* t == 7: per-head tensor-core products *)
noextract
let dots7_s (qn kvo:S.seq bf16) (c:int) : GTot (S.seq bf16) =
  S.init_ghost 128 (fun (j:nat{j < 128}) -> tb (fat (mmo 16 256 (keys_s kvo c (j / 16)) (ubs 256 qn ((j / 16) * 256)) 0 16) ((j % 16) * 8)))

noextract
let dots_good (qn kvo:S.seq bf16) (c t:nat) (dots:S.seq bf16) : prop =
  S.length dots == 128 /\
  (if t = 7 then dots == dots7_s qn kvo c
   else (forall (h p:nat). h < 8 /\ p < 16 /\ p <= t ==> S.index dots (h * 16 + p) == score_m qn kvo c h p))

noextract
let probs_s (dots:S.seq bf16) (t:nat) : S.seq bf16 =
  S.init 128 (fun (j:nat{j < 128}) -> prob_m dots t (j / 16) (j % 16))

noextract
let vals7_s (probs kvo:S.seq bf16) (c:int) : GTot (S.seq f32) =
  S.init_ghost 2048 (fun (j:nat{j < 2048}) -> fat (mmo 256 16 (valm_s kvo c (j / 256)) (ubs 16 probs ((j / 256) * 16)) 0 1) ((j % 256) * 8))

noextract
let vals_good (probs kvo:S.seq bf16) (c t:nat) (vals:S.seq f32) : prop =
  S.length vals == 2048 /\ (if t = 7 then vals == vals7_s probs kvo c else vals_ok probs kvo c t vals)

noextract
let gated_s (vals:S.seq f32) (pq:S.seq bf16) : S.seq bf16 =
  S.init 2048 (fun (j:nat{j < 2048}) -> gate_m vals pq j)

(* back: o_proj, residual, MLP *)
noextract
let resid_s (a b:S.seq bf16) : S.seq bf16 = S.init 1024 (fun (j:nat{j < 1024}) -> resid_m a b j)
noextract
let act_s (g u:S.seq bf16) : S.seq bf16 = S.init 3584 (fun (j:nat{j < 3584}) -> act_m g u j)
noextract
let col2_s (c0 c1:S.seq f32) : S.seq bf16 = S.init 1024 (fun (j:nat{j < 1024}) -> tb (add (fat c0 (j * 8)) (fat c1 (j * 8))))

noextract
let b_po (sw gated:S.seq bf16) : GTot (S.seq bf16) = col0 1024 (mmo 1024 2048 (wsl sw 16255488 2097152) (ubs 2048 gated 0) 0 128)
noextract
let b_x1 (sw sx gated:S.seq bf16) : GTot (S.seq bf16) = resid_s sx (b_po sw gated)
noextract
let b_h2 (sw sx gated:S.seq bf16) : GTot (S.seq bf16) = rms_s (b_x1 sw sx gated) sw 1024
noextract
let b_ub2 (sw sx gated:S.seq bf16) : GTot (S.seq bf16) = ubs 1024 (b_h2 sw sx gated) 0
noextract
let b_g (sw sx gated:S.seq bf16) : GTot (S.seq bf16) = col0 3584 (mmo 3584 1024 (wsl sw 2048 3670016) (b_ub2 sw sx gated) 0 64)
noextract
let b_u (sw sx gated:S.seq bf16) : GTot (S.seq bf16) = col0 3584 (mmo 3584 1024 (wsl sw 3672064 3670016) (b_ub2 sw sx gated) 0 64)
noextract
let b_uba (sw sx gated:S.seq bf16) : GTot (S.seq bf16) = ubs 3584 (act_s (b_g sw sx gated) (b_u sw sx gated)) 0
noextract
let b_pd (sw sx gated:S.seq bf16) : GTot (S.seq bf16) =
  col2_s (mmo 1024 3584 (wsl sw 7342080 3670016) (b_uba sw sx gated) 0 112)
         (mmo 1024 3584 (wsl sw 7342080 3670016) (b_uba sw sx gated) 112 224)
noextract
let b_xo (sw sx gated:S.seq bf16) : GTot (S.seq bf16) = resid_s (b_x1 sw sx gated) (b_pd sw sx gated)

noextract
let pipe_ok (sw sx skv sc ss:S.seq bf16) (c t:nat) (xo kvo:S.seq bf16) : prop =
  kvo == p_kvo sw sx skv sc ss c t /\
  (exists (dots:S.seq bf16) (vals:S.seq f32).
     dots_good (p_qn sw sx sc ss) kvo c t dots /\
     vals_good (probs_s dots t) kvo c t vals /\
     xo == b_xo sw sx (gated_s vals (p_pq sw sx)))

let pipe_intro (sw sx skv sc ss:S.seq bf16) (c t:nat) (dots:S.seq bf16) (vals:S.seq f32) (xo kvo:S.seq bf16) : Lemma
  (requires kvo == p_kvo sw sx skv sc ss c t /\ dots_good (p_qn sw sx sc ss) kvo c t dots /\
     vals_good (probs_s dots t) kvo c t vals /\ xo == b_xo sw sx (gated_s vals (p_pq sw sx)))
  (ensures pipe_ok sw sx skv sc ss c t xo kvo) = ()


let upd_step (#a:Type) (s:S.seq a) (n:nat) (v:a) (f:nat -> a)
  : Lemma (requires n < S.length s /\ (forall (j:nat). j < n ==> S.index s j == f j) /\ v == f n)
          (ensures S.length (S.upd s n v) == S.length s /\ (forall (j:nat). j < n + 1 ==> S.index (S.upd s n v) j == f j))
  = ()

let keys_bound (c h r j:nat) : Lemma (requires h < 8 /\ j < 256 /\ 8 <= c /\ c <= 12)
  (ensures (h / 4) * c <= 12 /\ (r < 8 ==> ((h / 4) * c + r) * 256 + j < c * 1024) /\ (r < 8 ==> c * 512 + ((h / 4) * c + r) * 256 + j < c * 1024))
  = if h / 4 = 0 then () else ()

let comp_bound (im q:nat) : Lemma (requires q > 0 /\ im < 2 * q) (ensures im / q < 2)
  = FStar.Math.Lemmas.lemma_div_le im (2 * q - 1) q;
    FStar.Math.Lemmas.lemma_div_plus (q - 1) 1 q;
    FStar.Math.Lemmas.small_div (q - 1) q

let kvw_bound (vi c:nat) : Lemma (requires c > 0) (ensures ((vi % (c * 512)) / (c * 256)) * 256 + vi % 256 < 512)
  = FStar.Math.Lemmas.lemma_mod_lt vi (c * 512); comp_bound (vi % (c * 512)) (c * 256)

let hn_bound (vh heads stride m m2:nat) : Lemma (requires vh < heads /\ heads * stride <= m /\ 256 <= stride /\ heads * 256 <= m2)
  (ensures vh * stride + 256 <= m /\ vh * stride + stride <= m /\ vh * 256 + 256 <= m2)
  = FStar.Math.Lemmas.lemma_mult_le_right stride (vh + 1) heads

let kvw_val (c t:nat) (kv kn pv:S.seq bf16) (i pos comp:nat) (a b o:bf16) : Lemma
  (requires c > 0 /\ pos == ((i % (c * 512)) % (c * 256)) / 256 /\ comp == ((i % (c * 512)) / (c * 256)) * 256 + i % 256 /\
            comp < 512 /\ S.length kn == 512 /\ S.length pv == 512 /\ i < S.length kv /\
            a == S.index kn comp /\ b == S.index pv comp /\ o == S.index kv i)
  (ensures (if pos = t then (if i < c * 512 then a else b) else o) == kvw_m c t kv kn pv i) = ()

let kvw_idx (c i half quarter im pos comp:nat) : Lemma
  (requires c > 0 /\ half == c * 512 /\ quarter == c * 256 /\ im == i % half /\ pos == (im % quarter) / 256 /\
            comp == (im / quarter) * 256 + i % 256)
  (ensures pos == ((i % (c * 512)) % (c * 256)) / 256 /\ comp == ((i % (c * 512)) / (c * 256)) * 256 + i % 256 /\ comp < 512)
  = kvw_bound i c
#push-options "--z3rlimit 40"

(*** Kernels with exact native posts *)

let halve_lem (s:S.seq f32) (o:nat) (s':S.seq f32)
  : Lemma (requires S.length s' == S.length s /\ 2 * o <= S.length s /\
                    (forall (j:nat). j < S.length s ==> S.index s' j == (if j < o then add (S.index s j) (S.index s (j + o)) else S.index s j)))
          (ensures s' == halve s o)
  = S.lemma_eq_intro s' (halve s o)

let tree_step (s:S.seq f32) (o:nat{o > 0}) : Lemma (tree s o == tree (halve s o) (o / 2)) = ()

(* in-place halving tree p[i] += p[i+off], off = count/2 .. 1 *)
inline_for_extraction noextract
fn tree_k (#m : erased nat) (buf : larray f32 m) (count : sz { SZ.v count <= m })
  (#o_buf : erased (S.seq f32))
  requires buf |-> o_buf
  ensures (exists* (s:S.seq f32). buf |-> s ** pure (S.length s == m /\ s == tree o_buf (SZ.v count / 2)))
{
  pts_to_len buf;
  let mut off : sz = count /^ 2sz;
  while (!off >^ 0sz)
    invariant exists* vo s. off |-> vo ** buf |-> s ** pure (2 * SZ.v vo <= SZ.v count /\ S.length s == m /\
      tree s (SZ.v vo) == tree o_buf (SZ.v count / 2))
    decreases (SZ.v !off)
  {
    let o = !off;
    with s0. assert (buf |-> s0);
    let mut i : sz = 0sz;
    while (!i <^ o)
      invariant exists* vi s. i |-> vi ** buf |-> s ** pure (SZ.v vi <= SZ.v o /\ S.length s == m /\
        (forall (j:nat). j < m ==> S.index s j == (if j < SZ.v vi then add (S.index s0 j) (S.index s0 (j + SZ.v o)) else S.index s0 j)))
      decreases (SZ.v o - SZ.v !i)
    {
      let vi = !i;
      let a = buf.(vi);
      let b = buf.(vi +^ o);
      buf.(vi) <- add a b;
      i := vi +^ 1sz;
    };
    with s1. assert (buf |-> s1);
    halve_lem s0 (SZ.v o) s1;
    tree_step s0 (SZ.v o);
    off := o /^ 2sz;
  };
}

(* dst[i] = bf16(f a[i] + f b[i]) *)
inline_for_extraction noextract
fn resid_k (a b : larray bf16 1024) (dst : larray bf16 1024) (#pa #pb : perm) (#sa #sb : erased (S.seq bf16))
  (#o_dst : erased (S.seq bf16))
  preserves gpu ** a |-> Frac pa sa ** b |-> Frac pb sb
  requires dst |-> o_dst
  ensures (exists* (s:S.seq bf16). dst |-> s ** pure (S.length s == 1024 /\ (forall (j:nat). j < 1024 ==> S.index s j == resid_m sa sb j)))
{
  pts_to_len a; pts_to_len b; pts_to_len dst;
  let mut i : sz = 0sz;
  while (!i <^ 1024sz)
    invariant exists* vi s. i |-> vi ** dst |-> s ** pure (SZ.v vi <= 1024 /\ S.length s == 1024 /\
      (forall (j:nat). j < SZ.v vi ==> S.index s j == resid_m sa sb j))
    decreases (1024 - SZ.v !i)
  {
    let vi = !i;
    let u = fb a.(vi);
    let v = fb b.(vi);
    dst.(vi) <- tb (add u v);
    i := vi +^ 1sz;
  };
}

#pop-options
#push-options "--z3rlimit 40"

(* MLP activation *)
inline_for_extraction noextract
fn act_k (g u : larray bf16 3584) (dst : larray bf16 3584)
  (#p1 #p2 : perm) (#s1 #s2 : erased (S.seq bf16))
  (#o_dst : erased (S.seq bf16))
  preserves gpu ** g |-> Frac p1 s1 ** u |-> Frac p2 s2
  requires dst |-> o_dst
  ensures (exists* (s:S.seq bf16). dst |-> s ** pure (S.length s == 3584 /\
    (forall (j:nat). j < 3584 ==> S.index s j == act_m s1 s2 j)))
{
  pts_to_len g; pts_to_len u; pts_to_len dst;
  let mut i : sz = 0sz;
  while (!i <^ 3584sz)
    invariant exists* vi s. i |-> vi ** dst |-> s ** pure (SZ.v vi <= 3584 /\ S.length s == 3584 /\
      (forall (j:nat). j < SZ.v vi ==> S.index s j == act_m s1 s2 j))
    decreases (3584 - SZ.v !i)
  {
    let vi = !i;
    let gv = fb g.(vi);
    let uv = fb u.(vi);
    dst.(vi) <- tb (mul (fb (tb (silu_f gv))) uv);
    i := vi +^ 1sz;
  };
}


#pop-options
#push-options "--z3rlimit 40"

(* dst[doff + i] = c[i*8] (f32) *)
inline_for_extraction noextract
fn colf_k (c : larray f32 2048) (dst : larray f32 2048) (doff : sz { SZ.v doff + 256 <= 2048 })
  (#p : perm) (#s : erased (S.seq f32))
  (#o_dst : erased (S.seq f32))
  preserves gpu ** c |-> Frac p s
  requires dst |-> o_dst
  ensures (exists* (s':S.seq f32). dst |-> s' ** pure (S.length s' == 2048 /\
    (forall (j:nat). j < 2048 ==> S.index s' j == (if SZ.v doff <= j && j < SZ.v doff + 256 then fat s ((j - SZ.v doff) * 8) else fat o_dst j))))
{
  pts_to_len c; pts_to_len dst;
  let mut i : sz = 0sz;
  while (!i <^ 256sz)
    invariant exists* vi s'. i |-> vi ** dst |-> s' ** pure (SZ.v vi <= 256 /\ S.length s' == 2048 /\
      (forall (j:nat). j < 2048 ==> S.index s' j == (if SZ.v doff <= j && j < SZ.v doff + SZ.v vi then fat s ((j - SZ.v doff) * 8) else fat o_dst j)))
    decreases (256 - SZ.v !i)
  {
    let vi = !i;
    let v = c.(vi *^ 8sz);
    dst.(doff +^ vi) <- v;
    i := vi +^ 1sz;
  };
}


#pop-options
#push-options "--z3rlimit 40"

(* dst[i] = bf16(c0[i*8] + c1[i*8]) *)
inline_for_extraction noextract
fn colx2_k (c0 c1 : larray f32 8192) (dst : larray bf16 1024) (#p0 #p1 : perm) (#s0 #s1 : erased (S.seq f32))
  (#o_dst : erased (S.seq bf16))
  preserves gpu ** c0 |-> Frac p0 s0 ** c1 |-> Frac p1 s1
  requires dst |-> o_dst
  ensures (exists* (s':S.seq bf16). dst |-> s' ** pure (S.length s' == 1024 /\
    (forall (j:nat). j < 1024 ==> S.index s' j == tb (add (fat s0 (j * 8)) (fat s1 (j * 8))))))
{
  pts_to_len c0; pts_to_len c1; pts_to_len dst;
  let mut i : sz = 0sz;
  while (!i <^ 1024sz)
    invariant exists* vi s'. i |-> vi ** dst |-> s' ** pure (SZ.v vi <= 1024 /\ S.length s' == 1024 /\
      (forall (j:nat). j < SZ.v vi ==> S.index s' j == tb (add (fat s0 (j * 8)) (fat s1 (j * 8)))))
    decreases (1024 - SZ.v !i)
  {
    let vi = !i;
    let u = c0.(vi *^ 8sz);
    let v = c1.(vi *^ 8sz);
    dst.(vi) <- tb (add u v);
    i := vi +^ 1sz;
  };
}


#pop-options
#push-options "--z3rlimit 40"

(* dst[doff + i] = bf16(c[i*8]) *)
inline_for_extraction noextract
fn colx_k (#m : erased nat) (n : sz { SZ.fits (SZ.v n * 8) }) (c : larray f32 (SZ.v n * 8))
  (dst : larray bf16 m) (doff : sz { SZ.v doff + SZ.v n <= m }) (#p : perm) (#s : erased (S.seq f32))
  (#o_dst : erased (S.seq bf16))
  preserves gpu ** c |-> Frac p s
  requires dst |-> o_dst
  ensures (exists* (s':S.seq bf16). dst |-> s' ** pure (S.length s' == m /\
    (forall (j:nat). j < m ==> S.index s' j == (if SZ.v doff <= j && j < SZ.v doff + SZ.v n then tb (fat s ((j - SZ.v doff) * 8)) else bat o_dst j))))
{
  pts_to_len c; pts_to_len dst;
  let mut i : sz = 0sz;
  while (!i <^ n)
    invariant exists* vi s'. i |-> vi ** dst |-> s' ** pure (SZ.v vi <= SZ.v n /\ S.length s' == m /\
      (forall (j:nat). j < m ==> S.index s' j == (if SZ.v doff <= j && j < SZ.v doff + SZ.v vi then tb (fat s ((j - SZ.v doff) * 8)) else bat o_dst j)))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let v = c.(vi *^ 8sz);
    dst.(doff +^ vi) <- tb v;
    i := vi +^ 1sz;
  };
}


#pop-options
#push-options "--z3rlimit 40"

(* gated *)
inline_for_extraction noextract
fn gate_k (vals : larray f32 2048) (pq : larray bf16 4096) (gated : larray bf16 2048)
  (#p1 #p2 : perm) (#s1 : erased (S.seq f32)) (#s2 : erased (S.seq bf16))
  (#o_gated : erased (S.seq bf16))
  preserves gpu ** vals |-> Frac p1 s1 ** pq |-> Frac p2 s2
  requires gated |-> o_gated
  ensures (exists* (s:S.seq bf16). gated |-> s ** pure (S.length s == 2048 /\
    (forall (j:nat). j < 2048 ==> S.index s j == gate_m s1 s2 j)))
{
  pts_to_len vals; pts_to_len pq; pts_to_len gated;
  let mut i : sz = 0sz;
  while (!i <^ 2048sz)
    invariant exists* vi s. i |-> vi ** gated |-> s ** pure (SZ.v vi <= 2048 /\ S.length s == 2048 /\
      (forall (j:nat). j < SZ.v vi ==> S.index s j == gate_m s1 s2 j))
    decreases (2048 - SZ.v !i)
  {
    let vi = !i;
    let v = vals.(vi);
    let g = fb pq.((vi /^ 256sz) *^ 512sz +^ 256sz +^ vi %^ 256sz);
    gated.(vi) <- tb (mul (fb (tb v)) (fb (tb (sigm g))));
    i := vi +^ 1sz;
  };
}


#pop-options

#push-options "--z3rlimit 40"
(* t == 7 key matrix *)
inline_for_extraction noextract
fn keys_k (nkv c h : sz) (#_ : squash (SZ.v nkv == SZ.v c * 1024 /\ 8 <= SZ.v c /\ SZ.v c <= 12 /\ SZ.v h < 8))
  (kvo : larray bf16 (SZ.v nkv)) (ka : larray bf16 4096) (#p : perm) (#s : erased (S.seq bf16))
  (#o_ka : erased (S.seq bf16))
  preserves gpu ** kvo |-> Frac p s
  requires ka |-> o_ka
  ensures (exists* (s':S.seq bf16). ka |-> s' ** pure (S.length s' == 4096 /\
    (forall (j:nat). j < 4096 ==> S.index s' j == keys_m s (SZ.v c) (SZ.v h) j)))
{
  pts_to_len kvo; pts_to_len ka;
  let mut i : sz = 0sz;
  while (!i <^ 4096sz)
    invariant exists* vi s'. i |-> vi ** ka |-> s' ** pure (SZ.v vi <= 4096 /\ S.length s' == 4096 /\
      (forall (j:nat). j < SZ.v vi ==> S.index s' j == keys_m s (SZ.v c) (SZ.v h) j))
    decreases (4096 - SZ.v !i)
  {
    let vi = !i;
    let r = vi /^ 256sz;
    let j = vi %^ 256sz;
    keys_bound (SZ.v c) (SZ.v h) (SZ.v r) (SZ.v j);
    let idx = (if r <^ 8sz then ((h /^ 4sz) *^ c +^ r) *^ 256sz +^ j else 0sz);
    let x = kvo.(idx);
    let z : bf16 = zero;
    let v = (if r <^ 8sz then x else z);
    with sc. assert (ka |-> sc);
    assert pure (v == keys_m s (SZ.v c) (SZ.v h) (SZ.v vi));
    upd_step sc (SZ.v vi) v (keys_m s (SZ.v c) (SZ.v h));
    ka.(vi) <- v;
    i := vi +^ 1sz;
  };
}

#pop-options

#push-options "--z3rlimit 40"
inline_for_extraction noextract
let kvw_pos (c:sz{1 <= SZ.v c /\ SZ.v c <= 12}) (vi:sz) : (r:sz{SZ.v r == base_kv_pos (SZ.v c) (SZ.v vi)}) =
  ((vi %^ (c *^ 512sz)) %^ (c *^ 256sz)) /^ 256sz

inline_for_extraction noextract
let kvw_comp (c:sz{1 <= SZ.v c /\ SZ.v c <= 12}) (vi:sz) : (r:sz{SZ.v r == base_kv_comp (SZ.v c) (SZ.v vi) /\ SZ.v r < 512}) =
  kvw_bound (SZ.v vi) (SZ.v c);
  comp_bound (SZ.v vi % (SZ.v c * 512)) (SZ.v c * 256);
  FStar.Math.Lemmas.lemma_mod_lt (SZ.v vi) (SZ.v c * 512);
  ((vi %^ (c *^ 512sz)) /^ (c *^ 256sz)) *^ 256sz +^ vi %^ 256sz

(* KV write *)
inline_for_extraction noextract
fn kvw_k (nkv c t : sz) (#_ : squash (SZ.v nkv == SZ.v c * 1024 /\ 1 <= SZ.v c /\ SZ.v c <= 12 /\ SZ.v t < SZ.v c))
  (kv : larray bf16 (SZ.v nkv)) (kn pv : larray bf16 512) (kvo : larray bf16 (SZ.v nkv))
  (#p1 #p2 #p3 : perm) (#s1 #s2 #s3 : erased (S.seq bf16))
  (#o_kvo : erased (S.seq bf16))
  preserves gpu ** kv |-> Frac p1 s1 ** kn |-> Frac p2 s2 ** pv |-> Frac p3 s3
  requires kvo |-> o_kvo
  ensures (exists* (s:S.seq bf16). kvo |-> s ** pure (S.length s == SZ.v nkv /\
    (forall (j:nat). j < SZ.v nkv ==> S.index s j == kvw_m (SZ.v c) (SZ.v t) s1 s2 s3 j)))
{
  pts_to_len kv; pts_to_len kn; pts_to_len pv; pts_to_len kvo;
  let half = c *^ 512sz;
  let quarter = c *^ 256sz;
  assert pure (SZ.v half == SZ.v c * 512 /\ SZ.v quarter == SZ.v c * 256);
  let mut i : sz = 0sz;
  while (!i <^ nkv)
    invariant exists* vi s. i |-> vi ** kvo |-> s ** pure (SZ.v vi <= SZ.v nkv /\ S.length s == SZ.v nkv /\
      (forall (j:nat). j < SZ.v vi ==> S.index s j == kvw_m (SZ.v c) (SZ.v t) s1 s2 s3 j))
    decreases (SZ.v nkv - SZ.v !i)
  {
    let vi = !i;
    let pos = kvw_pos c vi;
    let comp = kvw_comp c vi;
    let a = kn.(comp);
    let b = pv.(comp);
    let o = kv.(vi);
    let v = (if pos = t then (if vi <^ half then a else b) else o);
    with sc. assert (kvo |-> sc);
    assert pure (v == kvw_m (SZ.v c) (SZ.v t) s1 s2 s3 (SZ.v vi));
    upd_step sc (SZ.v vi) v (kvw_m (SZ.v c) (SZ.v t) s1 s2 s3);
    kvo.(vi) <- v;
    i := vi +^ 1sz;
  };
}

#pop-options
#push-options "--z3rlimit 40"

(* dst[i] = src[soff + i % k], i < k*8 *)
inline_for_extraction noextract
fn ub_k (#m : erased nat) (k : sz { SZ.v k > 0 /\ SZ.fits (SZ.v k * 8) }) (src : larray bf16 m) (soff : sz { SZ.v soff + SZ.v k <= m })
  (dst : larray bf16 (SZ.v k * 8)) (#p : perm) (#s : erased (S.seq bf16))
  (#o_dst : erased (S.seq bf16))
  preserves gpu ** src |-> Frac p s
  requires dst |-> o_dst
  ensures (exists* (s':S.seq bf16). dst |-> s' ** pure (S.length s' == SZ.v k * 8 /\
    (forall (j:nat). j < SZ.v k * 8 ==> S.index s' j == bat s (SZ.v soff + j % SZ.v k))))
{
  pts_to_len src; pts_to_len dst;
  let tot = k *^ 8sz;
  let mut i : sz = 0sz;
  while (!i <^ tot)
    invariant exists* vi s'. i |-> vi ** dst |-> s' ** pure (SZ.v vi <= SZ.v k * 8 /\ S.length s' == SZ.v k * 8 /\
      (forall (j:nat). j < SZ.v vi ==> S.index s' j == bat s (SZ.v soff + j % SZ.v k)))
    decreases (SZ.v k * 8 - SZ.v !i)
  {
    let vi = !i;
    let v = src.(soff +^ (vi %^ k));
    dst.(vi) <- v;
    i := vi +^ 1sz;
  };
}


#pop-options

#push-options "--z3rlimit 40"
(* t == 7 value matrix *)
inline_for_extraction noextract
fn valm_k (nkv c h : sz) (#_ : squash (SZ.v nkv == SZ.v c * 1024 /\ 8 <= SZ.v c /\ SZ.v c <= 12 /\ SZ.v h < 8))
  (kvo : larray bf16 (SZ.v nkv)) (va : larray bf16 4096) (#p : perm) (#s : erased (S.seq bf16))
  (#o_va : erased (S.seq bf16))
  preserves gpu ** kvo |-> Frac p s
  requires va |-> o_va
  ensures (exists* (s':S.seq bf16). va |-> s' ** pure (S.length s' == 4096 /\
    (forall (j:nat). j < 4096 ==> S.index s' j == valm_m s (SZ.v c) (SZ.v h) j)))
{
  pts_to_len kvo; pts_to_len va;
  let half = c *^ 512sz;
  let mut i : sz = 0sz;
  while (!i <^ 4096sz)
    invariant exists* vi s'. i |-> vi ** va |-> s' ** pure (SZ.v vi <= 4096 /\ S.length s' == 4096 /\
      (forall (j:nat). j < SZ.v vi ==> S.index s' j == valm_m s (SZ.v c) (SZ.v h) j))
    decreases (4096 - SZ.v !i)
  {
    let vi = !i;
    let j = vi /^ 16sz;
    let pp = vi %^ 16sz;
    keys_bound (SZ.v c) (SZ.v h) (SZ.v pp) (SZ.v j);
    let idx = (if pp <^ 8sz then half +^ ((h /^ 4sz) *^ c +^ pp) *^ 256sz +^ j else 0sz);
    let x = kvo.(idx);
    let z : bf16 = zero;
    let v = (if pp <^ 8sz then x else z);
    with sc. assert (va |-> sc);
    assert pure (v == valm_m s (SZ.v c) (SZ.v h) (SZ.v vi));
    upd_step sc (SZ.v vi) v (valm_m s (SZ.v c) (SZ.v h));
    va.(vi) <- v;
    i := vi +^ 1sz;
  };
}

#pop-options

noextract
let sqg (src:S.seq bf16) (base:int) (g:nat) (j:nat) : f32 = sqf (bat src (base + 4 * g + j))

#push-options "--fuel 5 --ifuel 1"
let sq4_eq (src:S.seq bf16) (base:int) (g:nat) : Lemma (sq4 src base g == fsum 4 (sqg src base g)) = ()
#pop-options

let fsum_step (n:nat) (f:nat -> f32) : Lemma (fsum (n + 1) f == add (fsum n f) (f n)) = ()

(* tmp[g] = sum of squares of src[base + 4g .. base + 4g + 4) for g < cnt *)
inline_for_extraction noextract
fn sqs_k (#m : erased nat) (src : larray bf16 m) (base cnt : sz)
  (#_ : squash (SZ.v base + 4 * SZ.v cnt <= m /\ SZ.v cnt <= 1024))
  (tmp : larray f32 1024) (#p : perm) (#s : erased (S.seq bf16)) (#o_tmp : erased (S.seq f32))
  preserves src |-> Frac p s
  requires tmp |-> o_tmp
  ensures (exists* (st:S.seq f32). tmp |-> st ** pure (S.length st == 1024 /\
    (forall (g:nat). g < SZ.v cnt ==> S.index st g == sq4 s (SZ.v base) g)))
{
  pts_to_len src; pts_to_len tmp;
  let mut g : sz = 0sz;
  while (!g <^ cnt)
    invariant exists* vg st. g |-> vg ** tmp |-> st ** pure (SZ.v vg <= SZ.v cnt /\ S.length st == 1024 /\
      (forall (g':nat). g' < SZ.v vg ==> S.index st g' == sq4 s (SZ.v base) g'))
    decreases (SZ.v cnt - SZ.v !g)
  {
    let vg = !g;
    let mut acc : f32 = zero;
    let mut j : sz = 0sz;
    while (!j <^ 4sz)
      invariant exists* vj va. j |-> vj ** acc |-> va ** pure (SZ.v vj <= 4 /\ va == fsum (SZ.v vj) (sqg s (SZ.v base) (SZ.v vg)))
      decreases (4 - SZ.v !j)
    {
      let vj = !j;
      let v = fb src.(base +^ 4sz *^ vg +^ vj);
      let a = !acc;
      fsum_step (SZ.v vj) (sqg s (SZ.v base) (SZ.v vg));
      acc := add a (mul v v);
      j := vj +^ 1sz;
    };
    let a = !acc;
    sq4_eq s (SZ.v base) (SZ.v vg);
    tmp.(vg) <- a;
    g := vg +^ 1sz;
  };
}

#push-options "--z3rlimit 40"
(* RMSNorm over 1024 with (1 + w[woff + i]) *)
inline_for_extraction noextract
fn rms_k (src : larray bf16 1024) (w : larray bf16 18352640) (woff : sz { SZ.v woff + 1024 <= 18352640 })
  (tmp : larray f32 1024) (dst : larray bf16 1024) (#ps #pw : perm) (#ss #sw : erased (S.seq bf16))
  (#o_tmp : erased (S.seq f32)) (#o_dst : erased (S.seq bf16))
  preserves gpu ** src |-> Frac ps ss ** w |-> Frac pw sw
  requires tmp |-> o_tmp ** dst |-> o_dst
  ensures (exists* (s:S.seq f32). tmp |-> s) ** (exists* (s:S.seq bf16). dst |-> s ** pure (S.length s == 1024 /\
    (forall (j:nat). j < 1024 ==> S.index s j == rms1024 ss sw (SZ.v woff) j)))
{
  pts_to_len src; pts_to_len w; pts_to_len dst;
  sqs_k src 0sz 256sz tmp;
  with s1. assert (tmp |-> s1);
  tree_k tmp 256sz;
  tree_prefix s1 256 (sq4 ss 0);
  let s = tmp.(0sz);
  let inv = FB.rsqrt (add (mul s f_inv1024) f_eps);
  assert pure (inv == rms_inv ss 0 256 f_inv1024);
  let mut i : sz = 0sz;
  while (!i <^ 1024sz)
    invariant exists* vi sd. i |-> vi ** dst |-> sd ** pure (SZ.v vi <= 1024 /\ S.length sd == 1024 /\
      (forall (j:nat). j < SZ.v vi ==> S.index sd j == rms1024 ss sw (SZ.v woff) j))
    decreases (1024 - SZ.v !i)
  {
    let vi = !i;
    let xv = fb src.(vi);
    let wv = fb w.(woff +^ vi);
    let v = tb (mul (mul xv inv) (add one wv));
    with sc. assert (dst |-> sc);
    assert pure (v == rms1024 ss sw (SZ.v woff) (SZ.v vi));
    upd_step sc (SZ.v vi) v (rms1024 ss sw (SZ.v woff));
    dst.(vi) <- v;
    i := vi +^ 1sz;
  };
}


noextract
let rng_ok (#a:Type) (s s0:S.seq a) (ob n:nat) (f:nat -> a) : prop =
  S.length s == S.length s0 /\ (forall (j:nat). j < S.length s ==> S.index s j == (if ob <= j && j < ob + n then f (j - ob) else S.index s0 j))

let rng_step (#a:Type) (s s0:S.seq a) (ob n:nat) (v:a) (f:nat -> a)
  : Lemma (requires rng_ok s s0 ob n f /\ ob + n < S.length s /\ v == f n) (ensures rng_ok (S.upd s (ob + n) v) s0 ob (n + 1) f) = ()

noextract
let rope_ok (s s0:S.seq bf16) (ob vr:nat) (f g:nat -> bf16) : prop =
  S.length s == S.length s0 /\ (forall (j:nat). j < S.length s ==> S.index s j ==
    (if ob <= j && j < ob + 256 then
       (if j < ob + vr || (ob + 32 <= j && j < ob + 32 + vr) then f (j - ob) else g (j - ob))
     else S.index s0 j))

let rope_init (s s0:S.seq bf16) (ob:nat) (f g:nat -> bf16)
  : Lemma (requires rng_ok s s0 ob 256 g) (ensures rope_ok s s0 ob 0 f g) = ()

let rope_read (s s0:S.seq bf16) (ob vr:nat) (f g:nat -> bf16)
  : Lemma (requires rope_ok s s0 ob vr f g /\ vr < 32 /\ ob + 256 <= S.length s)
          (ensures S.index s (ob + vr) == g vr /\ S.index s (ob + 32 + vr) == g (32 + vr)) = ()

let rope_step (s s0:S.seq bf16) (ob vr:nat) (f g:nat -> bf16) (va vb:bf16)
  : Lemma (requires rope_ok s s0 ob vr f g /\ vr < 32 /\ ob + 256 <= S.length s /\ va == f vr /\ vb == f (32 + vr))
          (ensures rope_ok (S.upd (S.upd s (ob + vr) va) (ob + 32 + vr) vb) s0 ob (vr + 1) f g) = ()

let hn_idx (j vh:nat) : Lemma (requires vh * 256 <= j /\ j < vh * 256 + 256) (ensures j / 256 == vh /\ j % 256 == j - vh * 256) = ()


let hn_step (s0 sd:S.seq bf16) (vh ob base vr:nat) (s sw sc ss:S.seq bf16) (stride woff:nat) : Lemma
  (requires (vh + 1) * 256 <= S.length sd /\ S.length s0 == S.length sd /\ ob == vh * 256 /\ base == vh * stride /\ vr == 32 /\
     (forall (j:nat). j < vh * 256 ==> S.index s0 j == hn_m s sw sc ss ((j / 256) * stride) woff (j % 256)) /\
     rope_ok sd s0 ob vr (hn_m s sw sc ss base woff) (hnorm s sw base woff))
  (ensures forall (j:nat). j < (vh + 1) * 256 ==> S.index sd j == hn_m s sw sc ss ((j / 256) * stride) woff (j % 256))
  = let aux (j:nat) : Lemma (j < (vh + 1) * 256 ==> S.index sd j == hn_m s sw sc ss ((j / 256) * stride) woff (j % 256)) =
      if j < vh * 256 then () else if j < (vh + 1) * 256 then hn_idx j vh else ()
    in
    Classical.forall_intro aux

(* per-head RMSNorm (1+w) and RoPE on the first 64 dims *)
inline_for_extraction noextract
fn hn_k (#m #m2 : erased nat) (heads stride : sz) (src : larray bf16 m)
  (#_ : squash (SZ.v heads * SZ.v stride <= m /\ SZ.v heads * 256 <= m2 /\ 256 <= SZ.v stride /\ SZ.fits m /\ SZ.fits m2))
  (w : larray bf16 18352640) (woff : sz { SZ.v woff + 256 <= 18352640 })
  (cs sn : larray bf16 32) (tmp : larray f32 1024) (dst : larray bf16 m2)
  (#p #pw #pc #ps : perm) (#s #sw #sc #ss : erased (S.seq bf16))
  (#o_tmp : erased (S.seq f32)) (#o_dst : erased (S.seq bf16))
  preserves gpu ** src |-> Frac p s ** w |-> Frac pw sw ** cs |-> Frac pc sc ** sn |-> Frac ps ss
  requires tmp |-> o_tmp ** dst |-> o_dst
  ensures (exists* (s:S.seq f32). tmp |-> s) ** (exists* (sd:S.seq bf16). dst |-> sd ** pure (S.length sd == m2 /\
    (forall (j:nat). j < SZ.v heads * 256 ==> S.index sd j == hn_m s sw sc ss ((j / 256) * SZ.v stride) (SZ.v woff) (j % 256))))
{
  pts_to_len src; pts_to_len w; pts_to_len cs; pts_to_len sn; pts_to_len dst;
  let mut h : sz = 0sz;
  while (!h <^ heads)
    invariant exists* vh st sd. h |-> vh ** tmp |-> st ** dst |-> sd ** pure (SZ.v vh <= SZ.v heads /\ S.length sd == m2 /\
      (forall (j:nat). j < SZ.v vh * 256 ==> S.index sd j == hn_m s sw sc ss ((j / 256) * SZ.v stride) (SZ.v woff) (j % 256)))
    decreases (SZ.v heads - SZ.v !h)
  {
    let vh = !h;
    hn_bound (SZ.v vh) (SZ.v heads) (SZ.v stride) m m2;
    let base = vh *^ stride;
    assert pure (SZ.v base == SZ.v vh * SZ.v stride);
    let ob = vh *^ 256sz;
    assert pure (SZ.v ob == SZ.v vh * 256);
    assert pure (SZ.v base + 256 <= S.length s /\ SZ.v ob + 256 <= m2);
    with s0. assert (dst |-> s0);
    sqs_k src base 64sz tmp;
    with s1. assert (tmp |-> s1);
    tree_k tmp 64sz;
    tree_prefix s1 64 (sq4 s (SZ.v base));
    let sm = tmp.(0sz);
    let inv = FB.rsqrt (add (mul sm f_inv256) f_eps);
    assert pure (inv == rms_inv s (SZ.v base) 64 f_inv256);
    let mut i : sz = 0sz;
    while (!i <^ 256sz)
      invariant exists* vi sd. i |-> vi ** dst |-> sd ** pure (SZ.v vi <= 256 /\ S.length sd == m2 /\
        rng_ok sd s0 (SZ.v ob) (SZ.v vi) (hnorm s sw (SZ.v base) (SZ.v woff)))
      decreases (256 - SZ.v !i)
    {
      let vi = !i;
      assert pure (SZ.v base + SZ.v vi < S.length s);
      let xv = fb src.(base +^ vi);
      let wv = fb w.(woff +^ vi);
      let v = tb (mul (mul xv inv) (add one wv));
      with scur. assert (dst |-> scur);
      assert pure (v == hnorm s sw (SZ.v base) (SZ.v woff) (SZ.v vi));
      rng_step scur s0 (SZ.v ob) (SZ.v vi) v (hnorm s sw (SZ.v base) (SZ.v woff));
      dst.(ob +^ vi) <- v;
      i := vi +^ 1sz;
    };
    with sd1. assert (dst |-> sd1);
    rope_init sd1 s0 (SZ.v ob) (hn_m s sw sc ss (SZ.v base) (SZ.v woff)) (hnorm s sw (SZ.v base) (SZ.v woff));
    let mut r : sz = 0sz;
    while (!r <^ 32sz)
      invariant exists* vr sd. r |-> vr ** dst |-> sd ** pure (SZ.v vr <= 32 /\ S.length sd == m2 /\
        rope_ok sd s0 (SZ.v ob) (SZ.v vr) (hn_m s sw sc ss (SZ.v base) (SZ.v woff)) (hnorm s sw (SZ.v base) (SZ.v woff)))
      decreases (32 - SZ.v !r)
    {
      let vr = !r;
      with scur. assert (dst |-> scur);
      rope_read scur s0 (SZ.v ob) (SZ.v vr) (hn_m s sw sc ss (SZ.v base) (SZ.v woff)) (hnorm s sw (SZ.v base) (SZ.v woff));
      let a = fb dst.(ob +^ vr);
      let b = fb dst.(ob +^ 32sz +^ vr);
      let c = fb cs.(vr);
      let sv = fb sn.(vr);
      let ac = fb (tb (mul a c));
      let bs = fb (tb (mul (FB.neg b) sv));
      let bc = fb (tb (mul b c));
      let as_ = fb (tb (mul a sv));
      let va = tb (add ac bs);
      let vb = tb (add bc as_);
      assert pure (va == hn_m s sw sc ss (SZ.v base) (SZ.v woff) (SZ.v vr));
      assert pure (vb == hn_m s sw sc ss (SZ.v base) (SZ.v woff) (32 + SZ.v vr));
      rope_step scur s0 (SZ.v ob) (SZ.v vr) (hn_m s sw sc ss (SZ.v base) (SZ.v woff)) (hnorm s sw (SZ.v base) (SZ.v woff)) va vb;
      dst.(ob +^ vr) <- va;
      dst.(ob +^ 32sz +^ vr) <- vb;
      r := vr +^ 1sz;
    };
    with vr sd. assert (r |-> vr ** dst |-> sd);
    hn_step s0 sd (SZ.v vh) (SZ.v ob) (SZ.v base) (SZ.v vr) s sw sc ss (SZ.v stride) (SZ.v woff);
    h := vh +^ 1sz;
  };
}
#pop-options

let tsum_ext (cnt:nat) (f g:nat -> f32)
  : Lemma (requires forall (i:nat). i < cnt ==> f i == g i) (ensures tsum cnt f == tsum cnt g)
  = S.lemma_eq_intro (S.init cnt (fun (i:nat{i < cnt}) -> f i)) (S.init cnt (fun (i:nat{i < cnt}) -> g i))

let rec fsum_ext (n:nat) (f g:nat -> f32)
  : Lemma (requires forall (i:nat). i < n ==> f i == g i) (ensures fsum n f == fsum n g)
  = if n = 0 then () else fsum_ext (n - 1) f g

noextract
let dotj (qn kvo:S.seq bf16) (c h p:int) (g:nat) (j:nat) : f32 =
  mul (fb (bat qn (h * 256 + 16 * g + j))) (fb (bat kvo (((h / 4) * c + p) * 256 + 16 * g + j)))

let dot16_eq (qn kvo:S.seq bf16) (c h p:int) (g:nat) : Lemma (dot16 qn kvo c h p g == fsum 16 (dotj qn kvo c h p g))
  = fsum_ext 16 (fun j -> mul (fb (bat qn (h * 256 + 16 * g + j))) (fb (bat kvo (((h / 4) * c + p) * 256 + 16 * g + j)))) (dotj qn kvo c h p g)

noextract
let sc_ok (sd qn kvo:S.seq bf16) (c t vh vp:nat) : prop =
  S.length sd == 128 /\
  (forall (h p:nat). h < 8 /\ p < 16 /\ ((h < vh /\ p <= t) \/ (h == vh /\ p < vp)) ==> S.index sd (h * 16 + p) == score_m qn kvo c h p)

let sc_step (sd qn kvo:S.seq bf16) (c t vh vp:nat) (v:bf16)
  : Lemma (requires sc_ok sd qn kvo c t vh vp /\ vh < 8 /\ vp <= t /\ t < 16 /\ v == score_m qn kvo c vh vp)
          (ensures sc_ok (S.upd sd (vh * 16 + vp) v) qn kvo c t vh (vp + 1)) = ()

let sc_next (sd qn kvo:S.seq bf16) (c t vh:nat)
  : Lemma (requires sc_ok sd qn kvo c t vh (t + 1)) (ensures sc_ok sd qn kvo c t (vh + 1) 0) = ()

#push-options "--z3rlimit 40"
(* scores (t <> 7): dots[h*16+p] = bf16(csum(q*key, 256, 16)) *)
inline_for_extraction noextract
fn score_k (nkv c t : sz) (#_ : squash (SZ.v nkv == SZ.v c * 1024 /\ 1 <= SZ.v c /\ SZ.v c <= 12 /\ SZ.v t < SZ.v c))
  (qn : larray bf16 2048) (kvo : larray bf16 (SZ.v nkv)) (tmp : larray f32 1024) (dots : larray bf16 128)
  (#p1 #p2 : perm) (#s1 #s2 : erased (S.seq bf16))
  (#o_tmp : erased (S.seq f32)) (#o_dots : erased (S.seq bf16))
  preserves gpu ** qn |-> Frac p1 s1 ** kvo |-> Frac p2 s2
  requires tmp |-> o_tmp ** dots |-> o_dots
  ensures (exists* (s:S.seq f32). tmp |-> s) ** (exists* (sd:S.seq bf16). dots |-> sd ** pure (sc_ok sd s1 s2 (SZ.v c) (SZ.v t) 8 0))
{
  pts_to_len qn; pts_to_len kvo; pts_to_len tmp; pts_to_len dots;
  let mut h : sz = 0sz;
  while (!h <^ 8sz)
    invariant exists* vh st sd. h |-> vh ** tmp |-> st ** dots |-> sd ** pure (SZ.v vh <= 8 /\ S.length st == 1024 /\
      sc_ok sd s1 s2 (SZ.v c) (SZ.v t) (SZ.v vh) 0)
    decreases (8 - SZ.v !h)
  {
    let vh = !h;
    let mut p : sz = 0sz;
    while (!p <=^ t)
      invariant exists* vp st sd. p |-> vp ** tmp |-> st ** dots |-> sd ** pure (SZ.v vp <= SZ.v t + 1 /\ S.length st == 1024 /\
        sc_ok sd s1 s2 (SZ.v c) (SZ.v t) (SZ.v vh) (SZ.v vp))
      decreases (SZ.v t + 1 - SZ.v !p)
    {
      let vp = !p;
      let kb = ((vh /^ 4sz) *^ c +^ vp) *^ 256sz;
      let mut g : sz = 0sz;
      while (!g <^ 16sz)
        invariant exists* vg st. g |-> vg ** tmp |-> st ** pure (SZ.v vg <= 16 /\ S.length st == 1024 /\
          (forall (g':nat). g' < SZ.v vg ==> S.index st g' == dot16 s1 s2 (SZ.v c) (SZ.v vh) (SZ.v vp) g'))
        decreases (16 - SZ.v !g)
      {
        let vg = !g;
        let mut acc : f32 = zero;
        let mut j : sz = 0sz;
        while (!j <^ 16sz)
          invariant exists* vj va. j |-> vj ** acc |-> va ** pure (SZ.v vj <= 16 /\
            va == fsum (SZ.v vj) (dotj s1 s2 (SZ.v c) (SZ.v vh) (SZ.v vp) (SZ.v vg)))
          decreases (16 - SZ.v !j)
        {
          let vj = !j;
          let ii = 16sz *^ vg +^ vj;
          let q = fb qn.(vh *^ 256sz +^ ii);
          let kk = fb kvo.(kb +^ ii);
          let a = !acc;
          fsum_step (SZ.v vj) (dotj s1 s2 (SZ.v c) (SZ.v vh) (SZ.v vp) (SZ.v vg));
          acc := add a (mul q kk);
          j := vj +^ 1sz;
        };
        let a = !acc;
        dot16_eq s1 s2 (SZ.v c) (SZ.v vh) (SZ.v vp) (SZ.v vg);
        with st. assert (tmp |-> st);
        upd_step st (SZ.v vg) a (dot16 s1 s2 (SZ.v c) (SZ.v vh) (SZ.v vp));
        tmp.(vg) <- a;
        g := vg +^ 1sz;
      };
      with st. assert (tmp |-> st);
      tree_k tmp 16sz;
      tree_prefix st 16 (dot16 s1 s2 (SZ.v c) (SZ.v vh) (SZ.v vp));
      let d = tmp.(0sz);
      let v = tb d;
      with sd. assert (dots |-> sd);
      sc_step sd s1 s2 (SZ.v c) (SZ.v t) (SZ.v vh) (SZ.v vp) v;
      dots.(vh *^ 16sz +^ vp) <- v;
      p := vp +^ 1sz;
    };
    with sd. assert (dots |-> sd);
    sc_next sd s1 s2 (SZ.v c) (SZ.v t) (SZ.v vh);
    h := vh +^ 1sz;
  };
}
#pop-options

noextract
let smf (dots:S.seq bf16) (t:nat) (h:int) (p:nat) : f32 = if p <= t then sm_e dots t h p else zero

let sm_sum_eq (dots:S.seq bf16) (t:nat) (h:int) : Lemma (sm_sum dots t h == tsum (sm_width t) (smf dots t h))
  = tsum_ext (sm_width t) (fun p -> if p <= t then sm_e dots t h p else zero) (smf dots t h)

let rec tree_frame (s:S.seq f32) (off:nat) (j:nat)
  : Lemma (requires off <= j /\ j < S.length s) (ensures S.length (tree s off) == S.length s /\ S.index (tree s off) j == S.index s j)
          (decreases off)
  = if off = 0 then () else tree_frame (halve s off) (off / 2) j

let rec tree_len (s:S.seq f32) (off:nat) : Lemma (ensures S.length (tree s off) == S.length s) (decreases off)
  = if off = 0 then () else tree_len (halve s off) (off / 2)

let tree_frame_all (s:S.seq f32) (off lo:nat)
  : Lemma (requires off <= lo /\ lo <= S.length s)
          (ensures S.length (tree s off) == S.length s /\ (forall (j:nat). lo <= j /\ j < S.length s ==> S.index (tree s off) j == S.index s j))
  = let aux (j:nat) : Lemma (lo <= j /\ j < S.length s ==> S.length (tree s off) == S.length s /\ S.index (tree s off) j == S.index s j) =
      if lo <= j && j < S.length s then tree_frame s off j else ()
    in
    Classical.forall_intro aux;
    if lo < S.length s then tree_frame s off lo else (if off < S.length s then tree_frame s off off else tree_len s off)

noextract
let ex_ok (st:S.seq f32) (dots:S.seq bf16) (t vh vp:nat) : prop =
  S.length st == 1024 /\
  (forall (p:nat). p < vp /\ p < 1024 ==> S.index st p == smf dots t vh p) /\
  (forall (p:nat). p <= t /\ p < 16 ==> S.index st (16 + p) == (if p < vp then sm_e dots t vh p else sm_s dots vh p))

let ex_step (st:S.seq f32) (dots:S.seq bf16) (t vh vp:nat) (sv:f32)
  : Lemma (requires ex_ok st dots t vh vp /\ vp < 16 /\ t < 16 /\ sv == S.index st (16 + vp))
          (ensures (let e = FB.fexp (FB.sub sv (sm_max dots vh t)) in
                    ex_ok (S.upd (S.upd st (16 + vp) (if vp <= t then e else sv)) vp (if vp <= t then e else zero)) dots t vh (vp + 1)))
  = let e = FB.fexp (FB.sub sv (sm_max dots vh t)) in
    let s1 = S.upd st (16 + vp) (if vp <= t then e else sv) in
    let s2 = S.upd s1 vp (if vp <= t then e else zero) in
    if vp <= t then assert (sv == sm_s dots vh vp /\ e == sm_e dots t vh vp);
    let a1 (p:nat) : Lemma (p < vp + 1 /\ p < 1024 ==> S.index s2 p == smf dots t vh p) =
      if p < vp then (assert (S.index s2 p == S.index s1 p); assert (S.index s1 p == S.index st p)) else ()
    in
    let a2 (p:nat) : Lemma (p <= t /\ p < 16 ==> S.index s2 (16 + p) == (if p < vp + 1 then sm_e dots t vh p else sm_s dots vh p)) =
      if p <= t && p < 16 then begin
        assert (S.index s2 (16 + p) == S.index s1 (16 + p));
        if p = vp then () else assert (S.index s1 (16 + p) == S.index st (16 + p))
      end else ()
    in
    Classical.forall_intro a1;
    Classical.forall_intro a2

noextract
let pm_ok (sd dots:S.seq bf16) (t vh vp:nat) : prop =
  S.length sd == 128 /\
  (forall (h p:nat). h < 8 /\ p < 16 /\ (h < vh \/ (h == vh /\ p < vp)) ==> S.index sd (h * 16 + p) == prob_m dots t h p)

let pm_step (sd dots:S.seq bf16) (t vh vp:nat) (v:bf16)
  : Lemma (requires pm_ok sd dots t vh vp /\ vh < 8 /\ vp < 16 /\ v == prob_m dots t vh vp)
          (ensures pm_ok (S.upd sd (vh * 16 + vp) v) dots t vh (vp + 1)) = ()

let pm_next (sd dots:S.seq bf16) (t vh:nat)
  : Lemma (requires pm_ok sd dots t vh 16) (ensures pm_ok sd dots t (vh + 1) 0) = ()

let sm_max_step (dots:S.seq bf16) (h:int) (n:pos)
  : Lemma (sm_max dots h n == (if lt (sm_max dots h (n - 1)) (sm_s dots h n) then sm_s dots h n else sm_max dots h (n - 1))) = ()

#push-options "--z3rlimit 40"
(* softmax over p <= t; probs[h*16+p], zero for p > t *)
inline_for_extraction noextract
fn softmax_k (t : sz { SZ.v t < 12 }) (dots : larray bf16 128) (tmp : larray f32 1024) (probs : larray bf16 128)
  (#p1 : perm) (#s1 : erased (S.seq bf16))
  (#o_tmp : erased (S.seq f32)) (#o_probs : erased (S.seq bf16))
  preserves gpu ** dots |-> Frac p1 s1
  requires tmp |-> o_tmp ** probs |-> o_probs
  ensures (exists* (s:S.seq f32). tmp |-> s) ** (exists* (sd:S.seq bf16). probs |-> sd ** pure (pm_ok sd s1 (SZ.v t) 8 0))
{
  pts_to_len dots; pts_to_len tmp; pts_to_len probs;
  let mut wd : sz = 1sz;
  while (!wd <=^ t)
    invariant exists* vw. wd |-> vw ** pure (SZ.v vw == 1 \/ (SZ.v vw == 2 /\ SZ.v t >= 1) \/ (SZ.v vw == 4 /\ SZ.v t >= 2) \/
      (SZ.v vw == 8 /\ SZ.v t >= 4) \/ (SZ.v vw == 16 /\ SZ.v t >= 8))
    decreases (16 - SZ.v !wd)
  {
    let v = !wd;
    wd := v *^ 2sz;
  };
  let width = !wd;
  assert pure (SZ.v width == sm_width (SZ.v t));
  let mut h : sz = 0sz;
  while (!h <^ 8sz)
    invariant exists* vh st sd. h |-> vh ** tmp |-> st ** probs |-> sd ** pure (SZ.v vh <= 8 /\ S.length st == 1024 /\
      pm_ok sd s1 (SZ.v t) (SZ.v vh) 0)
    decreases (8 - SZ.v !h)
  {
    let vh = !h;
    with st0. assert (tmp |-> st0);
    let mut p : sz = 0sz;
    while (!p <=^ t)
      invariant exists* vp st. p |-> vp ** tmp |-> st ** pure (SZ.v vp <= SZ.v t + 1 /\ S.length st == 1024 /\
        rng_ok st st0 16 (SZ.v vp) (sm_s s1 (SZ.v vh)))
      decreases (SZ.v t + 1 - SZ.v !p)
    {
      let vp = !p;
      let d = fb dots.(vh *^ 16sz +^ vp);
      let v = fb (tb (mul d f_sixteenth));
      with sc. assert (tmp |-> sc);
      rng_step sc st0 16 (SZ.v vp) v (sm_s s1 (SZ.v vh));
      tmp.(16sz +^ vp) <- v;
      p := vp +^ 1sz;
    };
    let s0 = tmp.(16sz);
    let mut mx : f32 = s0;
    let mut p2 : sz = 1sz;
    while (!p2 <=^ t)
      invariant exists* vp vm. p2 |-> vp ** mx |-> vm ** pure (1 <= SZ.v vp /\ SZ.v vp <= SZ.v t + 1 /\ vm == sm_max s1 (SZ.v vh) (SZ.v vp - 1))
      decreases (SZ.v t + 1 - SZ.v !p2)
    {
      let vp = !p2;
      let sv = tmp.(16sz +^ vp);
      let m = !mx;
      sm_max_step s1 (SZ.v vh) (SZ.v vp);
      mx := (if lt m sv then sv else m);
      p2 := vp +^ 1sz;
    };
    let m = !mx;
    with st1. assert (tmp |-> st1);
    assert pure (ex_ok st1 s1 (SZ.v t) (SZ.v vh) 0);
    let mut p3 : sz = 0sz;
    while (!p3 <^ width)
      invariant exists* vp st. p3 |-> vp ** tmp |-> st ** pure (SZ.v vp <= SZ.v width /\ ex_ok st s1 (SZ.v t) (SZ.v vh) (SZ.v vp))
      decreases (SZ.v width - SZ.v !p3)
    {
      let vp = !p3;
      with sc. assert (tmp |-> sc);
      let sv = tmp.(16sz +^ vp);
      let e = FB.fexp (FB.sub sv m);
      let le = (vp <=^ t);
      let z : f32 = zero;
      ex_step sc s1 (SZ.v t) (SZ.v vh) (SZ.v vp) sv;
      tmp.(16sz +^ vp) <- (if le then e else sv);
      tmp.(vp) <- (if le then e else z);
      p3 := vp +^ 1sz;
    };
    with st2. assert (tmp |-> st2);
    tree_k tmp width;
    tree_prefix st2 (SZ.v width) (smf s1 (SZ.v t) (SZ.v vh));
    tree_frame_all st2 (SZ.v width / 2) 16;
    sm_sum_eq s1 (SZ.v t) (SZ.v vh);
    let sum = tmp.(0sz);
    assert pure (sum == sm_sum s1 (SZ.v t) (SZ.v vh));
    let mut p4 : sz = 0sz;
    while (!p4 <^ 16sz)
      invariant exists* vp sd. p4 |-> vp ** probs |-> sd ** pure (SZ.v vp <= 16 /\ pm_ok sd s1 (SZ.v t) (SZ.v vh) (SZ.v vp))
      decreases (16 - SZ.v !p4)
    {
      let vp = !p4;
      let e = tmp.(16sz +^ vp);
      let r = tb (FB.div e sum);
      let z : bf16 = zero;
      let v = (if vp <=^ t then r else z);
      with sc. assert (probs |-> sc);
      pm_step sc s1 (SZ.v t) (SZ.v vh) (SZ.v vp) v;
      probs.(vh *^ 16sz +^ vp) <- v;
      p4 := vp +^ 1sz;
    };
    with sd. assert (probs |-> sd);
    pm_next sd s1 (SZ.v t) (SZ.v vh);
    h := vh +^ 1sz;
  };
}
#pop-options

noextract
let vhyp (probs kvo:S.seq bf16) (c t h i:nat) (rp rv:nat -> real) : prop =
  forall (p:nat). p <= t ==> pr probs h p %~ rp p /\ vv kvo c h p i %~ rv p

noextract
let val_at (probs kvo:S.seq bf16) (c t h i:nat) (x:f32) : prop =
  forall (rp rv:nat -> real). vhyp probs kvo c t h i rp rv ==> x %~ vsum (t + 1) (fun p -> rp p *. rv p)

noextract
let grp_ok (probs kvo:S.seq bf16) (c t h i n:nat) (st:S.seq f32) : prop =
  S.length st == 1024 /\
  (forall (rp rv:nat -> real). vhyp probs kvo c t h i rp rv ==>
     (forall (g:nat). g < 4 ==> S.index st g %~ rgrp rp rv g n))

let grp_init (probs kvo:S.seq bf16) (c t h i:nat) (st:S.seq f32)
  : Lemma (requires S.length st == 1024 /\ S.index st 0 == zero /\ S.index st 1 == zero /\ S.index st 2 == zero /\ S.index st 3 == zero)
          (ensures grp_ok probs kvo c t h i 0 st)
  = su_zero_f32 ()

let grp_step (probs kvo:S.seq bf16) (c t h i n:nat) (st:S.seq f32) (r:f32)
  : Lemma (requires grp_ok probs kvo c t h i n st /\ n <= t /\ t < 16 /\
            (forall (xr yr zr:real). pr probs h n %~ xr /\ vv kvo c h n i %~ yr /\ S.index st (n / 4) %~ zr ==> r %~ (xr *. yr +. zr)))
          (ensures grp_ok probs kvo c t h i (n + 1) (S.upd st (n / 4) r))
  = let aux (rp rv:nat -> real) : Lemma (vhyp probs kvo c t h i rp rv ==>
        (forall (g:nat). g < 4 ==> S.index (S.upd st (n / 4) r) g %~ rgrp rp rv g (n + 1))) = ()
    in
    Classical.forall_intro_2 aux

let val_fin (probs kvo:S.seq bf16) (c t h i:nat) (st:S.seq f32)
  : Lemma (requires grp_ok probs kvo c t h i (t + 1) st /\ t < 16)
          (ensures val_at probs kvo c t h i
            (if t = 0 then mul (pr probs h 0) (vv kvo c h 0 i)
             else add (add (S.index st 0) (S.index st 2)) (add (S.index st 1) (S.index st 3))))
  = let x = (if t = 0 then mul (pr probs h 0) (vv kvo c h 0 i)
             else add (add (S.index st 0) (S.index st 2)) (add (S.index st 1) (S.index st 3))) in
    let aux (rp rv:nat -> real) : Lemma (vhyp probs kvo c t h i rp rv ==> x %~ vsum (t + 1) (fun p -> rp p *. rv p)) =
      if t = 0 then begin
        rvals_one rp rv;
        introduce vhyp probs kvo c t h i rp rv ==> x %~ vsum (t + 1) (fun p -> rp p *. rv p)
        with a_mul (pr probs h 0) (vv kvo c h 0 i) (rp 0) (rv 0)
      end else begin
        rgrp_sum rp rv t;
        introduce vhyp probs kvo c t h i rp rv ==> x %~ vsum (t + 1) (fun p -> rp p *. rv p)
        with begin
          let r0 = rgrp rp rv 0 (t + 1) in let r1 = rgrp rp rv 1 (t + 1) in
          let r2 = rgrp rp rv 2 (t + 1) in let r3 = rgrp rp rv 3 (t + 1) in
          a_add (S.index st 0) (S.index st 2) r0 r2;
          a_add (S.index st 1) (S.index st 3) r1 r3;
          a_add (add (S.index st 0) (S.index st 2)) (add (S.index st 1) (S.index st 3)) (r0 +. r2) (r1 +. r3)
        end
      end
    in
    Classical.forall_intro_2 aux

noextract
let va_ok (sd:S.seq f32) (probs kvo:S.seq bf16) (c t vh vi:nat) : prop =
  S.length sd == 2048 /\
  (forall (h i:nat). h < 8 /\ i < 256 /\ (h < vh \/ (h == vh /\ i < vi)) ==> val_at probs kvo c t h i (S.index sd (h * 256 + i)))

let va_step (sd:S.seq f32) (probs kvo:S.seq bf16) (c t vh vi:nat) (x:f32)
  : Lemma (requires va_ok sd probs kvo c t vh vi /\ vh < 8 /\ vi < 256 /\ val_at probs kvo c t vh vi x)
          (ensures va_ok (S.upd sd (vh * 256 + vi) x) probs kvo c t vh (vi + 1)) = ()

let va_next (sd:S.seq f32) (probs kvo:S.seq bf16) (c t vh:nat)
  : Lemma (requires va_ok sd probs kvo c t vh 256) (ensures va_ok sd probs kvo c t (vh + 1) 0) = ()

let va_final (sd:S.seq f32) (probs kvo:S.seq bf16) (c t:nat)
  : Lemma (requires va_ok sd probs kvo c t 8 0) (ensures vals_ok probs kvo c t sd) = ()

#push-options "--z3rlimit 40"
(* attention values (t <> 7) into vals (f32) *)
inline_for_extraction noextract
fn vals_k (nkv c t : sz) (#_ : squash (SZ.v nkv == SZ.v c * 1024 /\ 1 <= SZ.v c /\ SZ.v c <= 12 /\ SZ.v t < SZ.v c))
  (probs : larray bf16 128) (kvo : larray bf16 (SZ.v nkv)) (tmp : larray f32 1024) (vals : larray f32 2048)
  (#p1 #p2 : perm) (#s1 #s2 : erased (S.seq bf16))
  (#o_tmp : erased (S.seq f32)) (#o_vals : erased (S.seq f32))
  preserves gpu ** probs |-> Frac p1 s1 ** kvo |-> Frac p2 s2
  requires tmp |-> o_tmp ** vals |-> o_vals
  ensures (exists* (s:S.seq f32). tmp |-> s) ** (exists* (s:S.seq f32). vals |-> s ** pure (vals_ok s1 s2 (SZ.v c) (SZ.v t) s))
{
  pts_to_len probs; pts_to_len kvo; pts_to_len tmp; pts_to_len vals;
  let half = c *^ 512sz;
  let mut h : sz = 0sz;
  while (!h <^ 8sz)
    invariant exists* vh st sd. h |-> vh ** tmp |-> st ** vals |-> sd ** pure (SZ.v vh <= 8 /\ S.length st == 1024 /\
      va_ok sd s1 s2 (SZ.v c) (SZ.v t) (SZ.v vh) 0)
    decreases (8 - SZ.v !h)
  {
    let vh = !h;
    let mut i : sz = 0sz;
    while (!i <^ 256sz)
      invariant exists* vi st sd. i |-> vi ** tmp |-> st ** vals |-> sd ** pure (SZ.v vi <= 256 /\ S.length st == 1024 /\
        va_ok sd s1 s2 (SZ.v c) (SZ.v t) (SZ.v vh) (SZ.v vi))
      decreases (256 - SZ.v !i)
    {
      let vi = !i;
      tmp.(0sz) <- zero;
      tmp.(1sz) <- zero;
      tmp.(2sz) <- zero;
      tmp.(3sz) <- zero;
      with st0. assert (tmp |-> st0);
      grp_init s1 s2 (SZ.v c) (SZ.v t) (SZ.v vh) (SZ.v vi) st0;
      let mut p : sz = 0sz;
      while (!p <=^ t)
        invariant exists* vp st. p |-> vp ** tmp |-> st ** pure (SZ.v vp <= SZ.v t + 1 /\
          grp_ok s1 s2 (SZ.v c) (SZ.v t) (SZ.v vh) (SZ.v vi) (SZ.v vp) st)
        decreases (SZ.v t + 1 - SZ.v !p)
      {
        let vp = !p;
        let v = fb kvo.(half +^ ((vh /^ 4sz) *^ c +^ vp) *^ 256sz +^ vi);
        let pr = fb probs.(vh *^ 16sz +^ vp);
        let g = vp /^ 4sz;
        let old = tmp.(g);
        let r = FM.fma_rn pr v old;
        with sc. assert (tmp |-> sc);
        grp_step s1 s2 (SZ.v c) (SZ.v t) (SZ.v vh) (SZ.v vi) (SZ.v vp) sc r;
        tmp.(g) <- r;
        p := vp +^ 1sz;
      };
      let a0 = tmp.(0sz);
      let a1 = tmp.(1sz);
      let a2 = tmp.(2sz);
      let a3 = tmp.(3sz);
      let v0 = fb kvo.(half +^ ((vh /^ 4sz) *^ c) *^ 256sz +^ vi);
      let pr0 = fb probs.(vh *^ 16sz);
      let res : f32 = (if (t = 0sz) then mul pr0 v0 else add (add a0 a2) (add a1 a3));
      with st1. assert (tmp |-> st1);
      val_fin s1 s2 (SZ.v c) (SZ.v t) (SZ.v vh) (SZ.v vi) st1;
      with sd. assert (vals |-> sd);
      va_step sd s1 s2 (SZ.v c) (SZ.v t) (SZ.v vh) (SZ.v vi) res;
      vals.(vh *^ 256sz +^ vi) <- res;
      i := vi +^ 1sz;
    };
    with sd. assert (vals |-> sd);
    va_next sd s1 s2 (SZ.v c) (SZ.v t) (SZ.v vh);
    h := vh +^ 1sz;
  };
  with sd. assert (vals |-> sd);
  va_final sd s1 s2 (SZ.v c) (SZ.v t);
}
#pop-options
let mm_eq (#n:pos) (e1 e2:chest2 f32 n 8) : Lemma
  (requires 16 /? n /\ (forall (r:natlt (n / 16)). ematrix_subtile e1 16 8 r 0 == ematrix_subtile e2 16 8 r 0))
  (ensures e1 == e2) =
  introduce forall (i:natlt n) (j:natlt 8). acc2 e1 i j == acc2 e2 i j with begin
    subtile_acc2 e1 16 8 (i / 16) 0 (i % 16) j;
    subtile_acc2 e2 16 8 (i / 16) 0 (i % 16) j
  end;
  assert (equal e1 e2)

noextract
let mtc (#n #k:pos) (eA:chest2 bf16 n k) (eB:chest2 bf16 k 8) (#_:squash (16 /? n /\ 16 /? k))
  (r:nat) (j0:nat) (j:nat) : chest2 f32 16 8 =
  if r < n / 16 && j <= k / 16 then mma_tile eA eB r j0 j else const (16 @| 8 @| INil) zero

noextract
let mm_tiles_ok (#n #k:pos) (eA:chest2 bf16 n k) (eB:chest2 bf16 k 8) (#_:squash (16 /? n /\ 16 /? k))
  (j0:nat) (j1:nat{j1 <= k / 16}) (eC:chest2 f32 n 8) (r:nat) : prop =
  forall (r':natlt (n / 16)). r' < r ==> ematrix_subtile eC 16 8 r' 0 == mma_tile eA eB r' j0 j1

let mm_fin (#n #k:pos) (eA:chest2 bf16 n k) (eB:chest2 bf16 k 8) (#_:squash (16 /? n /\ 16 /? k))
  (j0:nat) (j1:nat{j1 <= k / 16}) (eC:chest2 f32 n 8) : Lemma
  (requires mm_tiles_ok eA eB j0 j1 eC (n / 16)) (ensures eC == mma_model eA eB j0 j1) =
  mm_eq eC (mma_model eA eB j0 j1)

let mm_step (#n #k:pos) (eA:chest2 bf16 n k) (eB:chest2 bf16 k 8) (#_:squash (16 /? n /\ 16 /? k))
  (j0:nat) (j1:nat{j1 <= k / 16}) (eC:chest2 f32 n 8) (r:natlt (n / 16)) : Lemma
  (requires mm_tiles_ok eA eB j0 j1 eC r)
  (ensures mm_tiles_ok eA eB j0 j1 (update_tile eC 16 8 r 0 (mma_tile eA eB r j0 j1)) (r + 1)) = ()

#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn mma_thread
  (n k : szp)
  (#_ : squash (16 /?+ n /\ 16 /?+ k /\ SZ.fits (n * k) /\ SZ.fits (k * 8) /\ SZ.fits (n * 8)))
  (gA : array2 bf16 (row_major n k))
  (gB : array2 bf16 (col_major k 8))
  (gC : array2 f32 (row_major n 8))
  (j0 j1 : sz)
  (#_ : squash (SZ.v j0 <= SZ.v j1 /\ SZ.v j1 <= k / 16))
  (#pa #pb : perm)
  (#eA : chest2 bf16 n k) (#eB : chest2 bf16 k 8)
  preserves gpu ** gA |-> Frac pa eA ** gB |-> Frac pb eB
  requires exists* eC. gC |-> Frac (1.0R /. 32.0R) eC
  ensures gC |-> Frac (1.0R /. 32.0R) (mma_model eA eB (SZ.v j0) (SZ.v j1))
{
  let fr = MM.alloc_fragment ();
  let mut r : sz = 0sz;
  while (!r <^ n /^ 16sz)
    invariant live r ** pure (SZ.v !r <= n / 16)
    invariant exists* v. fr |-> v
    invariant exists* eC. gC |-> Frac (1.0R /. 32.0R) eC ** pure (mm_tiles_ok eA eB (SZ.v j0) (SZ.v j1) eC (SZ.v !r))
    decreases (n / 16 - SZ.v !r)
  {
    let vr = !r;
    MM.fill fr zero;
    rewrite (fr |-> const (16 @| 8 @| INil) (zero #f32)) as (fr |-> mtc eA eB (SZ.v vr) (SZ.v j0) (SZ.v j0));
    let mut j : sz = j0;
    while (!j <^ j1)
      invariant live j ** pure (SZ.v j0 <= SZ.v !j /\ SZ.v !j <= SZ.v j1)
      invariant fr |-> mtc eA eB (SZ.v vr) (SZ.v j0) (SZ.v !j)
      decreases (SZ.v j1 - SZ.v !j)
    {
      let vj = !j;
      let at = array2_extract_tile_ro' gA 16 16 (SZ.v vr) (SZ.v vj);
      let bt = array2_extract_tile_ro' gB 16 8 (SZ.v vj) 0;
      MM.mma_sync at bt fr;
      rewrite (fr |-> MM.emma (mtc eA eB (SZ.v vr) (SZ.v j0) (SZ.v vj)) (ematrix_subtile eA 16 16 (SZ.v vr) (SZ.v vj)) (ematrix_subtile eB 16 8 (SZ.v vj) 0))
        as (fr |-> mtc eA eB (SZ.v vr) (SZ.v j0) (SZ.v vj + 1));
      with etA. assert (tensor_pts_to at #pa etA);
      Pulse.Lib.Trade.elim_trade (at |-> Frac pa etA) (gA |-> Frac pa eA);
      with etB. assert (tensor_pts_to bt #pb etB);
      Pulse.Lib.Trade.elim_trade (bt |-> Frac pb etB) (gB |-> Frac pb eB);
      j := vj +^ 1sz;
    };
    with eC0. assert (gC |-> Frac (1.0R /. 32.0R) eC0);
    mm_step eA eB (SZ.v j0) (SZ.v j1) eC0 (SZ.v vr);
    array2_extract_tile gC 16 8 (SZ.v vr) 0;
    let ct = array2_subtile gC 16 8 (SZ.v vr) 0;
    assert (rewrites_to ct (array2_subtile gC 16 8 (SZ.v vr) 0));
    MM.store fr ct;
    with x3. assert tensor_pts_to ct #(1.0R /. 32.0R) x3;
    Pulse.Lib.Forall.elim_forall x3;
    Pulse.Lib.Trade.elim_trade (tensor_pts_to ct #(1.0R /. 32.0R) x3) _;
    r := vr +^ 1sz;
  };
  with x. assert fr |-> x; drop_ (fr |-> x);
  with eCf. assert (gC |-> Frac (1.0R /. 32.0R) eCf);
  mm_fin eA eB (SZ.v j0) (SZ.v j1) eCf;
  rewrite (gC |-> Frac (1.0R /. 32.0R) eCf) as (gC |-> Frac (1.0R /. 32.0R) (mma_model eA eB (SZ.v j0) (SZ.v j1)));
}
#pop-options

let mma_kpre (n k : nat) (gA : array2 bf16 (row_major n k)) (gB : array2 bf16 (col_major k 8))
  (gC : array2 f32 (row_major n 8)) (pa pb : perm) (eA : chest2 bf16 n k) (eB : chest2 bf16 k 8)
  (eC : chest2 f32 n 8) (tid : nat) : slprop =
  gA |-> Frac (pa /. 32) eA ** gB |-> Frac (pb /. 32) eB ** gC |-> Frac (1.0R /. 32) eC

let mma_kpost (n k : nat) (gA : array2 bf16 (row_major n k)) (gB : array2 bf16 (col_major k 8))
  (gC : array2 f32 (row_major n 8)) (pa pb : perm) (eA : chest2 bf16 n k) (eB : chest2 bf16 k 8) (j0 j1 : nat)
  (tid : nat) : slprop =
  gA |-> Frac (pa /. 32) eA ** gB |-> Frac (pb /. 32) eB **
  gC |-> Frac (1.0R /. 32) (mmr eA eB j0 j1)

ghost
fn mma_setup (n k : nat) (gA : array2 bf16 (row_major n k)) (gB : array2 bf16 (col_major k 8))
  (gC : array2 f32 (row_major n 8)) (pa pb : perm) (eA : chest2 bf16 n k) (eB : chest2 bf16 k 8)
  (eC : chest2 f32 n 8) ()
  norewrite
  requires gA |-> Frac pa eA ** gB |-> Frac pb eB ** gC |-> eC
  ensures (forall+ (tid : natlt 32). mma_kpre n k gA gB gC pa pb eA eB eC tid) ** emp
{
  tensor_share_n gA 32;
  tensor_share_n gB 32;
  tensor_share_n gC 32;
  forevery_zip3 #(natlt 32) (fun _ -> gA |-> Frac (pa /. 32) eA) (fun _ -> gB |-> Frac (pb /. 32) eB)
    (fun _ -> gC |-> Frac (1.0R /. 32) eC);
  forevery_ext #(natlt 32) _ (fun tid -> mma_kpre n k gA gB gC pa pb eA eB eC tid);
}

ghost
fn mma_teardown (n k : nat) (gA : array2 bf16 (row_major n k)) (gB : array2 bf16 (col_major k 8))
  (gC : array2 f32 (row_major n 8)) (pa pb : perm) (eA : chest2 bf16 n k) (eB : chest2 bf16 k 8) (j0 j1 : nat) ()
  norewrite
  requires (forall+ (tid : natlt 32). mma_kpost n k gA gB gC pa pb eA eB j0 j1 tid) ** emp
  ensures gA |-> Frac pa eA ** gB |-> Frac pb eB ** gC |-> mmr eA eB j0 j1
{
  forevery_ext #(natlt 32) (fun tid -> mma_kpost n k gA gB gC pa pb eA eB j0 j1 tid)
    (fun _ -> gA |-> Frac (pa /. 32) eA ** gB |-> Frac (pb /. 32) eB **
       gC |-> Frac (1.0R /. 32) (mmr eA eB j0 j1));
  forevery_unzip3 #(natlt 32) (fun _ -> gA |-> Frac (pa /. 32) eA) (fun _ -> gB |-> Frac (pb /. 32) eB)
    (fun _ -> gC |-> Frac (1.0R /. 32) (mmr eA eB j0 j1));
  tensor_gather_n gA 32;
  tensor_gather_n gB 32;
  tensor_gather_n gC 32;
}

inline_for_extraction noextract
fn mma_kf
  (n k : szp)
  (#_ : squash (16 /?+ n /\ 16 /?+ k /\ SZ.fits (n * k) /\ SZ.fits (k * 8) /\ SZ.fits (n * 8)))
  (gA : array2 bf16 (row_major n k))
  (gB : array2 bf16 (col_major k 8))
  (gC : array2 f32 (row_major n 8))
  (j0 j1 : sz)
  (#_ : squash (SZ.v j0 <= SZ.v j1 /\ SZ.v j1 <= k / 16))
  (pa pb : perm) (eA : chest2 bf16 n k) (eB : chest2 bf16 k 8) (eC : chest2 f32 n 8)
  (tid : szlt 32) ()
  norewrite
  requires gpu ** mma_kpre n k gA gB gC pa pb eA eB eC tid ** thread_id 32 tid
  ensures gpu ** mma_kpost n k gA gB gC pa pb eA eB (SZ.v j0) (SZ.v j1) tid ** thread_id 32 tid
{
  unfold mma_kpre;
  mma_thread n k gA gB gC j0 j1;
  rewrite (gC |-> Frac (1.0R /. 32.0R) (mma_model eA eB (SZ.v j0) (SZ.v j1)))
    as (gC |-> Frac (1.0R /. 32) (mmr eA eB (SZ.v j0) (SZ.v j1)));
  fold mma_kpost n k gA gB gC pa pb eA eB (SZ.v j0) (SZ.v j1) tid;
}

inline_for_extraction noextract
let mma_kdesc
  (n k : szp)
  (#_ : squash (16 /?+ n /\ 16 /?+ k /\ SZ.fits (n * k) /\ SZ.fits (k * 8) /\ SZ.fits (n * 8)))
  (gA : array2 bf16 (row_major n k) { is_global gA })
  (gB : array2 bf16 (col_major k 8) { is_global gB })
  (gC : array2 f32 (row_major n 8) { is_global gC })
  (j0 j1 : sz)
  (#_ : squash (SZ.v j0 <= SZ.v j1 /\ SZ.v j1 <= k / 16))
  (pa pb : perm) (eA : chest2 bf16 n k) (eB : chest2 bf16 k 8) (eC : chest2 f32 n 8)
  : kernel_desc_1_n (gA |-> Frac pa eA ** gB |-> Frac pb eB ** gC |-> eC)
                    (gA |-> Frac pa eA ** gB |-> Frac pb eB ** gC |-> mmr eA eB (SZ.v j0) (SZ.v j1))
= {
  nthr = 32sz;
  kpre = mma_kpre n k gA gB gC pa pb eA eB eC;
  kpost = mma_kpost n k gA gB gC pa pb eA eB (SZ.v j0) (SZ.v j1);
  f = mma_kf n k gA gB gC j0 j1 pa pb eA eB eC;
  frame = emp;
  block_setup = mma_setup n k gA gB gC pa pb eA eB eC;
  block_teardown = mma_teardown n k gA gB gC pa pb eA eB (SZ.v j0) (SZ.v j1);
  kpost_sendable = solve;
  kpre_sendable = solve;
  full_post_sendable = solve;
  full_pre_sendable = solve;
}

inline_for_extraction noextract
fn mma_launch
  (n k : szp)
  (#_ : squash (16 /?+ n /\ 16 /?+ k /\ SZ.fits (n * k) /\ SZ.fits (k * 8) /\ SZ.fits (n * 8)))
  (gA : array2 bf16 (row_major n k) { is_global gA })
  (gB : array2 bf16 (col_major k 8) { is_global gB })
  (gC : array2 f32 (row_major n 8) { is_global gC })
  (j0 j1 : sz)
  (#_ : squash (SZ.v j0 <= SZ.v j1 /\ SZ.v j1 <= k / 16))
  (#pa #pb : perm) (#eA : chest2 bf16 n k) (#eB : chest2 bf16 k 8) (#eC : chest2 f32 n 8)
  preserves cpu ** on gpu_loc (gA |-> Frac pa eA) ** on gpu_loc (gB |-> Frac pb eB)
  requires on gpu_loc (gC |-> eC)
  ensures on gpu_loc (gC |-> mmr eA eB (SZ.v j0) (SZ.v j1))
{
  launch_sync (mma_kdesc n k gA gB gC j0 j1 pa pb eA eB eC);
}

ghost
fn gpu_len (#a : Type0) (x : array a) (#f : perm) (#s : erased (S.seq a))
  preserves on gpu_loc (x |-> Frac f s)
  ensures pure (S.length s == Pulse.Lib.Array.length x)
{
  map_loc gpu_loc #(x |-> Frac f s) #((x |-> Frac f s) ** pure (S.length s == Pulse.Lib.Array.length x))
    fn _ { pts_to_len x };
  on_star_elim (x |-> Frac f s) (pure (S.length s == Pulse.Lib.Array.length x));
  on_pure_elim gpu_loc (S.length s == Pulse.Lib.Array.length x);
}

inline_for_extraction noextract
fn mma_arr
  (n k : szp)
  (#_ : squash (16 /?+ n /\ 16 /?+ k /\ SZ.fits (n * k) /\ SZ.fits (k * 8) /\ SZ.fits (n * 8)))
  (a : larray bf16 (n * k) { is_global a })
  (b : larray bf16 (k * 8) { is_global b })
  (c : larray f32 (n * 8) { is_global c })
  (j0 j1 : sz)
  (#_ : squash (SZ.v j0 <= SZ.v j1 /\ SZ.v j1 <= k / 16))
  (#pa #pb : perm) (#sa #sb #sc : erased (S.seq _))
  preserves cpu ** on gpu_loc (a |-> Frac pa sa) ** on gpu_loc (b |-> Frac pb sb)
  requires on gpu_loc (c |-> sc)
  ensures on gpu_loc (c |-> mmo n k sa sb (SZ.v j0) (SZ.v j1))
{
  gpu_len a; gpu_len b; gpu_len c;
  let la : (l : layout2 n k { is_full l }) = row_major n k;
  let lb : (l : layout2 k 8 { is_full l }) = col_major k 8;
  let lc : (l : layout2 n 8 { is_full l }) = row_major n 8;
  map_loc gpu_loc #(a |-> Frac pa sa) #(from_array la a |-> Frac pa (from_seq la sa))
    fn _ { pts_to_len a; tensor_abs' la a };
  map_loc gpu_loc #(b |-> Frac pb sb) #(from_array lb b |-> Frac pb (from_seq lb sb))
    fn _ { pts_to_len b; tensor_abs' lb b };
  map_loc gpu_loc #(c |-> sc) #(from_array lc c |-> from_seq lc sc)
    fn _ { pts_to_len c; tensor_abs' lc c };
  mma_launch n k (from_array la a) (from_array lb b) (from_array lc c) j0 j1;
  map_loc gpu_loc #(from_array la a |-> Frac pa (from_seq la sa)) #(a |-> Frac pa sa)
    fn _ { tensor_concr (from_array la a); rewrite (core (from_array la a) |-> Frac pa (to_seq la (from_seq la sa))) as (a |-> Frac pa sa) };
  map_loc gpu_loc #(from_array lb b |-> Frac pb (from_seq lb sb)) #(b |-> Frac pb sb)
    fn _ { tensor_concr (from_array lb b); rewrite (core (from_array lb b) |-> Frac pb (to_seq lb (from_seq lb sb))) as (b |-> Frac pb sb) };
  let ec : erased (chest2 f32 n 8) = mmr (from_seq la sa) (from_seq lb sb) (SZ.v j0) (SZ.v j1);
  rewrite on gpu_loc (from_array lc c |-> mmr (from_seq la sa) (from_seq lb sb) (SZ.v j0) (SZ.v j1))
    as on gpu_loc (from_array lc c |-> reveal ec);
  map_loc gpu_loc #(from_array lc c |-> reveal ec) #(c |-> to_seq lc ec)
    fn _ { tensor_concr (from_array lc c); rewrite (core (from_array lc c) |-> to_seq lc ec) as (c |-> to_seq lc ec) };
  rewrite on gpu_loc (c |-> to_seq lc ec)
    as on gpu_loc (c |-> mmo n k sa sb (SZ.v j0) (SZ.v j1));
}

(*** Host-side bridging lemmas: kernel pointwise posts -> pipeline seqs *)

let eq_wsl (gv v:S.seq bf16) (off n:nat) : Lemma
  (requires S.length gv == n /\ off + n <= S.length v)
  (ensures Kuiper.Seq.Common.seq_blit gv 0 v off n == wsl v off n)
  = assert (S.equal (Kuiper.Seq.Common.seq_blit gv 0 v off n) (wsl v off n))

let eq_rms (d src w:S.seq bf16) (woff:int) : Lemma
  (requires S.length d == 1024 /\ (forall (j:nat). j < 1024 ==> S.index d j == rms1024 src w woff j))
  (ensures d == rms_s src w woff)
  = assert (S.equal d (rms_s src w woff))

let eq_ubs (d src:S.seq bf16) (k:nat) (soff:int) : Lemma
  (requires S.length d == k * 8 /\ (forall (j:nat). j < k * 8 ==> S.index d j == bat src (soff + j % k)))
  (ensures d == ubs k src soff)
  = assert (S.equal d (ubs k src soff))

let eq_col0 (d:S.seq bf16) (n:nat) (c:S.seq f32) (o:S.seq bf16) : Lemma
  (requires S.length d == n /\
    (forall (j:nat). j < n ==> S.index d j == (if 0 <= j && j < 0 + n then tb (fat c ((j - 0) * 8)) else bat o j)))
  (ensures d == col0 n c)
  = assert (S.equal d (col0 n c))

let eq_hn (d:S.seq bf16) (heads stride:nat) (src w cs sn:S.seq bf16) (woff:int) : Lemma
  (requires S.length d == heads * 256 /\
    (forall (j:nat). j < heads * 256 ==> S.index d j == hn_m src w cs sn ((j / 256) * stride) woff (j % 256)))
  (ensures d == hn_s heads stride src w cs sn woff)
  = assert (S.equal d (hn_s heads stride src w cs sn woff))

let eq_kvw (d:S.seq bf16) (c t:int) (kv kn pv:S.seq bf16) : Lemma
  (requires S.length d == S.length kv /\ (forall (j:nat). j < S.length kv ==> S.index d j == kvw_m c t kv kn pv j))
  (ensures d == kvw_s c t kv kn pv)
  = assert (S.equal d (kvw_s c t kv kn pv))

let eq_keys (d kvo:S.seq bf16) (c h:int) : Lemma
  (requires S.length d == 4096 /\ (forall (j:nat). j < 4096 ==> S.index d j == keys_m kvo c h j))
  (ensures d == keys_s kvo c h)
  = assert (S.equal d (keys_s kvo c h))

let eq_valm (d kvo:S.seq bf16) (c h:int) : Lemma
  (requires S.length d == 4096 /\ (forall (j:nat). j < 4096 ==> S.index d j == valm_m kvo c h j))
  (ensures d == valm_s kvo c h)
  = assert (S.equal d (valm_s kvo c h))

let eq_probs (d dots:S.seq bf16) (t:nat) : Lemma
  (requires S.length d == 128 /\
    (forall (h p:nat). h < 8 /\ p < 16 /\ (h < 8 \/ (h == 8 /\ p < 0)) ==> S.index d (h * 16 + p) == prob_m dots t h p))
  (ensures d == probs_s dots t)
  = let aux (j:nat{j < 128}) : Lemma (S.index d j == S.index (probs_s dots t) j) =
      let h = j / 16 in let p = j % 16 in
      assert (j == h * 16 + p)
    in
    Classical.forall_intro aux;
    assert (S.equal d (probs_s dots t))

let eq_gated (d:S.seq bf16) (vals:S.seq f32) (pq:S.seq bf16) : Lemma
  (requires S.length d == 2048 /\ (forall (j:nat). j < 2048 ==> S.index d j == gate_m vals pq j))
  (ensures d == gated_s vals pq)
  = assert (S.equal d (gated_s vals pq))

let eq_resid (d a b:S.seq bf16) : Lemma
  (requires S.length d == 1024 /\ (forall (j:nat). j < 1024 ==> S.index d j == resid_m a b j))
  (ensures d == resid_s a b)
  = assert (S.equal d (resid_s a b))

let eq_act (d g u:S.seq bf16) : Lemma
  (requires S.length d == 3584 /\ (forall (j:nat). j < 3584 ==> S.index d j == act_m g u j))
  (ensures d == act_s g u)
  = assert (S.equal d (act_s g u))

let eq_col2 (d:S.seq bf16) (c0 c1:S.seq f32) : Lemma
  (requires S.length d == 1024 /\ (forall (j:nat). j < 1024 ==> S.index d j == tb (add (fat c0 (j * 8)) (fat c1 (j * 8)))))
  (ensures d == col2_s c0 c1)
  = assert (S.equal d (col2_s c0 c1))

(* t == 7 per-head loops *)
let h_div (j vh:nat) (k:pos) : Lemma (requires vh * k <= j /\ j < vh * k + k) (ensures j / k == vh /\ j % k == j - vh * k) =
  let a : nat = j - vh * k in
  FStar.Math.Lemmas.lemma_div_plus a vh k;
  FStar.Math.Lemmas.lemma_mod_plus a vh k;
  FStar.Math.Lemmas.small_div a k;
  FStar.Math.Lemmas.small_mod a k

noextract
let d7_ok (d qn kvo:S.seq bf16) (c:int) (vh:nat) : prop =
  S.length d == 128 /\ vh <= 8 /\ (forall (j:nat). j < vh * 16 ==> S.index d j == S.index (dots7_s qn kvo c) j)

#push-options "--z3rlimit 40"
let d7_step (d0 d qn kvo ka ubq:S.seq bf16) (csc:S.seq f32) (c:int) (vh:nat) : Lemma
  (requires vh < 8 /\ d7_ok d0 qn kvo c vh /\ ka == keys_s kvo c vh /\ ubq == ubs 256 qn (vh * 256) /\
            csc == mmo 16 256 ka ubq 0 16 /\ S.length d == 128 /\
            (forall (j:nat). j < 128 ==> S.index d j == (if vh * 16 <= j && j < vh * 16 + 16 then tb (fat csc ((j - vh * 16) * 8)) else bat d0 j)))
  (ensures d7_ok d qn kvo c (vh + 1))
  = let aux (j:nat) : Lemma (j < (vh + 1) * 16 ==> S.index d j == S.index (dots7_s qn kvo c) j) =
      if j < vh * 16 then () else if j < vh * 16 + 16 then h_div j vh 16 else ()
    in
    Classical.forall_intro aux

#pop-options

let d7_fin (d qn kvo:S.seq bf16) (c:int) : Lemma (requires d7_ok d qn kvo c 8) (ensures d == dots7_s qn kvo c)
  = assert (S.equal d (dots7_s qn kvo c))

noextract
let v7_ok (d:S.seq f32) (probs kvo:S.seq bf16) (c:int) (vh:nat) : prop =
  S.length d == 2048 /\ vh <= 8 /\ (forall (j:nat). j < vh * 256 ==> S.index d j == S.index (vals7_s probs kvo c) j)

#push-options "--z3rlimit 40"
let v7_step (d0 d:S.seq f32) (probs kvo va ubp:S.seq bf16) (cvl:S.seq f32) (c:int) (vh:nat) : Lemma
  (requires vh < 8 /\ v7_ok d0 probs kvo c vh /\ va == valm_s kvo c vh /\ ubp == ubs 16 probs (vh * 16) /\
            cvl == mmo 256 16 va ubp 0 1 /\ S.length d == 2048 /\
            (forall (j:nat). j < 2048 ==> S.index d j == (if vh * 256 <= j && j < vh * 256 + 256 then fat cvl ((j - vh * 256) * 8) else fat d0 j)))
  (ensures v7_ok d probs kvo c (vh + 1))
  = let aux (j:nat) : Lemma (j < (vh + 1) * 256 ==> S.index d j == S.index (vals7_s probs kvo c) j) =
      if j < vh * 256 then () else if j < vh * 256 + 256 then h_div j vh 256 else ()
    in
    Classical.forall_intro aux

#pop-options

let v7_fin (d:S.seq f32) (probs kvo:S.seq bf16) (c:int) : Lemma (requires v7_ok d probs kvo c 8) (ensures d == vals7_s probs kvo c)
  = assert (S.equal d (vals7_s probs kvo c))

let sc_good (d qn kvo:S.seq bf16) (c t:nat) : Lemma (requires t <> 7 /\ sc_ok d qn kvo c t 8 0) (ensures dots_good qn kvo c t d) = ()
#push-options "--z3rlimit 40"

let i64_to_sz (t : i64 { 0 <= I64.v t /\ I64.v t < 12 }) : (r : sz { SZ.v r == I64.v t }) =
  if t = 0L then 0sz else if t = 1L then 1sz else if t = 2L then 2sz else if t = 3L then 3sz
  else if t = 4L then 4sz else if t = 5L then 5sz else if t = 6L then 6sz else if t = 7L then 7sz
  else if t = 8L then 8sz else if t = 9L then 9sz else if t = 10L then 10sz else 11sz


inline_for_extraction noextract
fn galloc_bf (n : sz { SZ.v n > 0 })
  preserves cpu
  returns a : larray bf16 (SZ.v n)
  ensures (exists* (s : S.seq bf16). on gpu_loc (a |-> s)) ** pure (Pulse.Lib.Array.is_full_array a /\ is_global a)
{
  let a = gpu_array_alloc #bf16 n;
  a
}

inline_for_extraction noextract
fn galloc_f (n : sz { SZ.v n > 0 })
  preserves cpu
  returns a : larray f32 (SZ.v n)
  ensures (exists* (s : S.seq f32). on gpu_loc (a |-> s)) ** pure (Pulse.Lib.Array.is_full_array a /\ is_global a)
{
  let a = gpu_array_alloc #f32 n;
  a
}

inline_for_extraction noextract
fn gfree (#t : Type0) (a : array t)
  preserves cpu
  requires (exists* (s : S.seq t). on gpu_loc (a |-> s)) ** pure (Pulse.Lib.Array.is_full_array a)
  ensures emp
{
  gpu_array_free a;
  ()
}

inline_for_extraction noextract
fn load_w (hw : Pulse.Lib.Vec.vec bf16) (#sw : erased (S.seq bf16) { S.length sw == 18352640 })
  (off : sz) (n : sz { SZ.v n > 0 /\ SZ.v off + SZ.v n <= 18352640 })
  preserves cpu ** (hw |-> sw)
  returns a : larray bf16 (SZ.v n)
  ensures (exists* (s : S.seq bf16). on gpu_loc (a |-> s) ** pure (s == wsl sw (SZ.v off) (SZ.v n))) ** pure (Pulse.Lib.Array.is_full_array a /\ is_global a)
{
  let a = gpu_array_alloc #bf16 n;
  with gv. assert (on gpu_loc (a |-> gv));
  gpu_len a;
  memcpy_host_to_device' a 0sz #18352640 hw off n;
  eq_wsl gv sw (SZ.v off) (SZ.v n);
  a
}

(* launch wrappers with exact posts *)
inline_for_extraction noextract
fn l_rms (src : larray bf16 1024 { is_global src }) (w : larray bf16 18352640 { is_global w }) (woff : sz { SZ.v woff + 1024 <= 18352640 })
  (tmp : larray f32 1024 { is_global tmp }) (dst : larray bf16 1024 { is_global dst }) (#ps #pw : perm) (#ss #sw : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (src |-> Frac ps ss) ** on gpu_loc (w |-> Frac pw sw)
  requires (exists* (s : S.seq f32). on gpu_loc (tmp |-> s)) ** (exists* (s : S.seq bf16). on gpu_loc (dst |-> s))
  ensures (exists* (s : S.seq f32). on gpu_loc (tmp |-> s)) ** (exists* (s : S.seq bf16). on gpu_loc (dst |-> s) ** pure (s == rms_s ss sw (SZ.v woff)))
{
  with o1. assert (on gpu_loc (tmp |-> o1));
  with o2. assert (on gpu_loc (dst |-> o2));
  launch_kernel_1 (fun () -> rms_k src w woff tmp dst #ps #pw #ss #sw #o1 #o2);
  with d. assert (on gpu_loc (dst |-> d));
  eq_rms d ss sw (SZ.v woff);
  ()
}

inline_for_extraction noextract
fn l_resid (a : larray bf16 1024 { is_global a }) (b : larray bf16 1024 { is_global b }) (dst : larray bf16 1024 { is_global dst }) (#pa #pb : perm) (#sa #sb : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (a |-> Frac pa sa) ** on gpu_loc (b |-> Frac pb sb)
  requires exists* (s : S.seq bf16). on gpu_loc (dst |-> s)
  ensures exists* (s : S.seq bf16). on gpu_loc (dst |-> s) ** pure (s == resid_s sa sb)
{
  with o. assert (on gpu_loc (dst |-> o));
  launch_kernel_1 (fun () -> resid_k a b dst #pa #pb #sa #sb #o);
  with d. assert (on gpu_loc (dst |-> d));
  eq_resid d sa sb;
  ()
}

inline_for_extraction noextract
fn l_ub (#m : erased nat) (k : sz { SZ.v k > 0 /\ SZ.fits (SZ.v k * 8) }) (src : larray bf16 m { is_global src }) (soff : sz { SZ.v soff + SZ.v k <= m })
  (dst : larray bf16 (SZ.v k * 8) { is_global dst }) (#p : perm) (#s : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (src |-> Frac p s)
  requires exists* (s : S.seq bf16). on gpu_loc (dst |-> s)
  ensures exists* (s' : S.seq bf16). on gpu_loc (dst |-> s') ** pure (s' == ubs (SZ.v k) s (SZ.v soff))
{
  with o. assert (on gpu_loc (dst |-> o));
  launch_kernel_1 (fun () -> ub_k k src soff dst #p #s #o);
  with d. assert (on gpu_loc (dst |-> d));
  eq_ubs d s (SZ.v k) (SZ.v soff);
  ()
}

(* colx with the raw pointwise post (used per head) *)
inline_for_extraction noextract
fn l_colx (#m : erased nat) (n : sz { SZ.fits (SZ.v n * 8) }) (c : larray f32 (SZ.v n * 8) { is_global c })
  (dst : larray bf16 m { is_global dst }) (doff : sz { SZ.v doff + SZ.v n <= m }) (#p : perm) (#s : erased (S.seq f32))
  (#o : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (c |-> Frac p s)
  requires on gpu_loc (dst |-> o)
  ensures exists* (s' : S.seq bf16). on gpu_loc (dst |-> s') ** pure (S.length s' == m /\
    (forall (j:nat). j < m ==> S.index s' j == (if SZ.v doff <= j && j < SZ.v doff + SZ.v n then tb (fat s ((j - SZ.v doff) * 8)) else bat o j)))
{
  launch_kernel_1 (fun () -> colx_k n c dst doff #p #s #o);
  ()
}

(* full-buffer column extraction *)
inline_for_extraction noextract
fn l_col0 (n : sz { SZ.fits (SZ.v n * 8) }) (c : larray f32 (SZ.v n * 8) { is_global c })
  (dst : larray bf16 (SZ.v n) { is_global dst }) (#p : perm) (#s : erased (S.seq f32))
  preserves cpu ** on gpu_loc (c |-> Frac p s)
  requires exists* (s : S.seq bf16). on gpu_loc (dst |-> s)
  ensures exists* (s' : S.seq bf16). on gpu_loc (dst |-> s') ** pure (s' == col0 (SZ.v n) s)
{
  with o. assert (on gpu_loc (dst |-> o));
  l_colx #(SZ.v n) n c dst 0sz #p #s #o;
  with d. assert (on gpu_loc (dst |-> d));
  eq_col0 d (SZ.v n) s o;
  ()
}

inline_for_extraction noextract
fn l_colx2 (c0 : larray f32 8192 { is_global c0 }) (c1 : larray f32 8192 { is_global c1 }) (dst : larray bf16 1024 { is_global dst }) (#p0 #p1 : perm) (#s0 #s1 : erased (S.seq f32))
  preserves cpu ** on gpu_loc (c0 |-> Frac p0 s0) ** on gpu_loc (c1 |-> Frac p1 s1)
  requires exists* (s : S.seq bf16). on gpu_loc (dst |-> s)
  ensures exists* (s : S.seq bf16). on gpu_loc (dst |-> s) ** pure (s == col2_s s0 s1)
{
  with o. assert (on gpu_loc (dst |-> o));
  launch_kernel_1 (fun () -> colx2_k c0 c1 dst #p0 #p1 #s0 #s1 #o);
  with d. assert (on gpu_loc (dst |-> d));
  eq_col2 d s0 s1;
  ()
}

inline_for_extraction noextract
fn l_colf (c : larray f32 2048 { is_global c }) (dst : larray f32 2048 { is_global dst }) (doff : sz { SZ.v doff + 256 <= 2048 }) (#p : perm) (#s : erased (S.seq f32))
  (#o : erased (S.seq f32))
  preserves cpu ** on gpu_loc (c |-> Frac p s)
  requires on gpu_loc (dst |-> o)
  ensures exists* (s' : S.seq f32). on gpu_loc (dst |-> s') ** pure (S.length s' == 2048 /\
    (forall (j:nat). j < 2048 ==> S.index s' j == (if SZ.v doff <= j && j < SZ.v doff + 256 then fat s ((j - SZ.v doff) * 8) else fat o j)))
{
  launch_kernel_1 (fun () -> colf_k c dst doff #p #s #o);
  ()
}

inline_for_extraction noextract
fn l_hn (#m #m2 : erased nat) (heads stride : sz) (src : larray bf16 m { is_global src })
  (#_ : squash (SZ.v heads * SZ.v stride <= m /\ SZ.v heads * 256 == m2 /\ 256 <= SZ.v stride /\ SZ.fits m /\ SZ.fits m2))
  (w : larray bf16 18352640 { is_global w }) (woff : sz { SZ.v woff + 256 <= 18352640 })
  (cs : larray bf16 32 { is_global cs }) (sn : larray bf16 32 { is_global sn }) (tmp : larray f32 1024 { is_global tmp }) (dst : larray bf16 m2 { is_global dst })
  (#p #pw #pc #ps : perm) (#s #sw #sc #ss : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (src |-> Frac p s) ** on gpu_loc (w |-> Frac pw sw) ** on gpu_loc (cs |-> Frac pc sc) ** on gpu_loc (sn |-> Frac ps ss)
  requires (exists* (s : S.seq f32). on gpu_loc (tmp |-> s)) ** (exists* (s : S.seq bf16). on gpu_loc (dst |-> s))
  ensures (exists* (s : S.seq f32). on gpu_loc (tmp |-> s)) ** (exists* (d : S.seq bf16). on gpu_loc (dst |-> d) ** pure (d == hn_s (SZ.v heads) (SZ.v stride) s sw sc ss (SZ.v woff)))
{
  with o1. assert (on gpu_loc (tmp |-> o1));
  with o2. assert (on gpu_loc (dst |-> o2));
  launch_kernel_1 (fun () -> hn_k heads stride src w woff cs sn tmp dst #p #pw #pc #ps #s #sw #sc #ss #o1 #o2);
  with d. assert (on gpu_loc (dst |-> d));
  eq_hn d (SZ.v heads) (SZ.v stride) s sw sc ss (SZ.v woff);
  ()
}

inline_for_extraction noextract
fn l_kvw (nkv c t : sz) (#_ : squash (SZ.v nkv == SZ.v c * 1024 /\ 1 <= SZ.v c /\ SZ.v c <= 12 /\ SZ.v t < SZ.v c))
  (kv : larray bf16 (SZ.v nkv) { is_global kv }) (kn : larray bf16 512 { is_global kn }) (pv : larray bf16 512 { is_global pv }) (kvo : larray bf16 (SZ.v nkv) { is_global kvo })
  (#p1 #p2 #p3 : perm) (#s1 #s2 #s3 : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (kv |-> Frac p1 s1) ** on gpu_loc (kn |-> Frac p2 s2) ** on gpu_loc (pv |-> Frac p3 s3)
  requires exists* (s : S.seq bf16). on gpu_loc (kvo |-> s)
  ensures exists* (s : S.seq bf16). on gpu_loc (kvo |-> s) ** pure (s == kvw_s (SZ.v c) (SZ.v t) s1 s2 s3)
{
  gpu_len kv;
  with o. assert (on gpu_loc (kvo |-> o));
  launch_kernel_1 (fun () -> kvw_k nkv c t kv kn pv kvo #p1 #p2 #p3 #s1 #s2 #s3 #o);
  with d. assert (on gpu_loc (kvo |-> d));
  eq_kvw d (SZ.v c) (SZ.v t) s1 s2 s3;
  ()
}

inline_for_extraction noextract
fn l_score (nkv c t : sz) (#_ : squash (SZ.v nkv == SZ.v c * 1024 /\ 1 <= SZ.v c /\ SZ.v c <= 12 /\ SZ.v t < SZ.v c))
  (qn : larray bf16 2048 { is_global qn }) (kvo : larray bf16 (SZ.v nkv) { is_global kvo }) (tmp : larray f32 1024 { is_global tmp }) (dots : larray bf16 128 { is_global dots })
  (#p1 #p2 : perm) (#s1 #s2 : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (qn |-> Frac p1 s1) ** on gpu_loc (kvo |-> Frac p2 s2)
  requires (exists* (s : S.seq f32). on gpu_loc (tmp |-> s)) ** (exists* (s : S.seq bf16). on gpu_loc (dots |-> s))
  ensures (exists* (s : S.seq f32). on gpu_loc (tmp |-> s)) ** (exists* (s : S.seq bf16). on gpu_loc (dots |-> s) ** pure (sc_ok s s1 s2 (SZ.v c) (SZ.v t) 8 0))
{
  with o1. assert (on gpu_loc (tmp |-> o1));
  with o2. assert (on gpu_loc (dots |-> o2));
  launch_kernel_1 (fun () -> score_k nkv c t qn kvo tmp dots #p1 #p2 #s1 #s2 #o1 #o2);
  ()
}

inline_for_extraction noextract
fn l_softmax (t : sz { SZ.v t < 12 }) (dots : larray bf16 128 { is_global dots }) (tmp : larray f32 1024 { is_global tmp }) (probs : larray bf16 128 { is_global probs })
  (#p1 : perm) (#s1 : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (dots |-> Frac p1 s1)
  requires (exists* (s : S.seq f32). on gpu_loc (tmp |-> s)) ** (exists* (s : S.seq bf16). on gpu_loc (probs |-> s))
  ensures (exists* (s : S.seq f32). on gpu_loc (tmp |-> s)) ** (exists* (s : S.seq bf16). on gpu_loc (probs |-> s) ** pure (s == probs_s s1 (SZ.v t)))
{
  with o1. assert (on gpu_loc (tmp |-> o1));
  with o2. assert (on gpu_loc (probs |-> o2));
  launch_kernel_1 (fun () -> softmax_k t dots tmp probs #p1 #s1 #o1 #o2);
  with d. assert (on gpu_loc (probs |-> d));
  eq_probs d s1 (SZ.v t);
  ()
}

inline_for_extraction noextract
fn l_vals (nkv c t : sz) (#_ : squash (SZ.v nkv == SZ.v c * 1024 /\ 1 <= SZ.v c /\ SZ.v c <= 12 /\ SZ.v t < SZ.v c))
  (probs : larray bf16 128 { is_global probs }) (kvo : larray bf16 (SZ.v nkv) { is_global kvo }) (tmp : larray f32 1024 { is_global tmp }) (vals : larray f32 2048 { is_global vals })
  (#p1 #p2 : perm) (#s1 #s2 : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (probs |-> Frac p1 s1) ** on gpu_loc (kvo |-> Frac p2 s2)
  requires (exists* (s : S.seq f32). on gpu_loc (tmp |-> s)) ** (exists* (s : S.seq f32). on gpu_loc (vals |-> s))
  ensures (exists* (s : S.seq f32). on gpu_loc (tmp |-> s)) ** (exists* (s : S.seq f32). on gpu_loc (vals |-> s) ** pure (vals_ok s1 s2 (SZ.v c) (SZ.v t) s))
{
  with o1. assert (on gpu_loc (tmp |-> o1));
  with o2. assert (on gpu_loc (vals |-> o2));
  launch_kernel_1 (fun () -> vals_k nkv c t probs kvo tmp vals #p1 #p2 #s1 #s2 #o1 #o2);
  ()
}

inline_for_extraction noextract
fn l_gate (vals : larray f32 2048 { is_global vals }) (pq : larray bf16 4096 { is_global pq }) (gated : larray bf16 2048 { is_global gated })
  (#p1 #p2 : perm) (#s1 : erased (S.seq f32)) (#s2 : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (vals |-> Frac p1 s1) ** on gpu_loc (pq |-> Frac p2 s2)
  requires exists* (s : S.seq bf16). on gpu_loc (gated |-> s)
  ensures exists* (s : S.seq bf16). on gpu_loc (gated |-> s) ** pure (s == gated_s s1 s2)
{
  with o. assert (on gpu_loc (gated |-> o));
  launch_kernel_1 (fun () -> gate_k vals pq gated #p1 #p2 #s1 #s2 #o);
  with d. assert (on gpu_loc (gated |-> d));
  eq_gated d s1 s2;
  ()
}

inline_for_extraction noextract
fn l_act (g : larray bf16 3584 { is_global g }) (u : larray bf16 3584 { is_global u }) (dst : larray bf16 3584 { is_global dst }) (#p1 #p2 : perm) (#s1 #s2 : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (g |-> Frac p1 s1) ** on gpu_loc (u |-> Frac p2 s2)
  requires exists* (s : S.seq bf16). on gpu_loc (dst |-> s)
  ensures exists* (s : S.seq bf16). on gpu_loc (dst |-> s) ** pure (s == act_s s1 s2)
{
  with o. assert (on gpu_loc (dst |-> o));
  launch_kernel_1 (fun () -> act_k g u dst #p1 #p2 #s1 #s2 #o);
  with d. assert (on gpu_loc (dst |-> d));
  eq_act d s1 s2;
  ()
}

inline_for_extraction noextract
fn l_keys (nkv c h : sz) (#_ : squash (SZ.v nkv == SZ.v c * 1024 /\ 8 <= SZ.v c /\ SZ.v c <= 12 /\ SZ.v h < 8))
  (kvo : larray bf16 (SZ.v nkv) { is_global kvo }) (ka : larray bf16 4096 { is_global ka }) (#p : perm) (#s : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (kvo |-> Frac p s)
  requires exists* (s : S.seq bf16). on gpu_loc (ka |-> s)
  ensures exists* (s' : S.seq bf16). on gpu_loc (ka |-> s') ** pure (s' == keys_s s (SZ.v c) (SZ.v h))
{
  with o. assert (on gpu_loc (ka |-> o));
  launch_kernel_1 (fun () -> keys_k nkv c h kvo ka #p #s #o);
  with d. assert (on gpu_loc (ka |-> d));
  eq_keys d s (SZ.v c) (SZ.v h);
  ()
}

inline_for_extraction noextract
fn l_valm (nkv c h : sz) (#_ : squash (SZ.v nkv == SZ.v c * 1024 /\ 8 <= SZ.v c /\ SZ.v c <= 12 /\ SZ.v h < 8))
  (kvo : larray bf16 (SZ.v nkv) { is_global kvo }) (va : larray bf16 4096 { is_global va }) (#p : perm) (#s : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (kvo |-> Frac p s)
  requires exists* (s : S.seq bf16). on gpu_loc (va |-> s)
  ensures exists* (s' : S.seq bf16). on gpu_loc (va |-> s') ** pure (s' == valm_s s (SZ.v c) (SZ.v h))
{
  with o. assert (on gpu_loc (va |-> o));
  launch_kernel_1 (fun () -> valm_k nkv c h kvo va #p #s #o);
  with d. assert (on gpu_loc (va |-> d));
  eq_valm d s (SZ.v c) (SZ.v h);
  ()
}

inline_for_extraction noextract
fn l_mma (n k : szp)
  (#_ : squash (16 /?+ n /\ 16 /?+ k /\ SZ.fits (n * k) /\ SZ.fits (k * 8) /\ SZ.fits (n * 8)))
  (a : larray bf16 (n * k) { is_global a }) (b : larray bf16 (k * 8) { is_global b }) (c : larray f32 (n * 8) { is_global c })
  (j0 j1 : sz) (#_ : squash (SZ.v j0 <= SZ.v j1 /\ SZ.v j1 <= k / 16))
  (#pa #pb : perm) (#sa #sb : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (a |-> Frac pa sa) ** on gpu_loc (b |-> Frac pb sb)
  requires exists* (s : S.seq f32). on gpu_loc (c |-> s)
  ensures exists* (s : S.seq f32). on gpu_loc (c |-> s) ** pure (s == mmo n k sa sb (SZ.v j0) (SZ.v j1))
{
  with o. assert (on gpu_loc (c |-> o));
  mma_arr n k a b c j0 j1 #() #pa #pb #sa #sb #o;
  ()
}

(* t == 7 attention helpers: per-head tensor-core products *)
fn t7_scores (nkv c : sz) (#_ : squash (SZ.v nkv == SZ.v c * 1024 /\ 8 <= SZ.v c /\ SZ.v c <= 12))
  (kvo : larray bf16 (SZ.v nkv) { is_global kvo }) (qn : larray bf16 2048 { is_global qn }) (dots : larray bf16 128 { is_global dots })
  (#pk #pq : perm) (#skv #sq : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (kvo |-> Frac pk skv) ** on gpu_loc (qn |-> Frac pq sq)
  requires exists* (s : S.seq bf16). on gpu_loc (dots |-> s)
  ensures exists* (s : S.seq bf16). on gpu_loc (dots |-> s) ** pure (s == dots7_s sq skv (SZ.v c))
{
  gpu_len dots;
  let ka = galloc_bf 4096sz;
  let ubq = galloc_bf 2048sz;
  let csc = galloc_f 128sz;
  let mut h : sz = 0sz;
  while (!h <^ 8sz)
    invariant exists* (vh : sz). (h |-> vh) ** pure (SZ.v vh <= 8) **
      (exists* (s : S.seq bf16). on gpu_loc (ka |-> s)) ** (exists* (s : S.seq bf16). on gpu_loc (ubq |-> s)) **
      (exists* (s : S.seq f32). on gpu_loc (csc |-> s)) ** (exists* (s : S.seq bf16). on gpu_loc (dots |-> s) ** pure (d7_ok s sq skv (SZ.v c) (SZ.v vh)))
    decreases (8 - SZ.v !h)
  {
    let vh = !h;
    l_keys nkv c vh kvo ka;
    with ska. assert (on gpu_loc (ka |-> ska));
    l_ub 256sz qn (vh *^ 256sz) ubq;
    with subq. assert (on gpu_loc (ubq |-> subq));
    l_mma 16sz 256sz ka ubq csc 0sz 16sz;
    with scsc. assert (on gpu_loc (csc |-> scsc));
    with d0. assert (on gpu_loc (dots |-> d0));
    l_colx 16sz csc dots (vh *^ 16sz);
    with d1. assert (on gpu_loc (dots |-> d1));
    d7_step d0 d1 sq skv ska subq scsc (SZ.v c) (SZ.v vh);
    h := vh +^ 1sz;
    ()
  };
  with d. assert (on gpu_loc (dots |-> d));
  d7_fin d sq skv (SZ.v c);
  gfree ka; gfree ubq; gfree csc;
  ()
}

fn t7_values (nkv c : sz) (#_ : squash (SZ.v nkv == SZ.v c * 1024 /\ 8 <= SZ.v c /\ SZ.v c <= 12))
  (kvo : larray bf16 (SZ.v nkv) { is_global kvo }) (probs : larray bf16 128 { is_global probs }) (vals : larray f32 2048 { is_global vals })
  (#pk #pp : perm) (#skv #sp : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (kvo |-> Frac pk skv) ** on gpu_loc (probs |-> Frac pp sp)
  requires exists* (s : S.seq f32). on gpu_loc (vals |-> s)
  ensures exists* (s : S.seq f32). on gpu_loc (vals |-> s) ** pure (s == vals7_s sp skv (SZ.v c))
{
  gpu_len vals;
  let va = galloc_bf 4096sz;
  let ubp = galloc_bf 128sz;
  let cvl = galloc_f 2048sz;
  let mut h : sz = 0sz;
  while (!h <^ 8sz)
    invariant exists* (vh : sz). (h |-> vh) ** pure (SZ.v vh <= 8) **
      (exists* (s : S.seq bf16). on gpu_loc (va |-> s)) ** (exists* (s : S.seq bf16). on gpu_loc (ubp |-> s)) **
      (exists* (s : S.seq f32). on gpu_loc (cvl |-> s)) ** (exists* (s : S.seq f32). on gpu_loc (vals |-> s) ** pure (v7_ok s sp skv (SZ.v c) (SZ.v vh)))
    decreases (8 - SZ.v !h)
  {
    let vh = !h;
    l_valm nkv c vh kvo va;
    with sva. assert (on gpu_loc (va |-> sva));
    l_ub 16sz probs (vh *^ 16sz) ubp;
    with subp. assert (on gpu_loc (ubp |-> subp));
    l_mma 256sz 16sz va ubp cvl 0sz 1sz;
    with scvl. assert (on gpu_loc (cvl |-> scvl));
    with d0. assert (on gpu_loc (vals |-> d0));
    l_colf cvl vals (vh *^ 256sz);
    with d1. assert (on gpu_loc (vals |-> d1));
    v7_step d0 d1 sp skv sva subp scvl (SZ.v c) (SZ.v vh);
    h := vh +^ 1sz;
    ()
  };
  with d. assert (on gpu_loc (vals |-> d));
  v7_fin d sp skv (SZ.v c);
  gfree va; gfree ubp; gfree cvl;
  ()
}
#pop-options
#push-options "--z3rlimit 40"

inline_for_extraction noextract
let c_65536 : (r : sz { SZ.v r == 65536 }) = 64sz *^ 1024sz
inline_for_extraction noextract
let c_524288 : (r : sz { SZ.v r == 524288 }) = 512sz *^ 1024sz
inline_for_extraction noextract
let c_2097152 : (r : sz { SZ.v r == 2097152 }) = 2048sz *^ 1024sz
inline_for_extraction noextract
let c_3670016 : (r : sz { SZ.v r == 3670016 }) = 3584sz *^ 1024sz
inline_for_extraction noextract
let c_3672064 : (r : sz { SZ.v r == 3672064 }) = 3586sz *^ 1024sz
inline_for_extraction noextract
let c_4194304 : (r : sz { SZ.v r == 4194304 }) = 4096sz *^ 1024sz
inline_for_extraction noextract
let c_7342080 : (r : sz { SZ.v r == 7342080 }) = 7170sz *^ 1024sz
inline_for_extraction noextract
let c_11012096 : (r : sz { SZ.v r == 11012096 }) = 10754sz *^ 1024sz
inline_for_extraction noextract
let c_15206400 : (r : sz { SZ.v r == 15206400 }) = 14850sz *^ 1024sz
inline_for_extraction noextract
let c_15730688 : (r : sz { SZ.v r == 15730688 }) = 15362sz *^ 1024sz
inline_for_extraction noextract
let c_16254976 : (r : sz { SZ.v r == 16254976 }) = 15874sz *^ 1024sz
inline_for_extraction noextract
let c_16255232 : (r : sz { SZ.v r == 16255232 }) = 63497sz *^ 256sz
inline_for_extraction noextract
let c_16255488 : (r : sz { SZ.v r == 16255488 }) = 31749sz *^ 512sz
inline_for_extraction noextract
let c_18352640 : (r : sz { SZ.v r == 18352640 }) = 35845sz *^ 512sz


let chain_front (sw sx sc ss hin ub1 wq pq wk pk wv pv qn kn:S.seq bf16) (cq ck cv:S.seq f32) : Lemma
  (requires S.length sw == 18352640 /\ hin == rms_s sx sw 0 /\ ub1 == ubs 1024 hin 0 /\
     wq == wsl sw 11012096 4194304 /\ cq == mmo 4096 1024 wq ub1 0 64 /\ pq == col0 4096 cq /\
     wk == wsl sw 15206400 524288 /\ ck == mmo 512 1024 wk ub1 0 64 /\ pk == col0 512 ck /\
     wv == wsl sw 15730688 524288 /\ cv == mmo 512 1024 wv ub1 0 64 /\ pv == col0 512 cv /\
     qn == hn_s 8 512 pq sw sc ss 16254976 /\ kn == hn_s 2 256 pk sw sc ss 16255232)
  (ensures pq == p_pq sw sx /\ pv == p_pv sw sx /\ qn == p_qn sw sx sc ss /\ kn == p_kn sw sx sc ss)
  = ()

let chain_back (sw sx gated ubg wo po x1 h2 ub2 wg g wu u act uba wd pd xo:S.seq bf16) (co cg cu cd0 cd1:S.seq f32) : Lemma
  (requires ubg == ubs 2048 gated 0 /\ wo == wsl sw 16255488 2097152 /\ co == mmo 1024 2048 wo ubg 0 128 /\
     po == col0 1024 co /\ x1 == resid_s sx po /\ h2 == rms_s x1 sw 1024 /\ ub2 == ubs 1024 h2 0 /\
     wg == wsl sw 2048 3670016 /\ cg == mmo 3584 1024 wg ub2 0 64 /\ g == col0 3584 cg /\
     wu == wsl sw 3672064 3670016 /\ cu == mmo 3584 1024 wu ub2 0 64 /\ u == col0 3584 cu /\
     act == act_s g u /\ uba == ubs 3584 act 0 /\ wd == wsl sw 7342080 3670016 /\
     cd0 == mmo 1024 3584 wd uba 0 112 /\ cd1 == mmo 1024 3584 wd uba 112 224 /\ pd == col2_s cd0 cd1 /\
     xo == resid_s x1 pd)
  (ensures xo == b_xo sw sx gated)
  = ()

inline_for_extraction noextract
fn l_dots (nkv c ts : sz) (#_ : squash (SZ.v nkv == SZ.v c * 1024 /\ 1 <= SZ.v c /\ SZ.v c <= 12 /\ SZ.v ts < SZ.v c))
  (qn : larray bf16 2048 { is_global qn }) (kvo : larray bf16 (SZ.v nkv) { is_global kvo }) (tmp : larray f32 1024 { is_global tmp }) (dots : larray bf16 128 { is_global dots })
  (#p1 #p2 : perm) (#s1 #s2 : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (qn |-> Frac p1 s1) ** on gpu_loc (kvo |-> Frac p2 s2)
  requires (exists* (s : S.seq f32). on gpu_loc (tmp |-> s)) ** (exists* (s : S.seq bf16). on gpu_loc (dots |-> s))
  ensures (exists* (s : S.seq f32). on gpu_loc (tmp |-> s)) ** (exists* (s : S.seq bf16). on gpu_loc (dots |-> s) ** pure (dots_good s1 s2 (SZ.v c) (SZ.v ts) s))
{
  if (ts = 7sz) {
    t7_scores nkv c kvo qn dots;
    ()
  } else {
    l_score nkv c ts qn kvo tmp dots;
    with d. assert (on gpu_loc (dots |-> d));
    sc_good d s1 s2 (SZ.v c) (SZ.v ts);
    ()
  }
}

inline_for_extraction noextract
fn l_valsg (nkv c ts : sz) (#_ : squash (SZ.v nkv == SZ.v c * 1024 /\ 1 <= SZ.v c /\ SZ.v c <= 12 /\ SZ.v ts < SZ.v c))
  (probs : larray bf16 128 { is_global probs }) (kvo : larray bf16 (SZ.v nkv) { is_global kvo }) (tmp : larray f32 1024 { is_global tmp }) (vals : larray f32 2048 { is_global vals })
  (#p1 #p2 : perm) (#s1 #s2 : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (probs |-> Frac p1 s1) ** on gpu_loc (kvo |-> Frac p2 s2)
  requires (exists* (s : S.seq f32). on gpu_loc (tmp |-> s)) ** (exists* (s : S.seq f32). on gpu_loc (vals |-> s))
  ensures (exists* (s : S.seq f32). on gpu_loc (tmp |-> s)) ** (exists* (s : S.seq f32). on gpu_loc (vals |-> s) ** pure (vals_good s1 s2 (SZ.v c) (SZ.v ts) s))
{
  if (ts = 7sz) {
    t7_values nkv c kvo probs vals;
    ()
  } else {
    l_vals nkv c ts probs kvo tmp vals;
    ()
  }
}

#push-options "--z3rlimit 40"
fn run_main
  (nkv c ts : sz) (#_ : squash (SZ.v nkv == SZ.v c * 1024 /\ 1 <= SZ.v c /\ SZ.v c <= 12 /\ SZ.v ts < SZ.v c))
  (w : larray bf16 18352640 { is_global w })
  (x : larray bf16 1024 { is_global x })
  (kv : larray bf16 (SZ.v nkv) { is_global kv })
  (cs : larray bf16 32 { is_global cs })
  (sn : larray bf16 32 { is_global sn })
  (xo : larray bf16 1024 { is_global xo })
  (kvo : larray bf16 (SZ.v nkv) { is_global kvo })
  (#pw #px #pkv #pc #ps : perm)
  (#sw #sx #skv #sc #ss : erased (S.seq bf16))
  preserves
    cpu **
    on gpu_loc (w |-> Frac pw sw) **
    on gpu_loc (x |-> Frac px sx) **
    on gpu_loc (kv |-> Frac pkv skv) **
    on gpu_loc (cs |-> Frac pc sc) **
    on gpu_loc (sn |-> Frac ps ss)
  requires
    (exists* (s : S.seq bf16). on gpu_loc (xo |-> s)) **
    (exists* (s : S.seq bf16). on gpu_loc (kvo |-> s))
  ensures
    exists* (x' kv' : S.seq bf16).
      on gpu_loc (xo |-> x') ** on gpu_loc (kvo |-> kv') **
      pure (pipe_ok sw sx skv sc ss (SZ.v c) (SZ.v ts) x' kv')
{
  gpu_len w; gpu_len x; gpu_len kv; gpu_len cs; gpu_len sn;
  let hw = Pulse.Lib.Vec.alloc #bf16 zero c_18352640;
  memcpy_device_to_host hw w c_18352640;
  let tmp = galloc_f 1024sz;
  (* attention block *)
  let hin = galloc_bf 1024sz;
  l_rms x w 0sz tmp hin;
  with s_hin. assert (on gpu_loc (hin |-> s_hin));
  let ub1 = galloc_bf 8192sz;
  l_ub 1024sz hin 0sz ub1;
  with s_ub1. assert (on gpu_loc (ub1 |-> s_ub1));
  let wq = load_w hw c_11012096 c_4194304;
  with s_wq. assert (on gpu_loc (wq |-> s_wq));
  let cq = galloc_f 32768sz;
  l_mma 4096sz 1024sz wq ub1 cq 0sz 64sz;
  with s_cq. assert (on gpu_loc (cq |-> s_cq));
  gfree wq;
  let pq = galloc_bf 4096sz;
  l_col0 4096sz cq pq;
  with s_pq. assert (on gpu_loc (pq |-> s_pq));
  gfree cq;
  let wk = load_w hw c_15206400 c_524288;
  with s_wk. assert (on gpu_loc (wk |-> s_wk));
  let ck = galloc_f 4096sz;
  l_mma 512sz 1024sz wk ub1 ck 0sz 64sz;
  with s_ck. assert (on gpu_loc (ck |-> s_ck));
  gfree wk;
  let pk = galloc_bf 512sz;
  l_col0 512sz ck pk;
  with s_pk. assert (on gpu_loc (pk |-> s_pk));
  gfree ck;
  let wv = load_w hw c_15730688 c_524288;
  with s_wv. assert (on gpu_loc (wv |-> s_wv));
  let cv = galloc_f 4096sz;
  l_mma 512sz 1024sz wv ub1 cv 0sz 64sz;
  with s_cv. assert (on gpu_loc (cv |-> s_cv));
  gfree wv;
  let pv = galloc_bf 512sz;
  l_col0 512sz cv pv;
  with s_pv. assert (on gpu_loc (pv |-> s_pv));
  gfree cv;
  gfree ub1;
  let qn = galloc_bf 2048sz;
  l_hn 8sz 512sz pq w c_16254976 cs sn tmp qn;
  with s_qn. assert (on gpu_loc (qn |-> s_qn));
  let kn = galloc_bf 512sz;
  l_hn 2sz 256sz pk w c_16255232 cs sn tmp kn;
  with s_kn. assert (on gpu_loc (kn |-> s_kn));
  chain_front sw sx sc ss s_hin s_ub1 s_wq s_pq s_wk s_pk s_wv s_pv s_qn s_kn s_cq s_ck s_cv;
  l_kvw nkv c ts kv kn pv kvo;
  with s_kvo. assert (on gpu_loc (kvo |-> s_kvo));
  let dots = galloc_bf 128sz;
  l_dots nkv c ts qn kvo tmp dots;
  with s_dots. assert (on gpu_loc (dots |-> s_dots));
  let probs = galloc_bf 128sz;
  l_softmax ts dots tmp probs;
  with s_probs. assert (on gpu_loc (probs |-> s_probs));
  let vals = galloc_f 2048sz;
  l_valsg nkv c ts probs kvo tmp vals;
  with s_vals. assert (on gpu_loc (vals |-> s_vals));
  let gated = galloc_bf 2048sz;
  l_gate vals pq gated;
  with s_gated. assert (on gpu_loc (gated |-> s_gated));
  let ubg = galloc_bf 16384sz;
  l_ub 2048sz gated 0sz ubg;
  with s_ubg. assert (on gpu_loc (ubg |-> s_ubg));
  let wo = load_w hw c_16255488 c_2097152;
  with s_wo. assert (on gpu_loc (wo |-> s_wo));
  let co = galloc_f 8192sz;
  l_mma 1024sz 2048sz wo ubg co 0sz 128sz;
  with s_co. assert (on gpu_loc (co |-> s_co));
  gfree wo;
  let po = galloc_bf 1024sz;
  l_col0 1024sz co po;
  with s_po. assert (on gpu_loc (po |-> s_po));
  gfree co;
  let x1 = galloc_bf 1024sz;
  l_resid x po x1;
  with s_x1. assert (on gpu_loc (x1 |-> s_x1));
  (* mlp block *)
  let h2 = galloc_bf 1024sz;
  l_rms x1 w 1024sz tmp h2;
  with s_h2. assert (on gpu_loc (h2 |-> s_h2));
  let ub2 = galloc_bf 8192sz;
  l_ub 1024sz h2 0sz ub2;
  with s_ub2. assert (on gpu_loc (ub2 |-> s_ub2));
  let wg = load_w hw 2048sz c_3670016;
  with s_wg. assert (on gpu_loc (wg |-> s_wg));
  let cg = galloc_f 28672sz;
  l_mma 3584sz 1024sz wg ub2 cg 0sz 64sz;
  with s_cg. assert (on gpu_loc (cg |-> s_cg));
  gfree wg;
  let g = galloc_bf 3584sz;
  l_col0 3584sz cg g;
  with s_g. assert (on gpu_loc (g |-> s_g));
  let wu = load_w hw c_3672064 c_3670016;
  with s_wu. assert (on gpu_loc (wu |-> s_wu));
  l_mma 3584sz 1024sz wu ub2 cg 0sz 64sz;
  with s_cu. assert (on gpu_loc (cg |-> s_cu));
  gfree wu;
  let u = galloc_bf 3584sz;
  l_col0 3584sz cg u;
  with s_u. assert (on gpu_loc (u |-> s_u));
  gfree cg;
  let act = galloc_bf 3584sz;
  l_act g u act;
  with s_act. assert (on gpu_loc (act |-> s_act));
  let uba = galloc_bf 28672sz;
  l_ub 3584sz act 0sz uba;
  with s_uba. assert (on gpu_loc (uba |-> s_uba));
  let wd = load_w hw c_7342080 c_3670016;
  with s_wd. assert (on gpu_loc (wd |-> s_wd));
  let cd0 = galloc_f 8192sz;
  let cd1 = galloc_f 8192sz;
  l_mma 1024sz 3584sz wd uba cd0 0sz 112sz;
  with s_cd0. assert (on gpu_loc (cd0 |-> s_cd0));
  l_mma 1024sz 3584sz wd uba cd1 112sz 224sz;
  with s_cd1. assert (on gpu_loc (cd1 |-> s_cd1));
  gfree wd;
  let pd = galloc_bf 1024sz;
  l_colx2 cd0 cd1 pd;
  with s_pd. assert (on gpu_loc (pd |-> s_pd));
  l_resid x1 pd xo;
  with s_xo. assert (on gpu_loc (xo |-> s_xo));
  chain_back sw sx s_gated s_ubg s_wo s_po s_x1 s_h2 s_ub2 s_wg s_g s_wu s_u s_act s_uba s_wd s_pd s_xo s_co s_cg s_cu s_cd0 s_cd1;
  pipe_intro sw sx skv sc ss (SZ.v c) (SZ.v ts) s_dots s_vals s_xo s_kvo;
  gfree cd0; gfree cd1; gfree pd; gfree uba; gfree act; gfree u; gfree g; gfree ub2; gfree h2;
  gfree x1; gfree po; gfree ubg; gfree gated; gfree vals; gfree probs; gfree dots; gfree kn; gfree qn;
  gfree pv; gfree pk; gfree pq; gfree hin; gfree tmp;
  Pulse.Lib.Vec.free hw;
  ()
}
#pop-options

#pop-options

(*** Front composition: input norm, q/k/v projections, head norms, KV write *)

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"

(* sequence-level approximation from pointwise facts *)
let cf_seq_intro (s:S.seq bf16) (r:S.seq real)
  : Lemma (requires S.length s == S.length r /\ (forall (i:nat). i < S.length s ==> S.index s i %~ at r i))
          (ensures s %~ r) = ()

let cf_seq_elim (s:S.seq bf16) (r:S.seq real)
  : Lemma (requires s %~ r)
          (ensures S.length s == S.length r /\ (forall (i:nat). i < S.length s ==> S.index s i %~ at r i)) = ()

(* 1. attention input norm *)
let cf_hin (sw sx:S.seq bf16) (rw rx:S.seq real) : Lemma
  (requires S.length sw == 18352640 /\ S.length sx == 1024 /\ sw %~ rw /\ sx %~ rx)
  (ensures S.length (p_hin sw sx) == 1024 /\ p_hin sw sx %~ at_hin base_numerics rw 0 rx)
  = rms1024_approx sx sw 0 rx rw;
    assert (at_hin base_numerics rw 0 rx == rmsnorm1p eps_f32 rx 1024 rw 0);
    cf_seq_intro (p_hin sw sx) (at_hin base_numerics rw 0 rx)

(* 2. generic projection *)
let cf_mod (c k kk:nat) : Lemma (requires kk < k) (ensures (c * k + kk) % k == kk) =
  FStar.Math.Lemmas.lemma_mod_plus kk c k;
  FStar.Math.Lemmas.small_mod kk k

let cf_ub (k:nat) (src:S.seq bf16) (rsrc:S.seq real) : Lemma
  (requires S.length src == k /\ src %~ rsrc)
  (ensures S.length (ubs k src 0) == k * 8 /\
           (forall (kk c:nat). kk < k /\ c < 8 ==> S.index (ubs k src 0) (c * k + kk) %~ at rsrc kk))
  = introduce forall (kk c:nat). kk < k /\ c < 8 ==> S.index (ubs k src 0) (c * k + kk) %~ at rsrc kk
    with introduce _ ==> _ with begin
      cf_mod c k kk;
      FStar.Math.Lemmas.lemma_mult_le_right k (c + 1) 8;
      assert (S.index (ubs k src 0) (c * k + kk) == bat src kk);
      su_bat_approx src rsrc kk
    end

let cf_wsl (sw:S.seq bf16) (rw:S.seq real) (off m:nat) : Lemma
  (requires sw %~ rw /\ off + m <= S.length sw)
  (ensures S.length (wsl sw off m) == m /\
           (forall (idx:nat). idx < m ==> S.index (wsl sw off m) idx %~ at rw (off + idx)))
  = ()

noextract
let cf_ra (rw:S.seq real) (off:nat) (idx:nat) : real = at rw (off + idx)
noextract
let cf_rb (rsrc:S.seq real) (kk:nat) : real = at rsrc kk

let cf_dot (rw rsrc:S.seq real) (off n k i:nat) : Lemma
  (requires i < n /\ 16 /? k)
  (ensures at (matvec rw off n k rsrc) i ==
           vsum (16 * (k / 16 - 0)) (fun kk -> cf_ra rw off (i * k + 16 * 0 + kk) *. cf_rb rsrc (16 * 0 + kk)))
  = let wo = off + i * k in
    assert (16 * (k / 16 - 0) == k);
    assert (at (matvec rw off n k rsrc) i == dot rw wo rsrc k);
    assert (dot rw wo rsrc k == vsum k (fun kk -> at rw (wo + kk) *. at rsrc kk));
    vsum_ext k (fun kk -> at rw (wo + kk) *. at rsrc kk)
               (fun kk -> cf_ra rw off (i * k + 16 * 0 + kk) *. cf_rb rsrc (16 * 0 + kk))

let cf_bound8 (j n:nat) : Lemma (requires j < n) (ensures j * 8 < n * 8) = ()

let cf_proj (n k:pos) (off:nat) (sw src:S.seq bf16) (rw rsrc:S.seq real) : Lemma
  (requires 16 /? n /\ 16 /? k /\ S.length src == k /\ src %~ rsrc /\ sw %~ rw /\ off + n * k <= S.length sw)
  (ensures S.length (col0 n (mmo n k (wsl sw off (n * k)) (ubs k src 0) 0 (k / 16))) == n /\
           col0 n (mmo n k (wsl sw off (n * k)) (ubs k src 0) 0 (k / 16)) %~ matvec rw off n k rsrc)
  = let sa = wsl sw off (n * k) in
    let sb = ubs k src 0 in
    cf_wsl sw rw off (n * k);
    cf_ub k src rsrc;
    mma_col0_approx n k 0 (k / 16) sa sb (cf_ra rw off) (cf_rb rsrc);
    let eA = from_seq (row_major n k) sa in
    let eB = from_seq (col_major k 8) sb in
    let cm = to_seq (row_major n 8) (mma_model eA eB 0 (k / 16)) in
    assert (mmo n k sa sb 0 (k / 16) == cm);
    assert (S.length cm == n * 8);
    let res = col0 n cm in
    introduce forall (j:nat). j < n ==> S.index res j %~ at (matvec rw off n k rsrc) j
    with introduce _ ==> _ with begin
      cf_bound8 j n;
      assert (S.index res j == tb (S.index cm (j * 8)));
      cf_dot rw rsrc off n k j;
      su_tb (S.index cm (j * 8)) (at (matvec rw off n k rsrc) j)
    end;
    cf_seq_intro res (matvec rw off n k rsrc)

let cf_pq (sw sx:S.seq bf16) (rw rx:S.seq real) : Lemma
  (requires S.length sw == 18352640 /\ S.length sx == 1024 /\ sw %~ rw /\ sx %~ rx)
  (ensures S.length (p_pq sw sx) == 4096 /\ p_pq sw sx %~ at_qraw base_numerics rw 0 rx)
  = cf_hin sw sx rw rx;
    cf_proj 4096 1024 11012096 sw (p_hin sw sx) rw (at_hin base_numerics rw 0 rx)

let cf_pk (sw sx:S.seq bf16) (rw rx:S.seq real) : Lemma
  (requires S.length sw == 18352640 /\ S.length sx == 1024 /\ sw %~ rw /\ sx %~ rx)
  (ensures S.length (p_pk sw sx) == 512 /\ p_pk sw sx %~ at_kraw base_numerics rw 0 rx)
  = cf_hin sw sx rw rx;
    cf_proj 512 1024 15206400 sw (p_hin sw sx) rw (at_hin base_numerics rw 0 rx)

let cf_pv (sw sx:S.seq bf16) (rw rx:S.seq real) : Lemma
  (requires S.length sw == 18352640 /\ S.length sx == 1024 /\ sw %~ rw /\ sx %~ rx)
  (ensures S.length (p_pv sw sx) == 512 /\ p_pv sw sx %~ at_v base_numerics rw 0 rx)
  = cf_hin sw sx rw rx;
    cf_proj 512 1024 15730688 sw (p_hin sw sx) rw (at_hin base_numerics rw 0 rx)

#pop-options

(* 3. head norms + rope *)
#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let cf_hn_bound (j heads stride m:nat) : Lemma
  (requires j < heads * 256 /\ heads * stride <= m /\ 256 <= stride)
  (ensures (j / 256) * stride + 256 <= m)
  = let q = j / 256 in
    FStar.Math.Lemmas.lemma_div_mod j 256;
    if q >= heads then FStar.Math.Lemmas.lemma_mult_le_left 256 heads q;
    assert (q < heads);
    FStar.Math.Lemmas.lemma_mult_le_right stride (q + 1) heads

let cf_hn (heads stride:nat) (src w cs sn:S.seq bf16) (woff:nat) (rsrc rw rc rs:S.seq real) : Lemma
  (requires src %~ rsrc /\ heads * stride <= S.length src /\ 256 <= stride /\ w %~ rw /\ woff + 256 <= S.length w /\
            cs %~ rc /\ sn %~ rs /\ S.length cs == 32 /\ S.length sn == 32)
  (ensures S.length (hn_s heads stride src w cs sn woff) == heads * 256 /\
           (forall (j:nat). j < heads * 256 ==>
              S.index (hn_s heads stride src w cs sn woff) j %~
                at (rope (head_norm eps_f32 rsrc ((j / 256) * stride) rw woff) rc rs) (j % 256)))
  = introduce forall (j:nat). j < heads * 256 ==>
              S.index (hn_s heads stride src w cs sn woff) j %~
                at (rope (head_norm eps_f32 rsrc ((j / 256) * stride) rw woff) rc rs) (j % 256)
    with introduce _ ==> _ with begin
      cf_hn_bound j heads stride (S.length src);
      hn_approx src w cs sn ((j / 256) * stride) woff rsrc rw rc rs
    end
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let cf_atq_eq (rw rx rc rs:S.seq real) (j:nat) : Lemma
  (requires j < 2048)
  (ensures at (at_q base_numerics rw 0 rx rc rs) j ==
           at (rope (head_norm eps_f32 (at_qraw base_numerics rw 0 rx) ((j / 256) * 512) rw 16254976) rc rs) (j % 256))
  = assert (at_mix 0 + 5242880 == 16254976);
    assert (base_numerics.eps == eps_f32)

let cf_atk_eq (rw rx rc rs:S.seq real) (j:nat) : Lemma
  (requires j < 512)
  (ensures at (at_k base_numerics rw 0 rx rc rs) j ==
           at (rope (head_norm eps_f32 (at_kraw base_numerics rw 0 rx) ((j / 256) * 256) rw 16255232) rc rs) (j % 256))
  = assert (at_mix 0 + 5243136 == 16255232);
    assert (base_numerics.eps == eps_f32)

let cf_qn (sw sx sc ss:S.seq bf16) (rw rx rc rs:S.seq real) : Lemma
  (requires S.length sw == 18352640 /\ S.length sx == 1024 /\ S.length sc == 32 /\ S.length ss == 32 /\
            sw %~ rw /\ sx %~ rx /\ sc %~ rc /\ ss %~ rs)
  (ensures S.length (p_qn sw sx sc ss) == 2048 /\
           (forall (j:nat). j < 2048 ==> S.index (p_qn sw sx sc ss) j %~ at (at_q base_numerics rw 0 rx rc rs) j))
  = cf_pq sw sx rw rx;
    cf_hn 8 512 (p_pq sw sx) sw sc ss 16254976 (at_qraw base_numerics rw 0 rx) rw rc rs;
    FStar.Classical.forall_intro (FStar.Classical.move_requires (cf_atq_eq rw rx rc rs))

let cf_kn (sw sx sc ss:S.seq bf16) (rw rx rc rs:S.seq real) : Lemma
  (requires S.length sw == 18352640 /\ S.length sx == 1024 /\ S.length sc == 32 /\ S.length ss == 32 /\
            sw %~ rw /\ sx %~ rx /\ sc %~ rc /\ ss %~ rs)
  (ensures S.length (p_kn sw sx sc ss) == 512 /\ p_kn sw sx sc ss %~ at_k base_numerics rw 0 rx rc rs)
  = cf_pk sw sx rw rx;
    cf_hn 2 256 (p_pk sw sx) sw sc ss 16255232 (at_kraw base_numerics rw 0 rx) rw rc rs;
    FStar.Classical.forall_intro (FStar.Classical.move_requires (cf_atk_eq rw rx rc rs));
    cf_seq_intro (p_kn sw sx sc ss) (at_k base_numerics rw 0 rx rc rs)
#pop-options

(* 4. KV write *)
#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let cf_comp_bound (im q:nat) : Lemma (requires q > 0 /\ im < 2 * q) (ensures im / q < 2)
  = FStar.Math.Lemmas.lemma_div_le im (2 * q - 1) q;
    FStar.Math.Lemmas.lemma_div_plus (q - 1) 1 q;
    FStar.Math.Lemmas.small_div (q - 1) q

let cf_kv_comp_lt (c:pos) (i:nat) : Lemma (base_kv_comp c i < 512 /\ base_kv_comp c i >= 0)
  = FStar.Math.Lemmas.lemma_mod_lt i (c * 512);
    cf_comp_bound (i % (c * 512)) (c * 256)

let cf_kvo_i (c t:nat) (skv kn pv:S.seq bf16) (rkv rk rv:S.seq real) (i:nat) : Lemma
  (requires 1 <= c /\ S.length skv == S.length rkv /\ i < S.length skv /\ skv %~ rkv /\ kn %~ rk /\ pv %~ rv /\
            S.length kn == 512 /\ S.length pv == 512)
  (ensures S.index (kvw_s c t skv kn pv) i %~ at (base_kv_write c t rkv rk rv) i /\
           (base_kv_pos c i <> t ==> S.index (kvw_s c t skv kn pv) i == S.index skv i))
  = cf_kv_comp_lt c i;
    let p = base_kv_comp c i in
    su_bat_approx skv rkv i;
    su_bat_approx kn rk p;
    su_bat_approx pv rv p

let cf_kvo (c t:nat) (skv kn pv:S.seq bf16) (rkv rk rv:S.seq real) : Lemma
  (requires 1 <= c /\ S.length skv == S.length rkv /\ skv %~ rkv /\ kn %~ rk /\ pv %~ rv /\
            S.length kn == 512 /\ S.length pv == 512)
  (ensures S.length (kvw_s c t skv kn pv) == S.length skv /\
           kvw_s c t skv kn pv %~ base_kv_write c t rkv rk rv /\
           (forall (i:nat). i < S.length skv /\ base_kv_pos c i <> t ==> S.index (kvw_s c t skv kn pv) i == S.index skv i))
  = introduce forall (i:nat). i < S.length skv ==>
       (S.index (kvw_s c t skv kn pv) i %~ at (base_kv_write c t rkv rk rv) i /\
        (base_kv_pos c i <> t ==> S.index (kvw_s c t skv kn pv) i == S.index skv i))
    with introduce _ ==> _ with cf_kvo_i c t skv kn pv rkv rk rv i;
    cf_seq_intro (kvw_s c t skv kn pv) (base_kv_write c t rkv rk rv)
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 60"
let front_ok (sw sx skv sc ss:S.seq bf16) (c t:nat) (rw rx rkv rc rs:S.seq real) : Lemma
  (requires S.length sw == 18352640 /\ S.length sx == 1024 /\ S.length skv == c * 1024 /\ 1 <= c /\ c <= 12 /\ t < c /\
            S.length sc == 32 /\ S.length ss == 32 /\
            sw %~ rw /\ sx %~ rx /\ skv %~ rkv /\ sc %~ rc /\ ss %~ rs)
  (ensures
     S.length (p_pq sw sx) == 4096 /\ S.length (p_qn sw sx sc ss) == 2048 /\
     S.length (p_kvo sw sx skv sc ss c t) == c * 1024 /\
     (forall (j:nat). j < 4096 ==> S.index (p_pq sw sx) j %~ at (at_qraw base_numerics rw 0 rx) j) /\
     (forall (j:nat). j < 2048 ==> S.index (p_qn sw sx sc ss) j %~ at (at_q base_numerics rw 0 rx rc rs) j) /\
     p_kvo sw sx skv sc ss c t %~ base_kv_write c t rkv (at_k base_numerics rw 0 rx rc rs) (at_v base_numerics rw 0 rx) /\
     (forall (i:nat). i < c * 1024 /\ base_kv_pos c i <> t ==> S.index (p_kvo sw sx skv sc ss c t) i == S.index skv i))
  = cf_pq sw sx rw rx;
    cf_seq_elim (p_pq sw sx) (at_qraw base_numerics rw 0 rx);
    cf_qn sw sx sc ss rw rx rc rs;
    cf_kn sw sx sc ss rw rx rc rs;
    cf_pv sw sx rw rx;
    cf_kvo c t skv (p_kn sw sx sc ss) (p_pv sw sx) rkv (at_k base_numerics rw 0 rx rc rs) (at_v base_numerics rw 0 rx)
#pop-options

(*** comp_back: o_proj, residual, post-attention norm, SwiGLU MLP *)

#push-options "--z3rlimit 40"

let cb_mod (c k kk:nat) : Lemma (requires k > 0 /\ kk < k) (ensures (c * k + kk) % k == kk)
  = FStar.Math.Lemmas.lemma_mod_plus kk c k; FStar.Math.Lemmas.small_mod kk k

let cb_wsl (sw:S.seq bf16) (rw:S.seq real) (off len:nat) : Lemma
  (requires sw %~ rw /\ off + len <= S.length sw)
  (ensures S.length (wsl sw off len) == len /\
           (forall (idx:nat). idx < len ==> S.index (wsl sw off len) idx %~ at rw (off + idx)))
  = let aux (idx:nat) : Lemma (idx < len ==> S.index (wsl sw off len) idx %~ at rw (off + idx)) =
      if idx < len then su_bat_approx sw rw (off + idx) in
    FStar.Classical.forall_intro aux

let cb_ubs (k:pos) (src:S.seq bf16) (rs:S.seq real) : Lemma
  (requires S.length src == k /\ (forall (kk:nat). kk < k ==> S.index src kk %~ at rs kk))
  (ensures S.length (ubs k src 0) == k * 8 /\
           (forall (kk c:nat). kk < k /\ c < 8 ==> S.index (ubs k src 0) (c * k + kk) %~ at rs kk))
  = let aux (kk c:nat) : Lemma (kk < k /\ c < 8 ==> S.index (ubs k src 0) (c * k + kk) %~ at rs kk) =
      if kk < k && c < 8 then begin
        mm_bound c kk 8 k;
        cb_mod c k kk;
        assert (S.index (ubs k src 0) (c * k + kk) == bat src (0 + (c * k + kk) % k))
      end in
    FStar.Classical.forall_intro_2 aux

noextract
let cb_ra (rw:S.seq real) (off:nat) (idx:nat) : real = at rw (off + idx)
noextract
let cb_rb (rs:S.seq real) (kk:nat) : real = at rs kk
noextract
let cb_dterm (rw rs:S.seq real) (off k i:nat) (kk:nat) : real = at rw (off + i * k + kk) *. at rs kk

let cb_dot_eq (w x:S.seq real) (wo:int) (n:nat) (g:nat -> real) : Lemma
  (requires forall (kk:nat). kk < n ==> g kk == at w (wo + kk) *. at x kk)
  (ensures dot w wo x n == vsum n g)
  = vsum_ext n (fun k -> at w (wo + k) *. at x k) g

noextract
let cb_term (rw rs:S.seq real) (off k j0 i:nat) (kk:nat) : real =
  at rw (off + (i * k + 16 * j0 + kk)) *. at rs (16 * j0 + kk)

let cb_lam_eq (rw rs:S.seq real) (off k j0 i m:nat) : Lemma
  (vsum m (fun kk -> cb_ra rw off (i * k + 16 * j0 + kk) *. cb_rb rs (16 * j0 + kk)) == vsum m (cb_term rw rs off k j0 i))
  = vsum_ext m (fun kk -> cb_ra rw off (i * k + 16 * j0 + kk) *. cb_rb rs (16 * j0 + kk)) (cb_term rw rs off k j0 i)

let cb_mmo_eq (n k:pos) (sa sb:S.seq bf16) (j0 j1:nat) : Lemma
  (requires n % 16 == 0 /\ k % 16 == 0 /\ j1 <= k / 16 /\ S.length sa == n * k /\ S.length sb == k * 8)
  (ensures 16 /? n /\ 16 /? k /\
           mmo n k sa sb j0 j1 == to_seq (row_major n 8) (mma_model (from_seq (row_major n k) sa) (from_seq (col_major k 8) sb) j0 j1))
  = Kuiper.Divides.lemma_divides_mod 16 n; Kuiper.Divides.lemma_divides_mod 16 k

let cb_raw (n k:pos) (nk:nat) (j0 j1:nat) (sw:S.seq bf16) (rw:S.seq real) (off:nat) (src:S.seq bf16) (rs:S.seq real) : Lemma
  (requires n % 16 == 0 /\ k % 16 == 0 /\ j0 <= j1 /\ j1 <= k / 16 /\ nk == n * k /\ sw %~ rw /\ off + nk <= S.length sw /\
            S.length src == k /\ (forall (kk:nat). kk < k ==> S.index src kk %~ at rs kk))
  (ensures forall (i:nat). i < n ==>
     fat (mmo n k (wsl sw off nk) (ubs k src 0) j0 j1) (i * 8) %~ vsum (16 * (j1 - j0)) (cb_term rw rs off k j0 i))
  = let sa = wsl sw off nk in
    let sb = ubs k src 0 in
    cb_wsl sw rw off nk;
    cb_ubs k src rs;
    cb_mmo_eq n k sa sb j0 j1;
    mma_col0_approx n k j0 j1 sa sb (cb_ra rw off) (cb_rb rs);
    let m = mma_model (from_seq (row_major n k) sa) (from_seq (col_major k 8) sb) j0 j1 in
    introduce forall (i:nat). i < n ==>
       fat (mmo n k sa sb j0 j1) (i * 8) %~ vsum (16 * (j1 - j0)) (cb_term rw rs off k j0 i)
    with introduce _ ==> _ with begin
      mm_to_rm n 8 m i 0;
      assert (fat (mmo n k sa sb j0 j1) (i * 8) == S.index (to_seq (row_major n 8) m) (i * 8));
      cb_lam_eq rw rs off k j0 i (16 * (j1 - j0))
    end

let cb_matvec_idx (rw rs:S.seq real) (off n k:nat) (i:nat) : Lemma
  (requires i < n)
  (ensures at (matvec rw off n k rs) i == vsum k (cb_dterm rw rs off k i))
  = cb_dot_eq rw rs (off + i * k) k (cb_dterm rw rs off k i)

let cb_full (n k:pos) (nk:nat) (j1:nat) (sw:S.seq bf16) (rw:S.seq real) (off:nat) (src:S.seq bf16) (rs:S.seq real) : Lemma
  (requires n % 16 == 0 /\ k % 16 == 0 /\ 16 * j1 == k /\ nk == n * k /\ sw %~ rw /\ off + nk <= S.length sw /\
            S.length src == k /\ (forall (kk:nat). kk < k ==> S.index src kk %~ at rs kk))
  (ensures S.length (col0 n (mmo n k (wsl sw off nk) (ubs k src 0) 0 j1)) == n /\
           (forall (i:nat). i < n ==> S.index (col0 n (mmo n k (wsl sw off nk) (ubs k src 0) 0 j1)) i %~ at (matvec rw off n k rs) i))
  = cb_raw n k nk 0 j1 sw rw off src rs;
    let c = mmo n k (wsl sw off nk) (ubs k src 0) 0 j1 in
    introduce forall (i:nat). i < n ==> S.index (col0 n c) i %~ at (matvec rw off n k rs) i
    with introduce _ ==> _ with begin
      cb_matvec_idx rw rs off n k i;
      vsum_ext k (cb_term rw rs off k 0 i) (cb_dterm rw rs off k i);
      su_tb (fat c (i * 8)) (at (matvec rw off n k rs) i)
    end

let cb_split_real (rw rs:S.seq real) (off k i jm:nat) : Lemma
  (requires k == 16 * jm + 16 * jm)
  (ensures vsum k (cb_dterm rw rs off k i) == vsum (16 * jm) (cb_term rw rs off k 0 i) +. vsum (16 * jm) (cb_term rw rs off k jm i))
  = let f = cb_dterm rw rs off k i in
    vsum_ext (16 * jm) (cb_term rw rs off k 0 i) f;
    su_vsum_split (16 * jm) (16 * jm) f (cb_term rw rs off k jm i)

let cb_raw_at (n k:pos) (nk:nat) (j0 j1 m:nat) (sw:S.seq bf16) (rw:S.seq real) (off:nat) (src:S.seq bf16) (rs:S.seq real) (i:nat) : Lemma
  (requires n % 16 == 0 /\ k % 16 == 0 /\ j0 <= j1 /\ j1 <= k / 16 /\ nk == n * k /\ sw %~ rw /\ off + nk <= S.length sw /\
            S.length src == k /\ (forall (kk:nat). kk < k ==> S.index src kk %~ at rs kk) /\ i < n /\ m == 16 * (j1 - j0))
  (ensures fat (mmo n k (wsl sw off nk) (ubs k src 0) j0 j1) (i * 8) %~ vsum m (cb_term rw rs off k j0 i))
  = cb_raw n k nk j0 j1 sw rw off src rs

let cb_add_tb (x0 x1:f32) (r0 r1:real) : Lemma (requires x0 %~ r0 /\ x1 %~ r1) (ensures tb (add x0 x1) %~ (r0 +. r1))
  = a_add x0 x1 r0 r1; su_tb (add x0 x1) (r0 +. r1)

let cb_split (n k:pos) (nk:nat) (jm je:nat) (sw:S.seq bf16) (rw:S.seq real) (off:nat) (src:S.seq bf16) (rs:S.seq real) (i:nat) : Lemma
  (requires n % 16 == 0 /\ k % 16 == 0 /\ 16 * je == k /\ je == jm + jm /\ nk == n * k /\ sw %~ rw /\ off + nk <= S.length sw /\
            S.length src == k /\ (forall (kk:nat). kk < k ==> S.index src kk %~ at rs kk) /\ i < n)
  (ensures tb (add (fat (mmo n k (wsl sw off nk) (ubs k src 0) 0 jm) (i * 8))
                   (fat (mmo n k (wsl sw off nk) (ubs k src 0) jm je) (i * 8))) %~ at (matvec rw off n k rs) i)
  = assert (k == 16 * jm + 16 * jm /\ 16 * jm == 16 * (jm - 0) /\ 16 * jm == 16 * (je - jm) /\ jm <= je /\ je <= k / 16);
    cb_split_real rw rs off k i jm;
    cb_matvec_idx rw rs off n k i;
    cb_raw_at n k nk 0 jm (16 * jm) sw rw off src rs i;
    cb_raw_at n k nk jm je (16 * jm) sw rw off src rs i;
    cb_add_tb (fat (mmo n k (wsl sw off nk) (ubs k src 0) 0 jm) (i * 8))
              (fat (mmo n k (wsl sw off nk) (ubs k src 0) jm je) (i * 8))
              (vsum (16 * jm) (cb_term rw rs off k 0 i)) (vsum (16 * jm) (cb_term rw rs off k jm i))

let cb_seq (s:S.seq bf16) (r:S.seq real) : Lemma
  (requires S.length s == S.length r /\ (forall (i:nat). i < S.length s ==> S.index s i %~ S.index r i))
  (ensures s %~ r)
  = ()

(* stage 1: o_proj + residual *)
let cb_x1 (sw sx gated:S.seq bf16) (rw rx g:S.seq real) : Lemma
  (requires S.length sw == 18352640 /\ sw %~ rw /\ S.length sx == 1024 /\ sx %~ rx /\ S.length gated == 2048 /\
            (forall (j:nat). j < 2048 ==> S.index gated j %~ at g j))
  (ensures b_x1 sw sx gated %~ vadd rx (matvec rw 16255488 1024 2048 g) 1024)
  = cb_full 1024 2048 2097152 128 sw rw 16255488 gated g;
    let po = b_po sw gated in
    let mv = matvec rw 16255488 1024 2048 g in
    let x1 = b_x1 sw sx gated in
    let r = vadd rx mv 1024 in
    introduce forall (j:nat). j < 1024 ==> S.index x1 j %~ S.index r j
    with introduce _ ==> _ with begin
      su_bat_approx sx rx j;
      resid_approx sx po j (at rx j) (at mv j)
    end;
    cb_seq x1 r

(* stage 2: post-attention norm *)
let cb_h2 (sw x1:S.seq bf16) (rw rx1:S.seq real) : Lemma
  (requires S.length sw == 18352640 /\ sw %~ rw /\ S.length x1 == 1024 /\ x1 %~ rx1)
  (ensures rms_s x1 sw 1024 %~ mlp_in base_numerics rw 0 rx1)
  = rms1024_approx x1 sw 1024 rx1 rw;
    assert (mlp_in base_numerics rw 0 rx1 == rmsnorm1p eps_f32 rx1 1024 rw 1024);
    cb_seq (rms_s x1 sw 1024) (mlp_in base_numerics rw 0 rx1)

(* stage 3: gate/up projections + activation *)
let cb_act (sw h2:S.seq bf16) (rw rx1:S.seq real) : Lemma
  (requires S.length sw == 18352640 /\ sw %~ rw /\ S.length h2 == 1024 /\ h2 %~ mlp_in base_numerics rw 0 rx1)
  (ensures S.length (act_s (col0 3584 (mmo 3584 1024 (wsl sw 2048 3670016) (ubs 1024 h2 0) 0 64))
                           (col0 3584 (mmo 3584 1024 (wsl sw 3672064 3670016) (ubs 1024 h2 0) 0 64))) == 3584 /\
           (forall (i:nat). i < 3584 ==>
             S.index (act_s (col0 3584 (mmo 3584 1024 (wsl sw 2048 3670016) (ubs 1024 h2 0) 0 64))
                            (col0 3584 (mmo 3584 1024 (wsl sw 3672064 3670016) (ubs 1024 h2 0) 0 64))) i
               %~ at (mlp_act base_numerics rw 0 rx1) i))
  = let h = mlp_in base_numerics rw 0 rx1 in
    let aux (kk:nat) : Lemma (kk < 1024 ==> S.index h2 kk %~ at h kk) =
      if kk < 1024 then su_bat_approx h2 h kk in
    FStar.Classical.forall_intro aux;
    cb_full 3584 1024 3670016 64 sw rw 2048 h2 h;
    cb_full 3584 1024 3670016 64 sw rw 3672064 h2 h;
    let sg = col0 3584 (mmo 3584 1024 (wsl sw 2048 3670016) (ubs 1024 h2 0) 0 64) in
    let su = col0 3584 (mmo 3584 1024 (wsl sw 3672064 3670016) (ubs 1024 h2 0) 0 64) in
    let gr = matvec rw 2048 3584 1024 h in
    let ur = matvec rw 3672064 3584 1024 h in
    introduce forall (i:nat). i < 3584 ==> S.index (act_s sg su) i %~ at (mlp_act base_numerics rw 0 rx1) i
    with introduce _ ==> _ with begin
      act_approx sg su i (at gr i) (at ur i);
      assert (at (mlp_act base_numerics rw 0 rx1) i == silu (at gr i) *. at ur i)
    end

(* stage 4: down projection, two k-halves *)
let cb_pd (sw a:S.seq bf16) (rw ra:S.seq real) : Lemma
  (requires S.length sw == 18352640 /\ sw %~ rw /\ S.length a == 3584 /\
            (forall (kk:nat). kk < 3584 ==> S.index a kk %~ at ra kk))
  (ensures forall (i:nat). i < 1024 ==>
     S.index (col2_s (mmo 1024 3584 (wsl sw 7342080 3670016) (ubs 3584 a 0) 0 112)
                     (mmo 1024 3584 (wsl sw 7342080 3670016) (ubs 3584 a 0) 112 224)) i
       %~ at (matvec rw 7342080 1024 3584 ra) i)
  = introduce forall (i:nat). i < 1024 ==>
       S.index (col2_s (mmo 1024 3584 (wsl sw 7342080 3670016) (ubs 3584 a 0) 0 112)
                       (mmo 1024 3584 (wsl sw 7342080 3670016) (ubs 3584 a 0) 112 224)) i
         %~ at (matvec rw 7342080 1024 3584 ra) i
    with introduce _ ==> _ with
      cb_split 1024 3584 3670016 112 224 sw rw 7342080 a ra i

#pop-options

#push-options "--z3rlimit 60"
let back_ok (sw sx gated:S.seq bf16) (rw rx g:S.seq real) : Lemma
  (requires S.length sw == 18352640 /\ sw %~ rw /\ S.length sx == 1024 /\ sx %~ rx /\ S.length gated == 2048 /\
            (forall (j:nat). j < 2048 ==> S.index gated j %~ at g j))
  (ensures b_xo sw sx gated %~ mlp_out base_numerics rw 0 (vadd rx (matvec rw 16255488 1024 2048 g) 1024))
  = let rx1 = vadd rx (matvec rw 16255488 1024 2048 g) 1024 in
    let x1 = b_x1 sw sx gated in
    cb_x1 sw sx gated rw rx g;
    let h2 = b_h2 sw sx gated in
    cb_h2 sw x1 rw rx1;
    cb_act sw h2 rw rx1;
    let ac = act_s (b_g sw sx gated) (b_u sw sx gated) in
    let ra = mlp_act base_numerics rw 0 rx1 in
    assert (S.length ac == 3584);
    cb_pd sw ac rw ra;
    let pd = b_pd sw sx gated in
    let mv = matvec rw 7342080 1024 3584 ra in
    let xo = b_xo sw sx gated in
    let r = mlp_out base_numerics rw 0 rx1 in
    introduce forall (j:nat). j < 1024 ==> S.index xo j %~ S.index r j
    with introduce _ ==> _ with begin
      su_bat_approx x1 rx1 j;
      resid_approx x1 pd j (at rx1 j) (at mv j)
    end;
    cb_seq xo r
#pop-options
(*** comp_probs: softmax probabilities approximate at_prob *)

noextract
let cp_rk (kn:S.seq real) (hk:nat -> nat -> real) (t h p:nat) (j:nat) : real = hist_or hk kn t p ((h / 4) * 256 + j)
noextract
let cp_sf (rq kn:S.seq real) (hk:nat -> nat -> real) (t h p:nat) (i:nat) : real = at rq (h * 256 + i) *. cp_rk kn hk t h p i
noextract
let cp_rd (rq kn:S.seq real) (hk:nat -> nat -> real) (t h:nat) (p:nat) : real = vsum 256 (cp_sf rq kn hk t h p)

#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
let cp_score_eq (nm:numerics) (w:S.seq real) (wo:int) (x cs sn:S.seq real) (t:nat) (hk:nat -> nat -> real) (hd p:nat) : Lemma
  (at_score nm w wo x cs sn t hk hd p == cp_rd (at_q nm w wo x cs sn) (at_k nm w wo x cs sn) hk t hd p *. 0.0625R)
  = vsum_ext 256 (fun i -> at (at_q nm w wo x cs sn) (hd * 256 + i) *. hist_or hk (at_k nm w wo x cs sn) t p ((hd / 4) * 256 + i))
      (cp_sf (at_q nm w wo x cs sn) (at_k nm w wo x cs sn) hk t hd p)

let cp_prob_eq (nm:numerics) (w:S.seq real) (wo:int) (x cs sn:S.seq real) (t:nat) (hk:nat -> nat -> real) (hd p:nat)
  (rd:nat -> real) : Lemma
  (requires p <= t /\ (forall (p':nat). p' <= t ==> at_score nm w wo x cs sn t hk hd p' == rd p' *. 0.0625R))
  (ensures at_prob nm w wo x cs sn t hk hd p ==
           rdiv (FStar.Math.Exp.exp (rd p *. 0.0625R)) (vsum (t + 1) (fun p' -> FStar.Math.Exp.exp (rd p' *. 0.0625R))))
  = vsum_ext (t + 1) (fun p' -> FStar.Math.Exp.exp (at_score nm w wo x cs sn t hk hd p'))
                     (fun p' -> FStar.Math.Exp.exp (rd p' *. 0.0625R))
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let cp_arith (c:pos) (g:nat) (p:nat) (j:nat) : Lemma
  (requires g < 2 /\ p < c /\ j < 256)
  (ensures ((g * c + p) * 256 + j < c * 512 /\
            base_kv_index c p (g * 256 + j) == (g * c + p) * 256 + j /\
            base_kv_pos c ((g * c + p) * 256 + j) == p /\
            base_kv_comp c ((g * c + p) * 256 + j) == g * 256 + j))
  = let idx = (g * c + p) * 256 + j in
    let r = p * 256 + j in
    assert (idx == r + g * (c * 256));
    assert (idx == j + (g * c + p) * 256);
    FStar.Math.Lemmas.lemma_mod_plus j (g * c + p) 256;
    FStar.Math.Lemmas.small_mod j 256;
    assert (idx % 256 == j);
    assert (idx < c * 512);
    FStar.Math.Lemmas.small_mod idx (c * 512);
    FStar.Math.Lemmas.lemma_mod_plus r g (c * 256);
    FStar.Math.Lemmas.small_mod r (c * 256);
    FStar.Math.Lemmas.lemma_div_plus r g (c * 256);
    FStar.Math.Lemmas.small_div r (c * 256);
    FStar.Math.Lemmas.lemma_div_plus j p 256;
    FStar.Math.Lemmas.small_div j 256;
    FStar.Math.Lemmas.lemma_div_plus j g 256;
    FStar.Math.Lemmas.lemma_mod_plus j g 256;
    assert ((g * 256 + j) / 256 == g);
    assert ((g * 256 + j) % 256 == j)
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let cp_bkw_at (c t:nat) (rkv kn vn:S.seq real) (i:nat) : Lemma
  (requires i < S.length rkv)
  (ensures at (base_kv_write c t rkv kn vn) i ==
           (if base_kv_pos c i = t then (if i < c * 512 then at kn (base_kv_comp c i) else at vn (base_kv_comp c i))
            else at rkv i))
  = ()

let cp_bkw_hit (c t:nat) (rkv kn vn:S.seq real) (i jj:nat) : Lemma
  (requires i < S.length rkv /\ base_kv_pos c i == t /\ i < c * 512 /\ base_kv_comp c i == jj)
  (ensures at (base_kv_write c t rkv kn vn) i == at kn jj)
  = cp_bkw_at c t rkv kn vn i

let cp_bkw_miss (c t:nat) (rkv kn vn:S.seq real) (i:nat) : Lemma
  (requires i < S.length rkv /\ base_kv_pos c i <> t)
  (ensures at (base_kv_write c t rkv kn vn) i == at rkv i)
  = cp_bkw_at c t rkv kn vn i

let cp_hist (c t:nat) (rkv kn:S.seq real) (p jj:nat) : Lemma
  (hist_or (base_hk c rkv) kn t p jj == (if p < t then at rkv (base_kv_index c p jj) else at kn jj))
  = ()

let cp_key_core (c t:nat) (rkv kn vn:S.seq real) (p jj idx:nat) : Lemma
  (requires idx < S.length rkv /\ base_kv_index c p jj == idx /\ base_kv_pos c idx == p /\
            base_kv_comp c idx == jj /\ idx < c * 512 /\ p <= t)
  (ensures at (base_kv_write c t rkv kn vn) idx == hist_or (base_hk c rkv) kn t p jj)
  = cp_hist c t rkv kn p jj;
    if p < t then cp_bkw_miss c t rkv kn vn idx else cp_bkw_hit c t rkv kn vn idx jj

let cp_key (kvo:S.seq bf16) (c t:nat) (rkv kn vn:S.seq real) (h p j:nat) : Lemma
  (requires S.length kvo == c * 1024 /\ S.length rkv == c * 1024 /\ 1 <= c /\ t < c /\ h < 8 /\ p <= t /\ j < 256 /\
            kvo %~ base_kv_write c t rkv kn vn)
  (ensures bat kvo (((h / 4) * c + p) * 256 + j) %~ cp_rk kn (base_hk c rkv) t h p j)
  = let g = h / 4 in
    assert (g < 2);
    cp_arith c g p j;
    let idx = (g * c + p) * 256 + j in
    let jj = g * 256 + j in
    assert (idx < c * 1024);
    su_bat_approx kvo (base_kv_write c t rkv kn vn) idx;
    cp_key_core c t rkv kn vn p jj idx;
    assert (cp_rk kn (base_hk c rkv) t h p j == hist_or (base_hk c rkv) kn t p jj)
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
let cp_q_approx (qn:S.seq bf16) (rq:S.seq real) : Lemma
  (requires S.length qn == 2048 /\ S.length rq == 2048 /\ (forall (j:nat). j < 2048 ==> S.index qn j %~ at rq j))
  (ensures qn %~ rq) = ()

let cp_divmod16 (h p:nat) : Lemma (requires p < 16) (ensures (h * 16 + p) / 16 == h /\ (h * 16 + p) % 16 == p)
  = FStar.Math.Lemmas.lemma_div_plus p h 16; FStar.Math.Lemmas.small_div p 16;
    FStar.Math.Lemmas.lemma_mod_plus p h 16; FStar.Math.Lemmas.small_mod p 16

let cp_divmod256 (p kk:nat) : Lemma (requires kk < 256) (ensures (p * 256 + kk) / 256 == p /\ (p * 256 + kk) % 256 == kk)
  = FStar.Math.Lemmas.lemma_div_plus kk p 256; FStar.Math.Lemmas.small_div kk 256;
    FStar.Math.Lemmas.lemma_mod_plus kk p 256; FStar.Math.Lemmas.small_mod kk 256
#pop-options

(* t <> 7: scalar score kernel *)
#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let cp_caseA (qn kvo:S.seq bf16) (c t:nat) (rkv kn vn rq:S.seq real) (dots:S.seq bf16) (h p:nat) : Lemma
  (requires S.length qn == 2048 /\ S.length kvo == c * 1024 /\ S.length rkv == c * 1024 /\ 1 <= c /\ c <= 12 /\ t < c /\
            t <> 7 /\ qn %~ rq /\ kvo %~ base_kv_write c t rkv kn vn /\ dots_good qn kvo c t dots /\ h < 8 /\ p <= t)
  (ensures bat dots (h * 16 + p) %~ cp_rd rq kn (base_hk c rkv) t h p)
  = introduce forall (j:nat). j < 256 ==> bat kvo (((h / 4) * c + p) * 256 + j) %~ cp_rk kn (base_hk c rkv) t h p j
    with introduce _ ==> _ with cp_key kvo c t rkv kn vn h p j;
    score_approx qn kvo c h p rq (cp_rk kn (base_hk c rkv) t h p);
    vsum_ext 256 (fun i -> at rq (h * 256 + i) *. cp_rk kn (base_hk c rkv) t h p i) (cp_sf rq kn (base_hk c rkv) t h p);
    assert (S.index dots (h * 16 + p) == score_m qn kvo c h p)
#pop-options

(* t == 7: tensor-core score kernel *)
noextract
let cp_ra (kn:S.seq real) (hk:nat -> nat -> real) (t h:nat) (idx:nat) : real =
  if idx / 256 < 8 then cp_rk kn hk t h (idx / 256) (idx % 256) else 0.0R
noextract
let cp_rb (rq:S.seq real) (h:nat) (kk:nat) : real = at rq (h * 256 + kk)

#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
let cp_zero_bf () : Lemma ((zero #bf16) %~ 0.0R) = a0 #bf16
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let cp_keys_ok (kvo:S.seq bf16) (c t:nat) (rkv kn vn:S.seq real) (h idx:nat) : Lemma
  (requires S.length kvo == c * 1024 /\ S.length rkv == c * 1024 /\ 8 <= c /\ t == 7 /\ h < 8 /\ idx < 4096 /\
            kvo %~ base_kv_write c t rkv kn vn)
  (ensures S.index (keys_s kvo c h) idx %~ cp_ra kn (base_hk c rkv) t h idx)
  = if idx / 256 < 8 then cp_key kvo c t rkv kn vn h (idx / 256) (idx % 256) else cp_zero_bf ()

let cp_ub_ok (qn:S.seq bf16) (rq:S.seq real) (h kk c':nat) : Lemma
  (requires S.length qn == 2048 /\ qn %~ rq /\ h < 8 /\ kk < 256 /\ c' < 8)
  (ensures S.index (ubs 256 qn (h * 256)) (c' * 256 + kk) %~ cp_rb rq h kk)
  = FStar.Math.Lemmas.lemma_mod_plus kk c' 256; FStar.Math.Lemmas.small_mod kk 256;
    su_bat_approx qn rq (h * 256 + kk)

let cp_row (rq kn:S.seq real) (hk:nat -> nat -> real) (t h p kk:nat) : Lemma
  (requires p < 8 /\ kk < 256)
  (ensures cp_ra kn hk t h (p * 256 + 16 * 0 + kk) *. cp_rb rq h (16 * 0 + kk) == cp_sf rq kn hk t h p kk)
  = cp_divmod256 p kk
#pop-options

#push-options "--fuel 3 --ifuel 1 --z3rlimit 40"
let cp_len (m:chest2 f32 16 8) : Lemma (S.length (to_seq (row_major 16 8) m) == 128) = ()

let cp_mmo_eq (sa sb:S.seq bf16) : Lemma
  (requires S.length sa == 4096 /\ S.length sb == 2048)
  (ensures mmo 16 256 sa sb 0 16 ==
           to_seq (row_major 16 8) (mma_model (from_seq (row_major 16 256) sa) (from_seq (col_major 256 8) sb) 0 16)) = ()
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let cp_caseB1 (qn kvo:S.seq bf16) (c t:nat) (rkv kn rq:S.seq real) (dots:S.seq bf16) (h p:nat) : Lemma
  (requires t == 7 /\ S.length dots == 128 /\ dots == dots7_s qn kvo c /\ h < 8 /\ p < 8 /\
            S.length (keys_s kvo c h) == 4096 /\ S.length (ubs 256 qn (h * 256)) == 2048 /\
            S.index (to_seq (row_major 16 8) (mma_model (from_seq (row_major 16 256) (keys_s kvo c h))
                       (from_seq (col_major 256 8) (ubs 256 qn (h * 256))) 0 16)) (p * 8)
              %~ vsum (16 * (16 - 0)) (fun kk -> cp_ra kn (base_hk c rkv) t h (p * 256 + 16 * 0 + kk) *. cp_rb rq h (16 * 0 + kk)))
  (ensures bat dots (h * 16 + p) %~ cp_rd rq kn (base_hk c rkv) t h p)
  = let sa = keys_s kvo c h in
    let sb = ubs 256 qn (h * 256) in
    cp_mmo_eq sa sb;
    cp_len (mma_model (from_seq (row_major 16 256) sa) (from_seq (col_major 256 8) sb) 0 16);
    cp_divmod16 h p;
    assert (S.index dots (h * 16 + p) == tb (fat (mmo 16 256 sa sb 0 16) (p * 8)));
    introduce forall (kk:nat). kk < 16 * (16 - 0) ==>
      cp_ra kn (base_hk c rkv) t h (p * 256 + 16 * 0 + kk) *. cp_rb rq h (16 * 0 + kk) == cp_sf rq kn (base_hk c rkv) t h p kk
    with introduce _ ==> _ with cp_row rq kn (base_hk c rkv) t h p kk;
    vsum_ext (16 * (16 - 0)) (fun kk -> cp_ra kn (base_hk c rkv) t h (p * 256 + 16 * 0 + kk) *. cp_rb rq h (16 * 0 + kk))
      (cp_sf rq kn (base_hk c rkv) t h p);
    su_tb (fat (mmo 16 256 sa sb 0 16) (p * 8)) (cp_rd rq kn (base_hk c rkv) t h p)

let cp_keys_all (kvo:S.seq bf16) (c t:nat) (rkv kn vn:S.seq real) (h:nat) : Lemma
  (requires S.length kvo == c * 1024 /\ S.length rkv == c * 1024 /\ 8 <= c /\ t == 7 /\ h < 8 /\
            kvo %~ base_kv_write c t rkv kn vn)
  (ensures forall (idx:nat). idx < 16 * 256 ==> S.index (keys_s kvo c h) idx %~ cp_ra kn (base_hk c rkv) t h idx)
  = introduce forall (idx:nat). idx < 16 * 256 ==> S.index (keys_s kvo c h) idx %~ cp_ra kn (base_hk c rkv) t h idx
    with introduce _ ==> _ with cp_keys_ok kvo c t rkv kn vn h idx

let cp_ub_all (qn:S.seq bf16) (rq:S.seq real) (h:nat) : Lemma
  (requires S.length qn == 2048 /\ qn %~ rq /\ h < 8)
  (ensures forall (kk c':nat). kk < 256 /\ c' < 8 ==> S.index (ubs 256 qn (h * 256)) (c' * 256 + kk) %~ cp_rb rq h kk)
  = introduce forall (kk:nat) (c':nat). kk < 256 /\ c' < 8 ==> S.index (ubs 256 qn (h * 256)) (c' * 256 + kk) %~ cp_rb rq h kk
    with introduce _ ==> _ with cp_ub_ok qn rq h kk c'

let cp_caseB (qn kvo:S.seq bf16) (c t:nat) (rkv kn vn rq:S.seq real) (dots:S.seq bf16) (h p:nat) : Lemma
  (requires S.length qn == 2048 /\ S.length kvo == c * 1024 /\ S.length rkv == c * 1024 /\ c <= 12 /\ t < c /\
            t == 7 /\ qn %~ rq /\ kvo %~ base_kv_write c t rkv kn vn /\ dots_good qn kvo c t dots /\ h < 8 /\ p <= t)
  (ensures bat dots (h * 16 + p) %~ cp_rd rq kn (base_hk c rkv) t h p)
  = cp_keys_all kvo c t rkv kn vn h;
    cp_ub_all qn rq h;
    mma_col0_approx 16 256 0 16 (keys_s kvo c h) (ubs 256 qn (h * 256)) (cp_ra kn (base_hk c rkv) t h) (cp_rb rq h);
    cp_caseB1 qn kvo c t rkv kn rq dots h p
#pop-options

let cp_probs_at (dots:S.seq bf16) (t h p:nat) : Lemma
  (requires h < 8 /\ p < 16)
  (ensures S.index (probs_s dots t) (h * 16 + p) == prob_m dots t h p)
  = cp_divmod16 h p

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let cp_one (qn kvo:S.seq bf16) (c t:nat) (rw rx rkv rc rs:S.seq real) (dots:S.seq bf16) (h p:nat) : Lemma
  (requires S.length qn == 2048 /\ S.length kvo == c * 1024 /\ S.length rkv == c * 1024 /\ 1 <= c /\ c <= 12 /\ t < c /\
            qn %~ at_q base_numerics rw 0 rx rc rs /\
            kvo %~ base_kv_write c t rkv (at_k base_numerics rw 0 rx rc rs) (at_v base_numerics rw 0 rx) /\
            dots_good qn kvo c t dots /\ h < 8 /\ p <= t)
  (ensures S.index (probs_s dots t) (h * 16 + p) %~ at_prob base_numerics rw 0 rx rc rs t (base_hk c rkv) h p)
  = let rq = at_q base_numerics rw 0 rx rc rs in
    let kn = at_k base_numerics rw 0 rx rc rs in
    let vn = at_v base_numerics rw 0 rx in
    introduce forall (p':nat). p' <= t ==> bat dots (h * 16 + p') %~ cp_rd rq kn (base_hk c rkv) t h p'
    with introduce _ ==> _ with
      (if t = 7 then cp_caseB qn kvo c t rkv kn vn rq dots h p' else cp_caseA qn kvo c t rkv kn vn rq dots h p');
    prob_approx dots t h (cp_rd rq kn (base_hk c rkv) t h);
    introduce forall (p':nat). p' <= t ==>
      at_score base_numerics rw 0 rx rc rs t (base_hk c rkv) h p' == cp_rd rq kn (base_hk c rkv) t h p' *. 0.0625R
    with introduce _ ==> _ with cp_score_eq base_numerics rw 0 rx rc rs t (base_hk c rkv) h p';
    cp_prob_eq base_numerics rw 0 rx rc rs t (base_hk c rkv) h p (cp_rd rq kn (base_hk c rkv) t h);
    cp_probs_at dots t h p
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let probs_ok (qn kvo:S.seq bf16) (c t:nat) (rw rx rkv rc rs:S.seq real) (dots:S.seq bf16) : Lemma
  (requires S.length qn == 2048 /\ S.length kvo == c * 1024 /\ S.length rkv == c * 1024 /\ 1 <= c /\ c <= 12 /\ t < c /\
            (forall (j:nat). j < 2048 ==> S.index qn j %~ at (at_q base_numerics rw 0 rx rc rs) j) /\
            kvo %~ base_kv_write c t rkv (at_k base_numerics rw 0 rx rc rs) (at_v base_numerics rw 0 rx) /\
            dots_good qn kvo c t dots)
  (ensures forall (h p:nat). h < 8 /\ p <= t ==>
            S.index (probs_s dots t) (h * 16 + p) %~ at_prob base_numerics rw 0 rx rc rs t (base_hk c rkv) h p)
  = cp_q_approx qn (at_q base_numerics rw 0 rx rc rs);
    introduce forall (h:nat) (p:nat). h < 8 /\ p <= t ==>
      S.index (probs_s dots t) (h * 16 + p) %~ at_prob base_numerics rw 0 rx rc rs t (base_hk c rkv) h p
    with introduce _ ==> _ with cp_one qn kvo c t rw rx rkv rc rs dots h p
#pop-options
(*** Composition: gated attention output approximates at_gated *)

#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"

let cg_dm (a k:int) (n:pos) : Lemma (requires 0 <= a /\ a < n) (ensures (a + k * n) / n == k /\ (a + k * n) % n == a)
  = FStar.Math.Lemmas.lemma_div_plus a k n;
    FStar.Math.Lemmas.lemma_mod_plus a k n;
    FStar.Math.Lemmas.small_div a n;
    FStar.Math.Lemmas.small_mod a n

let cg_kv_idx (c g p e:nat) : Lemma
  (requires 1 <= c /\ g < 2 /\ p < c /\ e < 256)
  (ensures (let m = (g * c + p) * 256 + e in
            m < c * 512 /\ (c * 512 + m) % (c * 512) == m /\
            m % (c * 256) == p * 256 + e /\ m / (c * 256) == g /\
            (p * 256 + e) / 256 == p /\ (c * 512 + m) % 256 == e /\
            (g * 256 + e) / 256 == g /\ (g * 256 + e) % 256 == e))
  = let m = (g * c + p) * 256 + e in
    let q = p * 256 + e in
    assert (m == q + g * (c * 256));
    assert (q < c * 256);
    cg_dm q g (c * 256);
    assert (m < c * 512);
    cg_dm m 1 (c * 512);
    cg_dm e p 256;
    assert (c * 512 + m == e + (2 * c + g * c + p) * 256);
    cg_dm e (2 * c + g * c + p) 256;
    cg_dm e g 256

let cg_hist_lt (hh:nat -> nat -> real) (cur:S.seq real) (t p j:nat) : Lemma
  (requires p < t) (ensures hist_or hh cur t p j == hh p j) = ()

let cg_hist_eq (hh:nat -> nat -> real) (cur:S.seq real) (t p j:nat) : Lemma
  (requires p == t) (ensures hist_or hh cur t p j == at cur j) = ()

let cg_hv (c:nat) (rkv:S.seq real) (p j:nat) : Lemma
  (base_hv c rkv p j == at rkv (c * 512 + base_kv_index c p j)) = ()

#push-options "--z3rlimit 40"
let cg_kv (kvo:S.seq bf16) (rkv kn vn:S.seq real) (c t h p e:nat) : Lemma
  (requires S.length rkv == c * 1024 /\ kvo %~ base_kv_write c t rkv kn vn /\
            1 <= c /\ h < 8 /\ p <= t /\ t < c /\ e < 256)
  (ensures bat kvo (c * 512 + ((h / 4) * c + p) * 256 + e) %~ hist_or (base_hv c rkv) vn t p ((h / 4) * 256 + e))
  = let g = h / 4 in
    cg_kv_idx c g p e;
    let i = c * 512 + (g * c + p) * 256 + e in
    let w = base_kv_write c t rkv kn vn in
    assert (S.length w == c * 1024);
    assert (i < S.length kvo);
    assert (S.index kvo i %~ S.index w i);
    assert (base_kv_pos c i == p);
    assert (base_kv_comp c i == g * 256 + e);
    assert (base_kv_index c p (g * 256 + e) == (g * c + p) * 256 + e);
    assert (bat kvo i == S.index kvo i);
    if p = t then begin
      assert (S.index w i == at vn (g * 256 + e));
      cg_hist_eq (base_hv c rkv) vn t p (g * 256 + e)
    end else begin
      assert (S.index w i == at rkv i);
      cg_hv c rkv p (g * 256 + e);
      cg_hist_lt (base_hv c rkv) vn t p (g * 256 + e)
    end
#pop-options

#pop-options

noextract
let cg_mul (f g:nat -> real) (p:nat) : real = f p *. g p
noextract
let cg_rvf (rkv vn:S.seq real) (c t h e:nat) (p:nat) : real = hist_or (base_hv c rkv) vn t p ((h / 4) * 256 + e)
noextract
let cg_ra (rkv vn:S.seq real) (c t h:nat) (idx:nat) : real =
  if idx % 16 < 8 then hist_or (base_hv c rkv) vn t (idx % 16) ((h / 4) * 256 + idx / 16) else 0.0R
noextract
let cg_rb (rp:nat -> real) (t:nat) (kk:nat) : real = if kk <= t then rp kk else 0.0R
noextract
let cg_mmf (ra rb:nat -> real) (k j0 i:nat) (kk:nat) : real = ra (i * k + 16 * j0 + kk) *. rb (16 * j0 + kk)

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"

let cg_mma (n k:pos) (j0:nat) (j1:nat) (sa sb:S.seq bf16) (ra rb:nat -> real) : Lemma
  (requires 16 /? n /\ 16 /? k /\ j0 <= j1 /\ j1 <= k / 16 /\ S.length sa == n * k /\ S.length sb == k * 8 /\
            (forall (idx:nat). idx < n * k ==> S.index sa idx %~ ra idx) /\
            (forall (kk c:nat). kk < k /\ c < 8 ==> S.index sb (c * k + kk) %~ rb kk))
  (ensures forall (i:nat). i < n ==>
     S.index (to_seq (row_major n 8) (mma_model (from_seq (row_major n k) sa) (from_seq (col_major k 8) sb) j0 j1)) (i * 8)
       %~ vsum (16 * (j1 - j0)) (cg_mmf ra rb k j0 i))
  = mma_col0_approx n k j0 j1 sa sb ra rb;
    introduce forall (i:nat). i < n ==>
     S.index (to_seq (row_major n 8) (mma_model (from_seq (row_major n k) sa) (from_seq (col_major k 8) sb) j0 j1)) (i * 8)
       %~ vsum (16 * (j1 - j0)) (cg_mmf ra rb k j0 i)
    with introduce _ ==> _
    with vsum_ext (16 * (j1 - j0)) (fun kk -> ra (i * k + 16 * j0 + kk) *. rb (16 * j0 + kk)) (cg_mmf ra rb k j0 i)

#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"

let cg_vals_elim (probs kvo:S.seq bf16) (c t:nat) (vals:S.seq f32) (h i:nat) (rp rv:nat -> real) : Lemma
  (requires vals_ok probs kvo c t vals /\ h < 8 /\ i < 256 /\
            (forall (p:nat). p <= t ==> pr probs h p %~ rp p /\ vv kvo c h p i %~ rv p))
  (ensures S.index vals (h * 256 + i) %~ vsum (t + 1) (cg_mul rp rv))
  = assert (S.index vals (h * 256 + i) %~ vsum (t + 1) (fun p -> rp p *. rv p));
    vsum_ext (t + 1) (fun p -> rp p *. rv p) (cg_mul rp rv)

noextract
let cg_gp (nm:numerics) (w:S.seq real) (wo:int) (x cs sn:S.seq real) (t:nat)
          (hk hv:nat -> nat -> real) (hd e:nat) (p:nat) : real =
  at_prob nm w wo x cs sn t hk hd p *. hist_or hv (at_v nm w wo x) t p ((hd / 4) * 256 + e)

let cg_gated_at (nm:numerics) (w:S.seq real) (wo:int) (x cs sn:S.seq real) (t:nat)
             (hk hv:nat -> nat -> real) (c:nat{c < 2048}) : Lemma
  (at (at_gated nm w wo x cs sn t hk hv) c ==
     vsum (t + 1) (cg_gp nm w wo x cs sn t hk hv (c / 256) (c % 256)) *. sigmoid (at (at_qraw nm w wo x) ((c / 256) * 512 + 256 + c % 256)))
  = introduce forall (f:nat -> real). (forall (i:nat). i < t + 1 ==> f i == cg_gp nm w wo x cs sn t hk hv (c / 256) (c % 256) i) ==>
        vsum (t + 1) f == vsum (t + 1) (cg_gp nm w wo x cs sn t hk hv (c / 256) (c % 256))
    with introduce _ ==> _
    with vsum_ext (t + 1) f (cg_gp nm w wo x cs sn t hk hv (c / 256) (c % 256))

#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let cg_mmo_entry (sa sb:S.seq bf16) (e:nat) : Lemma
  (requires S.length sa == 256 * 16 /\ S.length sb == 16 * 8 /\ e < 256)
  (ensures fat (mmo 256 16 sa sb 0 1) (e * 8) ==
           S.index (to_seq (row_major 256 8) (mma_model (from_seq (row_major 256 16) sa) (from_seq (col_major 16 8) sb) 0 1)) (e * 8))
  = mm_to_rm 256 8 (mma_model (from_seq (row_major 256 16) sa) (from_seq (col_major 16 8) sb) 0 1) e 0

#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"

let cg_zero_bf16 () : Lemma ((zero #bf16) %~ 0.0R) = a0 #bf16

let cg_sa (kvo:S.seq bf16) (rkv kn vn:S.seq real) (c h:nat) : Lemma
  (requires S.length rkv == c * 1024 /\ kvo %~ base_kv_write c 7 rkv kn vn /\ 8 <= c /\ h < 8)
  (ensures forall (idx:nat). idx < 256 * 16 ==> S.index (valm_s kvo c h) idx %~ cg_ra rkv vn c 7 h idx)
  = introduce forall (idx:nat). idx < 256 * 16 ==> S.index (valm_s kvo c h) idx %~ cg_ra rkv vn c 7 h idx
    with introduce _ ==> _
    with (if idx % 16 < 8 then cg_kv kvo rkv kn vn c 7 h (idx % 16) (idx / 16) else cg_zero_bf16 ())

let cg_sb (probs:S.seq bf16) (rp:nat -> real) (h:nat) : Lemma
  (requires S.length probs == 128 /\ h < 8 /\
            (forall (p:nat). p <= 7 ==> S.index probs (h * 16 + p) %~ rp p) /\
            (forall (p:nat). 7 < p /\ p < 16 ==> S.index probs (h * 16 + p) == zero))
  (ensures forall (kk cc:nat). kk < 16 /\ cc < 8 ==> S.index (ubs 16 probs (h * 16)) (cc * 16 + kk) %~ cg_rb rp 7 kk)
  = introduce forall (kk cc:nat). kk < 16 /\ cc < 8 ==> S.index (ubs 16 probs (h * 16)) (cc * 16 + kk) %~ cg_rb rp 7 kk
    with introduce _ ==> _
    with (cg_dm kk cc 16; cg_zero_bf16 ())

let cg_pt (rkv vn:S.seq real) (c h e:nat) (rp:nat -> real) (kk:nat) : Lemma
  (requires kk < 8)
  (ensures cg_mmf (cg_ra rkv vn c 7 h) (cg_rb rp 7) 16 0 e kk == cg_mul rp (cg_rvf rkv vn c 7 h e) kk)
  = let idx = e * 16 + 16 * 0 + kk in
    assert (idx == kk + e * 16);
    cg_dm kk e 16;
    assert (idx % 16 == kk /\ idx / 16 == e);
    assert (cg_ra rkv vn c 7 h idx == hist_or (base_hv c rkv) vn 7 kk ((h / 4) * 256 + e));
    assert (cg_rb rp 7 (16 * 0 + kk) == rp kk)

let cg_vals7 (probs kvo:S.seq bf16) (rkv kn vn:S.seq real) (c h e:nat) (rp:nat -> real) : Lemma
  (requires S.length rkv == c * 1024 /\ kvo %~ base_kv_write c 7 rkv kn vn /\ 8 <= c /\ h < 8 /\ e < 256 /\
            S.length probs == 128 /\
            (forall (p:nat). p <= 7 ==> S.index probs (h * 16 + p) %~ rp p) /\
            (forall (p:nat). 7 < p /\ p < 16 ==> S.index probs (h * 16 + p) == zero))
  (ensures fat (mmo 256 16 (valm_s kvo c h) (ubs 16 probs (h * 16)) 0 1) (e * 8) %~ vsum (7 + 1) (cg_mul rp (cg_rvf rkv vn c 7 h e)))
  = let sa = valm_s kvo c h in
    let sb = ubs 16 probs (h * 16) in
    let ra = cg_ra rkv vn c 7 h in
    let rb = cg_rb rp 7 in
    cg_sa kvo rkv kn vn c h;
    cg_sb probs rp h;
    cg_mma 256 16 0 1 sa sb ra rb;
    cg_mmo_entry sa sb e;
    assert (fat (mmo 256 16 sa sb 0 1) (e * 8) %~ vsum (16 * (1 - 0)) (cg_mmf ra rb 16 0 e));
    vsum_zero_tail 8 16 (cg_mmf ra rb 16 0 e);
    introduce forall (kk:nat). kk < 8 ==> cg_mmf ra rb 16 0 e kk == cg_mul rp (cg_rvf rkv vn c 7 h e) kk
    with introduce _ ==> _
    with cg_pt rkv vn c h e rp kk;
    vsum_ext 8 (cg_mmf ra rb 16 0 e) (cg_mul rp (cg_rvf rkv vn c 7 h e))

let cg_valsn (probs kvo:S.seq bf16) (rkv kn vn:S.seq real) (c t:nat) (vals:S.seq f32) (h e:nat) (rp:nat -> real) : Lemma
  (requires S.length rkv == c * 1024 /\ kvo %~ base_kv_write c t rkv kn vn /\ 1 <= c /\ c <= 12 /\ t < c /\
            h < 8 /\ e < 256 /\ S.length probs == 128 /\ vals_ok probs kvo c t vals /\
            (forall (p:nat). p <= t ==> S.index probs (h * 16 + p) %~ rp p))
  (ensures S.index vals (h * 256 + e) %~ vsum (t + 1) (cg_mul rp (cg_rvf rkv vn c t h e)))
  = introduce forall (p:nat). p <= t ==> pr probs h p %~ rp p /\ vv kvo c h p e %~ cg_rvf rkv vn c t h e p
    with introduce _ ==> _
    with begin
      su_fb (bat probs (h * 16 + p)) (rp p);
      cg_kv kvo rkv kn vn c t h p e;
      su_fb (bat kvo (c * 512 + ((h / 4) * c + p) * 256 + e)) (cg_rvf rkv vn c t h e p)
    end;
    cg_vals_elim probs kvo c t vals h e rp (cg_rvf rkv vn c t h e)

#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 60"

let cg_one (pq kvo:S.seq bf16) (c t:nat) (rw rx rkv rc rs:S.seq real) (probs:S.seq bf16) (vals:S.seq f32) (j:nat) : Lemma
  (requires S.length pq == 4096 /\ S.length kvo == c * 1024 /\ S.length rkv == c * 1024 /\ S.length probs == 128 /\
            1 <= c /\ c <= 12 /\ t < c /\ j < 2048 /\
            (forall (j:nat). j < 4096 ==> S.index pq j %~ at (at_qraw base_numerics rw 0 rx) j) /\
            kvo %~ base_kv_write c t rkv (at_k base_numerics rw 0 rx rc rs) (at_v base_numerics rw 0 rx) /\
            (forall (h p:nat). h < 8 /\ p <= t ==> S.index probs (h * 16 + p) %~ at_prob base_numerics rw 0 rx rc rs t (base_hk c rkv) h p) /\
            (forall (h p:nat). h < 8 /\ t < p /\ p < 16 ==> S.index probs (h * 16 + p) == zero) /\
            vals_good probs kvo c t vals)
  (ensures S.index (gated_s vals pq) j %~ at (at_gated base_numerics rw 0 rx rc rs t (base_hk c rkv) (base_hv c rkv)) j)
  = let h = j / 256 in
    let e = j % 256 in
    let kn = at_k base_numerics rw 0 rx rc rs in
    let vn = at_v base_numerics rw 0 rx in
    let rp = at_prob base_numerics rw 0 rx rc rs t (base_hk c rkv) h in
    let rv = vsum (t + 1) (cg_mul rp (cg_rvf rkv vn c t h e)) in
    let rg = at (at_qraw base_numerics rw 0 rx) (h * 512 + 256 + e) in
    assert (j == h * 256 + e);
    (if t = 7 then cg_vals7 probs kvo rkv kn vn c h e rp
     else cg_valsn probs kvo rkv kn vn c t vals h e rp);
    assert (fat vals j %~ rv);
    assert (bat pq (h * 512 + 256 + e) %~ rg);
    gate_approx vals pq j rv rg;
    cg_gated_at base_numerics rw 0 rx rc rs t (base_hk c rkv) (base_hv c rkv) j;
    vsum_ext (t + 1) (cg_gp base_numerics rw 0 rx rc rs t (base_hk c rkv) (base_hv c rkv) h e) (cg_mul rp (cg_rvf rkv vn c t h e))

let gated_ok (pq kvo:S.seq bf16) (c t:nat) (rw rx rkv rc rs:S.seq real) (probs:S.seq bf16) (vals:S.seq f32) : Lemma
  (requires S.length pq == 4096 /\ S.length kvo == c * 1024 /\ S.length rkv == c * 1024 /\ S.length probs == 128 /\
            1 <= c /\ c <= 12 /\ t < c /\
            (forall (j:nat). j < 4096 ==> S.index pq j %~ at (at_qraw base_numerics rw 0 rx) j) /\
            kvo %~ base_kv_write c t rkv (at_k base_numerics rw 0 rx rc rs) (at_v base_numerics rw 0 rx) /\
            (forall (h p:nat). h < 8 /\ p <= t ==> S.index probs (h * 16 + p) %~ at_prob base_numerics rw 0 rx rc rs t (base_hk c rkv) h p) /\
            (forall (h p:nat). h < 8 /\ t < p /\ p < 16 ==> S.index probs (h * 16 + p) == zero) /\
            vals_good probs kvo c t vals)
  (ensures forall (j:nat). j < 2048 ==>
            S.index (gated_s vals pq) j %~ at (at_gated base_numerics rw 0 rx rc rs t (base_hk c rkv) (base_hv c rkv)) j)
  = introduce forall (j:nat). j < 2048 ==>
            S.index (gated_s vals pq) j %~ at (at_gated base_numerics rw 0 rx rc rs t (base_hk c rkv) (base_hv c rkv)) j
    with introduce _ ==> _
    with cg_one pq kvo c t rw rx rkv rc rs probs vals j

#pop-options

(*** Final combination: pipeline values satisfy the family relation *)

let fin_frame (sw sx skv sc ss:S.seq bf16) (c t:int) : Lemma
  (forall (i:nat). i < S.length skv /\ base_kv_pos c i <> t ==> S.index (p_kvo sw sx skv sc ss c t) i == S.index skv i) = ()

let fin_c (c:nat) : Lemma ((c * 1024) / 1024 == c) = FStar.Math.Lemmas.cancel_mul_div c 1024

let fin_hp (h p:nat) : Lemma (requires p < 16) (ensures (h * 16 + p) / 16 == h /\ (h * 16 + p) % 16 == p) =
  FStar.Math.Lemmas.lemma_div_plus p h 16;
  FStar.Math.Lemmas.lemma_mod_plus p h 16;
  FStar.Math.Lemmas.small_div p 16;
  FStar.Math.Lemmas.small_mod p 16

let fin_zero (dots:S.seq bf16) (t:nat) : Lemma
  (forall (h p:nat). h < 8 /\ t < p /\ p < 16 ==> S.index (probs_s dots t) (h * 16 + p) == zero)
  = let aux (h p:nat) : Lemma (h < 8 /\ t < p /\ p < 16 ==> S.index (probs_s dots t) (h * 16 + p) == zero) =
      if h < 8 && t < p && p < 16 then fin_hp h p else ()
    in
    Classical.forall_intro_2 aux

#push-options "--z3rlimit 40"
let final_ok0 (sw sx skv sc ss:S.seq bf16) (c t:nat) (dots:S.seq bf16) (vals:S.seq f32) (xo kvo:S.seq bf16) : Lemma
  (requires S.length sw == 18352640 /\ S.length sx == 1024 /\ S.length skv == c * 1024 /\ 1 <= c /\ c <= 12 /\ t < c /\
     S.length sc == 32 /\ S.length ss == 32 /\
     kvo == p_kvo sw sx skv sc ss c t /\
     dots_good (p_qn sw sx sc ss) kvo c t dots /\
     vals_good (probs_s dots t) kvo c t vals /\
     xo == b_xo sw sx (gated_s vals (p_pq sw sx)))
  (ensures base_full_attention_decoder_relation sw sx skv sc ss t xo kvo)
  = fin_c c;
    fin_frame sw sx skv sc ss c t;
    let qn = p_qn sw sx sc ss in
    let pq = p_pq sw sx in
    let probs = probs_s dots t in
    let gated = gated_s vals pq in
    fin_zero dots t;
    introduce forall (rw rx rkv rc rs:S.seq real).
      (sw %~ rw /\ sx %~ rx /\ skv %~ rkv /\ sc %~ rc /\ ss %~ rs) ==>
      (xo %~ at_out base_numerics rw 0 rx rc rs t (base_hk c rkv) (base_hv c rkv) /\
       kvo %~ base_kv_write c t rkv (at_k base_numerics rw 0 rx rc rs) (at_v base_numerics rw 0 rx))
    with introduce _ ==> _ with (
      front_ok sw sx skv sc ss c t rw rx rkv rc rs;
      probs_ok qn kvo c t rw rx rkv rc rs dots;
      gated_ok pq kvo c t rw rx rkv rc rs probs vals;
      let g = at_gated base_numerics rw 0 rx rc rs t (base_hk c rkv) (base_hv c rkv) in
      back_ok sw sx gated rw rx g
    )
#pop-options

noextract
let fin_pre (sw sx skv sc ss:S.seq bf16) (c t:nat) (dots:S.seq bf16) (vals:S.seq f32) (xo kvo:S.seq bf16) : prop =
  S.length sw == 18352640 /\ S.length sx == 1024 /\ S.length skv == c * 1024 /\ 1 <= c /\ c <= 12 /\ t < c /\
  S.length sc == 32 /\ S.length ss == 32 /\
  kvo == p_kvo sw sx skv sc ss c t /\
  dots_good (p_qn sw sx sc ss) kvo c t dots /\
  vals_good (probs_s dots t) kvo c t vals /\
  xo == b_xo sw sx (gated_s vals (p_pq sw sx))

let final_aux (sw sx skv sc ss:S.seq bf16) (c t:nat) (xo kvo:S.seq bf16) (dots:S.seq bf16) (vals:S.seq f32) : Lemma
  (fin_pre sw sx skv sc ss c t dots vals xo kvo ==> base_full_attention_decoder_relation sw sx skv sc ss t xo kvo)
  = Classical.move_requires (final_ok0 sw sx skv sc ss c t dots vals xo) kvo

#push-options "--z3rlimit 40"
let final_ok (sw sx skv sc ss:S.seq bf16) (c t:nat) (xo kvo:S.seq bf16) : Lemma
  (requires S.length sw == 18352640 /\ S.length sx == 1024 /\ S.length skv == c * 1024 /\ 1 <= c /\ c <= 12 /\ t < c /\
     S.length sc == 32 /\ S.length ss == 32 /\ pipe_ok sw sx skv sc ss c t xo kvo)
  (ensures base_full_attention_decoder_relation sw sx skv sc ss t xo kvo)
  = Classical.exists_elim (base_full_attention_decoder_relation sw sx skv sc ss t xo kvo)
      #(S.seq bf16)
      #(fun dots -> exists (vals:S.seq f32). dots_good (p_qn sw sx sc ss) kvo c t dots /\
          vals_good (probs_s dots t) kvo c t vals /\ xo == b_xo sw sx (gated_s vals (p_pq sw sx)))
      ()
      (fun dots -> Classical.exists_elim (base_full_attention_decoder_relation sw sx skv sc ss t xo kvo)
        #(S.seq f32)
        #(fun vals -> dots_good (p_qn sw sx sc ss) kvo c t dots /\
          vals_good (probs_s dots t) kvo c t vals /\ xo == b_xo sw sx (gated_s vals (p_pq sw sx)))
        ()
        (fun vals -> final_ok0 sw sx skv sc ss c t dots vals xo kvo))
#pop-options
let nkv_div (n:nat) : Lemma (requires n % 1024 = 0) (ensures n == (n / 1024) * 1024)
  = FStar.Math.Lemmas.euclidean_division_definition n 1024

#push-options "--z3rlimit 40"
let vacuous_rel (sw sx skv sc ss x' kv':S.seq bf16) (t:int) (nkv:nat) : Lemma
  (requires S.length skv == nkv /\
    ~(nkv % 1024 = 0 /\ 1 <= nkv / 1024 /\ nkv / 1024 <= 12 /\ 0 <= t /\ t < 12 /\ t < nkv / 1024))
  (ensures base_full_attention_decoder_relation sw sx skv sc ss t x' kv')
  = ()

fn kplay_run
  (nkv : sz)
  (w : larray bf16 18352640 { is_global w })
  (x : larray bf16 1024 { is_global x })
  (kv : larray bf16 (SZ.v nkv) { is_global kv })
  (cs : larray bf16 32 { is_global cs })
  (sn : larray bf16 32 { is_global sn })
  (t : i64)
  (xo : larray bf16 1024 { is_global xo })
  (kvo : larray bf16 (SZ.v nkv) { is_global kvo })
  (#pw #px #pkv #pc #ps : perm)
  (#sw #sx #skv #sc #ss : erased (S.seq bf16))
  (#ox #okv : erased (S.seq bf16))
  preserves
    cpu **
    on gpu_loc (w |-> Frac pw sw) **
    on gpu_loc (x |-> Frac px sx) **
    on gpu_loc (kv |-> Frac pkv skv) **
    on gpu_loc (cs |-> Frac pc sc) **
    on gpu_loc (sn |-> Frac ps ss)
  requires
    on gpu_loc (xo |-> ox) **
    on gpu_loc (kvo |-> okv)
  ensures
    exists* (x' kv' : S.seq bf16).
      on gpu_loc (xo |-> x') **
      on gpu_loc (kvo |-> kv') **
      pure (base_full_attention_decoder_relation sw sx skv sc ss (I64.v t) x' kv')
{
  gpu_len w; gpu_len x; gpu_len kv; gpu_len cs; gpu_len sn;
  let c = nkv /^ 1024sz;
  if (nkv %^ 1024sz = 0sz && 1sz <=^ c && c <=^ 12sz && I64.lte 0L t && I64.lt t 12L) {
    nkv_div (SZ.v nkv);
    assert pure (SZ.v nkv == SZ.v c * 1024);
    let ts = i64_to_sz t;
    if (ts <^ c) {
      run_main nkv c ts w x kv cs sn xo kvo;
      with x' kv'. assert (on gpu_loc (xo |-> x') ** on gpu_loc (kvo |-> kv'));
      final_ok sw sx skv sc ss (SZ.v c) (SZ.v ts) x' kv';
      ()
    } else {
      vacuous_rel sw sx skv sc ss ox okv (I64.v t) (SZ.v nkv);
      ()
    }
  } else {
    vacuous_rel sw sx skv sc ss ox okv (I64.v t) (SZ.v nkv);
    ()
  }
}
#pop-options
