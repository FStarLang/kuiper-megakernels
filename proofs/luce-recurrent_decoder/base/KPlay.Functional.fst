module KPlay.Functional

#lang-pulse

module S = FStar.Seq
module SZ = Kuiper.SizeT
module M = Kuiper.TensorCore.MMA
module RO = Kuiper.Float.Realops
module C = Kuiper.Float.Casts
module CF = Kuiper.Float32.ConstantFold
module Cert = Kuiper.Float32.ConstantFold.Certificates
module Lit = Kuiper.Float32.Literals
module Exp = FStar.Math.Exp
module Sqrt = FStar.Math.Sqrt
module L = FStar.Math.Lemmas
module SP = Luce.Softplus
module FM = Kuiper.Float32.FastMath
module A = Pulse.Lib.Array

(* ===== Views ===== *)
open Kuiper
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Bijection



#push-options "--z3rlimit 40 --fuel 0 --ifuel 1"

(* 16 x nc tile views over a flat array: element (a,b), a < 16, b < nc, lives at
   off + ld*a + b (row-major, cm=false) or off + ld*b + a (col-major, cm=true). *)
noextract
let tno (cm : bool) (nc : pos) : pos = if cm then nc else 16
noextract
let tni (cm : bool) (nc : pos) : pos = if cm then 16 else nc

noextract
let tix (cm : bool) (nc : pos) (off : nat) (ld : pos) (a : natlt 16) (b : natlt nc) : nat =
  if cm then off + ld * b + a else off + ld * a + b

noextract
let tile_ok (cm : bool) (nc : pos) (off ld n : nat) : prop =
  tni cm nc <= ld /\ off + ld * (tno cm nc - 1) + tni cm nc <= n

noextract
let in_tile (cm : bool) (nc : pos) (off : nat) (ld : pos) (i : nat) : bool =
  off <= i && (i - off) / ld < tno cm nc && (i - off) % ld < tni cm nc

noextract
let divmod_lemma (ld : pos) (a : nat) (b : nat{b < ld})
  : Lemma ((ld * a + b) / ld == a /\ (ld * a + b) % ld == b)
  = FStar.Math.Lemmas.lemma_div_plus b a ld;
    FStar.Math.Lemmas.small_div b ld;
    FStar.Math.Lemmas.lemma_mod_plus b a ld;
    FStar.Math.Lemmas.small_mod b ld

noextract
let tix_inv (cm : bool) (nc : pos) (off : nat) (ld : pos{ld >= tni cm nc}) (a : natlt 16) (b : natlt nc)
  : Lemma (let i = tix cm nc off ld a b in
           in_tile cm nc off ld i /\
           (if cm then (i - off) % ld == a /\ (i - off) / ld == b
                  else (i - off) / ld == a /\ (i - off) % ld == b))
  = if cm then divmod_lemma ld b a else divmod_lemma ld a b

noextract
let tix_back (cm : bool) (nc : pos) (off : nat) (ld : pos) (i : nat{in_tile cm nc off ld i})
  : Lemma (let a : natlt 16 = if cm then (i - off) % ld else (i - off) / ld in
           let b : natlt nc = if cm then (i - off) / ld else (i - off) % ld in
           tix cm nc off ld a b == i)
  = FStar.Math.Lemmas.euclidean_division_definition (i - off) ld

noextract
let tlay (cm : bool) (nc : pos) (off : nat) (ld : pos) (n : nat{tile_ok cm nc off ld n}) : layout2 16 nc =
  { ulen = n;
    imap = {
      f = (fun (x : abs (16 @| nc @| INil)) ->
             let (a, (b, ())) = x in (tix cm nc off ld a b <: natlt n));
      is_inj = (fun x y ->
        let (a, (b, ())) = x in
        let (c, (d, ())) = y in
        tix_inv cm nc off ld a b; tix_inv cm nc off ld c d) } }

inline_for_extraction noextract
instance srm_tlay (nc : pos) (off : sz) (ld : sz) (n : nat{tile_ok false nc (SZ.v off) (SZ.v ld) n})
  : strided_row_major (tlay false nc (SZ.v off) (SZ.v ld) n)
  = { offset = off; stride = ld; pf = (fun i j -> ()) }

inline_for_extraction noextract
instance scm_tlay (nc : pos) (off : sz) (ld : sz) (n : nat{tile_ok true nc (SZ.v off) (SZ.v ld) n})
  : strided_col_major (tlay true nc (SZ.v off) (SZ.v ld) n)
  = { offset = off; stride = ld; pf = (fun i j -> ()) }

noextract
let tile_val (#et : Type) (cm : bool) (nc : pos) (off : nat) (ld : pos) (s : S.seq et)
  (_ : squash (tile_ok cm nc off ld (S.length s)))
  : chest2 et 16 nc
  = mk2 (fun a b -> S.index s (tix cm nc off ld a b))

noextract
let upd_tile (#et : Type) (cm : bool) (nc : pos) (off : nat) (ld : pos) (s : S.seq et) (m : chest2 et 16 nc)
  : GTot (S.seq et)
  = S.init_ghost (S.length s) (fun i ->
      if in_tile cm nc off ld i
      then (tix_back cm nc off ld i;
            acc2 m (if cm then (i - off) % ld else (i - off) / ld)
                   (if cm then (i - off) / ld else (i - off) % ld))
      else S.index s i)

noextract
let upd_tile_lemmas (#et : Type) (cm : bool) (nc : pos) (off : nat) (ld : pos) (s : S.seq et) (m : chest2 et 16 nc)
  : Lemma (requires tile_ok cm nc off ld (S.length s))
          (ensures (let u = upd_tile cm nc off ld s m in
                    S.length u == S.length s /\
                    (forall (y : abs (16 @| nc @| INil)).
                       let (a, (b, ())) = y in S.index u (tix cm nc off ld a b) == acc m y) /\
                    (forall (i : nat). i < S.length s /\ ~(in_tile cm nc off ld i) ==> S.index u i == S.index s i)))
  = let aux (y : abs (16 @| nc @| INil))
      : Lemma (let (a, (b, ())) = y in S.index (upd_tile cm nc off ld s m) (tix cm nc off ld a b) == acc m y)
      = let (a, (b, ())) = y in tix_inv cm nc off ld a b in
    FStar.Classical.forall_intro aux

noextract
let tile_bij (cm : bool) (nc : pos) (off : nat) (ld : pos) (n : nat{tile_ok cm nc off ld n})
  : ((i : natlt n { in_tile cm nc off ld i }) =~ abs (16 @| nc @| INil))
  = { ff = (fun i -> tix_back cm nc off ld i;
                    ((if cm then (i - off) % ld else (i - off) / ld) <: natlt 16),
                    (((if cm then (i - off) / ld else (i - off) % ld) <: natlt nc), ()));
      gg = (fun (x : abs (16 @| nc @| INil)) -> let (a, (b, ())) = x in
                    tix_inv cm nc off ld a b; (tix cm nc off ld a b <: (i : natlt n { in_tile cm nc off ld i })));
      ff_gg = (fun x -> let (a, (b, ())) = x in tix_inv cm nc off ld a b);
      gg_ff = (fun i -> tix_back cm nc off ld i) }

noextract
let tile_frame (#et : Type0) (#n : nat) (a : larray et n) (cm : bool) (nc : pos) (off : nat) (ld : pos)
  (f : perm) (s : S.seq et{S.length s == n}) : slprop
  = forall+ (i : natlt n { ~ (in_tile cm nc off ld i) }). pts_to_cell a #f i (S.index s i)

ghost
fn view_tile (#et : Type0) (#n : nat) (a : larray et n) (cm : bool) (nc : pos) (off : nat) (ld : pos{tile_ok cm nc off ld n})
  (#f : perm) (#s : erased (S.seq et){S.length s == n})
  requires a |-> Frac f s
  ensures tensor_pts_to (from_array (tlay cm nc off ld n) a) #f (tile_val cm nc off ld s ())
          ** tile_frame a cm nc off ld f s
{
  pts_to_len a;
  array_slice_1 a;
  forevery_refine_split (fun (i : natlt n) -> pts_to_cell a #f i (S.index s i)) (fun i -> in_tile cm nc off ld i);
  fold (tile_frame a cm nc off ld f s);
  let l = tlay cm nc off ld n;
  let t = from_array l a;
  assert (rewrites_to t (from_array (tlay cm nc off ld n) a));
  forevery_iso (tile_bij cm nc off ld n) (fun (i : natlt n { in_tile cm nc off ld i }) -> pts_to_cell a #f i (S.index s i));
  forevery_ext (fun (y : abs (16 @| nc @| INil)) ->
                  pts_to_cell a #f ((tile_bij cm nc off ld n).gg y) (S.index s ((tile_bij cm nc off ld n).gg y)))
               (fun (y : abs (16 @| nc @| INil)) ->
                  pts_to_cell (core t) #f (l.imap.f y) (acc (tile_val cm nc off ld s ()) y));
  lem_from_array_core #et #2 #(16 @| nc @| INil) #l a;
  rewrite array_exists a as array_exists (core t);
  tensor_iraise t;
}

ghost
fn unview_tile (#et : Type0) (#n : nat) (a : larray et n) (cm : bool) (nc : pos) (off : nat) (ld : pos{tile_ok cm nc off ld n})
  (#f : perm) (#s : erased (S.seq et){S.length s == n}) (#m : chest2 et 16 nc)
  requires tensor_pts_to (from_array (tlay cm nc off ld n) a) #f m ** tile_frame a cm nc off ld f s
  ensures a |-> Frac f (upd_tile cm nc off ld s m)
{
  let l = tlay cm nc off ld n;
  let t = from_array l a;
  assert (rewrites_to t (from_array (tlay cm nc off ld n) a));
  let u : erased (S.seq et) = upd_tile cm nc off ld s m;
  upd_tile_lemmas cm nc off ld s m;
  tensor_ilower t;
  forevery_ext (fun (y : abs (16 @| nc @| INil)) ->
                  pts_to_cell (core t) #f (l.imap.f y) (acc m y))
               (fun (y : abs (16 @| nc @| INil)) ->
                  pts_to_cell a #f ((tile_bij cm nc off ld n).gg y) (S.index u ((tile_bij cm nc off ld n).gg y)));
  forevery_iso_back (tile_bij cm nc off ld n) (fun (i : natlt n { in_tile cm nc off ld i }) -> pts_to_cell a #f i (S.index u i));
  unfold (tile_frame a cm nc off ld f s);
  forevery_ext (fun (i : natlt n { ~ (in_tile cm nc off ld i) }) -> pts_to_cell a #f i (S.index s i))
               (fun (i : natlt n { ~ (in_tile cm nc off ld i) }) -> pts_to_cell a #f i (S.index u i));
  forevery_refine_join (fun (i : natlt n) -> pts_to_cell a #f i (S.index u i))
    (fun i -> in_tile cm nc off ld i) (fun i -> ~ (in_tile cm nc off ld i));
  forevery_unrefine (fun (i : natlt n) -> pts_to_cell a #f i (S.index u i));
  lem_from_array_core #et #2 #(16 @| nc @| INil) #l a;
  rewrite array_exists (core t) as array_exists a;
  array_unslice_1_with_exists a #f #u;
}

noextract
let upd_tile_same (#et : Type) (cm : bool) (nc : pos) (off : nat) (ld : pos) (s : S.seq et)
  : Lemma (requires tile_ok cm nc off ld (S.length s))
          (ensures upd_tile cm nc off ld s (tile_val cm nc off ld s ()) == s)
  = let u = upd_tile cm nc off ld s (tile_val cm nc off ld s ()) in
    let aux (i : nat{i < S.length s}) : Lemma (S.index u i == S.index s i)
      = if in_tile cm nc off ld i then tix_back cm nc off ld i in
    FStar.Classical.forall_intro aux;
    assert (S.equal u s)
#pop-options
(* ===== Share ===== *)
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }



#push-options "--z3rlimit 40 --fuel 0 --ifuel 1"

ghost
fn arr_untensor (#et : Type0) (#n : nat) (a : larray et n) (#f : perm) (#s : erased (S.seq et))
  (_ : squash (S.length s == n)) ()
  requires tensor_pts_to (from_array (l1_forward n) a) #f (from_seq (l1_forward n) s)
  ensures a |-> Frac f s
{
  tensor_concr (from_array (l1_forward n) a);
  rewrite (core (from_array (l1_forward n) a) |-> Frac f (to_seq (l1_forward n) (from_seq (l1_forward n) s)))
       as (a |-> Frac f s);
}

ghost
fn arr_share_n (#et : Type0) (#n : nat) (a : larray et n) (k : pos) (#f : perm) (#s : erased (S.seq et))
  requires a |-> Frac f s
  ensures forall+ (_ : natlt k). a |-> Frac (f /. k) s
{
  pts_to_len a;
  tensor_abs' (l1_forward n) a;
  tensor_share_n (from_array (l1_forward n) a) k;
  forevery_map #(natlt k)
    (fun _ -> tensor_pts_to (from_array (l1_forward n) a) #(f /. k) (from_seq (l1_forward n) s))
    (fun _ -> a |-> Frac (f /. k) s)
    (fun _ -> arr_untensor a #(f /. k) #s () ());
}

ghost
fn arr_tensor (#et : Type0) (#n : nat) (a : larray et n) (#f : perm) (#s : erased (S.seq et))
  (_ : squash (S.length s == n)) ()
  requires a |-> Frac f s
  ensures tensor_pts_to (from_array (l1_forward n) a) #f (from_seq (l1_forward n) s)
{
  tensor_abs' (l1_forward n) a;
}

ghost
fn arr_gather_n (#et : Type0) (#n : nat) (a : larray et n) (k : pos) (#f : perm) (#s : erased (S.seq et))
  (_ : squash (S.length s == n))
  requires forall+ (_ : natlt k). a |-> Frac (f /. k) s
  ensures a |-> Frac f s
{
  forevery_map #(natlt k)
    (fun _ -> a |-> Frac (f /. k) s)
    (fun _ -> tensor_pts_to (from_array (l1_forward n) a) #(f /. k) (from_seq (l1_forward n) s))
    (fun _ -> arr_tensor a #(f /. k) #s () ());
  tensor_gather_n (from_array (l1_forward n) a) k;
  arr_untensor a #f #s () ();
}
#pop-options
(* ===== Wmma ===== *)
open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.GEMM
open Kuiper.Bijection



#push-options "--z3rlimit 40 --fuel 0 --ifuel 1"

noextract
let nW : nat = 21555360
noextract
let nB : nat = 57344
noextract
let nC : nat = 49152

(* Parameters of one projection: weight offset, reduction length K = nch * chunk, nt row tiles. *)
noextract
let pok (woff kk nt nch chunk : nat) : prop =
  0 < nt /\ 0 < chunk /\ chunk % 16 == 0 /\ kk == nch * chunk /\ 16 <= kk /\
  woff + nt * 16 * kk <= nW /\ 16 * kk <= nB /\ nt * nch * 128 <= nC

noextract
let rec chain (sw : S.seq bf16{S.length sw == nW}) (sb : S.seq bf16{S.length sb == nB})
  (aoff boff : nat) (kk : pos) (j : nat)
  : Tot (chest2 float 16 8) (decreases j)
  = if j = 0 || not (16 <= kk && aoff + 16 * j + kk * 15 <= nW && boff + 16 * j + kk * 15 <= nB)
    then const (16 @| 8 @| INil) zero
    else M.emma (chain sw sb aoff boff kk (j - 1))
              (tile_val false 16 (aoff + 16 * (j - 1)) kk sw ())
              (tile_val true 8 (boff + 16 * (j - 1)) kk sb ())

#push-options "--fuel 1"
noextract
let chain_zero (sw : S.seq bf16{S.length sw == nW}) (sb : S.seq bf16{S.length sb == nB})
  (aoff boff : nat) (kk : pos)
  : Lemma (chain sw sb aoff boff kk 0 == const (16 @| 8 @| INil) zero)
  = ()

noextract
let chain_step (sw : S.seq bf16{S.length sw == nW}) (sb : S.seq bf16{S.length sb == nB})
  (aoff boff : nat) (kk : pos) (j : nat)
  : Lemma (requires 16 <= kk /\ aoff + 16 * j + kk * 15 + 16 <= nW /\ boff + 16 * j + kk * 15 + 16 <= nB)
          (ensures chain sw sb aoff boff kk (j + 1) ==
                   M.emma (chain sw sb aoff boff kk j)
                        (tile_val false 16 (aoff + 16 * j) kk sw ())
                        (tile_val true 8 (boff + 16 * j) kk sb ()))
  = ()
#pop-options

noextract
let gboff (nch chunk : nat) (g : nat) : nat = (g % (if nch = 0 then 1 else nch)) * chunk
noextract
let gaoff (woff kk nch chunk : nat) (g : nat) : nat =
  woff + (g / (if nch = 0 then 1 else nch)) * 16 * kk + gboff nch chunk g

noextract
let tile_bounds (woff kk nt nch chunk : nat) (g : nat) (j : nat)
  : Lemma (requires pok woff kk nt nch chunk /\ g < nt * nch /\ j < chunk / 16)
          (ensures (let boff = gboff nch chunk g in let aoff = gaoff woff kk nch chunk g in
                    aoff + 16 * j + kk * 15 + 16 <= nW /\ boff + 16 * j + kk * 15 + 16 <= nB /\
                    aoff + 16 * (chunk / 16) + kk * 15 <= nW /\ boff + 16 * (chunk / 16) + kk * 15 <= nB /\
                    16 * (chunk / 16) == chunk /\ g * 128 + 128 <= nC))
  = let t = g / nch in let ch = g % nch in
    FStar.Math.Lemmas.euclidean_division_definition g nch;
    FStar.Math.Lemmas.euclidean_division_definition chunk 16;
    assert (ch < nch);
    assert (t < nt);
    FStar.Math.Lemmas.lemma_mult_le_right chunk (ch + 1) nch;
    assert ((ch + 1) * chunk <= kk);
    FStar.Math.Lemmas.lemma_mult_le_right (16 * kk) (t + 1) nt;
    assert ((t + 1) * (16 * kk) <= nt * (16 * kk));
    assert (woff + nt * 16 * kk <= nW);
    assert (16 * j + 16 <= chunk);
    FStar.Math.Lemmas.lemma_mult_le_right 128 (g + 1) (nt * nch);
    assert ((g + 1) * 128 <= nt * nch * 128)

noextract
let fits_le (x : nat) : Lemma (requires x <= nW) (ensures SZ.fits x)
  = FStar.UInt.pow2_values 32

noextract
let kfits (woff kk nt nch chunk : nat) (g j : nat)
  : Lemma (requires pok woff kk nt nch chunk /\ g < nt * nch /\ j < chunk / 16)
          (ensures SZ.fits (nt * nch) /\ SZ.fits (g * 128) /\ SZ.fits (g + 1) /\
                   SZ.fits ((g % nch) * chunk) /\ SZ.fits ((g / nch) * 16) /\
                   SZ.fits ((g / nch) * 16 * kk) /\ SZ.fits (woff + (g / nch) * 16 * kk) /\
                   SZ.fits (gaoff woff kk nch chunk g) /\ SZ.fits (16 * j) /\ SZ.fits (j + 1) /\
                   SZ.fits (gaoff woff kk nch chunk g + 16 * j) /\
                   SZ.fits (gboff nch chunk g + 16 * j))
  = tile_bounds woff kk nt nch chunk g j;
    FStar.Math.Lemmas.euclidean_division_definition g nch;
    FStar.Math.Lemmas.lemma_mult_le_right 16 (g / nch) nt;
    FStar.Math.Lemmas.lemma_mult_le_right kk ((g / nch) * 16) (nt * 16);
    FStar.Math.Lemmas.lemma_mult_le_right 128 g (nt * nch);
    fits_le (nt * nch); fits_le (g * 128); fits_le (g + 1);
    fits_le ((g % nch) * chunk); fits_le ((g / nch) * 16); fits_le ((g / nch) * 16 * kk);
    fits_le (woff + (g / nch) * 16 * kk); fits_le (gaoff woff kk nch chunk g);
    fits_le (16 * j); fits_le (j + 1); fits_le (gaoff woff kk nch chunk g + 16 * j);
    fits_le (gboff nch chunk g + 16 * j)

noextract
let div_lt (i a : nat) (b : pos)
  : Lemma (requires i < a * b) (ensures i / b < a)
  = FStar.Math.Lemmas.euclidean_division_definition i b;
    if i / b >= a then FStar.Math.Lemmas.lemma_mult_le_right b a (i / b)

noextract
let cmodel_at (sw : S.seq bf16{S.length sw == nW}) (sb : S.seq bf16{S.length sb == nB})
  (woff kk nt nch chunk : nat) (i : nat)
  : GTot float
  = if pok woff kk nt nch chunk && i < nt * nch * 128 then begin
      let g = i / 128 in
      div_lt i (nt * nch) 128;
      tile_bounds woff kk nt nch chunk g 0;
      acc2 (chain sw sb (gaoff woff kk nch chunk g) (gboff nch chunk g) kk (chunk / 16))
           ((i % 128) / 8) (i % 8)
    end else zero

noextract
let cfin (sw : S.seq bf16{S.length sw == nW}) (sb : S.seq bf16{S.length sb == nB})
  (woff kk nt nch chunk : nat) (sc : S.seq float)
  : GTot (S.seq float)
  = S.init_ghost (S.length sc) (fun i ->
      if i < nt * nch * 128 then cmodel_at sw sb woff kk nt nch chunk i else S.index sc i)

[@@"opaque_to_smt"]
noextract
let partial_ok (sw : S.seq bf16{S.length sw == nW}) (sb : S.seq bf16{S.length sb == nB})
  (woff kk nt nch chunk : nat) (sc sc' : S.seq float) (g : nat) : prop
  = S.length sc' == S.length sc /\
    (forall (i : nat). i < S.length sc ==>
       S.index sc' i == (if i < g * 128 then cmodel_at sw sb woff kk nt nch chunk i else S.index sc i))

unfold let quarter = 1.0R /. warp_size

noextract
let partial_init (sw : S.seq bf16{S.length sw == nW}) (sb : S.seq bf16{S.length sb == nB})
  (woff kk nt nch chunk : nat) (sc : S.seq float)
  : Lemma (partial_ok sw sb woff kk nt nch chunk sc sc 0)
  = reveal_opaque (`%partial_ok) (partial_ok sw sb woff kk nt nch chunk sc sc 0)

noextract
let partial_fin (sw : S.seq bf16{S.length sw == nW}) (sb : S.seq bf16{S.length sb == nB})
  (woff kk nt nch chunk : nat) (sc sc' : S.seq float)
  : Lemma (requires partial_ok sw sb woff kk nt nch chunk sc sc' (nt * nch))
          (ensures sc' == cfin sw sb woff kk nt nch chunk sc)
  = reveal_opaque (`%partial_ok) (partial_ok sw sb woff kk nt nch chunk sc sc' (nt * nch));
    assert (S.equal sc' (cfin sw sb woff kk nt nch chunk sc))

noextract
let blk (g i : nat)
  : Lemma (in_tile false 8 (g * 128) 8 i <==> (g * 128 <= i /\ i < g * 128 + 128))
  = if g * 128 <= i then FStar.Math.Lemmas.euclidean_division_definition (i - g * 128) 8

noextract
let blk2 (g i : nat)
  : Lemma (requires g * 128 <= i /\ i < g * 128 + 128)
          (ensures i / 128 == g /\ i % 128 == i - g * 128 /\ (i - g * 128) % 8 == i % 8)
  = FStar.Math.Lemmas.lemma_div_plus (i - g * 128) g 128;
    FStar.Math.Lemmas.small_div (i - g * 128) 128;
    FStar.Math.Lemmas.lemma_mod_plus (i - g * 128) g 128;
    FStar.Math.Lemmas.small_mod (i - g * 128) 128;
    FStar.Math.Lemmas.lemma_mod_plus (i - g * 128) (16 * g) 8

noextract
let store_in (m : chest2 float 16 8) (sc' : S.seq float{S.length sc' == nC}) (g : nat) (i : nat)
  : Lemma (requires g * 128 <= i /\ i < g * 128 + 128 /\ g * 128 + 128 <= nC)
          (ensures S.index (upd_tile false 8 (g * 128) 8 sc' m) i == acc2 m ((i % 128) / 8) (i % 8))
  = blk g i; blk2 g i

noextract
let store_out (m : chest2 float 16 8) (sc' : S.seq float{S.length sc' == nC}) (g : nat) (i : nat)
  : Lemma (requires ~(g * 128 <= i /\ i < g * 128 + 128) /\ i < nC)
          (ensures S.index (upd_tile false 8 (g * 128) 8 sc' m) i == S.index sc' i)
  = blk g i

noextract
let cmodel_in (sw : S.seq bf16{S.length sw == nW}) (sb : S.seq bf16{S.length sb == nB})
  (woff kk nt nch chunk : nat) (g : nat) (i : nat)
  : Lemma (requires pok woff kk nt nch chunk /\ g < nt * nch /\ g * 128 <= i /\ i < g * 128 + 128)
          (ensures (tile_bounds woff kk nt nch chunk g 0;
                    i < nt * nch * 128 /\
                    cmodel_at sw sb woff kk nt nch chunk i ==
                    acc2 (chain sw sb (gaoff woff kk nch chunk g) (gboff nch chunk g) kk (chunk / 16))
                         ((i % 128) / 8) (i % 8)))
  = blk2 g i;
    FStar.Math.Lemmas.lemma_mult_le_right 128 (g + 1) (nt * nch)

noextract
let store_pt (sw : S.seq bf16{S.length sw == nW}) (sb : S.seq bf16{S.length sb == nB})
  (woff kk nt nch chunk : nat) (sc : S.seq float{S.length sc == nC}) (sc' : S.seq float{S.length sc' == nC}) (g : nat) (i : nat)
  : Lemma (requires pok woff kk nt nch chunk /\ g < nt * nch /\ i < nC /\
                    S.index sc' i == (if i < g * 128 then cmodel_at sw sb woff kk nt nch chunk i else S.index sc i))
          (ensures (tile_bounds woff kk nt nch chunk g 0;
                    S.index (upd_tile false 8 (g * 128) 8 sc'
                        (chain sw sb (gaoff woff kk nch chunk g) (gboff nch chunk g) kk (chunk / 16))) i ==
                    (if i < (g + 1) * 128 then cmodel_at sw sb woff kk nt nch chunk i else S.index sc i)))
  = tile_bounds woff kk nt nch chunk g 0;
    let m = chain sw sb (gaoff woff kk nch chunk g) (gboff nch chunk g) kk (chunk / 16) in
    assert ((g + 1) * 128 == g * 128 + 128);
    if g * 128 <= i && i < g * 128 + 128 then begin
      store_in m sc' g i;
      cmodel_in sw sb woff kk nt nch chunk g i
    end else
      store_out m sc' g i

noextract
let store_step (sw : S.seq bf16{S.length sw == nW}) (sb : S.seq bf16{S.length sb == nB})
  (woff kk nt nch chunk : nat) (sc sc' : S.seq float) (g : nat)
  : Lemma (requires pok woff kk nt nch chunk /\ g < nt * nch /\ S.length sc == nC /\
                    partial_ok sw sb woff kk nt nch chunk sc sc' g)
          (ensures (tile_bounds woff kk nt nch chunk g 0;
                    partial_ok sw sb woff kk nt nch chunk sc
                      (upd_tile false 8 (g * 128) 8 sc'
                        (chain sw sb (gaoff woff kk nch chunk g) (gboff nch chunk g) kk (chunk / 16))) (g + 1)))
  = reveal_opaque (`%partial_ok) (partial_ok sw sb woff kk nt nch chunk sc sc' g);
    tile_bounds woff kk nt nch chunk g 0;
    let m = chain sw sb (gaoff woff kk nch chunk g) (gboff nch chunk g) kk (chunk / 16) in
    let u = upd_tile false 8 (g * 128) 8 sc' m in
    let aux (i : nat{i < S.length sc})
      : Lemma (S.index u i == (if i < (g + 1) * 128 then cmodel_at sw sb woff kk nt nch chunk i else S.index sc i))
      = store_pt sw sb woff kk nt nch chunk sc sc' g i
    in
    FStar.Classical.forall_intro aux;
    reveal_opaque (`%partial_ok) (partial_ok sw sb woff kk nt nch chunk sc u (g + 1))

inline_for_extraction noextract
fn wmma_thread
  (w : larray bf16 nW) (b : larray bf16 nB) (c : larray float nC)
  (woff kk nt nch chunk : sz)
  (#fw #fb : perm) (#sw : erased (S.seq bf16){S.length sw == nW}) (#sb : erased (S.seq bf16){S.length sb == nB})
  (#sc : erased (S.seq float){S.length sc == nC})
  requires pure (pok (SZ.v woff) (SZ.v kk) (SZ.v nt) (SZ.v nch) (SZ.v chunk))
  preserves gpu ** (w |-> Frac fw sw) ** (b |-> Frac fb sb)
  requires c |-> Frac quarter sc
  ensures c |-> Frac quarter (cfin sw sb (SZ.v woff) (SZ.v kk) (SZ.v nt) (SZ.v nch) (SZ.v chunk) sc)
{
  pts_to_len w; pts_to_len b; pts_to_len c;
  let fc = M.alloc_fragment ();
  kfits (SZ.v woff) (SZ.v kk) (SZ.v nt) (SZ.v nch) (SZ.v chunk) 0 0;
  let ng = nt *^ nch;
  let nj = chunk /^ 16sz;
  partial_init sw sb (SZ.v woff) (SZ.v kk) (SZ.v nt) (SZ.v nch) (SZ.v chunk) sc;
  let mut g = 0sz;
  while (!g <^ ng)
    invariant exists* (vg : sz) (sc' : S.seq float) vc.
      (g |-> vg) ** (c |-> Frac quarter sc') ** (fc |-> vc) **
      pure (SZ.v vg <= SZ.v ng /\
            partial_ok sw sb (SZ.v woff) (SZ.v kk) (SZ.v nt) (SZ.v nch) (SZ.v chunk) sc sc' (SZ.v vg))
    decreases (SZ.v ng - SZ.v !g)
  {
    let vg = !g;
    tile_bounds (SZ.v woff) (SZ.v kk) (SZ.v nt) (SZ.v nch) (SZ.v chunk) (SZ.v vg) 0;
    kfits (SZ.v woff) (SZ.v kk) (SZ.v nt) (SZ.v nch) (SZ.v chunk) (SZ.v vg) 0;
    let ch = vg %^ nch;
    let t = vg /^ nch;
    assert (pure (SZ.v ch == SZ.v vg % SZ.v nch /\ SZ.v t == SZ.v vg / SZ.v nch));
    let boff = ch *^ chunk;
    let t16 = t *^ 16sz;
    let tk = t16 *^ kk;
    let aoff = (woff +^ tk) +^ boff;
    assert (pure (SZ.v boff == gboff (SZ.v nch) (SZ.v chunk) (SZ.v vg)));
    assert (pure (SZ.v aoff == gaoff (SZ.v woff) (SZ.v kk) (SZ.v nch) (SZ.v chunk) (SZ.v vg)));
    M.fill fc zero;
    chain_zero sw sb (SZ.v aoff) (SZ.v boff) (SZ.v kk);
    let mut j = 0sz;
    while (!j <^ nj)
      invariant exists* (vj : sz).
        (j |-> vj) **
        (fc |-> chain sw sb (SZ.v aoff) (SZ.v boff) (SZ.v kk) (SZ.v vj)) **
        pure (SZ.v vj <= SZ.v nj)
      decreases (SZ.v nj - SZ.v !j)
    {
      let vj = !j;
      tile_bounds (SZ.v woff) (SZ.v kk) (SZ.v nt) (SZ.v nch) (SZ.v chunk) (SZ.v vg) (SZ.v vj);
      kfits (SZ.v woff) (SZ.v kk) (SZ.v nt) (SZ.v nch) (SZ.v chunk) (SZ.v vg) (SZ.v vj);
      let ao = aoff +^ 16sz *^ vj;
      let bo = boff +^ 16sz *^ vj;
      view_tile w false 16 (SZ.v ao) (SZ.v kk);
      view_tile b true 8 (SZ.v bo) (SZ.v kk);
      M.mma_sync #_ #(srm_tlay 16 ao kk nW) (from_array (tlay false 16 (SZ.v ao) (SZ.v kk) nW) w)
                 #_ #(scm_tlay 8 bo kk nB) (from_array (tlay true 8 (SZ.v bo) (SZ.v kk) nB) b) fc;
      unview_tile w false 16 (SZ.v ao) (SZ.v kk);
      upd_tile_same false 16 (SZ.v ao) (SZ.v kk) sw;
      unview_tile b true 8 (SZ.v bo) (SZ.v kk);
      upd_tile_same true 8 (SZ.v bo) (SZ.v kk) sb;
      chain_step sw sb (SZ.v aoff) (SZ.v boff) (SZ.v kk) (SZ.v vj);
      rewrite (fc |-> M.emma (chain sw sb (SZ.v aoff) (SZ.v boff) (SZ.v kk) (SZ.v vj))
                        (tile_val false 16 (SZ.v ao) (SZ.v kk) sw ())
                        (tile_val true 8 (SZ.v bo) (SZ.v kk) sb ()))
           as (fc |-> chain sw sb (SZ.v aoff) (SZ.v boff) (SZ.v kk) (SZ.v vj + 1));
      j := vj +^ 1sz;
    };
    with sc'. assert (c |-> Frac quarter sc');
    store_step sw sb (SZ.v woff) (SZ.v kk) (SZ.v nt) (SZ.v nch) (SZ.v chunk) sc sc' (SZ.v vg);
    let co = vg *^ 128sz;
    pts_to_len c;
    view_tile c false 8 (SZ.v co) (SZ.v 8sz);
    M.store fc #_ #(srm_tlay 8 co 8sz nC) (from_array (tlay false 8 (SZ.v co) (SZ.v 8sz) nC) c);
    unview_tile c false 8 (SZ.v co) (SZ.v 8sz);
    g := vg +^ 1sz;
  };
  with sc'. assert (c |-> Frac quarter sc');
  partial_fin sw sb (SZ.v woff) (SZ.v kk) (SZ.v nt) (SZ.v nch) (SZ.v chunk) sc sc';
  with x. assert fc |-> x; drop_ (fc |-> x);
}

noextract
instance is_send_global_pts_to (#a:Type0) (x:array a { is_global x }) (#f:perm) (v:S.seq a)
  : is_send_across gpu_of (pts_to x #f v)
  = let i : is_send_across (visibility_of x) (pts_to x #f v) = solve in i

noextract
let wsw (sw : erased (S.seq bf16)) = squash (S.length sw == nW)

unfold
noextract
let wpre (w : larray bf16 nW) (b : larray bf16 nB) (c : larray float nC)
  (fw fb : perm) (sw sb : S.seq bf16) (sc : S.seq float) (tid : nat) : slprop
  = (w |-> Frac (fw /. 32) sw) ** (b |-> Frac (fb /. 32) sb) ** (c |-> Frac quarter sc)

ghost
fn wmma_setup (w : larray bf16 nW) (b : larray bf16 nB) (c : larray float nC)
  (fw fb : perm) (sw sb : erased (S.seq bf16)) (sc : erased (S.seq float)) ()
  norewrite
  requires (w |-> Frac fw sw) ** (b |-> Frac fb sb) ** (c |-> sc)
  ensures (forall+ (tid : natlt 32). wpre w b c fw fb sw sb sc tid) ** emp
{
  arr_share_n w 32;
  arr_share_n b 32;
  arr_share_n c 32;
  forevery_zip3 #(natlt 32) (fun _ -> w |-> Frac (fw /. 32) sw) (fun _ -> b |-> Frac (fb /. 32) sb)
    (fun _ -> c |-> Frac (1.0R /. 32) sc);
  forevery_ext #(natlt 32)
    (fun _ -> (w |-> Frac (fw /. 32) sw) ** (b |-> Frac (fb /. 32) sb) ** (c |-> Frac (1.0R /. 32) sc))
    (fun tid -> wpre w b c fw fb sw sb sc tid);
}

ghost
fn wmma_teardown (w : larray bf16 nW) (b : larray bf16 nB) (c : larray float nC)
  (fw fb : perm) (sw sb : erased (S.seq bf16)) (sc : erased (S.seq float))
  (_ : squash (S.length sw == nW /\ S.length sb == nB /\ S.length sc == nC)) ()
  norewrite
  requires (forall+ (tid : natlt 32). wpre w b c fw fb sw sb sc tid) ** emp
  ensures (w |-> Frac fw sw) ** (b |-> Frac fb sb) ** (c |-> sc)
{
  forevery_ext #(natlt 32)
    (fun tid -> wpre w b c fw fb sw sb sc tid)
    (fun _ -> (w |-> Frac (fw /. 32) sw) ** (b |-> Frac (fb /. 32) sb) ** (c |-> Frac (1.0R /. 32) sc));
  forevery_unzip3 #(natlt 32) (fun _ -> w |-> Frac (fw /. 32) sw) (fun _ -> b |-> Frac (fb /. 32) sb)
    (fun _ -> c |-> Frac (1.0R /. 32) sc);
  arr_gather_n w 32 ();
  arr_gather_n b 32 ();
  arr_gather_n c 32 #1.0R ();
}

noextract
let cfinv (sw : erased (S.seq bf16){S.length sw == nW}) (sb : erased (S.seq bf16){S.length sb == nB})
  (woff kk nt nch chunk : sz) (sc : erased (S.seq float))
  : erased (S.seq float)
  = hide (cfin sw sb (SZ.v woff) (SZ.v kk) (SZ.v nt) (SZ.v nch) (SZ.v chunk) sc)

inline_for_extraction noextract
fn wmma_kf (w : larray bf16 nW) (b : larray bf16 nB) (c : larray float nC)
  (woff kk nt nch chunk : sz)
  (fw fb : perm) (sw : erased (S.seq bf16){S.length sw == nW}) (sb : erased (S.seq bf16){S.length sb == nB})
  (sc : erased (S.seq float){S.length sc == nC})
  (_ : squash (pok (SZ.v woff) (SZ.v kk) (SZ.v nt) (SZ.v nch) (SZ.v chunk)))
  (tid : szlt 32sz) ()
  norewrite
  requires gpu ** wpre w b c fw fb sw sb sc tid ** thread_id 32sz tid
  ensures gpu ** wpre w b c fw fb sw sb (cfinv sw sb woff kk nt nch chunk sc) tid ** thread_id 32sz tid
{
  wmma_thread w b c woff kk nt nch chunk;
}

inline_for_extraction noextract
let wmma_kdesc (w : larray bf16 nW) (b : larray bf16 nB) (c : larray float nC)
  (#_ : squash (is_global w)) (#_ : squash (is_global b)) (#_ : squash (is_global c))
  (woff kk nt nch chunk : sz)
  (fw fb : perm) (sw : erased (S.seq bf16){S.length sw == nW}) (sb : erased (S.seq bf16){S.length sb == nB})
  (sc : erased (S.seq float){S.length sc == nC})
  (pf : squash (pok (SZ.v woff) (SZ.v kk) (SZ.v nt) (SZ.v nch) (SZ.v chunk)))
  : kernel_desc_1_n ((w |-> Frac fw sw) ** (b |-> Frac fb sb) ** (c |-> sc))
                    ((w |-> Frac fw sw) ** (b |-> Frac fb sb) ** (c |-> cfinv sw sb woff kk nt nch chunk sc))
  = {
    nthr = 32sz;
    frame = emp;
    kpre = (fun tid -> wpre w b c fw fb sw sb sc tid);
    kpost = (fun tid -> wpre w b c fw fb sw sb (cfinv sw sb woff kk nt nch chunk sc) tid);
    block_setup = wmma_setup w b c fw fb sw sb sc;
    block_teardown = wmma_teardown w b c fw fb sw sb (cfinv sw sb woff kk nt nch chunk sc) ();
    f = (fun tid -> wmma_kf w b c woff kk nt nch chunk fw fb sw sb sc pf tid);
    full_pre_sendable = solve;
    full_post_sendable = solve;
    kpre_sendable = (fun _ -> solve);
    kpost_sendable = (fun _ -> solve);
  }

inline_for_extraction noextract
fn wmma_proj (w : larray bf16 nW) (b : larray bf16 nB) (c : larray float nC)
  (#_ : squash (is_global w)) (#_ : squash (is_global b)) (#_ : squash (is_global c))
  (woff kk nt nch chunk : sz)
  (#fw #fb : perm) (#sw : erased (S.seq bf16){S.length sw == nW}) (#sb : erased (S.seq bf16){S.length sb == nB})
  (#sc : erased (S.seq float){S.length sc == nC})
  requires pure (pok (SZ.v woff) (SZ.v kk) (SZ.v nt) (SZ.v nch) (SZ.v chunk))
  preserves cpu ** on gpu_loc (w |-> Frac fw sw) ** on gpu_loc (b |-> Frac fb sb)
  requires on gpu_loc (c |-> sc)
  ensures on gpu_loc (c |-> cfin sw sb (SZ.v woff) (SZ.v kk) (SZ.v nt) (SZ.v nch) (SZ.v chunk) sc)
{
  on_star_intro #gpu_loc (w |-> Frac fw sw) (b |-> Frac fb sb);
  on_star_intro #gpu_loc ((w |-> Frac fw sw) ** (b |-> Frac fb sb)) (c |-> sc);
  launch_sync (wmma_kdesc w b c woff kk nt nch chunk fw fb sw sb sc ());
  on_star_elim #gpu_loc ((w |-> Frac fw sw) ** (b |-> Frac fb sb)) (c |-> cfinv sw sb woff kk nt nch chunk sc);
  on_star_elim #gpu_loc (w |-> Frac fw sw) (b |-> Frac fb sb);
  rewrite on gpu_loc (c |-> cfinv sw sb woff kk nt nch chunk sc)
       as on gpu_loc (c |-> cfin sw sb (SZ.v woff) (SZ.v kk) (SZ.v nt) (SZ.v nch) (SZ.v chunk) sc);
}
#pop-options
(* ===== RL ===== *)
open FStar.Real
open Kuiper.Real
open KPlay.SharedSpec


#push-options "--z3rlimit 40 --fuel 0 --ifuel 0"

(*** Generic real summation lemmas over vsum *)

(* same lambda as in vsum, so its SMT encoding coincides *)
noextract
let vs (n : nat) (f : nat -> real) : S.seq real = S.init n (fun (k:nat{k < n}) -> f k)

noextract
let vs_def (n : nat) (f : nat -> real) : Lemma (vsum n f == rsum (vs n f) /\ S.length (vs n f) == n /\
  (forall (k : nat). k < n ==> S.index (vs n f) k == f k)) = ()

noextract
let vsum_ext (n : nat) (f g : nat -> real)
  : Lemma (requires forall (k : nat). k < n ==> f k == g k)
          (ensures vsum n f == vsum n g)
  = vs_def n f; vs_def n g; assert (S.equal (vs n f) (vs n g))

#push-options "--fuel 2"
noextract
let vsum_zero (f : nat -> real) : Lemma (vsum 0 f == 0.0R)
  = vs_def 0 f; assert (S.equal (vs 0 f) S.empty)

noextract
let rsum_one (x : real) : Lemma (rsum (S.create 1 x) == x)
  = let s = S.create 1 x in
    assert (S.equal (S.tail s) S.empty);
    assert (Kuiper.Seq.Common.view_seq s == Kuiper.Seq.Common.SCons x S.empty)
#pop-options

noextract
let vsum_split (n m : nat) (f : nat -> real)
  : Lemma (vsum (n + m) f == vsum n f +. vsum m (fun k -> f (n + k)))
  = let g : nat -> real = fun k -> f (n + k) in
    vs_def (n + m) f; vs_def n f; vs_def m g;
    let s1 = vs n f in let s2 = vs m g in
    assert (S.equal (vs (n + m) f) (S.append s1 s2));
    rsum_append s1 s2

noextract
let vsum_succ (n : nat) (f : nat -> real)
  : Lemma (vsum (n + 1) f == vsum n f +. f n)
  = vsum_split n 1 f;
    let g : nat -> real = fun k -> f (n + k) in
    vs_def 1 g;
    assert (S.equal (vs 1 g) (S.create 1 (f n)));
    rsum_one (f n)

noextract
let rec vsum_add (n : nat) (f g : nat -> real)
  : Lemma (ensures vsum n (fun k -> f k +. g k) == vsum n f +. vsum n g) (decreases n)
  = if n = 0 then (vsum_zero f; vsum_zero g; vsum_zero (fun k -> f k +. g k))
    else begin
      vsum_add (n - 1) f g;
      vsum_succ (n - 1) f; vsum_succ (n - 1) g; vsum_succ (n - 1) (fun k -> f k +. g k)
    end

noextract
let vsum_one (f : nat -> real) : Lemma (vsum 1 f == f 0)
  = vsum_succ 0 f; vsum_zero f

(* block decomposition: index i*w + j *)
noextract
let rec vsum_block (c w : nat) (f : nat -> real)
  : Lemma (ensures vsum (c * w) f == vsum c (fun i -> vsum w (fun j -> f (i * w + j)))) (decreases c)
  = if c = 0 then (vsum_zero f; vsum_zero (fun i -> vsum w (fun j -> f (i * w + j))))
    else begin
      vsum_block (c - 1) w f;
      vsum_split ((c - 1) * w) w f;
      vsum_succ (c - 1) (fun i -> vsum w (fun j -> f (i * w + j)));
      vsum_ext w (fun k -> f ((c - 1) * w + k)) (fun j -> f ((c - 1) * w + j))
    end

(* even/odd split *)
noextract
let rec vsum_interleave (n : nat) (h : nat -> real)
  : Lemma (ensures vsum (2 * n) h == vsum n (fun t -> h (2 * t)) +. vsum n (fun t -> h (2 * t + 1))) (decreases n)
  = if n = 0 then (vsum_zero h; vsum_zero (fun t -> h (2 * t)); vsum_zero (fun t -> h (2 * t + 1)))
    else begin
      vsum_interleave (n - 1) h;
      vsum_succ (2 * (n - 1)) h;
      vsum_succ (2 * (n - 1) + 1) h;
      vsum_succ (n - 1) (fun t -> h (2 * t));
      vsum_succ (n - 1) (fun t -> h (2 * t + 1))
    end

noextract
let vsum_nonneg_sq (n : nat) (f : nat -> real)
  : Lemma (vsum n (fun k -> f k *. f k) >=. 0.0R)
  = let rec aux (m : nat) : Lemma (ensures vsum m (fun k -> f k *. f k) >=. 0.0R) (decreases m)
      = if m = 0 then vsum_zero (fun k -> f k *. f k)
        else (aux (m - 1); vsum_succ (m - 1) (fun k -> f k *. f k))
    in aux n


(*** Named combinators (no lambdas in statements) *)

noextract
let shf (f : nat -> real) (o : nat) (k : nat) : real = f (o + k)
noextract
let strd (f : nat -> real) (o s : nat) (t : nat) : real = f (o + s * t)
noextract
let rblk (f : nat -> real) (w : nat) (i : nat) : real = vsum w (shf f (i * w))
noextract
let addf (f g : nat -> real) (k : nat) : real = f k +. g k

noextract
let vsum_ext_all (n : nat) (f : nat -> real)
  : Lemma (forall (g : nat -> real). {:pattern (vsum n g)} (forall (k : nat). k < n ==> f k == g k) ==> vsum n f == vsum n g)
  = let aux (g : nat -> real) : Lemma ((forall (k : nat). k < n ==> f k == g k) ==> vsum n f == vsum n g)
      = FStar.Classical.move_requires (vsum_ext n f) g in
    FStar.Classical.forall_intro aux

noextract
let vsum_split2 (n m : nat) (f : nat -> real)
  : Lemma (vsum (n + m) f == vsum n f +. vsum m (shf f n))
  = vsum_split n m f; vsum_ext m (fun k -> f (n + k)) (shf f n)

noextract
let vsum_succ2 (n : nat) (f : nat -> real)
  : Lemma (vsum (n + 1) f == vsum n f +. f n) = vsum_succ n f

noextract
let vsum_add2 (n : nat) (f g : nat -> real)
  : Lemma (vsum n (addf f g) == vsum n f +. vsum n g)
  = vsum_add n f g; vsum_ext n (fun k -> f k +. g k) (addf f g)

noextract
let vsum_block2 (c w : nat) (f : nat -> real)
  : Lemma (vsum (c * w) f == vsum c (rblk f w))
  = vsum_block c w f;
    let aux (i : nat) : Lemma (vsum w (fun j -> f (i * w + j)) == rblk f w i)
      = vsum_ext w (fun j -> f (i * w + j)) (shf f (i * w)) in
    FStar.Classical.forall_intro aux;
    vsum_ext c (fun i -> vsum w (fun j -> f (i * w + j))) (rblk f w)

noextract
let vsum_interleave2 (n : nat) (h : nat -> real)
  : Lemma (vsum (2 * n) h == vsum n (strd h 0 2) +. vsum n (strd h 1 2))
  = vsum_interleave n h;
    vsum_ext n (fun t -> h (2 * t)) (strd h 0 2);
    vsum_ext n (fun t -> h (2 * t + 1)) (strd h 1 2)

noextract
let vsum_nonneg (n : nat) (f : nat -> real)
  : Lemma (requires forall (k : nat). k < n ==> f k >=. 0.0R) (ensures vsum n f >=. 0.0R)
  = let rec aux (m : nat) : Lemma (requires m <= n) (ensures vsum m f >=. 0.0R) (decreases m)
      = if m = 0 then vsum_zero f else (aux (m - 1); vsum_succ (m - 1) f)
    in aux n

#pop-options
(* ===== Ap ===== *)
open FStar.Real
open Kuiper
open KPlay.SharedSpec


(*** Native helpers shared by the kernels *)

inline_for_extraction noextract
let b2f (x : bf16) : float = C.c_bf16_32.fcast x
inline_for_extraction noextract
let f2b (x : float) : bf16 = C.c_32_bf16.fcast x
inline_for_extraction noextract
let silu_f (x : float) : float = div x (add one (fexp (neg x)))
inline_for_extraction noextract
let sigm_f (x : float) : float = div one (add one (fexp (neg x)))
inline_for_extraction noextract
let eps_v : float = of_literal #float "0.0000009999999974752427078783512115478515625"
inline_for_extraction noextract
let qscale_v : float = of_literal #float "0.088388346135616302490234375"
inline_for_extraction noextract
let inv1024_v : float = of_literal #float "0.0009765625"
inline_for_extraction noextract
let inv128_v : float = of_literal #float "0.0078125"

(*** Literal certificates (no normalization: explicit evaluation steps) *)

noextract
let lit_fstep (c:FStar.Char.char) (l:list FStar.Char.char) (n:nat) (d:pos) (v:nat)
  : Lemma (requires Lit.digit c == Some v) (ensures Lit.fraction (c::l) n d == Lit.fraction l (10*n+v) (10*d)) = ()

noextract
let lit_fnil (n:nat) (d:pos) : Lemma (Lit.fraction [] n d == Some (n, d)) = ()

noextract
let lit_istep (c:FStar.Char.char) (r:list FStar.Char.char)
  : Lemma (Lit.integral ('0'::'.'::c::r) 0 false == Lit.fraction (c::r) 0 1) = ()

noextract
let lit_bp (n:nat) : Lemma (Lit.binary_power (n + 1) == 2 * Lit.binary_power n) = ()

noextract
let lit_value_of (s:string) (n:nat) (d:pos)
  : Lemma (requires Lit.integral (FStar.String.list_of_string s) 0 false == Some (n, d))
          (ensures Lit.value s == FStar.Real.of_int n /. FStar.Real.of_int d) = ()

noextract
let lit_cert_of (s:string) (m:nat) (e:int) (n:nat) (d:pos)
  : Lemma (requires Cert.normal_encoding m e /\ Lit.integral (FStar.String.list_of_string s) 0 false == Some (n, d) /\
                    (let (un, ud) = Cert.unit_fraction e in n * ud = m * un * d))
          (ensures Cert.exact_normal s m e) = ()

noextract
let eps_frac () : Lemma (Lit.integral ['0';'.';'0';'0';'0';'0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 false == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) =
  lit_istep '0' ['0';'0';'0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'];
  lit_fstep '0' ['0';'0';'0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 1 0;
  lit_fstep '0' ['0';'0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 10 0;
  lit_fstep '0' ['0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 100 0;
  lit_fstep '0' ['0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 1000 0;
  lit_fstep '0' ['0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 10000 0;
  lit_fstep '0' ['9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 100000 0;
  lit_fstep '9' ['9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 1000000 9;
  lit_fstep '9' ['9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9 10000000 9;
  lit_fstep '9' ['9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99 100000000 9;
  lit_fstep '9' ['9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999 1000000000 9;
  lit_fstep '9' ['9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999 10000000000 9;
  lit_fstep '9' ['9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999 100000000000 9;
  lit_fstep '9' ['9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999 1000000000000 9;
  lit_fstep '9' ['7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999 10000000000000 9;
  lit_fstep '7' ['4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999 100000000000000 7;
  lit_fstep '4' ['7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997 1000000000000000 4;
  lit_fstep '7' ['5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974 10000000000000000 7;
  lit_fstep '5' ['2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747 100000000000000000 5;
  lit_fstep '2' ['4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475 1000000000000000000 2;
  lit_fstep '4' ['2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752 10000000000000000000 4;
  lit_fstep '2' ['7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524 100000000000000000000 2;
  lit_fstep '7' ['0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242 1000000000000000000000 7;
  lit_fstep '0' ['7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427 10000000000000000000000 0;
  lit_fstep '7' ['8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524270 100000000000000000000000 7;
  lit_fstep '8' ['7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242707 1000000000000000000000000 8;
  lit_fstep '7' ['8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427078 10000000000000000000000000 7;
  lit_fstep '8' ['3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524270787 100000000000000000000000000 8;
  lit_fstep '3' ['5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242707878 1000000000000000000000000000 3;
  lit_fstep '5' ['1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427078783 10000000000000000000000000000 5;
  lit_fstep '1' ['2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524270787835 100000000000000000000000000000 1;
  lit_fstep '2' ['1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242707878351 1000000000000000000000000000000 2;
  lit_fstep '1' ['1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427078783512 10000000000000000000000000000000 1;
  lit_fstep '1' ['5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524270787835121 100000000000000000000000000000000 1;
  lit_fstep '5' ['4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242707878351211 1000000000000000000000000000000000 5;
  lit_fstep '4' ['7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427078783512115 10000000000000000000000000000000000 4;
  lit_fstep '7' ['8';'5';'1';'5';'6';'2';'5'] 99999999747524270787835121154 100000000000000000000000000000000000 7;
  lit_fstep '8' ['5';'1';'5';'6';'2';'5'] 999999997475242707878351211547 1000000000000000000000000000000000000 8;
  lit_fstep '5' ['1';'5';'6';'2';'5'] 9999999974752427078783512115478 10000000000000000000000000000000000000 5;
  lit_fstep '1' ['5';'6';'2';'5'] 99999999747524270787835121154785 100000000000000000000000000000000000000 1;
  lit_fstep '5' ['6';'2';'5'] 999999997475242707878351211547851 1000000000000000000000000000000000000000 5;
  lit_fstep '6' ['2';'5'] 9999999974752427078783512115478515 10000000000000000000000000000000000000000 6;
  lit_fstep '2' ['5'] 99999999747524270787835121154785156 100000000000000000000000000000000000000000 2;
  lit_fstep '5' [] 999999997475242707878351211547851562 1000000000000000000000000000000000000000000 5;
  lit_fnil 9999999974752427078783512115478515625 10000000000000000000000000000000000000000000

noextract
let eps_bp () : Lemma (Lit.binary_power 43 == 8796093022208) =
  assert (Lit.binary_power 0 == 1);
  lit_bp 0;
  lit_bp 1;
  lit_bp 2;
  lit_bp 3;
  lit_bp 4;
  lit_bp 5;
  lit_bp 6;
  lit_bp 7;
  lit_bp 8;
  lit_bp 9;
  lit_bp 10;
  lit_bp 11;
  lit_bp 12;
  lit_bp 13;
  lit_bp 14;
  lit_bp 15;
  lit_bp 16;
  lit_bp 17;
  lit_bp 18;
  lit_bp 19;
  lit_bp 20;
  lit_bp 21;
  lit_bp 22;
  lit_bp 23;
  lit_bp 24;
  lit_bp 25;
  lit_bp 26;
  lit_bp 27;
  lit_bp 28;
  lit_bp 29;
  lit_bp 30;
  lit_bp 31;
  lit_bp 32;
  lit_bp 33;
  lit_bp 34;
  lit_bp 35;
  lit_bp 36;
  lit_bp 37;
  lit_bp 38;
  lit_bp 39;
  lit_bp 40;
  lit_bp 41;
  lit_bp 42;
  ()

#push-options "--fuel 0 --ifuel 0 --z3rlimit 50"
noextract
let eps_cert () : Lemma (Cert.exact_normal "0.0000009999999974752427078783512115478515625" 8796093 (-43)) =
  eps_frac ();
  eps_bp ();
  assert (Cert.normal_encoding 8796093 (-43));
  lit_cert_of "0.0000009999999974752427078783512115478515625" 8796093 (-43) 9999999974752427078783512115478515625 10000000000000000000000000000000000000000000
#pop-options

noextract
let qs_frac () : Lemma (Lit.integral ['0';'.';'0';'8';'8';'3';'8';'8';'3';'4';'6';'1';'3';'5';'6';'1';'6';'3';'0';'2';'4';'9';'0';'2';'3';'4';'3';'7';'5'] 0 false == Some (88388346135616302490234375, 1000000000000000000000000000)) =
  lit_istep '0' ['8';'8';'3';'8';'8';'3';'4';'6';'1';'3';'5';'6';'1';'6';'3';'0';'2';'4';'9';'0';'2';'3';'4';'3';'7';'5'];
  lit_fstep '0' ['8';'8';'3';'8';'8';'3';'4';'6';'1';'3';'5';'6';'1';'6';'3';'0';'2';'4';'9';'0';'2';'3';'4';'3';'7';'5'] 0 1 0;
  lit_fstep '8' ['8';'3';'8';'8';'3';'4';'6';'1';'3';'5';'6';'1';'6';'3';'0';'2';'4';'9';'0';'2';'3';'4';'3';'7';'5'] 0 10 8;
  lit_fstep '8' ['3';'8';'8';'3';'4';'6';'1';'3';'5';'6';'1';'6';'3';'0';'2';'4';'9';'0';'2';'3';'4';'3';'7';'5'] 8 100 8;
  lit_fstep '3' ['8';'8';'3';'4';'6';'1';'3';'5';'6';'1';'6';'3';'0';'2';'4';'9';'0';'2';'3';'4';'3';'7';'5'] 88 1000 3;
  lit_fstep '8' ['8';'3';'4';'6';'1';'3';'5';'6';'1';'6';'3';'0';'2';'4';'9';'0';'2';'3';'4';'3';'7';'5'] 883 10000 8;
  lit_fstep '8' ['3';'4';'6';'1';'3';'5';'6';'1';'6';'3';'0';'2';'4';'9';'0';'2';'3';'4';'3';'7';'5'] 8838 100000 8;
  lit_fstep '3' ['4';'6';'1';'3';'5';'6';'1';'6';'3';'0';'2';'4';'9';'0';'2';'3';'4';'3';'7';'5'] 88388 1000000 3;
  lit_fstep '4' ['6';'1';'3';'5';'6';'1';'6';'3';'0';'2';'4';'9';'0';'2';'3';'4';'3';'7';'5'] 883883 10000000 4;
  lit_fstep '6' ['1';'3';'5';'6';'1';'6';'3';'0';'2';'4';'9';'0';'2';'3';'4';'3';'7';'5'] 8838834 100000000 6;
  lit_fstep '1' ['3';'5';'6';'1';'6';'3';'0';'2';'4';'9';'0';'2';'3';'4';'3';'7';'5'] 88388346 1000000000 1;
  lit_fstep '3' ['5';'6';'1';'6';'3';'0';'2';'4';'9';'0';'2';'3';'4';'3';'7';'5'] 883883461 10000000000 3;
  lit_fstep '5' ['6';'1';'6';'3';'0';'2';'4';'9';'0';'2';'3';'4';'3';'7';'5'] 8838834613 100000000000 5;
  lit_fstep '6' ['1';'6';'3';'0';'2';'4';'9';'0';'2';'3';'4';'3';'7';'5'] 88388346135 1000000000000 6;
  lit_fstep '1' ['6';'3';'0';'2';'4';'9';'0';'2';'3';'4';'3';'7';'5'] 883883461356 10000000000000 1;
  lit_fstep '6' ['3';'0';'2';'4';'9';'0';'2';'3';'4';'3';'7';'5'] 8838834613561 100000000000000 6;
  lit_fstep '3' ['0';'2';'4';'9';'0';'2';'3';'4';'3';'7';'5'] 88388346135616 1000000000000000 3;
  lit_fstep '0' ['2';'4';'9';'0';'2';'3';'4';'3';'7';'5'] 883883461356163 10000000000000000 0;
  lit_fstep '2' ['4';'9';'0';'2';'3';'4';'3';'7';'5'] 8838834613561630 100000000000000000 2;
  lit_fstep '4' ['9';'0';'2';'3';'4';'3';'7';'5'] 88388346135616302 1000000000000000000 4;
  lit_fstep '9' ['0';'2';'3';'4';'3';'7';'5'] 883883461356163024 10000000000000000000 9;
  lit_fstep '0' ['2';'3';'4';'3';'7';'5'] 8838834613561630249 100000000000000000000 0;
  lit_fstep '2' ['3';'4';'3';'7';'5'] 88388346135616302490 1000000000000000000000 2;
  lit_fstep '3' ['4';'3';'7';'5'] 883883461356163024902 10000000000000000000000 3;
  lit_fstep '4' ['3';'7';'5'] 8838834613561630249023 100000000000000000000000 4;
  lit_fstep '3' ['7';'5'] 88388346135616302490234 1000000000000000000000000 3;
  lit_fstep '7' ['5'] 883883461356163024902343 10000000000000000000000000 7;
  lit_fstep '5' [] 8838834613561630249023437 100000000000000000000000000 5;
  lit_fnil 88388346135616302490234375 1000000000000000000000000000

noextract
let qs_bp () : Lemma (Lit.binary_power 27 == 134217728) =
  assert (Lit.binary_power 0 == 1);
  lit_bp 0;
  lit_bp 1;
  lit_bp 2;
  lit_bp 3;
  lit_bp 4;
  lit_bp 5;
  lit_bp 6;
  lit_bp 7;
  lit_bp 8;
  lit_bp 9;
  lit_bp 10;
  lit_bp 11;
  lit_bp 12;
  lit_bp 13;
  lit_bp 14;
  lit_bp 15;
  lit_bp 16;
  lit_bp 17;
  lit_bp 18;
  lit_bp 19;
  lit_bp 20;
  lit_bp 21;
  lit_bp 22;
  lit_bp 23;
  lit_bp 24;
  lit_bp 25;
  lit_bp 26;
  ()

#push-options "--fuel 0 --ifuel 0 --z3rlimit 50"
noextract
let qs_cert () : Lemma (Cert.exact_normal "0.088388346135616302490234375" 11863283 (-27)) =
  qs_frac ();
  qs_bp ();
  assert (Cert.normal_encoding 11863283 (-27));
  lit_cert_of "0.088388346135616302490234375" 11863283 (-27) 88388346135616302490234375 1000000000000000000000000000
#pop-options

noextract
let i1024_frac () : Lemma (Lit.integral ['0';'.';'0';'0';'0';'9';'7';'6';'5';'6';'2';'5'] 0 false == Some (9765625, 10000000000)) =
  lit_istep '0' ['0';'0';'9';'7';'6';'5';'6';'2';'5'];
  lit_fstep '0' ['0';'0';'9';'7';'6';'5';'6';'2';'5'] 0 1 0;
  lit_fstep '0' ['0';'9';'7';'6';'5';'6';'2';'5'] 0 10 0;
  lit_fstep '0' ['9';'7';'6';'5';'6';'2';'5'] 0 100 0;
  lit_fstep '9' ['7';'6';'5';'6';'2';'5'] 0 1000 9;
  lit_fstep '7' ['6';'5';'6';'2';'5'] 9 10000 7;
  lit_fstep '6' ['5';'6';'2';'5'] 97 100000 6;
  lit_fstep '5' ['6';'2';'5'] 976 1000000 5;
  lit_fstep '6' ['2';'5'] 9765 10000000 6;
  lit_fstep '2' ['5'] 97656 100000000 2;
  lit_fstep '5' [] 976562 1000000000 5;
  lit_fnil 9765625 10000000000

noextract
let i1024_bp () : Lemma (Lit.binary_power 33 == 8589934592) =
  assert (Lit.binary_power 0 == 1);
  lit_bp 0;
  lit_bp 1;
  lit_bp 2;
  lit_bp 3;
  lit_bp 4;
  lit_bp 5;
  lit_bp 6;
  lit_bp 7;
  lit_bp 8;
  lit_bp 9;
  lit_bp 10;
  lit_bp 11;
  lit_bp 12;
  lit_bp 13;
  lit_bp 14;
  lit_bp 15;
  lit_bp 16;
  lit_bp 17;
  lit_bp 18;
  lit_bp 19;
  lit_bp 20;
  lit_bp 21;
  lit_bp 22;
  lit_bp 23;
  lit_bp 24;
  lit_bp 25;
  lit_bp 26;
  lit_bp 27;
  lit_bp 28;
  lit_bp 29;
  lit_bp 30;
  lit_bp 31;
  lit_bp 32;
  ()

#push-options "--fuel 0 --ifuel 0 --z3rlimit 50"
noextract
let i1024_cert () : Lemma (Cert.exact_normal "0.0009765625" 8388608 (-33)) =
  i1024_frac ();
  i1024_bp ();
  assert (Cert.normal_encoding 8388608 (-33));
  lit_cert_of "0.0009765625" 8388608 (-33) 9765625 10000000000
#pop-options

noextract
let i128_frac () : Lemma (Lit.integral ['0';'.';'0';'0';'7';'8';'1';'2';'5'] 0 false == Some (78125, 10000000)) =
  lit_istep '0' ['0';'7';'8';'1';'2';'5'];
  lit_fstep '0' ['0';'7';'8';'1';'2';'5'] 0 1 0;
  lit_fstep '0' ['7';'8';'1';'2';'5'] 0 10 0;
  lit_fstep '7' ['8';'1';'2';'5'] 0 100 7;
  lit_fstep '8' ['1';'2';'5'] 7 1000 8;
  lit_fstep '1' ['2';'5'] 78 10000 1;
  lit_fstep '2' ['5'] 781 100000 2;
  lit_fstep '5' [] 7812 1000000 5;
  lit_fnil 78125 10000000

noextract
let i128_bp () : Lemma (Lit.binary_power 30 == 1073741824) =
  assert (Lit.binary_power 0 == 1);
  lit_bp 0;
  lit_bp 1;
  lit_bp 2;
  lit_bp 3;
  lit_bp 4;
  lit_bp 5;
  lit_bp 6;
  lit_bp 7;
  lit_bp 8;
  lit_bp 9;
  lit_bp 10;
  lit_bp 11;
  lit_bp 12;
  lit_bp 13;
  lit_bp 14;
  lit_bp 15;
  lit_bp 16;
  lit_bp 17;
  lit_bp 18;
  lit_bp 19;
  lit_bp 20;
  lit_bp 21;
  lit_bp 22;
  lit_bp 23;
  lit_bp 24;
  lit_bp 25;
  lit_bp 26;
  lit_bp 27;
  lit_bp 28;
  lit_bp 29;
  ()

#push-options "--fuel 0 --ifuel 0 --z3rlimit 50"
noextract
let i128_cert () : Lemma (Cert.exact_normal "0.0078125" 8388608 (-30)) =
  i128_frac ();
  i128_bp ();
  assert (Cert.normal_encoding 8388608 (-30));
  lit_cert_of "0.0078125" 8388608 (-30) 78125 10000000
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 50"
noextract
let eps_value () : Lemma (Lit.value "0.0000009999999974752427078783512115478515625" == FStar.Real.of_int 9999999974752427078783512115478515625 /. FStar.Real.of_int 10000000000000000000000000000000000000000000) =
  eps_frac ();
  lit_value_of "0.0000009999999974752427078783512115478515625" 9999999974752427078783512115478515625 10000000000000000000000000000000000000000000
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 50"
noextract
let qs_value () : Lemma (Lit.value "0.088388346135616302490234375" == FStar.Real.of_int 88388346135616302490234375 /. FStar.Real.of_int 1000000000000000000000000000) =
  qs_frac ();
  lit_value_of "0.088388346135616302490234375" 88388346135616302490234375 1000000000000000000000000000
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 50"
noextract
let i1024_value () : Lemma (Lit.value "0.0009765625" == FStar.Real.of_int 9765625 /. FStar.Real.of_int 10000000000) =
  i1024_frac ();
  lit_value_of "0.0009765625" 9765625 10000000000
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 50"
noextract
let i128_value () : Lemma (Lit.value "0.0078125" == FStar.Real.of_int 78125 /. FStar.Real.of_int 10000000) =
  i128_frac ();
  lit_value_of "0.0078125" 78125 10000000
#pop-options


(*** Literal approximations *)

#push-options "--z3rlimit 40 --fuel 0 --ifuel 0"

noextract
let eps_ap () : Lemma (eps_v %~ eps_f32 /\ eps_f32 >. 0.0R)
  = eps_cert ();
    CF.literal_approx "0.0000009999999974752427078783512115478515625" 8796093 (-43);
    eps_value ()

noextract
let qscale_ap () : Lemma (qscale_v %~ qscale_f32)
  = qs_cert ();
    CF.literal_approx "0.088388346135616302490234375" 11863283 (-27)

noextract
let inv1024_ap () : Lemma (inv1024_v %~ (1.0R /. 1024.0R))
  = i1024_cert ();
    CF.literal_approx "0.0009765625" 8388608 (-33);
    i1024_value ()

noextract
let inv128_ap () : Lemma (inv128_v %~ (1.0R /. 128.0R))
  = i128_cert ();
    CF.literal_approx "0.0078125" 8388608 (-30);
    i128_value ()

(*** Scalar approximation steps (explicit instances of the library laws) *)

noextract
let ap_zero () : Lemma ((zero #float) %~ 0.0R) = let _ = (a0 #float) in ()
noextract
let ap_one () : Lemma ((one #float) %~ 1.0R) = let _ = (a1 #float) in ()

noextract
let ap_add (x y : float) (r s : real)
  : Lemma (requires x %~ r /\ y %~ s) (ensures add x y %~ (r +. s)) = a_add x y r s
noextract
let ap_mul (x y : float) (r s : real)
  : Lemma (requires x %~ r /\ y %~ s) (ensures mul x y %~ (r *. s)) = a_mul x y r s
noextract
let ap_sub (x y : float) (r s : real)
  : Lemma (requires x %~ r /\ y %~ s) (ensures sub x y %~ (r -. s))
  = let _ = approx_sub #float in
    approx_apply2 #float #real #float #real #float #real (sub #float) RO.r_sub x y r s
noextract
let ap_neg (x : float) (r : real)
  : Lemma (requires x %~ r) (ensures neg x %~ (0.0R -. r))
  = let _ = approx_neg #float in
    approx_apply #float #real #float #real (neg #float) RO.r_neg x r
noextract
let ap_exp (x : float) (r : real)
  : Lemma (requires x %~ r) (ensures fexp x %~ Exp.exp r)
  = let _ = approx_fexp #float in
    approx_apply #float #real #float #real (fexp #float) RO.r_exp x r
noextract
let ap_div (x y : float) (r s : real)
  : Lemma (requires x %~ r /\ y %~ s /\ s =!= 0.0R) (ensures div x y %~ (r /. s))
  = let _ = approx_div #float in
    approx_apply2_dom #float #real #float #real #float #real #(fun _ -> True) #(fun s -> s =!= 0.0R)
      (div #float) RO.r_div x y r s
noextract
let ap_rsqrt (x : float) (r : real)
  : Lemma (requires x %~ r /\ r >. 0.0R) (ensures rsqrt x %~ Sqrt.rsqrt r)
  = let _ = approx_rsqrt #float in
    approx_apply_dom #float #real #float #real #(fun r -> r >. 0.0R) (rsqrt #float) RO.r_rsqrt x r
noextract
let ap_b2f (x : bf16) (r : real)
  : Lemma (requires x %~ r) (ensures b2f x %~ r) = C.c_bf16_32.fcast_approx x r
noextract
let ap_f2b (x : float) (r : real)
  : Lemma (requires x %~ r) (ensures f2b x %~ r) = C.c_32_bf16.fcast_approx x r

noextract
let exp_pos (r : real) : Lemma (Exp.exp r >. 0.0R) = Exp.exp_positive r

noextract
let ap_sigm (x : float) (r : real)
  : Lemma (requires x %~ r) (ensures sigm_f x %~ sigmoid r)
  = ap_neg x r;
    ap_exp (neg x) (0.0R -. r);
    exp_pos (0.0R -. r);
    ap_one ();
    ap_add one (fexp (neg x)) 1.0R (Exp.exp (0.0R -. r));
    ap_div one (add one (fexp (neg x))) 1.0R (1.0R +. Exp.exp (0.0R -. r))

noextract
let silu_eq (r d : real) : Lemma (requires d =!= 0.0R) (ensures r /. d == r *. (1.0R /. d)) = ()

noextract
let ap_silu (x : float) (r : real)
  : Lemma (requires x %~ r) (ensures silu_f x %~ silu r)
  = ap_neg x r;
    ap_exp (neg x) (0.0R -. r);
    exp_pos (0.0R -. r);
    ap_one ();
    ap_add one (fexp (neg x)) 1.0R (Exp.exp (0.0R -. r));
    ap_div x (add one (fexp (neg x))) r (1.0R +. Exp.exp (0.0R -. r));
    silu_eq r (1.0R +. Exp.exp (0.0R -. r))

#pop-options
(* ===== Sums ===== *)
open FStar.Real
open Kuiper
open KPlay.SharedSpec


#push-options "--z3rlimit 40 --fuel 0 --ifuel 0"

(*** Reduction contracts: a float sum approximates vsum of any real model *)

noextract
let fa (s : S.seq float) (n : nat) (f : nat -> real) : prop =
  n <= S.length s /\ (forall (i : nat). i < n ==> S.index s i %~ f i)

noextract
let csum_ok (s : S.seq float) (n : nat) (x : float) : prop =
  forall (f : nat -> real). fa s n f ==> x %~ vsum n f

noextract
let strd_il (n : nat) (f : nat -> real) (o s : nat)
  : Lemma (vsum (2 * n) (strd f o s) == vsum n (strd f o (2 * s)) +. vsum n (strd f (o + s) (2 * s)))
  = vsum_interleave2 n (strd f o s);
    let a1 (t : nat) : Lemma (strd (strd f o s) 0 2 t == strd f o (2 * s) t)
      = L.paren_mul_right s 2 t; L.swap_mul s 2 in
    let a2 (t : nat) : Lemma (strd (strd f o s) 1 2 t == strd f (o + s) (2 * s) t)
      = L.distributivity_add_right s 1 (2 * t); L.paren_mul_right s 2 t; L.swap_mul s 2 in
    FStar.Classical.forall_intro a1;
    FStar.Classical.forall_intro a2;
    vsum_ext n (strd (strd f o s) 0 2) (strd f o (2 * s));
    vsum_ext n (strd (strd f o s) 1 2) (strd f (o + s) (2 * s))

(*** contiguous_sum: leaves then halving tree *)

noextract
let inner (s : S.seq float) (n w vi vj : nat) (acc : float) : prop =
  forall (f : nat -> real). fa s n f ==> acc %~ vsum vj (shf f (vi * w))

noextract
let inner_init (s : S.seq float) (n w vi : nat) : Lemma (inner s n w vi 0 (zero #float))
  = let aux (f : nat -> real) : Lemma (fa s n f ==> (zero #float) %~ vsum 0 (shf f (vi * w)))
      = vsum_zero (shf f (vi * w)); ap_zero () in
    FStar.Classical.forall_intro aux

noextract
let inner_step (s : S.seq float) (n w vi vj : nat) (acc x : float)
  : Lemma (requires inner s n w vi vj acc /\ vi * w + vj < n /\ n <= S.length s /\ x == S.index s (vi * w + vj))
          (ensures inner s n w vi (vj + 1) (add acc x))
  = let aux (f : nat -> real) : Lemma (requires fa s n f) (ensures add acc x %~ vsum (vj + 1) (shf f (vi * w)))
      = vsum_succ2 vj (shf f (vi * w));
        ap_add acc x (vsum vj (shf f (vi * w))) (f (vi * w + vj)) in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux)

noextract
let leaves (s : S.seq float) (n w : nat) (p : S.seq float) (vi : nat) : prop =
  vi <= S.length p /\
  (forall (f : nat -> real). fa s n f ==> (forall (k : nat). k < vi ==> S.index p k %~ rblk f w k))

noextract
let leaves_init (s : S.seq float) (n w : nat) (p : S.seq float) : Lemma (leaves s n w p 0) = ()

noextract
let leaves_step (s : S.seq float) (n w : nat) (p : S.seq float) (vi : nat) (acc : float)
  : Lemma (requires leaves s n w p vi /\ vi < S.length p /\ inner s n w vi w acc)
          (ensures leaves s n w (S.upd p vi acc) (vi + 1))
  = ()

noextract
let tree (s : S.seq float) (n w count : nat) (p : S.seq float) (m : nat) : prop =
  0 < m /\ m <= S.length p /\
  (forall (f : nat -> real). fa s n f ==>
     (forall (k : nat). k < m ==> S.index p k %~ vsum (count / m) (strd (rblk f w) k m)))

noextract
let leaves_tree (s : S.seq float) (n w count : nat) (p : S.seq float)
  : Lemma (requires leaves s n w p count /\ 0 < count) (ensures tree s n w count p count)
  = let aux (f : nat -> real) (k : nat)
      : Lemma (requires k < count) (ensures vsum (count / count) (strd (rblk f w) k count) == rblk f w k)
      = vsum_one (strd (rblk f w) k count) in
    FStar.Classical.forall_intro_2 (fun f k -> FStar.Classical.move_requires (aux f) k)

noextract
let round (s : S.seq float) (n w count : nat) (p p' : S.seq float) (vo : nat)
  : Lemma (requires tree s n w count p (2 * vo) /\ 0 < vo /\ count % (2 * vo) == 0 /\
                    S.length p' == S.length p /\
                    (forall (k : nat). k < S.length p ==>
                       S.index p' k == (if k < vo then add (S.index p k) (S.index p (k + vo)) else S.index p k)))
          (ensures tree s n w count p' vo)
  = let q = count / (2 * vo) in
    L.euclidean_division_definition count (2 * vo);
    L.paren_mul_right q 2 vo;
    L.cancel_mul_div (q * 2) vo;
    assert (count / vo == 2 * q);
    let aux (f : nat -> real) (k : nat)
      : Lemma (requires fa s n f /\ k < vo /\ k < S.length p' /\ k + vo < S.length p)
              (ensures k < S.length p' /\ S.index p' k %~ vsum (count / vo) (strd (rblk f w) k vo))
      = strd_il q (rblk f w) k vo;
        ap_add (S.index p k) (S.index p (k + vo))
          (vsum q (strd (rblk f w) k (2 * vo))) (vsum q (strd (rblk f w) (k + vo) (2 * vo))) in
    FStar.Classical.forall_intro_2 (fun f k -> FStar.Classical.move_requires (aux f) k)

noextract
let tree_fin (s : S.seq float) (n w count : nat) (p : S.seq float)
  : Lemma (requires tree s n w count p 1 /\ n == count * w) (ensures csum_ok s n (S.index p 0))
  = let aux (f : nat -> real) : Lemma (requires fa s n f) (ensures S.index p 0 %~ vsum n f)
      = vsum_ext count (strd (rblk f w) 0 1) (rblk f w);
        vsum_block2 count w f in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux)

(* The halving offsets of a power-of-two count *)
noextract
let pw (v : nat) : bool = v = 1 || v = 2 || v = 4 || v = 8 || v = 16 || v = 32 || v = 64 || v = 128

noextract
let cnt_ok (c : nat) : bool = c = 16 || c = 32 || c = 256

noextract
let pw_step (count vo : nat)
  : Lemma (requires cnt_ok count /\ pw vo /\ 2 * vo <= count)
          (ensures count % (2 * vo) == 0 /\ (vo = 1 \/ (pw (vo / 2) /\ 2 * (vo / 2) <= count)))
  = ()

noextract
let pw_start (count : nat)
  : Lemma (requires cnt_ok count) (ensures pw (count / 2) /\ 2 * (count / 2) == count) = ()

(*** recurrent_sum(values[128]): stride-4 groups of four accumulators *)

noextract
let rin (s : S.seq float) (y vt : nat) (x0 x1 x2 x3 : float) : prop =
  forall (f : nat -> real). fa s 128 f ==>
    (x0 %~ vsum vt (strd f y 16) /\ x1 %~ vsum vt (strd f (y + 4) 16) /\
     x2 %~ vsum vt (strd f (y + 8) 16) /\ x3 %~ vsum vt (strd f (y + 12) 16))

noextract
let rin_init (s : S.seq float) (y : nat)
  : Lemma (rin s y 0 (zero #float) (zero #float) (zero #float) (zero #float))
  = let aux (f : nat -> real) : Lemma (requires fa s 128 f)
      (ensures (zero #float) %~ vsum 0 (strd f y 16) /\ (zero #float) %~ vsum 0 (strd f (y + 4) 16) /\
               (zero #float) %~ vsum 0 (strd f (y + 8) 16) /\ (zero #float) %~ vsum 0 (strd f (y + 12) 16))
      = ap_zero (); vsum_zero (strd f y 16); vsum_zero (strd f (y + 4) 16);
        vsum_zero (strd f (y + 8) 16); vsum_zero (strd f (y + 12) 16) in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux)

noextract
let rin_step (s : S.seq float) (y vt : nat) (x0 x1 x2 x3 e0 e1 e2 e3 : float)
  : Lemma (requires rin s y vt x0 x1 x2 x3 /\ y < 4 /\ vt < 8 /\ S.length s == 128 /\
                    e0 == S.index s (y + 16 * vt) /\ e1 == S.index s (y + 16 * vt + 4) /\
                    e2 == S.index s (y + 16 * vt + 8) /\ e3 == S.index s (y + 16 * vt + 12))
          (ensures rin s y (vt + 1) (add x0 e0) (add x1 e1) (add x2 e2) (add x3 e3))
  = let aux (f : nat -> real) : Lemma (requires fa s 128 f)
      (ensures add x0 e0 %~ vsum (vt + 1) (strd f y 16) /\ add x1 e1 %~ vsum (vt + 1) (strd f (y + 4) 16) /\
               add x2 e2 %~ vsum (vt + 1) (strd f (y + 8) 16) /\ add x3 e3 %~ vsum (vt + 1) (strd f (y + 12) 16))
      = vsum_succ2 vt (strd f y 16); vsum_succ2 vt (strd f (y + 4) 16);
        vsum_succ2 vt (strd f (y + 8) 16); vsum_succ2 vt (strd f (y + 12) 16);
        ap_add x0 e0 (vsum vt (strd f y 16)) (f (y + 16 * vt));
        ap_add x1 e1 (vsum vt (strd f (y + 4) 16)) (f (y + 4 + 16 * vt));
        ap_add x2 e2 (vsum vt (strd f (y + 8) 16)) (f (y + 8 + 16 * vt));
        ap_add x3 e3 (vsum vt (strd f (y + 12) 16)) (f (y + 12 + 16 * vt)) in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux)

noextract
let rpart (s : S.seq float) (y : nat) (v : float) : prop =
  forall (f : nat -> real). fa s 128 f ==> v %~ vsum 32 (strd f y 4)

noextract
let rin_fin (s : S.seq float) (y : nat) (x0 x1 x2 x3 : float)
  : Lemma (requires rin s y 8 x0 x1 x2 x3) (ensures rpart s y (add (add (add x0 x1) x2) x3))
  = let aux (f : nat -> real) : Lemma (requires fa s 128 f)
      (ensures add (add (add x0 x1) x2) x3 %~ vsum 32 (strd f y 4))
      = let a0 = vsum 8 (strd f y 16) in let a1 = vsum 8 (strd f (y + 4) 16) in
        let a2 = vsum 8 (strd f (y + 8) 16) in let a3 = vsum 8 (strd f (y + 12) 16) in
        ap_add x0 x1 a0 a1; ap_add (add x0 x1) x2 (a0 +. a1) a2; ap_add (add (add x0 x1) x2) x3 ((a0 +. a1) +. a2) a3;
        strd_il 16 f y 4; strd_il 8 f y 8; strd_il 8 f (y + 4) 8 in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux)

noextract
let rsum_fin (s : S.seq float) (a b c d : float)
  : Lemma (requires rpart s 0 a /\ rpart s 1 b /\ rpart s 2 c /\ rpart s 3 d)
          (ensures csum_ok s 128 (add (add a c) (add b d)))
  = let aux (f : nat -> real) : Lemma (requires fa s 128 f) (ensures add (add a c) (add b d) %~ vsum 128 f)
      = let v0 = vsum 32 (strd f 0 4) in let v1 = vsum 32 (strd f 1 4) in
        let v2 = vsum 32 (strd f 2 4) in let v3 = vsum 32 (strd f 3 4) in
        ap_add a c v0 v2; ap_add b d v1 v3; ap_add (add a c) (add b d) (v0 +. v2) (v1 +. v3);
        strd_il 64 f 0 1; strd_il 32 f 0 2; strd_il 32 f 1 2;
        vsum_ext 128 (strd f 0 1) f in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux)

#pop-options
(* ===== KS ===== *)
open FStar.Real
open Kuiper
open KPlay.SharedSpec


#push-options "--z3rlimit 40 --fuel 0 --ifuel 0"

(*** Prefix approximation of a sequence by a real sequence *)

noextract
let pb (s : S.seq bf16) (n : nat) (r : S.seq real) : prop =
  n <= S.length s /\ (forall (i : nat). i < n ==> S.index s i %~ at r i)

noextract
let pf (s : S.seq float) (n : nat) (r : S.seq real) : prop =
  n <= S.length s /\ (forall (i : nat). i < n ==> S.index s i %~ at r i)

(*** Spec-lambda bridges *)

noextract
let sqf (x : S.seq real) (o : int) (k : nat) : real = at x (o + k) *. at x (o + k)

noextract
let sq_sum_eq (x : S.seq real) (o : int) (n : nat) : Lemma (sq_sum x o n == vsum n (sqf x o))
  = vsum_ext_all n (sqf x o)

noextract
let dotf (w : S.seq real) (wo : int) (x : S.seq real) (k : nat) : real = at w (wo + k) *. at x k

noextract
let dot_eq (w : S.seq real) (wo : int) (x : S.seq real) (n : nat) : Lemma (dot w wo x n == vsum n (dotf w wo x))
  = vsum_ext_all n (dotf w wo x)

noextract
let sq_nonneg (a : real) : Lemma (a *. a >=. 0.0R) = ()

noextract
let sq_sum_nonneg (x : S.seq real) (o : int) (n : nat) : Lemma (sq_sum x o n >=. 0.0R)
  = sq_sum_eq x o n;
    let aux (k : nat) : Lemma (sqf x o k >=. 0.0R) = sq_nonneg (at x (o + k)) in
    FStar.Classical.forall_intro aux;
    vsum_nonneg n (sqf x o)

noextract
let nonneg_mul (a b : real) : Lemma (requires a >=. 0.0R /\ b >=. 0.0R) (ensures a *. b >=. 0.0R) = ()

noextract
let rdiv_1024 (x : real) : Lemma (rdiv x (of_int 1024) == x *. (1.0R /. 1024.0R)) = ()

noextract
let rdiv_128 (x : real) : Lemma (rdiv x 128.0R == x *. (1.0R /. 128.0R)) = ()

noextract
let rsqrt_t_pos (x : real) : Lemma (requires x >. 0.0R) (ensures rsqrt_t x == Sqrt.rsqrt x) = ()

(* rsqrt(s*c + eps) where s sums the squares f *)
noextract
let inv_core (sv : S.seq float) (n : nat) (s c : float) (cr : real) (f : nat -> real)
  : Lemma (requires csum_ok sv n s /\ fa sv n f /\ c %~ cr /\ cr >=. 0.0R /\ vsum n f >=. 0.0R)
          (ensures rsqrt (add (mul s c) eps_v) %~ rsqrt_t (vsum n f *. cr +. eps_f32))
  = ap_mul s c (vsum n f) cr;
    eps_ap ();
    ap_add (mul s c) eps_v (vsum n f *. cr) eps_f32;
    nonneg_mul (vsum n f) cr;
    ap_rsqrt (add (mul s c) eps_v) (vsum n f *. cr +. eps_f32);
    rsqrt_t_pos (vsum n f *. cr +. eps_f32)

(* sv[k] = x*x with x = b2f sr[o+k] *)
noextract
let sq_vals (sr : S.seq bf16) (o : nat) (sv : S.seq float) (n : nat) : prop =
  n <= S.length sv /\ o + n <= S.length sr /\
  (forall (k : nat). k < n ==> S.index sv k == mul (b2f (S.index sr (o + k))) (b2f (S.index sr (o + k))))

noextract
let sq_fa (sr : S.seq bf16) (o : nat) (sv : S.seq float) (n m : nat) (rx : S.seq real)
  : Lemma (requires sq_vals sr o sv n /\ pb sr m rx /\ o + n <= m) (ensures fa sv n (sqf rx o))
  = let e (k : nat) : Lemma (k < n ==> S.index sv k %~ sqf rx o k)
      = if k < n then begin
          ap_b2f (S.index sr (o + k)) (at rx (o + k));
          ap_mul (b2f (S.index sr (o + k))) (b2f (S.index sr (o + k))) (at rx (o + k)) (at rx (o + k))
        end in
    FStar.Classical.forall_intro e

(*** RMSNorm(1+w) over 1024 *)

noextract
let norm_inv_r (rx : S.seq real) : real = rsqrt_t (rdiv (sq_sum rx 0 1024) (of_int 1024) +. eps_f32)

noextract
let norm_inv_ok (sr : S.seq bf16) (inv : float) : prop =
  forall (rx : S.seq real). pb sr 1024 rx ==> inv %~ norm_inv_r rx

noextract
let norm_inv (sr : S.seq bf16) (sv : S.seq float) (s : float)
  : Lemma (requires sq_vals sr 0 sv 1024 /\ csum_ok sv 1024 s)
          (ensures norm_inv_ok sr (rsqrt (add (mul s inv1024_v) eps_v)))
  = let aux (rx : S.seq real) : Lemma (requires pb sr 1024 rx)
        (ensures rsqrt (add (mul s inv1024_v) eps_v) %~ norm_inv_r rx)
      = sq_fa sr 0 sv 1024 1024 rx;
        sq_sum_eq rx 0 1024; sq_sum_nonneg rx 0 1024;
        inv1024_ap ();
        inv_core sv 1024 s inv1024_v (1.0R /. 1024.0R) (sqf rx 0);
        rdiv_1024 (sq_sum rx 0 1024) in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux)

noextract
let norm_el_r (rx rw : S.seq real) (woff : nat) (i : nat) : real =
  (at rx i *. norm_inv_r rx) *. (1.0R +. at rw (woff + i))

noextract
let norm_el_ok (sr sw : S.seq bf16) (woff i : nat) (v : bf16) : prop =
  forall (rx rw : S.seq real). pb sr 1024 rx /\ pb sw (S.length sw) rw ==> v %~ norm_el_r rx rw woff i

noextract
let norm_el (sr sw : S.seq bf16) (woff i : nat) (inv : float)
  : Lemma (requires norm_inv_ok sr inv /\ i < 1024 /\ i < S.length sr /\ woff + i < S.length sw)
          (ensures norm_el_ok sr sw woff i
                     (f2b (mul (mul (b2f (S.index sr i)) inv) (add one (b2f (S.index sw (woff + i)))))))
  = let x = S.index sr i in let w = S.index sw (woff + i) in
    let aux (rx rw : S.seq real) : Lemma (requires pb sr 1024 rx /\ pb sw (S.length sw) rw)
        (ensures f2b (mul (mul (b2f x) inv) (add one (b2f w))) %~ norm_el_r rx rw woff i)
      = ap_b2f x (at rx i); ap_b2f w (at rw (woff + i)); ap_one ();
        ap_mul (b2f x) inv (at rx i) (norm_inv_r rx);
        ap_add one (b2f w) 1.0R (at rw (woff + i));
        ap_mul (mul (b2f x) inv) (add one (b2f w)) (at rx i *. norm_inv_r rx) (1.0R +. at rw (woff + i));
        ap_f2b (mul (mul (b2f x) inv) (add one (b2f w))) (norm_el_r rx rw woff i) in
    FStar.Classical.forall_intro_2 (fun rx rw -> FStar.Classical.move_requires (aux rx) rw)

(*** Gated RMSNorm over 16 heads of 128 *)

noextract
let o_gn_n : nat = 19458080

noextract
let gn_inv_r (rc : S.seq real) (h : nat) : real =
  rsqrt_t (rdiv (sq_sum rc (h * 128) 128) 128.0R +. eps_f32)

noextract
let gn_inv_ok (sc : S.seq bf16) (h : nat) (inv : float) : prop =
  forall (rc : S.seq real). pb sc 2048 rc ==> inv %~ gn_inv_r rc h

noextract
let gn_inv (sc : S.seq bf16) (h : nat) (sv : S.seq float) (s : float)
  : Lemma (requires h < 16 /\ sq_vals sc (h * 128) sv 128 /\ csum_ok sv 128 s)
          (ensures gn_inv_ok sc h (rsqrt (add (mul s inv128_v) eps_v)))
  = let aux (rc : S.seq real) : Lemma (requires pb sc 2048 rc)
        (ensures rsqrt (add (mul s inv128_v) eps_v) %~ gn_inv_r rc h)
      = sq_fa sc (h * 128) sv 128 2048 rc;
        sq_sum_eq rc (h * 128) 128; sq_sum_nonneg rc (h * 128) 128;
        inv128_ap ();
        inv_core sv 128 s inv128_v (1.0R /. 128.0R) (sqf rc (h * 128));
        rdiv_128 (sq_sum rc (h * 128) 128) in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux)

noextract
let gn_el_r (rw rc rz : S.seq real) (c : nat) : real =
  (at rw (o_gn_n + c % 128) *. (at rc c *. gn_inv_r rc (c / 128))) *. silu (at rz c)

noextract
let gn_el_ok (sw sc sz : S.seq bf16) (c : nat) (v : bf16) : prop =
  forall (rw rc rz : S.seq real). pb sw (S.length sw) rw /\ pb sc 2048 rc /\ pb sz 2048 rz ==>
    v %~ gn_el_r rw rc rz c

noextract
let hdec (h i : nat) : Lemma (requires i < 128) (ensures (h * 128 + i) / 128 == h /\ (h * 128 + i) % 128 == i)
  = L.lemma_div_plus i h 128; L.small_div i 128; L.lemma_mod_plus i h 128; L.small_mod i 128

noextract
let gn_el (sw sc sz : S.seq bf16) (h i : nat) (inv : float)
  : Lemma (requires gn_inv_ok sc h inv /\ h < 16 /\ i < 128 /\ S.length sc == 2048 /\ S.length sz == 2048 /\
                    o_gn_n + i < S.length sw)
          (ensures gn_el_ok sw sc sz (h * 128 + i)
             (f2b (mul (b2f (f2b (mul (b2f (S.index sw (o_gn_n + i)))
                                      (b2f (f2b (mul (b2f (S.index sc (h * 128 + i))) inv))))))
                       (silu_f (b2f (S.index sz (h * 128 + i)))))))
  = let c = h * 128 + i in
    hdec h i;
    let x = S.index sc c in let w = S.index sw (o_gn_n + i) in let zz = S.index sz c in
    let aux (rw rc rz : S.seq real) : Lemma (requires pb sw (S.length sw) rw /\ pb sc 2048 rc /\ pb sz 2048 rz)
        (ensures f2b (mul (b2f (f2b (mul (b2f w) (b2f (f2b (mul (b2f x) inv)))))) (silu_f (b2f zz)))
                 %~ gn_el_r rw rc rz c)
      = ap_b2f x (at rc c); ap_mul (b2f x) inv (at rc c) (gn_inv_r rc h);
        ap_f2b (mul (b2f x) inv) (at rc c *. gn_inv_r rc h);
        ap_b2f (f2b (mul (b2f x) inv)) (at rc c *. gn_inv_r rc h);
        ap_b2f w (at rw (o_gn_n + i));
        ap_mul (b2f w) (b2f (f2b (mul (b2f x) inv))) (at rw (o_gn_n + i)) (at rc c *. gn_inv_r rc h);
        let sc_r = at rw (o_gn_n + i) *. (at rc c *. gn_inv_r rc h) in
        ap_f2b (mul (b2f w) (b2f (f2b (mul (b2f x) inv)))) sc_r;
        ap_b2f (f2b (mul (b2f w) (b2f (f2b (mul (b2f x) inv))))) sc_r;
        ap_b2f zz (at rz c); ap_silu (b2f zz) (at rz c);
        ap_mul (b2f (f2b (mul (b2f w) (b2f (f2b (mul (b2f x) inv)))))) (silu_f (b2f zz)) sc_r (silu (at rz c));
        ap_f2b (mul (b2f (f2b (mul (b2f w) (b2f (f2b (mul (b2f x) inv)))))) (silu_f (b2f zz))) (sc_r *. silu (at rz c)) in
    let aux3 (rw rc rz : S.seq real)
      : Lemma ((pb sw (S.length sw) rw /\ pb sc 2048 rc /\ pb sz 2048 rz) ==>
               f2b (mul (b2f (f2b (mul (b2f w) (b2f (f2b (mul (b2f x) inv)))))) (silu_f (b2f zz)))
                 %~ gn_el_r rw rc rz c)
      = FStar.Classical.move_requires (aux rw rc) rz in
    FStar.Classical.forall_intro_3 aux3

(*** Per-head l2 norm of q and k (heads at multiples of 128 below 4096) *)

noextract
let qk_inv_r (rq : S.seq real) (st : int) : real = rsqrt_t (sq_sum rq st 128 +. eps_f32)

noextract
let qk_inv_ok (sq : S.seq bf16) (st : nat) (inv : float) : prop =
  forall (rq : S.seq real). pb sq 6144 rq ==> inv %~ qk_inv_r rq st

noextract
let qk_vals (sq : S.seq bf16) (st : nat) (sv : S.seq float) : prop =
  128 <= S.length sv /\ st + 128 <= S.length sq /\
  (forall (k : nat). k < 128 ==>
     S.index sv k == b2f (f2b (mul (b2f (S.index sq (st + k))) (b2f (S.index sq (st + k))))))

noextract
let qk_inv (sq : S.seq bf16) (st : nat) (sv : S.seq float) (s : float)
  : Lemma (requires qk_vals sq st sv /\ st + 128 <= 6144 /\ csum_ok sv 128 s)
          (ensures qk_inv_ok sq st (b2f (f2b (rsqrt (b2f (f2b (add (b2f (f2b s)) eps_v)))))))
  = let aux (rq : S.seq real) : Lemma (requires pb sq 6144 rq)
        (ensures b2f (f2b (rsqrt (b2f (f2b (add (b2f (f2b s)) eps_v))))) %~ qk_inv_r rq st)
      = let f = sqf rq st in
        let e (k : nat) : Lemma (k < 128 ==> S.index sv k %~ f k)
          = if k < 128 then begin
              let x = S.index sq (st + k) in
              ap_b2f x (at rq (st + k));
              ap_mul (b2f x) (b2f x) (at rq (st + k)) (at rq (st + k));
              ap_f2b (mul (b2f x) (b2f x)) (f k); ap_b2f (f2b (mul (b2f x) (b2f x))) (f k)
            end in
        FStar.Classical.forall_intro e;
        sq_sum_eq rq st 128; sq_sum_nonneg rq st 128;
        let ss = vsum 128 f in
        ap_f2b s ss; ap_b2f (f2b s) ss; eps_ap ();
        ap_add (b2f (f2b s)) eps_v ss eps_f32;
        ap_f2b (add (b2f (f2b s)) eps_v) (ss +. eps_f32);
        ap_b2f (f2b (add (b2f (f2b s)) eps_v)) (ss +. eps_f32);
        ap_rsqrt (b2f (f2b (add (b2f (f2b s)) eps_v))) (ss +. eps_f32);
        rsqrt_t_pos (ss +. eps_f32);
        ap_f2b (rsqrt (b2f (f2b (add (b2f (f2b s)) eps_v)))) (qk_inv_r rq st);
        ap_b2f (f2b (rsqrt (b2f (f2b (add (b2f (f2b s)) eps_v))))) (qk_inv_r rq st) in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux)

noextract
let qk_el_r (rq : S.seq real) (i : nat) : real = at rq i *. qk_inv_r rq ((i / 128) * 128)

noextract
let qk_el_ok (sq : S.seq bf16) (i : nat) (v : bf16) : prop =
  forall (rq : S.seq real). pb sq 6144 rq ==> v %~ qk_el_r rq i

noextract
let qk_el (sq : S.seq bf16) (hh k : nat) (inv : float) (x : bf16)
  : Lemma (requires qk_inv_ok sq (hh * 128) inv /\ k < 128 /\ hh * 128 + k < 6144 /\ S.length sq == 6144 /\
                    x == S.index sq (hh * 128 + k))
          (ensures qk_el_ok sq (hh * 128 + k) (f2b (mul (b2f x) inv)))
  = hdec hh k;
    let aux (rq : S.seq real) : Lemma (requires pb sq 6144 rq)
        (ensures f2b (mul (b2f x) inv) %~ qk_el_r rq (hh * 128 + k))
      = ap_b2f x (at rq (hh * 128 + k));
        ap_mul (b2f x) inv (at rq (hh * 128 + k)) (qk_inv_r rq (hh * 128));
        ap_f2b (mul (b2f x) inv) (qk_el_r rq (hh * 128 + k)) in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux)

(*** Causal conv1d (4 taps) + silu *)

noextract
let o_conv_n : nat = 19433472

noextract
let fma_ok (v x y z : float) : prop =
  forall (xr yr zr : real). x %~ xr /\ y %~ yr /\ z %~ zr ==> v %~ RO.r_fma xr yr zr

noextract
let fma_el (v x y z : float) (xr yr zr : real)
  : Lemma (requires fma_ok v x y z /\ x %~ xr /\ y %~ yr /\ z %~ zr) (ensures v %~ (xr *. yr +. zr)) = ()

noextract
let cv_r (rw rc rq : S.seq real) (c : nat) : real =
  at rq c *. at rw (o_conv_n + 4 * c + 3) +.
  (at rc (4 * c + 3) *. at rw (o_conv_n + 4 * c + 2) +.
   (at rc (4 * c + 2) *. at rw (o_conv_n + 4 * c + 1) +.
    (at rc (4 * c + 1) *. at rw (o_conv_n + 4 * c) +. 0.0R)))

noextract
let conv_ok (sw sc sq : S.seq bf16) (c : nat) (y : bf16) : prop =
  forall (rw rc rq : S.seq real). pb sw (S.length sw) rw /\ pb sc 24576 rc /\ pb sq 6144 rq ==>
    y %~ silu (cv_r rw rc rq c)

noextract
let conv_el (sw sc sq : S.seq bf16) (c : nat) (v0 v1 v2 v3 : float)
  : Lemma (requires c < 6144 /\ S.length sc == 24576 /\ S.length sq == 6144 /\ o_conv_n + 4 * c + 3 < S.length sw /\
                    fma_ok v0 (b2f (S.index sc (4 * c + 1))) (b2f (S.index sw (o_conv_n + 4 * c))) (zero #float) /\
                    fma_ok v1 (b2f (S.index sc (4 * c + 2))) (b2f (S.index sw (o_conv_n + 4 * c + 1))) v0 /\
                    fma_ok v2 (b2f (S.index sc (4 * c + 3))) (b2f (S.index sw (o_conv_n + 4 * c + 2))) v1 /\
                    fma_ok v3 (b2f (S.index sq c)) (b2f (S.index sw (o_conv_n + 4 * c + 3))) v2)
          (ensures conv_ok sw sc sq c (f2b (silu_f (b2f (f2b v3)))))
  = let aux (rw rc rq : S.seq real)
      : Lemma (requires pb sw (S.length sw) rw /\ pb sc 24576 rc /\ pb sq 6144 rq)
              (ensures f2b (silu_f (b2f (f2b v3))) %~ silu (cv_r rw rc rq c))
      = ap_zero ();
        ap_b2f (S.index sc (4 * c + 1)) (at rc (4 * c + 1));
        ap_b2f (S.index sc (4 * c + 2)) (at rc (4 * c + 2));
        ap_b2f (S.index sc (4 * c + 3)) (at rc (4 * c + 3));
        ap_b2f (S.index sq c) (at rq c);
        ap_b2f (S.index sw (o_conv_n + 4 * c)) (at rw (o_conv_n + 4 * c));
        ap_b2f (S.index sw (o_conv_n + 4 * c + 1)) (at rw (o_conv_n + 4 * c + 1));
        ap_b2f (S.index sw (o_conv_n + 4 * c + 2)) (at rw (o_conv_n + 4 * c + 2));
        ap_b2f (S.index sw (o_conv_n + 4 * c + 3)) (at rw (o_conv_n + 4 * c + 3));
        let r0 = at rc (4 * c + 1) *. at rw (o_conv_n + 4 * c) +. 0.0R in
        let r1 = at rc (4 * c + 2) *. at rw (o_conv_n + 4 * c + 1) +. r0 in
        let r2 = at rc (4 * c + 3) *. at rw (o_conv_n + 4 * c + 2) +. r1 in
        fma_el v0 (b2f (S.index sc (4 * c + 1))) (b2f (S.index sw (o_conv_n + 4 * c))) (zero #float)
          (at rc (4 * c + 1)) (at rw (o_conv_n + 4 * c)) 0.0R;
        fma_el v1 (b2f (S.index sc (4 * c + 2))) (b2f (S.index sw (o_conv_n + 4 * c + 1))) v0
          (at rc (4 * c + 2)) (at rw (o_conv_n + 4 * c + 1)) r0;
        fma_el v2 (b2f (S.index sc (4 * c + 3))) (b2f (S.index sw (o_conv_n + 4 * c + 2))) v1
          (at rc (4 * c + 3)) (at rw (o_conv_n + 4 * c + 2)) r1;
        fma_el v3 (b2f (S.index sq c)) (b2f (S.index sw (o_conv_n + 4 * c + 3))) v2
          (at rq c) (at rw (o_conv_n + 4 * c + 3)) r2;
        ap_f2b v3 (cv_r rw rc rq c); ap_b2f (f2b v3) (cv_r rw rc rq c);
        ap_silu (b2f (f2b v3)) (cv_r rw rc rq c);
        ap_f2b (silu_f (b2f (f2b v3))) (silu (cv_r rw rc rq c)) in
    let aux3 (rw rc rq : S.seq real)
      : Lemma ((pb sw (S.length sw) rw /\ pb sc 24576 rc /\ pb sq 6144 rq) ==>
               f2b (silu_f (b2f (f2b v3))) %~ silu (cv_r rw rc rq c))
      = FStar.Classical.move_requires (aux rw rc) rq in
    FStar.Classical.forall_intro_3 aux3


(*** Gated delta-rule recurrence *)

noextract
let o_alog_n : nat = 19458048
noextract
let o_dtb_n : nat = 19458064

noeq type krset = { rw : S.seq real; rq : S.seq real; ra : S.seq real; rb : S.seq real; rs : S.seq real }

noeq type rins = { iw : S.seq bf16; iq : S.seq bf16; ip : S.seq bf16; ib : S.seq bf16; ist : S.seq float }

noextract
let mk_ins (sw sq sp sb : S.seq bf16) (ss : S.seq float) : rins = { iw = sw; iq = sq; ip = sp; ib = sb; ist = ss }

noextract
let rprem (ii : rins) (r : krset) : prop =
  pb ii.iw (S.length ii.iw) r.rw /\ pb ii.iq 6144 r.rq /\ pb ii.ip 16 r.ra /\ pb ii.ib 16 r.rb /\
  pf ii.ist 262144 r.rs

noextract
let rlens (ii : rins) : prop =
  o_dtb_n + 16 <= S.length ii.iw /\ S.length ii.iq == 6144 /\ S.length ii.ip >= 16 /\
  S.length ii.ib >= 16 /\ S.length ii.ist == 262144

noextract
let sidx (h i j : nat) : nat = h * 16384 + i * 128 + j

noextract
let sidx_dec (h i j : nat)
  : Lemma (requires h < 16 /\ i < 128 /\ j < 128)
          (ensures sidx h i j < 262144 /\ sidx h i j / 16384 == h /\ (sidx h i j % 16384) / 128 == i /\
                   sidx h i j % 128 == j)
  = let r = i * 128 + j in
    assert (r < 16384);
    L.lemma_div_plus r h 16384; L.small_div r 16384;
    L.lemma_mod_plus r h 16384; L.small_mod r 16384;
    hdec i j;
    L.lemma_mod_plus (i * 128 + j) (h * 128) 128;
    assert (h * 16384 == (h * 128) * 128)

noextract
let dec_r (r : krset) (h : nat) : real =
  Exp.exp ((0.0R -. Exp.exp (at r.rw (o_alog_n + h))) *. softplus (at r.ra h +. at r.rw (o_dtb_n + h)))

noextract
let memf (r : krset) (dec : real) (h j : nat) (i : nat) : real =
  (at r.rs (sidx h i j) *. dec) *. at r.rq (2048 + h * 128 + i)

noextract
let delta_r (r : krset) (h j : nat) : real =
  (at r.rq (4096 + h * 128 + j) -. vsum 128 (memf r (dec_r r h) h j)) *. at r.rb h

noextract
let st_r (r : krset) (h i j : nat) : real =
  at r.rs (sidx h i j) *. dec_r r h +. at r.rq (2048 + h * 128 + i) *. delta_r r h j

noextract
let coref (r : krset) (h j : nat) (i : nat) : real = st_r r h i j *. (at r.rq (h * 128 + i) *. qscale_f32)

noextract
let core_r (r : krset) (h j : nat) : real = vsum 128 (coref r h j)

noextract
let dec_f (sw sp : S.seq bf16) (h : nat{o_dtb_n + h < S.length sw /\ h < S.length sp}) : float =
  fexp (mul (neg (fexp (b2f (S.index sw (o_alog_n + h)))))
            (SP.softplus20 (add (b2f (S.index sp h)) (b2f (S.index sw (o_dtb_n + h))))))

noextract
let dec_ok (ii : rins) (h : nat) (d : float) : prop = forall (r : krset). rprem ii r ==> d %~ dec_r r h

noextract
let sp_eq (x : real) : Lemma (SP.model x == softplus x) = ()

noextract
let dec_lemma (ii : rins) (h : nat)
  : Lemma (requires rlens ii /\ h < 16) (ensures dec_ok ii h (dec_f ii.iw ii.ip h))
  = let aux (r : krset) : Lemma (requires rprem ii r) (ensures dec_f ii.iw ii.ip h %~ dec_r r h)
      = let pa = S.index ii.ip h in let wd = S.index ii.iw (o_dtb_n + h) in let wl = S.index ii.iw (o_alog_n + h) in
        ap_b2f pa (at r.ra h); ap_b2f wd (at r.rw (o_dtb_n + h));
        ap_add (b2f pa) (b2f wd) (at r.ra h) (at r.rw (o_dtb_n + h));
        SP.softplus20_approx (add (b2f pa) (b2f wd)) (at r.ra h +. at r.rw (o_dtb_n + h));
        sp_eq (at r.ra h +. at r.rw (o_dtb_n + h));
        ap_b2f wl (at r.rw (o_alog_n + h));
        ap_exp (b2f wl) (at r.rw (o_alog_n + h));
        ap_neg (fexp (b2f wl)) (Exp.exp (at r.rw (o_alog_n + h)));
        ap_mul (neg (fexp (b2f wl))) (SP.softplus20 (add (b2f pa) (b2f wd)))
          (0.0R -. Exp.exp (at r.rw (o_alog_n + h))) (softplus (at r.ra h +. at r.rw (o_dtb_n + h)));
        ap_exp (mul (neg (fexp (b2f wl))) (SP.softplus20 (add (b2f pa) (b2f wd))))
          ((0.0R -. Exp.exp (at r.rw (o_alog_n + h))) *. softplus (at r.ra h +. at r.rw (o_dtb_n + h))) in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux)

(* first pass of a column: upd[i] = st*decay, prod[i] = upd[i]*k[i] *)
noextract
let col1 (ii : rins) (h j : nat) (decay : float) (su sr : S.seq float) (n : nat) : prop =
  n <= 128 /\ h < 16 /\ j < 128 /\ S.length su == 128 /\ S.length sr == 128 /\ rlens ii /\
  (forall (i : nat). i < n ==>
     (sidx_dec h i j;
      S.index su i == mul (S.index ii.ist (sidx h i j)) decay /\
      S.index sr i == mul (S.index su i) (b2f (S.index ii.iq (2048 + h * 128 + i)))))

noextract
let delta_ok (ii : rins) (h j : nat) (d : float) : prop = forall (r : krset). rprem ii r ==> d %~ delta_r r h j

noextract
let delta_lemma (ii : rins) (h j : nat) (decay : float) (su sr : S.seq float) (m : float)
  : Lemma (requires col1 ii h j decay su sr 128 /\ dec_ok ii h decay /\ csum_ok sr 128 m)
          (ensures delta_ok ii h j (mul (sub (b2f (S.index ii.iq (4096 + h * 128 + j))) m) (b2f (S.index ii.ib h))))
  = let aux (r : krset) : Lemma (requires rprem ii r)
        (ensures mul (sub (b2f (S.index ii.iq (4096 + h * 128 + j))) m) (b2f (S.index ii.ib h)) %~ delta_r r h j)
      = let dr = dec_r r h in
        let e (i : nat) : Lemma (i < 128 ==> S.index sr i %~ memf r dr h j i)
          = if i < 128 then begin
              sidx_dec h i j;
              let k = S.index ii.iq (2048 + h * 128 + i) in
              ap_mul (S.index ii.ist (sidx h i j)) decay (at r.rs (sidx h i j)) dr;
              ap_b2f k (at r.rq (2048 + h * 128 + i));
              ap_mul (S.index su i) (b2f k) (at r.rs (sidx h i j) *. dr) (at r.rq (2048 + h * 128 + i))
            end in
        FStar.Classical.forall_intro e;
        assert (fa sr 128 (memf r dr h j));
        let v = S.index ii.iq (4096 + h * 128 + j) in
        ap_b2f v (at r.rq (4096 + h * 128 + j));
        ap_sub (b2f v) m (at r.rq (4096 + h * 128 + j)) (vsum 128 (memf r dr h j));
        ap_b2f (S.index ii.ib h) (at r.rb h);
        ap_mul (sub (b2f v) m) (b2f (S.index ii.ib h))
          (at r.rq (4096 + h * 128 + j) -. vsum 128 (memf r dr h j)) (at r.rb h) in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux)

noextract
let apf (v : float) (r : real) : prop = v %~ r
noextract
let apb (v : bf16) (r : real) : prop = v %~ r

[@@"opaque_to_smt"]
noextract
let st_ok (ii : rins) (h i j : nat) (v : float) : prop = forall (r : krset). rprem ii r ==> apf v (st_r r h i j)

[@@"opaque_to_smt"]
noextract
let p2_ok (ii : rins) (h j i : nat) (v : float) : prop = forall (r : krset). rprem ii r ==> apf v (coref r h j i)

noextract
let st_intro (ii : rins) (h i j : nat) (v : float)
  : Lemma (requires forall (r : krset). rprem ii r ==> v %~ st_r r h i j) (ensures st_ok ii h i j v)
  = reveal_opaque (`%st_ok) (st_ok ii h i j v)

noextract
let p2_intro (ii : rins) (h j i : nat) (v : float)
  : Lemma (requires forall (r : krset). rprem ii r ==> v %~ coref r h j i) (ensures p2_ok ii h j i v)
  = reveal_opaque (`%p2_ok) (p2_ok ii h j i v)

noextract
let nx_f (ii : rins) (h i : nat{h < 16 /\ i < 128 /\ S.length ii.iq == 6144}) (u delta : float) : float =
  add u (mul (b2f (S.index ii.iq (2048 + h * 128 + i))) delta)

noextract
let st_el (ii : rins) (h i j : nat) (decay delta u : float)
  : Lemma (requires rlens ii /\ h < 16 /\ i < 128 /\ j < 128 /\ dec_ok ii h decay /\ delta_ok ii h j delta /\
                    (sidx_dec h i j; u == mul (S.index ii.ist (sidx h i j)) decay))
          (ensures st_ok ii h i j (nx_f ii h i u delta) /\
                   p2_ok ii h j i (mul (nx_f ii h i u delta) (mul (b2f (S.index ii.iq (h * 128 + i))) qscale_v)))
  = sidx_dec h i j;
    let k = S.index ii.iq (2048 + h * 128 + i) in
    let q = S.index ii.iq (h * 128 + i) in
    let nx = nx_f ii h i u delta in
    let aux (r : krset) : Lemma (requires rprem ii r)
        (ensures nx %~ st_r r h i j /\ mul nx (mul (b2f q) qscale_v) %~ coref r h j i)
      = ap_mul (S.index ii.ist (sidx h i j)) decay (at r.rs (sidx h i j)) (dec_r r h);
        ap_b2f k (at r.rq (2048 + h * 128 + i));
        ap_mul (b2f k) delta (at r.rq (2048 + h * 128 + i)) (delta_r r h j);
        ap_add u (mul (b2f k) delta) (at r.rs (sidx h i j) *. dec_r r h) (at r.rq (2048 + h * 128 + i) *. delta_r r h j);
        ap_b2f q (at r.rq (h * 128 + i)); qscale_ap ();
        ap_mul (b2f q) qscale_v (at r.rq (h * 128 + i)) qscale_f32;
        ap_mul nx (mul (b2f q) qscale_v) (st_r r h i j) (at r.rq (h * 128 + i) *. qscale_f32) in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux);
    assert (forall (r : krset). rprem ii r ==> nx %~ st_r r h i j);
    assert (forall (r : krset). rprem ii r ==> mul nx (mul (b2f q) qscale_v) %~ coref r h j i);
    reveal_opaque (`%st_ok) (st_ok ii h i j nx);
    reveal_opaque (`%p2_ok) (p2_ok ii h j i (mul nx (mul (b2f q) qscale_v)))

[@@"opaque_to_smt"]
noextract
let core_ok (ii : rins) (h j : nat) (v : bf16) : prop = forall (r : krset). rprem ii r ==> apb v (core_r r h j)

noextract
let core_lemma (ii : rins) (h j : nat) (sr : S.seq float) (cv : float)
  : Lemma (requires S.length sr == 128 /\ (forall (i : nat). i < 128 ==> p2_ok ii h j i (S.index sr i)) /\
                    csum_ok sr 128 cv)
          (ensures core_ok ii h j (f2b cv))
  = let aux (r : krset) : Lemma (requires rprem ii r) (ensures f2b cv %~ core_r r h j)
      = let e (i : nat) : Lemma (i < 128 ==> S.index sr i %~ coref r h j i)
          = if i < 128 then reveal_opaque (`%p2_ok) (p2_ok ii h j i (S.index sr i)) in
        FStar.Classical.forall_intro e;
        assert (fa sr 128 (coref r h j));
        ap_f2b cv (core_r r h j) in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux);
    reveal_opaque (`%core_ok) (core_ok ii h j (f2b cv))

(* kernel-level progress: heads < vh done, and columns < vj of head vh *)
noextract
let sdone (h j vh vj : nat) : bool = h < vh || (h = vh && j < vj)

noextract
let fat (s : S.seq float) (k : nat) : float = if k < S.length s then S.index s k else zero #float
noextract
let bat (s : S.seq bf16) (k : nat) : bf16 = if k < S.length s then S.index s k else f2b (zero #float)

[@@"opaque_to_smt"]
noextract
let rec_done (ii : rins) (so : S.seq float) (sc : S.seq bf16) (vh vj : nat) : prop =
  (forall (h i j : nat). h < 16 /\ i < 128 /\ j < 128 /\ sdone h j vh vj ==> st_ok ii h i j (fat so (sidx h i j))) /\
  (forall (h j : nat). h < 16 /\ j < 128 /\ sdone h j vh vj ==> core_ok ii h j (bat sc (h * 128 + j)))

[@@"opaque_to_smt"]
noextract
let rec_mid (ii : rins) (so : S.seq float) (sr : S.seq float) (h j vi : nat) : prop =
  (forall (i : nat). i < vi /\ i < 128 ==> st_ok ii h i j (fat so (sidx h i j)) /\ p2_ok ii h j i (fat sr i))

noextract
let rec_init (ii : rins) (so : S.seq float) (sc : S.seq bf16) : Lemma (rec_done ii so sc 0 0)
  = reveal_opaque (`%rec_done) (rec_done ii so sc 0 0)

noextract
let mid_init (ii : rins) (so sr : S.seq float) (h j : nat) : Lemma (rec_mid ii so sr h j 0)
  = reveal_opaque (`%rec_mid) (rec_mid ii so sr h j 0)

noextract
let sidx_ne (h i j h' i' j' : nat)
  : Lemma (requires h < 16 /\ i < 128 /\ j < 128 /\ h' < 16 /\ i' < 128 /\ j' < 128 /\ ~(h == h' /\ i == i' /\ j == j'))
          (ensures sidx h i j <> sidx h' i' j')
  = sidx_dec h i j; sidx_dec h' i' j'

noextract
let mid_step (ii : rins) (so sr : S.seq float) (sc : S.seq bf16) (h j vi : nat) (nx pv : float)
  : Lemma (requires S.length so == 262144 /\ S.length sr == 128 /\ h < 16 /\ j < 128 /\
                    rec_done ii so sc h j /\ rec_mid ii so sr h j vi /\ vi < 128 /\
                    st_ok ii h vi j nx /\ p2_ok ii h j vi pv)
          (ensures sidx h vi j < 262144 /\
                   rec_done ii (S.upd so (sidx h vi j) nx) sc h j /\
                   rec_mid ii (S.upd so (sidx h vi j) nx) (S.upd sr vi pv) h j (vi + 1))
  = sidx_dec h vi j;
    let so' = S.upd so (sidx h vi j) nx in
    let sr' = S.upd sr vi pv in
    reveal_opaque (`%rec_done) (rec_done ii so sc h j);
    reveal_opaque (`%rec_done) (rec_done ii so' sc h j);
    reveal_opaque (`%rec_mid) (rec_mid ii so sr h j vi);
    reveal_opaque (`%rec_mid) (rec_mid ii so' sr' h j (vi + 1));
    let a1 (h' i' j' : nat) : Lemma (h' < 16 /\ i' < 128 /\ j' < 128 /\ sdone h' j' h j ==>
                                      st_ok ii h' i' j' (fat so' (sidx h' i' j')))
      = if h' < 16 && i' < 128 && j' < 128 && sdone h' j' h j then begin
          sidx_ne h' i' j' h vi j; sidx_dec h' i' j'
        end in
    FStar.Classical.forall_intro_3 a1;
    let a2 (i' : nat) : Lemma (i' < vi + 1 /\ i' < 128 ==>
                               st_ok ii h i' j (fat so' (sidx h i' j)) /\ p2_ok ii h j i' (fat sr' i'))
      = if i' < vi + 1 && i' < 128 then begin
          sidx_dec h i' j;
          if i' < vi then sidx_ne h i' j h vi j
        end in
    FStar.Classical.forall_intro a2

noextract
let col_fin (ii : rins) (so sr : S.seq float) (sc : S.seq bf16) (h j : nat) (cv : float)
  : Lemma (requires S.length sr == 128 /\ S.length sc == 2048 /\ h < 16 /\ j < 128 /\
                    rec_done ii so sc h j /\ rec_mid ii so sr h j 128 /\ csum_ok sr 128 cv)
          (ensures h * 128 + j < 2048 /\ rec_done ii so (S.upd sc (h * 128 + j) (f2b cv)) h (j + 1))
  = hdec h j;
    reveal_opaque (`%rec_done) (rec_done ii so sc h j);
    reveal_opaque (`%rec_mid) (rec_mid ii so sr h j 128);
    let sc' = S.upd sc (h * 128 + j) (f2b cv) in
    reveal_opaque (`%rec_done) (rec_done ii so sc' h (j + 1));
    let e (i : nat) : Lemma (i < 128 ==> p2_ok ii h j i (S.index sr i)) = () in
    FStar.Classical.forall_intro e;
    core_lemma ii h j sr cv;
    let a1 (h' i' j' : nat) : Lemma (h' < 16 /\ i' < 128 /\ j' < 128 /\ sdone h' j' h (j + 1) ==>
                                      st_ok ii h' i' j' (fat so (sidx h' i' j')))
      = if h' < 16 && i' < 128 && j' < 128 && sdone h' j' h (j + 1) then begin
          if h' = h && j' = j then () else assert (sdone h' j' h j)
        end in
    FStar.Classical.forall_intro_3 a1;
    let a2 (h' j' : nat) : Lemma (h' < 16 /\ j' < 128 /\ sdone h' j' h (j + 1) ==>
                                   core_ok ii h' j' (bat sc' (h' * 128 + j')))
      = if h' < 16 && j' < 128 && sdone h' j' h (j + 1) then begin
          hdec h' j';
          if h' = h && j' = j then () else assert (h' * 128 + j' <> h * 128 + j)
        end in
    FStar.Classical.forall_intro_2 a2

noextract
let head_fin (ii : rins) (so : S.seq float) (sc : S.seq bf16) (h : nat)
  : Lemma (requires rec_done ii so sc h 128) (ensures rec_done ii so sc (h + 1) 0)
  = reveal_opaque (`%rec_done) (rec_done ii so sc h 128);
    reveal_opaque (`%rec_done) (rec_done ii so sc (h + 1) 0)

noextract
let rec_ok (ii : rins) (so : S.seq float) (sc : S.seq bf16) : prop = rec_done ii so sc 16 0

#pop-options
(* ===== Proj ===== *)
open FStar.Real
open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.GEMM
open KPlay.SharedSpec


#push-options "--z3rlimit 40 --fuel 0 --ifuel 1"

(*** matmul_single over reals is a vsum *)

noextract
let mmt (m1 : chest2 real 16 16) (m2 : chest2 real 16 8) (a : natlt 16) (n : natlt 8) (k : nat) : real =
  if k < 16 then acc2 m1 a k *. acc2 m2 k n else 0.0R

noextract
let rec mm_vsum (m1 : chest2 real 16 16) (m2 : chest2 real 16 8) (a : natlt 16) (n : natlt 8) (t : nat{t <= 16})
  : Lemma (ensures __matmul_single m1 m2 a n t == vsum t (mmt m1 m2 a n)) (decreases t)
  = if t = 0 then vsum_zero (mmt m1 m2 a n)
    else begin
      mm_vsum m1 m2 a n (t - 1);
      vsum_succ2 (t - 1) (mmt m1 m2 a n)
    end

(*** The real model of a WMMA accumulator chain *)

noextract
let bfilled (sb si : S.seq bf16) (kk : nat) : prop =
  S.length sb == nB /\ 0 < kk /\ 16 * kk <= nB /\ kk <= S.length si /\
  (forall (idx : nat). idx < 16 * kk ==> S.index sb idx == S.index si (idx % kk))

noextract
let pterm (rw rh : S.seq real) (aoff boff kk : nat) (a : nat) (k : nat) : real =
  at rw (aoff + kk * a + k) *. at rh (boff + k)

noextract
let rchain (rw rh : S.seq real) (aoff boff kk : nat) (j : nat) : chest2 real 16 8 =
  mk2 (fun a n -> vsum (16 * j) (pterm rw rh aoff boff kk a))

noextract
let rA (rw : S.seq real) (off kk : nat) : chest2 real 16 16 = mk2 (fun a k -> at rw (off + kk * a + k))

noextract
let rB (rh : S.seq real) (off : nat) : chest2 real 16 8 = mk2 (fun k n -> at rh (off + k))

noextract
let cprem (sw sb si : S.seq bf16) (rw rh : S.seq real) (kk : nat) : prop =
  S.length sw == nW /\ bfilled sb si kk /\ pb sw (S.length sw) rw /\ pb si kk rh

noextract
let a_ap (sw : S.seq bf16{S.length sw == nW}) (rw : S.seq real) (off kk : nat)
  : Lemma (requires pb sw (S.length sw) rw /\ 16 <= kk /\ off + kk * 15 + 16 <= nW)
          (ensures tile_val false 16 off kk sw () %~ rA rw off kk)
  = let aux (y : abs (16 @| 16 @| INil))
      : Lemma (acc (tile_val false 16 off kk sw ()) y %~ acc (rA rw off kk) y)
      = let (a, (k, ())) = y in
        assert (off + kk * a + k < nW) in
    FStar.Classical.forall_intro aux

noextract
let mod_small (boff j k kk n : nat)
  : Lemma (requires boff + 16 * j + 16 <= kk /\ k < 16 /\ 0 < kk)
          (ensures (boff + 16 * j + kk * n + k) % kk == boff + 16 * j + k)
  = L.lemma_mod_plus (boff + 16 * j + k) n kk; L.small_mod (boff + 16 * j + k) kk

noextract
let b_ap (sb si : S.seq bf16) (rh : S.seq real) (boff kk j : nat)
  : Lemma (requires bfilled sb si kk /\ pb si kk rh /\ 16 <= kk /\ boff + 16 * j + 16 <= kk)
          (ensures tile_val true 8 (boff + 16 * j) kk sb () %~ rB rh (boff + 16 * j))
  = let aux (y : abs (16 @| 8 @| INil))
      : Lemma (acc (tile_val true 8 (boff + 16 * j) kk sb ()) y %~ acc (rB rh (boff + 16 * j)) y)
      = let (k, (n, ())) = y in
        L.lemma_mult_le_left kk n 7;
        assert (boff + 16 * j + kk * n + k < 16 * kk);
        mod_small boff j k kk n in
    FStar.Classical.forall_intro aux

noextract
let comb_acc (f : real -> real -> real) (c1 c2 : chest2 real 16 8) (i : abs (16 @| 8 @| INil))
  : Lemma (acc (Kuiper.Chest.chest_comb f c1 c2) i == f (acc c1 i) (acc c2 i))
  = ()

noextract
let rstep (rw rh : S.seq real) (aoff boff kk j : nat) (a : natlt 16) (n : natlt 8)
  : Lemma (acc2 (matplus (rchain rw rh aoff boff kk j) (matmul (rA rw (aoff + 16 * j) kk) (rB rh (boff + 16 * j)))) a n
           == acc2 (rchain rw rh aoff boff kk (j + 1)) a n)
  = let ra = rA rw (aoff + 16 * j) kk in
    let rb = rB rh (boff + 16 * j) in
    let f = pterm rw rh aoff boff kk a in
    lemma_matmul_index ra rb a n;
    mm_vsum ra rb a n 16;
    let e (k : nat) : Lemma (k < 16 ==> mmt ra rb a n k == shf f (16 * j) k)
      = if k < 16 then assert (aoff + 16 * j + kk * a + k == aoff + kk * a + (16 * j + k)) in
    FStar.Classical.forall_intro e;
    vsum_ext 16 (mmt ra rb a n) (shf f (16 * j));
    vsum_split2 (16 * j) 16 f;
    assert (16 * (j + 1) == 16 * j + 16);
    let rc = rchain rw rh aoff boff kk j in
    let mm = matmul ra rb in
    assert (acc2 rc a n == vsum (16 * j) f);
    assert (acc2 mm a n == vsum 16 (shf f (16 * j)));
    comb_acc (add #real) rc mm (a, (n, ()));
    assert (acc2 (matplus rc mm) a n == add #real (acc2 rc a n) (acc2 mm a n));
    assert (add #real (acc2 rc a n) (acc2 mm a n) == acc2 rc a n +. acc2 mm a n);
    assert (acc2 (rchain rw rh aoff boff kk (j + 1)) a n == vsum (16 * (j + 1)) f)

noextract
let bt_ok (boff j kk : nat)
  : Lemma (requires boff + 16 * j + 16 <= kk /\ 16 * kk <= nB /\ 16 <= kk)
          (ensures tile_ok true 8 (boff + 16 * j) kk nB)
  = L.lemma_mult_le_left kk 7 16

noextract
let chain_step_ap (sw sb si : S.seq bf16) (rw rh : S.seq real) (aoff boff kk : nat) (j : nat)
  : Lemma (requires cprem sw sb si rw rh kk /\ 16 <= kk /\ aoff + 16 * j + kk * 15 + 16 <= nW /\
                    boff + 16 * j + 16 <= kk /\
                    chain sw sb aoff boff kk j %~ rchain rw rh aoff boff kk j)
          (ensures chain sw sb aoff boff kk (j + 1) %~ rchain rw rh aoff boff kk (j + 1))
  = bt_ok boff j kk;
    chain_step sw sb aoff boff kk j;
    a_ap sw rw (aoff + 16 * j) kk;
    b_ap sb si rh boff kk j;
    M.emma_approx_lemma (chain sw sb aoff boff kk j)
      (tile_val false 16 (aoff + 16 * j) kk sw ()) (tile_val true 8 (boff + 16 * j) kk sb ())
      (rchain rw rh aoff boff kk j) (rA rw (aoff + 16 * j) kk) (rB rh (boff + 16 * j));
    let aux (y : abs (16 @| 8 @| INil))
      : Lemma (acc (chain sw sb aoff boff kk (j + 1)) y %~ acc (rchain rw rh aoff boff kk (j + 1)) y)
      = let (a, (n, ())) = y in rstep rw rh aoff boff kk j a n in
    FStar.Classical.forall_intro aux

noextract
let chain_zero_ap (sw sb : S.seq bf16) (rw rh : S.seq real) (aoff boff kk : nat)
  : Lemma (requires S.length sw == nW /\ S.length sb == nB /\ 0 < kk)
          (ensures chain sw sb aoff boff kk 0 %~ rchain rw rh aoff boff kk 0)
  = chain_zero sw sb aoff boff kk;
    let aux (y : abs (16 @| 8 @| INil))
      : Lemma (acc (chain sw sb aoff boff kk 0) y %~ acc (rchain rw rh aoff boff kk 0) y)
      = let (a, (n, ())) = y in vsum_zero (pterm rw rh aoff boff kk a); ap_zero () in
    FStar.Classical.forall_intro aux

noextract
let rec chain_ap (sw sb si : S.seq bf16) (rw rh : S.seq real) (aoff boff kk : nat) (j : nat)
  : Lemma (requires cprem sw sb si rw rh kk /\ 16 <= kk /\ aoff + 16 * j + kk * 15 <= nW /\ boff + 16 * j <= kk)
          (ensures chain sw sb aoff boff kk j %~ rchain rw rh aoff boff kk j)
          (decreases j)
  = if j = 0 then chain_zero_ap sw sb rw rh aoff boff kk
    else begin
      chain_ap sw sb si rw rh aoff boff kk (j - 1);
      chain_step_ap sw sb si rw rh aoff boff kk (j - 1)
    end

(*** Combining the stored tiles: projected[r] approximates the dot product of row r *)

noextract
let combv (sc : S.seq float) (nch : nat) (r : nat) : float =
  let g = (r / 16) * nch in
  let i = r % 16 in
  if nch = 1 then fat sc (g * 128 + i * 8) else add (fat sc (g * 128 + i * 8)) (fat sc ((g + 1) * 128 + i * 8))

noextract
let comb_ok (sc : S.seq float) (nch nt : nat) (sd : S.seq bf16) : prop =
  nt * 16 <= S.length sd /\ (forall (r : nat). r < nt * 16 ==> S.index sd r == f2b (combv sc nch r))

noextract
let rowf (rw rh : S.seq real) (wo : nat) (k : nat) : real = at rw (wo + k) *. at rh k

noextract
let proj_g1 (sw si : S.seq bf16) (woff kk nt : nat) (sd : S.seq bf16) : prop =
  nt * 16 <= S.length sd /\
  (forall (rw rh : S.seq real). pb sw (S.length sw) rw /\ pb si kk rh ==>
     (forall (r : nat). r < nt * 16 ==> S.index sd r %~ dot rw (woff + r * kk) rh kk))

noextract
let cm_idx (g i : nat)
  : Lemma (requires i < 16) (ensures (g * 128 + i * 8) / 128 == g /\ ((g * 128 + i * 8) % 128) / 8 == i /\
                                     (g * 128 + i * 8) % 8 == 0)
  = L.lemma_div_plus (i * 8) g 128; L.small_div (i * 8) 128;
    L.lemma_mod_plus (i * 8) g 128; L.small_mod (i * 8) 128;
    L.cancel_mul_div i 8; L.lemma_mod_plus 0 (g * 16 + i) 8

#push-options "--fuel 2 --ifuel 2"
noextract
let chest_el (c1 : chest2 float 16 8) (c2 : chest2 real 16 8) (a : natlt 16) (n : natlt 8)
  : Lemma (requires c1 %~ c2) (ensures acc2 c1 a n %~ acc2 c2 a n)
  = let y : abs (16 @| 8 @| INil) = (a, (n, ())) in
    assert (acc c1 y %~ acc c2 y)
#pop-options

noextract
let cell_arith (woff kk nt nch chunk g : nat)
  : Lemma (requires pok woff kk nt nch chunk /\ g < nt * nch)
          (ensures gboff nch chunk g + 16 * (chunk / 16) <= kk /\ 16 * (chunk / 16) == chunk /\
                   gaoff woff kk nch chunk g + 16 * (chunk / 16) + kk * 15 <= nW /\ g * 128 + 128 <= nC)
  = tile_bounds woff kk nt nch chunk g 0;
    L.euclidean_division_definition g nch;
    L.lemma_mult_le_right chunk (g % nch + 1) nch

(* the stored cell of tile g, row i approximates the chunk sum *)
noextract
let cell_ap (sw sb si : S.seq bf16) (rw rh : S.seq real) (sc : S.seq float)
  (woff kk nt nch chunk : nat) (g i : nat)
  : Lemma (requires pok woff kk nt nch chunk /\ S.length sb == nB /\ S.length sc == nC /\
                    cprem sw sb si rw rh kk /\ g < nt * nch /\ i < 16)
          (ensures g * 128 + i * 8 < nC /\
                   S.index (cfin sw sb woff kk nt nch chunk sc) (g * 128 + i * 8) %~
                   vsum chunk (pterm rw rh (gaoff woff kk nch chunk g) (gboff nch chunk g) kk i))
  = let c = g * 128 + i * 8 in
    cell_arith woff kk nt nch chunk g;
    cm_idx g i;
    L.lemma_mult_le_right 128 (g + 1) (nt * nch);
    let aoff = gaoff woff kk nch chunk g in
    let boff = gboff nch chunk g in
    let nj = chunk / 16 in
    chain_ap sw sb si rw rh aoff boff kk nj;
    let ch = chain sw sb aoff boff kk nj in
    let rc = rchain rw rh aoff boff kk nj in
    chest_el ch rc i 0;
    assert (acc2 rc i 0 == vsum chunk (pterm rw rh aoff boff kk i));
    assert (cmodel_at sw sb woff kk nt nch chunk c == acc2 ch i 0);
    assert (S.index (cfin sw sb woff kk nt nch chunk sc) c == cmodel_at sw sb woff kk nt nch chunk c)

noextract
let dot_split (rw rh : S.seq real) (wo kk chunk : nat)
  : Lemma (requires kk == 2 * chunk)
          (ensures dot rw wo rh kk == vsum chunk (rowf rw rh wo) +. vsum chunk (shf (rowf rw rh wo) chunk))
  = dot_eq rw wo rh kk;
    vsum_ext kk (dotf rw wo rh) (rowf rw rh wo);
    vsum_split2 chunk chunk (rowf rw rh wo)

noextract
let row_offs (woff kk nch chunk t i r : nat)
  : Lemma (requires (nch == 1 \/ nch == 2) /\ r == t * 16 + i /\ kk == nch * chunk)
          (ensures (nch == 1 ==> chunk == kk) /\ (nch == 2 ==> kk == 2 * chunk) /\ gaoff woff kk nch chunk (t * nch) == woff + t * 16 * kk /\ gboff nch chunk (t * nch) == 0 /\
                   r * kk == t * 16 * kk + kk * i /\
                   (nch == 2 ==> (gaoff woff kk nch chunk (t * nch + 1) == woff + t * 16 * kk + chunk /\
                                  gboff nch chunk (t * nch + 1) == chunk)))
  = L.lemma_mod_plus 0 t nch; L.lemma_div_plus 0 t nch;
    L.distributivity_add_left (t * 16) i kk;
    if nch = 2 then begin L.lemma_mod_plus 1 t 2; L.lemma_div_plus 1 t 2 end

noextract
let ext0 (rw rh : S.seq real) (ao kk i wo : nat)
  : Lemma (requires wo == ao + kk * i) (ensures vsum kk (pterm rw rh ao 0 kk i) == vsum kk (rowf rw rh wo) /\
             (forall (c : nat). vsum c (pterm rw rh ao 0 kk i) == vsum c (rowf rw rh wo)))
  = let e0 (k : nat) : Lemma (pterm rw rh ao 0 kk i k == rowf rw rh wo k) = () in
    FStar.Classical.forall_intro e0;
    let e (c : nat) : Lemma (vsum c (pterm rw rh ao 0 kk i) == vsum c (rowf rw rh wo)) =
      vsum_ext c (pterm rw rh ao 0 kk i) (rowf rw rh wo) in
    FStar.Classical.forall_intro e

noextract
let ext1 (rw rh : S.seq real) (ao kk i wo chunk : nat)
  : Lemma (requires wo + chunk == ao + kk * i)
          (ensures vsum chunk (pterm rw rh ao chunk kk i) == vsum chunk (shf (rowf rw rh wo) chunk))
  = let e1 (k : nat) : Lemma (pterm rw rh ao chunk kk i k == shf (rowf rw rh wo) chunk k) = () in
    FStar.Classical.forall_intro e1;
    vsum_ext chunk (pterm rw rh ao chunk kk i) (shf (rowf rw rh wo) chunk)

noextract
let dot_one (rw rh : S.seq real) (wo kk : nat)
  : Lemma (dot rw wo rh kk == vsum kk (rowf rw rh wo))
  = dot_eq rw wo rh kk; vsum_ext kk (dotf rw wo rh) (rowf rw rh wo)

noextract
let fin_1 (rw rh : S.seq real) (x : float) (ao wo kk i : nat)
  : Lemma (requires x %~ vsum kk (pterm rw rh ao 0 kk i) /\ wo == ao + kk * i)
          (ensures x %~ dot rw wo rh kk)
  = ext0 rw rh ao kk i wo; dot_one rw rh wo kk

noextract
let fin_2 (rw rh : S.seq real) (x y : float) (ao wo kk i chunk : nat)
  : Lemma (requires x %~ vsum chunk (pterm rw rh ao 0 kk i) /\
                    y %~ vsum chunk (pterm rw rh (ao + chunk) chunk kk i) /\
                    wo == ao + kk * i /\ kk == 2 * chunk)
          (ensures add x y %~ dot rw wo rh kk)
  = ext0 rw rh ao kk i wo;
    ext1 rw rh (ao + chunk) kk i wo chunk;
    dot_split rw rh wo kk chunk;
    ap_add x y (vsum chunk (rowf rw rh wo)) (vsum chunk (shf (rowf rw rh wo) chunk))

noextract
let row_ap (sw sb si : S.seq bf16) (rw rh : S.seq real) (sc : S.seq float)
  (woff kk nt nch chunk : nat) (r : nat)
  : Lemma (requires pok woff kk nt nch chunk /\ (nch == 1 \/ nch == 2) /\ S.length sb == nB /\
                    S.length sc == nC /\ cprem sw sb si rw rh kk /\ r < nt * 16)
          (ensures combv (cfin sw sb woff kk nt nch chunk sc) nch r %~ dot rw (woff + r * kk) rh kk)
  = let t = r / 16 in let i = r % 16 in
    L.euclidean_division_definition r 16;
    div_lt r nt 16;
    let g = t * nch in
    L.lemma_mult_le_right nch (t + 1) nt;
    assert (g + nch <= nt * nch);
    let wo = woff + r * kk in
    let ao = woff + t * 16 * kk in
    let cf = cfin sw sb woff kk nt nch chunk sc in
    row_offs woff kk nch chunk t i r;
    assert (wo == ao + kk * i);
    cell_ap sw sb si rw rh sc woff kk nt nch chunk g i;
    let x = S.index cf (g * 128 + i * 8) in
    assert (fat cf (g * 128 + i * 8) == x);
    assert (x %~ vsum chunk (pterm rw rh ao 0 kk i));
    if nch = 1 then begin fin_1 rw rh x ao wo kk i end
    else begin
      cell_ap sw sb si rw rh sc woff kk nt nch chunk (g + 1) i;
      let y = S.index cf ((g + 1) * 128 + i * 8) in
      assert (fat cf ((g + 1) * 128 + i * 8) == y);
      fin_2 rw rh x y ao wo kk i chunk
    end

noextract
let proj_lemma (sw sb si : S.seq bf16) (sc : S.seq float) (woff kk nt nch chunk : nat) (sd : S.seq bf16)
  : Lemma (requires pok woff kk nt nch chunk /\ (nch == 1 \/ nch == 2) /\ S.length sw == nW /\
                    S.length sc == nC /\ bfilled sb si kk /\
                    comb_ok (cfin sw sb woff kk nt nch chunk sc) nch nt sd)
          (ensures proj_g1 sw si woff kk nt sd)
  = let aux (rw rh : S.seq real)
      : Lemma (requires pb sw (S.length sw) rw /\ pb si kk rh)
              (ensures forall (r : nat). r < nt * 16 ==> S.index sd r %~ dot rw (woff + r * kk) rh kk)
      = let e (r : nat) : Lemma (r < nt * 16 ==> S.index sd r %~ dot rw (woff + r * kk) rh kk)
          = if r < nt * 16 then begin
              row_ap sw sb si rw rh sc woff kk nt nch chunk r;
              ap_f2b (combv (cfin sw sb woff kk nt nch chunk sc) nch r) (dot rw (woff + r * kk) rh kk)
            end in
        FStar.Classical.forall_intro e in
    FStar.Classical.forall_intro_2 (fun rw rh -> FStar.Classical.move_requires (aux rw) rh)

#pop-options
(* ===== Post ===== *)
open FStar.Real
open Kuiper
open KPlay.SharedSpec


#push-options "--z3rlimit 40 --fuel 0 --ifuel 0"

(*** Kernel post-conditions *)

noextract
let copy_post (sx s' : S.seq bf16) : prop =
  S.length s' == 1024 /\ 1024 <= S.length sx /\ (forall (i : nat). i < 1024 ==> S.index s' i == S.index sx i)

noextract
let norm_post (sr sw : S.seq bf16) (woff : nat) (si : S.seq bf16) : prop =
  S.length si == 3584 /\ (forall (i : nat). i < 1024 ==> norm_el_ok sr sw woff i (S.index si i))

noextract
let beta_post (sp sb : S.seq bf16) : prop =
  S.length sb == 16 /\ 16 <= S.length sp /\
  (forall (i : nat). i < 16 ==> S.index sb i == f2b (sigm_f (b2f (S.index sp i))))

noextract
let conv_post (sw sc sq so sq' : S.seq bf16) : prop =
  S.length so == 24576 /\ S.length sq' == 6144 /\ S.length sc == 24576 /\ S.length sq == 6144 /\
  (forall (c : nat). c < 6144 ==>
     S.index so (4 * c) == S.index sc (4 * c + 1) /\ S.index so (4 * c + 1) == S.index sc (4 * c + 2) /\
     S.index so (4 * c + 2) == S.index sc (4 * c + 3) /\ S.index so (4 * c + 3) == S.index sq c /\
     conv_ok sw sc sq c (S.index sq' c))

noextract
let qk_post (sq sq' : S.seq bf16) : prop =
  S.length sq == 6144 /\ S.length sq' == 6144 /\
  (forall (i : nat). i < 4096 ==> qk_el_ok sq i (S.index sq' i)) /\
  (forall (i : nat). 4096 <= i /\ i < 6144 ==> S.index sq' i == S.index sq i)

noextract
let gn_post (sw sc sz si : S.seq bf16) : prop =
  S.length si == 3584 /\ (forall (c : nat). c < 2048 ==> gn_el_ok sw sc sz c (S.index si c))

noextract
let resadd_post (sr sp s' : S.seq bf16) : prop =
  S.length s' == 1024 /\ 1024 <= S.length sr /\ 1024 <= S.length sp /\
  (forall (i : nat). i < 1024 ==> S.index s' i == f2b (add (b2f (S.index sr i)) (b2f (S.index sp i))))

noextract
let silumul_post (sg sp s' : S.seq bf16) : prop =
  S.length s' == 3584 /\ 3584 <= S.length sg /\ 3584 <= S.length sp /\
  (forall (i : nat). i < 3584 ==>
     S.index s' i == f2b (mul (b2f (f2b (silu_f (b2f (S.index sg i))))) (b2f (S.index sp i))))

(*** contiguous_sum tree halving *)

noextract
let pw_half (v : nat) : Lemma (requires pw v) (ensures v = 1 \/ 2 * (v / 2) == v) = ()

(*** qk-norm head order: hp -> start = (hp % 2) * 2048 + (hp / 2) * 128 *)

noextract
let qhd (vh : nat) : nat = (vh % 2) * 16 + vh / 2

noextract
let qst (vh : nat) : nat = qhd vh * 128

noextract
let qpos (hh : nat) : nat = (hh % 16) * 2 + hh / 16

noextract
let qdone (i vh : nat) : bool = qpos (i / 128) < vh

[@@"opaque_to_smt"]
noextract
let qk_mid (sq sq' : S.seq bf16) (vh : nat) : prop =
  S.length sq == 6144 /\ S.length sq' == 6144 /\
  (forall (i : nat). i < 6144 ==>
     (if i < 4096 && qdone i vh then qk_el_ok sq i (S.index sq' i) else S.index sq' i == S.index sq i))

noextract
let qp_inj (hh vh : nat)
  : Lemma (requires hh < 32 /\ vh < 32) (ensures (qpos hh == vh) <==> (hh == qhd vh))
  = ()

noextract
let qst_facts (vh : nat)
  : Lemma (requires vh < 32)
          (ensures qhd vh < 32 /\ qst vh + 128 <= 4096 /\ qpos (qhd vh) == vh /\ qst vh == (vh % 2) * 2048 + (vh / 2) * 128 /\
                   qst vh % 128 == 0)
  = L.cancel_mul_mod (qhd vh) 128

noextract
let qk_init (sq : S.seq bf16) : Lemma (requires S.length sq == 6144) (ensures qk_mid sq sq 0)
  = reveal_opaque (`%qk_mid) (qk_mid sq sq 0)

noextract
let qk_fresh (sq sq1 : S.seq bf16) (vh : nat)
  : Lemma (requires qk_mid sq sq1 vh /\ vh < 32 /\ S.length sq == 6144 /\ S.length sq1 == 6144)
          (ensures qst vh + 128 <= 4096 /\
                   (forall (k : nat). k < 128 ==> S.index sq1 (qst vh + k) == S.index sq (qst vh + k)))
  = qst_facts vh;
    reveal_opaque (`%qk_mid) (qk_mid sq sq1 vh);
    let e (k : nat) : Lemma (k < 128 ==> S.index sq1 (qst vh + k) == S.index sq (qst vh + k))
      = if k < 128 then begin hdec (qhd vh) k; assert (not (qdone (qst vh + k) vh)) end in
    FStar.Classical.forall_intro e

noextract
let qd_other (i vh : nat)
  : Lemma (requires i < 4096 /\ vh < 32 /\ (i < qst vh \/ qst vh + 128 <= i))
          (ensures qdone i (vh + 1) == qdone i vh)
  = let hh : nat = i / 128 in
    L.euclidean_division_definition i 128;
    if hh >= 32 then L.lemma_mult_le_right 128 32 hh;
    qp_inj hh vh;
    if qpos hh = vh then begin
      assert (hh == qhd vh);
      assert (qst vh == hh * 128)
    end

noextract
let qk_step (sq sq1 sq2 : S.seq bf16) (vh : nat)
  : Lemma (requires qk_mid sq sq1 vh /\ vh < 32 /\ S.length sq == 6144 /\ S.length sq1 == 6144 /\ S.length sq2 == 6144 /\ qst vh + 128 <= 6144 /\
                    (forall (i : nat). i < 6144 /\ (i < qst vh \/ qst vh + 128 <= i) ==> S.index sq2 i == S.index sq1 i) /\
                    (forall (k : nat). k < 128 ==> qk_el_ok sq (qst vh + k) (S.index sq2 (qst vh + k))))
          (ensures qk_mid sq sq2 (vh + 1))
  = qst_facts vh;
    reveal_opaque (`%qk_mid) (qk_mid sq sq1 vh);
    reveal_opaque (`%qk_mid) (qk_mid sq sq2 (vh + 1));
    let e (i : nat)
      : Lemma (i < 6144 ==>
               (if i < 4096 && qdone i (vh + 1) then qk_el_ok sq i (S.index sq2 i) else S.index sq2 i == S.index sq i))
      = if i < 6144 then begin
          if qst vh <= i && i < qst vh + 128 then begin
            hdec (qhd vh) (i - qst vh);
            assert (qk_el_ok sq (qst vh + (i - qst vh)) (S.index sq2 (qst vh + (i - qst vh))))
          end else if i < 4096 then qd_other i vh
        end in
    FStar.Classical.forall_intro e

[@@"opaque_to_smt"]
noextract
let qk_row (sq sq1 sq' : S.seq bf16) (st vi : nat) : prop =
  S.length sq' == 6144 /\ S.length sq1 == 6144 /\ S.length sq == 6144 /\ st + 128 <= 4096 /\ vi <= 128 /\
  (forall (c : nat). c < 6144 /\ (c < st \/ st + 128 <= c) ==> S.index sq' c == S.index sq1 c) /\
  (forall (k : nat). k < 128 /\ vi <= k ==> S.index sq' (st + k) == S.index sq (st + k)) /\
  (forall (k : nat). k < vi ==> qk_el_ok sq (st + k) (S.index sq' (st + k)))

noextract
let qk_row_init (sq sq1 : S.seq bf16) (st : nat)
  : Lemma (requires S.length sq == 6144 /\ S.length sq1 == 6144 /\ st + 128 <= 4096 /\
                    (forall (k : nat). k < 128 ==> S.index sq1 (st + k) == S.index sq (st + k)))
          (ensures qk_row sq sq1 sq1 st 0)
  = reveal_opaque (`%qk_row) (qk_row sq sq1 sq1 st 0)

noextract
let qk_row_read (sq sq1 sq' : S.seq bf16) (st vi : nat)
  : Lemma (requires qk_row sq sq1 sq' st vi /\ vi < 128)
          (ensures S.length sq' == 6144 /\ S.length sq == 6144 /\ st + 128 <= 4096 /\
                   S.index sq' (st + vi) == S.index sq (st + vi))
  = reveal_opaque (`%qk_row) (qk_row sq sq1 sq' st vi)

noextract
let qk_row_step (sq sq1 sq' : S.seq bf16) (st vi : nat) (y : bf16)
  : Lemma (requires qk_row sq sq1 sq' st vi /\ vi < 128 /\ qk_el_ok sq (st + vi) y)
          (ensures st + vi < 6144 /\ S.length sq' == 6144 /\ qk_row sq sq1 (S.upd sq' (st + vi) y) st (vi + 1))
  = reveal_opaque (`%qk_row) (qk_row sq sq1 sq' st vi);
    let sq2 = S.upd sq' (st + vi) y in
    reveal_opaque (`%qk_row) (qk_row sq sq1 sq2 st (vi + 1));
    let a (c : nat) : Lemma (c < 6144 /\ (c < st \/ st + 128 <= c) ==> S.index sq2 c == S.index sq1 c) = () in
    let b (k : nat) : Lemma (k < 128 /\ vi + 1 <= k ==> S.index sq2 (st + k) == S.index sq (st + k)) = () in
    let d (k : nat) : Lemma (k < vi + 1 ==> qk_el_ok sq (st + k) (S.index sq2 (st + k)))
      = if k < vi then assert (S.index sq2 (st + k) == S.index sq' (st + k)) in
    FStar.Classical.forall_intro a;
    FStar.Classical.forall_intro b;
    FStar.Classical.forall_intro d

noextract
let qk_row_fin (sq sq1 sq2 : S.seq bf16) (vh st : nat)
  : Lemma (requires qk_mid sq sq1 vh /\ vh < 32 /\ st == qst vh /\ qk_row sq sq1 sq2 st 128)
          (ensures S.length sq2 == 6144 /\ qk_mid sq sq2 (vh + 1))
  = reveal_opaque (`%qk_row) (qk_row sq sq1 sq2 st 128);
    qk_step sq sq1 sq2 vh

noextract
let qk_fin (sq sq' : S.seq bf16)
  : Lemma (requires qk_mid sq sq' 32) (ensures qk_post sq sq')
  = reveal_opaque (`%qk_mid) (qk_mid sq sq' 32);
    let e (i : nat) : Lemma (i < 4096 ==> qdone i 32) = () in
    FStar.Classical.forall_intro e

noextract
let qk_el2 (sq : S.seq bf16) (st k : nat) (inv : float) (x : bf16)
  : Lemma (requires qk_inv_ok sq st inv /\ st % 128 == 0 /\ st + 128 <= 6144 /\ k < 128 /\ S.length sq == 6144 /\
                    x == S.index sq (st + k))
          (ensures qk_el_ok sq (st + k) (f2b (mul (b2f x) inv)))
  = L.euclidean_division_definition st 128;
    qk_el sq (st / 128) k inv x

#pop-options
(* ===== Stages ===== *)
open Kuiper



#push-options "--z3rlimit 40 --fuel 0 --ifuel 1"

inline_for_extraction noextract
let c32 (n : FStar.UInt32.t) : r:sz{SZ.v r == FStar.UInt32.v n} = SZ.uint32_to_sizet n
inline_for_extraction noextract
let o_conv : sz = c32 19433472ul
inline_for_extraction noextract
let o_alog : sz = c32 19458048ul
inline_for_extraction noextract
let o_dtb : sz = c32 19458064ul
inline_for_extraction noextract
let o_gn : sz = c32 19458080ul

noextract
let rem_lemma (a b : nat) (c : int)
  : Lemma (requires b > 0 /\ c == a - (a / b) * b) (ensures c == a % b /\ c < b)
  = FStar.Math.Lemmas.euclidean_division_definition a b

(* contiguous_sum(values, size, width) with partial[] in p *)
inline_for_extraction noextract
fn csum (v : larray float 1024) (p : larray float 256) (size width : sz)
  (#fv : perm) (#sv : erased (S.seq float)) (#sp : erased (S.seq float))
  requires pure (0 < SZ.v width /\ SZ.v size <= 1024 /\ cnt_ok (SZ.v size / SZ.v width) /\
                 SZ.v size == (SZ.v size / SZ.v width) * SZ.v width)
  preserves gpu ** (v |-> Frac fv sv)
  requires p |-> sp
  returns r : float
  ensures exists* sp'. (p |-> sp') ** pure (csum_ok sv (SZ.v size) r)
{
  pts_to_len v; pts_to_len p;
  let count = size /^ width;
  leaves_init sv (SZ.v size) (SZ.v width) sp;
  let mut i = 0sz;
  while (!i <^ count)
    invariant exists* vi sp'. (i |-> vi) ** (p |-> sp') **
      pure (SZ.v vi <= SZ.v count /\ S.length sp' == 256 /\ leaves sv (SZ.v size) (SZ.v width) sp' (SZ.v vi))
    decreases (SZ.v count - SZ.v !i)
  {
    let vi = !i;
    inner_init sv (SZ.v size) (SZ.v width) (SZ.v vi);
    let mut acc = zero #float #_;
    let mut j = 0sz;
    while (!j <^ width)
      invariant exists* vj va. (j |-> vj) ** (acc |-> va) **
        pure (SZ.v vj <= SZ.v width /\ inner sv (SZ.v size) (SZ.v width) (SZ.v vi) (SZ.v vj) va)
      decreases (SZ.v width - SZ.v !j)
    {
      let vj = !j;
      FStar.Math.Lemmas.lemma_mult_le_right (SZ.v width) (SZ.v vi + 1) (SZ.v count);
      let x = v.(vi *^ width +^ vj);
      let a = !acc;
      inner_step sv (SZ.v size) (SZ.v width) (SZ.v vi) (SZ.v vj) a x;
      acc := add a x;
      j := vj +^ 1sz;
    };
    let a = !acc;
    with sp1. assert (p |-> sp1);
    leaves_step sv (SZ.v size) (SZ.v width) sp1 (SZ.v vi) a;
    p.(vi) <- a;
    i := vi +^ 1sz;
  };
  with sp2. assert (p |-> sp2);
  leaves_tree sv (SZ.v size) (SZ.v width) (SZ.v count) sp2;
  pw_start (SZ.v count);
  let mut off = count /^ 2sz;
  while (!off >^ 0sz)
    invariant exists* vo sp'. (off |-> vo) ** (p |-> sp') **
      pure (SZ.v vo <= 128 /\ S.length sp' == 256 /\
            (SZ.v vo == 0 \/ (pw (SZ.v vo) /\ 2 * SZ.v vo <= SZ.v count)) /\
            tree sv (SZ.v size) (SZ.v width) (SZ.v count) sp' (if SZ.v vo = 0 then 1 else 2 * SZ.v vo))
    decreases (SZ.v !off)
  {
    let vo = !off;
    with sp0. assert (p |-> sp0);
    pw_step (SZ.v count) (SZ.v vo);
    let mut k = 0sz;
    while (!k <^ vo)
      invariant exists* vk sp'. (k |-> vk) ** (p |-> sp') **
        pure (SZ.v vk <= SZ.v vo /\ S.length sp' == 256 /\
              (forall (q : nat). q < 256 ==>
                 fat sp' q == (if q < SZ.v vk then add (fat sp0 q) (fat sp0 (q + SZ.v vo)) else fat sp0 q)))
      decreases (SZ.v vo - SZ.v !k)
    {
      let vk = !k;
      let a = p.(vk);
      let b = p.(vk +^ vo);
      p.(vk) <- add a b;
      k := vk +^ 1sz;
    };
    with sp3. assert (p |-> sp3);
    round sv (SZ.v size) (SZ.v width) (SZ.v count) sp0 sp3 (SZ.v vo);
    pw_half (SZ.v vo);
    off := vo /^ 2sz;
  };
  with sp4. assert (p |-> sp4);
  tree_fin sv (SZ.v size) (SZ.v width) (SZ.v count) sp4;
  p.(0sz)
}

noextract
let rinv (sv : S.seq float) (vy : nat) (a b c d : float) : prop =
  (vy > 0 ==> rpart sv 0 a) /\ (vy > 1 ==> rpart sv 1 b) /\ (vy > 2 ==> rpart sv 2 c) /\ (vy > 3 ==> rpart sv 3 d)

(* recurrent_sum(values[128]) *)
inline_for_extraction noextract
fn rsum128 (v : larray float 128) (#fv : perm) (#sv : erased (S.seq float))
  preserves gpu ** (v |-> Frac fv sv)
  returns r : float
  ensures pure (csum_ok sv 128 r)
{
  pts_to_len v;
  let mut pa = zero #float #_;
  let mut pb = zero #float #_;
  let mut pc = zero #float #_;
  let mut pd = zero #float #_;
  let mut y = 0sz;
  while (!y <^ 4sz)
    invariant exists* vy a b c d. (y |-> vy) ** (pa |-> a) ** (pb |-> b) ** (pc |-> c) ** (pd |-> d) **
      pure (SZ.v vy <= 4 /\ rinv sv (SZ.v vy) a b c d)
    decreases (4 - SZ.v !y)
  {
    let vy = !y;
    rin_init sv (SZ.v vy);
    let mut a0 = zero #float #_;
    let mut a1 = zero #float #_;
    let mut a2 = zero #float #_;
    let mut a3 = zero #float #_;
    let mut t = 0sz;
    while (!t <^ 8sz)
      invariant exists* vt x0 x1 x2 x3. (t |-> vt) ** (a0 |-> x0) ** (a1 |-> x1) ** (a2 |-> x2) ** (a3 |-> x3) **
        pure (SZ.v vt <= 8 /\ rin sv (SZ.v vy) (SZ.v vt) x0 x1 x2 x3)
      decreases (8 - SZ.v !t)
    {
      let vt = !t;
      let i = vy +^ 16sz *^ vt;
      let e0 = v.(i);
      let e1 = v.(i +^ 4sz);
      let e2 = v.(i +^ 8sz);
      let e3 = v.(i +^ 12sz);
      let x0 = !a0; a0 := add x0 e0;
      let x1 = !a1; a1 := add x1 e1;
      let x2 = !a2; a2 := add x2 e2;
      let x3 = !a3; a3 := add x3 e3;
      rin_step sv (SZ.v vy) (SZ.v vt) x0 x1 x2 x3 e0 e1 e2 e3;
      t := vt +^ 1sz;
    };
    let x0 = !a0; let x1 = !a1; let x2 = !a2; let x3 = !a3;
    rin_fin sv (SZ.v vy) x0 x1 x2 x3;
    let s = add (add (add x0 x1) x2) x3;
    with a b c d. assert ((pa |-> a) ** (pb |-> b) ** (pc |-> c) ** (pd |-> d));
    if (vy = 0sz)
      requires (pa |-> a) ** (pb |-> b) ** (pc |-> c) ** (pd |-> d) ** pure (rinv sv (SZ.v vy) a b c d /\ rpart sv (SZ.v vy) s)
      ensures exists* a b c d. (pa |-> a) ** (pb |-> b) ** (pc |-> c) ** (pd |-> d) ** pure (rinv sv (SZ.v vy + 1) a b c d)
    { pa := s }
    else if (vy = 1sz) { pb := s } else if (vy = 2sz) { pc := s } else { pd := s };
    y := vy +^ 1sz;
  };
  let a = !pa; let b = !pb; let c = !pc; let d = !pd;
  rsum_fin sv a b c d;
  add (add a c) (add b d)
}

(* x -> out0 copy *)
inline_for_extraction noextract
fn k_copy (x : larray bf16 1024) (o : larray bf16 1024)
  (#px : perm) (#sx : erased (S.seq bf16)) (#so : erased (S.seq bf16))
  preserves gpu ** (x |-> Frac px sx)
  requires o |-> so
  ensures exists* s'. (o |-> s') ** pure (copy_post sx s')
{
  pts_to_len x; pts_to_len o;
  let mut i = 0sz;
  while (!i <^ 1024sz)
    invariant exists* vi s'. (i |-> vi) ** (o |-> s') **
      pure (SZ.v vi <= 1024 /\ S.length s' == 1024 /\ (forall (k : nat). k < SZ.v vi ==> S.index s' k == S.index sx k))
    decreases (1024 - SZ.v !i)
  {
    let vi = !i;
    let e = x.(vi);
    o.(vi) <- e;
    i := vi +^ 1sz;
  };
}

(* RMSNorm(1+w) of res with weight at woff into inp[0..1024) *)
inline_for_extraction noextract
fn k_norm (res : larray bf16 1024) (w : larray bf16 nW) (woff : sz) (inp : larray bf16 3584)
  (vals : larray float 1024) (part : larray float 256)
  (#pr #pw : perm) (#sr #sw : erased (S.seq bf16)) (#si : erased (S.seq bf16))
  (#sv #sp : erased (S.seq float))
  requires pure (SZ.v woff + 1024 <= nW)
  preserves gpu ** (res |-> Frac pr sr) ** (w |-> Frac pw sw)
  requires (inp |-> si) ** (vals |-> sv) ** (part |-> sp)
  ensures exists* si' sv' sp'. (inp |-> si') ** (vals |-> sv') ** (part |-> sp') ** pure (norm_post sr sw (SZ.v woff) si')
{
  pts_to_len res; pts_to_len w; pts_to_len inp; pts_to_len vals;
  let mut i = 0sz;
  while (!i <^ 1024sz)
    invariant exists* vi s'. (i |-> vi) ** (vals |-> s') **
      pure (SZ.v vi <= 1024 /\ S.length s' == 1024 /\ sq_vals sr 0 s' (SZ.v vi))
    decreases (1024 - SZ.v !i)
  {
    let vi = !i;
    let r = b2f res.(vi);
    vals.(vi) <- mul r r;
    i := vi +^ 1sz;
  };
  with sv1. assert (vals |-> sv1);
  let s = csum vals part 1024sz 4sz;
  norm_inv sr sv1 s;
  let inv = rsqrt (add (mul s inv1024_v) eps_v);
  let mut i = 0sz;
  while (!i <^ 1024sz)
    invariant exists* vi s'. (i |-> vi) ** (inp |-> s') **
      pure (SZ.v vi <= 1024 /\ S.length s' == 3584 /\
            (forall (k : nat). k < SZ.v vi ==> norm_el_ok sr sw (SZ.v woff) k (S.index s' k)))
    decreases (1024 - SZ.v !i)
  {
    let vi = !i;
    let r = b2f res.(vi);
    let wv = b2f w.(woff +^ vi);
    norm_el sr sw (SZ.v woff) (SZ.v vi) inv;
    inp.(vi) <- f2b (mul (mul r inv) (add one wv));
    i := vi +^ 1sz;
  };
}

(* B operand: bb[c*k + j] = inp[j] for c < 16 *)
inline_for_extraction noextract
fn k_bfill (inp : larray bf16 3584) (bb : larray bf16 nB) (k : sz)
  (#pi : perm) (#si : erased (S.seq bf16)) (#sb : erased (S.seq bf16))
  requires pure (0 < SZ.v k /\ SZ.v k <= 3584)
  preserves gpu ** (inp |-> Frac pi si)
  requires bb |-> sb
  ensures exists* s'. (bb |-> s') ** pure (bfilled s' si (SZ.v k))
{
  pts_to_len inp; pts_to_len bb;
  let n = 16sz *^ k;
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* vi s'. (i |-> vi) ** (bb |-> s') **
      pure (SZ.v vi <= SZ.v n /\ S.length s' == nB /\
            (forall (idx : nat). idx < SZ.v vi ==> S.index s' idx == S.index si (idx % SZ.v k)))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let ix = vi %^ k;
    rem_lemma (SZ.v vi) (SZ.v k) (SZ.v ix);
    let e = inp.(ix);
    bb.(vi) <- e;
    i := vi +^ 1sz;
  };
}

noextract
let comb_idx (vr nt nch t i : nat)
  : Lemma (requires vr < nt * 16 /\ (nch == 1 \/ nch == 2) /\ nt * nch * 128 <= nC /\
                    t == vr / 16 /\ i == vr - (vr / 16) * 16)
          (ensures i == vr % 16 /\ i < 16 /\ t < nt /\ t * nch < nC /\ (t * nch) * 128 + i * 8 < nC /\
                   (nch == 2 ==> (t * nch + 1) * 128 + i * 8 < nC))
  = FStar.Math.Lemmas.euclidean_division_definition vr 16;
    if t >= nt then FStar.Math.Lemmas.lemma_mult_le_right 16 nt t;
    FStar.Math.Lemmas.lemma_mult_le_right (nch * 128) (t + 1) nt;
    FStar.Math.Lemmas.paren_mul_right nt nch 128;
    FStar.Math.Lemmas.paren_mul_right t nch 128

noextract
let comb_val (sc : S.seq float) (vr nch t i g : nat)
  : Lemma (requires (nch == 1 \/ nch == 2) /\ S.length sc == nC /\ t == vr / 16 /\ i == vr % 16 /\ g == t * nch /\
                    g * 128 + i * 8 < nC /\ (nch == 2 ==> (g + 1) * 128 + i * 8 < nC))
          (ensures combv sc nch vr == (if nch = 1 then fat sc (g * 128 + i * 8)
                                       else add (fat sc (g * 128 + i * 8)) (fat sc ((g + 1) * 128 + i * 8))))
  = ()

(* projected[r] = bf16(combined) from the WMMA tiles *)
#push-options "--z3rlimit 100"
inline_for_extraction noextract
fn k_comb (#n : erased nat) (cc : larray float nC) (dst : larray bf16 n) (nt nch : sz)
  (#pc : perm) (#sc : erased (S.seq float)) (#sd : erased (S.seq bf16))
  requires pure ((SZ.v nch == 1 \/ SZ.v nch == 2) /\ SZ.v nt * 16 <= n /\ SZ.v nt * SZ.v nch * 128 <= nC)
  preserves gpu ** (cc |-> Frac pc sc)
  requires dst |-> sd
  ensures exists* s'. (dst |-> s') ** pure (S.length s' == reveal n /\ comb_ok sc (SZ.v nch) (SZ.v nt) s')
{
  pts_to_len cc; pts_to_len dst;
  let m = nt *^ 16sz;
  let mut r = 0sz;
  while (!r <^ m)
    invariant exists* vr s'. (r |-> vr) ** (dst |-> s') **
      pure (SZ.v vr <= SZ.v m /\ S.length s' == reveal n /\
            (forall (q : nat). q < SZ.v vr ==> S.index s' q == f2b (combv sc (SZ.v nch) q)))
    decreases (SZ.v m - SZ.v !r)
  {
    let vr = !r;
    let t = vr /^ 16sz;
    let i = vr %^ 16sz;
    comb_idx (SZ.v vr) (SZ.v nt) (SZ.v nch) (SZ.v t) (SZ.v i);
    let g = t *^ nch;
    comb_val sc (SZ.v vr) (SZ.v nch) (SZ.v t) (SZ.v i) (SZ.v g);
    assert (pure (SZ.v g * 128 + SZ.v i * 8 < S.length sc));
    let ix = g *^ 128sz +^ i *^ 8sz;
    assert (pure (SZ.v ix == SZ.v g * 128 + SZ.v i * 8));
    let c0 = cc.(ix);
    let mut cv = c0;
    if (nch = 2sz)
      requires (cv |-> c0) ** pure (c0 == fat sc (SZ.v g * 128 + SZ.v i * 8) /\
        combv sc (SZ.v nch) (SZ.v vr) == (if SZ.v nch = 1 then fat sc (SZ.v g * 128 + SZ.v i * 8)
          else add (fat sc (SZ.v g * 128 + SZ.v i * 8)) (fat sc ((SZ.v g + 1) * 128 + SZ.v i * 8))) /\
        (SZ.v nch == 2 ==> (SZ.v g + 1) * 128 + SZ.v i * 8 < nC))
      ensures exists* x. (cv |-> x) ** pure (x == combv sc (SZ.v nch) (SZ.v vr))
    {
      let c1 = cc.((g +^ 1sz) *^ 128sz +^ i *^ 8sz);
      cv := add c0 c1;
    };
    let v = !cv;
    dst.(vr) <- f2b v;
    r := vr +^ 1sz;
  };
}
#pop-options

inline_for_extraction noextract
fn k_beta (prj : larray bf16 6144) (beta : larray bf16 16)
  (#pp : perm) (#sp : erased (S.seq bf16)) (#sb : erased (S.seq bf16))
  preserves gpu ** (prj |-> Frac pp sp)
  requires beta |-> sb
  ensures exists* s'. (beta |-> s') ** pure (beta_post sp s')
{
  pts_to_len prj; pts_to_len beta;
  let mut i = 0sz;
  while (!i <^ 16sz)
    invariant exists* vi s'. (i |-> vi) ** (beta |-> s') **
      pure (SZ.v vi <= 16 /\ S.length s' == 16 /\
            (forall (k : nat). k < SZ.v vi ==> S.index s' k == f2b (sigm_f (b2f (S.index sp k)))))
    decreases (16 - SZ.v !i)
  {
    let vi = !i;
    let e = b2f prj.(vi);
    beta.(vi) <- f2b (sigm_f e);
    i := vi +^ 1sz;
  };
}

noextract
let shift_done (sc sq so : S.seq bf16) (vi : nat) : prop =
  vi <= 6144 /\ S.length so == 24576 /\ S.length sc == 24576 /\ S.length sq == 6144 /\
  (forall (c : nat). c < vi ==>
     S.index so (4 * c) == S.index sc (4 * c + 1) /\ S.index so (4 * c + 1) == S.index sc (4 * c + 2) /\
     S.index so (4 * c + 2) == S.index sc (4 * c + 3) /\ S.index so (4 * c + 3) == S.index sq c)

noextract
let convs_done (sw sc sq sq' : S.seq bf16) (vi : nat) : prop =
  vi <= 6144 /\ S.length sq' == 6144 /\ S.length sq == 6144 /\
  (forall (c : nat). c < vi ==> conv_ok sw sc sq c (S.index sq' c)) /\
  (forall (c : nat). vi <= c /\ c < 6144 ==> S.index sq' c == S.index sq c)

noextract
let conv_done (sw sc sq so sq' : S.seq bf16) (vi : nat) : prop =
  shift_done sc sq so vi /\ convs_done sw sc sq sq' vi

noextract
let upd4 (so : S.seq bf16) (vi : nat) (x0 x1 x2 x3 : bf16)
  : Pure (S.seq bf16) (requires S.length so == 24576 /\ vi < 6144) (ensures fun r -> S.length r == 24576)
  = S.upd (S.upd (S.upd (S.upd so (4 * vi) x0) (4 * vi + 1) x1) (4 * vi + 2) x2) (4 * vi + 3) x3

noextract
let upd4_at (so : S.seq bf16) (vi : nat) (x0 x1 x2 x3 : bf16) (k : nat)
  : Lemma (requires S.length so == 24576 /\ vi < 6144 /\ k < 24576)
          (ensures S.index (upd4 so vi x0 x1 x2 x3) k ==
                   (if k = 4 * vi then x0 else if k = 4 * vi + 1 then x1 else if k = 4 * vi + 2 then x2
                    else if k = 4 * vi + 3 then x3 else S.index so k))
  = ()

noextract
let shift_step (sc sq so : S.seq bf16) (vi : nat) (x0 x1 x2 x3 : bf16)
  : Lemma (requires shift_done sc sq so vi /\ vi < 6144 /\
                    x0 == S.index sc (4 * vi + 1) /\ x1 == S.index sc (4 * vi + 2) /\
                    x2 == S.index sc (4 * vi + 3) /\ x3 == S.index sq vi)
          (ensures shift_done sc sq (upd4 so vi x0 x1 x2 x3) (vi + 1))
  = let so1 = upd4 so vi x0 x1 x2 x3 in
    let e (c : nat)
      : Lemma (c < vi + 1 ==>
                 S.index so1 (4 * c) == S.index sc (4 * c + 1) /\ S.index so1 (4 * c + 1) == S.index sc (4 * c + 2) /\
                 S.index so1 (4 * c + 2) == S.index sc (4 * c + 3) /\ S.index so1 (4 * c + 3) == S.index sq c)
      = if c < vi + 1 then begin
          upd4_at so vi x0 x1 x2 x3 (4 * c);
          upd4_at so vi x0 x1 x2 x3 (4 * c + 1);
          upd4_at so vi x0 x1 x2 x3 (4 * c + 2);
          upd4_at so vi x0 x1 x2 x3 (4 * c + 3)
        end in
    FStar.Classical.forall_intro e

noextract
let convs_step (sw sc sq sq' : S.seq bf16) (vi : nat) (y : bf16)
  : Lemma (requires convs_done sw sc sq sq' vi /\ vi < 6144 /\ conv_ok sw sc sq vi y)
          (ensures convs_done sw sc sq (S.upd sq' vi y) (vi + 1))
  = let sq1 = S.upd sq' vi y in
    let e (c : nat) : Lemma (c < vi + 1 ==> conv_ok sw sc sq c (S.index sq1 c))
      = if c < vi then assert (S.index sq1 c == S.index sq' c) in
    let f (c : nat) : Lemma (vi + 1 <= c /\ c < 6144 ==> S.index sq1 c == S.index sq c)
      = if vi + 1 <= c && c < 6144 then assert (S.index sq1 c == S.index sq' c) in
    FStar.Classical.forall_intro e;
    FStar.Classical.forall_intro f

noextract
let conv_fin (sw sc sq so sq' : S.seq bf16)
  : Lemma (requires conv_done sw sc sq so sq' 6144) (ensures conv_post sw sc sq so sq')
  = ()

(* causal conv1d window shift + silu *)
inline_for_extraction noextract
fn k_conv (w : larray bf16 nW) (conv : larray bf16 24576) (out1 : larray bf16 24576) (qkv : larray bf16 6144)
  (#pw #pc : perm) (#sw #sc : erased (S.seq bf16)) (#so #sq : erased (S.seq bf16))
  preserves gpu ** (w |-> Frac pw sw) ** (conv |-> Frac pc sc)
  requires (out1 |-> so) ** (qkv |-> sq)
  ensures exists* so' sq'. (out1 |-> so') ** (qkv |-> sq') ** pure (conv_post sw sc sq so' sq')
{
  pts_to_len w; pts_to_len conv; pts_to_len out1; pts_to_len qkv;
  let mut i = 0sz;
  while (!i <^ 6144sz)
    invariant exists* vi so' sq'. (i |-> vi) ** (out1 |-> so') ** (qkv |-> sq') **
      pure (SZ.v vi <= 6144 /\ conv_done sw sc sq so' sq' (SZ.v vi))
    decreases (6144 - SZ.v !i)
  {
    let vi = !i;
    with so1 sq1. assert ((out1 |-> so1) ** (qkv |-> sq1));
    let b4 = vi *^ 4sz;
    let wb = o_conv +^ b4;
    let x0 = conv.(b4 +^ 1sz);
    out1.(b4) <- x0;
    let v0 = FM.fma_rn (b2f x0) (b2f w.(wb)) (zero #float #_);
    let x1 = conv.(b4 +^ 2sz);
    out1.(b4 +^ 1sz) <- x1;
    let v1 = FM.fma_rn (b2f x1) (b2f w.(wb +^ 1sz)) v0;
    let x2 = conv.(b4 +^ 3sz);
    out1.(b4 +^ 2sz) <- x2;
    let v2 = FM.fma_rn (b2f x2) (b2f w.(wb +^ 2sz)) v1;
    let x3 = qkv.(vi);
    out1.(b4 +^ 3sz) <- x3;
    let v3 = FM.fma_rn (b2f x3) (b2f w.(wb +^ 3sz)) v2;
    conv_el sw sc sq (SZ.v vi) v0 v1 v2 v3;
    shift_step sc sq so1 (SZ.v vi) x0 x1 x2 x3;
    convs_step sw sc sq sq1 (SZ.v vi) (f2b (silu_f (b2f (f2b v3))));
    qkv.(vi) <- f2b (silu_f (b2f (f2b v3)));
    i := vi +^ 1sz;
  };
  with so2 sq2. assert ((out1 |-> so2) ** (qkv |-> sq2));
  conv_fin sw sc sq so2 sq2;
}

noextract
let qk_start (vh h2 hd : nat)
  : Lemma (requires vh < 32 /\ hd == vh / 2 /\ h2 == vh - (vh / 2) * 2)
          (ensures h2 < 2 /\ hd < 16 /\ h2 * 2048 <= 2048 /\ hd * 128 <= 1920 /\ h2 * 2048 + hd * 128 == qst vh)
  = qst_facts vh

(* per-head l2 norm of q and k *)
inline_for_extraction noextract
fn k_qknorm (qkv : larray bf16 6144) (vals : larray float 1024) (part : larray float 256)
  (#sq : erased (S.seq bf16)) (#sv #sp : erased (S.seq float))
  preserves gpu
  requires (qkv |-> sq) ** (vals |-> sv) ** (part |-> sp)
  ensures exists* sq' sv' sp'. (qkv |-> sq') ** (vals |-> sv') ** (part |-> sp') ** pure (qk_post sq sq')
{
  pts_to_len qkv; pts_to_len vals;
  qk_init sq;
  let mut hp = 0sz;
  while (!hp <^ 32sz)
    invariant exists* vh sq' sv' sp'. (hp |-> vh) ** (qkv |-> sq') ** (vals |-> sv') ** (part |-> sp') **
      pure (SZ.v vh <= 32 /\ S.length sv' == 1024 /\ S.length sq' == 6144 /\ qk_mid sq sq' (SZ.v vh))
    decreases (32 - SZ.v !hp)
  {
    let vh = !hp;
    with sq1. assert (qkv |-> sq1);
    qst_facts (SZ.v vh);
    qk_fresh sq sq1 (SZ.v vh);
    let h2 = vh %^ 2sz;
    let hd = vh /^ 2sz;
    qk_start (SZ.v vh) (SZ.v h2) (SZ.v hd);
    let start = h2 *^ 2048sz +^ hd *^ 128sz;
    let mut i = 0sz;
    while (!i <^ 128sz)
      invariant exists* vi sv'. (i |-> vi) ** (vals |-> sv') **
        pure (SZ.v vi <= 128 /\ S.length sv' == 1024 /\
              (forall (k : nat). k < SZ.v vi ==>
                 S.index sv' k == b2f (f2b (mul (b2f (S.index sq (SZ.v start + k))) (b2f (S.index sq (SZ.v start + k)))))))
      decreases (128 - SZ.v !i)
    {
      let vi = !i;
      let q = b2f qkv.(start +^ vi);
      vals.(vi) <- b2f (f2b (mul q q));
      i := vi +^ 1sz;
    };
    with sv1. assert (vals |-> sv1);
    let s0 = csum vals part 128sz 8sz;
    qk_inv sq (SZ.v start) sv1 s0;
    let s = b2f (f2b s0);
    let inv = b2f (f2b (rsqrt (b2f (f2b (add s eps_v)))));
    qk_row_init sq sq1 (SZ.v start);
    let mut i = 0sz;
    while (!i <^ 128sz)
      invariant exists* vi sq'. (i |-> vi) ** (qkv |-> sq') **
        pure (SZ.v vi <= 128 /\ qk_row sq sq1 sq' (SZ.v start) (SZ.v vi))
      decreases (128 - SZ.v !i)
    {
      let vi = !i;
      with sq'. assert (qkv |-> sq');
      qk_row_read sq sq1 sq' (SZ.v start) (SZ.v vi);
      let x = qkv.(start +^ vi);
      let q = b2f x;
      qk_el2 sq (SZ.v start) (SZ.v vi) inv x;
      qk_row_step sq sq1 sq' (SZ.v start) (SZ.v vi) (f2b (mul q inv));
      qkv.(start +^ vi) <- f2b (mul q inv);
      i := vi +^ 1sz;
    };
    with sq2. assert (qkv |-> sq2);
    qk_row_fin sq sq1 sq2 (SZ.v vh) (SZ.v start);
    hp := vh +^ 1sz;
  };
  with sq3. assert (qkv |-> sq3);
  let vh = !hp;
  assert (pure (SZ.v vh == 32));
  assert (pure (qk_mid sq sq3 32));
  qk_fin sq sq3;
  assert (pure (qk_post sq sq3));
}

(* gated delta-rule recurrence; core[h*128+j] *)
inline_for_extraction noextract
fn k_rec (w : larray bf16 nW) (qkv : larray bf16 6144) (prj : larray bf16 6144) (beta : larray bf16 16)
  (st : larray float 262144) (out2 : larray float 262144) (core : larray bf16 2048)
  (upd : larray float 128) (prod : larray float 128)
  (#pw #pq #pp #pb #ps : perm) (#sw #sq #sp #sb : erased (S.seq bf16)) (#ss : erased (S.seq float))
  (#so : erased (S.seq float)) (#sc : erased (S.seq bf16)) (#su #sr : erased (S.seq float))
  preserves gpu ** (w |-> Frac pw sw) ** (qkv |-> Frac pq sq) ** (prj |-> Frac pp sp) ** (beta |-> Frac pb sb) ** (st |-> Frac ps ss)
  requires (out2 |-> so) ** (core |-> sc) ** (upd |-> su) ** (prod |-> sr)
  ensures exists* so' sc' su' sr'. (out2 |-> so') ** (core |-> sc') ** (upd |-> su') ** (prod |-> sr') **
    pure (S.length so' == 262144 /\ S.length sc' == 2048 /\ rec_ok (mk_ins sw sq sp sb ss) so' sc')
{
  pts_to_len w; pts_to_len qkv; pts_to_len prj; pts_to_len beta; pts_to_len st;
  pts_to_len out2; pts_to_len core; pts_to_len upd; pts_to_len prod;
  rec_init (mk_ins sw sq sp sb ss) so sc;
  let mut h = 0sz;
  while (!h <^ 16sz)
    invariant exists* vh so' sc' su' sr'. (h |-> vh) ** (out2 |-> so') ** (core |-> sc') ** (upd |-> su') ** (prod |-> sr') **
      pure (SZ.v vh <= 16 /\ S.length so' == 262144 /\ S.length sc' == 2048 /\ S.length su' == 128 /\ S.length sr' == 128 /\
            rec_done (mk_ins sw sq sp sb ss) so' sc' (SZ.v vh) 0)
    decreases (16 - SZ.v !h)
  {
    let vh = !h;
    dec_lemma (mk_ins sw sq sp sb ss) (SZ.v vh);
    let a = add (b2f prj.(vh)) (b2f w.(o_dtb +^ vh));
    let spv = SP.softplus20 a;
    let decay = fexp (mul (neg (fexp (b2f w.(o_alog +^ vh)))) spv);
    assert (pure (dec_ok (mk_ins sw sq sp sb ss) (SZ.v vh) decay));
    let bh = b2f beta.(vh);
    let sbase = vh *^ 16384sz;
    let hb = vh *^ 128sz;
    let mut j = 0sz;
    while (!j <^ 128sz)
      invariant exists* vj so' sc' su' sr'. (j |-> vj) ** (out2 |-> so') ** (core |-> sc') ** (upd |-> su') ** (prod |-> sr') **
        pure (SZ.v vj <= 128 /\ S.length so' == 262144 /\ S.length sc' == 2048 /\ S.length su' == 128 /\ S.length sr' == 128 /\
              rec_done (mk_ins sw sq sp sb ss) so' sc' (SZ.v vh) (SZ.v vj))
      decreases (128 - SZ.v !j)
    {
      let vj = !j;
      with so1 sc1. assert ((out2 |-> so1) ** (core |-> sc1));
      let mut i = 0sz;
      while (!i <^ 128sz)
        invariant exists* vi su' sr'. (i |-> vi) ** (upd |-> su') ** (prod |-> sr') **
          pure (SZ.v vi <= 128 /\ col1 (mk_ins sw sq sp sb ss) (SZ.v vh) (SZ.v vj) decay su' sr' (SZ.v vi))
        decreases (128 - SZ.v !i)
      {
        let vi = !i;
        sidx_dec (SZ.v vh) (SZ.v vi) (SZ.v vj);
        let u = mul st.(sbase +^ vi *^ 128sz +^ vj) decay;
        upd.(vi) <- u;
        prod.(vi) <- mul u (b2f qkv.(2048sz +^ hb +^ vi));
        i := vi +^ 1sz;
      };
      with su1 sr1. assert ((upd |-> su1) ** (prod |-> sr1));
      let m = rsum128 prod;
      delta_lemma (mk_ins sw sq sp sb ss) (SZ.v vh) (SZ.v vj) decay su1 sr1 m;
      let delta = mul (sub (b2f qkv.(4096sz +^ hb +^ vj)) m) bh;
      mid_init (mk_ins sw sq sp sb ss) so1 sr1 (SZ.v vh) (SZ.v vj);
      let mut i = 0sz;
      while (!i <^ 128sz)
        invariant exists* vi so' sr'. (i |-> vi) ** (out2 |-> so') ** (prod |-> sr') **
          pure (SZ.v vi <= 128 /\ S.length so' == 262144 /\ S.length sr' == 128 /\
                rec_done (mk_ins sw sq sp sb ss) so' sc1 (SZ.v vh) (SZ.v vj) /\
                rec_mid (mk_ins sw sq sp sb ss) so' sr' (SZ.v vh) (SZ.v vj) (SZ.v vi))
        decreases (128 - SZ.v !i)
      {
        let vi = !i;
        with so2 sr2. assert ((out2 |-> so2) ** (prod |-> sr2));
        sidx_dec (SZ.v vh) (SZ.v vi) (SZ.v vj);
        let u = upd.(vi);
        let nx = add u (mul (b2f qkv.(2048sz +^ hb +^ vi)) delta);
        st_el (mk_ins sw sq sp sb ss) (SZ.v vh) (SZ.v vi) (SZ.v vj) decay delta u;
        let qv = mul (b2f qkv.(hb +^ vi)) qscale_v;
        mid_step (mk_ins sw sq sp sb ss) so2 sr2 sc1 (SZ.v vh) (SZ.v vj) (SZ.v vi) nx (mul nx qv);
        out2.(sbase +^ vi *^ 128sz +^ vj) <- nx;
        prod.(vi) <- mul nx qv;
        i := vi +^ 1sz;
      };
      with so3 sr3. assert ((out2 |-> so3) ** (prod |-> sr3));
      let cv = rsum128 prod;
      col_fin (mk_ins sw sq sp sb ss) so3 sr3 sc1 (SZ.v vh) (SZ.v vj) cv;
      core.(hb +^ vj) <- f2b cv;
      j := vj +^ 1sz;
    };
    with so4 sc4. assert ((out2 |-> so4) ** (core |-> sc4));
    head_fin (mk_ins sw sq sp sb ss) so4 sc4 (SZ.v vh);
    h := vh +^ 1sz;
  };
}

[@@"opaque_to_smt"] noextract
let gn_pre (sw sc sz si : S.seq bf16) (n : nat) : prop =
  S.length si == 3584 /\ n <= 2048 /\ (forall (c : nat). c < n ==> gn_el_ok sw sc sz c (S.index si c))

noextract
let gn_pre_init (sw sc sz si : S.seq bf16)
  : Lemma (requires S.length si == 3584) (ensures gn_pre sw sc sz si 0)
  = reveal_opaque (`%gn_pre) (gn_pre sw sc sz si 0)

noextract
let gn_pre_len (sw sc sz si : S.seq bf16) (n : nat)
  : Lemma (requires gn_pre sw sc sz si n) (ensures S.length si == 3584 /\ n <= 2048)
  = reveal_opaque (`%gn_pre) (gn_pre sw sc sz si n)

noextract
let gn_step (sw sc sz si : S.seq bf16) (h i c : nat) (inv : float) (x : bf16)
  : Lemma (requires gn_pre sw sc sz si c /\ c == h * 128 + i /\ gn_inv_ok sc h inv /\ h < 16 /\ i < 128 /\
                    S.length sc == 2048 /\ S.length sz == 2048 /\ o_gn_n + i < S.length sw /\
                    x == f2b (mul (b2f (f2b (mul (b2f (S.index sw (o_gn_n + i)))
                                      (b2f (f2b (mul (b2f (S.index sc c)) inv))))))
                       (silu_f (b2f (S.index sz c)))))
          (ensures S.length si == 3584 /\ c < 3584 /\ gn_pre sw sc sz (S.upd si c x) (c + 1))
  = reveal_opaque (`%gn_pre) (gn_pre sw sc sz si c);
    reveal_opaque (`%gn_pre) (gn_pre sw sc sz (S.upd si c x) (c + 1));
    gn_el sw sc sz h i inv

noextract
let gn_pre_fin (sw sc sz si : S.seq bf16)
  : Lemma (requires gn_pre sw sc sz si 2048) (ensures gn_post sw sc sz si)
  = reveal_opaque (`%gn_pre) (gn_pre sw sc sz si 2048)

(* gated RMSNorm into inp[0..2048) *)
inline_for_extraction noextract
fn k_gnorm (w : larray bf16 nW) (core : larray bf16 2048) (z : larray bf16 2048) (inp : larray bf16 3584)
  (vals : larray float 1024) (part : larray float 256)
  (#pw #pc #pz : perm) (#sw #sc #sz_ : erased (S.seq bf16)) (#si : erased (S.seq bf16)) (#sv #sp : erased (S.seq float))
  preserves gpu ** (w |-> Frac pw sw) ** (core |-> Frac pc sc) ** (z |-> Frac pz sz_)
  requires (inp |-> si) ** (vals |-> sv) ** (part |-> sp)
  ensures exists* si' sv' sp'. (inp |-> si') ** (vals |-> sv') ** (part |-> sp') ** pure (gn_post sw sc sz_ si')
{
  pts_to_len w; pts_to_len core; pts_to_len z; pts_to_len inp; pts_to_len vals;
  gn_pre_init sw sc sz_ si;
  let mut h = 0sz;
  while (!h <^ 16sz)
    invariant exists* vh si' sv' sp'. (h |-> vh) ** (inp |-> si') ** (vals |-> sv') ** (part |-> sp') **
      pure (SZ.v vh <= 16 /\ S.length sv' == 1024 /\ gn_pre sw sc sz_ si' (SZ.v vh * 128))
    decreases (16 - SZ.v !h)
  {
    let vh = !h;
    let hb = vh *^ 128sz;
    let mut i = 0sz;
    while (!i <^ 128sz)
      invariant exists* vi sv'. (i |-> vi) ** (vals |-> sv') **
        pure (SZ.v vi <= 128 /\ S.length sv' == 1024 /\ sq_vals sc (SZ.v hb) sv' (SZ.v vi))
      decreases (128 - SZ.v !i)
    {
      let vi = !i;
      let c = b2f core.(hb +^ vi);
      vals.(vi) <- mul c c;
      i := vi +^ 1sz;
    };
    with sv1. assert (vals |-> sv1);
    let s = csum vals part 128sz 4sz;
    gn_inv sc (SZ.v vh) sv1 s;
    let inv = rsqrt (add (mul s inv128_v) eps_v);
    let mut i = 0sz;
    while (!i <^ 128sz)
      invariant exists* vi si'. (i |-> vi) ** (inp |-> si') **
        pure (SZ.v vi <= 128 /\ gn_pre sw sc sz_ si' (SZ.v hb + SZ.v vi))
      decreases (128 - SZ.v !i)
    {
      let vi = !i;
      with si2. assert (inp |-> si2);
      gn_pre_len sw sc sz_ si2 (SZ.v hb + SZ.v vi);
      let unit = f2b (mul (b2f core.(hb +^ vi)) inv);
      let scaled = f2b (mul (b2f w.(o_gn +^ vi)) (b2f unit));
      let y = f2b (mul (b2f scaled) (silu_f (b2f z.(hb +^ vi))));
      gn_step sw sc sz_ si2 (SZ.v vh) (SZ.v vi) (SZ.v hb + SZ.v vi) inv y;
      inp.(hb +^ vi) <- y;
      i := vi +^ 1sz;
    };
    h := vh +^ 1sz;
  };
  with si3. assert (inp |-> si3);
  let vh = !h;
  assert (pure (SZ.v vh == 16));
  assert (pure (gn_pre sw sc sz_ si3 2048));
  gn_pre_fin sw sc sz_ si3;
  assert (pure (gn_post sw sc sz_ si3));
}

(* res = bf16(res + prj) *)
inline_for_extraction noextract
fn k_resadd (res : larray bf16 1024) (prj : larray bf16 6144)
  (#pp : perm) (#sr : erased (S.seq bf16)) (#sp : erased (S.seq bf16))
  preserves gpu ** (prj |-> Frac pp sp)
  requires res |-> sr
  ensures exists* s'. (res |-> s') ** pure (resadd_post sr sp s')
{
  pts_to_len res; pts_to_len prj;
  let mut i = 0sz;
  while (!i <^ 1024sz)
    invariant exists* vi s'. (i |-> vi) ** (res |-> s') **
      pure (SZ.v vi <= 1024 /\ S.length s' == 1024 /\
            (forall (k : nat). k < 1024 ==>
               S.index s' k == (if k < SZ.v vi then f2b (add (b2f (S.index sr k)) (b2f (S.index sp k))) else S.index sr k)))
    decreases (1024 - SZ.v !i)
  {
    let vi = !i;
    let r = b2f res.(vi);
    let p = b2f prj.(vi);
    res.(vi) <- f2b (add r p);
    i := vi +^ 1sz;
  };
}

(* inp = bf16(bf16(silu(gate)) * up) *)
inline_for_extraction noextract
fn k_silumul (gate : larray bf16 3584) (prj : larray bf16 6144) (inp : larray bf16 3584)
  (#pg #pp : perm) (#sg #sp : erased (S.seq bf16)) (#si : erased (S.seq bf16))
  preserves gpu ** (gate |-> Frac pg sg) ** (prj |-> Frac pp sp)
  requires inp |-> si
  ensures exists* s'. (inp |-> s') ** pure (silumul_post sg sp s')
{
  pts_to_len gate; pts_to_len prj; pts_to_len inp;
  let mut i = 0sz;
  while (!i <^ 3584sz)
    invariant exists* vi s'. (i |-> vi) ** (inp |-> s') **
      pure (SZ.v vi <= 3584 /\ S.length s' == 3584 /\
            (forall (k : nat). k < SZ.v vi ==>
               S.index s' k == f2b (mul (b2f (f2b (silu_f (b2f (S.index sg k))))) (b2f (S.index sp k)))))
    decreases (3584 - SZ.v !i)
  {
    let vi = !i;
    let g = b2f (f2b (silu_f (b2f gate.(vi))));
    let p = b2f prj.(vi);
    inp.(vi) <- f2b (mul g p);
    i := vi +^ 1sz;
  };
}

#pop-options
(* ===== Comp ===== *)
open FStar.Real
open Kuiper
open KPlay.SharedSpec


#push-options "--z3rlimit 40 --fuel 0 --ifuel 0"

(*** Sequence approximation <-> prefix approximation *)

noextract
let seq_pb (s : S.seq bf16) (r : S.seq real)
  : Lemma (requires s %~ r) (ensures pb s (S.length s) r /\ S.length r == S.length s)
  = ()

noextract
let seq_pf (s : S.seq float) (r : S.seq real)
  : Lemma (requires s %~ r) (ensures pf s (S.length s) r /\ S.length r == S.length s)
  = ()

noextract
let pb_seq (s : S.seq bf16) (r : S.seq real)
  : Lemma (requires pb s (S.length s) r /\ S.length r == S.length s) (ensures s %~ r)
  = ()

noextract
let pf_seq (s : S.seq float) (r : S.seq real)
  : Lemma (requires pf s (S.length s) r /\ S.length r == S.length s) (ensures s %~ r)
  = ()

(*** Index arithmetic *)

noextract
let d4 (c k : nat) : Lemma (requires k < 4) (ensures (c * 4 + k) % 4 == k /\ (c * 4 + k) / 4 == c)
  = L.lemma_div_plus k c 4; L.small_div k 4; L.lemma_mod_plus k c 4; L.small_mod k 4

noextract
let d4_of (j : nat) : Lemma (j == (j / 4) * 4 + j % 4 /\ j % 4 < 4)
  = L.euclidean_division_definition j 4

noextract
let hd_of (i : nat) : Lemma (i == (i / 128) * 128 + i % 128 /\ i % 128 < 128)
  = L.euclidean_division_definition i 128

noextract
let sidx_of (idx : nat)
  : Lemma (requires idx < 262144)
          (ensures idx / 16384 < 16 /\ (idx % 16384) / 128 < 128 /\ idx % 128 < 128 /\
                   sidx (idx / 16384) ((idx % 16384) / 128) (idx % 128) == idx)
  = let h = idx / 16384 in let r = idx % 16384 in
    L.euclidean_division_definition idx 16384;
    L.euclidean_division_definition r 128;
    assert (h * 16384 == (h * 128) * 128);
    L.lemma_mod_plus r (h * 128) 128

(*** Stage lemmas over reals *)

noextract
let norm_eq (rx rw : S.seq real) (woff i : nat)
  : Lemma (requires i < 1024) (ensures at (rmsnorm1p eps_f32 rx 1024 rw woff) i == norm_el_r rx rw woff i)
  = ()

noextract
let norm_real (sr sw : S.seq bf16) (woff : nat) (si : S.seq bf16) (rx rw : S.seq real)
  : Lemma (requires norm_post sr sw woff si /\ pb sr 1024 rx /\ pb sw (S.length sw) rw)
          (ensures pb si 1024 (rmsnorm1p eps_f32 rx 1024 rw woff))
  = let e (i : nat) : Lemma (i < 1024 ==> S.index si i %~ at (rmsnorm1p eps_f32 rx 1024 rw woff) i)
      = if i < 1024 then begin norm_eq rx rw woff i; assert (norm_el_ok sr sw woff i (S.index si i)) end in
    FStar.Classical.forall_intro e

noextract
let proj_real (sw si : S.seq bf16) (woff kk nt : nat) (sd : S.seq bf16) (rw rh : S.seq real)
  : Lemma (requires proj_g1 sw si woff kk nt sd /\ pb sw (S.length sw) rw /\ pb si kk rh)
          (ensures pb sd (nt * 16) (matvec rw woff (nt * 16) kk rh))
  = ()

noextract
let rbf (rw rx : S.seq real) (h : nat) : real = dn_beta base_numerics rw 0 rx h

noextract
let beta_real (sp sb : S.seq bf16) (rw rx : S.seq real)
  : Lemma (requires beta_post sp sb /\ pb sp 16 (matvec rw 19400704 16 1024 (dn_hin base_numerics rw 0 rx)))
          (ensures pb sb 16 (S.init 16 (rbf rw rx)))
  = let rh = dn_hin base_numerics rw 0 rx in
    let e (h : nat) : Lemma (h < 16 ==> S.index sb h %~ at (S.init 16 (rbf rw rx)) h)
      = if h < 16 then begin
          let d = dot rw (19400704 + h * 1024) rh 1024 in
          assert (S.index sp h %~ d);
          ap_b2f (S.index sp h) d;
          ap_sigm (b2f (S.index sp h)) d;
          ap_f2b (sigm_f (b2f (S.index sp h))) (sigmoid d)
        end in
    FStar.Classical.forall_intro e

noextract
let cvf (rw rx rc : S.seq real) (c : nat) (k : nat) : real =
  at (dn_conv base_numerics rw 0 rx rc) (c * 4 + k) *. at rw (dn_mix 0 + 8421376 + c * 4 + k)

#push-options "--fuel 1"
noextract
let vsum4 (f : nat -> real) : Lemma (vsum 4 f == (((0.0R +. f 0) +. f 1) +. f 2) +. f 3)
  = vsum_succ 3 f; vsum_succ 2 f; vsum_succ 1 f; vsum_succ 0 f; vsum_zero f
#pop-options

noextract
let cv_eq (rw rx rc : S.seq real) (c : nat)
  : Lemma (requires c < 6144)
          (ensures cv_r rw rc (dn_qkv base_numerics rw 0 rx) c == vsum 4 (cvf rw rx rc c))
  = let m = dn_qkv base_numerics rw 0 rx in
    let cc = dn_conv base_numerics rw 0 rx rc in
    d4 c 0; d4 c 1; d4 c 2; d4 c 3;
    assert (at cc (c * 4) == at rc (c * 4 + 1));
    assert (at cc (c * 4 + 1) == at rc (c * 4 + 2));
    assert (at cc (c * 4 + 2) == at rc (c * 4 + 3));
    assert (at cc (c * 4 + 3) == at m c);
    vsum4 (cvf rw rx rc c)

noextract
let co_el (rw rx rc : S.seq real) (c : nat)
  : Lemma (requires c < 6144)
          (ensures at (dn_conv_out base_numerics rw 0 rx rc) c == silu (cv_r rw rc (dn_qkv base_numerics rw 0 rx) c))
  = cv_eq rw rx rc c;
    vsum_ext_all 4 (cvf rw rx rc c)

noextract
let conv_sh (sc sq so : S.seq bf16) (rw rx rc : S.seq real) (c r : nat)
  : Lemma (requires c < 6144 /\ r < 4 /\ S.length so == 24576 /\ S.length sc == 24576 /\ S.length sq == 6144 /\
                    S.index so (4 * c) == S.index sc (4 * c + 1) /\ S.index so (4 * c + 1) == S.index sc (4 * c + 2) /\
                    S.index so (4 * c + 2) == S.index sc (4 * c + 3) /\ S.index so (4 * c + 3) == S.index sq c /\
                    pb sc 24576 rc /\ pb sq 6144 (dn_qkv base_numerics rw 0 rx))
          (ensures S.index so (c * 4 + r) %~ at (dn_conv base_numerics rw 0 rx rc) (c * 4 + r))
  = d4 c r;
    let m = dn_qkv base_numerics rw 0 rx in
    let cc = dn_conv base_numerics rw 0 rx rc in
    assert (at cc (c * 4 + r) == (if r < 3 then at rc (c * 4 + r + 1) else at m c));
    if r < 3 then assert (S.index sc (c * 4 + r + 1) %~ at rc (c * 4 + r + 1))
    else assert (S.index sq c %~ at m c)

noextract
let conv_real (sw sc sq so sq' : S.seq bf16) (rw rx rc : S.seq real)
  : Lemma (requires conv_post sw sc sq so sq' /\ pb sw (S.length sw) rw /\ pb sc 24576 rc /\
                    pb sq 6144 (dn_qkv base_numerics rw 0 rx))
          (ensures pb sq' 6144 (dn_conv_out base_numerics rw 0 rx rc) /\
                   pb so 24576 (dn_conv base_numerics rw 0 rx rc))
  = let m = dn_qkv base_numerics rw 0 rx in
    let e1 (c : nat) : Lemma (c < 6144 ==> S.index sq' c %~ at (dn_conv_out base_numerics rw 0 rx rc) c)
      = if c < 6144 then begin
          co_el rw rx rc c;
          assert (conv_ok sw sc sq c (S.index sq' c))
        end in
    FStar.Classical.forall_intro e1;
    let e2 (j : nat) : Lemma (j < 24576 ==> S.index so j %~ at (dn_conv base_numerics rw 0 rx rc) j)
      = if j < 24576 then begin
          d4_of j;
          let c = j / 4 in
          conv_sh sc sq so rw rx rc c (j % 4)
        end in
    FStar.Classical.forall_intro e2

(*** qk-norm *)

noextract
let qkr (co : S.seq real) (i : nat) : real = if i < 4096 then qk_el_r co i else at co i

noextract
let rqr (co : S.seq real) : S.seq real = S.init 6144 (qkr co)

noextract
let qk_real (sq sq' : S.seq bf16) (co : S.seq real)
  : Lemma (requires qk_post sq sq' /\ pb sq 6144 co) (ensures pb sq' 6144 (rqr co))
  = let e (i : nat) : Lemma (i < 6144 ==> S.index sq' i %~ at (rqr co) i)
      = if i < 4096 then assert (qk_el_ok sq i (S.index sq' i))
        else if i < 6144 then assert (S.index sq' i == S.index sq i) in
    FStar.Classical.forall_intro e

(*** Recurrence *)

noextract
let co_ (rw rx rc : S.seq real) : S.seq real = dn_conv_out base_numerics rw 0 rx rc

noextract
let rset_of (rw rx rc rs : S.seq real) : krset =
  { rw = rw; rq = rqr (co_ rw rx rc); ra = matvec rw 19417088 16 1024 (dn_hin base_numerics rw 0 rx);
    rb = S.init 16 (rbf rw rx); rs = rs }

noextract
let rq_q (rw rx rc rs : S.seq real) (h i : nat)
  : Lemma (requires h < 16 /\ i < 128)
          (ensures at (rset_of rw rx rc rs).rq (h * 128 + i) *. qscale_f32 == at (dn_q base_numerics rw 0 rx rc) (h * 128 + i))
  = hdec h i

noextract
let rq_k (rw rx rc rs : S.seq real) (h i : nat)
  : Lemma (requires h < 16 /\ i < 128)
          (ensures at (rset_of rw rx rc rs).rq (2048 + h * 128 + i) == at (dn_k base_numerics rw 0 rx rc) (h * 128 + i))
  = hdec h i; hdec (16 + h) i;
    assert (2048 + h * 128 + i == (16 + h) * 128 + i)

noextract
let rq_v (rw rx rc rs : S.seq real) (h j : nat)
  : Lemma (requires h < 16 /\ j < 128)
          (ensures at (rset_of rw rx rc rs).rq (4096 + h * 128 + j) == at (dn_v base_numerics rw 0 rx rc) (h * 128 + j))
  = ()

noextract
let dec_eq (rw rx rc rs : S.seq real) (h : nat)
  : Lemma (requires h < 16) (ensures dec_r (rset_of rw rx rc rs) h == dn_decay base_numerics rw 0 rx h)
  = ()

noextract
let mem_g (rw rx rc rs : S.seq real) (h j : nat) (i : nat) : real =
  (at rs (h * 16384 + i * 128 + j) *. dn_decay base_numerics rw 0 rx h) *. at (dn_k base_numerics rw 0 rx rc) (h * 128 + i)

noextract
let delta_eq (rw rx rc rs : S.seq real) (h j : nat)
  : Lemma (requires h < 16 /\ j < 128)
          (ensures delta_r (rset_of rw rx rc rs) h j == dn_delta base_numerics rw 0 rx rc rs h j)
  = let r = rset_of rw rx rc rs in
    dec_eq rw rx rc rs h;
    let e (i : nat) : Lemma (i < 128 ==> memf r (dec_r r h) h j i == mem_g rw rx rc rs h j i)
      = if i < 128 then rq_k rw rx rc rs h i in
    FStar.Classical.forall_intro e;
    vsum_ext 128 (memf r (dec_r r h) h j) (mem_g rw rx rc rs h j);
    vsum_ext_all 128 (mem_g rw rx rc rs h j);
    rq_v rw rx rc rs h j

noextract
let st_eq (rw rx rc rs : S.seq real) (h i j : nat)
  : Lemma (requires h < 16 /\ i < 128 /\ j < 128)
          (ensures sidx h i j < 262144 /\
                   st_r (rset_of rw rx rc rs) h i j == at (dn_state base_numerics rw 0 rx rc rs) (sidx h i j))
  = sidx_dec h i j;
    dec_eq rw rx rc rs h; rq_k rw rx rc rs h i; delta_eq rw rx rc rs h j

noextract
let core_g (rw rx rc rs : S.seq real) (h j : nat) (i : nat) : real =
  at (dn_state base_numerics rw 0 rx rc rs) (h * 16384 + i * 128 + j) *. at (dn_q base_numerics rw 0 rx rc) (h * 128 + i)

noextract
let core_eq (rw rx rc rs : S.seq real) (h j : nat)
  : Lemma (requires h < 16 /\ j < 128)
          (ensures core_r (rset_of rw rx rc rs) h j == at (dn_core base_numerics rw 0 rx rc rs) (h * 128 + j))
  = let r = rset_of rw rx rc rs in
    hdec h j;
    let e (i : nat) : Lemma (i < 128 ==> coref r h j i == core_g rw rx rc rs h j i)
      = if i < 128 then begin st_eq rw rx rc rs h i j; rq_q rw rx rc rs h i end in
    FStar.Classical.forall_intro e;
    vsum_ext 128 (coref r h j) (core_g rw rx rc rs h j);
    vsum_ext_all 128 (core_g rw rx rc rs h j)

noextract
let st_get (ii : rins) (h i j : nat) (v : float) (r : krset)
  : Lemma (requires st_ok ii h i j v /\ rprem ii r) (ensures v %~ st_r r h i j)
  = reveal_opaque (`%st_ok) (st_ok ii h i j v)

noextract
let core_get (ii : rins) (h j : nat) (v : bf16) (r : krset)
  : Lemma (requires core_ok ii h j v /\ rprem ii r) (ensures v %~ core_r r h j)
  = reveal_opaque (`%core_ok) (core_ok ii h j v)

noextract
let rd_st (ii : rins) (so : S.seq float) (sc : S.seq bf16) (h i j : nat)
  : Lemma (requires rec_done ii so sc 16 0 /\ h < 16 /\ i < 128 /\ j < 128)
          (ensures st_ok ii h i j (fat so (sidx h i j)))
  = reveal_opaque (`%rec_done) (rec_done ii so sc 16 0)

noextract
let rd_core (ii : rins) (so : S.seq float) (sc : S.seq bf16) (h j : nat)
  : Lemma (requires rec_done ii so sc 16 0 /\ h < 16 /\ j < 128)
          (ensures core_ok ii h j (bat sc (h * 128 + j)))
  = reveal_opaque (`%rec_done) (rec_done ii so sc 16 0)

noextract
let hd16 (c : nat)
  : Lemma (requires c < 2048) (ensures c / 128 < 16 /\ c % 128 < 128 /\ c == (c / 128) * 128 + c % 128)
  = L.euclidean_division_definition c 128

noextract
let rec_e1 (ii : rins) (r : krset) (so : S.seq float) (sc : S.seq bf16) (rw rx rc rs : S.seq real) (idx : nat)
  : Lemma (requires idx < 262144 /\ S.length so == 262144 /\ rec_done ii so sc 16 0 /\ rprem ii r /\
                    r == rset_of rw rx rc rs)
          (ensures S.index so idx %~ at (dn_state base_numerics rw 0 rx rc rs) idx)
  = sidx_of idx;
    let h = idx / 16384 in let i = (idx % 16384) / 128 in let j = idx % 128 in
    rd_st ii so sc h i j;
    st_get ii h i j (fat so (sidx h i j)) r;
    st_eq rw rx rc rs h i j

noextract
let rec_e2 (ii : rins) (r : krset) (so : S.seq float) (sc : S.seq bf16) (rw rx rc rs : S.seq real) (c : nat)
  : Lemma (requires c < 2048 /\ S.length sc == 2048 /\ rec_done ii so sc 16 0 /\ rprem ii r /\
                    r == rset_of rw rx rc rs)
          (ensures S.index sc c %~ at (dn_core base_numerics rw 0 rx rc rs) c)
  = hd16 c;
    let h = c / 128 in let j = c % 128 in
    rd_core ii so sc h j;
    core_get ii h j (bat sc (h * 128 + j)) r;
    core_eq rw rx rc rs h j

noextract
let rec_real (sw sq sp sb : S.seq bf16) (ss : S.seq float) (so : S.seq float) (sc : S.seq bf16)
             (rw rx rc rs : S.seq real)
  : Lemma (requires S.length so == 262144 /\ S.length sc == 2048 /\ rec_ok (mk_ins sw sq sp sb ss) so sc /\
                    rprem (mk_ins sw sq sp sb ss) (rset_of rw rx rc rs))
          (ensures pf so 262144 (dn_state base_numerics rw 0 rx rc rs) /\
                   pb sc 2048 (dn_core base_numerics rw 0 rx rc rs))
  = let ii = mk_ins sw sq sp sb ss in
    let r = rset_of rw rx rc rs in
    let e1 (idx : nat) : Lemma (idx < 262144 ==> S.index so idx %~ at (dn_state base_numerics rw 0 rx rc rs) idx)
      = if idx < 262144 then rec_e1 ii r so sc rw rx rc rs idx in
    FStar.Classical.forall_intro e1;
    let e2 (c : nat) : Lemma (c < 2048 ==> S.index sc c %~ at (dn_core base_numerics rw 0 rx rc rs) c)
      = if c < 2048 then rec_e2 ii r so sc rw rx rc rs c in
    FStar.Classical.forall_intro e2

(*** Gated norm, residual adds, MLP activation *)

noextract
let gn_eq (rw rx rc rs : S.seq real) (c : nat)
  : Lemma (requires c < 2048)
          (ensures gn_el_r rw (dn_core base_numerics rw 0 rx rc rs)
                     (matvec rw (dn_mix 0 + 6291456) 2048 1024 (dn_hin base_numerics rw 0 rx)) c ==
                   at (dn_gated base_numerics rw 0 rx rc rs) c)
  = ()

noextract
let gn_real (sw sc sz si : S.seq bf16) (rw rx rc rs : S.seq real)
  : Lemma (requires gn_post sw sc sz si /\ pb sw (S.length sw) rw /\
                    pb sc 2048 (dn_core base_numerics rw 0 rx rc rs) /\
                    pb sz 2048 (matvec rw (dn_mix 0 + 6291456) 2048 1024 (dn_hin base_numerics rw 0 rx)))
          (ensures pb si 2048 (dn_gated base_numerics rw 0 rx rc rs))
  = let e (c : nat) : Lemma (c < 2048 ==> S.index si c %~ at (dn_gated base_numerics rw 0 rx rc rs) c)
      = if c < 2048 then begin gn_eq rw rx rc rs c; assert (gn_el_ok sw sc sz c (S.index si c)) end in
    FStar.Classical.forall_intro e

noextract
let copy_real (sx s' : S.seq bf16) (rx : S.seq real)
  : Lemma (requires copy_post sx s' /\ pb sx 1024 rx) (ensures pb s' 1024 rx)
  = ()

noextract
let resadd_real (sr sp s' : S.seq bf16) (rx ry : S.seq real)
  : Lemma (requires resadd_post sr sp s' /\ pb sr 1024 rx /\ pb sp 1024 ry) (ensures pb s' 1024 (vadd rx ry 1024))
  = let e (i : nat) : Lemma (i < 1024 ==> S.index s' i %~ at (vadd rx ry 1024) i)
      = if i < 1024 then begin
          ap_b2f (S.index sr i) (at rx i); ap_b2f (S.index sp i) (at ry i);
          ap_add (b2f (S.index sr i)) (b2f (S.index sp i)) (at rx i) (at ry i);
          ap_f2b (add (b2f (S.index sr i)) (b2f (S.index sp i))) (at rx i +. at ry i)
        end in
    FStar.Classical.forall_intro e

noextract
let silumul_real (sg sp s' : S.seq bf16) (rw x1 : S.seq real)
  : Lemma (requires silumul_post sg sp s' /\
                    pb sg 3584 (matvec rw 2048 3584 1024 (mlp_in base_numerics rw 0 x1)) /\
                    pb sp 3584 (matvec rw 3672064 3584 1024 (mlp_in base_numerics rw 0 x1)))
          (ensures pb s' 3584 (mlp_act base_numerics rw 0 x1))
  = let h = mlp_in base_numerics rw 0 x1 in
    let g = matvec rw 2048 3584 1024 h in
    let u = matvec rw 3672064 3584 1024 h in
    let e (i : nat) : Lemma (i < 3584 ==> S.index s' i %~ at (mlp_act base_numerics rw 0 x1) i)
      = if i < 3584 then begin
          let a = S.index sg i in let b = S.index sp i in
          ap_b2f a (at g i); ap_silu (b2f a) (at g i);
          ap_f2b (silu_f (b2f a)) (silu (at g i)); ap_b2f (f2b (silu_f (b2f a))) (silu (at g i));
          ap_b2f b (at u i);
          ap_mul (b2f (f2b (silu_f (b2f a)))) (b2f b) (silu (at g i)) (at u i);
          ap_f2b (mul (b2f (f2b (silu_f (b2f a)))) (b2f b)) (silu (at g i) *. at u i)
        end in
    FStar.Classical.forall_intro e

(*** Whole-layer composition *)

noextract
let rprem_of (sw sq sp sb : S.seq bf16) (ss : S.seq float) (rw rx rc rs : S.seq real)
  : Lemma (requires pb sw (S.length sw) rw /\ pb sq 6144 (rqr (co_ rw rx rc)) /\
                    pb sp 16 (matvec rw 19417088 16 1024 (dn_hin base_numerics rw 0 rx)) /\
                    pb sb 16 (S.init 16 (rbf rw rx)) /\ pf ss 262144 rs)
          (ensures rprem (mk_ins sw sq sp sb ss) (rset_of rw rx rc rs))
  = ()

noeq type stages = {
  sw : S.seq bf16; sx : S.seq bf16; sc : S.seq bf16; ss : S.seq float;
  a0 : S.seq bf16; i1 : S.seq bf16; q0 : S.seq bf16; zz : S.seq bf16; pbb : S.seq bf16; bt : S.seq bf16;
  pa : S.seq bf16; o1 : S.seq bf16; q1 : S.seq bf16; q2 : S.seq bf16; o2 : S.seq float; cr : S.seq bf16;
  i2 : S.seq bf16; po : S.seq bf16; a1 : S.seq bf16; i3 : S.seq bf16; g : S.seq bf16; pu : S.seq bf16;
  i4 : S.seq bf16; pd : S.seq bf16; a2 : S.seq bf16
}

noextract
let stages_ok (t : stages) : prop =
  S.length t.sw == dn_len /\ S.length t.sx == 1024 /\ S.length t.sc == 24576 /\ S.length t.ss == 262144 /\
  copy_post t.sx t.a0 /\ norm_post t.a0 t.sw 0 t.i1 /\
  proj_g1 t.sw t.i1 11012096 1024 384 t.q0 /\ proj_g1 t.sw t.i1 17303552 1024 128 t.zz /\
  proj_g1 t.sw t.i1 19400704 1024 1 t.pbb /\ beta_post t.pbb t.bt /\
  proj_g1 t.sw t.i1 19417088 1024 1 t.pa /\
  conv_post t.sw t.sc t.q0 t.o1 t.q1 /\ qk_post t.q1 t.q2 /\
  S.length t.o2 == 262144 /\ S.length t.cr == 2048 /\ rec_ok (mk_ins t.sw t.q2 t.pa t.bt t.ss) t.o2 t.cr /\
  gn_post t.sw t.cr t.zz t.i2 /\ proj_g1 t.sw t.i2 19458208 2048 64 t.po /\ resadd_post t.a0 t.po t.a1 /\
  norm_post t.a1 t.sw 1024 t.i3 /\ proj_g1 t.sw t.i3 2048 1024 224 t.g /\ proj_g1 t.sw t.i3 3672064 1024 224 t.pu /\
  silumul_post t.g t.pu t.i4 /\ proj_g1 t.sw t.i4 7342080 3584 64 t.pd /\ resadd_post t.a1 t.pd t.a2

noextract
let comp_chain (t : stages) (rw rx rc rs : S.seq real)
  : Lemma (requires stages_ok t /\ t.sw %~ rw /\ t.sx %~ rx /\ t.sc %~ rc /\ t.ss %~ rs)
          (ensures t.a2 %~ dn_out base_numerics rw 0 rx rc rs /\
                   t.o1 %~ dn_conv base_numerics rw 0 rx rc /\
                   t.o2 %~ dn_state base_numerics rw 0 rx rc rs)
  = seq_pb t.sw rw; seq_pb t.sx rx; seq_pb t.sc rc; seq_pf t.ss rs;
    copy_real t.sx t.a0 rx;
    norm_real t.a0 t.sw 0 t.i1 rx rw;
    let rh = dn_hin base_numerics rw 0 rx in
    assert (rmsnorm1p eps_f32 rx 1024 rw 0 == rh);
    proj_real t.sw t.i1 11012096 1024 384 t.q0 rw rh;
    proj_real t.sw t.i1 17303552 1024 128 t.zz rw rh;
    proj_real t.sw t.i1 19400704 1024 1 t.pbb rw rh;
    proj_real t.sw t.i1 19417088 1024 1 t.pa rw rh;
    assert (matvec rw 11012096 6144 1024 rh == dn_qkv base_numerics rw 0 rx);
    beta_real t.pbb t.bt rw rx;
    conv_real t.sw t.sc t.q0 t.o1 t.q1 rw rx rc;
    qk_real t.q1 t.q2 (co_ rw rx rc);
    rprem_of t.sw t.q2 t.pa t.bt t.ss rw rx rc rs;
    rec_real t.sw t.q2 t.pa t.bt t.ss t.o2 t.cr rw rx rc rs;
    assert (matvec rw 17303552 2048 1024 rh == matvec rw (dn_mix 0 + 6291456) 2048 1024 rh);
    gn_real t.sw t.cr t.zz t.i2 rw rx rc rs;
    let rg = dn_gated base_numerics rw 0 rx rc rs in
    proj_real t.sw t.i2 19458208 2048 64 t.po rw rg;
    resadd_real t.a0 t.po t.a1 rx (matvec rw 19458208 1024 2048 rg);
    let x1 = dn_x1 base_numerics rw 0 rx rc rs in
    assert (vadd rx (matvec rw 19458208 1024 2048 rg) 1024 == x1);
    norm_real t.a1 t.sw 1024 t.i3 x1 rw;
    let h2 = mlp_in base_numerics rw 0 x1 in
    assert (rmsnorm1p eps_f32 x1 1024 rw 1024 == h2);
    proj_real t.sw t.i3 2048 1024 224 t.g rw h2;
    proj_real t.sw t.i3 3672064 1024 224 t.pu rw h2;
    silumul_real t.g t.pu t.i4 rw x1;
    let act = mlp_act base_numerics rw 0 x1 in
    proj_real t.sw t.i4 7342080 3584 64 t.pd rw act;
    resadd_real t.a1 t.pd t.a2 x1 (matvec rw 7342080 1024 3584 act);
    assert (vadd x1 (matvec rw 7342080 1024 3584 act) 1024 == dn_out base_numerics rw 0 rx rc rs);
    pb_seq t.a2 (dn_out base_numerics rw 0 rx rc rs);
    pb_seq t.o1 (dn_conv base_numerics rw 0 rx rc);
    pf_seq t.o2 (dn_state base_numerics rw 0 rx rc rs)

noextract
let shift_c (sc so : S.seq bf16) (c r : nat)
  : Lemma (requires c < 6144 /\ r < 3 /\ S.length so == 24576 /\ S.length sc == 24576 /\
                    S.index so (4 * c) == S.index sc (4 * c + 1) /\ S.index so (4 * c + 1) == S.index sc (4 * c + 2) /\
                    S.index so (4 * c + 2) == S.index sc (4 * c + 3))
          (ensures S.index so (c * 4 + r) == S.index sc (c * 4 + r + 1))
  = ()

noextract
let shift_real (sw sc sq so sq' : S.seq bf16)
  : Lemma (requires conv_post sw sc sq so sq')
          (ensures forall (j : nat). j + 1 < 24576 /\ j % 4 < 3 ==> S.index so j == S.index sc (j + 1))
  = let e (j : nat) : Lemma (j + 1 < 24576 /\ j % 4 < 3 ==> S.index so j == S.index sc (j + 1))
      = if j + 1 < 24576 && j % 4 < 3 then begin
          d4_of j;
          let c = j / 4 in
          assert (c < 6144);
          shift_c sc so c (j % 4)
        end in
    FStar.Classical.forall_intro e

noextract
let compose (t : stages)
  : Lemma (requires stages_ok t)
          (ensures base_recurrent_decoder_relation t.sw t.sx t.sc t.ss t.a2 t.o1 t.o2)
  = shift_real t.sw t.sc t.q0 t.o1 t.q1;
    let aux (rw rx rc rs : S.seq real)
      : Lemma ((t.sw %~ rw /\ t.sx %~ rx /\ t.sc %~ rc /\ t.ss %~ rs) ==>
               (t.a2 %~ dn_out base_numerics rw 0 rx rc rs /\
                t.o1 %~ dn_conv base_numerics rw 0 rx rc /\
                t.o2 %~ dn_state base_numerics rw 0 rx rc rs))
      = FStar.Classical.move_requires (comp_chain t rw rx rc) rs in
    FStar.Classical.forall_intro_4 aux

#pop-options
(* ===== Host ===== *)
open Kuiper
open KPlay.SharedSpec



#push-options "--z3rlimit 80 --fuel 0 --ifuel 1"

ghost fn on_len (#a : Type0) (x : array a) (#f : perm) (#s : erased (S.seq a))
  requires on gpu_loc (x |-> Frac f s)
  ensures on gpu_loc (x |-> Frac f s) ** pure (S.length s == A.length x)
{
  ghost_impersonate gpu_loc (on gpu_loc (x |-> Frac f s))
    (on gpu_loc (x |-> Frac f s) ** pure (S.length s == A.length x)) fn _ {
    on_elim (x |-> Frac f s);
    pts_to_len x;
    on_intro (x |-> Frac f s);
  };
}

noextract
let gl (#a:Type0) (x : array a) = squash (is_global x)

fn h_copy (x : larray bf16 1024) (o : larray bf16 1024) (#_ : gl x) (#_ : gl o)
  (#px : perm) (#sx : erased (S.seq bf16)) (#so : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (x |-> Frac px sx)
  requires on gpu_loc (o |-> so)
  ensures exists* s'. on gpu_loc (o |-> s') ** pure (copy_post sx s')
{ launch_kernel_1 (fun () -> k_copy x o #px #sx #so) }

fn h_norm (res : larray bf16 1024) (w : larray bf16 nW) (woff : sz) (inp : larray bf16 3584)
  (vals : larray float 1024) (part : larray float 256)
  (#_ : gl res) (#_ : gl w) (#_ : gl inp) (#_ : gl vals) (#_ : gl part)
  (#pr #pw : perm) (#sr #sw : erased (S.seq bf16)) (#si : erased (S.seq bf16))
  (#sv #sp : erased (S.seq float))
  requires pure (SZ.v woff + 1024 <= nW)
  preserves cpu ** on gpu_loc (res |-> Frac pr sr) ** on gpu_loc (w |-> Frac pw sw)
  requires on gpu_loc (inp |-> si) ** on gpu_loc (vals |-> sv) ** on gpu_loc (part |-> sp)
  ensures exists* si' sv' sp'. on gpu_loc (inp |-> si') ** on gpu_loc (vals |-> sv') ** on gpu_loc (part |-> sp') **
    pure (norm_post sr sw (SZ.v woff) si')
{ launch_kernel_1 (fun () -> k_norm res w woff inp vals part #pr #pw #sr #sw #si #sv #sp) }

fn h_bfill (inp : larray bf16 3584) (bb : larray bf16 nB) (k : sz) (#_ : gl inp) (#_ : gl bb)
  (#pi : perm) (#si : erased (S.seq bf16)) (#sb : erased (S.seq bf16))
  requires pure (0 < SZ.v k /\ SZ.v k <= 3584)
  preserves cpu ** on gpu_loc (inp |-> Frac pi si)
  requires on gpu_loc (bb |-> sb)
  ensures exists* s'. on gpu_loc (bb |-> s') ** pure (bfilled s' si (SZ.v k))
{ launch_kernel_1 (fun () -> k_bfill inp bb k #pi #si #sb) }

fn h_beta (prj : larray bf16 6144) (beta : larray bf16 16) (#_ : gl prj) (#_ : gl beta)
  (#pp : perm) (#sp : erased (S.seq bf16)) (#sb : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (prj |-> Frac pp sp)
  requires on gpu_loc (beta |-> sb)
  ensures exists* s'. on gpu_loc (beta |-> s') ** pure (beta_post sp s')
{ launch_kernel_1 (fun () -> k_beta prj beta #pp #sp #sb) }

fn h_conv (w : larray bf16 nW) (conv : larray bf16 24576) (out1 : larray bf16 24576) (qkv : larray bf16 6144)
  (#_ : gl w) (#_ : gl conv) (#_ : gl out1) (#_ : gl qkv)
  (#pw #pc : perm) (#sw #sc : erased (S.seq bf16)) (#so #sq : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (w |-> Frac pw sw) ** on gpu_loc (conv |-> Frac pc sc)
  requires on gpu_loc (out1 |-> so) ** on gpu_loc (qkv |-> sq)
  ensures exists* so' sq'. on gpu_loc (out1 |-> so') ** on gpu_loc (qkv |-> sq') ** pure (conv_post sw sc sq so' sq')
{ launch_kernel_1 (fun () -> k_conv w conv out1 qkv #pw #pc #sw #sc #so #sq) }

fn h_qknorm (qkv : larray bf16 6144) (vals : larray float 1024) (part : larray float 256)
  (#_ : gl qkv) (#_ : gl vals) (#_ : gl part)
  (#sq : erased (S.seq bf16)) (#sv #sp : erased (S.seq float))
  preserves cpu
  requires on gpu_loc (qkv |-> sq) ** on gpu_loc (vals |-> sv) ** on gpu_loc (part |-> sp)
  ensures exists* sq' sv' sp'. on gpu_loc (qkv |-> sq') ** on gpu_loc (vals |-> sv') ** on gpu_loc (part |-> sp') **
    pure (qk_post sq sq')
{ launch_kernel_1 (fun () -> k_qknorm qkv vals part #sq #sv #sp) }

fn h_rec (w : larray bf16 nW) (qkv : larray bf16 6144) (prj : larray bf16 6144) (beta : larray bf16 16)
  (st : larray float 262144) (out2 : larray float 262144) (core : larray bf16 2048)
  (upd : larray float 128) (prod : larray float 128)
  (#_ : gl w) (#_ : gl qkv) (#_ : gl prj) (#_ : gl beta) (#_ : gl st) (#_ : gl out2) (#_ : gl core) (#_ : gl upd) (#_ : gl prod)
  (#pw #pq #pp #pb #ps : perm) (#sw #sq #sp #sb : erased (S.seq bf16)) (#ss : erased (S.seq float))
  (#so : erased (S.seq float)) (#sc : erased (S.seq bf16)) (#su #sr : erased (S.seq float))
  preserves cpu ** on gpu_loc (w |-> Frac pw sw) ** on gpu_loc (qkv |-> Frac pq sq) ** on gpu_loc (prj |-> Frac pp sp) **
    on gpu_loc (beta |-> Frac pb sb) ** on gpu_loc (st |-> Frac ps ss)
  requires on gpu_loc (out2 |-> so) ** on gpu_loc (core |-> sc) ** on gpu_loc (upd |-> su) ** on gpu_loc (prod |-> sr)
  ensures exists* so' sc' su' sr'. on gpu_loc (out2 |-> so') ** on gpu_loc (core |-> sc') ** on gpu_loc (upd |-> su') ** on gpu_loc (prod |-> sr') **
    pure (S.length so' == 262144 /\ S.length sc' == 2048 /\ rec_ok (mk_ins sw sq sp sb ss) so' sc')
{ launch_kernel_1 (fun () -> k_rec w qkv prj beta st out2 core upd prod #pw #pq #pp #pb #ps #sw #sq #sp #sb #ss #so #sc #su #sr) }

fn h_gnorm (w : larray bf16 nW) (core : larray bf16 2048) (z : larray bf16 2048) (inp : larray bf16 3584)
  (vals : larray float 1024) (part : larray float 256)
  (#_ : gl w) (#_ : gl core) (#_ : gl z) (#_ : gl inp) (#_ : gl vals) (#_ : gl part)
  (#pw #pc #pz : perm) (#sw #sc #sz_ : erased (S.seq bf16)) (#si : erased (S.seq bf16)) (#sv #sp : erased (S.seq float))
  preserves cpu ** on gpu_loc (w |-> Frac pw sw) ** on gpu_loc (core |-> Frac pc sc) ** on gpu_loc (z |-> Frac pz sz_)
  requires on gpu_loc (inp |-> si) ** on gpu_loc (vals |-> sv) ** on gpu_loc (part |-> sp)
  ensures exists* si' sv' sp'. on gpu_loc (inp |-> si') ** on gpu_loc (vals |-> sv') ** on gpu_loc (part |-> sp') **
    pure (gn_post sw sc sz_ si')
{ launch_kernel_1 (fun () -> k_gnorm w core z inp vals part #pw #pc #pz #sw #sc #sz_ #si #sv #sp) }

fn h_resadd (res : larray bf16 1024) (prj : larray bf16 6144) (#_ : gl res) (#_ : gl prj)
  (#pp : perm) (#sr : erased (S.seq bf16)) (#sp : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (prj |-> Frac pp sp)
  requires on gpu_loc (res |-> sr)
  ensures exists* s'. on gpu_loc (res |-> s') ** pure (resadd_post sr sp s')
{ launch_kernel_1 (fun () -> k_resadd res prj #pp #sr #sp) }

fn h_silumul (gate : larray bf16 3584) (prj : larray bf16 6144) (inp : larray bf16 3584)
  (#_ : gl gate) (#_ : gl prj) (#_ : gl inp)
  (#pg #pp : perm) (#sg #sp : erased (S.seq bf16)) (#si : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (gate |-> Frac pg sg) ** on gpu_loc (prj |-> Frac pp sp)
  requires on gpu_loc (inp |-> si)
  ensures exists* s'. on gpu_loc (inp |-> s') ** pure (silumul_post sg sp s')
{ launch_kernel_1 (fun () -> k_silumul gate prj inp #pg #pp #sg #sp #si) }

fn proj (w : larray bf16 nW) (bb : larray bf16 nB) (cc : larray float nC) (#n : erased nat) (dst : larray bf16 n)
  (#_ : gl w) (#_ : gl bb) (#_ : gl cc) (#_ : gl dst)
  (woff kk nt nch chunk : sz) (si : erased (S.seq bf16))
  (#fw #fb : perm) (#sw : erased (S.seq bf16)) (#sb : erased (S.seq bf16))
  (#sc : erased (S.seq float)) (#sd : erased (S.seq bf16))
  requires pure (pok (SZ.v woff) (SZ.v kk) (SZ.v nt) (SZ.v nch) (SZ.v chunk) /\ SZ.v nt * 16 <= n /\ (SZ.v nch == 1 \/ SZ.v nch == 2))
  requires pure (S.length sw == nW /\ S.length sc == nC /\ bfilled sb si (SZ.v kk))
  preserves cpu ** on gpu_loc (w |-> Frac fw sw) ** on gpu_loc (bb |-> Frac fb sb)
  requires on gpu_loc (cc |-> sc) ** on gpu_loc (dst |-> sd)
  ensures exists* sc' sd'. on gpu_loc (cc |-> sc') ** on gpu_loc (dst |-> sd') **
    pure (S.length sc' == nC /\ proj_g1 sw si (SZ.v woff) (SZ.v kk) (SZ.v nt) sd')
{
  on_len cc;
  wmma_proj w bb cc woff kk nt nch chunk #fw #fb #sw #sb #sc;
  with sc1. assert on gpu_loc (cc |-> sc1);
  launch_kernel_1 (fun () -> k_comb cc dst nt nch #1.0R #sc1 #sd);
  with sd1. assert on gpu_loc (dst |-> sd1);
  proj_lemma sw sb si sc (SZ.v woff) (SZ.v kk) (SZ.v nt) (SZ.v nch) (SZ.v chunk) sd1;
  ()
}

noextract
let pok_all () : Lemma (pok 11012096 1024 384 1 1024 /\ pok 17303552 1024 128 1 1024 /\ pok 19400704 1024 1 1 1024 /\
                        pok 19417088 1024 1 1 1024 /\ pok 19458208 2048 64 1 2048 /\ pok 2048 1024 224 1 1024 /\
                        pok 3672064 1024 224 1 1024 /\ pok 7342080 3584 64 2 1792) = ()

fn kplay_run
  (w : larray bf16 21555360 { is_global w })
  (x : larray bf16 1024 { is_global x })
  (conv : larray bf16 24576 { is_global conv })
  (st : larray float 262144 { is_global st })
  (out0 : larray bf16 1024 { is_global out0 })
  (out1 : larray bf16 24576 { is_global out1 })
  (out2 : larray float 262144 { is_global out2 })
  (#pw #px #pc #ps : perm)
  (#sw #sx #sc : erased (S.seq bf16)) (#ss : erased (S.seq float))
  (#o0 #o1 : erased (S.seq bf16)) (#o2 : erased (S.seq float))
  preserves cpu ** on gpu_loc (w |-> Frac pw sw) ** on gpu_loc (x |-> Frac px sx) **
    on gpu_loc (conv |-> Frac pc sc) ** on gpu_loc (st |-> Frac ps ss)
  requires on gpu_loc (out0 |-> o0) ** on gpu_loc (out1 |-> o1) ** on gpu_loc (out2 |-> o2)
  ensures exists* (a b : S.seq bf16) (c : S.seq float).
    on gpu_loc (out0 |-> a) ** on gpu_loc (out1 |-> b) ** on gpu_loc (out2 |-> c) **
    pure (base_recurrent_decoder_relation sw sx sc ss a b c)
{
  on_len w; on_len x; on_len conv; on_len st; on_len out0; on_len out1; on_len out2;
  let inp = gpu_array_alloc #bf16 3584sz;
  let bb = gpu_array_alloc #bf16 57344sz;
  let prj = gpu_array_alloc #bf16 6144sz;
  let qkv = gpu_array_alloc #bf16 6144sz;
  let z = gpu_array_alloc #bf16 2048sz;
  let beta = gpu_array_alloc #bf16 16sz;
  let gate = gpu_array_alloc #bf16 3584sz;
  let core = gpu_array_alloc #bf16 2048sz;
  let cc = gpu_array_alloc #float 49152sz;
  let vals = gpu_array_alloc #float 1024sz;
  let part = gpu_array_alloc #float 256sz;
  let upd = gpu_array_alloc #float 128sz;
  let prod = gpu_array_alloc #float 128sz;
  on_len cc;
  pok_all ();
  (* stage 1 *)
  h_copy x out0;
  with a0. assert on gpu_loc (out0 |-> a0);
  h_norm out0 w 0sz inp vals part;
  with i1. assert on gpu_loc (inp |-> i1);
  (* p0..p3 *)
  h_bfill inp bb 1024sz;
  proj w bb cc qkv (c32 11012096ul) 1024sz 384sz 1sz 1024sz i1;
  with q0. assert on gpu_loc (qkv |-> q0);
  proj w bb cc z (c32 17303552ul) 1024sz 128sz 1sz 1024sz i1;
  with zz. assert on gpu_loc (z |-> zz);
  proj w bb cc prj (c32 19400704ul) 1024sz 1sz 1sz 1024sz i1;
  with pbb. assert on gpu_loc (prj |-> pbb);
  h_beta prj beta;
  with bt. assert on gpu_loc (beta |-> bt);
  proj w bb cc prj (c32 19417088ul) 1024sz 1sz 1sz 1024sz i1;
  with pa. assert on gpu_loc (prj |-> pa);
  h_conv w conv out1 qkv;
  with o1 q1. assert (on gpu_loc (out1 |-> o1) ** on gpu_loc (qkv |-> q1));
  h_qknorm qkv vals part;
  with q2. assert on gpu_loc (qkv |-> q2);
  h_rec w qkv prj beta st out2 core upd prod;
  with o2 cr. assert (on gpu_loc (out2 |-> o2) ** on gpu_loc (core |-> cr));
  h_gnorm w core z inp vals part;
  with i2. assert on gpu_loc (inp |-> i2);
  (* p4 *)
  h_bfill inp bb 2048sz;
  proj w bb cc prj (c32 19458208ul) 2048sz 64sz 1sz 2048sz i2;
  with po. assert on gpu_loc (prj |-> po);
  h_resadd out0 prj;
  with a1. assert on gpu_loc (out0 |-> a1);
  h_norm out0 w 1024sz inp vals part;
  with i3. assert on gpu_loc (inp |-> i3);
  (* p5, p6 *)
  h_bfill inp bb 1024sz;
  proj w bb cc gate 2048sz 1024sz 224sz 1sz 1024sz i3;
  with g. assert on gpu_loc (gate |-> g);
  proj w bb cc prj (c32 3672064ul) 1024sz 224sz 1sz 1024sz i3;
  with pu. assert on gpu_loc (prj |-> pu);
  h_silumul gate prj inp;
  with i4. assert on gpu_loc (inp |-> i4);
  (* p7 *)
  h_bfill inp bb 3584sz;
  proj w bb cc prj (c32 7342080ul) 3584sz 64sz 2sz 1792sz i4;
  with pd. assert on gpu_loc (prj |-> pd);
  h_resadd out0 prj;
  with a2. assert on gpu_loc (out0 |-> a2);
  compose ({ sw = sw; sx = sx; sc = sc; ss = ss; a0 = a0; i1 = i1; q0 = q0; zz = zz; pbb = pbb; bt = bt;
             pa = pa; o1 = o1; q1 = q1; q2 = q2; o2 = o2; cr = cr; i2 = i2; po = po; a1 = a1; i3 = i3;
             g = g; pu = pu; i4 = i4; pd = pd; a2 = a2 });
  gpu_array_free inp; gpu_array_free bb; gpu_array_free prj; gpu_array_free qkv;
  gpu_array_free z; gpu_array_free beta; gpu_array_free gate; gpu_array_free core;
  gpu_array_free cc; gpu_array_free vals; gpu_array_free part; gpu_array_free upd;
  gpu_array_free prod;
  ()
}
#pop-options
