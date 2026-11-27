module KPlay.Functional

#lang-pulse

open FStar.Real
open Kuiper
open Kuiper.Approximates
open Kuiper.Seq.Common
open KPlay.SharedSpec
module S = FStar.Seq
module Exp = FStar.Math.Exp
module SZ = Kuiper.SizeT
module F32 = Kuiper.Float32
module Sqrt = FStar.Math.Sqrt
module BF = Kuiper.BFloat16
module C = Kuiper.Float.Casts.Base
module Lit = Kuiper.Float32.Literals
module Cert = Kuiper.Float32.ConstantFold.Certificates
module FL = Kuiper.Floating.Base
module AB = Kuiper.Approximates.Base
module I64 = FStar.Int64
module FM = Kuiper.Float32.FastMath
module CF = Kuiper.Float32.ConstantFold
module ML = FStar.Math.Lemmas
module AS = Kuiper.Approximates.Scalar
module U32 = FStar.UInt32
module Cast = FStar.Int.Cast

(**** Section R ****)
#set-options "--fuel 0 --ifuel 0 --z3rlimit 20"

(*** Size-generic finite sums *)

noextract let shift (g:nat -> real) (m:nat) (k:nat) : real = g (m + k)
noextract let stride (g:nat -> real) (o st:nat) (k:nat) : real = g (o + k * st)
noextract let zf (k:nat) : real = 0.0R

noextract let vinit (n:nat) (f:nat -> real) : S.seq real = S.init n (fun (k:nat{k < n}) -> f k)

noextract let vsum_vinit (n:nat) (f:nat -> real) : Lemma (vsum n f == rsum (vinit n f)) = ()

noextract let vsum_ext (n:nat) (f g:nat -> real)
  : Lemma (requires forall (k:nat). k < n ==> f k == g k) (ensures vsum n f == vsum n g)
  = vsum_vinit n f; vsum_vinit n g;
    assert (S.equal (vinit n f) (vinit n g))

#push-options "--fuel 2 --ifuel 1"
noextract let rsum_one (x:real) : Lemma (rsum (S.create 1 x) == x) =
  let s = S.create 1 x in
  assert (S.equal (S.tail s) S.empty);
  assert (view_seq s == SCons x (S.tail s));
  assert (view_seq (S.tail s) == SNil)

noextract let vsum_zero (f:nat -> real) : Lemma (vsum 0 f == 0.0R) =
  vsum_vinit 0 f;
  assert (S.equal (vinit 0 f) S.empty)
#pop-options

noextract let vsum_succ (n:nat) (f:nat -> real) : Lemma (vsum (n + 1) f == vsum n f +. f n) =
  vsum_vinit (n + 1) f; vsum_vinit n f;
  assert (S.equal (vinit (n + 1) f) (S.append (vinit n f) (S.create 1 (f n))));
  rsum_append (vinit n f) (S.create 1 (f n));
  rsum_one (f n)

noextract let rec vsum_split (m n:nat) (f:nat -> real)
  : Lemma (ensures vsum (m + n) f == vsum m f +. vsum n (shift f m)) (decreases n)
  = if n = 0 then (vsum_zero (shift f m))
    else (vsum_split m (n - 1) f; vsum_succ (m + n - 1) f; vsum_succ (n - 1) (shift f m))

noextract let fadd (f g:nat -> real) (k:nat) : real = f k +. g k
noextract let fscale (c:real) (f:nat -> real) (k:nat) : real = c *. f k

noextract let rec vsum_add (n:nat) (f g:nat -> real)
  : Lemma (ensures vsum n (fadd f g) == vsum n f +. vsum n g) (decreases n)
  = if n = 0 then (vsum_zero f; vsum_zero g; vsum_zero (fadd f g))
    else (vsum_add (n - 1) f g; vsum_succ (n - 1) f; vsum_succ (n - 1) g; vsum_succ (n - 1) (fadd f g))

noextract let rec vsum_scale (n:nat) (c:real) (f:nat -> real)
  : Lemma (ensures vsum n (fscale c f) == c *. vsum n f) (decreases n)
  = if n = 0 then (vsum_zero f; vsum_zero (fscale c f))
    else (vsum_scale (n - 1) c f; vsum_succ (n - 1) f; vsum_succ (n - 1) (fscale c f))

noextract let rec vsum_zeros (n:nat) : Lemma (ensures vsum n zf == 0.0R) (decreases n) =
  if n = 0 then vsum_zero zf else (vsum_zeros (n - 1); vsum_succ (n - 1) zf)

noextract let rec vsum_nonneg (n:nat) (f:nat -> real)
  : Lemma (requires forall (k:nat). k < n ==> f k >=. 0.0R) (ensures vsum n f >=. 0.0R) (decreases n)
  = if n = 0 then vsum_zero f else (vsum_nonneg (n - 1) f; vsum_succ (n - 1) f)

noextract let rec vsum_pos (n:pos) (f:nat -> real)
  : Lemma (requires forall (k:nat). k < n ==> f k >. 0.0R) (ensures vsum n f >. 0.0R) (decreases n)
  = if n = 1 then (vsum_zero f; vsum_succ 0 f) else (vsum_pos (n - 1) f; vsum_succ (n - 1) f)

(* double sums *)
noextract let row (h:nat -> nat -> real) (a:nat) (b:nat) : real = h a b
noextract let col (h:nat -> nat -> real) (b:nat) (a:nat) : real = h a b
noextract let rowsum (bn:nat) (h:nat -> nat -> real) (a:nat) : real = vsum bn (row h a)
noextract let colsum (an:nat) (h:nat -> nat -> real) (b:nat) : real = vsum an (col h b)

noextract let rec vsum_swap (an bn:nat) (h:nat -> nat -> real)
  : Lemma (ensures vsum an (rowsum bn h) == vsum bn (colsum an h)) (decreases an)
  = if an = 0 then (
      vsum_zero (rowsum bn h);
      introduce forall (b:nat). b < bn ==> colsum an h b == zf b with vsum_zero (col h b);
      vsum_ext bn (colsum an h) zf;
      vsum_zeros bn)
    else (
      vsum_swap (an - 1) bn h;
      vsum_succ (an - 1) (rowsum bn h);
      introduce forall (b:nat). b < bn ==> colsum an h b == fadd (colsum (an - 1) h) (row h (an - 1)) b
        with vsum_succ (an - 1) (col h b);
      vsum_ext bn (colsum an h) (fadd (colsum (an - 1) h) (row h (an - 1)));
      vsum_add bn (colsum (an - 1) h) (row h (an - 1)))

(* blocked: sum over a*bn = sum over blocks b of (sum over a < an of g (b*an + a)) *)
noextract let blk (g:nat -> real) (an:nat) (b:nat) : real = vsum an (shift g (b * an))

noextract let rec vsum_block (an bn:nat) (g:nat -> real)
  : Lemma (ensures vsum (an * bn) g == vsum bn (blk g an)) (decreases bn)
  = if bn = 0 then (vsum_zero g; vsum_zero (blk g an))
    else (
      vsum_block an (bn - 1) g;
      vsum_split (an * (bn - 1)) an g;
      vsum_succ (bn - 1) (blk g an);
      assert (an * (bn - 1) == (bn - 1) * an))

(* strided: lane a sums g (a + k*an) for k < bn *)
noextract let strd (g:nat -> real) (an bn:nat) (a:nat) : real = vsum bn (stride g a an)
noextract let strh (g:nat -> real) (an:nat) (a k:nat) : real = g (k * an + a)

noextract let vsum_strided (an bn:nat) (g:nat -> real)
  : Lemma (ensures vsum an (strd g an bn) == vsum (an * bn) g)
  = vsum_block an bn g;
    introduce forall (a:nat). a < an ==> strd g an bn a == rowsum bn (strh g an) a
      with vsum_ext bn (stride g a an) (row (strh g an) a);
    vsum_ext an (strd g an bn) (rowsum bn (strh g an));
    vsum_swap an bn (strh g an);
    introduce forall (b:nat). b < bn ==> colsum an (strh g an) b == blk g an b
      with vsum_ext an (col (strh g an) b) (shift g (b * an));
    vsum_ext bn (colsum an (strh g an)) (blk g an)

noextract let even (g:nat -> real) (k:nat) : real = g (2 * k)
noextract let odd (g:nat -> real) (k:nat) : real = g (2 * k + 1)

noextract let rec vsum_even_odd (m:nat) (g:nat -> real)
  : Lemma (ensures vsum (2 * m) g == vsum m (even g) +. vsum m (odd g)) (decreases m)
  = if m = 0 then (vsum_zero g; vsum_zero (even g); vsum_zero (odd g))
    else (
      vsum_even_odd (m - 1) g;
      vsum_succ (2 * (m - 1)) g;
      vsum_succ (2 * (m - 1) + 1) g;
      vsum_succ (m - 1) (even g);
      vsum_succ (m - 1) (odd g))

(*** Pairwise tree reduction model *)

noextract let tv (g:nat -> real) (d:pos) (i:nat) : real = vsum (32 / d) (stride g i d)

noextract let lvl (d:nat) : prop = d == 1 \/ d == 2 \/ d == 4 \/ d == 8 \/ d == 16

noextract let tv_step (g:nat -> real) (d:pos{lvl d}) (i:nat)
  : Lemma (tv g d i == tv g (2 * d) i +. tv g (2 * d) (i + d))
  = let m = 32 / (2 * d) in
    assert (32 / d == 2 * m);
    vsum_even_odd m (stride g i d);
    introduce forall (k:nat). k < m ==> even (stride g i d) k == stride g i (2 * d) k
      with (assert (2 * k * d == k * (2 * d)));
    introduce forall (k:nat). k < m ==> odd (stride g i d) k == stride g (i + d) (2 * d) k
      with (assert ((2 * k + 1) * d == d + k * (2 * d)));
    vsum_ext m (even (stride g i d)) (stride g i (2 * d));
    vsum_ext m (odd (stride g i d)) (stride g (i + d) (2 * d))

noextract let tv_top (g:nat -> real) (i:nat) : Lemma (tv g 32 i == g i) =
  vsum_succ 0 (stride g i 32); vsum_zero (stride g i 32)

noextract let tv_bot (g:nat -> real) : Lemma (tv g 1 0 == vsum 32 g) =
  vsum_ext 32 (stride g 0 1) g

noextract noeq type ins = {
  rw: S.seq real; rx: S.seq real; rk: S.seq real; rv: S.seq real; rc: S.seq real; rs: S.seq real
}

noextract let tvf (f:ins -> nat -> real) (d:pos) (r:ins) (i:nat) : real = tv (f r) d i


(**** Section T ****)
#set-options "--fuel 0 --ifuel 0 --z3rlimit 20"

noextract let apx (p:ins -> prop) (s:S.seq f32) (n:nat) (f:ins -> nat -> real) : prop =
  n <= S.length s /\ (forall (r:ins) (j:nat). p r /\ j < n ==> S.index s j %~ f r j)

noextract let apx_tv_top (p:ins -> prop) (s:S.seq f32) (f:ins -> nat -> real)
  : Lemma (requires apx p s 32 f) (ensures apx p s 32 (tvf f 32))
  = introduce forall (r:ins) (j:nat). tvf f 32 r j == f r j with tv_top (f r) j

noextract let tv_step_all (f:ins -> nat -> real) (d:pos{lvl d}) (i:nat)
  : Lemma (forall (r:ins). tvf f d r i == tvf f (2 * d) r i +. tvf f (2 * d) r (i + d))
  = introduce forall (r:ins). tvf f d r i == tvf f (2 * d) r i +. tvf f (2 * d) r (i + d)
    with tv_step (f r) d i

noextract let tv_bot_all (f:ins -> nat -> real) : Lemma (forall (r:ins). tvf f 1 r 0 == vsum 32 (f r)) =
  introduce forall (r:ins). tvf f 1 r 0 == vsum 32 (f r) with tv_bot (f r)

inline_for_extraction noextract
fn tree_level (lanes:larray f32 32) (d:sz{lvl (SZ.v d)})
  (p:erased (ins -> prop)) (f:erased (ins -> nat -> real)) (#s0:erased (S.seq f32))
  preserves gpu
  requires lanes |-> s0 ** pure (apx p s0 (2 * SZ.v d) (tvf f (2 * SZ.v d)))
  ensures exists* s1. lanes |-> s1 ** pure (S.length s1 == 32 /\ apx p s1 (SZ.v d) (tvf f (SZ.v d)))
{
  pts_to_len lanes;
  let mut i = 0sz;
  while (!i <^ d)
    invariant exists* (k:sz) s. i |-> k ** lanes |-> s ** pure (
      SZ.v k <= SZ.v d /\ S.length s == 32 /\ apx p s (SZ.v k) (tvf f (SZ.v d)) /\
      (forall (j:nat). SZ.v k <= j /\ j < 32 ==> S.index s j == S.index s0 j))
    decreases (SZ.v d - SZ.v !i)
  {
    let k = !i;
    let a = lanes.(k);
    let b = lanes.(k +^ d);
    tv_step_all f (SZ.v d) (SZ.v k);
    let c = F32.add_rn_ftz a b;
    lanes.(k) <- c;
    i := k +^ 1sz;
  }
}

inline_for_extraction noextract
fn tree32 (lanes:larray f32 32)
  (p:erased (ins -> prop)) (f:erased (ins -> nat -> real)) (#s0:erased (S.seq f32))
  preserves gpu
  requires lanes |-> s0 ** pure (apx p s0 32 f)
  returns res:f32
  ensures exists* s1. lanes |-> s1 ** pure (S.length s1 == 32 /\ (forall (r:ins). reveal p r ==> res %~ vsum 32 (reveal f r)))
{
  apx_tv_top p s0 f;
  tree_level lanes 16sz p f;
  tree_level lanes 8sz p f;
  tree_level lanes 4sz p f;
  tree_level lanes 2sz p f;
  tree_level lanes 1sz p f;
  let res = lanes.(0sz);
  tv_bot_all f;
  res
}


(**** Section L ****)
#set-options "--fuel 0 --ifuel 0 --z3rlimit 20"

(*** Approximation predicates over all admissible input ideals *)

noextract let apxs (p:ins -> prop) (s:S.seq F32.t) (n:nat) (f:ins -> S.seq real) : prop =
  n <= S.length s /\ (forall (r:ins) (j:nat). p r /\ j < n ==> S.index s j %~ at (f r) j)

noextract let wok (p:ins -> prop) (sw:S.seq BF.t) : prop = forall (r:ins). p r ==> sw %~ r.rw

noextract let wcast_all (p:ins -> prop) (sw:S.seq BF.t) (j:nat)
  : Lemma (requires wok p sw /\ j < S.length sw)
          (ensures forall (r:ins). p r ==> C.cast_bf16_to_f32 (S.index sw j) %~ at r.rw j)
  = introduce forall (r:ins). p r ==> C.cast_bf16_to_f32 (S.index sw j) %~ at r.rw j
    with introduce _ ==> _ with C.cast_bf16_to_f32_ok (S.index sw j) (at r.rw j)

noextract let rt_cast (t:F32.t) (y:real)
  : Lemma (requires t %~ y) (ensures C.cast_bf16_to_f32 (C.cast_f32_to_bf16 t) %~ y)
  = C.cast_f32_to_bf16_ok t y; C.cast_bf16_to_f32_ok (C.cast_f32_to_bf16 t) y

noextract let rt_cast_all (p:ins -> prop) (t:F32.t) (f:ins -> real)
  : Lemma (requires forall (r:ins). p r ==> t %~ f r)
          (ensures forall (r:ins). p r ==> C.cast_bf16_to_f32 (C.cast_f32_to_bf16 t) %~ f r)
  = introduce forall (r:ins). p r ==> C.cast_bf16_to_f32 (C.cast_f32_to_bf16 t) %~ f r
    with introduce _ ==> _ with rt_cast t (f r)

(*** Dot products: 32 lanes x nc chunks x 8 *)

noextract let chunk (g:nat -> real) (l:nat) (c:nat) : real = vsum 8 (shift g (c * 256 + l * 8))
noextract let lane (g:nat -> real) (nc:nat) (l:nat) : real = vsum nc (chunk g l)
noextract let hch (g:nat -> real) (c l:nat) : real = chunk g l c

noextract let dot_lanes (g:nat -> real) (nc:nat)
  : Lemma (vsum 32 (lane g nc) == vsum (256 * nc) g)
  = vsum_block 256 nc g;
    introduce forall (c:nat). c < nc ==> blk g 256 c == rowsum 32 (hch g) c
    with (
      vsum_block 8 32 (shift g (c * 256));
      introduce forall (l:nat). l < 32 ==> blk (shift g (c * 256)) 8 l == row (hch g) c l
        with vsum_ext 8 (shift (shift g (c * 256)) (l * 8)) (shift g (c * 256 + l * 8));
      vsum_ext 32 (blk (shift g (c * 256)) 8) (row (hch g) c));
    vsum_ext nc (blk g 256) (rowsum 32 (hch g));
    vsum_swap nc 32 (hch g);
    introduce forall (l:nat). l < 32 ==> colsum nc (hch g) l == lane g nc l
      with vsum_ext nc (col (hch g) l) (chunk g l);
    vsum_ext 32 (colsum nc (hch g)) (lane g nc)

noextract let dotg (wo:int) (xf:ins -> S.seq real) (r:ins) (k:nat) : real = at r.rw (wo + k) *. at (xf r) k
noextract let laneF (wo:int) (xf:ins -> S.seq real) (nc:nat) (r:ins) (l:nat) : real = lane (dotg wo xf r) nc l

noextract let dot_eq (wo:int) (xf:ins -> S.seq real) (r:ins) (n:nat)
  : Lemma (dot r.rw wo (xf r) n == vsum n (dotg wo xf r))
  = norm_spec [delta_only [`%dot]] (dot r.rw wo (xf r) n);
    vsum_ext n (fun k -> at r.rw (wo + k) *. at (xf r) k) (dotg wo xf r)

noextract let gdot_ok (wo:int) (xf:ins -> S.seq real) (nc:nat)
  : Lemma (forall (r:ins). vsum 32 (laneF wo xf nc r) == dot r.rw wo (xf r) (256 * nc))
  = introduce forall (r:ins). vsum 32 (laneF wo xf nc r) == dot r.rw wo (xf r) (256 * nc)
    with (
      vsum_ext 32 (laneF wo xf nc r) (lane (dotg wo xf r) nc);
      dot_lanes (dotg wo xf r) nc;
      dot_eq wo xf r (256 * nc))

noextract let step_shift (g:ins -> nat -> real) (kk j:nat)
  : Lemma (forall (r:ins). vsum (j + 1) (shift (g r) kk) == vsum j (shift (g r) kk) +. g r (kk + j))
  = introduce forall (r:ins). vsum (j + 1) (shift (g r) kk) == vsum j (shift (g r) kk) +. g r (kk + j)
    with vsum_succ j (shift (g r) kk)

noextract let step_chunk (g:ins -> nat -> real) (l c:nat)
  : Lemma (forall (r:ins). vsum (c + 1) (chunk (g r) l) == vsum c (chunk (g r) l) +. chunk (g r) l c)
  = introduce forall (r:ins). vsum (c + 1) (chunk (g r) l) == vsum c (chunk (g r) l) +. chunk (g r) l c
    with vsum_succ c (chunk (g r) l)

noextract let zero_all (g:ins -> nat -> real)
  : Lemma (forall (r:ins). vsum 0 (g r) == 0.0R)
  = introduce forall (r:ins). vsum 0 (g r) == 0.0R with vsum_zero (g r)

noextract let zero_shift (g:ins -> nat -> real) (kk:nat)
  : Lemma (forall (r:ins). vsum 0 (shift (g r) kk) == 0.0R)
  = introduce forall (r:ins). vsum 0 (shift (g r) kk) == 0.0R with vsum_zero (shift (g r) kk)

noextract let zero_chunk (g:ins -> nat -> real) (l:nat)
  : Lemma (forall (r:ins). vsum 0 (chunk (g r) l) == 0.0R)
  = introduce forall (r:ins). vsum 0 (chunk (g r) l) == 0.0R with vsum_zero (chunk (g r) l)

(*** Sum of squares *)

noextract let sqg (xf:ins -> S.seq real) (o:int) (r:ins) (k:nat) : real = at (xf r) (o + k) *. at (xf r) (o + k)

noextract let sq_eq (xf:ins -> S.seq real) (o:int) (r:ins) (n:nat)
  : Lemma (sq_sum (xf r) o n == vsum n (sqg xf o r))
  = norm_spec [delta_only [`%sq_sum]] (sq_sum (xf r) o n);
    vsum_ext n (fun k -> at (xf r) (o + k) *. at (xf r) (o + k)) (sqg xf o r)

noextract let sq_nonneg (xf:ins -> S.seq real) (o:int) (r:ins) (n:nat)
  : Lemma (sq_sum (xf r) o n >=. 0.0R)
  = sq_eq xf o r n; vsum_nonneg n (sqg xf o r)

(* 1024: warp w, lane l sums k < 2 of g (w*32 + l + k*512) *)
noextract let ssl (g:nat -> real) (w l:nat) : real = vsum 2 (stride g (w * 32 + l) 512)
noextract let sslF (xf:ins -> S.seq real) (w:nat) (r:ins) (l:nat) : real = ssl (sqg xf 0 r) w l
noextract let ssw (g:nat -> real) (w:nat) : real = if w < 16 then vsum 32 (ssl g w) else 0.0R
noextract let sswF (xf:ins -> S.seq real) (r:ins) (w:nat) : real = ssw (sqg xf 0 r) w

noextract let ss1024_sum (g:nat -> real)
  : Lemma (vsum 32 (ssw g) == vsum 1024 g)
  = vsum_split 16 16 (ssw g);
    vsum_ext 16 (shift (ssw g) 16) zf; vsum_zeros 16;
    introduce forall (w:nat). w < 16 ==> ssw g w == blk (strd g 512 2) 32 w
      with vsum_ext 32 (ssl g w) (shift (strd g 512 2) (w * 32));
    vsum_ext 16 (ssw g) (blk (strd g 512 2) 32);
    vsum_block 32 16 (strd g 512 2);
    vsum_strided 512 2 g

noextract let ss1024_ok (xf:ins -> S.seq real)
  : Lemma (forall (r:ins). vsum 32 (sswF xf r) == sq_sum (xf r) 0 1024)
  = introduce forall (r:ins). vsum 32 (sswF xf r) == sq_sum (xf r) 0 1024
    with (vsum_ext 32 (sswF xf r) (ssw (sqg xf 0 r)); ss1024_sum (sqg xf 0 r); sq_eq xf 0 r 1024)

noextract let ssw_lane_ok (xf:ins -> S.seq real) (w:nat{w < 16})
  : Lemma (forall (r:ins). vsum 32 (sslF xf w r) == sswF xf r w)
  = introduce forall (r:ins). vsum 32 (sslF xf w r) == sswF xf r w
    with vsum_ext 32 (sslF xf w r) (ssl (sqg xf 0 r) w)

noextract let step_stride (g:ins -> nat -> real) (o st j:nat)
  : Lemma (forall (r:ins). vsum (j + 1) (stride (g r) o st) == vsum j (stride (g r) o st) +. g r (o + j * st))
  = introduce forall (r:ins). vsum (j + 1) (stride (g r) o st) == vsum j (stride (g r) o st) +. g r (o + j * st)
    with vsum_succ j (stride (g r) o st)

noextract let zero_stride (g:ins -> nat -> real) (o st:nat)
  : Lemma (forall (r:ins). vsum 0 (stride (g r) o st) == 0.0R)
  = introduce forall (r:ins). vsum 0 (stride (g r) o st) == 0.0R with vsum_zero (stride (g r) o st)

(* 256 (head norm): lane l sums k < 8 of g (l + k*32) *)
noextract let hsF (xf:ins -> S.seq real) (o:int) (r:ins) (l:nat) : real = vsum 8 (stride (sqg xf o r) l 32)

noextract let ss256_ok (xf:ins -> S.seq real) (o:int)
  : Lemma (forall (r:ins). vsum 32 (hsF xf o r) == sq_sum (xf r) o 256)
  = introduce forall (r:ins). vsum 32 (hsF xf o r) == sq_sum (xf r) o 256
    with (vsum_ext 32 (hsF xf o r) (strd (sqg xf o r) 32 8);
          vsum_strided 32 8 (sqg xf o r); sq_eq xf o r 256)

(*** Constants *)

noextract
let eps_lit : string = "0.0000009999999974752427078783512115478515625"

noextract let eps_cert () : Lemma (Cert.exact_normal eps_lit 8796093 (-43)) =
  assert_norm (Cert.exact_normal eps_lit 8796093 (-43))

#push-options "--ifuel 1"
noextract let eps_pos () : Lemma (eps_f32 >. 0.0R /\ eps_f32 == Lit.value eps_lit) =
  eps_cert ();
  match Lit.integral (FStar.String.list_of_string eps_lit) 0 false with
  | Some (n, d) -> ()
  | None -> ()
#pop-options

noextract let inv_ok (q:real) (n:pos)
  : Lemma (requires q >=. 0.0R /\ eps_f32 >. 0.0R)
          (ensures q *. (1.0R /. of_int n) +. eps_f32 >. 0.0R /\
                   rsqrt_t (rdiv q (of_int n) +. eps_f32) == Sqrt.rsqrt (q *. (1.0R /. of_int n) +. eps_f32))
  = assert (of_int n >. 0.0R);
    assert (rdiv q (of_int n) == q *. (1.0R /. of_int n))

noextract let consts () : Lemma (of_int 256 == 256.0R /\ 1.0R /. of_int 16 == 0.0625R) = ()


noextract let neg_ok (b:F32.t) (y:real)
  : Lemma (requires b %~ y) (ensures FL.neg b %~ (0.0R -. y))
  = AB.neg_models_pat F32.t

noextract let fmax_ok (a b:F32.t) (x y:real)
  : Lemma (requires a %~ x /\ b %~ y) (ensures FL.fmax a b %~ rmax x y)
  = AB.fmax_models_pat F32.t

noextract let of_int_ok (n:FStar.Int64.t)
  : Lemma (FL.of_int #F32.t n %~ of_int (FStar.Int64.v n))
  = AB.of_int_models_pat F32.t;
    assert (Kuiper.Approximates.Core.int64_can_approximate.approximates n n)

noextract let neg_all (p:ins -> prop) (b:F32.t) (f:ins -> real)
  : Lemma (requires forall (r:ins). p r ==> b %~ f r)
          (ensures forall (r:ins). p r ==> FL.neg b %~ (0.0R -. f r))
  = introduce forall (r:ins). p r ==> FL.neg b %~ (0.0R -. f r)
    with introduce _ ==> _ with neg_ok b (f r)

noextract let neg_forall (b:F32.t)
  : Lemma (forall (y:real). b %~ y ==> FL.neg b %~ (0.0R -. y))
  = introduce forall (y:real). b %~ y ==> FL.neg b %~ (0.0R -. y)
    with introduce _ ==> _ with neg_ok b y

noextract let sub_fix (a b res:F32.t)
  : Lemma (requires forall (xr yr:real). a %~ xr /\ FL.neg b %~ yr ==> res %~ (xr +. yr))
          (ensures forall (x y:real). a %~ x /\ b %~ y ==> res %~ (x -. y))
  = introduce forall (x y:real). a %~ x /\ b %~ y ==> res %~ (x -. y)
    with introduce _ ==> _ with (
      neg_ok b y;
      assert (res %~ (x +. (0.0R -. y)));
      assert (x +. (0.0R -. y) == x -. y))

noextract let zero_ok () : Lemma (Kuiper.Scalars.zero #F32.t #_ %~ 0.0R) =
  Kuiper.Approximates.Scalar.a0 #F32.t #_ #_
noextract let one_ok () : Lemma (Kuiper.Scalars.one #F32.t #_ %~ 1.0R) =
  Kuiper.Approximates.Scalar.a1 #F32.t #_ #_

(*** Lifting op contracts to all admissible input ideals *)
noextract let add_lift (p:ins -> prop) (a b res:F32.t) (fa fb fr:ins -> real)
  : Lemma (requires (forall (xr yr:real). a %~ xr /\ b %~ yr ==> res %~ (xr +. yr)) /\
                    (forall (r:ins). p r ==> a %~ fa r /\ b %~ fb r /\ fr r == fa r +. fb r))
          (ensures forall (r:ins). p r ==> res %~ fr r)
  = ()
noextract let mul_lift (p:ins -> prop) (a b res:F32.t) (fa fb fr:ins -> real)
  : Lemma (requires (forall (xr yr:real). a %~ xr /\ b %~ yr ==> res %~ (xr *. yr)) /\
                    (forall (r:ins). p r ==> a %~ fa r /\ b %~ fb r /\ fr r == fa r *. fb r))
          (ensures forall (r:ins). p r ==> res %~ fr r)
  = ()
noextract let fma_lift (p:ins -> prop) (a b c res:F32.t) (fa fb fc fr:ins -> real)
  : Lemma (requires (forall (xr yr zr:real). a %~ xr /\ b %~ yr /\ c %~ zr ==> res %~ (xr *. yr +. zr)) /\
                    (forall (r:ins). p r ==> a %~ fa r /\ b %~ fb r /\ c %~ fc r /\ fr r == fa r *. fb r +. fc r))
          (ensures forall (r:ins). p r ==> res %~ fr r)
  = ()


(**** Section K1 ****)
#set-options "--fuel 0 --ifuel 0 --z3rlimit 20"

noextract let rmsF (xf:ins -> S.seq real) (wo:int) (r:ins) : S.seq real = rmsnorm1p eps_f32 (xf r) 1024 r.rw wo
noextract let invI (q:real) (n:pos) : real = rsqrt_t (rdiv q (of_int n) +. eps_f32)

noextract let sq_nonneg_all (xf:ins -> S.seq real) (o:int) (n:nat)
  : Lemma (forall (r:ins). sq_sum (xf r) o n >=. 0.0R)
  = introduce forall (r:ins). sq_sum (xf r) o n >=. 0.0R with sq_nonneg xf o r n

noextract let inv_all (p:ins -> prop) (q:ins -> real) (n:pos)
  : Lemma (requires forall (r:ins). p r ==> q r >=. 0.0R)
          (ensures forall (r:ins). p r ==> q r *. (1.0R /. of_int n) +. eps_f32 >. 0.0R /\
                     invI (q r) n == Sqrt.rsqrt (q r *. (1.0R /. of_int n) +. eps_f32))
  = eps_pos ();
    introduce forall (r:ins). p r ==> q r *. (1.0R /. of_int n) +. eps_f32 >. 0.0R /\
                     invI (q r) n == Sqrt.rsqrt (q r *. (1.0R /. of_int n) +. eps_f32)
    with introduce _ ==> _ with inv_ok (q r) n

noextract let inv_chain (p:ins -> prop) (q:ins -> real) (n:I64.t{I64.v n > 0}) (ss c m a res:F32.t)
  : Lemma (requires (forall (r:ins). p r ==> ss %~ q r /\ q r >=. 0.0R) /\
                    (forall (xr:real{xr =!= 0.0R}). FL.of_int #F32.t n %~ xr ==> c %~ (1.0R /. xr)) /\
                    (forall (xr yr:real). ss %~ xr /\ c %~ yr ==> m %~ (xr *. yr)) /\
                    (forall (xr yr:real). m %~ xr /\ FL.of_literal #F32.t eps_lit %~ yr ==> a %~ (xr +. yr)) /\
                    (forall (xr:Sqrt.rpos). a %~ xr ==> res %~ Sqrt.rsqrt xr))
          (ensures forall (r:ins). p r ==> res %~ invI (q r) (I64.v n))
  = eps_cert ();
    CF.literal_approx eps_lit 8796093 (-43);
    eps_pos ();
    of_int_ok n;
    let ni = of_int (I64.v n) in
    assert (ni >. 0.0R);
    assert (c %~ (1.0R /. ni));
    introduce forall (r:ins). p r ==> res %~ invI (q r) (I64.v n)
    with introduce _ ==> _ with (
      inv_ok (q r) (I64.v n);
      assert (m %~ (q r *. (1.0R /. ni)));
      assert (a %~ (q r *. (1.0R /. ni) +. eps_f32));
      assert (res %~ Sqrt.rsqrt (q r *. (1.0R /. ni) +. eps_f32)))

inline_for_extraction noextract
fn sub_ftz (a b : f32)
  preserves gpu
  returns r : F32.t
  ensures pure (forall (x y:real). a %~ x /\ b %~ y ==> r %~ (x -. y))
{
  let res = F32.add_rn_ftz a (FL.neg b);
  sub_fix a b res;
  res
}

inline_for_extraction noextract
fn inv_rms (ss : f32) (n : I64.t{I64.v n > 0}) (p : erased (ins -> prop)) (q : erased (ins -> real))
  preserves gpu
  requires pure (forall (r:ins). reveal p r ==> ss %~ reveal q r /\ reveal q r >=. 0.0R)
  returns res : F32.t
  ensures pure (forall (r:ins). reveal p r ==> res %~ invI (reveal q r) (I64.v n))
{
  let c = F32.rcp_approx_ftz (FL.of_int #f32 n);
  let m = F32.mul_rn_ftz ss c;
  let a = F32.add_rn_ftz m (FL.of_literal #f32 "0.0000009999999974752427078783512115478515625");
  let res = F32.rsqrt_approx_ftz a;
  inv_chain p q n ss c m a res;
  res
}

noextract let ssk (xf:ins -> S.seq real) (o:nat) (k:nat) (r:ins) : real = vsum k (stride (sqg xf 0 r) o 512)
noextract let xat (xf:ins -> S.seq real) (i:int) (r:ins) : real = at (xf r) i

inline_for_extraction noextract
fn ss1024 (x : larray f32 1024) (lanes warps : larray f32 32)
  (p:erased (ins -> prop)) (xf:erased (ins -> S.seq real))
  (#sx #sl #sw : erased (S.seq f32))
  preserves gpu ** x |-> sx
  requires lanes |-> sl ** warps |-> sw ** pure (apxs p sx 1024 xf)
  returns res : F32.t
  ensures exists* sl' sw'. lanes |-> sl' ** warps |-> sw' **
    pure (forall (r:ins). reveal p r ==> res %~ sq_sum (reveal xf r) 0 1024)
{
  pts_to_len x; pts_to_len lanes; pts_to_len warps;
  let mut w = 0sz;
  while (!w <^ 16sz)
    invariant exists* (vw:sz) s1 s2. w |-> vw ** lanes |-> s1 ** warps |-> s2 **
      pure (SZ.v vw <= 16 /\ S.length s1 == 32 /\ S.length s2 == 32 /\ apx p s2 (SZ.v vw) (sswF xf))
    decreases (16 - SZ.v !w)
  {
    let vw = !w;
    let mut l = 0sz;
    while (!l <^ 32sz)
      invariant exists* (vl:sz) s1. l |-> vl ** lanes |-> s1 **
        pure (SZ.v vl <= 32 /\ S.length s1 == 32 /\ apx p s1 (SZ.v vl) (sslF xf (SZ.v vw)))
      decreases (32 - SZ.v !l)
    {
      let vl = !l;
      zero_stride (sqg xf 0) (SZ.v vw * 32 + SZ.v vl) 512;
      zero_ok ();
      let mut s = zero #f32 #_;
      let mut k = 0sz;
      while (!k <^ 2sz)
        invariant exists* (vk:sz) (vs:f32). k |-> vk ** s |-> vs ** pure (SZ.v vk <= 2 /\
          (forall (r:ins). reveal p r ==> vs %~ ssk xf (SZ.v vw * 32 + SZ.v vl) (SZ.v vk) r))
        decreases (2 - SZ.v !k)
      {
        let vk = !k;
        let idx = vw *^ 32sz +^ vl +^ vk *^ 512sz;
        let a = x.(idx);
        let vs = !s;
        let res = F32.fma_rn_ftz a a vs;
        step_stride (sqg xf 0) (SZ.v vw * 32 + SZ.v vl) 512 (SZ.v vk);
        fma_lift p a a vs res (xat xf (SZ.v idx)) (xat xf (SZ.v idx))
          (ssk xf (SZ.v vw * 32 + SZ.v vl) (SZ.v vk)) (ssk xf (SZ.v vw * 32 + SZ.v vl) (SZ.v vk + 1));
        s := res;
        k := vk +^ 1sz;
      };
      let vs = !s;
      lanes.(vl) <- vs;
      l := vl +^ 1sz;
    };
    let t = tree32 lanes p (sslF xf (SZ.v vw));
    ssw_lane_ok xf (SZ.v vw);
    warps.(vw) <- t;
    w := vw +^ 1sz;
  };
  let mut j = 16sz;
  zero_ok ();
  while (!j <^ 32sz)
    invariant exists* (vj:sz) s2. j |-> vj ** warps |-> s2 ** pure (16 <= SZ.v vj /\ SZ.v vj <= 32 /\
      S.length s2 == 32 /\ apx p s2 (SZ.v vj) (sswF xf))
    decreases (32 - SZ.v !j)
  {
    let vj = !j;
    warps.(vj) <- zero #f32 #_;
    j := vj +^ 1sz;
  };
  let res = tree32 warps p (sswF xf);
  ss1024_ok xf;
  res
}

noextract let ssq (xf:ins -> S.seq real) (o:int) (n:nat) (r:ins) : real = sq_sum (xf r) o n
noextract let invF (xf:ins -> S.seq real) (r:ins) : real = invI (sq_sum (xf r) 0 1024) 1024
noextract let wat (wo:int) (r:ins) : real = at r.rw wo
noextract let cst (c:real) (r:ins) : real = c
noextract let mulF (f g:ins -> real) (r:ins) : real = f r *. g r
noextract let addF (f g:ins -> real) (r:ins) : real = f r +. g r
noextract let rmsI (xf:ins -> S.seq real) (wo:int) (i:nat) (r:ins) : real = at (rmsF xf wo r) i

noextract let rms_at (xf:ins -> S.seq real) (wo:int) (i:nat{i < 1024})
  : Lemma (forall (r:ins). rmsI xf wo i r ==
             mulF (mulF (xat xf i) (invF xf)) (addF (cst 1.0R) (wat (wo + i))) r)
  = introduce forall (r:ins). rmsI xf wo i r ==
             mulF (mulF (xat xf i) (invF xf)) (addF (cst 1.0R) (wat (wo + i))) r
    with norm_spec [delta_only [`%rmsnorm1p]; zeta] (rmsnorm1p eps_f32 (xf r) 1024 r.rw wo)

inline_for_extraction noextract
fn rmsnorm_kernel (w : array bf16) (wo : sz) (x out : larray f32 1024) (lanes warps : larray f32 32)
  (p:erased (ins -> prop)) (xf:erased (ins -> S.seq real))
  (#sw : erased (S.seq bf16)) (#sx #so #sl #swp : erased (S.seq f32))
  preserves gpu ** w |-> sw ** x |-> sx
  requires out |-> so ** lanes |-> sl ** warps |-> swp **
    pure (SZ.fits (S.length sw) /\ SZ.v wo + 1024 <= S.length sw /\ wok p sw /\ apxs p sx 1024 xf)
  ensures exists* so' sl' swp'. out |-> so' ** lanes |-> sl' ** warps |-> swp' **
    pure (S.length so' == 1024 /\ apxs p so' 1024 (rmsF xf (SZ.v wo)))
{
  pts_to_len out; pts_to_len x;
  let ss = ss1024 x lanes warps p xf;
  sq_nonneg_all xf 0 1024;
  let scale = inv_rms ss 1024L p (ssq xf 0 1024);
  let mut i = 0sz;
  one_ok ();
  while (!i <^ 1024sz)
    invariant exists* (vi:sz) s. i |-> vi ** out |-> s ** pure (SZ.v vi <= 1024 /\ S.length s == 1024 /\
      apxs p s (SZ.v vi) (rmsF xf (SZ.v wo)))
    decreases (1024 - SZ.v !i)
  {
    let vi = !i;
    let xi = x.(vi);
    let wi = w.(wo +^ vi);
    wcast_all p sw (SZ.v wo + SZ.v vi);
    let cw = C.cast_bf16_to_f32 wi;
    let m1 = F32.mul_rn_ftz xi scale;
    mul_lift p xi scale m1 (xat xf (SZ.v vi)) (invF xf) (mulF (xat xf (SZ.v vi)) (invF xf));
    let a1 = F32.add_rn_ftz (one #f32 #_) cw;
    add_lift p (one #f32 #_) cw a1 (cst 1.0R) (wat (SZ.v wo + SZ.v vi)) (addF (cst 1.0R) (wat (SZ.v wo + SZ.v vi)));
    let t = F32.mul_rn_ftz m1 a1;
    rms_at xf (SZ.v wo) (SZ.v vi);
    mul_lift p m1 a1 t (mulF (xat xf (SZ.v vi)) (invF xf)) (addF (cst 1.0R) (wat (SZ.v wo + SZ.v vi))) (rmsI xf (SZ.v wo) (SZ.v vi));
    rt_cast_all p t (rmsI xf (SZ.v wo) (SZ.v vi));
    out.(vi) <- C.cast_bf16_to_f32 (C.cast_f32_to_bf16 t);
    i := vi +^ 1sz;
  }
}

(*** Dot products *)
noextract let partF (wo:int) (xf:ins -> S.seq real) (kk j:nat) (r:ins) : real = vsum j (shift (dotg wo xf r) kk)
noextract let accF (wo:int) (xf:ins -> S.seq real) (l c:nat) (r:ins) : real = vsum c (chunk (dotg wo xf r) l)
noextract let contF (wo:int) (xf:ins -> S.seq real) (l c kk j:nat) (r:ins) : real =
  accF wo xf l c r +. partF wo xf kk j r
noextract let dotF (wo:int) (xf:ins -> S.seq real) (n:nat) (r:ins) : real = dot r.rw wo (xf r) n

inline_for_extraction noextract
fn grouped_dot (w : array bf16) (wo : sz) (n : sz{SZ.v n % 256 == 0}) (x : larray f32 (SZ.v n))
  (lanes : larray f32 32) (p:erased (ins -> prop)) (xf:erased (ins -> S.seq real))
  (#sw : erased (S.seq bf16)) (#sx #sl : erased (S.seq f32))
  preserves gpu ** w |-> sw ** x |-> sx
  requires lanes |-> sl ** pure (SZ.fits (S.length sw) /\ SZ.v wo + SZ.v n <= S.length sw /\
    wok p sw /\ apxs p sx (SZ.v n) xf)
  returns res : F32.t
  ensures exists* sl'. lanes |-> sl' ** pure (forall (r:ins). reveal p r ==> res %~ dotF (SZ.v wo) xf (SZ.v n) r)
{
  pts_to_len lanes; pts_to_len x;
  let nc = n /^ 256sz;
  let mut l = 0sz;
  while (!l <^ 32sz)
    invariant exists* (vl:sz) s1. l |-> vl ** lanes |-> s1 ** pure (SZ.v vl <= 32 /\ S.length s1 == 32 /\
      apx p s1 (SZ.v vl) (laneF (SZ.v wo) xf (SZ.v nc)))
    decreases (32 - SZ.v !l)
  {
    let vl = !l;
    zero_ok ();
    zero_chunk (dotg (SZ.v wo) xf) (SZ.v vl);
    let mut s = zero #f32 #_;
    let mut c = 0sz;
    while (!c <^ nc)
      invariant exists* (vc:sz) (vs:f32). c |-> vc ** s |-> vs ** pure (SZ.v vc <= SZ.v nc /\
        (forall (r:ins). reveal p r ==> vs %~ accF (SZ.v wo) xf (SZ.v vl) (SZ.v vc) r))
      decreases (SZ.v nc - SZ.v !c)
    {
      let vc = !c;
      let kk = vc *^ 256sz +^ vl *^ 8sz;
      zero_shift (dotg (SZ.v wo) xf) (SZ.v kk);
      let mut part = zero #f32 #_;
      let mut j = 0sz;
      while (!j <^ 8sz)
        invariant exists* (vj:sz) (vp:f32). j |-> vj ** part |-> vp ** pure (SZ.v vj <= 8 /\
          (forall (r:ins). reveal p r ==> vp %~ partF (SZ.v wo) xf (SZ.v kk) (SZ.v vj) r))
        decreases (8 - SZ.v !j)
      {
        let vj = !j;
        let a = w.(wo +^ kk +^ vj);
        let b = x.(kk +^ vj);
        let vp = !part;
        wcast_all p sw (SZ.v wo + SZ.v kk + SZ.v vj);
        let ca = C.cast_bf16_to_f32 a;
        let np = F32.fma_rn_ftz ca b vp;
        step_shift (dotg (SZ.v wo) xf) (SZ.v kk) (SZ.v vj);
        fma_lift p ca b vp np (wat (SZ.v wo + SZ.v kk + SZ.v vj)) (xat xf (SZ.v kk + SZ.v vj))
          (partF (SZ.v wo) xf (SZ.v kk) (SZ.v vj)) (partF (SZ.v wo) xf (SZ.v kk) (SZ.v vj + 1));
        part := np;
        j := vj +^ 1sz;
      };
      let vs = !s;
      let vp = !part;
      let ns = F32.add_rn_ftz vs vp;
      step_chunk (dotg (SZ.v wo) xf) (SZ.v vl) (SZ.v vc);
      add_lift p vs vp ns (accF (SZ.v wo) xf (SZ.v vl) (SZ.v vc)) (partF (SZ.v wo) xf (SZ.v kk) 8)
        (accF (SZ.v wo) xf (SZ.v vl) (SZ.v vc + 1));
      s := ns;
      c := vc +^ 1sz;
    };
    let vs = !s;
    lanes.(vl) <- vs;
    l := vl +^ 1sz;
  };
  let res = tree32 lanes p (laneF (SZ.v wo) xf (SZ.v nc));
  gdot_ok (SZ.v wo) xf (SZ.v nc);
  assert (pure (256 * SZ.v nc == SZ.v n));
  res
}

inline_for_extraction noextract
fn continuous_dot (w : array bf16) (wo : sz) (n : sz{SZ.v n % 256 == 0}) (x : larray f32 (SZ.v n))
  (lanes : larray f32 32) (p:erased (ins -> prop)) (xf:erased (ins -> S.seq real))
  (#sw : erased (S.seq bf16)) (#sx #sl : erased (S.seq f32))
  preserves gpu ** w |-> sw ** x |-> sx
  requires lanes |-> sl ** pure (SZ.fits (S.length sw) /\ SZ.v wo + SZ.v n <= S.length sw /\
    wok p sw /\ apxs p sx (SZ.v n) xf)
  returns res : F32.t
  ensures exists* sl'. lanes |-> sl' ** pure (forall (r:ins). reveal p r ==> res %~ dotF (SZ.v wo) xf (SZ.v n) r)
{
  pts_to_len lanes; pts_to_len x;
  let nc = n /^ 256sz;
  let mut l = 0sz;
  while (!l <^ 32sz)
    invariant exists* (vl:sz) s1. l |-> vl ** lanes |-> s1 ** pure (SZ.v vl <= 32 /\ S.length s1 == 32 /\
      apx p s1 (SZ.v vl) (laneF (SZ.v wo) xf (SZ.v nc)))
    decreases (32 - SZ.v !l)
  {
    let vl = !l;
    zero_ok ();
    zero_chunk (dotg (SZ.v wo) xf) (SZ.v vl);
    let mut s = zero #f32 #_;
    let mut c = 0sz;
    while (!c <^ nc)
      invariant exists* (vc:sz) (vs:f32). c |-> vc ** s |-> vs ** pure (SZ.v vc <= SZ.v nc /\
        (forall (r:ins). reveal p r ==> vs %~ accF (SZ.v wo) xf (SZ.v vl) (SZ.v vc) r))
      decreases (SZ.v nc - SZ.v !c)
    {
      let vc = !c;
      let kk = vc *^ 256sz +^ vl *^ 8sz;
      zero_shift (dotg (SZ.v wo) xf) (SZ.v kk);
      let mut j = 0sz;
      while (!j <^ 8sz)
        invariant exists* (vj:sz) (vs:f32). j |-> vj ** s |-> vs ** pure (SZ.v vj <= 8 /\
          (forall (r:ins). reveal p r ==> vs %~ contF (SZ.v wo) xf (SZ.v vl) (SZ.v vc) (SZ.v kk) (SZ.v vj) r))
        decreases (8 - SZ.v !j)
      {
        let vj = !j;
        let a = w.(wo +^ kk +^ vj);
        let b = x.(kk +^ vj);
        let vs = !s;
        wcast_all p sw (SZ.v wo + SZ.v kk + SZ.v vj);
        let ca = C.cast_bf16_to_f32 a;
        let ns = F32.fma_rn_ftz ca b vs;
        step_shift (dotg (SZ.v wo) xf) (SZ.v kk) (SZ.v vj);
        fma_lift p ca b vs ns (wat (SZ.v wo + SZ.v kk + SZ.v vj)) (xat xf (SZ.v kk + SZ.v vj))
          (contF (SZ.v wo) xf (SZ.v vl) (SZ.v vc) (SZ.v kk) (SZ.v vj))
          (contF (SZ.v wo) xf (SZ.v vl) (SZ.v vc) (SZ.v kk) (SZ.v vj + 1));
        s := ns;
        j := vj +^ 1sz;
      };
      step_chunk (dotg (SZ.v wo) xf) (SZ.v vl) (SZ.v vc);
      c := vc +^ 1sz;
    };
    let vs = !s;
    lanes.(vl) <- vs;
    l := vl +^ 1sz;
  };
  let res = tree32 lanes p (laneF (SZ.v wo) xf (SZ.v nc));
  gdot_ok (SZ.v wo) xf (SZ.v nc);
  assert (pure (256 * SZ.v nc == SZ.v n));
  res
}

noextract let mvF (wo:int) (rows cols:nat) (xf:ins -> S.seq real) (r:ins) : S.seq real = matvec r.rw wo rows cols (xf r)
noextract let resF (hf xf:ins -> S.seq real) (wo:int) (n:nat) (r:ins) : S.seq real =
  vadd (hf r) (matvec r.rw wo 1024 n (xf r)) 1024

noextract let mv_at (wo:int) (rows cols:nat) (xf:ins -> S.seq real) (i:nat{i < rows})
  : Lemma (forall (r:ins). at (mvF wo rows cols xf r) i == dotF (wo + i * cols) xf cols r)
  = introduce forall (r:ins). at (mvF wo rows cols xf r) i == dotF (wo + i * cols) xf cols r
    with norm_spec [delta_only [`%matvec]] (matvec r.rw wo rows cols (xf r))

noextract let res_at (hf xf:ins -> S.seq real) (wo:int) (n:nat) (i:nat{i < 1024})
  : Lemma (forall (r:ins). at (resF hf xf wo n r) i == dotF (wo + i * n) xf n r +. xat hf i r)
  = introduce forall (r:ins). at (resF hf xf wo n r) i == dotF (wo + i * n) xf n r +. xat hf i r
    with (norm_spec [delta_only [`%vadd]] (vadd (hf r) (matvec r.rw wo 1024 n (xf r)) 1024);
          norm_spec [delta_only [`%matvec]] (matvec r.rw wo 1024 n (xf r)))

inline_for_extraction noextract
fn matvec_kernel (w : array bf16) (wo : sz) (rows : sz) (x : larray f32 1024) (out : larray f32 (SZ.v rows))
  (lanes : larray f32 32) (p:erased (ins -> prop)) (xf:erased (ins -> S.seq real))
  (#sw : erased (S.seq bf16)) (#sx #so #sl : erased (S.seq f32))
  preserves gpu ** w |-> sw ** x |-> sx
  requires out |-> so ** lanes |-> sl ** pure (SZ.fits (S.length sw) /\ SZ.v wo + SZ.v rows * 1024 <= S.length sw /\
    wok p sw /\ apxs p sx 1024 xf)
  ensures exists* so' sl'. out |-> so' ** lanes |-> sl' **
    pure (S.length so' == SZ.v rows /\ apxs p so' (SZ.v rows) (mvF (SZ.v wo) (SZ.v rows) 1024 xf))
{
  pts_to_len out; pts_to_len x;
  let mut i = 0sz;
  while (!i <^ rows)
    invariant exists* (vi:sz) s sl1. i |-> vi ** out |-> s ** lanes |-> sl1 **
      pure (SZ.v vi <= SZ.v rows /\ S.length s == SZ.v rows /\ apxs p s (SZ.v vi) (mvF (SZ.v wo) (SZ.v rows) 1024 xf))
    decreases (SZ.v rows - SZ.v !i)
  {
    let vi = !i;
    let d = grouped_dot w (wo +^ vi *^ 1024sz) 1024sz x lanes p xf;
    mv_at (SZ.v wo) (SZ.v rows) 1024 xf (SZ.v vi);
    out.(vi) <- d;
    i := vi +^ 1sz;
  }
}

inline_for_extraction noextract
fn residual_kernel (w : array bf16) (wo : sz) (n : sz{SZ.v n % 256 == 0}) (x : larray f32 (SZ.v n))
  (res out : larray f32 1024)
  (lanes : larray f32 32) (p:erased (ins -> prop)) (xf hf:erased (ins -> S.seq real))
  (#sw : erased (S.seq bf16)) (#sx #sr #so #sl : erased (S.seq f32))
  preserves gpu ** w |-> sw ** x |-> sx ** res |-> sr
  requires out |-> so ** lanes |-> sl ** pure (SZ.fits (S.length sw) /\ SZ.v wo + 1024 * SZ.v n <= S.length sw /\
    wok p sw /\ apxs p sx (SZ.v n) xf /\ apxs p sr 1024 hf)
  ensures exists* so' sl'. out |-> so' ** lanes |-> sl' **
    pure (S.length so' == 1024 /\ apxs p so' 1024 (resF hf xf (SZ.v wo) (SZ.v n)))
{
  pts_to_len out; pts_to_len res;
  let mut i = 0sz;
  while (!i <^ 1024sz)
    invariant exists* (vi:sz) s sl1. i |-> vi ** out |-> s ** lanes |-> sl1 **
      pure (SZ.v vi <= 1024 /\ S.length s == 1024 /\ apxs p s (SZ.v vi) (resF hf xf (SZ.v wo) (SZ.v n)))
    decreases (1024 - SZ.v !i)
  {
    let vi = !i;
    let d = continuous_dot w (wo +^ vi *^ n) n x lanes p xf;
    let r = res.(vi);
    let t = F32.add_rn_ftz d r;
    res_at hf xf (SZ.v wo) (SZ.v n) (SZ.v vi);
    add_lift p d r t (dotF (SZ.v wo + SZ.v vi * SZ.v n) xf (SZ.v n)) (xat hf (SZ.v vi))
      (fun r -> at (resF hf xf (SZ.v wo) (SZ.v n) r) (SZ.v vi));
    rt_cast_all p t (fun r -> at (resF hf xf (SZ.v wo) (SZ.v n) r) (SZ.v vi));
    out.(vi) <- C.cast_bf16_to_f32 (C.cast_f32_to_bf16 t);
    i := vi +^ 1sz;
  }
}


(**** Section K2 ****)
#set-options "--fuel 0 --ifuel 0 --z3rlimit 20"

noextract let hnF (rawF:ins -> S.seq real) (o wo:int) (r:ins) : S.seq real = head_norm eps_f32 (rawF r) o r.rw wo
noextract let hrF (rawF:ins -> S.seq real) (o wo:int) (r:ins) : S.seq real = rope (hnF rawF o wo r) r.rc r.rs
noextract let hnI (rawF:ins -> S.seq real) (o wo:int) (i:nat) (r:ins) : real = at (hnF rawF o wo r) i
noextract let hinvF (rawF:ins -> S.seq real) (o:int) (r:ins) : real = invI (sq_sum (rawF r) o 256) 256
noextract let csI (i:nat) (r:ins) : real = at r.rc i
noextract let snI (i:nat) (r:ins) : real = at r.rs i
noextract let csok (p:ins -> prop) (sc ss:S.seq F32.t) : prop = forall (r:ins). p r ==> sc %~ r.rc /\ ss %~ r.rs

noextract let hn_at (rawF:ins -> S.seq real) (o wo:int) (i:nat{i < 256})
  : Lemma (forall (r:ins). hnI rawF o wo i r ==
             mulF (mulF (xat rawF (o + i)) (hinvF rawF o)) (addF (cst 1.0R) (wat (wo + i))) r)
  = consts ();
    introduce forall (r:ins). hnI rawF o wo i r ==
             mulF (mulF (xat rawF (o + i)) (hinvF rawF o)) (addF (cst 1.0R) (wat (wo + i))) r
    with norm_spec [delta_only [`%head_norm]; zeta] (head_norm eps_f32 (rawF r) o r.rw wo)

noextract let ropeA (rawF:ins -> S.seq real) (o wo:int) (i:nat) (r:ins) : real =
  hnI rawF o wo i r *. csI i r +. (0.0R -. hnI rawF o wo (i + 32) r *. snI i r)
noextract let ropeB (rawF:ins -> S.seq real) (o wo:int) (i:nat{i >= 32}) (r:ins) : real =
  hnI rawF o wo (i - 32) r *. snI (i - 32) r +. hnI rawF o wo i r *. csI (i - 32) r
noextract let hrI (rawF:ins -> S.seq real) (o wo:int) (i:nat) (r:ins) : real = at (hrF rawF o wo r) i

noextract let rope_at (rawF:ins -> S.seq real) (o wo:int) (i:nat{i < 256})
  : Lemma (forall (r:ins). hrI rawF o wo i r ==
      (if i < 32 then ropeA rawF o wo i r else if i < 64 then ropeB rawF o wo i r else hnI rawF o wo i r))
  = introduce forall (r:ins). hrI rawF o wo i r ==
      (if i < 32 then ropeA rawF o wo i r else if i < 64 then ropeB rawF o wo i r else hnI rawF o wo i r)
    with norm_spec [delta_only [`%rope]] (rope (hnF rawF o wo r) r.rc r.rs)

noextract let mneg (f:ins -> real) (r:ins) : real = 0.0R -. f r

inline_for_extraction noextract
fn head_value (raw : array f32) (o : sz) (rstd : f32) (w : array bf16) (wo : sz) (i : sz{SZ.v i < 256})
  (p:erased (ins -> prop)) (rawF:erased (ins -> S.seq real))
  (#sr : erased (S.seq f32)) (#sw : erased (S.seq bf16))
  preserves gpu ** raw |-> sr ** w |-> sw
  requires pure (SZ.fits (S.length sr) /\ SZ.fits (S.length sw) /\ SZ.v o + 256 <= S.length sr /\ SZ.v wo + 256 <= S.length sw /\
    wok p sw /\ apxs p sr (SZ.v o + 256) rawF /\
    (forall (r:ins). reveal p r ==> rstd %~ hinvF rawF (SZ.v o) r))
  returns res : F32.t
  ensures pure (forall (r:ins). reveal p r ==> res %~ hnI rawF (SZ.v o) (SZ.v wo) (SZ.v i) r)
{
  let a = raw.(o +^ i);
  let b = w.(wo +^ i);
  wcast_all p sw (SZ.v wo + SZ.v i);
  one_ok ();
  let cb = C.cast_bf16_to_f32 b;
  let m1 = F32.mul_rn_ftz a rstd;
  mul_lift p a rstd m1 (xat rawF (SZ.v o + SZ.v i)) (hinvF rawF (SZ.v o)) (mulF (xat rawF (SZ.v o + SZ.v i)) (hinvF rawF (SZ.v o)));
  let a1 = F32.add_rn_ftz (one #f32 #_) cb;
  add_lift p (one #f32 #_) cb a1 (cst 1.0R) (wat (SZ.v wo + SZ.v i)) (addF (cst 1.0R) (wat (SZ.v wo + SZ.v i)));
  let t = F32.mul_rn_ftz m1 a1;
  hn_at rawF (SZ.v o) (SZ.v wo) (SZ.v i);
  mul_lift p m1 a1 t (mulF (xat rawF (SZ.v o + SZ.v i)) (hinvF rawF (SZ.v o))) (addF (cst 1.0R) (wat (SZ.v wo + SZ.v i)))
    (hnI rawF (SZ.v o) (SZ.v wo) (SZ.v i));
  t
}

inline_for_extraction noextract
fn rope_value (raw : array f32) (o : sz) (rstd : f32) (w : array bf16) (wo : sz) (cs sn : larray f32 32)
  (vi : sz{SZ.v vi < 256})
  (p:erased (ins -> prop)) (rawF:erased (ins -> S.seq real))
  (#sr : erased (S.seq f32)) (#sw : erased (S.seq bf16)) (#sc #ss : erased (S.seq f32))
  preserves gpu ** raw |-> sr ** w |-> sw ** cs |-> sc ** sn |-> ss
  requires pure (SZ.fits (S.length sr) /\ SZ.fits (S.length sw) /\ SZ.v o + 256 <= S.length sr /\ SZ.v wo + 256 <= S.length sw /\
    wok p sw /\ apxs p sr (SZ.v o + 256) rawF /\ csok p sc ss /\
    (forall (r:ins). reveal p r ==> rstd %~ hinvF rawF (SZ.v o) r))
  returns res : F32.t
  ensures pure (forall (r:ins). reveal p r ==> res %~ hrI rawF (SZ.v o) (SZ.v wo) (SZ.v vi) r)
{
  pts_to_len cs; pts_to_len sn;
  rope_at rawF (SZ.v o) (SZ.v wo) (SZ.v vi);
  if (vi <^ 32sz) {
    let u0 = head_value raw o rstd w wo vi p rawF;
    let u1 = head_value raw o rstd w wo (vi +^ 32sz) p rawF;
    let c = cs.(vi);
    let sv = sn.(vi);
    let m = F32.mul_rn_ftz u1 sv;
    mul_lift p u1 sv m (hnI rawF (SZ.v o) (SZ.v wo) (SZ.v vi + 32)) (snI (SZ.v vi))
      (mulF (hnI rawF (SZ.v o) (SZ.v wo) (SZ.v vi + 32)) (snI (SZ.v vi)));
    neg_all p m (mulF (hnI rawF (SZ.v o) (SZ.v wo) (SZ.v vi + 32)) (snI (SZ.v vi)));
    let nm = FL.neg m;
    let r0 = F32.fma_rn_ftz u0 c nm;
    fma_lift p u0 c nm r0 (hnI rawF (SZ.v o) (SZ.v wo) (SZ.v vi)) (csI (SZ.v vi))
      (mneg (mulF (hnI rawF (SZ.v o) (SZ.v wo) (SZ.v vi + 32)) (snI (SZ.v vi))))
      (hrI rawF (SZ.v o) (SZ.v wo) (SZ.v vi));
    r0
  } else if (vi <^ 64sz) {
    let u0 = head_value raw o rstd w wo (vi -^ 32sz) p rawF;
    let u1 = head_value raw o rstd w wo vi p rawF;
    let c = cs.(vi -^ 32sz);
    let sv = sn.(vi -^ 32sz);
    let m = F32.mul_rn_ftz u1 c;
    mul_lift p u1 c m (hnI rawF (SZ.v o) (SZ.v wo) (SZ.v vi)) (csI (SZ.v vi - 32))
      (mulF (hnI rawF (SZ.v o) (SZ.v wo) (SZ.v vi)) (csI (SZ.v vi - 32)));
    let r1 = F32.fma_rn_ftz u0 sv m;
    fma_lift p u0 sv m r1 (hnI rawF (SZ.v o) (SZ.v wo) (SZ.v vi - 32)) (snI (SZ.v vi - 32))
      (mulF (hnI rawF (SZ.v o) (SZ.v wo) (SZ.v vi)) (csI (SZ.v vi - 32)))
      (hrI rawF (SZ.v o) (SZ.v wo) (SZ.v vi));
    r1
  } else {
    let r2 = head_value raw o rstd w wo vi p rawF;
    r2
  }
}

noextract let hsk (rawF:ins -> S.seq real) (o:int) (l k:nat) (r:ins) : real = vsum k (stride (sqg rawF o r) l 32)

noextract let hsq (rawF:ins -> S.seq real) (st wq:int) (n:nat) (r:ins) : S.seq real =
  S.init n (fun (c:nat{c < n}) -> hrI rawF ((c / 256) * st) wq (c % 256) r)

noextract let divmod256 (h j:nat)
  : Lemma (requires j < 256) (ensures (h * 256 + j) / 256 == h /\ (h * 256 + j) % 256 == j)
  = ML.lemma_div_plus j h 256; ML.small_div j 256; ML.lemma_mod_plus j h 256; ML.small_mod j 256

noextract let hsq_ext (p:ins -> prop) (s s0:S.seq F32.t) (h:nat) (oo o:nat) (st wq:int) (rawF:ins -> S.seq real) (n:nat)
  : Lemma (requires (h + 1) * 256 <= n /\ n <= S.length s /\ oo == h * 256 /\ o == h * st /\
                    apxs p s0 (h * 256) (hsq rawF st wq n) /\ S.length s == S.length s0 /\
                    (forall (j:nat). j < S.length s0 /\ (j < oo \/ oo + 256 <= j) ==> S.index s j == S.index s0 j) /\
                    (forall (r:ins) (j:nat). p r /\ j < 256 ==> S.index s (oo + j) %~ hrI rawF o wq j r))
          (ensures apxs p s ((h + 1) * 256) (hsq rawF st wq n))
  = introduce forall (r:ins) (c:nat). p r /\ c < (h + 1) * 256 ==> S.index s c %~ at (hsq rawF st wq n r) c
    with introduce _ ==> _ with (
      if c < h * 256 then ()
      else (let j = c - h * 256 in divmod256 h j; assert (c == h * 256 + j)))

inline_for_extraction noextract
fn head_kernel (raw : array f32) (o : sz) (w : array bf16) (wo : sz) (cs sn : larray f32 32)
  (out : array f32) (oo : sz) (lanes : larray f32 32)
  (p:erased (ins -> prop)) (rawF:erased (ins -> S.seq real))
  (#sr : erased (S.seq f32)) (#sw : erased (S.seq bf16)) (#sc #ss #so #sl : erased (S.seq f32))
  preserves gpu ** raw |-> sr ** w |-> sw ** cs |-> sc ** sn |-> ss
  requires out |-> so ** lanes |-> sl **
    pure (SZ.fits (S.length sr) /\ SZ.fits (S.length sw) /\ SZ.v o + 256 <= S.length sr /\ SZ.v wo + 256 <= S.length sw /\
      SZ.v oo + 256 <= S.length so /\ wok p sw /\ apxs p sr (SZ.v o + 256) rawF /\ csok p sc ss)
  ensures exists* so' sl'. out |-> so' ** lanes |-> sl' ** pure (S.length so' == S.length so /\ S.length sl' == 32 /\
    SZ.v oo + 256 <= S.length so /\
    (forall (j:nat). j < S.length so /\ (j < SZ.v oo \/ SZ.v oo + 256 <= j) ==> S.index so' j == S.index so j) /\
    (forall (r:ins) (j:nat). reveal p r /\ j < 256 ==> S.index so' (SZ.v oo + j) %~ hrI rawF (SZ.v o) (SZ.v wo) j r))
{
  pts_to_len lanes; pts_to_len raw; pts_to_len out; pts_to_len cs; pts_to_len sn;
  let mut l = 0sz;
  while (!l <^ 32sz)
    invariant exists* (vl:sz) s1. l |-> vl ** lanes |-> s1 ** pure (SZ.v vl <= 32 /\ S.length s1 == 32 /\
      apx p s1 (SZ.v vl) (hsF rawF (SZ.v o)))
    decreases (32 - SZ.v !l)
  {
    let vl = !l;
    zero_ok ();
    zero_stride (sqg rawF (SZ.v o)) (SZ.v vl) 32;
    let mut s = zero #f32 #_;
    let mut k = 0sz;
    while (!k <^ 8sz)
      invariant exists* (vk:sz) (vs:f32). k |-> vk ** s |-> vs ** pure (SZ.v vk <= 8 /\
        (forall (r:ins). reveal p r ==> vs %~ hsk rawF (SZ.v o) (SZ.v vl) (SZ.v vk) r))
      decreases (8 - SZ.v !k)
    {
      let vk = !k;
      let a = raw.(o +^ vl +^ vk *^ 32sz);
      let vs = !s;
      let ns = F32.fma_rn_ftz a a vs;
      step_stride (sqg rawF (SZ.v o)) (SZ.v vl) 32 (SZ.v vk);
      fma_lift p a a vs ns (xat rawF (SZ.v o + SZ.v vl + SZ.v vk * 32)) (xat rawF (SZ.v o + SZ.v vl + SZ.v vk * 32))
        (hsk rawF (SZ.v o) (SZ.v vl) (SZ.v vk)) (hsk rawF (SZ.v o) (SZ.v vl) (SZ.v vk + 1));
      s := ns;
      k := vk +^ 1sz;
    };
    let vs = !s;
    lanes.(vl) <- vs;
    l := vl +^ 1sz;
  };
  let ss = tree32 lanes p (hsF rawF (SZ.v o));
  ss256_ok rawF (SZ.v o);
  sq_nonneg_all rawF (SZ.v o) 256;
  let rstd = inv_rms ss 256L p (ssq rawF (SZ.v o) 256);
  let mut i = 0sz;
  while (!i <^ 256sz)
    invariant exists* (vi:sz) s. i |-> vi ** out |-> s ** pure (SZ.v vi <= 256 /\ S.length s == S.length so /\
      SZ.v oo + 256 <= S.length so /\
      (forall (j:nat). j < S.length so /\ (j < SZ.v oo \/ SZ.v oo + SZ.v vi <= j) ==> S.index s j == S.index so j) /\
      (forall (r:ins) (j:nat). reveal p r /\ j < SZ.v vi ==> S.index s (SZ.v oo + j) %~ hrI rawF (SZ.v o) (SZ.v wo) j r))
    decreases (256 - SZ.v !i)
  {
    let vi = !i;
    let r = rope_value raw o rstd w wo cs sn vi p rawF;
    out.(oo +^ vi) <- r;
    i := vi +^ 1sz;
  }
}

(* All 8 query heads and 2 key heads *)
inline_for_extraction noextract
fn heads_kernel (w : array bf16) (mix : sz) (proj : larray f32 4096) (zk : larray f32 512)
  (cs sn : larray f32 32) (qn : larray f32 2048) (kn : larray f32 512) (lanes : larray f32 32)
  (p:erased (ins -> prop)) (projF zkF:erased (ins -> S.seq real))
  (#sw : erased (S.seq bf16)) (#sp #szk #sc #ss #sq #sk #sl : erased (S.seq f32))
  preserves gpu ** w |-> sw ** proj |-> sp ** zk |-> szk ** cs |-> sc ** sn |-> ss
  requires qn |-> sq ** kn |-> sk ** lanes |-> sl ** pure (SZ.fits (S.length sw) /\ SZ.v mix + 5243392 <= S.length sw /\
    wok p sw /\ apxs p sp 4096 projF /\ apxs p szk 512 zkF /\ csok p sc ss)
  ensures exists* sq' sk' sl'. qn |-> sq' ** kn |-> sk' ** lanes |-> sl' ** pure (
    S.length sq' == 2048 /\ S.length sk' == 512 /\
    apxs p sq' 2048 (hsq projF 512 (SZ.v mix + 5242880) 2048) /\
    apxs p sk' 512 (hsq zkF 256 (SZ.v mix + 5243136) 512))
{
  pts_to_len qn; pts_to_len zk; pts_to_len proj; pts_to_len kn; pts_to_len lanes;
  let mut h = 0sz;
  while (!h <^ 2sz)
    invariant exists* (vh:sz) s sl1. h |-> vh ** kn |-> s ** lanes |-> sl1 ** pure (SZ.v vh <= 2 /\ S.length s == 512 /\
      S.length sl1 == 32 /\ apxs p s (SZ.v vh * 256) (hsq zkF 256 (SZ.v mix + 5243136) 512))
    decreases (2 - SZ.v !h)
  {
    let vh = !h;
    with s0. assert (kn |-> s0);
    let off = vh *^ 256sz;
    let wk = mix +^ (SZ.uint32_to_sizet 5243136ul);
    assert (pure (SZ.v wk == SZ.v mix + 5243136));
    head_kernel zk off w wk cs sn kn off lanes p zkF;
    with s1. assert (kn |-> s1);
    hsq_ext p s1 s0 (SZ.v vh) (SZ.v off) (SZ.v off) 256 (SZ.v wk) zkF 512;
    pts_to_len kn; pts_to_len lanes;
    h := vh +^ 1sz;
  };
  let mut q = 0sz;
  while (!q <^ 8sz)
    invariant exists* (vq:sz) s sl1. q |-> vq ** qn |-> s ** lanes |-> sl1 ** pure (SZ.v vq <= 8 /\ S.length s == 2048 /\
      S.length sl1 == 32 /\ apxs p s (SZ.v vq * 256) (hsq projF 512 (SZ.v mix + 5242880) 2048))
    decreases (8 - SZ.v !q)
  {
    let vq = !q;
    with s0. assert (qn |-> s0);
    let ro = vq *^ 512sz;
    let off = vq *^ 256sz;
    let wq = mix +^ (SZ.uint32_to_sizet 5242880ul);
    assert (pure (SZ.v wq == SZ.v mix + 5242880));
    head_kernel proj ro w wq cs sn qn off lanes p projF;
    with s1. assert (qn |-> s1);
    hsq_ext p s1 s0 (SZ.v vq) (SZ.v off) (SZ.v ro) 512 (SZ.v wq) projF 2048;
    pts_to_len qn; pts_to_len lanes;
    q := vq +^ 1sz;
  }
}


(**** Section K3 ****)
#set-options "--fuel 0 --ifuel 0 --z3rlimit 20"

noextract let apxb (p:ins -> prop) (s:S.seq BF.t) (n:nat) (f:ins -> S.seq real) : prop =
  n <= S.length s /\ (forall (r:ins) (j:nat). p r /\ j < n ==> S.index s j %~ at (f r) j)

noextract let kv_ci (fa pos c:nat) : nat = fa * 1048576 + (c / 256) * 524288 + pos * 256 + c % 256

#push-options "--z3rlimit 60"
noextract let kv_ci_ok (fa:nat{fa < 6}) (pos:nat{pos < 2048}) (c:nat{c < 512})
  : Lemma (kv_ci fa pos c < 6291456 /\ opt_kv_hit fa pos (kv_ci fa pos c) /\ opt_kv_comp (kv_ci fa pos c) == c)
  = let h = c / 256 in
    let e = c % 256 in
    let y = pos * 256 + e in
    let x = h * 524288 + y in
    let ci = kv_ci fa pos c in
    assert (h <= 1);
    assert (y < 524288);
    assert (x < 1048576);
    assert (ci == x + fa * 1048576);
    ML.lemma_div_plus x fa 1048576; ML.small_div x 1048576;
    ML.lemma_mod_plus x fa 1048576; ML.small_mod x 1048576;
    ML.lemma_div_plus y h 524288; ML.small_div y 524288;
    ML.lemma_mod_plus y h 524288; ML.small_mod y 524288;
    assert (ci == x + (2 * fa) * 524288);
    ML.lemma_mod_plus x (2 * fa) 524288;
    assert (ci % 524288 == y);
    ML.lemma_div_plus e pos 256; ML.small_div e 256;
    ML.lemma_mod_plus e pos 256; ML.small_mod e 256;
    assert (ci == (fa * 4096 + h * 2048 + pos) * 256 + e);
    ML.lemma_mod_plus e (fa * 4096 + h * 2048 + pos) 256;
    assert (ci % 256 == e);
    ML.euclidean_division_definition c 256

noextract let kv_inj (fa:nat{fa < 6}) (pos:nat{pos < 2048}) (i:nat)
  : Lemma (requires i < 6291456 /\ opt_kv_hit fa pos i)
          (ensures opt_kv_comp i < 512 /\ i == kv_ci fa pos (opt_kv_comp i))
  = let x = i % 1048576 in
    let h = x / 524288 in
    ML.euclidean_division_definition i 1048576;
    ML.euclidean_division_definition x 524288;
    ML.modulo_modulo_lemma i 524288 2;
    assert (x % 524288 == i % 524288);
    let y = i % 524288 in
    ML.euclidean_division_definition y 256;
    ML.modulo_modulo_lemma i 256 2048;
    assert (y % 256 == i % 256);
    assert (h <= 1);
    let c = opt_kv_comp i in
    assert (c == h * 256 + i % 256);
    ML.lemma_div_plus (i % 256) h 256; ML.small_div (i % 256) 256;
    ML.lemma_mod_plus (i % 256) h 256; ML.small_mod (i % 256) 256
#pop-options

noextract let kvw_at (fa pos:int) (rk kn:S.seq real) (i:nat{i < 6291456})
  : Lemma (at (opt_kv_write fa pos rk kn) i == (if opt_kv_hit fa pos i then at kn (opt_kv_comp i) else at rk i))
  = norm_spec [delta_only [`%opt_kv_write]] (opt_kv_write fa pos rk kn)

noextract let kv_read (fa:nat{fa < 6}) (pos:nat{pos < 2048}) (rk kn:S.seq real) (p:nat{p <= pos}) (j:nat{j < 512})
  : Lemma (opt_kv_index fa p j == kv_ci fa p j /\ kv_ci fa p j < 6291456 /\
           at (opt_kv_write fa pos rk kn) (kv_ci fa p j) == hist_or (opt_hk fa rk) kn pos p j)
  = kv_ci_ok fa p j;
    kvw_at fa pos rk kn (kv_ci fa p j);
    if p = pos then ()
    else (
      if opt_kv_hit fa pos (kv_ci fa p j) then (kv_inj fa pos (kv_ci fa p j); kv_ci_ok fa pos (opt_kv_comp (kv_ci fa p j)))
      else ())

noextract let c2b_all (p:ins -> prop) (t:F32.t) (f:ins -> real)
  : Lemma (requires forall (r:ins). p r ==> t %~ f r)
          (ensures forall (r:ins). p r ==> C.cast_f32_to_bf16 t %~ f r)
  = introduce forall (r:ins). p r ==> C.cast_f32_to_bf16 t %~ f r
    with introduce _ ==> _ with C.cast_f32_to_bf16_ok t (f r)

noextract let kvI (fa pos:int) (sel:ins -> S.seq real) (nf:ins -> S.seq real) (r:ins) : S.seq real =
  opt_kv_write fa pos (sel r) (nf r)

(* invariant of the cache write after the first vc components *)
noextract let kvinv (p:ins -> prop) (fa:nat{fa < 6}) (pos:nat{pos < 2048}) (vc:nat) (s s0:S.seq BF.t) (nf:ins -> S.seq real) : prop =
  S.length s == 6291456 /\ S.length s0 == 6291456 /\
  (forall (i:nat). i < 6291456 /\ ~(opt_kv_hit fa pos i /\ opt_kv_comp i < vc) ==> S.index s i == S.index s0 i) /\
  (forall (r:ins) (i:nat). p r /\ i < 6291456 /\ opt_kv_hit fa pos i /\ opt_kv_comp i < vc ==>
     S.index s i %~ at (nf r) (opt_kv_comp i))

noextract let kvinv_step (p:ins -> prop) (fa:nat{fa < 6}) (pos:nat{pos < 2048}) (vc:nat{vc < 512}) (s s0:S.seq BF.t)
  (nf:ins -> S.seq real) (b:BF.t)
  : Lemma (requires kvinv p fa pos vc s s0 nf /\ (forall (r:ins). p r ==> b %~ at (nf r) vc))
          (ensures kv_ci fa pos vc < 6291456 /\ kvinv p fa pos (vc + 1) (S.upd s (kv_ci fa pos vc) b) s0 nf)
  = kv_ci_ok fa pos vc;
    let ci = kv_ci fa pos vc in
    introduce forall (i:nat). i < 6291456 /\ opt_kv_hit fa pos i /\ opt_kv_comp i == vc ==> i == ci
    with introduce _ ==> _ with kv_inj fa pos i

noextract let kvinv_done (p:ins -> prop) (fa:nat{fa < 6}) (pos:nat{pos < 2048}) (s s0:S.seq BF.t) (sel nf:ins -> S.seq real)
  : Lemma (requires kvinv p fa pos 512 s s0 nf /\ (forall (r:ins). p r ==> s0 %~ sel r))
          (ensures (forall (i:nat). i < 6291456 /\ not (opt_kv_hit fa pos i) ==> S.index s i == S.index s0 i) /\
                   apxb p s 6291456 (kvI fa pos sel nf))
  = introduce forall (r:ins) (i:nat). p r /\ i < 6291456 ==> S.index s i %~ at (kvI fa pos sel nf r) i
    with introduce _ ==> _ with (
      kvw_at fa pos (sel r) (nf r) i;
      if opt_kv_hit fa pos i then kv_inj fa pos i else ())

noextract let rkF (r:ins) : S.seq real = r.rk
noextract let rvF (r:ins) : S.seq real = r.rv

noextract let kvpost (p:ins -> prop) (fa pos:int) (s1 s2 sko svo:S.seq BF.t) (knF zvF:ins -> S.seq real) : prop =
  S.length s1 == 6291456 /\ S.length s2 == 6291456 /\
  S.length sko == 6291456 /\ S.length svo == 6291456 /\
  (forall (i:nat). i < 6291456 /\ not (opt_kv_hit fa pos i) ==>
     S.index s1 i == S.index sko i /\ S.index s2 i == S.index svo i) /\
  apxb p s1 6291456 (kvI fa pos rkF knF) /\
  apxb p s2 6291456 (kvI fa pos rvF zvF)

noextract let kv_done2 (p:ins -> prop) (fa:nat{fa < 6}) (pos:nat{pos < 2048}) (s1 s2 sko svo:S.seq BF.t) (knF zvF:ins -> S.seq real)
  : Lemma (requires kvinv p fa pos 512 s1 sko knF /\ kvinv p fa pos 512 s2 svo zvF /\
                    (forall (r:ins). p r ==> sko %~ r.rk /\ svo %~ r.rv))
          (ensures kvpost p fa pos s1 s2 sko svo knF zvF)
  = kvinv_done p fa pos s1 sko rkF knF;
    kvinv_done p fa pos s2 svo rvF zvF

noextract let kvinv_init (p:ins -> prop) (fa:nat{fa < 6}) (pos:nat{pos < 2048}) (s:S.seq BF.t) (nf:ins -> S.seq real)
  : Lemma (requires S.length s == 6291456) (ensures kvinv p fa pos 0 s s nf) = ()

inline_for_extraction noextract
let kv_idx (fa:SZ.t{SZ.v fa < 6}) (pos:SZ.t{SZ.v pos < 2048}) (vc:SZ.t{SZ.v vc < 512})
  : ci:SZ.t{SZ.v ci == kv_ci (SZ.v fa) (SZ.v pos) (SZ.v vc)}
  = assert (SZ.v vc / 256 <= 1);
    fa *^ (SZ.uint32_to_sizet 1048576ul) +^ (vc /^ 256sz) *^ (SZ.uint32_to_sizet 524288ul) +^ pos *^ 256sz +^ vc %^ 256sz

#push-options "--z3rlimit 80"
inline_for_extraction noextract
fn kv_kernel (kn zv : larray f32 512) (ko vo : larray bf16 6291456) (fa : sz{SZ.v fa < 6}) (pos : sz{SZ.v pos < 2048})
  (p:erased (ins -> prop)) (knF zvF:erased (ins -> S.seq real))
  (#sk #sv : erased (S.seq f32)) (#sko #svo : erased (S.seq bf16))
  preserves gpu ** kn |-> sk ** zv |-> sv
  requires ko |-> sko ** vo |-> svo ** pure (apxs p sk 512 knF /\ apxs p sv 512 zvF /\
    (forall (r:ins). reveal p r ==> reveal sko %~ r.rk /\ reveal svo %~ r.rv))
  ensures exists* sko' svo'. ko |-> sko' ** vo |-> svo' ** pure (
    kvpost p (SZ.v fa) (SZ.v pos) sko' svo' sko svo knF zvF)
{
  pts_to_len ko; pts_to_len vo; pts_to_len kn; pts_to_len zv;
  kvinv_init p (SZ.v fa) (SZ.v pos) sko knF;
  kvinv_init p (SZ.v fa) (SZ.v pos) svo zvF;
  let mut c = 0sz;
  while (!c <^ 512sz)
    invariant exists* (vc:sz) s1 s2. c |-> vc ** ko |-> s1 ** vo |-> s2 **
      pure (SZ.v vc <= 512 /\ kvinv p (SZ.v fa) (SZ.v pos) (SZ.v vc) s1 sko knF /\
        kvinv p (SZ.v fa) (SZ.v pos) (SZ.v vc) s2 svo zvF)
    decreases (512 - SZ.v !c)
  {
    let vc = !c;
    with s1. assert (ko |-> s1);
    with s2. assert (vo |-> s2);
    kv_ci_ok (SZ.v fa) (SZ.v pos) (SZ.v vc);
    let ci = kv_idx fa pos vc;
    let a = kn.(vc);
    let b = zv.(vc);
    c2b_all p a (xat knF (SZ.v vc));
    c2b_all p b (xat zvF (SZ.v vc));
    kvinv_step p (SZ.v fa) (SZ.v pos) (SZ.v vc) s1 sko knF (C.cast_f32_to_bf16 a);
    kvinv_step p (SZ.v fa) (SZ.v pos) (SZ.v vc) s2 svo zvF (C.cast_f32_to_bf16 b);
    ko.(ci) <- C.cast_f32_to_bf16 a;
    vo.(ci) <- C.cast_f32_to_bf16 b;
    c := vc +^ 1sz;
  };
  with s1. assert (ko |-> s1);
  with s2. assert (vo |-> s2);
  assert (pure (kvinv p (SZ.v fa) (SZ.v pos) 512 s1 sko knF));
  assert (pure (kvinv p (SZ.v fa) (SZ.v pos) 512 s2 svo zvF));
  assert (pure (forall (r:ins). reveal p r ==> reveal sko %~ r.rk /\ reveal svo %~ r.rv));
  kv_done2 p (SZ.v fa) (SZ.v pos) s1 s2 sko svo knF zvF;
  assert (pure (kvpost p (SZ.v fa) (SZ.v pos) s1 s2 sko svo knF zvF));
}
#pop-options


(**** Section K4 ****)
#set-options "--fuel 0 --ifuel 0 --z3rlimit 20"

(*** Window partition *)
noextract let kf (w pos:nat) : nat = if w <= pos then (pos - w) / 16 + 1 else 0

noextract let kf_lt (w pos k:nat)
  : Lemma (w + k * 16 <= pos <==> k < kf w pos)
  = if w <= pos then (
      let n = pos - w in
      ML.lemma_div_mod n 16;
      if k * 16 <= n then (ML.lemma_div_le (k * 16) n 16; ML.cancel_mul_div k 16)
      else (if k <= n / 16 then ML.lemma_mult_le_right 16 k (n / 16) else ()))
    else ()

noextract let kf_le (w pos:nat) : Lemma (requires w < 16 /\ pos < 2048) (ensures kf w pos <= 128) =
  if w <= pos then ML.lemma_div_le (pos - w) 2047 16 else ()

noextract let awh (h:nat -> real) (wi k:nat) : real = vsum k (stride h wi 16)
noextract let gsum (h:nat -> real) (pos:nat) (w:nat) : real = if w <= pos then awh h w (kf w pos) else 0.0R
noextract let pad (h:nat -> real) (pos:nat) (p:nat) : real = if p <= pos then h p else 0.0R

noextract let part_one (h:nat -> real) (pos:nat{pos < 2048}) (w:nat)
  : Lemma (requires w < 16) (ensures strd (pad h pos) 16 128 w == gsum h pos w)
  = let k = kf w pos in
    kf_le w pos;
    let g = stride (pad h pos) w 16 in
    vsum_split k (128 - k) g;
    introduce forall (j:nat). j < k ==> g j == stride h w 16 j
      with kf_lt w pos j;
    vsum_ext k g (stride h w 16);
    introduce forall (j:nat). j < 128 - k ==> shift g k j == zf j
      with kf_lt w pos (k + j);
    vsum_ext (128 - k) (shift g k) zf;
    vsum_zeros (128 - k);
    if w <= pos then () else vsum_zero (stride h w 16)

noextract let partition (h:nat -> real) (pos:nat{pos < 2048})
  : Lemma (vsum 16 (gsum h pos) == vsum (pos + 1) h)
  = vsum_strided 16 128 (pad h pos);
    introduce forall (w:nat). w < 16 ==> strd (pad h pos) 16 128 w == gsum h pos w
      with introduce _ ==> _ with part_one h pos w;
    vsum_ext 16 (strd (pad h pos) 16 128) (gsum h pos);
    vsum_split (pos + 1) (2047 - pos) (pad h pos);
    vsum_ext (pos + 1) (pad h pos) h;
    vsum_ext (2047 - pos) (shift (pad h pos) (pos + 1)) zf;
    vsum_zeros (2047 - pos)

(*** Real algebra of the online softmax *)
noextract let div_mul (a b c:real)
  : Lemma (requires b =!= 0.0R /\ c =!= 0.0R) (ensures (a /. b) *. (b /. c) == a /. c)
  = assert (a /. b == a *. (1.0R /. b));
    assert (b /. c == b *. (1.0R /. c));
    assert ((1.0R /. b) *. b == 1.0R)

noextract let div_add (a b c:real)
  : Lemma (requires c =!= 0.0R) (ensures a /. c +. b /. c == (a +. b) /. c)
  = ()

noextract let alg1 (a eo em es:real)
  : Lemma (requires eo >. 0.0R /\ em >. 0.0R)
          (ensures (a /. eo) *. (eo /. em) +. es /. em == (a +. es) /. em)
  = div_mul a eo em; div_add a es em

noextract let alg2 (b eo em es v:real)
  : Lemma (requires eo >. 0.0R /\ em >. 0.0R)
          (ensures (b /. eo) *. (eo /. em) +. (es /. em) *. v == (b +. es *. v) /. em)
  = div_mul b eo em;
    assert ((es /. em) *. v == (es *. v) /. em);
    div_add b (es *. v) em

noextract let alg3 (x y e g:real)
  : Lemma (requires e >. 0.0R /\ y >. 0.0R)
          (ensures y /. e =!= 0.0R /\ ((x /. e) *. (1.0R /. (y /. e))) *. g == (x /. y) *. g)
  = assert (y /. e >. 0.0R);
    assert (1.0R /. (y /. e) == e /. y);
    div_mul x e y

(*** Float facts *)
noextract let tr (x:F32.t) : real = AS.to_real #F32.t #_ #_ x
noextract let tr_ok (x:F32.t) : Lemma (x %~ tr x) = AS.to_real_ok #F32.t #_ #_ x
noextract let eT (x:F32.t) : real = Exp.exp (tr x)
noextract let eT_pos (x:F32.t) : Lemma (eT x >. 0.0R) = Exp.exp_positive (tr x)

noextract let subc (a b res:F32.t) : prop = forall (x y:real). a %~ x /\ b %~ y ==> res %~ (x -. y)
noextract let expc (a res:F32.t) : prop = forall (xr:real). a %~ xr ==> res %~ Exp.exp xr
noextract let addc (a b res:F32.t) : prop = forall (xr yr:real). a %~ xr /\ b %~ yr ==> res %~ (xr +. yr)
noextract let mulc (a b res:F32.t) : prop = forall (xr yr:real). a %~ xr /\ b %~ yr ==> res %~ (xr *. yr)
noextract let fmac (a b c res:F32.t) : prop =
  forall (xr yr zr:real). a %~ xr /\ b %~ yr /\ c %~ zr ==> res %~ (xr *. yr +. zr)
noextract let rcpc (a res:F32.t) : prop = forall (xr:real{xr =!= 0.0R}). a %~ xr ==> res %~ (1.0R /. xr)

noextract let bcast_all (p:ins -> prop) (s:S.seq BF.t) (n:nat) (f:ins -> S.seq real) (j:nat)
  : Lemma (requires apxb p s n f /\ j < n)
          (ensures forall (r:ins). p r ==> C.cast_bf16_to_f32 (S.index s j) %~ at (f r) j)
  = introduce forall (r:ins). p r ==> C.cast_bf16_to_f32 (S.index s j) %~ at (f r) j
    with introduce _ ==> _ with C.cast_bf16_to_f32_ok (S.index s j) (at (f r) j)

noextract let exp_diff (a b d e:F32.t)
  : Lemma (requires subc a b d /\ expc d e) (ensures e %~ (eT a /. eT b))
  = tr_ok a; tr_ok b; eT_pos b;
    assert (d %~ (tr a -. tr b));
    Exp.exp_sub (tr a) (tr b)

noextract let wt_ok (p:ins -> prop) (sc m d wt:F32.t) (fs:ins -> real)
  : Lemma (requires (forall (r:ins). p r ==> sc %~ fs r) /\ subc sc m d /\ expc d wt)
          (ensures forall (r:ins). p r ==> wt %~ (Exp.exp (fs r) /. eT m))
  = tr_ok m; eT_pos m;
    introduce forall (r:ins). p r ==> wt %~ (Exp.exp (fs r) /. eT m)
    with introduce _ ==> _ with (
      assert (d %~ (fs r -. tr m));
      Exp.exp_sub (fs r) (tr m))

noextract let acc_step (p:ins -> prop) (a0 b0 c1:ins -> real) (m g t f vt nt:F32.t)
  : Lemma (requires f %~ (eT m /. eT g) /\ fmac t f vt nt /\
                    (forall (r:ins). p r ==> t %~ (a0 r /. eT m) /\ vt %~ (b0 r /. eT g) /\ c1 r == b0 r +. a0 r))
          (ensures forall (r:ins). p r ==> nt %~ (c1 r /. eT g))
  = eT_pos m; eT_pos g;
    introduce forall (r:ins). p r ==> nt %~ (c1 r /. eT g)
    with introduce _ ==> _ with (
      assert (nt %~ ((a0 r /. eT m) *. (eT m /. eT g) +. b0 r /. eT g));
      alg1 (a0 r) (eT m) (eT g) (b0 r))

noextract let pv_step (p:ins -> prop) (b0 es v b1:ins -> real) (o m po diff wt cv mw np:F32.t)
  : Lemma (requires diff %~ (eT o /. eT m) /\ mulc wt cv mw /\ fmac po diff mw np /\
                    (forall (r:ins). p r ==> po %~ (b0 r /. eT o) /\ wt %~ (es r /. eT m) /\ cv %~ v r /\
                       b1 r == b0 r +. es r *. v r))
          (ensures forall (r:ins). p r ==> np %~ (b1 r /. eT m))
  = eT_pos o; eT_pos m;
    introduce forall (r:ins). p r ==> np %~ (b1 r /. eT m)
    with introduce _ ==> _ with (
      assert (mw %~ ((es r /. eT m) *. v r));
      assert (np %~ ((b0 r /. eT o) *. (eT o /. eT m) +. (es r /. eT m) *. v r));
      alg2 (b0 r) (eT o) (eT m) (es r) (v r))

noextract let rdiv_pos (x y:real) : Lemma (requires y >. 0.0R) (ensures rdiv x y == x /. y) = ()

noextract let fin_step (p:ins -> prop) (ob ta sg:ins -> real) (g vs vt rc o1 sv res:F32.t)
  : Lemma (requires rcpc vt rc /\ mulc vs rc o1 /\ mulc o1 sv res /\
                    (forall (r:ins). p r ==> vs %~ (ob r /. eT g) /\ vt %~ (ta r /. eT g) /\ ta r >. 0.0R /\ sv %~ sg r))
          (ensures forall (r:ins). p r ==> res %~ (rdiv (ob r) (ta r) *. sg r))
  = eT_pos g;
    introduce forall (r:ins). p r ==> res %~ (rdiv (ob r) (ta r) *. sg r)
    with introduce _ ==> _ with (
      alg3 (ob r) (ta r) (eT g) (sg r);
      assert (rc %~ (1.0R /. (ta r /. eT g)));
      assert (o1 %~ ((ob r /. eT g) *. (1.0R /. (ta r /. eT g))));
      assert (res %~ (((ob r /. eT g) *. (1.0R /. (ta r /. eT g))) *. sg r));
      rdiv_pos (ob r) (ta r))

noextract let sig_eq (x:real) : Lemma (1.0R +. Exp.exp (0.0R -. x) >. 0.0R /\ sigmoid x == 1.0R /. (1.0R +. Exp.exp (0.0R -. x))) =
  Exp.exp_positive (0.0R -. x)

noextract let sig_chain (p:ins -> prop) (gf:ins -> real) (ng e d res:F32.t)
  : Lemma (requires (forall (r:ins). p r ==> ng %~ (0.0R -. gf r)) /\ expc ng e /\
                    addc (Kuiper.Scalars.one #F32.t #_) e d /\ rcpc d res)
          (ensures forall (r:ins). p r ==> res %~ sigmoid (gf r))
  = one_ok ();
    introduce forall (r:ins). p r ==> res %~ sigmoid (gf r)
    with introduce _ ==> _ with (
      sig_eq (gf r);
      assert (e %~ Exp.exp (0.0R -. gf r));
      assert (d %~ (1.0R +. Exp.exp (0.0R -. gf r))))

(*** Attention ideals *)
noextract let scg (qF kF:ins -> S.seq real) (hd kb:nat) (r:ins) (i:nat) : real =
  at (qF r) (hd * 256 + i) *. at (kF r) (kb + i)
noextract let scP (qF kF:ins -> S.seq real) (hd kb l e:nat) (r:ins) : real = vsum e (shift (scg qF kF hd kb r) (l * 8))
noextract let scL (qF kF:ins -> S.seq real) (hd kb:nat) (r:ins) (l:nat) : real = lane (scg qF kF hd kb r) 1 l
noextract let sc32 (qF kF:ins -> S.seq real) (hd kb:nat) (r:ins) : real = vsum 32 (scL qF kF hd kb r)
noextract let scI (qF kF:ins -> S.seq real) (hd kb:nat) (r:ins) : real = vsum 256 (scg qF kF hd kb r) *. (1.0R /. of_int 16)
noextract let esc (qF kF:ins -> S.seq real) (hd kb:nat) (r:ins) : real = Exp.exp (scI qF kF hd kb r)

noextract let cbf (fa hd:nat) : nat = fa * 1048576 + (hd / 4) * 524288
noextract let kbi (cb p:nat) : nat = cb + p * 256
noextract let eS (qF kF:ins -> S.seq real) (hd cb:nat) (r:ins) (p:nat) : real = esc qF kF hd (kbi cb p) r
noextract let eSV (qF kF vF:ins -> S.seq real) (hd cb e:nat) (r:ins) (p:nat) : real =
  eS qF kF hd cb r p *. at (vF r) (kbi cb p + e)
noextract let wA (qF kF:ins -> S.seq real) (hd cb wi k:nat) (r:ins) : real = awh (eS qF kF hd cb r) wi k
noextract let wB (qF kF vF:ins -> S.seq real) (hd cb e wi k:nat) (r:ins) : real = awh (eSV qF kF vF hd cb e r) wi k
noextract let tA (qF kF:ins -> S.seq real) (hd cb pos n:nat) (r:ins) : real = vsum n (gsum (eS qF kF hd cb r) pos)
noextract let oB (qF kF vF:ins -> S.seq real) (hd cb e pos n:nat) (r:ins) : real = vsum n (gsum (eSV qF kF vF hd cb e r) pos)
noextract let taF (qF kF:ins -> S.seq real) (hd cb pos:nat) (r:ins) : real = vsum (pos + 1) (eS qF kF hd cb r)
noextract let obF (qF kF vF:ins -> S.seq real) (hd cb e pos:nat) (r:ins) : real = vsum (pos + 1) (eSV qF kF vF hd cb e r)
noextract let sgF (projF:ins -> S.seq real) (hd e:nat) (r:ins) : real = sigmoid (at (projF r) (hd * 512 + 256 + e))
noextract let attO (qF kF vF projF:ins -> S.seq real) (fa pos hd e:nat) (r:ins) : real =
  rdiv (obF qF kF vF hd (cbf fa hd) e pos r) (taF qF kF hd (cbf fa hd) pos r) *. sgF projF hd e r
noextract let attI (qF kF vF projF:ins -> S.seq real) (fa pos:nat) (r:ins) : S.seq real =
  S.init 2048 (fun (c:nat{c < 2048}) -> attO qF kF vF projF fa pos (c / 256) (c % 256) r)

(*** Ideal step lemmas *)
noextract let lane1 (g:nat -> real) (l:nat) : Lemma (lane g 1 l == vsum 8 (shift g (l * 8))) =
  vsum_succ 0 (chunk g l); vsum_zero (chunk g l);
  vsum_ext 8 (shift g (0 * 256 + l * 8)) (shift g (l * 8))

noextract let lane1_all (qF kF:ins -> S.seq real) (hd kb l:nat)
  : Lemma (forall (r:ins). scL qF kF hd kb r l == scP qF kF hd kb l 8 r)
  = introduce forall (r:ins). scL qF kF hd kb r l == scP qF kF hd kb l 8 r
    with lane1 (scg qF kF hd kb r) l

noextract let sc_sum (qF kF:ins -> S.seq real) (hd kb:nat)
  : Lemma (forall (r:ins). sc32 qF kF hd kb r == vsum 256 (scg qF kF hd kb r))
  = introduce forall (r:ins). sc32 qF kF hd kb r == vsum 256 (scg qF kF hd kb r)
    with (vsum_ext 32 (scL qF kF hd kb r) (lane (scg qF kF hd kb r) 1);
          dot_lanes (scg qF kF hd kb r) 1)

noextract let kq (wi vp:nat) : nat = if vp >= wi then (vp - wi) / 16 else 0

noextract let kq_init (wi:nat) : Lemma (kq wi wi == 0) = ()

noextract let kq_step (wi vp:nat)
  : Lemma (requires vp == wi + 16 * kq wi vp)
          (ensures kq wi (vp + 16) == kq wi vp + 1 /\ vp + 16 == wi + 16 * kq wi (vp + 16))
  = let k = kq wi vp in
    ML.cancel_mul_div (k + 1) 16;
    assert (vp + 16 - wi == (k + 1) * 16)

noextract let wA_succ (qF kF:ins -> S.seq real) (hd cb wi k kb:nat)
  : Lemma (requires kb == kbi cb (wi + k * 16))
          (ensures forall (r:ins). wA qF kF hd cb wi (k + 1) r == esc qF kF hd kb r +. wA qF kF hd cb wi k r)
  = step_stride (eS qF kF hd cb) wi 16 k

noextract let wB_succ (qF kF vF:ins -> S.seq real) (hd cb e wi k kb:nat)
  : Lemma (requires kb == kbi cb (wi + k * 16))
          (ensures forall (r:ins). wB qF kF vF hd cb e wi (k + 1) r ==
                     wB qF kF vF hd cb e wi k r +. esc qF kF hd kb r *. at (vF r) (kb + e))
  = step_stride (eSV qF kF vF hd cb e) wi 16 k

noextract let wA0 (qF kF:ins -> S.seq real) (hd cb wi:nat)
  : Lemma (forall (r:ins). wA qF kF hd cb wi 0 r == 0.0R)
  = zero_stride (eS qF kF hd cb) wi 16

noextract let wB0 (qF kF vF:ins -> S.seq real) (hd cb wi:nat)
  : Lemma (forall (r:ins) (e:nat). wB qF kF vF hd cb e wi 0 r == 0.0R)
  = introduce forall (r:ins) (e:nat). wB qF kF vF hd cb e wi 0 r == 0.0R
    with vsum_zero (stride (eSV qF kF vF hd cb e r) wi 16)

noextract let tA_succ (qF kF:ins -> S.seq real) (hd cb pos n:nat)
  : Lemma (forall (r:ins). tA qF kF hd cb pos (n + 1) r ==
             tA qF kF hd cb pos n r +. (if n <= pos then wA qF kF hd cb n (kf n pos) r else 0.0R))
  = introduce forall (r:ins). tA qF kF hd cb pos (n + 1) r ==
             tA qF kF hd cb pos n r +. (if n <= pos then wA qF kF hd cb n (kf n pos) r else 0.0R)
    with vsum_succ n (gsum (eS qF kF hd cb r) pos)

noextract let oB_succ (qF kF vF:ins -> S.seq real) (hd cb e pos n:nat)
  : Lemma (forall (r:ins). oB qF kF vF hd cb e pos (n + 1) r ==
             oB qF kF vF hd cb e pos n r +. (if n <= pos then wB qF kF vF hd cb e n (kf n pos) r else 0.0R))
  = introduce forall (r:ins). oB qF kF vF hd cb e pos (n + 1) r ==
             oB qF kF vF hd cb e pos n r +. (if n <= pos then wB qF kF vF hd cb e n (kf n pos) r else 0.0R)
    with vsum_succ n (gsum (eSV qF kF vF hd cb e r) pos)

noextract let tA0 (qF kF:ins -> S.seq real) (hd cb pos:nat) : Lemma (forall (r:ins). tA qF kF hd cb pos 0 r == 0.0R) =
  introduce forall (r:ins). tA qF kF hd cb pos 0 r == 0.0R with vsum_zero (gsum (eS qF kF hd cb r) pos)
noextract let oB0 (qF kF vF:ins -> S.seq real) (hd cb e pos:nat) : Lemma (forall (r:ins). oB qF kF vF hd cb e pos 0 r == 0.0R) =
  introduce forall (r:ins). oB qF kF vF hd cb e pos 0 r == 0.0R with vsum_zero (gsum (eSV qF kF vF hd cb e r) pos)

noextract let ta_done (qF kF:ins -> S.seq real) (hd cb:nat) (pos:nat{pos < 2048})
  : Lemma (forall (r:ins). tA qF kF hd cb pos 16 r == taF qF kF hd cb pos r /\ taF qF kF hd cb pos r >. 0.0R)
  = introduce forall (r:ins). tA qF kF hd cb pos 16 r == taF qF kF hd cb pos r /\ taF qF kF hd cb pos r >. 0.0R
    with (
      partition (eS qF kF hd cb r) pos;
      introduce forall (k:nat). k < pos + 1 ==> eS qF kF hd cb r k >. 0.0R
        with Exp.exp_positive (scI qF kF hd (kbi cb k) r);
      vsum_pos (pos + 1) (eS qF kF hd cb r))

noextract let ob_done (qF kF vF:ins -> S.seq real) (hd cb e:nat) (pos:nat{pos < 2048})
  : Lemma (forall (r:ins). oB qF kF vF hd cb e pos 16 r == obF qF kF vF hd cb e pos r)
  = introduce forall (r:ins). oB qF kF vF hd cb e pos 16 r == obF qF kF vF hd cb e pos r
    with partition (eSV qF kF vF hd cb e r) pos

noextract let tA_init (p:ins -> prop) (qF kF:ins -> S.seq real) (hd cb pos:nat) (g:F32.t)
  : Lemma (forall (r:ins). p r ==> zero #F32.t #_ %~ (tA qF kF hd cb pos 0 r /. eT g))
  = tA0 qF kF hd cb pos; zero_ok (); eT_pos g;
    assert (0.0R /. eT g == 0.0R)
noextract let oB_init (p:ins -> prop) (qF kF vF:ins -> S.seq real) (hd cb e pos:nat) (g:F32.t)
  : Lemma (forall (r:ins). p r ==> zero #F32.t #_ %~ (oB qF kF vF hd cb e pos 0 r /. eT g))
  = oB0 qF kF vF hd cb e pos; zero_ok (); eT_pos g;
    assert (0.0R /. eT g == 0.0R)

(*** Window results *)
noextract let winv (p:ins -> prop) (qF kF vF:ins -> S.seq real) (hd cb wi k:nat) (vs vm:F32.t) (s sp:S.seq F32.t) : prop =
  S.length s == 4096 /\ S.length sp == 4096 /\ wi < 16 /\
  (forall (r:ins). p r ==> vs %~ (wA qF kF hd cb wi k r /. eT vm)) /\
  (forall (r:ins) (i:nat). p r /\ i < 256 ==> S.index s (wi * 256 + i) %~ (wB qF kF vF hd cb i wi k r /. eT vm)) /\
  (forall (j:nat). j < 4096 /\ (j < wi * 256 \/ wi * 256 + 256 <= j) ==> S.index s j == S.index sp j)

noextract let wdone (p:ins -> prop) (qF kF vF:ins -> S.seq real) (hd cb pos:nat) (sp sm st:S.seq F32.t) (w:nat) : prop =
  S.length sp == 4096 /\ S.length sm == 16 /\ S.length st == 16 /\ w < 16 /\
  (forall (r:ins). p r ==> S.index st w %~ (wA qF kF hd cb w (kf w pos) r /. eT (S.index sm w))) /\
  (forall (r:ins) (i:nat). p r /\ i < 256 ==>
     S.index sp (w * 256 + i) %~ (wB qF kF vF hd cb i w (kf w pos) r /. eT (S.index sm w)))

noextract let wdone_frame (p:ins -> prop) (qF kF vF:ins -> S.seq real) (hd cb pos:nat) (sp sm st sp' sm' st':S.seq F32.t) (wi n:nat)
  : Lemma (requires (forall (w:nat). w < n ==> wdone p qF kF vF hd cb pos sp sm st w) /\ n == wi /\ wi < 16 /\
                    S.length sp' == 4096 /\ S.length sm' == 16 /\ S.length st' == 16 /\
                    wdone p qF kF vF hd cb pos sp' sm' st' wi /\
                    S.length sp == 4096 /\ S.length sm == 16 /\ S.length st == 16 /\
                    (forall (j:nat). j < 16 /\ j <> wi ==> S.index sm' j == S.index sm j /\ S.index st' j == S.index st j) /\
                    (forall (j:nat). j < 4096 /\ (j < wi * 256 \/ wi * 256 + 256 <= j) ==> S.index sp' j == S.index sp j))
          (ensures forall (w:nat). w < n + 1 ==> wdone p qF kF vF hd cb pos sp' sm' st' w)
  = introduce forall (w:nat). w < n + 1 ==> wdone p qF kF vF hd cb pos sp' sm' st' w
    with introduce _ ==> _ with (
      if w = wi then () else (
        assert (forall (i:nat). i < 256 ==> w * 256 + i < wi * 256);
        assert (forall (i:nat). i < 256 ==> S.index sp' (w * 256 + i) == S.index sp (w * 256 + i))))

noextract let head_step (p:ins -> prop) (qF kF vF projF:ins -> S.seq real) (fa pos h:nat) (so so':S.seq F32.t)
  : Lemma (requires h < 8 /\ S.length so == 2048 /\ S.length so' == 2048 /\ apxs p so (h * 256) (attI qF kF vF projF fa pos) /\
                    (forall (r:ins) (i:nat). p r /\ i < 256 ==> S.index so' (h * 256 + i) %~ attO qF kF vF projF fa pos h i r) /\
                    (forall (j:nat). j < 2048 /\ (j < h * 256 \/ h * 256 + 256 <= j) ==> S.index so' j == S.index so j))
          (ensures apxs p so' ((h + 1) * 256) (attI qF kF vF projF fa pos))
  = introduce forall (r:ins) (j:nat). p r /\ j < (h + 1) * 256 ==> S.index so' j %~ at (attI qF kF vF projF fa pos r) j
    with introduce _ ==> _ with (
      if j < h * 256 then () else (let i = j - h * 256 in divmod256 h i; assert (j == h * 256 + i)))

(*** Kernels *)
inline_for_extraction noextract
fn sigm (g : f32) (p:erased (ins -> prop)) (gf:erased (ins -> real))
  preserves gpu
  requires pure (forall (r:ins). reveal p r ==> g %~ reveal gf r)
  returns res : F32.t
  ensures pure (forall (r:ins). reveal p r ==> res %~ sigmoid (reveal gf r))
{
  let ng = FL.neg g;
  neg_all p g gf;
  let e = FM.exp ng;
  let d = F32.add_rn_ftz (one #f32 #_) e;
  let res = F32.rcp_approx_ftz d;
  sig_chain p gf ng e d res;
  res
}

inline_for_extraction noextract
fn score (qn : larray f32 2048) (hd : sz{SZ.v hd < 8}) (kc : larray bf16 6291456) (kb : sz{SZ.v kb + 256 <= 6291456})
  (lanes : larray f32 32) (p:erased (ins -> prop)) (qF kF:erased (ins -> S.seq real))
  (#sq #sl : erased (S.seq f32)) (#sk : erased (S.seq bf16))
  preserves gpu ** qn |-> sq ** kc |-> sk
  requires lanes |-> sl ** pure (apxs p sq 2048 qF /\ apxb p sk 6291456 kF)
  returns res : F32.t
  ensures exists* sl'. lanes |-> sl' ** pure (forall (r:ins). reveal p r ==> res %~ scI qF kF (SZ.v hd) (SZ.v kb) r)
{
  pts_to_len lanes; pts_to_len qn; pts_to_len kc;
  let mut l = 0sz;
  while (!l <^ 32sz)
    invariant exists* (vl:sz) s1. l |-> vl ** lanes |-> s1 ** pure (SZ.v vl <= 32 /\ S.length s1 == 32 /\
      apx p s1 (SZ.v vl) (scL qF kF (SZ.v hd) (SZ.v kb)))
    decreases (32 - SZ.v !l)
  {
    let vl = !l;
    zero_ok ();
    zero_shift (scg qF kF (SZ.v hd) (SZ.v kb)) (SZ.v vl * 8);
    let mut s = zero #f32 #_;
    let mut e = 0sz;
    while (!e <^ 8sz)
      invariant exists* (ve:sz) (vs:f32). e |-> ve ** s |-> vs ** pure (SZ.v ve <= 8 /\
        (forall (r:ins). reveal p r ==> vs %~ scP qF kF (SZ.v hd) (SZ.v kb) (SZ.v vl) (SZ.v ve) r))
      decreases (8 - SZ.v !e)
    {
      let ve = !e;
      let i = vl *^ 8sz +^ ve;
      let qi = hd *^ 256sz +^ i;
      let ki = kb +^ i;
      assert (pure (SZ.v qi == SZ.v hd * 256 + (SZ.v vl * 8 + SZ.v ve) /\ SZ.v ki == SZ.v kb + (SZ.v vl * 8 + SZ.v ve)));
      let a = qn.(qi);
      let b = kc.(ki);
      let vs = !s;
      bcast_all p sk 6291456 kF (SZ.v ki);
      let cb = C.cast_bf16_to_f32 b;
      let ns = F32.fma_rn_ftz a cb vs;
      step_shift (scg qF kF (SZ.v hd) (SZ.v kb)) (SZ.v vl * 8) (SZ.v ve);
      fma_lift p a cb vs ns (xat qF (SZ.v qi)) (xat kF (SZ.v ki))
        (scP qF kF (SZ.v hd) (SZ.v kb) (SZ.v vl) (SZ.v ve)) (scP qF kF (SZ.v hd) (SZ.v kb) (SZ.v vl) (SZ.v ve + 1));
      s := ns;
      e := ve +^ 1sz;
    };
    let vs = !s;
    lane1_all qF kF (SZ.v hd) (SZ.v kb) (SZ.v vl);
    lanes.(vl) <- vs;
    l := vl +^ 1sz;
  };
  let t = tree32 lanes p (scL qF kF (SZ.v hd) (SZ.v kb));
  sc_sum qF kF (SZ.v hd) (SZ.v kb);
  let c16 = FL.of_int #f32 16L;
  of_int_ok 16L;
  let rc = F32.rcp_approx_ftz c16;
  assert (pure (rc %~ (1.0R /. of_int 16)));
  let res = F32.mul_rn_ftz t rc;
  mul_lift p t rc res (sc32 qF kF (SZ.v hd) (SZ.v kb)) (cst (1.0R /. of_int 16)) (scI qF kF (SZ.v hd) (SZ.v kb));
  res
}

inline_for_extraction noextract
fn zero_block (partial : larray f32 4096) (wi : sz{SZ.v wi < 16}) (#sp : erased (S.seq f32))
  preserves gpu
  requires partial |-> sp
  ensures exists* s. partial |-> s ** pure (S.length s == 4096 /\ S.length sp == 4096 /\
    (forall (i:nat). i < 256 ==> S.index s (SZ.v wi * 256 + i) == zero #f32 #_) /\
    (forall (j:nat). j < 4096 /\ (j < SZ.v wi * 256 \/ SZ.v wi * 256 + 256 <= j) ==> S.index s j == S.index sp j))
{
  pts_to_len partial;
  let mut i = 0sz;
  while (!i <^ 256sz)
    invariant exists* (vi:sz) s. i |-> vi ** partial |-> s ** pure (SZ.v vi <= 256 /\ S.length s == 4096 /\
      (forall (j:nat). j < SZ.v vi ==> S.index s (SZ.v wi * 256 + j) == zero #f32 #_) /\
      (forall (j:nat). j < 4096 /\ (j < SZ.v wi * 256 \/ SZ.v wi * 256 + SZ.v vi <= j) ==> S.index s j == S.index sp j))
    decreases (256 - SZ.v !i)
  {
    let vi = !i;
    partial.(wi *^ 256sz +^ vi) <- zero #f32 #_;
    i := vi +^ 1sz;
  }
}

noextract let winv0 (p:ins -> prop) (qF kF vF:ins -> S.seq real) (hd cb wi:nat) (vm:F32.t) (s sp:S.seq F32.t)
  : Lemma (requires wi < 16 /\ S.length s == 4096 /\ S.length sp == 4096 /\
                    (forall (i:nat). i < 256 ==> S.index s (wi * 256 + i) == zero #f32 #_) /\
                    (forall (j:nat). j < 4096 /\ (j < wi * 256 \/ wi * 256 + 256 <= j) ==> S.index s j == S.index sp j))
          (ensures winv p qF kF vF hd cb wi 0 (zero #f32 #_) vm s sp)
  = zero_ok (); wA0 qF kF hd cb wi; wB0 qF kF vF hd cb wi; eT_pos vm

noextract let po_ok (p:ins -> prop) (qF kF vF:ins -> S.seq real) (hd cb wi k vi:nat) (vs vm:F32.t) (s s1 sp:S.seq F32.t)
  : Lemma (requires winv p qF kF vF hd cb wi k vs vm s1 sp /\ vi < 256 /\ S.length s == 4096 /\
                    S.index s (wi * 256 + vi) == S.index s1 (wi * 256 + vi))
          (ensures forall (r:ins). p r ==> S.index s (wi * 256 + vi) %~ (wB qF kF vF hd cb vi wi k r /. eT vm))
  = ()

inline_for_extraction noextract
fn window (qn : larray f32 2048) (hd : sz{SZ.v hd < 8}) (kc vc : larray bf16 6291456)
  (cb : sz{SZ.v cb + 524288 <= 6291456}) (pos : sz{SZ.v pos < 2048}) (wi : sz{SZ.v wi < 16})
  (lanes : larray f32 32) (partial : larray f32 4096) (maxs totals : larray f32 16)
  (p:erased (ins -> prop)) (qF kF vF:erased (ins -> S.seq real))
  (#sq #sl #sp #sm #st : erased (S.seq f32)) (#sk #sv : erased (S.seq bf16))
  preserves gpu ** qn |-> sq ** kc |-> sk ** vc |-> sv
  requires lanes |-> sl ** partial |-> sp ** maxs |-> sm ** totals |-> st **
    pure (apxs p sq 2048 qF /\ apxb p sk 6291456 kF /\ apxb p sv 6291456 vF)
  ensures exists* sl' sp' sm' st'. lanes |-> sl' ** partial |-> sp' ** maxs |-> sm' ** totals |-> st' ** pure (
    S.length sp == 4096 /\ S.length sm == 16 /\ S.length st == 16 /\
    S.length sp' == 4096 /\ S.length sm' == 16 /\ S.length st' == 16 /\
    wdone p qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v pos) sp' sm' st' (SZ.v wi) /\
    (forall (j:nat). j < 16 /\ j <> SZ.v wi ==> S.index sm' j == S.index sm j /\ S.index st' j == S.index st j) /\
    (forall (j:nat). j < 4096 /\ (j < SZ.v wi * 256 \/ SZ.v wi * 256 + 256 <= j) ==> S.index sp' j == S.index sp j))
{
  pts_to_len partial; pts_to_len maxs; pts_to_len totals; pts_to_len vc; pts_to_len kc; pts_to_len qn; pts_to_len lanes;
  zero_block partial wi;
  with s0. assert (partial |-> s0);
  let mx0 = FL.neg (FL.infinity #f32);
  winv0 p qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v wi) mx0 s0 sp;
  kq_init (SZ.v wi);
  let mut mx = mx0;
  let mut sum = zero #f32 #_;
  let mut pp = wi;
  while (!pp <=^ pos)
    invariant exists* (vp:sz) (vm vs:f32) s sl1. pp |-> vp ** mx |-> vm ** sum |-> vs ** partial |-> s ** lanes |-> sl1 **
      pure (SZ.v vp <= SZ.v pos + 16 /\ SZ.v vp == SZ.v wi + 16 * kq (SZ.v wi) (SZ.v vp) /\
        kq (SZ.v wi) (SZ.v vp) <= kf (SZ.v wi) (SZ.v pos) /\
        winv p qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v wi) (kq (SZ.v wi) (SZ.v vp)) vs vm s sp)
    decreases (SZ.v pos + 16 - SZ.v !pp)
  {
    let vp = !pp;
    kf_lt (SZ.v wi) (SZ.v pos) (kq (SZ.v wi) (SZ.v vp));
    let kb = cb +^ vp *^ 256sz;
    assert (pure (SZ.v kb == kbi (SZ.v cb) (SZ.v wi + (kq (SZ.v wi) (SZ.v vp)) * 16)));
    let sc = score qn hd kc kb lanes p qF kF;
    let old = !mx;
    let m = FL.fmax old sc;
    mx := m;
    let d1 = sub_ftz old m;
    let diff = FM.exp d1;
    let d2 = sub_ftz sc m;
    let wt = FM.exp d2;
    let vs = !sum;
    let ns = F32.fma_rn_ftz vs diff wt;
    exp_diff old m d1 diff;
    wt_ok p sc m d2 wt (scI qF kF (SZ.v hd) (SZ.v kb));
    wA_succ qF kF (SZ.v hd) (SZ.v cb) (SZ.v wi) (kq (SZ.v wi) (SZ.v vp)) (SZ.v kb);
    acc_step p (wA qF kF (SZ.v hd) (SZ.v cb) (SZ.v wi) (kq (SZ.v wi) (SZ.v vp))) (esc qF kF (SZ.v hd) (SZ.v kb))
      (wA qF kF (SZ.v hd) (SZ.v cb) (SZ.v wi) ((kq (SZ.v wi) (SZ.v vp)) + 1)) old m vs diff wt ns;
    sum := ns;
    with s1. assert (partial |-> s1);
    let mut i = 0sz;
    while (!i <^ 256sz)
      invariant exists* (vi:sz) s. i |-> vi ** partial |-> s ** pure (SZ.v vi <= 256 /\ S.length s == 4096 /\
        (forall (j:nat). j < 4096 /\ (j < SZ.v wi * 256 \/ SZ.v wi * 256 + SZ.v vi <= j) ==> S.index s j == S.index s1 j) /\
        (forall (r:ins) (j:nat). reveal p r /\ j < SZ.v vi ==>
           S.index s (SZ.v wi * 256 + j) %~ (wB qF kF vF (SZ.v hd) (SZ.v cb) j (SZ.v wi) ((kq (SZ.v wi) (SZ.v vp)) + 1) r /. eT m)))
      decreases (256 - SZ.v !i)
    {
      let vi = !i;
      let ix = wi *^ 256sz +^ vi;
      let vix = kb +^ vi;
      let vv = vc.(vix);
      with s2. assert (partial |-> s2);
      let old_p = partial.(ix);
      po_ok p qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v wi) (kq (SZ.v wi) (SZ.v vp)) (SZ.v vi) vs old s2 s1 sp;
      bcast_all p sv 6291456 vF (SZ.v vix);
      let cv = C.cast_bf16_to_f32 vv;
      let mw = F32.mul_rn_ftz wt cv;
      let np = F32.fma_rn_ftz old_p diff mw;
      wB_succ qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v vi) (SZ.v wi) (kq (SZ.v wi) (SZ.v vp)) (SZ.v kb);
      pv_step p (wB qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v vi) (SZ.v wi) (kq (SZ.v wi) (SZ.v vp))) (esc qF kF (SZ.v hd) (SZ.v kb))
        (xat vF (SZ.v vix)) (wB qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v vi) (SZ.v wi) ((kq (SZ.v wi) (SZ.v vp)) + 1))
        old m old_p diff wt cv mw np;
      partial.(ix) <- np;
      i := vi +^ 1sz;
    };
    kq_step (SZ.v wi) (SZ.v vp);
    pp := vp +^ 16sz;
  };
  let vp = !pp;
  kf_lt (SZ.v wi) (SZ.v pos) (kq (SZ.v wi) (SZ.v vp));
  let vm = !mx;
  let vs = !sum;
  maxs.(wi) <- vm;
  totals.(wi) <- vs;
}

noextract let wall (p:ins -> prop) (qF kF vF:ins -> S.seq real) (hd cb pos:nat) (sp sm st:S.seq F32.t) : prop =
  S.length sp == 4096 /\ S.length sm == 16 /\ S.length st == 16 /\
  (forall (w:nat). w < 16 ==> wdone p qF kF vF hd cb pos sp sm st w)

inline_for_extraction noextract
fn ts_body (maxs totals factor : larray f32 16) (ts : ref f32) (vw : sz{SZ.v vw < 16}) (pos : sz{SZ.v pos < 2048}) (g : f32)
  (p:erased (ins -> prop)) (qF kF vF:erased (ins -> S.seq real)) (hd cb : sz) (sp : erased (S.seq f32))
  (#sm #st #sf : erased (S.seq f32)) (#vt : erased f32)
  preserves gpu ** maxs |-> sm ** totals |-> st
  requires factor |-> sf ** ts |-> vt ** pure (S.length sf == 16 /\
    wall p qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v pos) sp sm st /\
    (forall (r:ins). reveal p r ==> reveal vt %~ (tA qF kF (SZ.v hd) (SZ.v cb) (SZ.v pos) (SZ.v vw) r /. eT g)) /\
    (forall (w:nat). w < SZ.v vw /\ w <= SZ.v pos ==> S.index sf w %~ (eT (S.index sm w) /. eT g)))
  ensures exists* sf' vt'. factor |-> sf' ** ts |-> vt' ** pure (S.length sf' == 16 /\ S.length sm == 16 /\
    (forall (r:ins). reveal p r ==> vt' %~ (tA qF kF (SZ.v hd) (SZ.v cb) (SZ.v pos) (SZ.v vw + 1) r /. eT g)) /\
    (forall (w:nat). w < SZ.v vw + 1 /\ w <= SZ.v pos ==> S.index sf' w %~ (eT (S.index sm w) /. eT g)))
{
  pts_to_len factor;
  tA_succ qF kF (SZ.v hd) (SZ.v cb) (SZ.v pos) (SZ.v vw);
  if (vw <=^ pos) {
    let m = maxs.(vw);
    let d = sub_ftz m g;
    let f = FM.exp d;
    factor.(vw) <- f;
    let t = totals.(vw);
    let vt = !ts;
    let nt = F32.fma_rn_ftz t f vt;
    exp_diff m g d f;
    acc_step p (wA qF kF (SZ.v hd) (SZ.v cb) (SZ.v vw) (kf (SZ.v vw) (SZ.v pos)))
      (tA qF kF (SZ.v hd) (SZ.v cb) (SZ.v pos) (SZ.v vw)) (tA qF kF (SZ.v hd) (SZ.v cb) (SZ.v pos) (SZ.v vw + 1))
      m g t f vt nt;
    ts := nt;
  } else {
    ()
  }
}

inline_for_extraction noextract
fn os_body (partial : larray f32 4096) (factor : larray f32 16) (s : ref f32) (vw : sz{SZ.v vw < 16}) (vi : sz{SZ.v vi < 256})
  (pos : sz{SZ.v pos < 2048}) (g : f32)
  (p:erased (ins -> prop)) (qF kF vF:erased (ins -> S.seq real)) (hd cb : sz) (sm st : erased (S.seq f32))
  (#sp #sf : erased (S.seq f32)) (#vs : erased f32)
  preserves gpu ** partial |-> sp ** factor |-> sf
  requires s |-> vs ** pure (S.length sf == 16 /\
    wall p qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v pos) sp sm st /\
    (forall (w:nat). w < 16 /\ w <= SZ.v pos ==> S.index sf w %~ (eT (S.index sm w) /. eT g)) /\
    (forall (r:ins). reveal p r ==> reveal vs %~ (oB qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v vi) (SZ.v pos) (SZ.v vw) r /. eT g)))
  ensures exists* vs'. s |-> vs' ** pure (
    (forall (r:ins). reveal p r ==> vs' %~ (oB qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v vi) (SZ.v pos) (SZ.v vw + 1) r /. eT g)))
{
  pts_to_len factor; pts_to_len partial;
  oB_succ qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v vi) (SZ.v pos) (SZ.v vw);
  if (vw <=^ pos) {
    let ix = vw *^ 256sz +^ vi;
    assert (pure (SZ.v ix == SZ.v vw * 256 + SZ.v vi));
    let pv = partial.(ix);
    let f = factor.(vw);
    let vs = !s;
    let ns = F32.fma_rn_ftz pv f vs;
    acc_step p (wB qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v vi) (SZ.v vw) (kf (SZ.v vw) (SZ.v pos)))
      (oB qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v vi) (SZ.v pos) (SZ.v vw))
      (oB qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v vi) (SZ.v pos) (SZ.v vw + 1))
      (S.index sm (SZ.v vw)) g pv f vs ns;
    s := ns;
  } else {
    ()
  }
}

inline_for_extraction noextract
fn attn_head (qn : larray f32 2048) (proj : larray f32 4096) (hd : sz{SZ.v hd < 8}) (kc vc : larray bf16 6291456)
  (fa : sz{SZ.v fa < 6}) (pos : sz{SZ.v pos < 2048})
  (lanes : larray f32 32) (partial : larray f32 4096) (maxs totals factor : larray f32 16) (out : larray f32 2048)
  (p:erased (ins -> prop)) (qF kF vF projF:erased (ins -> S.seq real))
  (#sq #spj #sl #sp #sm #st #sf #so : erased (S.seq f32)) (#sk #sv : erased (S.seq bf16))
  preserves gpu ** qn |-> sq ** proj |-> spj ** kc |-> sk ** vc |-> sv
  requires lanes |-> sl ** partial |-> sp ** maxs |-> sm ** totals |-> st ** factor |-> sf ** out |-> so **
    pure (apxs p sq 2048 qF /\ apxb p sk 6291456 kF /\ apxb p sv 6291456 vF /\ apxs p spj 4096 projF)
  ensures exists* sl' sp' sm' st' sf' so'. lanes |-> sl' ** partial |-> sp' ** maxs |-> sm' ** totals |-> st' **
    factor |-> sf' ** out |-> so' ** pure (S.length so == 2048 /\ S.length so' == 2048 /\
    (forall (r:ins) (i:nat). reveal p r /\ i < 256 ==>
       S.index so' (SZ.v hd * 256 + i) %~ attO qF kF vF projF (SZ.v fa) (SZ.v pos) (SZ.v hd) i r) /\
    (forall (j:nat). j < 2048 /\ (j < SZ.v hd * 256 \/ SZ.v hd * 256 + 256 <= j) ==> S.index so' j == S.index so j))
{
  pts_to_len maxs; pts_to_len totals; pts_to_len factor; pts_to_len out; pts_to_len partial; pts_to_len proj; pts_to_len qn;
  let cb = fa *^ (SZ.uint32_to_sizet 1048576ul) +^ (hd /^ 4sz) *^ (SZ.uint32_to_sizet 524288ul);
  assert (pure (SZ.v cb == cbf (SZ.v fa) (SZ.v hd)));
  let mut wi = 0sz;
  while (!wi <^ 16sz)
    invariant exists* (vw:sz) sl1 sp1 sm1 st1. wi |-> vw ** lanes |-> sl1 ** partial |-> sp1 ** maxs |-> sm1 ** totals |-> st1 **
      pure (SZ.v vw <= 16 /\ S.length sp1 == 4096 /\ S.length sm1 == 16 /\ S.length st1 == 16 /\
        (forall (w:nat). w < SZ.v vw ==> wdone p qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v pos) sp1 sm1 st1 w))
    decreases (16 - SZ.v !wi)
  {
    let vw = !wi;
    with sp1 sm1 st1. assert (partial |-> sp1 ** maxs |-> sm1 ** totals |-> st1);
    window qn hd kc vc cb pos vw lanes partial maxs totals p qF kF vF;
    with sp2 sm2 st2. assert (partial |-> sp2 ** maxs |-> sm2 ** totals |-> st2);
    wdone_frame p qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v pos) sp1 sm1 st1 sp2 sm2 st2 (SZ.v vw) (SZ.v vw);
    wi := vw +^ 1sz;
  };
  with sp2 sm2 st2. assert (partial |-> sp2 ** maxs |-> sm2 ** totals |-> st2);
  assert (pure (wall p qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v pos) sp2 sm2 st2));
  let mut gm = FL.neg (FL.infinity #f32);
  let mut w2 = 0sz;
  while (!w2 <^ 16sz)
    invariant exists* (vw:sz) (vg:f32). w2 |-> vw ** gm |-> vg ** pure (SZ.v vw <= 16)
    decreases (16 - SZ.v !w2)
  {
    let vw = !w2;
    if (vw <=^ pos) {
      let m = maxs.(vw);
      let g = !gm;
      gm := FL.fmax g m;
    };
    w2 := vw +^ 1sz;
  };
  let g = !gm;
  zero_ok (); eT_pos g;
  tA_init p qF kF (SZ.v hd) (SZ.v cb) (SZ.v pos) g;
  let mut ts = zero #f32 #_;
  let mut w3 = 0sz;
  while (!w3 <^ 16sz)
    invariant exists* (vw:sz) (vt:f32) sf1. w3 |-> vw ** ts |-> vt ** factor |-> sf1 ** pure (SZ.v vw <= 16 /\ S.length sf1 == 16 /\
      (forall (r:ins). reveal p r ==> vt %~ (tA qF kF (SZ.v hd) (SZ.v cb) (SZ.v pos) (SZ.v vw) r /. eT g)) /\
      (forall (w:nat). w < SZ.v vw /\ w <= SZ.v pos ==> S.index sf1 w %~ (eT (S.index sm2 w) /. eT g)))
    decreases (16 - SZ.v !w3)
  {
    let vw = !w3;
    ts_body maxs totals factor ts vw pos g p qF kF vF hd cb sp2;
    w3 := vw +^ 1sz;
  };
  let vt = !ts;
  ta_done qF kF (SZ.v hd) (SZ.v cb) (SZ.v pos);
  let rcp = F32.rcp_approx_ftz vt;
  with sf2. assert (factor |-> sf2);
  let mut i = 0sz;
  while (!i <^ 256sz)
    invariant exists* (vi:sz) so1. i |-> vi ** out |-> so1 ** pure (SZ.v vi <= 256 /\ S.length so1 == 2048 /\
      (forall (r:ins) (j:nat). reveal p r /\ j < SZ.v vi ==>
         S.index so1 (SZ.v hd * 256 + j) %~ attO qF kF vF projF (SZ.v fa) (SZ.v pos) (SZ.v hd) j r) /\
      (forall (j:nat). j < 2048 /\ (j < SZ.v hd * 256 \/ SZ.v hd * 256 + SZ.v vi <= j) ==> S.index so1 j == S.index so j))
    decreases (256 - SZ.v !i)
  {
    let vi = !i;
    oB_init p qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v vi) (SZ.v pos) g;
    let mut s = zero #f32 #_;
    let mut w4 = 0sz;
    while (!w4 <^ 16sz)
      invariant exists* (vw:sz) (vs:f32). w4 |-> vw ** s |-> vs ** pure (SZ.v vw <= 16 /\
        (forall (r:ins). reveal p r ==> vs %~ (oB qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v vi) (SZ.v pos) (SZ.v vw) r /. eT g)))
      decreases (16 - SZ.v !w4)
    {
      let vw = !w4;
      os_body partial factor s vw vi pos g p qF kF vF hd cb sm2 st2;
      w4 := vw +^ 1sz;
    };
    let vs = !s;
    ob_done qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v vi) (SZ.v pos);
    let gi = hd *^ 512sz +^ 256sz +^ vi;
    assert (pure (SZ.v gi == SZ.v hd * 512 + 256 + SZ.v vi));
    let gt = proj.(gi);
    let sg = sigm gt p (xat projF (SZ.v gi));
    let o1 = F32.mul_rn_ftz vs rcp;
    let o2 = F32.mul_rn_ftz o1 sg;
    fin_step p (obF qF kF vF (SZ.v hd) (SZ.v cb) (SZ.v vi) (SZ.v pos)) (taF qF kF (SZ.v hd) (SZ.v cb) (SZ.v pos))
      (sgF projF (SZ.v hd) (SZ.v vi)) g vs vt rcp o1 sg o2;
    let oi = hd *^ 256sz +^ vi;
    assert (pure (SZ.v oi == SZ.v hd * 256 + SZ.v vi));
    out.(oi) <- o2;
    i := vi +^ 1sz;
  }
}

inline_for_extraction noextract
fn attn_kernel (qn : larray f32 2048) (proj : larray f32 4096) (kc vc : larray bf16 6291456)
  (fa : sz{SZ.v fa < 6}) (pos : sz{SZ.v pos < 2048})
  (lanes : larray f32 32) (partial : larray f32 4096) (maxs totals factor : larray f32 16) (out : larray f32 2048)
  (p:erased (ins -> prop)) (qF kF vF projF:erased (ins -> S.seq real))
  (#sq #spj #sl #sp #sm #st #sf #so : erased (S.seq f32)) (#sk #sv : erased (S.seq bf16))
  preserves gpu ** qn |-> sq ** proj |-> spj ** kc |-> sk ** vc |-> sv
  requires lanes |-> sl ** partial |-> sp ** maxs |-> sm ** totals |-> st ** factor |-> sf ** out |-> so **
    pure (apxs p sq 2048 qF /\ apxb p sk 6291456 kF /\ apxb p sv 6291456 vF /\ apxs p spj 4096 projF)
  ensures exists* sl' sp' sm' st' sf' so'. lanes |-> sl' ** partial |-> sp' ** maxs |-> sm' ** totals |-> st' **
    factor |-> sf' ** out |-> so' ** pure (S.length so' == 2048 /\ apxs p so' 2048 (attI qF kF vF projF (SZ.v fa) (SZ.v pos)))
{
  pts_to_len out;
  let mut h = 0sz;
  while (!h <^ 8sz)
    invariant exists* (vh:sz) sl1 sp1 sm1 st1 sf1 so1. h |-> vh ** lanes |-> sl1 ** partial |-> sp1 ** maxs |-> sm1 ** totals |-> st1 **
      factor |-> sf1 ** out |-> so1 ** pure (SZ.v vh <= 8 /\ S.length so1 == 2048 /\
      apxs p so1 (SZ.v vh * 256) (attI qF kF vF projF (SZ.v fa) (SZ.v pos)))
    decreases (8 - SZ.v !h)
  {
    let vh = !h;
    with so1. assert (out |-> so1);
    attn_head qn proj vh kc vc fa pos lanes partial maxs totals factor out p qF kF vF projF;
    with so2. assert (out |-> so2);
    head_step p qF kF vF projF (SZ.v fa) (SZ.v pos) (SZ.v vh) so1 so2;
    h := vh +^ 1sz;
  }
}


(**** Section E ****)
#set-options "--fuel 0 --ifuel 0 --z3rlimit 20"

(*** Ideal chain of the attention layer *)
noextract let rxF (r:ins) : S.seq real = r.rx

noextract unfold let iN1 (b:int) = rmsF rxF b
noextract unfold let iPr (b:int) = mvF (b + 11012096) 4096 1024 (iN1 b)
noextract unfold let iZk (b:int) = mvF (b + 11012096 + 4194304) 512 1024 (iN1 b)
noextract unfold let iZv (b:int) = mvF (b + 11012096 + 4718592) 512 1024 (iN1 b)
noextract unfold let iQ (b:int) = hsq (iPr b) 512 (b + 11012096 + 5242880) 2048
noextract unfold let iKn (b:int) = hsq (iZk b) 256 (b + 11012096 + 5243136) 512
noextract unfold let iK (b fa pos:int) = kvI fa pos rkF (iKn b)
noextract unfold let iV (b fa pos:int) = kvI fa pos rvF (iZv b)
noextract unfold let iAt (b:int) (fa pos:nat) = attI (iQ b) (iK b fa pos) (iV b fa pos) (iPr b) fa pos
noextract unfold let iH1 (b:int) (fa pos:nat) = resF rxF (iAt b fa pos) (b + 11012096 + 5243392) 2048
noextract unfold let iN2 (b:int) (fa pos:nat) = rmsF (iH1 b fa pos) (b + 1024)
noextract unfold let iGt (b:int) (fa pos:nat) = mvF (b + 2048) 3584 1024 (iN2 b fa pos)
noextract unfold let iUp (b:int) (fa pos:nat) = mvF (b + 3672064) 3584 1024 (iN2 b fa pos)

noextract let actF (gF uF:ins -> S.seq real) (r:ins) : S.seq real =
  S.init 3584 (fun (i:nat{i < 3584}) -> silu (at (gF r) i) *. at (uF r) i)

noextract unfold let iAc (b:int) (fa pos:nat) = actF (iGt b fa pos) (iUp b fa pos)
noextract unfold let iXo (b:int) (fa pos:nat) = resF (iH1 b fa pos) (iAc b fa pos) (b + 7342080) 3584

(*** Projections and head norms *)
noextract let e_hin (b:int) (r:ins) : Lemma (iN1 b r == at_hin opt_numerics r.rw b r.rx) = ()
noextract let e_qraw (b:int) (r:ins) : Lemma (iPr b r == at_qraw opt_numerics r.rw b r.rx) = e_hin b r
noextract let e_kraw (b:int) (r:ins) : Lemma (iZk b r == at_kraw opt_numerics r.rw b r.rx) = e_hin b r
noextract let e_v (b:int) (r:ins) : Lemma (iZv b r == at_v opt_numerics r.rw b r.rx) = e_hin b r

noextract let e_q (b:int) (r:ins) : Lemma (iQ b r == at_q opt_numerics r.rw b r.rx r.rc r.rs) =
  e_qraw b r;
  norm_spec [delta_only [`%at_q]; zeta] (at_q opt_numerics r.rw b r.rx r.rc r.rs);
  norm_spec [delta_only [`%hsq]] (hsq (iPr b) 512 (b + 11012096 + 5242880) 2048 r);
  assert (S.equal (iQ b r) (at_q opt_numerics r.rw b r.rx r.rc r.rs))

noextract let e_k (b:int) (r:ins) : Lemma (iKn b r == at_k opt_numerics r.rw b r.rx r.rc r.rs) =
  e_kraw b r;
  norm_spec [delta_only [`%at_k]; zeta] (at_k opt_numerics r.rw b r.rx r.rc r.rs);
  norm_spec [delta_only [`%hsq]] (hsq (iZk b) 256 (b + 11012096 + 5243136) 512 r);
  assert (S.equal (iKn b r) (at_k opt_numerics r.rw b r.rx r.rc r.rs))

(*** Attention scores and softmax *)
noextract let sS (b:int) (fa pos:nat) (r:ins) (hd p:nat) : real =
  at_score opt_numerics r.rw b r.rx r.rc r.rs pos (opt_hk fa r.rk) hd p
noextract let sG (b:int) (fa pos:nat) (r:ins) (hd p:nat) (i:nat) : real =
  at (at_q opt_numerics r.rw b r.rx r.rc r.rs) (hd * 256 + i) *.
  hist_or (opt_hk fa r.rk) (at_k opt_numerics r.rw b r.rx r.rc r.rs) pos p ((hd / 4) * 256 + i)

noextract let e_score_def (b:int) (fa pos:nat) (r:ins) (hd p:nat)
  : Lemma (sS b fa pos r hd p == vsum 256 (sG b fa pos r hd p) *. 0.0625R)
  = norm_spec [delta_only [`%at_score]; zeta] (at_score opt_numerics r.rw b r.rx r.rc r.rs pos (opt_hk fa r.rk) hd p);
    vsum_ext 256 (fun i -> at (at_q opt_numerics r.rw b r.rx r.rc r.rs) (hd * 256 + i) *.
                           hist_or (opt_hk fa r.rk) (at_k opt_numerics r.rw b r.rx r.rc r.rs) pos p ((hd / 4) * 256 + i))
      (sG b fa pos r hd p)

#push-options "--z3rlimit 60"
noextract let e_score (b:int) (fa:nat{fa < 6}) (pos:nat{pos < 2048}) (r:ins) (hd:nat{hd < 8}) (p:nat{p <= pos})
  : Lemma (scI (iQ b) (iK b fa pos) hd (kbi (cbf fa hd) p) r == sS b fa pos r hd p)
  = e_score_def b fa pos r hd p; e_q b r; e_k b r; consts ();
    introduce forall (i:nat). i < 256 ==> scg (iQ b) (iK b fa pos) hd (kbi (cbf fa hd) p) r i == sG b fa pos r hd p i
    with introduce _ ==> _ with (
      divmod256 (hd / 4) i;
      kv_read fa pos r.rk (iKn b r) p ((hd / 4) * 256 + i));
    vsum_ext 256 (scg (iQ b) (iK b fa pos) hd (kbi (cbf fa hd) p) r) (sG b fa pos r hd p)
#pop-options

noextract let eE (b:int) (fa pos:nat) (r:ins) (hd p:nat) : real = Exp.exp (sS b fa pos r hd p)

noextract let e_prob_def (b:int) (fa pos:nat) (r:ins) (hd p:nat)
  : Lemma (at_prob opt_numerics r.rw b r.rx r.rc r.rs pos (opt_hk fa r.rk) hd p ==
           rdiv (eE b fa pos r hd p) (vsum (pos + 1) (eE b fa pos r hd)))
  = norm_spec [delta_only [`%at_prob]] (at_prob opt_numerics r.rw b r.rx r.rc r.rs pos (opt_hk fa r.rk) hd p);
    vsum_ext (pos + 1) (fun p' -> Exp.exp (at_score opt_numerics r.rw b r.rx r.rc r.rs pos (opt_hk fa r.rk) hd p'))
      (eE b fa pos r hd)

noextract let gB (b:int) (fa pos:nat) (r:ins) (hd e p:nat) : real =
  at_prob opt_numerics r.rw b r.rx r.rc r.rs pos (opt_hk fa r.rk) hd p *.
  hist_or (opt_hk fa r.rv) (at_v opt_numerics r.rw b r.rx) pos p ((hd / 4) * 256 + e)

noextract let e_gated_def (b:int) (fa pos:nat) (r:ins) (c:nat{c < 2048})
  : Lemma (S.index (at_gated opt_numerics r.rw b r.rx r.rc r.rs pos (opt_hk fa r.rk) (opt_hk fa r.rv)) c ==
           vsum (pos + 1) (gB b fa pos r (c / 256) (c % 256)) *.
           sigmoid (at (at_qraw opt_numerics r.rw b r.rx) ((c / 256) * 512 + 256 + c % 256)))
  = norm_spec [delta_only [`%at_gated]; zeta] (at_gated opt_numerics r.rw b r.rx r.rc r.rs pos (opt_hk fa r.rk) (opt_hk fa r.rv));
    vsum_ext (pos + 1) (fun p -> at_prob opt_numerics r.rw b r.rx r.rc r.rs pos (opt_hk fa r.rk) (c / 256) p *.
                                 hist_or (opt_hk fa r.rv) (at_v opt_numerics r.rw b r.rx) pos p (((c / 256) / 4) * 256 + c % 256))
      (gB b fa pos r (c / 256) (c % 256))

noextract let alg_w (x ta v:real)
  : Lemma (requires ta >. 0.0R) (ensures rdiv x ta *. v == (1.0R /. ta) *. (x *. v))
  = ()

noextract let alg_o (o ta g:real)
  : Lemma (requires ta >. 0.0R) (ensures rdiv o ta *. g == ((1.0R /. ta) *. o) *. g)
  = assert (o /. ta == o *. (1.0R /. ta))

#push-options "--z3rlimit 80"
noextract let e_gated (b:int) (fa:nat{fa < 6}) (pos:nat{pos < 2048}) (r:ins) (c:nat{c < 2048})
  : Lemma (S.index (iAt b fa pos r) c ==
           S.index (at_gated opt_numerics r.rw b r.rx r.rc r.rs pos (opt_hk fa r.rk) (opt_hk fa r.rv)) c)
  = let hd = c / 256 in
    let e = c % 256 in
    let cb = cbf fa hd in
    let qF = iQ b in
    let kF = iK b fa pos in
    let vF = iV b fa pos in
    e_gated_def b fa pos r c;
    e_qraw b r; e_v b r;
    ta_done qF kF hd cb pos;
    let ta = taF qF kF hd cb pos r in
    introduce forall (p:nat). p < pos + 1 ==> eS qF kF hd cb r p == eE b fa pos r hd p
      with introduce _ ==> _ with e_score b fa pos r hd p;
    vsum_ext (pos + 1) (eS qF kF hd cb r) (eE b fa pos r hd);
    introduce forall (p:nat). p < pos + 1 ==> gB b fa pos r hd e p == fscale (1.0R /. ta) (eSV qF kF vF hd cb e r) p
      with introduce _ ==> _ with (
        e_prob_def b fa pos r hd p;
        e_score b fa pos r hd p;
        divmod256 (hd / 4) e;
        kv_read fa pos r.rv (iZv b r) p ((hd / 4) * 256 + e);
        alg_w (eE b fa pos r hd p) ta (at (vF r) (kbi cb p + e)));
    vsum_ext (pos + 1) (gB b fa pos r hd e) (fscale (1.0R /. ta) (eSV qF kF vF hd cb e r));
    vsum_scale (pos + 1) (1.0R /. ta) (eSV qF kF vF hd cb e r);
    alg_o (obF qF kF vF hd cb e pos r) ta (sgF (iPr b) hd e r)
#pop-options

noextract let e_attn (b:int) (fa:nat{fa < 6}) (pos:nat{pos < 2048}) (r:ins)
  : Lemma (iAt b fa pos r == at_gated opt_numerics r.rw b r.rx r.rc r.rs pos (opt_hk fa r.rk) (opt_hk fa r.rv))
  = introduce forall (c:nat). c < 2048 ==>
      S.index (iAt b fa pos r) c ==
      S.index (at_gated opt_numerics r.rw b r.rx r.rc r.rs pos (opt_hk fa r.rk) (opt_hk fa r.rv)) c
    with introduce _ ==> _ with e_gated b fa pos r c;
    assert (S.equal (iAt b fa pos r) (at_gated opt_numerics r.rw b r.rx r.rc r.rs pos (opt_hk fa r.rk) (opt_hk fa r.rv)))

(*** Residuals and MLP *)
noextract let e_x1 (b:int) (fa:nat{fa < 6}) (pos:nat{pos < 2048}) (r:ins)
  : Lemma (iH1 b fa pos r == at_x1 opt_numerics r.rw b r.rx r.rc r.rs pos (opt_hk fa r.rk) (opt_hk fa r.rv))
  = e_attn b fa pos r

noextract let e_act (b:int) (fa:nat{fa < 6}) (pos:nat{pos < 2048}) (r:ins)
  : Lemma (iAc b fa pos r == mlp_act opt_numerics r.rw b (iH1 b fa pos r))
  = norm_spec [delta_only [`%mlp_act]; zeta] (mlp_act opt_numerics r.rw b (iH1 b fa pos r));
    assert (S.equal (iAc b fa pos r) (mlp_act opt_numerics r.rw b (iH1 b fa pos r)))

noextract let e_out (b:int) (fa:nat{fa < 6}) (pos:nat{pos < 2048}) (r:ins)
  : Lemma (iXo b fa pos r == at_out opt_numerics r.rw b r.rx r.rc r.rs pos (opt_hk fa r.rk) (opt_hk fa r.rv))
  = e_act b fa pos r; e_x1 b fa pos r

(*** Admissible inputs and the final relation *)
noextract let pI (sw:S.seq BF.t) (sx:S.seq F32.t) (sk sv:S.seq BF.t) (sc ss:S.seq F32.t) (r:ins) : prop =
  sw %~ r.rw /\ sx %~ r.rx /\ sk %~ r.rk /\ sv %~ r.rv /\ sc %~ r.rc /\ ss %~ r.rs

noextract let p_facts (sw:S.seq BF.t) (sx:S.seq F32.t) (sk sv:S.seq BF.t) (sc ss:S.seq F32.t)
  : Lemma (requires S.length sx == 1024)
          (ensures wok (pI sw sx sk sv sc ss) sw /\ apxs (pI sw sx sk sv sc ss) sx 1024 rxF /\
                   csok (pI sw sx sk sv sc ss) sc ss /\
                   (forall (r:ins). pI sw sx sk sv sc ss r ==> sk %~ r.rk /\ sv %~ r.rv))
  = ()

noextract let sq1 (s:S.seq F32.t) (t:S.seq real)
  : Lemma (requires S.length s == S.length t /\ (forall (j:nat). j < S.length s ==> S.index s j %~ at t j))
          (ensures s %~ t)
  = assert (seq_approximates s t)

noextract let sq2 (s:S.seq BF.t) (t:S.seq real)
  : Lemma (requires S.length s == S.length t /\ (forall (j:nat). j < S.length s ==> S.index s j %~ at t j))
          (ensures s %~ t)
  = assert (seq_approximates s t)

noextract let apxs_seq (p:ins -> prop) (s:S.seq F32.t) (n:nat) (f:ins -> S.seq real) (r:ins)
  : Lemma (requires apxs p s n f /\ S.length s == n /\ S.length (f r) == n /\ p r) (ensures s %~ f r)
  = sq1 s (f r)

noextract let apxb_seq (p:ins -> prop) (s:S.seq BF.t) (n:nat) (f:ins -> S.seq real) (r:ins)
  : Lemma (requires apxb p s n f /\ S.length s == n /\ S.length (f r) == n /\ p r) (ensures s %~ f r)
  = sq2 s (f r)

#push-options "--z3rlimit 60"
noextract let final_rel (sw:S.seq BF.t) (sx:S.seq F32.t) (sk sv:S.seq BF.t) (sc ss:S.seq F32.t)
  (b:nat) (fa:nat{fa < 6}) (pos:nat{pos < 2048}) (sxo:S.seq F32.t) (sko svo:S.seq BF.t)
  : Lemma (requires S.length sxo == 1024 /\ apxs (pI sw sx sk sv sc ss) sxo 1024 (iXo b fa pos) /\
                    kvpost (pI sw sx sk sv sc ss) fa pos sko svo sk sv (iKn b) (iZv b))
          (ensures optimized_attention_decoder_relation sw sx sk sv sc ss b fa pos sxo sko svo)
  = let p = pI sw sx sk sv sc ss in
    introduce forall (rw rx rk rv rc rs:S.seq real).
      sw %~ rw /\ sx %~ rx /\ sk %~ rk /\ sv %~ rv /\ sc %~ rc /\ ss %~ rs ==>
      sxo %~ at_out opt_numerics rw b rx rc rs pos (opt_hk fa rk) (opt_hk fa rv) /\
      sko %~ opt_kv_write fa pos rk (at_k opt_numerics rw b rx rc rs) /\
      svo %~ opt_kv_write fa pos rv (at_v opt_numerics rw b rx)
    with introduce _ ==> _ with (
      let r : ins = { rw = rw; rx = rx; rk = rk; rv = rv; rc = rc; rs = rs } in
      assert (p r);
      e_out b fa pos r; e_k b r; e_v b r;
      apxs_seq p sxo 1024 (iXo b fa pos) r;
      apxb_seq p sko 6291456 (iK b fa pos) r;
      apxb_seq p svo 6291456 (iV b fa pos) r)
#pop-options


(**** Section K5 ****)
#set-options "--fuel 0 --ifuel 0 --z3rlimit 20"

instance send_global_array_contents
  (#a : Type0) (arr : array a{is_global arr}) (#f : perm) (contents : S.seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

(*** SwiGLU activation *)
noextract let sgI (gF:ins -> S.seq real) (i:nat) (r:ins) : real = sigmoid (at (gF r) i)

noextract let act_at (gF uF:ins -> S.seq real) (i:nat{i < 3584})
  : Lemma (forall (r:ins). at (actF gF uF r) i == mulF (mulF (xat gF i) (sgI gF i)) (xat uF i) r)
  = ()

inline_for_extraction noextract
fn act_kernel (gate up act : larray f32 3584) (p:erased (ins -> prop)) (gF uF:erased (ins -> S.seq real))
  (#sg #su #sa : erased (S.seq f32))
  preserves gpu ** gate |-> sg ** up |-> su
  requires act |-> sa ** pure (apxs p sg 3584 gF /\ apxs p su 3584 uF)
  ensures exists* sa'. act |-> sa' ** pure (S.length sa' == 3584 /\ apxs p sa' 3584 (actF gF uF))
{
  pts_to_len act; pts_to_len gate; pts_to_len up;
  let mut i = 0sz;
  while (!i <^ 3584sz)
    invariant exists* (vi:sz) s. i |-> vi ** act |-> s **
      pure (SZ.v vi <= 3584 /\ S.length s == 3584 /\ apxs p s (SZ.v vi) (actF gF uF))
    decreases (3584 - SZ.v !i)
  {
    let vi = !i;
    let g = gate.(vi);
    let u = up.(vi);
    let sg = sigm g p (xat gF (SZ.v vi));
    let m1 = F32.mul_rn_ftz g sg;
    let a = F32.mul_rn_ftz m1 u;
    act_at gF uF (SZ.v vi);
    mul_lift p g sg m1 (xat gF (SZ.v vi)) (sgI gF (SZ.v vi)) (mulF (xat gF (SZ.v vi)) (sgI gF (SZ.v vi)));
    mul_lift p m1 u a (mulF (xat gF (SZ.v vi)) (sgI gF (SZ.v vi))) (xat uF (SZ.v vi))
      (fun r -> at (actF gF uF r) (SZ.v vi));
    act.(vi) <- a;
    i := vi +^ 1sz;
  }
}

(*** Host-side facts *)
noextract let szb (x:SZ.t) : Lemma (SZ.v x < 4294967296) = FStar.UInt.pow2_values 32

noextract let cast_ok (x:I64.t)
  : Lemma (requires 0 <= I64.v x /\ I64.v x <= 4294967295) (ensures U32.v (Cast.int64_to_uint32 x) == I64.v x)
  = FStar.UInt.pow2_values 32;
    FStar.Math.Lemmas.small_mod (I64.v x) (pow2 32)

noextract let guard_fail (sw:S.seq BF.t) (sx:S.seq F32.t) (sk sv:S.seq BF.t) (sc ss:S.seq F32.t)
  (base fa pos:int) (sxo:S.seq F32.t) (sko svo:S.seq BF.t)
  : Lemma (requires S.length sw < 4294967296 /\
                    (base < 0 \/ base > 4294967295 \/ fa < 0 \/ fa >= 6 \/ pos < 0 \/ pos >= 2048 \/
                     base > S.length sw \/ S.length sw - base < 18352640))
          (ensures optimized_attention_decoder_relation sw sx sk sv sc ss base fa pos sxo sko svo)
  = ()

(*** Host *)
#push-options "--z3rlimit 100"
fn kplay_run
  (w : array bf16{is_global w}) (x : larray f32 1024{is_global x})
  (k : larray bf16 6291456{is_global k}) (v : larray bf16 6291456{is_global v})
  (cs : larray f32 32{is_global cs}) (sn : larray f32 32{is_global sn})
  (xo : larray f32 1024{is_global xo})
  (ko : larray bf16 6291456{is_global ko}) (vo : larray bf16 6291456{is_global vo})
  (wn : SZ.t) (base fa pos : I64.t)
  (#sw : erased (S.seq bf16)) (#sx : erased (S.seq f32))
  (#sk #sv : erased (S.seq bf16)) (#sc #ss : erased (S.seq f32))
  preserves cpu ** on gpu_loc (w |-> sw) ** on gpu_loc (x |-> sx) **
    on gpu_loc (k |-> sk) ** on gpu_loc (v |-> sv) **
    on gpu_loc (cs |-> sc) ** on gpu_loc (sn |-> ss)
  requires pure (SZ.v wn == S.length sw /\ S.length sx == 1024 /\ S.length sk == 6291456 /\ S.length sv == 6291456 /\ S.length sc == 32 /\ S.length ss == 32) **
    (exists* s. on gpu_loc (xo |-> s)) ** (exists* s. on gpu_loc (ko |-> s)) ** (exists* s. on gpu_loc (vo |-> s))
  ensures exists* sx' sk' sv'.
    on gpu_loc (xo |-> sx') ** on gpu_loc (ko |-> sk') ** on gpu_loc (vo |-> sv') **
    pure (optimized_attention_decoder_relation sw sx sk sv sc ss (I64.v base) (I64.v fa) (I64.v pos) sx' sk' sv')
{
  szb wn;
  if (I64.lt base 0L || I64.gt base 4294967295L || I64.lt fa 0L || I64.gte fa 6L || I64.lt pos 0L || I64.gte pos 2048L) {
    with s1. assert on gpu_loc (xo |-> s1);
    with s2. assert on gpu_loc (ko |-> s2);
    with s3. assert on gpu_loc (vo |-> s3);
    guard_fail sw sx sk sv sc ss (I64.v base) (I64.v fa) (I64.v pos) s1 s2 s3;
  } else {
    cast_ok base; cast_ok fa; cast_ok pos;
    let b = SZ.uint32_to_sizet (Cast.int64_to_uint32 base);
    let f = SZ.uint32_to_sizet (Cast.int64_to_uint32 fa);
    let p = SZ.uint32_to_sizet (Cast.int64_to_uint32 pos);
    if (b >^ wn || wn -^ b <^ (SZ.uint32_to_sizet 18352640ul)) {
      with s1. assert on gpu_loc (xo |-> s1);
      with s2. assert on gpu_loc (ko |-> s2);
      with s3. assert on gpu_loc (vo |-> s3);
      guard_fail sw sx sk sv sc ss (I64.v base) (I64.v fa) (I64.v pos) s1 s2 s3;
    } else {
      p_facts sw sx sk sv sc ss;
      memcpy_device_to_device ko k (SZ.uint32_to_sizet 6291456ul);
      memcpy_device_to_device vo v (SZ.uint32_to_sizet 6291456ul);
      let norm1 = gpu_array_alloc #f32 1024sz;
      let lanes = gpu_array_alloc #f32 32sz;
      let warps = gpu_array_alloc #f32 32sz;
      let proj = gpu_array_alloc #f32 4096sz;
      let zk = gpu_array_alloc #f32 512sz;
      let zv = gpu_array_alloc #f32 512sz;
      let qn = gpu_array_alloc #f32 2048sz;
      let kn = gpu_array_alloc #f32 512sz;
      let partial = gpu_array_alloc #f32 4096sz;
      let maxs = gpu_array_alloc #f32 16sz;
      let totals = gpu_array_alloc #f32 16sz;
      let factor = gpu_array_alloc #f32 16sz;
      let attn = gpu_array_alloc #f32 2048sz;
      let h1 = gpu_array_alloc #f32 1024sz;
      let norm2 = gpu_array_alloc #f32 1024sz;
      let gate = gpu_array_alloc #f32 3584sz;
      let up = gpu_array_alloc #f32 3584sz;
      let act = gpu_array_alloc #f32 3584sz;
      let mix = b +^ (SZ.uint32_to_sizet 11012096ul);
      assert (pure (SZ.v mix == SZ.v b + 11012096));
      with s1. assert on gpu_loc (norm1 |-> s1);
      with s2. assert on gpu_loc (lanes |-> s2);
      with s3. assert on gpu_loc (warps |-> s3);
      launch_kernel_1 (fun () -> rmsnorm_kernel w b x norm1 lanes warps (pI sw sx sk sv sc ss) rxF #sw #sx #s1 #s2 #s3);
      with s1. assert on gpu_loc (norm1 |-> s1);
      with s2. assert on gpu_loc (lanes |-> s2);
      with s4. assert on gpu_loc (proj |-> s4);
      launch_kernel_1 (fun () -> matvec_kernel w mix 4096sz norm1 proj lanes (pI sw sx sk sv sc ss) (iN1 (SZ.v b)) #sw #s1 #s4 #s2);
      with s2. assert on gpu_loc (lanes |-> s2);
      with s4. assert on gpu_loc (zk |-> s4);
      launch_kernel_1 (fun () -> matvec_kernel w (mix +^ (SZ.uint32_to_sizet 4194304ul)) 512sz norm1 zk lanes (pI sw sx sk sv sc ss) (iN1 (SZ.v b)) #sw #s1 #s4 #s2);
      with s2. assert on gpu_loc (lanes |-> s2);
      with s4. assert on gpu_loc (zv |-> s4);
      launch_kernel_1 (fun () -> matvec_kernel w (mix +^ (SZ.uint32_to_sizet 4718592ul)) 512sz norm1 zv lanes (pI sw sx sk sv sc ss) (iN1 (SZ.v b)) #sw #s1 #s4 #s2);
      with s2. assert on gpu_loc (lanes |-> s2);
      with s4. assert on gpu_loc (proj |-> s4);
      with s5. assert on gpu_loc (zk |-> s5);
      with s6. assert on gpu_loc (qn |-> s6);
      with s7. assert on gpu_loc (kn |-> s7);
      launch_kernel_1 (fun () -> heads_kernel w mix proj zk cs sn qn kn lanes (pI sw sx sk sv sc ss) (iPr (SZ.v b)) (iZk (SZ.v b)) #sw #s4 #s5 #sc #ss #s6 #s7 #s2);
      with s7. assert on gpu_loc (kn |-> s7);
      with s8. assert on gpu_loc (zv |-> s8);
      with s9. assert on gpu_loc (ko |-> s9);
      with s10. assert on gpu_loc (vo |-> s10);
      launch_kernel_1 (fun () -> kv_kernel kn zv ko vo f p (pI sw sx sk sv sc ss) (iKn (SZ.v b)) (iZv (SZ.v b)) #s7 #s8 #s9 #s10);
      with s2. assert on gpu_loc (lanes |-> s2);
      with s6. assert on gpu_loc (qn |-> s6);
      with s9. assert on gpu_loc (ko |-> s9);
      with s10. assert on gpu_loc (vo |-> s10);
      with s11. assert on gpu_loc (partial |-> s11);
      with s12. assert on gpu_loc (maxs |-> s12);
      with s13. assert on gpu_loc (totals |-> s13);
      with s14. assert on gpu_loc (factor |-> s14);
      with s15. assert on gpu_loc (attn |-> s15);
      launch_kernel_1 (fun () -> attn_kernel qn proj ko vo f p lanes partial maxs totals factor attn
        (pI sw sx sk sv sc ss) (iQ (SZ.v b)) (iK (SZ.v b) (SZ.v f) (SZ.v p)) (iV (SZ.v b) (SZ.v f) (SZ.v p)) (iPr (SZ.v b))
        #s6 #s4 #s2 #s11 #s12 #s13 #s14 #s15 #s9 #s10);
      with s2. assert on gpu_loc (lanes |-> s2);
      with s15. assert on gpu_loc (attn |-> s15);
      with s16. assert on gpu_loc (h1 |-> s16);
      launch_kernel_1 (fun () -> residual_kernel w (mix +^ (SZ.uint32_to_sizet 5243392ul)) 2048sz attn x h1 lanes
        (pI sw sx sk sv sc ss) (iAt (SZ.v b) (SZ.v f) (SZ.v p)) rxF #sw #s15 #sx #s16 #s2);
      with s2. assert on gpu_loc (lanes |-> s2);
      with s3. assert on gpu_loc (warps |-> s3);
      with s16. assert on gpu_loc (h1 |-> s16);
      with s17. assert on gpu_loc (norm2 |-> s17);
      launch_kernel_1 (fun () -> rmsnorm_kernel w (b +^ 1024sz) h1 norm2 lanes warps
        (pI sw sx sk sv sc ss) (iH1 (SZ.v b) (SZ.v f) (SZ.v p)) #sw #s16 #s17 #s2 #s3);
      with s2. assert on gpu_loc (lanes |-> s2);
      with s17. assert on gpu_loc (norm2 |-> s17);
      with s18. assert on gpu_loc (gate |-> s18);
      launch_kernel_1 (fun () -> matvec_kernel w (b +^ 2048sz) 3584sz norm2 gate lanes
        (pI sw sx sk sv sc ss) (iN2 (SZ.v b) (SZ.v f) (SZ.v p)) #sw #s17 #s18 #s2);
      with s2. assert on gpu_loc (lanes |-> s2);
      with s19. assert on gpu_loc (up |-> s19);
      launch_kernel_1 (fun () -> matvec_kernel w (b +^ (SZ.uint32_to_sizet 3672064ul)) 3584sz norm2 up lanes
        (pI sw sx sk sv sc ss) (iN2 (SZ.v b) (SZ.v f) (SZ.v p)) #sw #s17 #s19 #s2);
      with s18. assert on gpu_loc (gate |-> s18);
      with s19. assert on gpu_loc (up |-> s19);
      with s20. assert on gpu_loc (act |-> s20);
      launch_kernel_1 (fun () -> act_kernel gate up act
        (pI sw sx sk sv sc ss) (iGt (SZ.v b) (SZ.v f) (SZ.v p)) (iUp (SZ.v b) (SZ.v f) (SZ.v p)) #s18 #s19 #s20);
      with s2. assert on gpu_loc (lanes |-> s2);
      with s20. assert on gpu_loc (act |-> s20);
      with s21. assert on gpu_loc (xo |-> s21);
      launch_kernel_1 (fun () -> residual_kernel w (b +^ (SZ.uint32_to_sizet 7342080ul)) 3584sz act h1 xo lanes
        (pI sw sx sk sv sc ss) (iAc (SZ.v b) (SZ.v f) (SZ.v p)) (iH1 (SZ.v b) (SZ.v f) (SZ.v p)) #sw #s20 #s16 #s21 #s2);
      with s21. assert on gpu_loc (xo |-> s21);
      with s9. assert on gpu_loc (ko |-> s9);
      with s10. assert on gpu_loc (vo |-> s10);
      final_rel sw sx sk sv sc ss (SZ.v b) (SZ.v f) (SZ.v p) s21 s9 s10;
      gpu_array_free norm1;
      gpu_array_free lanes;
      gpu_array_free warps;
      gpu_array_free proj;
      gpu_array_free zk;
      gpu_array_free zv;
      gpu_array_free qn;
      gpu_array_free kn;
      gpu_array_free partial;
      gpu_array_free maxs;
      gpu_array_free totals;
      gpu_array_free factor;
      gpu_array_free attn;
      gpu_array_free h1;
      gpu_array_free norm2;
      gpu_array_free gate;
      gpu_array_free up;
      gpu_array_free act;
    }
  }
}
#pop-options
