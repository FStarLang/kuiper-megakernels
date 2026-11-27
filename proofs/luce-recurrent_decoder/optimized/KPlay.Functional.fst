module KPlay.Functional

#lang-pulse

open FStar.Real
open KPlay.SharedSpec
open Kuiper
module S = FStar.Seq
module SZ = Kuiper.SizeT
module F32 = Kuiper.Float32
module BF = Kuiper.BFloat16
module C = Kuiper.Float.Casts
module Exp = FStar.Math.Exp
module Sqrt = FStar.Math.Sqrt
module Lit = Kuiper.Float32.Literals
module CF = Kuiper.Float32.ConstantFold
module Cert = Kuiper.Float32.ConstantFold.Certificates
module RO = Kuiper.Float.Realops
module Fast = Kuiper.Float32.FastMath
module A = Pulse.Lib.Array
module I64 = FStar.Int64
module SP = Luce.Softplus.Fast

(**** from dev/KPlay.Sums ****)

(*** Real sums in fma-chain order *)

let rec sumr (n : nat) (f : nat -> real) : Tot real (decreases n) =
  if n = 0 then 0.0R else f (n - 1) +. sumr (n - 1) f

let shift (f : nat -> real) (k : nat) (i : nat) : real = f (k + i)
let blk (f : nat -> real) (b : nat) (i : nat) : real = sumr b (shift f (i * b))
let rows (g : nat -> nat -> real) (b : nat) (i : nat) : real = sumr b (g i)
let flip (g : nat -> nat -> real) (j : nat) (i : nat) : real = g i j
let strd (f : nat -> real) (b s : nat) (t : nat) : real = f (b + t * s)
let pr (f : nat -> real) (s : nat) (l : nat) (t : nat) : real = f (t * s + l)

let rec sumr_ext (n : nat) (f g : nat -> real)
  : Lemma (requires forall (i : nat). i < n ==> f i == g i) (ensures sumr n f == sumr n g)
  = if n = 0 then () else sumr_ext (n - 1) f g

let rec sumr_add (n : nat) (f g h : nat -> real)
  : Lemma (requires forall (i : nat). i < n ==> h i == f i +. g i)
          (ensures sumr n h == sumr n f +. sumr n g)
  = if n = 0 then () else sumr_add (n - 1) f g h

let rec sumr_scale (n : nat) (c : real) (f g : nat -> real)
  : Lemma (requires forall (i : nat). i < n ==> g i == c *. f i)
          (ensures sumr n g == c *. sumr n f)
  = if n = 0 then () else sumr_scale (n - 1) c f g

let rec sumr_zero (n : nat) (f : nat -> real)
  : Lemma (requires forall (i : nat). i < n ==> f i == 0.0R) (ensures sumr n f == 0.0R)
  = if n = 0 then () else sumr_zero (n - 1) f

let rec sumr_nonneg (n : nat) (f : nat -> real)
  : Lemma (requires forall (i : nat). i < n ==> f i >=. 0.0R) (ensures sumr n f >=. 0.0R)
  = if n = 0 then () else sumr_nonneg (n - 1) f

let rec sumr_split (a b : nat) (f : nat -> real)
  : Lemma (ensures sumr (a + b) f == sumr a f +. sumr b (shift f a)) (decreases b)
  = if b = 0 then () else sumr_split a (b - 1) f

let rec sumr_block (a b : nat) (f : nat -> real)
  : Lemma (ensures sumr (a * b) f == sumr a (blk f b)) (decreases a)
  = if a = 0 then ()
    else begin
      sumr_block (a - 1) b f;
      sumr_split ((a - 1) * b) b f
    end

let rec sumr_swap (a b : nat) (g : nat -> nat -> real)
  : Lemma (ensures sumr a (rows g b) == sumr b (rows (flip g) a)) (decreases a)
  = if a = 0 then sumr_zero b (rows (flip g) 0)
    else begin
      sumr_swap (a - 1) b g;
      sumr_add b (g (a - 1)) (rows (flip g) (a - 1)) (rows (flip g) a)
    end

let rec sumr_zero_tail (n m : nat) (f : nat -> real)
  : Lemma (requires n <= m /\ (forall (i : nat). n <= i /\ i < m ==> f i == 0.0R))
          (ensures sumr m f == sumr n f) (decreases m)
  = if m = n then () else sumr_zero_tail n (m - 1) f

let rec sumr_evenodd (m : nat) (f : nat -> real)
  : Lemma (ensures sumr (2 * m) f == sumr m (strd f 0 2) +. sumr m (strd f 1 2)) (decreases m)
  = if m = 0 then () else sumr_evenodd (m - 1) f

(*** Bridge to the spec's vsum *)

let rsum_empty () : Lemma (rsum (S.empty #real) == 0.0R) = ()

let rsum_snoc (s : S.seq real) (x : real)
  : Lemma (rsum (S.snoc s x) == rsum s +. x)
  = assert (S.equal (S.snoc s x) (S.append s (S.create 1 x)));
    rsum_append s (S.create 1 x);
    assert (S.equal (S.create 1 x) (S.cons x (S.empty #real)));
    Kuiper.Seq.Common.lemma_view_seq_cons x (S.empty #real)

let vsum_unfold (n : nat) (f : nat -> real)
  : Lemma (vsum n f == rsum (S.init n (fun (k : nat{k < n}) -> f k))) = ()

#push-options "--z3rlimit 40"
let rec vsum_sumr (n : nat) (f : nat -> real)
  : Lemma (ensures vsum n f == sumr n f) (decreases n)
  = vsum_unfold n f;
    if n = 0 then begin
      assert (S.equal (S.init 0 (fun (k : nat{k < 0}) -> f k)) (S.empty #real))
    end else begin
      vsum_sumr (n - 1) f;
      vsum_unfold (n - 1) f;
      let s1 = S.init n (fun (k : nat{k < n}) -> f k) in
      let s0 = S.init (n - 1) (fun (k : nat{k < n - 1}) -> f k) in
      assert (S.equal s1 (S.snoc s0 (f (n - 1))));
      rsum_snoc s0 (f (n - 1));
      assert (rsum s1 == rsum s0 +. f (n - 1))
    end
#pop-options

(*** Tree reduction over 32 lanes *)

let tl (f : nat -> real) (dd i : nat) : real = if dd = 0 then 0.0R else sumr (32 / dd) (strd f i dd)

let tl_init (f : nat -> real) (i : nat) : Lemma (tl f 32 i == f i) = ()

let tl_step (f : nat -> real) (dv : nat) (i : nat)
  : Lemma (requires dv == 1 \/ dv == 2 \/ dv == 4 \/ dv == 8 \/ dv == 16)
          (ensures tl f dv i == tl f (2 * dv) i +. tl f (2 * dv) (i + dv))
  = let m = 32 / (2 * dv) in
    assert (32 / dv == 2 * m);
    sumr_evenodd m (strd f i dv);
    sumr_ext m (strd (strd f i dv) 0 2) (strd f i (2 * dv));
    sumr_ext m (strd (strd f i dv) 1 2) (strd f (i + dv) (2 * dv))

let tl_final (f : nat -> real) : Lemma (tl f 1 0 == sumr 32 f)
  = sumr_ext 32 (strd f 0 1) f

(*** Reindexing of the kernels' lane layouts *)

(* lanes l < s, each summing c terms at stride s *)
let sum_strided (f : nat -> real) (s c : nat)
  : Lemma (sumr s (rows (pr f s) c) == sumr (c * s) f)
  = sumr_swap s c (pr f s);
    let aux (t : nat) : Lemma (rows (flip (pr f s)) s t == blk f s t)
      = sumr_ext s (flip (pr f s) t) (shift f (t * s))
    in
    FStar.Classical.forall_intro aux;
    sumr_ext c (rows (flip (pr f s)) s) (blk f s);
    sumr_block c s f

(* wdot lane l, block m: 8 contiguous terms at m*256 + l*8 *)
let wb (f : nat -> real) (l m : nat) : real = sumr 8 (shift f (m * 256 + l * 8))

let sum_wdot (f : nat -> real) (nm : nat)
  : Lemma (sumr 32 (rows (wb f) nm) == sumr (nm * 256) f)
  = sumr_swap 32 nm (wb f);
    let aux (m : nat) : Lemma (rows (flip (wb f)) 32 m == blk f 256 m)
      = let inner (l : nat) : Lemma (flip (wb f) m l == blk (shift f (m * 256)) 8 l)
          = sumr_ext 8 (shift f (m * 256 + l * 8)) (shift (shift f (m * 256)) (l * 8))
        in
        FStar.Classical.forall_intro inner;
        sumr_ext 32 (flip (wb f) m) (blk (shift f (m * 256)) 8);
        sumr_block 32 8 (shift f (m * 256))
    in
    FStar.Classical.forall_intro aux;
    sumr_ext nm (rows (flip (wb f)) 32) (blk f 256);
    sumr_block nm 256 f

(**** from dev/KPlay.Ax ****)

inline_for_extraction noextract
let wcast (b : BF.t) : F32.t = C.c_bf16_32.fcast b

inline_for_extraction noextract
let bround (v : F32.t) : F32.t = C.c_bf16_32.fcast (C.c_32_bf16.fcast v)

(*** Ideal environments and the approximation predicate *)

noextract noeq type env = { rw : S.seq real; rx : S.seq real; rr : S.seq real; rcv : S.seq real }

let pred = env -> prop
let sf = env -> real
let af = env -> nat -> real

[@@"opaque_to_smt"]
let ax (p : pred) (v : F32.t) (f : sf) : prop = forall (e : env). p e ==> v %~ f e

let ax_def (p : pred) (v : F32.t) (f : sf)
  : Lemma (ax p v f <==> (forall (e : env). p e ==> v %~ f e))
  = assert_norm (ax p v f == (forall (e : env). p e ==> v %~ f e))

let fadd (f g : sf) (e : env) : real = f e +. g e
let fmul (f g : sf) (e : env) : real = f e *. g e
let ffma (f g h : sf) (e : env) : real = f e *. g e +. h e
let fneg (f : sf) (e : env) : real = 0.0R -. f e
let fexp (f : sf) (e : env) : real = Exp.exp (f e)
let frcp (f : sf) (e : env) : real = rdiv 1.0R (f e)
let frsq (f : sf) (e : env) : real = rsqrt_t (f e)
let fsp (f : sf) (e : env) : real = softplus (f e)
let fconst (c : real) (e : env) : real = c
let elt (a : af) (i : nat) (e : env) : real = a e i

let ax_ext (p : pred) (v : F32.t) (f g : sf)
  : Lemma (requires ax p v f /\ (forall (e : env). p e ==> f e == g e)) (ensures ax p v g)
  = ax_def p v f; ax_def p v g

let ax_add (p : pred) (x y r : F32.t) (f g : sf)
  : Lemma (requires ax p x f /\ ax p y g /\
            (forall (xr yr : real). x %~ xr /\ y %~ yr ==> r %~ RO.r_add xr yr))
          (ensures ax p r (fadd f g))
  = ax_def p x f; ax_def p y g; ax_def p r (fadd f g)

let ax_mul (p : pred) (x y r : F32.t) (f g : sf)
  : Lemma (requires ax p x f /\ ax p y g /\
            (forall (xr yr : real). x %~ xr /\ y %~ yr ==> r %~ RO.r_mul xr yr))
          (ensures ax p r (fmul f g))
  = ax_def p x f; ax_def p y g; ax_def p r (fmul f g)

let ax_fma (p : pred) (x y z r : F32.t) (f g h : sf)
  : Lemma (requires ax p x f /\ ax p y g /\ ax p z h /\
            (forall (xr yr zr : real). x %~ xr /\ y %~ yr /\ z %~ zr ==> r %~ RO.r_fma xr yr zr))
          (ensures ax p r (ffma f g h))
  = ax_def p x f; ax_def p y g; ax_def p z h;
    ax_def p r (ffma f g h)

let ax_exp (p : pred) (x r : F32.t) (f : sf)
  : Lemma (requires ax p x f /\ (forall (xr : real). x %~ xr ==> r %~ RO.r_exp xr))
          (ensures ax p r (fexp f))
  = ax_def p x f; ax_def p r (fexp f)

let ax_rcp (p : pred) (x r : F32.t) (f : sf)
  : Lemma (requires ax p x f /\ (forall (e : env). p e ==> f e =!= 0.0R) /\
            (forall (xr : real{xr =!= 0.0R}). x %~ xr ==> r %~ RO.r_rcp xr))
          (ensures ax p r (frcp f))
  = ax_def p x f; ax_def p r (frcp f)

let ax_rsq (p : pred) (x r : F32.t) (f : sf)
  : Lemma (requires ax p x f /\ (forall (e : env). p e ==> f e >. 0.0R) /\
            (forall (xr : Sqrt.rpos). x %~ xr ==> r %~ RO.r_rsqrt xr))
          (ensures ax p r (frsq f))
  = ax_def p x f; ax_def p r (frsq f)

let ax_sp (p : pred) (x r : F32.t) (f : sf)
  : Lemma (requires ax p x f /\ (forall (xr : real). x %~ xr ==> r %~ Luce.Softplus.model xr))
          (ensures ax p r (fsp f))
  = ax_def p x f; ax_def p r (fsp f)

let ax_neg (p : pred) (x : F32.t) (f : sf)
  : Lemma (requires ax p x f) (ensures ax p (neg x) (fneg f))
  = ax_def p x f; ax_def p (neg x) (fneg f);
    Kuiper.Approximates.Base.neg_models_pat F32.t

let ax_bround (p : pred) (x : F32.t) (f : sf)
  : Lemma (requires ax p x f) (ensures ax p (bround x) f)
  = ax_def p x f; ax_def p (bround x) f

let z1 () : Lemma (zero #F32.t %~ 0.0R) = ()
let o1 () : Lemma (one #F32.t %~ 1.0R) = ()
let ax_intro (p : pred) (v : F32.t) (f : sf)
  : Lemma (requires forall (e : env). p e ==> v %~ f e) (ensures ax p v f)
  = assert_norm (ax p v f == (forall (e : env). p e ==> v %~ f e))

let ax_const (p : pred) (v : F32.t) (c : real)
  : Lemma (requires v %~ c) (ensures ax p v (fconst c))
  = ax_intro p v (fconst c)

let ax_zero (p : pred) : Lemma (ax p (zero #F32.t) (fconst 0.0R))
  = z1 (); ax_const p (zero #F32.t) 0.0R

let ax_one (p : pred) : Lemma (ax p (one #F32.t) (fconst 1.0R))
  = o1 (); ax_const p (one #F32.t) 1.0R

let ax_lit (p : pred) (s : string) (m : nat) (x : int)
  : Lemma (requires Cert.exact_normal s m x) (ensures ax p (of_literal #F32.t s) (fconst (Lit.value s)))
  = CF.literal_approx s m x; ax_const p (of_literal #F32.t s) (Lit.value s)

noextract let eps_s = "0.0000009999999974752427078783512115478515625"
noextract let qsc_s = "0.088388346135616302490234375"
noextract let i1024_s = "0.0009765625"
noextract let i128_s = "0.0078125"

let cert_eps () : Lemma (Cert.exact_normal eps_s 8796093 (-43)) = assert_norm (Cert.exact_normal eps_s 8796093 (-43))
let cert_qsc () : Lemma (Cert.exact_normal qsc_s 11863283 (-27)) = assert_norm (Cert.exact_normal qsc_s 11863283 (-27))
let cert_i1024 () : Lemma (Cert.exact_normal i1024_s 8388608 (-33)) = assert_norm (Cert.exact_normal i1024_s 8388608 (-33))
let cert_i128 () : Lemma (Cert.exact_normal i128_s 8388608 (-30)) = assert_norm (Cert.exact_normal i128_s 8388608 (-30))

let val_i1024 () : Lemma (Lit.value i1024_s == 1.0R /. 1024.0R)
  = assert_norm (Lit.value i1024_s == of_int 9765625 /. of_int 10000000000)
let val_i128 () : Lemma (Lit.value i128_s == 1.0R /. 128.0R)
  = assert_norm (Lit.value i128_s == of_int 78125 /. of_int 10000000)
let val_eps () : Lemma (Lit.value eps_s >. 0.0R)
  = assert_norm (Lit.value eps_s == of_int 9999999974752427078783512115478515625 /. of_int 10000000000000000000000000000000000000000000)

let sp_model (r : real) : Lemma (Luce.Softplus.model r == softplus r) = ()

let ax_elim (p : pred) (v : F32.t) (f : sf) (e : env)
  : Lemma (requires ax p v f /\ p e) (ensures v %~ f e)
  = ax_def p v f

(*** Ideals of array ranges *)

[@@"opaque_to_smt"]
let axr (p : pred) (s : S.seq F32.t) (a : af) (lo hi : nat) : prop =
  forall (i : nat). lo <= i /\ i < hi ==> (i < S.length s /\ ax p (S.index s i) (elt a i))

let axr_def (p : pred) (s : S.seq F32.t) (a : af) (lo hi : nat)
  : Lemma (axr p s a lo hi <==> (forall (i : nat). lo <= i /\ i < hi ==> (i < S.length s /\ ax p (S.index s i) (elt a i))))
  = assert_norm (axr p s a lo hi == (forall (i : nat). lo <= i /\ i < hi ==> (i < S.length s /\ ax p (S.index s i) (elt a i))))

let axr_intro (p : pred) (s : S.seq F32.t) (a : af) (lo hi : nat)
  : Lemma (requires forall (i : nat). lo <= i /\ i < hi ==> (i < S.length s /\ ax p (S.index s i) (elt a i)))
          (ensures axr p s a lo hi)
  = assert_norm (axr p s a lo hi == (forall (i : nat). lo <= i /\ i < hi ==> (i < S.length s /\ ax p (S.index s i) (elt a i))))

let axr_get (p : pred) (s : S.seq F32.t) (a : af) (lo hi i : nat)
  : Lemma (requires axr p s a lo hi /\ lo <= i /\ i < hi)
          (ensures i < S.length s /\ ax p (S.index s i) (elt a i))
  = axr_def p s a lo hi

let axr_empty (p : pred) (s : S.seq F32.t) (a : af) (n : nat)
  : Lemma (axr p s a n n)
  = axr_intro p s a n n

let axr_mono (p : pred) (s : S.seq F32.t) (a : af) (lo hi lo' hi' : nat)
  : Lemma (requires axr p s a lo hi /\ lo <= lo' /\ hi' <= hi) (ensures axr p s a lo' hi')
  = axr_def p s a lo hi; axr_intro p s a lo' hi'

let axr_snoc (p : pred) (s : S.seq F32.t) (a : af) (lo hi : nat) (v : F32.t)
  : Lemma (requires axr p s a lo hi /\ lo <= hi /\ hi < S.length s /\ ax p v (elt a hi))
          (ensures axr p (S.upd s hi v) a lo (hi + 1))
  = axr_def p s a lo hi; axr_intro p (S.upd s hi v) a lo (hi + 1)

let axr_frame (p : pred) (s : S.seq F32.t) (a : af) (lo hi j : nat) (v : F32.t)
  : Lemma (requires axr p s a lo hi /\ (j < lo \/ j >= hi) /\ j < S.length s)
          (ensures axr p (S.upd s j v) a lo hi)
  = axr_def p s a lo hi; axr_intro p (S.upd s j v) a lo hi

let axr_ext (p : pred) (s : S.seq F32.t) (a b : af) (lo hi : nat)
  : Lemma (requires axr p s a lo hi /\ (forall (e : env) (i : nat). p e /\ lo <= i /\ i < hi ==> a e i == b e i))
          (ensures axr p s b lo hi)
  = axr_def p s a lo hi;
    let aux (i : nat) : Lemma (requires lo <= i /\ i < hi) (ensures i < S.length s /\ ax p (S.index s i) (elt b i))
      = ax_ext p (S.index s i) (elt a i) (elt b i)
    in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux);
    axr_intro p s b lo hi

(*** Inputs *)

let okp (ws : S.seq BF.t) (xs rs cs : S.seq F32.t) (e : env) : prop =
  ws %~ e.rw /\ xs %~ e.rx /\ rs %~ e.rr /\ cs %~ e.rcv

let fw (i : int) (e : env) : real = at e.rw i
let fx (i : int) (e : env) : real = at e.rx i
let fr (i : int) (e : env) : real = at e.rr i
let fc (i : int) (e : env) : real = at e.rcv i

[@@"opaque_to_smt"]
let wok (p : pred) (ws : S.seq BF.t) : prop = forall (e : env). p e ==> ws %~ e.rw
[@@"opaque_to_smt"]
let xok (p : pred) (xs : S.seq F32.t) : prop = forall (e : env). p e ==> xs %~ e.rx
[@@"opaque_to_smt"]
let rok (p : pred) (rs : S.seq F32.t) : prop = forall (e : env). p e ==> rs %~ e.rr
[@@"opaque_to_smt"]
let cok (p : pred) (cs : S.seq F32.t) : prop = forall (e : env). p e ==> cs %~ e.rcv

let okp_ok (ws : S.seq BF.t) (xs rs cs : S.seq F32.t)
  : Lemma (let p = okp ws xs rs cs in wok p ws /\ xok p xs /\ rok p rs /\ cok p cs)
  = let p = okp ws xs rs cs in
    assert_norm (wok p ws == (forall (e : env). p e ==> ws %~ e.rw));
    assert_norm (xok p xs == (forall (e : env). p e ==> xs %~ e.rx));
    assert_norm (rok p rs == (forall (e : env). p e ==> rs %~ e.rr));
    assert_norm (cok p cs == (forall (e : env). p e ==> cs %~ e.rcv))

let ax_w (p : pred) (ws : S.seq BF.t) (i : nat)
  : Lemma (requires wok p ws /\ i < S.length ws) (ensures ax p (wcast (S.index ws i)) (fw i))
  = assert_norm (wok p ws == (forall (e : env). p e ==> ws %~ e.rw));
    let aux (e : env) : Lemma (requires p e) (ensures wcast (S.index ws i) %~ fw i e) = () in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux);
    ax_intro p (wcast (S.index ws i)) (fw i)

let ax_x (p : pred) (xs : S.seq F32.t) (i : nat)
  : Lemma (requires xok p xs /\ i < S.length xs) (ensures ax p (S.index xs i) (fx i))
  = assert_norm (xok p xs == (forall (e : env). p e ==> xs %~ e.rx));
    let aux (e : env) : Lemma (requires p e) (ensures S.index xs i %~ fx i e) = () in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux);
    ax_intro p (S.index xs i) (fx i)

let ax_r (p : pred) (rs : S.seq F32.t) (i : nat)
  : Lemma (requires rok p rs /\ i < S.length rs) (ensures ax p (S.index rs i) (fr i))
  = assert_norm (rok p rs == (forall (e : env). p e ==> rs %~ e.rr));
    let aux (e : env) : Lemma (requires p e) (ensures S.index rs i %~ fr i e) = () in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux);
    ax_intro p (S.index rs i) (fr i)

let ax_c (p : pred) (cs : S.seq F32.t) (i : nat)
  : Lemma (requires cok p cs /\ i < S.length cs) (ensures ax p (S.index cs i) (fc i))
  = assert_norm (cok p cs == (forall (e : env). p e ==> cs %~ e.rcv));
    let aux (e : env) : Lemma (requires p e) (ensures S.index cs i %~ fc i e) = () in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux);
    ax_intro p (S.index cs i) (fc i)

(*** Tree reduction at the ideal level *)

let ssum (n : nat) (l : af) (e : env) : real = sumr n (l e)
let tla (l : af) (dd : nat) (e : env) (i : nat) : real = tl (l e) dd i

let tla_init (p : pred) (s : S.seq F32.t) (l : af)
  : Lemma (requires axr p s l 0 32) (ensures axr p s (tla l 32) 0 32)
  = axr_ext p s l (tla l 32) 0 32

let tla_step (p : pred) (x y r : F32.t) (l : af) (dv i : nat)
  : Lemma (requires (dv == 1 \/ dv == 2 \/ dv == 4 \/ dv == 8 \/ dv == 16) /\
            ax p x (elt (tla l (2 * dv)) i) /\ ax p y (elt (tla l (2 * dv)) (i + dv)) /\
            (forall (xr yr : real). x %~ xr /\ y %~ yr ==> r %~ RO.r_add xr yr))
          (ensures ax p r (elt (tla l dv) i))
  = ax_add p x y r (elt (tla l (2 * dv)) i) (elt (tla l (2 * dv)) (i + dv));
    let aux (e : env) : Lemma (fadd (elt (tla l (2 * dv)) i) (elt (tla l (2 * dv)) (i + dv)) e == elt (tla l dv) i e)
      = tl_step (l e) dv i
    in
    FStar.Classical.forall_intro aux;
    ax_ext p r (fadd (elt (tla l (2 * dv)) i) (elt (tla l (2 * dv)) (i + dv))) (elt (tla l dv) i)

let tla_final (p : pred) (r : F32.t) (l : af)
  : Lemma (requires ax p r (elt (tla l 1) 0)) (ensures ax p r (ssum 32 l))
  = let aux (e : env) : Lemma (elt (tla l 1) 0 e == ssum 32 l e) = tl_final (l e) in
    FStar.Classical.forall_intro aux;
    ax_ext p r (elt (tla l 1) 0) (ssum 32 l)

let axr_join (p : pred) (s : S.seq F32.t) (a : af) (lo mid hi : nat)
  : Lemma (requires axr p s a lo mid /\ axr p s a mid hi) (ensures axr p s a lo hi)
  = axr_def p s a lo mid; axr_def p s a mid hi; axr_intro p s a lo hi

let zf (e : env) (i : nat) : real = 0.0R

let axr_zero (p : pred) (n : nat)
  : Lemma (axr p (S.create n (zero #F32.t)) zf 0 n)
  = let s = S.create n (zero #F32.t) in
    let aux (i : nat) : Lemma (requires i < n) (ensures i < S.length s /\ ax p (S.index s i) (elt zf i))
      = ax_zero p; ax_ext p (zero #F32.t) (fconst 0.0R) (elt zf i)
    in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux);
    axr_intro p s zf 0 n

(*** Lane accumulators *)

let lacc (t : af) (s l n : nat) (e : env) : real = sumr n (pr (t e) s l)
let lg (t : af) (s c g : nat) (e : env) (l : nat) : real = lacc t s (g * 32 + l) c e

let lacc_zero (p : pred) (t : af) (s l : nat)
  : Lemma (ax p (zero #F32.t) (lacc t s l 0))
  = ax_zero p; ax_ext p (zero #F32.t) (fconst 0.0R) (lacc t s l 0)

let lacc_step (p : pred) (x y acc r : F32.t) (fx fy : sf) (t : af) (s l n : nat)
  : Lemma (requires ax p x fx /\ ax p y fy /\ ax p acc (lacc t s l n) /\
            (forall (xr yr zr : real). x %~ xr /\ y %~ yr /\ acc %~ zr ==> r %~ RO.r_fma xr yr zr) /\
            (forall (e : env). p e ==> fx e *. fy e == t e (n * s + l)))
          (ensures ax p r (lacc t s l (n + 1)))
  = ax_fma p x y acc r fx fy (lacc t s l n);
    ax_ext p r (ffma fx fy (lacc t s l n)) (lacc t s l (n + 1))

let lanes1_final (p : pred) (r : F32.t) (t : af) (c : nat)
  : Lemma (requires ax p r (ssum 32 (lg t 32 c 0))) (ensures ax p r (ssum (c * 32) t))
  = let aux (e : env) : Lemma (ssum 32 (lg t 32 c 0) e == ssum (c * 32) t e)
      = sumr_ext 32 (lg t 32 c 0 e) (rows (pr (t e) 32) c);
        sum_strided (t e) 32 c
    in
    FStar.Classical.forall_intro aux;
    ax_ext p r (ssum 32 (lg t 32 c 0)) (ssum (c * 32) t)

let wz (t : af) (s c nw : nat) (e : env) (g : nat) : real = if g < nw then ssum 32 (lg t s c g) e else 0.0R

let warps_final (p : pred) (r : F32.t) (t : af) (c nw : nat)
  : Lemma (requires nw <= 32 /\ ax p r (ssum 32 (wz t (nw * 32) c nw))) (ensures ax p r (ssum (c * (nw * 32)) t))
  = let s = nw * 32 in
    let aux (e : env) : Lemma (ssum 32 (wz t s c nw) e == ssum (c * s) t e)
      = let lf = lacc t s in
        sumr_zero_tail nw 32 (wz t s c nw e);
        let inner (g : nat) : Lemma (requires g < nw) (ensures wz t s c nw e g == blk (rows (pr (t e) s) c) 32 g)
          = sumr_ext 32 (lg t s c g e) (shift (rows (pr (t e) s) c) (g * 32))
        in
        FStar.Classical.forall_intro (FStar.Classical.move_requires inner);
        sumr_ext nw (wz t s c nw e) (blk (rows (pr (t e) s) c) 32);
        sumr_block nw 32 (rows (pr (t e) s) c);
        sum_strided (t e) s c
    in
    FStar.Classical.forall_intro aux;
    ax_ext p r (ssum 32 (wz t s c nw)) (ssum (c * s) t)

(*** wdot accumulators *)

let wacc (t : af) (l m : nat) (e : env) : real = sumr m (wb (t e) l)
let wl (t : af) (nm : nat) (e : env) (l : nat) : real = wacc t l nm e
let wpart (t : af) (g : bool) (l m j : nat) (e : env) : real =
  (if g then 0.0R else wacc t l m e) +. sumr j (shift (t e) (m * 256 + l * 8))

let wacc_zero (p : pred) (t : af) (l : nat)
  : Lemma (ax p (zero #F32.t) (wacc t l 0))
  = ax_zero p; ax_ext p (zero #F32.t) (fconst 0.0R) (wacc t l 0)

let wpart_init_g (p : pred) (t : af) (l m : nat)
  : Lemma (ax p (zero #F32.t) (wpart t true l m 0))
  = ax_zero p; ax_ext p (zero #F32.t) (fconst 0.0R) (wpart t true l m 0)

let wpart_init_c (p : pred) (a : F32.t) (t : af) (l m : nat)
  : Lemma (requires ax p a (wacc t l m)) (ensures ax p a (wpart t false l m 0))
  = ax_ext p a (wacc t l m) (wpart t false l m 0)

let wpart_step (p : pred) (x y pv r : F32.t) (fx fy : sf) (t : af) (g : bool) (l m j : nat)
  : Lemma (requires ax p x fx /\ ax p y fy /\ ax p pv (wpart t g l m j) /\
            (forall (xr yr zr : real). x %~ xr /\ y %~ yr /\ pv %~ zr ==> r %~ RO.r_fma xr yr zr) /\
            (forall (e : env). p e ==> fx e *. fy e == t e (m * 256 + l * 8 + j)))
          (ensures ax p r (wpart t g l m (j + 1)))
  = ax_fma p x y pv r fx fy (wpart t g l m j);
    ax_ext p r (ffma fx fy (wpart t g l m j)) (wpart t g l m (j + 1))

let wacc_grouped (p : pred) (a pv r : F32.t) (t : af) (l m : nat)
  : Lemma (requires ax p a (wacc t l m) /\ ax p pv (wpart t true l m 8) /\
            (forall (xr yr : real). a %~ xr /\ pv %~ yr ==> r %~ RO.r_add xr yr))
          (ensures ax p r (wacc t l (m + 1)))
  = ax_add p a pv r (wacc t l m) (wpart t true l m 8);
    ax_ext p r (fadd (wacc t l m) (wpart t true l m 8)) (wacc t l (m + 1))

let wacc_cont (p : pred) (pv : F32.t) (t : af) (l m : nat)
  : Lemma (requires ax p pv (wpart t false l m 8)) (ensures ax p pv (wacc t l (m + 1)))
  = ax_ext p pv (wpart t false l m 8) (wacc t l (m + 1))

let wdot_final (p : pred) (r : F32.t) (t : af) (nm : nat)
  : Lemma (requires ax p r (ssum 32 (wl t nm))) (ensures ax p r (ssum (nm * 256) t))
  = let aux (e : env) : Lemma (ssum 32 (wl t nm) e == ssum (nm * 256) t e)
      = sumr_ext 32 (wl t nm e) (rows (wb (t e)) nm);
        sum_wdot (t e) nm
    in
    FStar.Classical.forall_intro aux;
    ax_ext p r (ssum 32 (wl t nm)) (ssum (nm * 256) t)

let sqt (x : af) (xo : nat) (e : env) (i : nat) : real = x e (xo + i) *. x e (xo + i)
let dt (a : af) (ao : nat) (b : af) (e : env) (i : nat) : real = a e (ao + i) *. b e i
let wt (wo : nat) (x : af) (e : env) (k : nat) : real = at e.rw (wo + k) *. x e k

let lg_elt (p : pred) (v : F32.t) (t : af) (s c g l : nat)
  : Lemma (requires ax p v (lacc t s (g * 32 + l) c)) (ensures ax p v (elt (lg t s c g) l))
  = ax_ext p v (lacc t s (g * 32 + l) c) (elt (lg t s c g) l)

let wz_elt (p : pred) (v : F32.t) (t : af) (s c nw g : nat)
  : Lemma (requires g < nw /\ ax p v (ssum 32 (lg t s c g))) (ensures ax p v (elt (wz t s c nw) g))
  = ax_ext p v (ssum 32 (lg t s c g)) (elt (wz t s c nw) g)

let wz_zero (p : pred) (v : S.seq F32.t) (t : af) (s c nw : nat)
  : Lemma (requires axr p v zf nw 32) (ensures axr p v (wz t s c nw) nw 32)
  = axr_ext p v zf (wz t s c nw) nw 32

let wz_single (p : pred) (v : F32.t) (t : af) (c : nat)
  : Lemma (requires ax p v (elt (wz t 32 c 1) 0)) (ensures ax p v (ssum 32 (lg t 32 c 0)))
  = ax_ext p v (elt (wz t 32 c 1) 0) (ssum 32 (lg t 32 c 0))

let wpart_init (p : pred) (g : bool) (a : F32.t) (t : af) (l m : nat)
  : Lemma (requires ax p a (wacc t l m)) (ensures ax p (if g then zero #F32.t else a) (wpart t g l m 0))
  = if g then wpart_init_g p t l m else wpart_init_c p a t l m

let wl_elt (p : pred) (v : F32.t) (t : af) (l nm : nat)
  : Lemma (requires ax p v (wacc t l nm)) (ensures ax p v (elt (wl t nm) l))
  = ax_ext p v (wacc t l nm) (elt (wl t nm) l)

(**** from dev/KPlay.Conn ****)

(*** vsum bridges *)

let sqs (x : S.seq real) (o : int) (k : nat) : real = at x (o + k) *. at x (o + k)
let dts (w : S.seq real) (wo : int) (x : S.seq real) (k : nat) : real = at w (wo + k) *. at x k

let sq_sum_sumr (x : S.seq real) (o : int) (n : nat)
  : Lemma (sq_sum x o n == sumr n (sqs x o))
  = vsum_sumr n (fun k -> at x (o + k) *. at x (o + k));
    sumr_ext n (fun k -> at x (o + k) *. at x (o + k)) (sqs x o)

let dot_sumr (w : S.seq real) (wo : int) (x : S.seq real) (n : nat)
  : Lemma (dot w wo x n == sumr n (dts w wo x))
  = vsum_sumr n (fun k -> at w (wo + k) *. at x k);
    sumr_ext n (fun k -> at w (wo + k) *. at x k) (dts w wo x)

let sq_nonneg (x : S.seq real) (o : int) (n : nat)
  : Lemma (sq_sum x o n >=. 0.0R)
  = sq_sum_sumr x o n;
    sumr_nonneg n (sqs x o)

(*** Spec-shaped ideals *)

let atf (q : env -> S.seq real) (e : env) (i : nat) : real = at (q e) i
let atfo (q : env -> S.seq real) (o : nat) (e : env) (i : nat) : real = at (q e) (o + i)

let wt_dot (wo : nat) (q : env -> S.seq real) (n : nat) (e : env)
  : Lemma (ssum n (wt wo (atf q)) e == dot e.rw wo (q e) n)
  = dot_sumr e.rw wo (q e) n;
    sumr_ext n (wt wo (atf q) e) (dts e.rw wo (q e))

let sqt_sq (q : env -> S.seq real) (o xo : nat) (n : nat) (e : env)
  : Lemma (ssum n (sqt (atfo q o) xo) e == sq_sum (q e) (o + xo) n)
  = sq_sum_sumr (q e) (o + xo) n;
    sumr_ext n (sqt (atfo q o) xo e) (sqs (q e) (o + xo))

let sqt_sq0 (q : env -> S.seq real) (xo : nat) (n : nat) (e : env)
  : Lemma (ssum n (sqt (atf q) xo) e == sq_sum (q e) xo n)
  = sq_sum_sumr (q e) xo n;
    sumr_ext n (sqt (atf q) xo e) (sqs (q e) xo)

(* conv: the silu argument of dn_conv_out *)
let conv_pre (nm : numerics) (w : S.seq real) (wo : int) (x conv : S.seq real) (c : nat) : real =
  let c' = dn_conv nm w wo x conv in
  vsum 4 (fun k -> at c' (c * 4 + k) *. at w (dn_mix wo + 8421376 + c * 4 + k))

let conv_out_pre (nm : numerics) (w : S.seq real) (wo : int) (x conv : S.seq real) (c : nat)
  : Lemma (requires c < 6144) (ensures at (dn_conv_out nm w wo x conv) c == silu (conv_pre nm w wo x conv c))
  = ()

let cterm (c' w : S.seq real) (wo : int) (c : nat) (k : nat) : real =
  at c' (c * 4 + k) *. at w (dn_mix wo + 8421376 + c * 4 + k)

#push-options "--fuel 0 --ifuel 0"
let conv_pre_sum (nm : numerics) (w : S.seq real) (wo : int) (x conv : S.seq real) (c : nat)
  : Lemma (conv_pre nm w wo x conv c == sumr 4 (cterm (dn_conv nm w wo x conv) w wo c))
  = vsum_sumr 4 (fun k -> at (dn_conv nm w wo x conv) (c * 4 + k) *. at w (dn_mix wo + 8421376 + c * 4 + k));
    sumr_ext 4 (fun k -> at (dn_conv nm w wo x conv) (c * 4 + k) *. at w (dn_mix wo + 8421376 + c * 4 + k))
      (cterm (dn_conv nm w wo x conv) w wo c)
#pop-options

let sumr4 (f : nat -> real) : Lemma (sumr 4 f == f 3 +. (f 2 +. (f 1 +. (f 0 +. 0.0R))))
  = assert_norm (sumr 4 f == f 3 +. (f 2 +. (f 1 +. (f 0 +. 0.0R))))

(* delta: mem = dec * sum_i st[h, i, j] * k[h, i] *)
let stk (st k : S.seq real) (h j : int) (i : nat) : real = at st (h * 16384 + i * 128 + j) *. at k (h * 128 + i)

#push-options "--fuel 0 --ifuel 0"
let delta_sum (nm : numerics) (w : S.seq real) (wo : int) (x conv st : S.seq real) (h j : int)
  : Lemma (dn_delta nm w wo x conv st h j ==
           (at (dn_v nm w wo x conv) (h * 128 + j) -. dn_decay nm w wo x h *. sumr 128 (stk st (dn_k nm w wo x conv) h j)) *.
           dn_beta nm w wo x h)
  = vsum_sumr 128 (fun i -> (at st (h * 16384 + i * 128 + j) *. dn_decay nm w wo x h) *. at (dn_k nm w wo x conv) (h * 128 + i));
    sumr_scale 128 (dn_decay nm w wo x h) (stk st (dn_k nm w wo x conv) h j)
      (fun i -> (at st (h * 16384 + i * 128 + j) *. dn_decay nm w wo x h) *. at (dn_k nm w wo x conv) (h * 128 + i))
#pop-options

let ctm (st' q : S.seq real) (c : int) (i : nat) : real =
  at st' ((c / 128) * 16384 + i * 128 + c % 128) *. at q ((c / 128) * 128 + i)

#push-options "--fuel 0 --ifuel 0"
let core_sum (nm : numerics) (w : S.seq real) (wo : int) (x conv st : S.seq real) (c : nat)
  : Lemma (requires c < 2048)
          (ensures at (dn_core nm w wo x conv st) c == sumr 128 (ctm (dn_state nm w wo x conv st) (dn_q nm w wo x conv) c))
  = vsum_sumr 128 (fun i -> at (dn_state nm w wo x conv st) ((c / 128) * 16384 + i * 128 + c % 128) *. at (dn_q nm w wo x conv) ((c / 128) * 128 + i));
    sumr_ext 128 (fun i -> at (dn_state nm w wo x conv st) ((c / 128) * 16384 + i * 128 + c % 128) *. at (dn_q nm w wo x conv) ((c / 128) * 128 + i))
      (ctm (dn_state nm w wo x conv st) (dn_q nm w wo x conv) c)
#pop-options

(*** Index arithmetic *)

let idx3 (h a c : nat)
  : Lemma (requires h < 16 /\ a < 128 /\ c < 128)
          (ensures (let m = h * 16384 + a * 128 + c in
                    m < 262144 /\ m / 16384 == h /\ (m % 16384) / 128 == a /\ m % 128 == c /\
                    rec_perm m == h * 16384 + c * 128 + a))
  = let m = h * 16384 + a * 128 + c in
    let r = a * 128 + c in
    assert (r < 16384);
    FStar.Math.Lemmas.lemma_div_plus r h 16384;
    FStar.Math.Lemmas.small_div r 16384;
    FStar.Math.Lemmas.lemma_mod_plus r h 16384;
    FStar.Math.Lemmas.small_mod r 16384;
    FStar.Math.Lemmas.lemma_div_plus c a 128;
    FStar.Math.Lemmas.small_div c 128;
    assert (m == c + (h * 128 + a) * 128);
    FStar.Math.Lemmas.lemma_mod_plus c (h * 128 + a) 128;
    FStar.Math.Lemmas.small_mod c 128

let idx_slot (d m : nat)
  : Lemma (requires m < 262144) (ensures (d * 262144 + m) / 262144 == d /\ (d * 262144 + m) % 262144 == m)
  = FStar.Math.Lemmas.lemma_div_plus m d 262144;
    FStar.Math.Lemmas.small_div m 262144;
    FStar.Math.Lemmas.lemma_mod_plus m d 262144;
    FStar.Math.Lemmas.small_mod m 262144

let idx_hc (h i : nat)
  : Lemma (requires i < 128) (ensures (h * 128 + i) / 128 == h /\ (h * 128 + i) % 128 == i)
  = FStar.Math.Lemmas.lemma_div_plus i h 128;
    FStar.Math.Lemmas.small_div i 128;
    FStar.Math.Lemmas.lemma_mod_plus i h 128;
    FStar.Math.Lemmas.small_mod i 128

let idx_c4 (c k : nat)
  : Lemma (requires k < 4) (ensures (c * 4 + k) / 4 == c /\ (c * 4 + k) % 4 == k)
  = FStar.Math.Lemmas.lemma_div_plus k c 4;
    FStar.Math.Lemmas.small_div k 4;
    FStar.Math.Lemmas.lemma_mod_plus k c 4;
    FStar.Math.Lemmas.small_mod k 4

(*** Spec-shaped ideals of the kernel's buffers *)

noextract let xq (e : env) : S.seq real = e.rx
noextract let rmsq (q : env -> S.seq real) (wo : nat) (e : env) : S.seq real = rmsnorm1p eps_f32 (q e) 1024 e.rw wo
noextract let mvq (q : env -> S.seq real) (wo rows n : nat) (e : env) : S.seq real = matvec e.rw wo rows n (q e)
noextract let hinq (b : nat) (e : env) : S.seq real = dn_hin opt_numerics e.rw b e.rx
noextract let qkvq (b : nat) (e : env) : S.seq real = dn_qkv opt_numerics e.rw b e.rx
noextract let zq (b : nat) (e : env) : S.seq real = matvec e.rw (dn_mix b + 6291456) 2048 1024 (hinq b e)
noextract let convq (d : nat) (e : env) : S.seq real = opt_conv_slot e.rcv d
noextract let stq (d : nat) (e : env) : S.seq real = opt_rec_slot e.rr d
noextract let coq (b d : nat) (e : env) : S.seq real = dn_conv_out opt_numerics e.rw b e.rx (convq d e)
noextract let qq (b d : nat) (e : env) : S.seq real = dn_q opt_numerics e.rw b e.rx (convq d e)
noextract let kq (b d : nat) (e : env) : S.seq real = dn_k opt_numerics e.rw b e.rx (convq d e)
let betaf (b h : nat) (e : env) : real = dn_beta opt_numerics e.rw b e.rx h
let decf (b h : nat) (e : env) : real = dn_decay opt_numerics e.rw b e.rx h
let deltaf (b d h j : nat) (e : env) : real = dn_delta opt_numerics e.rw b e.rx (convq d e) (stq d e) h j
noextract let coreq (b d : nat) (e : env) : S.seq real = dn_core opt_numerics e.rw b e.rx (convq d e) (stq d e)
noextract let gatedq (b d : nat) (e : env) : S.seq real = dn_gated opt_numerics e.rw b e.rx (convq d e) (stq d e)
noextract let x1q (b d : nat) (e : env) : S.seq real = dn_x1 opt_numerics e.rw b e.rx (convq d e) (stq d e)
noextract let actq (q : env -> S.seq real) (b : nat) (e : env) : S.seq real = mlp_act opt_numerics e.rw b (q e)
noextract let resq (q r : env -> S.seq real) (wo n : nat) (e : env) : S.seq real = vadd (r e) (matvec e.rw wo 1024 n (q e)) 1024
noextract let outq (b d : nat) (e : env) : S.seq real = dn_out opt_numerics e.rw b e.rx (convq d e) (stq d e)
noextract let cvq (b d : nat) (e : env) : S.seq real = opt_conv_put e.rcv d (dn_conv opt_numerics e.rw b e.rx (convq d e))
noextract let rcq (b d : nat) (e : env) : S.seq real =
  opt_rec_put e.rr d (dn_state opt_numerics e.rw b e.rx (convq d e) (stq d e))
let rI (e : env) (i : nat) : real = at e.rr i

let axr_q (p : pred) (s : S.seq F32.t) (q1 q2 : env -> S.seq real) (lo hi : nat)
  : Lemma (requires axr p s (atf q1) lo hi /\ (forall (e : env). q1 e == q2 e)) (ensures axr p s (atf q2) lo hi)
  = axr_ext p s (atf q1) (atf q2) lo hi

let hin_eq (b : nat) (e : env) : Lemma (rmsq xq b e == hinq b e) = ()
let qkv_eq (b : nat) (e : env) : Lemma (mvq (hinq b) (dn_mix b) 6144 1024 e == qkvq b e) = ()
let z_eq (b : nat) (e : env) : Lemma (mvq (hinq b) (dn_mix b + 6291456) 2048 1024 e == zq b e) = ()
let x1_eq (b d : nat) (e : env) : Lemma (resq (gatedq b d) xq (dn_mix b + 8446112) 2048 e == x1q b d e) = ()
let mlpin_eq (b d : nat) (e : env) : Lemma (rmsq (x1q b d) (b + 1024) e == mlp_in opt_numerics e.rw b (x1q b d e)) = ()
let out_eq (b d : nat) (e : env) : Lemma (resq (actq (x1q b d) b) (x1q b d) (b + 7342080) 3584 e == outq b d e) = ()

(*** Activations *)

let fsig (f : sf) (e : env) : real = sigmoid (f e)
let fsilu (f : sf) (e : env) : real = silu (f e)

let sig_ax (p : pred) (x e1 a1 r : F32.t) (f : sf)
  : Lemma (requires ax p x f /\
            (forall (xr : real). neg x %~ xr ==> e1 %~ RO.r_exp xr) /\
            (forall (xr yr : real). one #F32.t %~ xr /\ e1 %~ yr ==> a1 %~ RO.r_add xr yr) /\
            (forall (xr : real{xr =!= 0.0R}). a1 %~ xr ==> r %~ RO.r_rcp xr))
          (ensures ax p r (fsig f))
  = ax_neg p x f;
    ax_exp p (neg x) e1 (fneg f);
    ax_one p;
    ax_add p (one #F32.t) e1 a1 (fconst 1.0R) (fexp (fneg f));
    let g = fadd (fconst 1.0R) (fexp (fneg f)) in
    let pos (e : env) : Lemma (g e >. 0.0R) = Exp.exp_positive (0.0R -. f e) in
    FStar.Classical.forall_intro pos;
    ax_rcp p a1 r g;
    ax_ext p r (frcp g) (fsig f)

let silu_ax (p : pred) (x s r : F32.t) (f : sf)
  : Lemma (requires ax p x f /\ ax p s (fsig f) /\
            (forall (xr yr : real). x %~ xr /\ s %~ yr ==> r %~ RO.r_mul xr yr))
          (ensures ax p r (fsilu f))
  = ax_mul p x s r f (fsig f);
    ax_ext p r (fmul f (fsig f)) (fsilu f)

(*** RMSNorm *)

let rms_s (q : env -> S.seq real) (e : env) : real = ssum (2 * (16 * 32)) (sqt (atf q) 0) e
let rms_inv (q : env -> S.seq real) (e : env) : real = rsqrt_t (rdiv (sq_sum (q e) 0 1024) (of_int 1024) +. eps_f32)

let lit_i1024 (p : pred) : Lemma (ax p (of_literal #F32.t "0.0009765625") (fconst (1.0R /. 1024.0R)))
  = cert_i1024 (); ax_lit p i1024_s 8388608 (-33); val_i1024 ()
let lit_i128 (p : pred) : Lemma (ax p (of_literal #F32.t "0.0078125") (fconst (1.0R /. 128.0R)))
  = cert_i128 (); ax_lit p i128_s 8388608 (-30); val_i128 ()
let lit_eps (p : pred) : Lemma (ax p (of_literal #F32.t "0.0000009999999974752427078783512115478515625") (fconst eps_f32) /\ eps_f32 >. 0.0R)
  = cert_eps (); ax_lit p eps_s 8796093 (-43); val_eps ()
let lit_qsc (p : pred) : Lemma (ax p (of_literal #F32.t "0.088388346135616302490234375") (fconst qscale_f32))
  = cert_qsc (); ax_lit p qsc_s 11863283 (-27)

let rdiv_1024 (x : real) : Lemma (rdiv x (of_int 1024) == x *. (1.0R /. 1024.0R)) = ()

let rms_pos (q : env -> S.seq real) (e : env)
  : Lemma (rms_s q e *. (1.0R /. 1024.0R) +. eps_f32 == rdiv (sq_sum (q e) 0 1024) (of_int 1024) +. eps_f32 /\
           rms_s q e *. (1.0R /. 1024.0R) >=. 0.0R)
  = sqt_sq0 q 0 1024 e; sq_nonneg (q e) 0 1024; rdiv_1024 (sq_sum (q e) 0 1024)

let rms_scale (p : pred) (s m a sc : F32.t) (q : env -> S.seq real)
  : Lemma (requires ax p s (rms_s q) /\
            (forall (xr yr : real). s %~ xr /\ of_literal #F32.t "0.0009765625" %~ yr ==> m %~ RO.r_mul xr yr) /\
            (forall (xr yr : real). m %~ xr /\ of_literal #F32.t "0.0000009999999974752427078783512115478515625" %~ yr ==> a %~ RO.r_add xr yr) /\
            (forall (xr : Sqrt.rpos). a %~ xr ==> sc %~ RO.r_rsqrt xr))
          (ensures ax p sc (rms_inv q))
  = lit_i1024 p; lit_eps p;
    ax_mul p s (of_literal #F32.t "0.0009765625") m (rms_s q) (fconst (1.0R /. 1024.0R));
    let g = fmul (rms_s q) (fconst (1.0R /. 1024.0R)) in
    ax_add p m (of_literal #F32.t "0.0000009999999974752427078783512115478515625") a g (fconst eps_f32);
    let h = fadd g (fconst eps_f32) in
    let pos (e : env) : Lemma (h e >. 0.0R /\ h e == rdiv (sq_sum (q e) 0 1024) (of_int 1024) +. eps_f32)
      = rms_pos q e; lit_eps p in
    FStar.Classical.forall_intro pos;
    ax_rsq p a sc h;
    ax_ext p sc (frsq h) (rms_inv q)

let rms_elt (p : pred) (q : env -> S.seq real) (wo i : nat) (h sc w1 m r : F32.t) (wv : BF.t)
  : Lemma (requires i < 1024 /\ ax p h (elt (atf q) i) /\ ax p sc (rms_inv q) /\ ax p (wcast wv) (fw (wo + i)) /\
            (forall (xr yr : real). one #F32.t %~ xr /\ wcast wv %~ yr ==> w1 %~ RO.r_add xr yr) /\
            (forall (xr yr : real). h %~ xr /\ sc %~ yr ==> m %~ RO.r_mul xr yr) /\
            (forall (xr yr : real). m %~ xr /\ w1 %~ yr ==> r %~ RO.r_mul xr yr))
          (ensures ax p (bround r) (elt (atf (rmsq q wo)) i))
  = ax_one p;
    ax_add p (one #F32.t) (wcast wv) w1 (fconst 1.0R) (fw (wo + i));
    ax_mul p h sc m (elt (atf q) i) (rms_inv q);
    ax_mul p m w1 r (fmul (elt (atf q) i) (rms_inv q)) (fadd (fconst 1.0R) (fw (wo + i)));
    ax_bround p r (fmul (fmul (elt (atf q) i) (rms_inv q)) (fadd (fconst 1.0R) (fw (wo + i))));
    ax_ext p (bround r) (fmul (fmul (elt (atf q) i) (rms_inv q)) (fadd (fconst 1.0R) (fw (wo + i)))) (elt (atf (rmsq q wo)) i)

(*** Matrix rows *)

let mv_elt (p : pred) (r : F32.t) (q : env -> S.seq real) (wo rows n i : nat)
  : Lemma (requires i < rows /\ ax p r (ssum n (wt (wo + i * n) (atf q))))
          (ensures ax p r (elt (atf (mvq q wo rows n)) i))
  = let aux (e : env) : Lemma (ssum n (wt (wo + i * n) (atf q)) e == elt (atf (mvq q wo rows n)) i e)
      = wt_dot (wo + i * n) q n e in
    FStar.Classical.forall_intro aux;
    ax_ext p r (ssum n (wt (wo + i * n) (atf q))) (elt (atf (mvq q wo rows n)) i)

let res_elt (p : pred) (q r : env -> S.seq real) (wo n i : nat) (dv rv s : F32.t)
  : Lemma (requires i < 1024 /\ ax p dv (ssum n (wt (wo + i * n) (atf q))) /\ ax p rv (elt (atf r) i) /\
            (forall (xr yr : real). dv %~ xr /\ rv %~ yr ==> s %~ RO.r_add xr yr))
          (ensures ax p (bround s) (elt (atf (resq q r wo n)) i))
  = ax_add p dv rv s (ssum n (wt (wo + i * n) (atf q))) (elt (atf r) i);
    ax_bround p s (fadd (ssum n (wt (wo + i * n) (atf q))) (elt (atf r) i));
    let aux (e : env) : Lemma (fadd (ssum n (wt (wo + i * n) (atf q))) (elt (atf r) i) e == elt (atf (resq q r wo n)) i e)
      = wt_dot (wo + i * n) q n e in
    FStar.Classical.forall_intro aux;
    ax_ext p (bround s) (fadd (ssum n (wt (wo + i * n) (atf q))) (elt (atf r) i)) (elt (atf (resq q r wo n)) i)

#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
let act_elt (p : pred) (q : env -> S.seq real) (b i : nat) (g u sg r : F32.t)
  : Lemma (requires i < 3584 /\ ax p g (ssum 1024 (wt (b + 2048 + i * 1024) (atf (rmsq q (b + 1024))))) /\
            ax p u (ssum 1024 (wt (b + 3672064 + i * 1024) (atf (rmsq q (b + 1024))))) /\
            ax p sg (fsilu (ssum 1024 (wt (b + 2048 + i * 1024) (atf (rmsq q (b + 1024)))))) /\
            (forall (xr yr : real). sg %~ xr /\ u %~ yr ==> r %~ RO.r_mul xr yr))
          (ensures ax p r (elt (atf (actq q b)) i))
  = let fg = ssum 1024 (wt (b + 2048 + i * 1024) (atf (rmsq q (b + 1024)))) in
    let fu = ssum 1024 (wt (b + 3672064 + i * 1024) (atf (rmsq q (b + 1024)))) in
    ax_mul p sg u r (fsilu fg) fu;
    let aux (e : env) : Lemma (fmul (fsilu fg) fu e == elt (atf (actq q b)) i e)
      = wt_dot (b + 2048 + i * 1024) (rmsq q (b + 1024)) 1024 e;
        wt_dot (b + 3672064 + i * 1024) (rmsq q (b + 1024)) 1024 e;
        let h = mlp_in opt_numerics e.rw b (q e) in
        assert (h == rmsq q (b + 1024) e);
        assert (at (matvec e.rw (b + 2048) 3584 1024 h) i == dot e.rw (b + 2048 + i * 1024) h 1024);
        assert (at (matvec e.rw (b + 3672064) 3584 1024 h) i == dot e.rw (b + 3672064 + i * 1024) h 1024) in
    FStar.Classical.forall_intro aux;
    ax_ext p r (fmul (fsilu fg) fu) (elt (atf (actq q b)) i)
#pop-options

(*** Per-head scalars *)

let bdot (b h : nat) : sf = ssum 1024 (wt (dn_mix b + 8388608 + h * 1024) (atf (hinq b)))

let beta_fin (p : pred) (r : F32.t) (b h : nat)
  : Lemma (requires ax p r (fsig (bdot b h))) (ensures ax p r (betaf b h))
  = let aux (e : env) : Lemma (fsig (bdot b h) e == betaf b h e)
      = wt_dot (dn_mix b + 8388608 + h * 1024) (hinq b) 1024 e in
    FStar.Classical.forall_intro aux;
    ax_ext p r (fsig (bdot b h)) (betaf b h)

#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
let dec_ax (p : pred) (b h : nat) (d dtv a sp alv ea m r : F32.t)
  : Lemma (requires ax p d (ssum 1024 (wt (dn_mix b + 8404992 + h * 1024) (atf (hinq b)))) /\
            ax p dtv (fw (dn_mix b + 8445968 + h)) /\
            (forall (xr yr : real). d %~ xr /\ dtv %~ yr ==> a %~ RO.r_add xr yr) /\
            (forall (xr : real). a %~ xr ==> sp %~ Luce.Softplus.model xr) /\
            ax p alv (fw (dn_mix b + 8445952 + h)) /\
            (forall (xr : real). alv %~ xr ==> ea %~ RO.r_exp xr) /\
            (forall (xr yr : real). ea %~ xr /\ sp %~ yr ==> m %~ RO.r_mul xr yr) /\
            (forall (xr : real). neg m %~ xr ==> r %~ RO.r_exp xr))
          (ensures ax p r (decf b h))
  = let fd = ssum 1024 (wt (dn_mix b + 8404992 + h * 1024) (atf (hinq b))) in
    ax_add p d dtv a fd (fw (dn_mix b + 8445968 + h));
    ax_sp p a sp (fadd fd (fw (dn_mix b + 8445968 + h)));
    ax_exp p alv ea (fw (dn_mix b + 8445952 + h));
    ax_mul p ea sp m (fexp (fw (dn_mix b + 8445952 + h))) (fsp (fadd fd (fw (dn_mix b + 8445968 + h))));
    let fm = fmul (fexp (fw (dn_mix b + 8445952 + h))) (fsp (fadd fd (fw (dn_mix b + 8445968 + h)))) in
    ax_neg p m fm;
    ax_exp p (neg m) r (fneg fm);
    let aux (e : env) : Lemma (fexp (fneg fm) e == decf b h e)
      = wt_dot (dn_mix b + 8404992 + h * 1024) (hinq b) 1024 e;
        let ea = Exp.exp (at e.rw (dn_mix b + 8445952 + h)) in
        let sv = softplus (dot e.rw (dn_mix b + 8404992 + h * 1024) (hinq b e) 1024 +. at e.rw (dn_mix b + 8445968 + h)) in
        assert (fneg fm e == 0.0R -. ea *. sv);
        assert ((0.0R -. ea) *. sv == 0.0R -. ea *. sv)
    in
    FStar.Classical.forall_intro aux;
    ax_ext p r (fexp (fneg fm)) (decf b h)
#pop-options

(*** Convolution *)

let cpre (b d c : nat) (e : env) : real = conv_pre opt_numerics e.rw b e.rx (convq d e) c

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let cpre_eq (b d c : nat) (e : env)
  : Lemma (requires c < 6144 /\ d < 18)
          (ensures cpre b d c e ==
             at (qkvq b e) c *. at e.rw (dn_mix b + 8421376 + c * 4 + 3) +.
             (at e.rcv (d * 24576 + c * 4 + 3) *. at e.rw (dn_mix b + 8421376 + c * 4 + 2) +.
             (at e.rcv (d * 24576 + c * 4 + 2) *. at e.rw (dn_mix b + 8421376 + c * 4 + 1) +.
             (at e.rcv (d * 24576 + c * 4 + 1) *. at e.rw (dn_mix b + 8421376 + c * 4) +. 0.0R))))
  = let c' = dn_conv opt_numerics e.rw b e.rx (convq d e) in
    conv_pre_sum opt_numerics e.rw b e.rx (convq d e) c;
    sumr4 (cterm c' e.rw b c);
    idx_c4 c 0; idx_c4 c 1; idx_c4 c 2; idx_c4 c 3;
    assert (at c' (c * 4 + 0) == at (convq d e) (c * 4 + 1));
    assert (at c' (c * 4 + 1) == at (convq d e) (c * 4 + 2));
    assert (at c' (c * 4 + 2) == at (convq d e) (c * 4 + 3));
    assert (at c' (c * 4 + 3) == at (qkvq b e) c)
#pop-options

let conv_ax (p : pred) (b d c : nat) (c0 c1 c2 c3 w0 w1 w2 w3 s0 s1 s2 s3 : F32.t)
  : Lemma (requires c < 6144 /\ d < 18 /\
            ax p c0 (fc (d * 24576 + c * 4 + 1)) /\ ax p c1 (fc (d * 24576 + c * 4 + 2)) /\
            ax p c2 (fc (d * 24576 + c * 4 + 3)) /\ ax p c3 (elt (atf (qkvq b)) c) /\
            ax p w0 (fw (dn_mix b + 8421376 + c * 4)) /\ ax p w1 (fw (dn_mix b + 8421376 + c * 4 + 1)) /\
            ax p w2 (fw (dn_mix b + 8421376 + c * 4 + 2)) /\ ax p w3 (fw (dn_mix b + 8421376 + c * 4 + 3)) /\
            (forall (xr yr zr : real). c0 %~ xr /\ w0 %~ yr /\ zero #F32.t %~ zr ==> s0 %~ RO.r_fma xr yr zr) /\
            (forall (xr yr zr : real). c1 %~ xr /\ w1 %~ yr /\ s0 %~ zr ==> s1 %~ RO.r_fma xr yr zr) /\
            (forall (xr yr zr : real). c2 %~ xr /\ w2 %~ yr /\ s1 %~ zr ==> s2 %~ RO.r_fma xr yr zr) /\
            (forall (xr yr zr : real). c3 %~ xr /\ w3 %~ yr /\ s2 %~ zr ==> s3 %~ RO.r_fma xr yr zr))
          (ensures ax p s3 (cpre b d c))
  = ax_zero p;
    let f0 = ffma (fc (d * 24576 + c * 4 + 1)) (fw (dn_mix b + 8421376 + c * 4)) (fconst 0.0R) in
    ax_fma p c0 w0 (zero #F32.t) s0 (fc (d * 24576 + c * 4 + 1)) (fw (dn_mix b + 8421376 + c * 4)) (fconst 0.0R);
    let f1 = ffma (fc (d * 24576 + c * 4 + 2)) (fw (dn_mix b + 8421376 + c * 4 + 1)) f0 in
    ax_fma p c1 w1 s0 s1 (fc (d * 24576 + c * 4 + 2)) (fw (dn_mix b + 8421376 + c * 4 + 1)) f0;
    let f2 = ffma (fc (d * 24576 + c * 4 + 3)) (fw (dn_mix b + 8421376 + c * 4 + 2)) f1 in
    ax_fma p c2 w2 s1 s2 (fc (d * 24576 + c * 4 + 3)) (fw (dn_mix b + 8421376 + c * 4 + 2)) f1;
    let f3 = ffma (elt (atf (qkvq b)) c) (fw (dn_mix b + 8421376 + c * 4 + 3)) f2 in
    ax_fma p c3 w3 s2 s3 (elt (atf (qkvq b)) c) (fw (dn_mix b + 8421376 + c * 4 + 3)) f2;
    let aux (e : env) : Lemma (f3 e == cpre b d c e) = cpre_eq b d c e in
    FStar.Classical.forall_intro aux;
    ax_ext p s3 f3 (cpre b d c)

let co_fin (p : pred) (b d c : nat) (r : F32.t)
  : Lemma (requires c < 6144 /\ ax p r (fsilu (cpre b d c))) (ensures ax p r (elt (atf (coq b d)) c))
  = let aux (e : env) : Lemma (fsilu (cpre b d c) e == elt (atf (coq b d)) c e)
      = conv_out_pre opt_numerics e.rw b e.rx (convq d e) c in
    FStar.Classical.forall_intro aux;
    ax_ext p r (fsilu (cpre b d c)) (elt (atf (coq b d)) c)

(*** q / k normalization *)

let nscf (b d o : nat) (e : env) : real = rsqrt_t (sq_sum (coq b d e) o 128 +. eps_f32)

let nsc_ax (p : pred) (b d o : nat) (sq a rs : F32.t)
  : Lemma (requires ax p sq (ssum (4 * (1 * 32)) (sqt (atf (coq b d)) o)) /\
            (forall (xr yr : real). sq %~ xr /\ of_literal #F32.t "0.0000009999999974752427078783512115478515625" %~ yr ==> a %~ RO.r_add xr yr) /\
            (forall (xr : Sqrt.rpos). a %~ xr ==> rs %~ RO.r_rsqrt xr))
          (ensures ax p rs (nscf b d o))
  = lit_eps p;
    let g = ssum (4 * (1 * 32)) (sqt (atf (coq b d)) o) in
    ax_add p sq (of_literal #F32.t "0.0000009999999974752427078783512115478515625") a g (fconst eps_f32);
    let h = fadd g (fconst eps_f32) in
    let pos (e : env) : Lemma (h e >. 0.0R /\ h e == sq_sum (coq b d e) o 128 +. eps_f32)
      = sqt_sq0 (coq b d) o 128 e; sq_nonneg (coq b d e) o 128 in
    FStar.Classical.forall_intro pos;
    ax_rsq p a rs h;
    ax_ext p rs (frsq h) (nscf b d o)

let qsc_ax (p : pred) (b d o : nat) (rs qs : F32.t)
  : Lemma (requires ax p rs (nscf b d o) /\
            (forall (xr yr : real). rs %~ xr /\ of_literal #F32.t "0.088388346135616302490234375" %~ yr ==> qs %~ RO.r_mul xr yr))
          (ensures ax p qs (fmul (nscf b d o) (fconst qscale_f32)))
  = lit_qsc p;
    ax_mul p rs (of_literal #F32.t "0.088388346135616302490234375") qs (nscf b d o) (fconst qscale_f32)

#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
let q_elt (p : pred) (b d h i : nat) (x qs r : F32.t)
  : Lemma (requires h < 16 /\ i < 128 /\ ax p x (elt (atf (coq b d)) (h * 128 + i)) /\
            ax p qs (fmul (nscf b d (h * 128)) (fconst qscale_f32)) /\
            (forall (xr yr : real). x %~ xr /\ qs %~ yr ==> r %~ RO.r_mul xr yr))
          (ensures ax p r (elt (atfo (qq b d) (h * 128)) i))
  = ax_mul p x qs r (elt (atf (coq b d)) (h * 128 + i)) (fmul (nscf b d (h * 128)) (fconst qscale_f32));
    let aux (e : env) : Lemma (fmul (elt (atf (coq b d)) (h * 128 + i)) (fmul (nscf b d (h * 128)) (fconst qscale_f32)) e ==
                               elt (atfo (qq b d) (h * 128)) i e)
      = idx_hc h i;
        let x = at (coq b d e) (h * 128 + i) in
        let r = nscf b d (h * 128) e in
        assert (at (qq b d e) (h * 128 + i) == (x *. r) *. qscale_f32);
        assert (x *. (r *. qscale_f32) == (x *. r) *. qscale_f32) in
    FStar.Classical.forall_intro aux;
    ax_ext p r (fmul (elt (atf (coq b d)) (h * 128 + i)) (fmul (nscf b d (h * 128)) (fconst qscale_f32))) (elt (atfo (qq b d) (h * 128)) i)

#pop-options

let k_elt (p : pred) (b d h i : nat) (x ks r : F32.t)
  : Lemma (requires h < 16 /\ i < 128 /\ ax p x (elt (atf (coq b d)) (2048 + h * 128 + i)) /\
            ax p ks (nscf b d (2048 + h * 128)) /\
            (forall (xr yr : real). x %~ xr /\ ks %~ yr ==> r %~ RO.r_mul xr yr))
          (ensures ax p r (elt (atfo (kq b d) (h * 128)) i))
  = ax_mul p x ks r (elt (atf (coq b d)) (2048 + h * 128 + i)) (nscf b d (2048 + h * 128));
    let aux (e : env) : Lemma (fmul (elt (atf (coq b d)) (2048 + h * 128 + i)) (nscf b d (2048 + h * 128)) e ==
                               elt (atfo (kq b d) (h * 128)) i e)
      = idx_hc h i in
    FStar.Classical.forall_intro aux;
    ax_ext p r (fmul (elt (atf (coq b d)) (2048 + h * 128 + i)) (nscf b d (2048 + h * 128))) (elt (atfo (kq b d) (h * 128)) i)

(*** Delta rule *)

noextract let si (d h j : nat) : nat = d * 262144 + h * 16384 + j * 128

let rr_slot (d h i j : nat) (e : env)
  : Lemma (requires d < 18 /\ h < 16 /\ i < 128 /\ j < 128)
          (ensures at (stq d e) (h * 16384 + i * 128 + j) == rI e (si d h j + i))
  = idx3 h i j

let ring_err (dv s v : real) : Lemma ((0.0R -. dv) *. s +. v == v -. dv *. s) = ()

#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
let err_ax (p : pred) (b d h j : nat) (decay sk vj beta m1 err : F32.t)
  : Lemma (requires d < 18 /\ h < 16 /\ j < 128 /\ ax p decay (decf b h) /\
            ax p sk (ssum 128 (dt rI (si d h j) (atfo (kq b d) (h * 128)))) /\
            ax p vj (elt (atf (coq b d)) (4096 + h * 128 + j)) /\ ax p beta (betaf b h) /\
            (forall (xr yr zr : real). neg decay %~ xr /\ sk %~ yr /\ vj %~ zr ==> m1 %~ RO.r_fma xr yr zr) /\
            (forall (xr yr : real). m1 %~ xr /\ beta %~ yr ==> err %~ RO.r_mul xr yr))
          (ensures ax p err (deltaf b d h j))
  = let fsk = ssum 128 (dt rI (si d h j) (atfo (kq b d) (h * 128))) in
    let fv = elt (atf (coq b d)) (4096 + h * 128 + j) in
    ax_neg p decay (decf b h);
    ax_fma p (neg decay) sk vj m1 (fneg (decf b h)) fsk fv;
    ax_mul p m1 beta err (ffma (fneg (decf b h)) fsk fv) (betaf b h);
    let aux (e : env) : Lemma (fmul (ffma (fneg (decf b h)) fsk fv) (betaf b h) e == deltaf b d h j e)
      = delta_sum opt_numerics e.rw b e.rx (convq d e) (stq d e) h j;
        let inner (i : nat) : Lemma (requires i < 128)
            (ensures stk (stq d e) (kq b d e) h j i == dt rI (si d h j) (atfo (kq b d) (h * 128)) e i)
          = rr_slot d h i j e in
        FStar.Classical.forall_intro (FStar.Classical.move_requires inner);
        sumr_ext 128 (stk (stq d e) (kq b d e) h j) (dt rI (si d h j) (atfo (kq b d) (h * 128)) e);
        let dv = decf b h e in
        let s = fsk e in
        let v = fv e in
        assert (at (dn_v opt_numerics e.rw b e.rx (convq d e)) (h * 128 + j) == v);
        ring_err dv s v
    in
    FStar.Classical.forall_intro aux;
    ax_ext p err (fmul (ffma (fneg (decf b h)) fsk fv) (betaf b h)) (deltaf b d h j)

#pop-options

(**** from dev/KPlay.Conn2 ****)

#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
let scl (c : real) (f : nat -> real) (i : nat) : real = c *. f i

let st_val (b d h i j : nat) (e : env)
  : Lemma (requires d < 18 /\ h < 16 /\ i < 128 /\ j < 128)
          (ensures at (dn_state opt_numerics e.rw b e.rx (convq d e) (stq d e)) (h * 16384 + i * 128 + j) ==
                   rI e (si d h j + i) *. decf b h e +. at (kq b d e) (h * 128 + i) *. deltaf b d h j e)
  = idx3 h i j; rr_slot d h i j e

let rdist (a b c : real) : Lemma ((a +. b) *. c == a *. c +. b *. c) = ()
let rassoc (a b c : real) : Lemma ((a *. b) *. c == b *. (a *. c)) = ()

let ring_core (r dv k del qv : real)
  : Lemma ((r *. dv +. k *. del) *. qv == dv *. (r *. qv) +. del *. (k *. qv))
  = rdist (r *. dv) (k *. del) qv;
    rassoc r dv qv;
    rassoc k del qv

let core_term (b d h j i : nat) (e : env)
  : Lemma (requires d < 18 /\ h < 16 /\ j < 128 /\ i < 128)
          (ensures ctm (dn_state opt_numerics e.rw b e.rx (convq d e) (stq d e)) (qq b d e) (h * 128 + j) i ==
                   scl (decf b h e) (dt rI (si d h j) (atfo (qq b d) (h * 128)) e) i +.
                   scl (deltaf b d h j e) (dt (atfo (kq b d) (h * 128)) 0 (atfo (qq b d) (h * 128)) e) i)
  = idx_hc h j;
    st_val b d h i j e;
    ring_core (rI e (si d h j + i)) (decf b h e) (at (kq b d e) (h * 128 + i)) (deltaf b d h j e) (at (qq b d e) (h * 128 + i))

let core_val (b d h j : nat) (e : env)
  : Lemma (requires d < 18 /\ h < 16 /\ j < 128)
          (ensures elt (atfo (coreq b d) (h * 128)) j e ==
                   decf b h e *. ssum 128 (dt rI (si d h j) (atfo (qq b d) (h * 128))) e +.
                   deltaf b d h j e *. ssum 128 (dt (atfo (kq b d) (h * 128)) 0 (atfo (qq b d) (h * 128))) e)
  = let st' = dn_state opt_numerics e.rw b e.rx (convq d e) (stq d e) in
    let dv = decf b h e in
    let del = deltaf b d h j e in
    let f1 = dt rI (si d h j) (atfo (qq b d) (h * 128)) e in
    let f2 = dt (atfo (kq b d) (h * 128)) 0 (atfo (qq b d) (h * 128)) e in
    core_sum opt_numerics e.rw b e.rx (convq d e) (stq d e) (h * 128 + j);
    let inner (i : nat) : Lemma (requires i < 128) (ensures ctm st' (qq b d e) (h * 128 + j) i == scl dv f1 i +. scl del f2 i)
      = core_term b d h j i e in
    FStar.Classical.forall_intro (FStar.Classical.move_requires inner);
    sumr_scale 128 dv f1 (scl dv f1);
    sumr_scale 128 del f2 (scl del f2);
    sumr_add 128 (scl dv f1) (scl del f2) (ctm st' (qq b d e) (h * 128 + j))

let core_ax (p : pred) (b d h j : nat) (decay sq err kqv m2 cj : F32.t)
  : Lemma (requires d < 18 /\ h < 16 /\ j < 128 /\ ax p decay (decf b h) /\
            ax p sq (ssum 128 (dt rI (si d h j) (atfo (qq b d) (h * 128)))) /\ ax p err (deltaf b d h j) /\
            ax p kqv (ssum 128 (dt (atfo (kq b d) (h * 128)) 0 (atfo (qq b d) (h * 128)))) /\
            (forall (xr yr : real). err %~ xr /\ kqv %~ yr ==> m2 %~ RO.r_mul xr yr) /\
            (forall (xr yr zr : real). decay %~ xr /\ sq %~ yr /\ m2 %~ zr ==> cj %~ RO.r_fma xr yr zr))
          (ensures ax p cj (elt (atfo (coreq b d) (h * 128)) j))
  = let fsq = ssum 128 (dt rI (si d h j) (atfo (qq b d) (h * 128))) in
    let fkq = ssum 128 (dt (atfo (kq b d) (h * 128)) 0 (atfo (qq b d) (h * 128))) in
    ax_mul p err kqv m2 (deltaf b d h j) fkq;
    ax_fma p decay sq m2 cj (decf b h) fsq (fmul (deltaf b d h j) fkq);
    let aux (e : env) : Lemma (ffma (decf b h) fsq (fmul (deltaf b d h j) fkq) e == elt (atfo (coreq b d) (h * 128)) j e)
      = core_val b d h j e in
    FStar.Classical.forall_intro aux;
    ax_ext p cj (ffma (decf b h) fsq (fmul (deltaf b d h j) fkq)) (elt (atfo (coreq b d) (h * 128)) j)

let st_ax (p : pred) (b d h j i : nat) (s decay ki err m3 r : F32.t)
  : Lemma (requires d < 18 /\ h < 16 /\ j < 128 /\ i < 128 /\
            ax p s (elt rI (si d h j + i)) /\ ax p decay (decf b h) /\
            ax p ki (elt (atfo (kq b d) (h * 128)) i) /\ ax p err (deltaf b d h j) /\
            (forall (xr yr : real). ki %~ xr /\ err %~ yr ==> m3 %~ RO.r_mul xr yr) /\
            (forall (xr yr zr : real). s %~ xr /\ decay %~ yr /\ m3 %~ zr ==> r %~ RO.r_fma xr yr zr))
          (ensures ax p r (elt (atf (rcq b d)) (si d h j + i)))
  = ax_mul p ki err m3 (elt (atfo (kq b d) (h * 128)) i) (deltaf b d h j);
    let fm = fmul (elt (atfo (kq b d) (h * 128)) i) (deltaf b d h j) in
    ax_fma p s decay m3 r (elt rI (si d h j + i)) (decf b h) fm;
    let aux (e : env) : Lemma (ffma (elt rI (si d h j + i)) (decf b h) fm e == elt (atf (rcq b d)) (si d h j + i) e)
      = let m = h * 16384 + j * 128 + i in
        idx3 h j i;
        idx_slot d m;
        assert (si d h j + i == d * 262144 + m);
        st_val b d h i j e
    in
    FStar.Classical.forall_intro aux;
    ax_ext p r (ffma (elt rI (si d h j + i)) (decf b h) fm) (elt (atf (rcq b d)) (si d h j + i))
#pop-options

(**** from dev/KPlay.Conn3 ****)

(*** Gated RMSNorm *)

let gscf (b d h : nat) (e : env) : real = rsqrt_t (rdiv (sq_sum (coreq b d e) (h * 128) 128) 128.0R +. eps_f32)

let rdiv128 (x : real) : Lemma (x *. (1.0R /. 128.0R) == rdiv x 128.0R) = ()

let gsc_ax (p : pred) (b d h : nat) (ss m a rs : F32.t)
  : Lemma (requires ax p ss (ssum (1 * (4 * 32)) (sqt (atfo (coreq b d) (h * 128)) 0)) /\
            (forall (xr yr : real). ss %~ xr /\ of_literal #F32.t "0.0078125" %~ yr ==> m %~ RO.r_mul xr yr) /\
            (forall (xr yr : real). m %~ xr /\ of_literal #F32.t "0.0000009999999974752427078783512115478515625" %~ yr ==> a %~ RO.r_add xr yr) /\
            (forall (xr : Sqrt.rpos). a %~ xr ==> rs %~ RO.r_rsqrt xr))
          (ensures ax p rs (gscf b d h))
  = lit_i128 p; lit_eps p;
    let g0 = ssum (1 * (4 * 32)) (sqt (atfo (coreq b d) (h * 128)) 0) in
    ax_mul p ss (of_literal #F32.t "0.0078125") m g0 (fconst (1.0R /. 128.0R));
    let g = fmul g0 (fconst (1.0R /. 128.0R)) in
    ax_add p m (of_literal #F32.t "0.0000009999999974752427078783512115478515625") a g (fconst eps_f32);
    let hh = fadd g (fconst eps_f32) in
    let pos (e : env) : Lemma (hh e >. 0.0R /\ hh e == rdiv (sq_sum (coreq b d e) (h * 128) 128) 128.0R +. eps_f32)
      = sqt_sq (coreq b d) (h * 128) 0 128 e; sq_nonneg (coreq b d e) (h * 128) 128;
        let sv = sq_sum (coreq b d e) (h * 128) 128 in
        assert (g0 e == sv);
        rdiv128 sv;
        assert (g e == rdiv sv 128.0R);
        assert (rdiv sv 128.0R >=. 0.0R) in
    FStar.Classical.forall_intro pos;
    ax_rsq p a rs hh;
    ax_ext p rs (frsq hh) (gscf b d h)

let rcomm (a b : real) : Lemma (a *. b == b *. a) = ()

#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
let gate_elt (p : pred) (b d h i : nat) (c rs zs m1 m2 r : F32.t) (gv : BF.t)
  : Lemma (requires d < 18 /\ h < 16 /\ i < 128 /\ ax p c (elt (atfo (coreq b d) (h * 128)) i) /\
            ax p rs (gscf b d h) /\ ax p (wcast gv) (fw (dn_mix b + 8445984 + i)) /\
            ax p zs (fsilu (elt (atf (zq b)) (h * 128 + i))) /\
            (forall (xr yr : real). c %~ xr /\ rs %~ yr ==> m1 %~ RO.r_mul xr yr) /\
            (forall (xr yr : real). m1 %~ xr /\ wcast gv %~ yr ==> m2 %~ RO.r_mul xr yr) /\
            (forall (xr yr : real). m2 %~ xr /\ zs %~ yr ==> r %~ RO.r_mul xr yr))
          (ensures ax p r (elt (atf (gatedq b d)) (h * 128 + i)))
  = let fc0 = elt (atfo (coreq b d) (h * 128)) i in
    let fz = fsilu (elt (atf (zq b)) (h * 128 + i)) in
    let fg = fw (dn_mix b + 8445984 + i) in
    ax_mul p c rs m1 fc0 (gscf b d h);
    ax_mul p m1 (wcast gv) m2 (fmul fc0 (gscf b d h)) fg;
    ax_mul p m2 zs r (fmul (fmul fc0 (gscf b d h)) fg) fz;
    let aux (e : env) : Lemma (fmul (fmul (fmul fc0 (gscf b d h)) fg) fz e == elt (atf (gatedq b d)) (h * 128 + i) e)
      = idx_hc h i;
        let cv = at (coreq b d e) (h * 128 + i) in
        let iv = gscf b d h e in
        let gw = at e.rw (dn_mix b + 8445984 + i) in
        let sv = silu (at (zq b e) (h * 128 + i)) in
        assert (at (gatedq b d e) (h * 128 + i) == (gw *. (cv *. iv)) *. sv);
        rcomm (cv *. iv) gw
    in
    FStar.Classical.forall_intro aux;
    ax_ext p r (fmul (fmul (fmul fc0 (gscf b d h)) fg) fz) (elt (atf (gatedq b d)) (h * 128 + i))
#pop-options

(*** Final assembly *)

let seq_of_axr (p : pred) (s : S.seq F32.t) (q : env -> S.seq real) (n : nat) (e : env)
  : Lemma (requires axr p s (atf q) 0 n /\ S.length s == n /\ S.length (q e) == n /\ p e)
          (ensures s %~ q e)
  = axr_def p s (atf q) 0 n;
    let aux (i : nat) : Lemma (i < n ==> (i < S.length s /\ i < S.length (q e) /\ S.index s i %~ S.index (q e) i))
      = if i < n then ax_elim p (S.index s i) (elt (atf q) i) e in
    FStar.Classical.forall_intro aux

let slot_rng (i d k : nat)
  : Lemma (requires k > 0 /\ i / k == d) (ensures d * k <= i /\ i < d * k + k /\ i % k == i - d * k)
  = FStar.Math.Lemmas.lemma_div_mod i k;
    FStar.Math.Lemmas.lemma_mod_lt i k

let idx_slot24 (d j : nat)
  : Lemma (requires j < 24576) (ensures (d * 24576 + j) / 24576 == d /\ (d * 24576 + j) % 24576 == j)
  = FStar.Math.Lemmas.lemma_div_plus j d 24576;
    FStar.Math.Lemmas.small_div j 24576;
    FStar.Math.Lemmas.lemma_mod_plus j d 24576;
    FStar.Math.Lemmas.small_mod j 24576

(* Exact-copy invariants of the state outputs *)

[@@"opaque_to_smt"]
let rco_pre (d : nat) (rs rco : S.seq F32.t) : prop =
  d < 18 /\ S.length rs == 4718592 /\ S.length rco == 4718592 /\
  (forall (i : nat). i < 4718592 /\ i / 262144 <> d ==> S.index rco i == S.index rs i)

let rco_pre_def (d : nat) (rs rco : S.seq F32.t)
  : Lemma (rco_pre d rs rco <==> (d < 18 /\ S.length rs == 4718592 /\ S.length rco == 4718592 /\
            (forall (i : nat). i < 4718592 /\ i / 262144 <> d ==> S.index rco i == S.index rs i)))
  = assert_norm (rco_pre d rs rco == (d < 18 /\ S.length rs == 4718592 /\ S.length rco == 4718592 /\
            (forall (i : nat). i < 4718592 /\ i / 262144 <> d ==> S.index rco i == S.index rs i)))

let rco_init (d : nat) (rs : S.seq F32.t)
  : Lemma (requires d < 18 /\ S.length rs == 4718592) (ensures rco_pre d rs rs)
  = rco_pre_def d rs rs

let rco_upd (d : nat) (rs rco : S.seq F32.t) (j : nat) (v : F32.t)
  : Lemma (requires rco_pre d rs rco /\ S.length rco == 4718592 /\ j < 4718592 /\ j / 262144 == d) (ensures rco_pre d rs (S.upd rco j v))
  = rco_pre_def d rs rco; rco_pre_def d rs (S.upd rco j v)

let rco_len (d : nat) (rs rco : S.seq F32.t)
  : Lemma (requires rco_pre d rs rco) (ensures d < 18 /\ S.length rs == 4718592 /\ S.length rco == 4718592)
  = rco_pre_def d rs rco

let rco_get (d : nat) (rs rco : S.seq F32.t) (i : nat)
  : Lemma (requires rco_pre d rs rco /\ i < 4718592 /\ i / 262144 <> d)
          (ensures i < S.length rco /\ i < S.length rs /\ S.index rco i == S.index rs i)
  = rco_pre_def d rs rco

let cvo_body (d : nat) (cs cvo proj : S.seq F32.t) (cc : nat) : prop =
  d < 18 /\ cc <= 6144 /\ S.length cs == 442368 /\ S.length cvo == 442368 /\ S.length proj == 6144 /\
  (forall (i : nat). i < 442368 /\ i / 24576 <> d ==> S.index cvo i == S.index cs i) /\
  (forall (c k : nat). c < cc /\ k < 3 ==> S.index cvo (d * 24576 + c * 4 + k) == S.index cs (d * 24576 + c * 4 + k + 1)) /\
  (forall (c : nat). c < cc ==> S.index cvo (d * 24576 + c * 4 + 3) == S.index proj c)

[@@"opaque_to_smt"]
let cvo_pre (d : nat) (cs cvo proj : S.seq F32.t) (cc : nat) : prop = cvo_body d cs cvo proj cc

let cvo_pre_def (d : nat) (cs cvo proj : S.seq F32.t) (cc : nat)
  : Lemma (cvo_pre d cs cvo proj cc <==> cvo_body d cs cvo proj cc)
  = assert_norm (cvo_pre d cs cvo proj cc == cvo_body d cs cvo proj cc)

let cvo_init (d : nat) (cs proj : S.seq F32.t)
  : Lemma (requires d < 18 /\ S.length cs == 442368 /\ S.length proj == 6144) (ensures cvo_pre d cs cs proj 0)
  = cvo_pre_def d cs cs proj 0

let cvo_len (d : nat) (cs cvo proj : S.seq F32.t) (cc : nat)
  : Lemma (requires cvo_pre d cs cvo proj cc)
          (ensures d < 18 /\ cc <= 6144 /\ S.length cs == 442368 /\ S.length cvo == 442368 /\ S.length proj == 6144)
  = cvo_pre_def d cs cvo proj cc

let cvo_get_out (d : nat) (cs cvo proj : S.seq F32.t) (cc i : nat)
  : Lemma (requires cvo_pre d cs cvo proj cc /\ i < 442368 /\ i / 24576 <> d)
          (ensures i < S.length cvo /\ i < S.length cs /\ S.index cvo i == S.index cs i)
  = cvo_pre_def d cs cvo proj cc

let cvo_get_lo (d : nat) (cs cvo proj : S.seq F32.t) (cc c k : nat)
  : Lemma (requires cvo_pre d cs cvo proj cc /\ c < cc /\ k < 3)
          (ensures d * 24576 + c * 4 + k + 1 < 442368 /\ S.length cvo == 442368 /\ S.length cs == 442368 /\
                   S.index cvo (d * 24576 + c * 4 + k) == S.index cs (d * 24576 + c * 4 + k + 1))
  = cvo_pre_def d cs cvo proj cc

let cvo_get_hi (d : nat) (cs cvo proj : S.seq F32.t) (cc c : nat)
  : Lemma (requires cvo_pre d cs cvo proj cc /\ c < cc)
          (ensures c < 6144 /\ d * 24576 + c * 4 + 3 < 442368 /\ S.length cvo == 442368 /\ S.length proj == 6144 /\
                   S.index cvo (d * 24576 + c * 4 + 3) == S.index proj c)
  = cvo_pre_def d cs cvo proj cc

let slot_c4 (d c k : nat)
  : Lemma (requires d < 18 /\ c < 6144 /\ k < 4)
          (ensures d * 24576 + c * 4 + k < 442368 /\ (d * 24576 + c * 4 + k) / 24576 == d)
  = idx_slot24 d (c * 4 + k)

#push-options "--z3rlimit 20"
let cvo_step (d : nat) (cs cvo proj : S.seq F32.t) (cc : nat) (c0 c1 c2 c3 : F32.t)
  : Lemma (requires cvo_pre d cs cvo proj cc /\ cc < 6144 /\
            d * 24576 + cc * 4 + 3 < 442368 /\ S.length cs == 442368 /\ S.length proj == 6144 /\
            c0 == S.index cs (d * 24576 + cc * 4 + 1) /\ c1 == S.index cs (d * 24576 + cc * 4 + 2) /\
            c2 == S.index cs (d * 24576 + cc * 4 + 3) /\ c3 == S.index proj cc)
          (ensures (S.length cvo == 442368 /\
                    cvo_pre d cs (S.upd (S.upd (S.upd (S.upd cvo (d * 24576 + cc * 4) c0) (d * 24576 + cc * 4 + 1) c1)
                                         (d * 24576 + cc * 4 + 2) c2) (d * 24576 + cc * 4 + 3) c3) proj (cc + 1)))
  = cvo_pre_def d cs cvo proj cc;
    let base = d * 24576 + cc * 4 in
    let cvo' = S.upd (S.upd (S.upd (S.upd cvo base c0) (base + 1) c1) (base + 2) c2) (base + 3) c3 in
    slot_c4 d cc 0; slot_c4 d cc 1; slot_c4 d cc 2; slot_c4 d cc 3;
    assert (S.length cvo' == 442368);
    let out (i : nat) : Lemma (i < 442368 /\ i / 24576 <> d ==> S.index cvo' i == S.index cs i)
      = () in
    FStar.Classical.forall_intro out;
    let lo (c k : nat) : Lemma (c < cc + 1 /\ k < 3 ==>
        (d * 24576 + c * 4 + k + 1 < 442368 /\ S.index cvo' (d * 24576 + c * 4 + k) == S.index cs (d * 24576 + c * 4 + k + 1)))
      = () in
    FStar.Classical.forall_intro_2 lo;
    let hi (c : nat) : Lemma (c < cc + 1 ==>
        (d * 24576 + c * 4 + 3 < 442368 /\ S.index cvo' (d * 24576 + c * 4 + 3) == S.index proj c))
      = () in
    FStar.Classical.forall_intro hi;
    cvo_pre_def d cs cvo' proj (cc + 1)
#pop-options

let rco_fin (p : pred) (rs rco : S.seq F32.t) (b d : nat) (e : env)
  : Lemma (requires rok p rs /\ p e /\ rco_pre d rs rco /\
            axr p rco (atf (rcq b d)) (d * 262144) (d * 262144 + 262144))
          (ensures rco %~ rcq b d e)
  = rco_len d rs rco;
    let aux (i : nat) : Lemma (i < 4718592 ==> (i < S.length rco /\ i < S.length (rcq b d e) /\ S.index rco i %~ S.index (rcq b d e) i))
      = if i >= 4718592 then () else if i / 262144 = d then begin
          slot_rng i d 262144;
          axr_get p rco (atf (rcq b d)) (d * 262144) (d * 262144 + 262144) i;
          ax_elim p (S.index rco i) (elt (atf (rcq b d)) i) e
        end else begin
          rco_get d rs rco i;
          ax_r p rs i;
          ax_elim p (S.index rs i) (fr i) e
        end
    in
    FStar.Classical.forall_intro aux

let cv_in (b d j : nat) (e : env)
  : Lemma (requires d < 18 /\ j < 24576)
          (ensures at (cvq b d e) (d * 24576 + j) ==
                   (if j % 4 < 3 then at e.rcv (d * 24576 + j + 1) else at (qkvq b e) (j / 4)))
  = idx_slot24 d j

let cv_out (b d i : nat) (e : env)
  : Lemma (requires d < 18 /\ i < 442368 /\ i / 24576 <> d) (ensures at (cvq b d e) i == at e.rcv i)
  = ()

let cvo_lo (p : pred) (cs : S.seq F32.t) (b d j : nat) (e : env) (v : F32.t)
  : Lemma (requires d < 18 /\ j < 24576 /\ j % 4 < 3 /\ cok p cs /\ p e /\ S.length cs == 442368 /\
            v == S.index cs (d * 24576 + j + 1))
          (ensures v %~ at (cvq b d e) (d * 24576 + j))
  = cv_in b d j e;
    ax_c p cs (d * 24576 + j + 1);
    ax_elim p (S.index cs (d * 24576 + j + 1)) (fc (d * 24576 + j + 1)) e

let cvo_hi (p : pred) (proj : S.seq F32.t) (b d j : nat) (e : env) (v : F32.t)
  : Lemma (requires d < 18 /\ j < 24576 /\ j % 4 = 3 /\ p e /\ S.length proj == 6144 /\
            axr p proj (atf (qkvq b)) 0 6144 /\ v == S.index proj (j / 4))
          (ensures v %~ at (cvq b d e) (d * 24576 + j))
  = cv_in b d j e;
    axr_get p proj (atf (qkvq b)) 0 6144 (j / 4);
    ax_elim p (S.index proj (j / 4)) (elt (atf (qkvq b)) (j / 4)) e

let cvo_other (p : pred) (cs : S.seq F32.t) (b d i : nat) (e : env) (v : F32.t)
  : Lemma (requires d < 18 /\ i < 442368 /\ i / 24576 <> d /\ cok p cs /\ p e /\ S.length cs == 442368 /\
            v == S.index cs i)
          (ensures v %~ at (cvq b d e) i)
  = cv_out b d i e;
    ax_c p cs i;
    ax_elim p (S.index cs i) (fc i) e

let j_split (j : nat)
  : Lemma (j == (j / 4) * 4 + j % 4 /\ j % 4 < 4)
  = FStar.Math.Lemmas.lemma_div_mod j 4; FStar.Math.Lemmas.lemma_mod_lt j 4

let cvo_rel_lo (d : nat) (cs cvo proj : S.seq F32.t) (j : nat)
  : Lemma (requires cvo_pre d cs cvo proj 6144 /\ j < 24576 /\ j % 4 < 3)
          (ensures d * 24576 + j + 1 < 442368 /\ S.length cvo == 442368 /\ S.length cs == 442368 /\
                   S.index cvo (d * 24576 + j) == S.index cs (d * 24576 + j + 1))
  = j_split j; cvo_get_lo d cs cvo proj 6144 (j / 4) (j % 4)

let cvo_rel_hi (d : nat) (cs cvo proj : S.seq F32.t) (j : nat)
  : Lemma (requires cvo_pre d cs cvo proj 6144 /\ j < 24576 /\ j % 4 = 3)
          (ensures j / 4 < 6144 /\ d * 24576 + j < 442368 /\ S.length cvo == 442368 /\ S.length proj == 6144 /\
                   S.index cvo (d * 24576 + j) == S.index proj (j / 4))
  = j_split j; cvo_get_hi d cs cvo proj 6144 (j / 4)

let cvo_elt (p : pred) (cs cvo proj : S.seq F32.t) (b d : nat) (e : env) (i : nat)
  : Lemma (requires cok p cs /\ p e /\ axr p proj (atf (qkvq b)) 0 6144 /\ i < 442368 /\
            cvo_pre d cs cvo proj 6144)
          (ensures i < S.length cvo /\ S.index cvo i %~ at (cvq b d e) i)
  = cvo_len d cs cvo proj 6144;
    if i / 24576 = d then begin
      slot_rng i d 24576;
      let j : nat = i % 24576 in
      assert (i == d * 24576 + j);
      j_split j;
      if j % 4 < 3 then begin
        cvo_rel_lo d cs cvo proj j;
        cvo_lo p cs b d j e (S.index cvo i)
      end else begin
        cvo_rel_hi d cs cvo proj j;
        cvo_hi p proj b d j e (S.index cvo i)
      end
    end else begin
      cvo_get_out d cs cvo proj 6144 i;
      cvo_other p cs b d i e (S.index cvo i)
    end

let cvo_fin (p : pred) (cs cvo proj : S.seq F32.t) (b d : nat) (e : env)
  : Lemma (requires cok p cs /\ p e /\ axr p proj (atf (qkvq b)) 0 6144 /\ cvo_pre d cs cvo proj 6144)
          (ensures cvo %~ cvq b d e)
  = cvo_len d cs cvo proj 6144;
    let aux (i : nat) : Lemma (i < 442368 ==> (i < S.length cvo /\ i < S.length (cvq b d e) /\ S.index cvo i %~ S.index (cvq b d e) i))
      = if i < 442368 then cvo_elt p cs cvo proj b d e i in
    FStar.Classical.forall_intro aux

let rel_fin (ws : S.seq BF.t) (xs rs cs xo rco cvo proj : S.seq F32.t) (b d : nat)
  : Lemma (requires S.length xs == 1024 /\ S.length xo == 1024 /\
            axr (okp ws xs rs cs) xo (atf (outq b d)) 0 1024 /\
            axr (okp ws xs rs cs) proj (atf (qkvq b)) 0 6144 /\
            rco_pre d rs rco /\
            axr (okp ws xs rs cs) rco (atf (rcq b d)) (d * 262144) (d * 262144 + 262144) /\
            cvo_pre d cs cvo proj 6144)
          (ensures optimized_recurrent_decoder_relation ws xs rs cs b d xo rco cvo)
  = let p = okp ws xs rs cs in
    okp_ok ws xs rs cs;
    rco_len d rs rco;
    cvo_len d cs cvo proj 6144;
    let r1 (i : nat) : Lemma (i < opt_rec_len /\ i / 262144 <> d ==> S.index rco i == S.index rs i)
      = if i < opt_rec_len && i / 262144 <> d then rco_get d rs rco i in
    FStar.Classical.forall_intro r1;
    let r2 (i : nat) : Lemma (i < opt_conv_len /\ i / 24576 <> d ==> S.index cvo i == S.index cs i)
      = if i < opt_conv_len && i / 24576 <> d then cvo_get_out d cs cvo proj 6144 i in
    FStar.Classical.forall_intro r2;
    let r3 (j : nat) : Lemma (j + 1 < 24576 /\ j % 4 < 3 ==> S.index cvo (d * 24576 + j) == S.index cs (d * 24576 + j + 1))
      = if j + 1 < 24576 && j % 4 < 3 then cvo_rel_lo d cs cvo proj j in
    FStar.Classical.forall_intro r3;
    let aux (rw rx rr rcv : S.seq real)
      : Lemma (requires ws %~ rw /\ xs %~ rx /\ rs %~ rr /\ cs %~ rcv)
              (ensures xo %~ dn_out opt_numerics rw b rx (opt_conv_slot rcv d) (opt_rec_slot rr d) /\
                       cvo %~ opt_conv_put rcv d (dn_conv opt_numerics rw b rx (opt_conv_slot rcv d)) /\
                       rco %~ opt_rec_put rr d (dn_state opt_numerics rw b rx (opt_conv_slot rcv d) (opt_rec_slot rr d)))
      = let e : env = { rw = rw; rx = rx; rr = rr; rcv = rcv } in
        assert (p e);
        seq_of_axr p xo (outq b d) 1024 e;
        rco_fin p rs rco b d e;
        cvo_fin p cs cvo proj b d e
    in
    FStar.Classical.forall_intro_4 (fun rw rx rr rcv -> FStar.Classical.move_requires (aux rw rx rr) rcv)

(**** from dev/KPlay.Kern ****)

let lvl_ok (dv : nat) : prop = dv == 16 \/ dv == 8 \/ dv == 4 \/ dv == 2 \/ dv == 1 \/ dv == 0
noextract let dd_of (dv : nat) : nat = if dv = 0 then 1 else 2 * dv

inline_for_extraction noextract
fn tree32 (p : erased pred) (l : erased af) (#s : erased (Seq.seq F32.t)) (a : A.array F32.t)
  preserves gpu
  requires a |-> s ** pure (Seq.length s == 32 /\ axr p s l 0 32)
  returns r : F32.t
  ensures exists* s'. a |-> s' ** pure (Seq.length s' == 32 /\ ax p r (ssum 32 l))
{
  tla_init p s l;
  let mut d = 16sz;
  while (!d >^ 0sz)
    invariant exists* (dv : sz). d |-> dv ** pure (lvl_ok (SZ.v dv))
    invariant exists* v. a |-> v ** pure (Seq.length v == 32 /\ axr p v (tla l (dd_of (SZ.v !d))) 0 (dd_of (SZ.v !d)))
    decreases (SZ.v !d)
  {
    let dv = !d;
    with v0. assert a |-> v0;
    axr_empty p v0 (tla l (SZ.v dv)) 0;
    let mut i = 0sz;
    while (!i <^ dv)
      invariant exists* (iv : sz{iv <= dv}). i |-> iv
      invariant exists* v. a |-> v ** pure (Seq.length v == 32 /\ axr p v (tla l (SZ.v dv)) 0 (SZ.v !i) /\
                                           axr p v (tla l (2 * SZ.v dv)) (SZ.v !i) (2 * SZ.v dv))
      decreases (SZ.v dv - SZ.v !i)
    {
      let iv = !i;
      with v. assert a |-> v;
      axr_get p v (tla l (2 * SZ.v dv)) (SZ.v iv) (2 * SZ.v dv) (SZ.v iv);
      axr_get p v (tla l (2 * SZ.v dv)) (SZ.v iv) (2 * SZ.v dv) (SZ.v iv + SZ.v dv);
      let x = a.(iv);
      let y = a.(iv +^ dv);
      let r = F32.add_rn_ftz x y;
      tla_step p x y r l (SZ.v dv) (SZ.v iv);
      axr_snoc p v (tla l (SZ.v dv)) 0 (SZ.v iv) r;
      axr_mono p v (tla l (2 * SZ.v dv)) (SZ.v iv) (2 * SZ.v dv) (SZ.v iv + 1) (2 * SZ.v dv);
      axr_frame p v (tla l (2 * SZ.v dv)) (SZ.v iv + 1) (2 * SZ.v dv) (SZ.v iv) r;
      a.(iv) <- r;
      i := iv +^ 1sz;
    };
    d := dv /^ 2sz;
  };
  with v. assert a |-> v;
  axr_get p v (tla l 1) 0 1 0;
  let r = a.(0sz);
  tla_final p r l;
  r
}

let sq_idx (xo g l t nw c : nat)
  : Lemma (requires g < nw /\ l < 32 /\ t < c)
          (ensures xo + g * 32 + l + t * (nw * 32) < xo + c * (nw * 32) /\
                   xo + g * 32 + l + t * (nw * 32) == xo + (t * (nw * 32) + (g * 32 + l)))
  = FStar.Math.Lemmas.lemma_mult_le_right (nw * 32) (t + 1) c

(* Sum of squares of x[xo + i], i < cnt * nw * 32, over [nw] groups of 32 lanes;
   lane (g, l) accumulates [cnt] terms at stride nw * 32. *)
inline_for_extraction noextract
fn sqsum (p : erased pred) (xi : erased af) (#s : erased (Seq.seq F32.t)) (x : A.array F32.t) (xo cnt nw : sz) (single : bool)
  preserves gpu ** x |-> s
  requires pure (SZ.v nw <= 32 /\ SZ.v nw > 0 /\ SZ.v cnt > 0 /\ (single ==> SZ.v nw == 1) /\
                 SZ.v xo + SZ.v cnt * (SZ.v nw * 32) <= Seq.length s /\
                 axr p s xi (SZ.v xo) (SZ.v xo + SZ.v cnt * (SZ.v nw * 32)))
  returns r : F32.t
  ensures pure (ax p r (ssum (SZ.v cnt * (SZ.v nw * 32)) (sqt xi (SZ.v xo))))
{
  pts_to_len x;
  let st = nw *^ 32sz;
  let mut warps = [| zero #F32.t #_; 32sz |];
  let mut lanes = [| zero #F32.t #_; 32sz |];
  axr_zero p 32;
  axr_empty p (Seq.create 32 (zero #F32.t)) (wz (sqt xi (SZ.v xo)) (SZ.v st) (SZ.v cnt) (SZ.v nw)) 0;
  let mut g = 0sz;
  while (!g <^ nw)
    invariant exists* (gv : sz{gv <= nw}). g |-> gv
    invariant exists* v. warps |-> v ** pure (Seq.length v == 32 /\
      axr p v (wz (sqt xi (SZ.v xo)) (SZ.v st) (SZ.v cnt) (SZ.v nw)) 0 (SZ.v !g) /\ axr p v zf (SZ.v !g) 32)
    invariant exists* v. lanes |-> v ** pure (Seq.length v == 32)
    decreases (SZ.v nw - SZ.v !g)
  {
    let gv = !g;
    with lv0. assert lanes |-> lv0;
    axr_empty p lv0 (lg (sqt xi (SZ.v xo)) (SZ.v st) (SZ.v cnt) (SZ.v gv)) 0;
    let mut l = 0sz;
    while (!l <^ 32sz)
      invariant exists* (lv : sz{lv <= 32}). l |-> lv
      invariant exists* v. lanes |-> v ** pure (Seq.length v == 32 /\
        axr p v (lg (sqt xi (SZ.v xo)) (SZ.v st) (SZ.v cnt) (SZ.v gv)) 0 (SZ.v !l))
      decreases (32 - SZ.v !l)
    {
      let lv = !l;
      let mut acc = zero #F32.t #_;
      lacc_zero p (sqt xi (SZ.v xo)) (SZ.v st) (SZ.v gv * 32 + SZ.v lv);
      let mut t = 0sz;
      while (!t <^ cnt)
        invariant exists* (tv : sz{tv <= cnt}). t |-> tv
        invariant exists* av. acc |-> av ** pure (ax p av (lacc (sqt xi (SZ.v xo)) (SZ.v st) (SZ.v gv * 32 + SZ.v lv) (SZ.v !t)))
        decreases (SZ.v cnt - SZ.v !t)
      {
        let tv = !t;
        sq_idx (SZ.v xo) (SZ.v gv) (SZ.v lv) (SZ.v tv) (SZ.v nw) (SZ.v cnt);
        let idx = xo +^ gv *^ 32sz +^ lv +^ tv *^ st;
        axr_get p s xi (SZ.v xo) (SZ.v xo + SZ.v cnt * (SZ.v nw * 32)) (SZ.v idx);
        let e = x.(idx);
        let av = !acc;
        let r = F32.fma_rn_ftz e e av;
        lacc_step p e e av r (elt xi (SZ.v idx)) (elt xi (SZ.v idx)) (sqt xi (SZ.v xo)) (SZ.v st) (SZ.v gv * 32 + SZ.v lv) (SZ.v tv);
        acc := r;
        t := tv +^ 1sz;
      };
      let av = !acc;
      lg_elt p av (sqt xi (SZ.v xo)) (SZ.v st) (SZ.v cnt) (SZ.v gv) (SZ.v lv);
      with lv1. assert lanes |-> lv1;
      axr_snoc p lv1 (lg (sqt xi (SZ.v xo)) (SZ.v st) (SZ.v cnt) (SZ.v gv)) 0 (SZ.v lv) av;
      lanes.(lv) <- av;
      l := lv +^ 1sz;
    };
    let r = tree32 p (lg (sqt xi (SZ.v xo)) (SZ.v st) (SZ.v cnt) (SZ.v gv)) lanes;
    wz_elt p r (sqt xi (SZ.v xo)) (SZ.v st) (SZ.v cnt) (SZ.v nw) (SZ.v gv);
    with wv1. assert warps |-> wv1;
    axr_snoc p wv1 (wz (sqt xi (SZ.v xo)) (SZ.v st) (SZ.v cnt) (SZ.v nw)) 0 (SZ.v gv) r;
    axr_mono p wv1 zf (SZ.v gv) 32 (SZ.v gv + 1) 32;
    axr_frame p wv1 zf (SZ.v gv + 1) 32 (SZ.v gv) r;
    warps.(gv) <- r;
    g := gv +^ 1sz;
  };
  with wv. assert warps |-> wv;
  if single {
    axr_get p wv (wz (sqt xi (SZ.v xo)) (SZ.v st) (SZ.v cnt) (SZ.v nw)) 0 1 0;
    let r = warps.(0sz);
    wz_single p r (sqt xi (SZ.v xo)) (SZ.v cnt);
    lanes1_final p r (sqt xi (SZ.v xo)) (SZ.v cnt);
    r
  } else {
    wz_zero p wv (sqt xi (SZ.v xo)) (SZ.v st) (SZ.v cnt) (SZ.v nw);
    axr_join p wv (wz (sqt xi (SZ.v xo)) (SZ.v st) (SZ.v cnt) (SZ.v nw)) 0 (SZ.v nw) 32;
    let r = tree32 p (wz (sqt xi (SZ.v xo)) (SZ.v st) (SZ.v cnt) (SZ.v nw)) warps;
    warps_final p r (sqt xi (SZ.v xo)) (SZ.v cnt) (SZ.v nw);
    r
  }
}

(* sum_i a[ao + i] * b[i] over 128, 32 lanes with stride 32 *)
inline_for_extraction noextract
fn dot128 (p : erased pred) (ai bi : erased af) (#as_ : erased (Seq.seq F32.t)) (#bs : erased (Seq.seq F32.t))
  (a : A.array F32.t) (ao : sz) (b : A.array F32.t)
  preserves gpu ** a |-> as_ ** b |-> bs
  requires pure (SZ.v ao + 128 <= Seq.length as_ /\ SZ.fits (SZ.v ao + 128) /\ Seq.length bs == 128 /\
                 axr p as_ ai (SZ.v ao) (SZ.v ao + 128) /\ axr p bs bi 0 128)
  returns r : F32.t
  ensures pure (ax p r (ssum 128 (dt ai (SZ.v ao) bi)))
{
  let mut lanes = [| zero #F32.t #_; 32sz |];
  axr_empty p (Seq.create 32 (zero #F32.t)) (lg (dt ai (SZ.v ao) bi) 32 4 0) 0;
  let mut l = 0sz;
  while (!l <^ 32sz)
    invariant exists* (lv : sz{lv <= 32}). l |-> lv
    invariant exists* v. lanes |-> v ** pure (Seq.length v == 32 /\ axr p v (lg (dt ai (SZ.v ao) bi) 32 4 0) 0 (SZ.v !l))
    decreases (32 - SZ.v !l)
  {
    let lv = !l;
    let mut acc = zero #F32.t #_;
    lacc_zero p (dt ai (SZ.v ao) bi) 32 (0 * 32 + SZ.v lv);
    let mut t = 0sz;
    while (!t <^ 4sz)
      invariant exists* (tv : sz{tv <= 4}). t |-> tv
      invariant exists* av. acc |-> av ** pure (ax p av (lacc (dt ai (SZ.v ao) bi) 32 (0 * 32 + SZ.v lv) (SZ.v !t)))
      decreases (4 - SZ.v !t)
    {
      let tv = !t;
      let idx = lv +^ tv *^ 32sz;
      axr_get p as_ ai (SZ.v ao) (SZ.v ao + 128) (SZ.v ao + SZ.v idx);
      axr_get p bs bi 0 128 (SZ.v idx);
      let x = a.(ao +^ idx);
      let y = b.(idx);
      let av = !acc;
      let r = F32.fma_rn_ftz x y av;
      lacc_step p x y av r (elt ai (SZ.v ao + SZ.v idx)) (elt bi (SZ.v idx)) (dt ai (SZ.v ao) bi) 32 (0 * 32 + SZ.v lv) (SZ.v tv);
      acc := r;
      t := tv +^ 1sz;
    };
    let av = !acc;
    lg_elt p av (dt ai (SZ.v ao) bi) 32 4 0 (SZ.v lv);
    with lv1. assert lanes |-> lv1;
    axr_snoc p lv1 (lg (dt ai (SZ.v ao) bi) 32 4 0) 0 (SZ.v lv) av;
    lanes.(lv) <- av;
    l := lv +^ 1sz;
  };
  let r = tree32 p (lg (dt ai (SZ.v ao) bi) 32 4 0) lanes;
  lanes1_final p r (dt ai (SZ.v ao) bi) 4;
  r
}

inline_for_extraction noextract
fn wacc_next (p : erased pred) (t : erased af) (l m : erased nat) (grouped : bool) (a0 pv : F32.t)
  preserves gpu
  requires pure (ax p a0 (wacc t l m) /\ ax p pv (wpart t grouped l m 8))
  returns r : F32.t
  ensures pure (ax p r (wacc t l (m + 1)))
{
  if grouped {
    let r = F32.add_rn_ftz a0 pv;
    wacc_grouped p a0 pv r t l m;
    r
  } else {
    wacc_cont p pv t l m;
    pv
  }
}

(* 32 lanes; lane l reads blocks of 8 at l*8 + 256*m, m < nm. *)
inline_for_extraction noextract
fn wdot (p : erased pred) (xi : erased af) (#ws : erased (Seq.seq BF.t)) (#xs : erased (Seq.seq F32.t))
  (grouped : bool) (w : A.array BF.t) (wo : sz) (x : A.array F32.t) (nm : sz)
  preserves gpu ** w |-> ws ** x |-> xs
  requires pure (SZ.v nm * 256 <= Seq.length xs /\ SZ.v wo + SZ.v nm * 256 <= Seq.length ws /\
                 wok p ws /\ axr p xs xi 0 (SZ.v nm * 256))
  returns r : F32.t
  ensures pure (ax p r (ssum (SZ.v nm * 256) (wt (SZ.v wo) xi)))
{
  pts_to_len w;
  pts_to_len x;
  let mut lanes = [| zero #F32.t #_; 32sz |];
  axr_empty p (Seq.create 32 (zero #F32.t)) (wl (wt (SZ.v wo) xi) (SZ.v nm)) 0;
  let mut l = 0sz;
  while (!l <^ 32sz)
    invariant exists* (lv : sz{lv <= 32}). l |-> lv
    invariant exists* v. lanes |-> v ** pure (Seq.length v == 32 /\ axr p v (wl (wt (SZ.v wo) xi) (SZ.v nm)) 0 (SZ.v !l))
    decreases (32 - SZ.v !l)
  {
    let lv = !l;
    let mut acc = zero #F32.t #_;
    wacc_zero p (wt (SZ.v wo) xi) (SZ.v lv);
    let mut m = 0sz;
    while (!m <^ nm)
      invariant exists* (mv : sz{mv <= nm}). m |-> mv
      invariant exists* av. acc |-> av ** pure (ax p av (wacc (wt (SZ.v wo) xi) (SZ.v lv) (SZ.v !m)))
      decreases (SZ.v nm - SZ.v !m)
    {
      let mv = !m;
      let k = lv *^ 8sz +^ mv *^ 256sz;
      let a0 = !acc;
      wpart_init p grouped a0 (wt (SZ.v wo) xi) (SZ.v lv) (SZ.v mv);
      let mut part = (if grouped then zero #F32.t #_ else a0);
      let mut j = 0sz;
      while (!j <^ 8sz)
        invariant exists* (jv : sz{jv <= 8}). j |-> jv
        invariant exists* pv. part |-> pv ** pure (ax p pv (wpart (wt (SZ.v wo) xi) grouped (SZ.v lv) (SZ.v mv) (SZ.v !j)))
        decreases (8 - SZ.v !j)
      {
        let jv = !j;
        ax_w p ws (SZ.v wo + SZ.v k + SZ.v jv);
        axr_get p xs xi 0 (SZ.v nm * 256) (SZ.v k + SZ.v jv);
        let wv = w.(wo +^ k +^ jv);
        let xv = x.(k +^ jv);
        let pv = !part;
        let r = F32.fma_rn_ftz (wcast wv) xv pv;
        wpart_step p (wcast wv) xv pv r (fw (SZ.v wo + SZ.v k + SZ.v jv)) (elt xi (SZ.v k + SZ.v jv))
          (wt (SZ.v wo) xi) grouped (SZ.v lv) (SZ.v mv) (SZ.v jv);
        part := r;
        j := jv +^ 1sz;
      };
      let pv = !part;
      let na = wacc_next p (wt (SZ.v wo) xi) (SZ.v lv) (SZ.v mv) grouped a0 pv;
      acc := na;
      m := mv +^ 1sz;
    };
    let av = !acc;
    wl_elt p av (wt (SZ.v wo) xi) (SZ.v lv) (SZ.v nm);
    with lv1. assert lanes |-> lv1;
    axr_snoc p lv1 (wl (wt (SZ.v wo) xi) (SZ.v nm)) 0 (SZ.v lv) av;
    lanes.(lv) <- av;
    l := lv +^ 1sz;
  };
  let r = tree32 p (wl (wt (SZ.v wo) xi) (SZ.v nm)) lanes;
  wdot_final p r (wt (SZ.v wo) xi) (SZ.v nm);
  r
}

(**** from dev/KPlay.Kern2 ****)

inline_for_extraction noextract
let c32 (n : FStar.UInt32.t) : sz = SZ.uint32_to_sizet n

let rok_axr (p : pred) (rs : S.seq F32.t)
  : Lemma (requires rok p rs /\ S.length rs == 4718592) (ensures axr p rs rI 0 4718592)
  = let aux (i : nat) : Lemma (i < 4718592 ==> (i < S.length rs /\ ax p (S.index rs i) (elt rI i)))
      = if i < 4718592 then begin ax_r p rs i; ax_ext p (S.index rs i) (fr i) (elt rI i) end in
    FStar.Classical.forall_intro aux;
    axr_intro p rs rI 0 4718592

let xok_axr (p : pred) (xs : S.seq F32.t)
  : Lemma (requires xok p xs /\ S.length xs == 1024) (ensures axr p xs (atf xq) 0 1024)
  = let aux (i : nat) : Lemma (i < 1024 ==> (i < S.length xs /\ ax p (S.index xs i) (elt (atf xq) i)))
      = if i < 1024 then begin ax_x p xs i; ax_ext p (S.index xs i) (fx i) (elt (atf xq) i) end in
    FStar.Classical.forall_intro aux;
    axr_intro p xs (atf xq) 0 1024

inline_for_extraction noextract
fn copy_f32 (#ss : erased (Seq.seq F32.t)) (#ds : erased (Seq.seq F32.t)) (src dst : A.array F32.t) (n : sz)
  preserves gpu ** src |-> ss
  requires dst |-> ds ** pure (Seq.length ss == SZ.v n /\ Seq.length ds == SZ.v n)
  ensures dst |-> ss
{
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (iv : sz{iv <= n}). i |-> iv
    invariant exists* v. dst |-> v ** pure (Seq.length v == SZ.v n /\
      (forall (j : nat). j < SZ.v !i ==> Seq.index v j == Seq.index ss j))
    decreases (SZ.v n - SZ.v !i)
  {
    let iv = !i;
    let e = src.(iv);
    dst.(iv) <- e;
    i := iv +^ 1sz;
  };
  with v. assert dst |-> v;
  Seq.lemma_eq_elim v ss;
  rewrite (dst |-> v) as (dst |-> ss);
}

inline_for_extraction noextract
fn sigmoid32 (p : erased pred) (f : erased sf) (x : F32.t)
  preserves gpu
  requires pure (ax p x f)
  returns r : F32.t
  ensures pure (ax p r (fsig f))
{
  let e = Fast.exp (neg x);
  let a = F32.add_rn_ftz (one #F32.t) e;
  let r = F32.rcp_approx_ftz a;
  sig_ax p x e a r f;
  r
}

inline_for_extraction noextract
fn silu32 (p : erased pred) (f : erased sf) (x : F32.t)
  preserves gpu
  requires pure (ax p x f)
  returns r : F32.t
  ensures pure (ax p r (fsilu f))
{
  let s = sigmoid32 p f x;
  let r = F32.mul_rn_ftz x s;
  silu_ax p x s r f;
  r
}

(* RMSNorm with (1 + w) scaling of src[0..1024), rounded through BF16. *)
inline_for_extraction noextract
fn rmsnorm (p : erased pred) (q : erased (env -> S.seq real))
  (#ss : erased (Seq.seq F32.t)) (#ws : erased (Seq.seq BF.t)) (#ds : erased (Seq.seq F32.t))
  (src : A.array F32.t) (w : A.array BF.t) (wo : sz) (dst : A.array F32.t)
  preserves gpu ** src |-> ss ** w |-> ws
  requires dst |-> ds ** pure (Seq.length ss == 1024 /\ Seq.length ds == 1024 /\ SZ.v wo + 1024 <= Seq.length ws /\
    wok p ws /\ axr p ss (atf q) 0 1024)
  ensures exists* v. dst |-> v ** pure (Seq.length v == 1024 /\ axr p v (atf (rmsq q (SZ.v wo))) 0 1024)
{
  pts_to_len w;
  let s = sqsum p (atf q) src 0sz 2sz 16sz false;
  ax_ext p s (ssum (SZ.v 2sz * (SZ.v 16sz * 32)) (sqt (atf q) (SZ.v 0sz))) (rms_s q);
  let m = F32.mul_rn_ftz s (of_literal #F32.t "0.0009765625");
  let a = F32.add_rn_ftz m (of_literal #F32.t "0.0000009999999974752427078783512115478515625");
  let scale = F32.rsqrt_approx_ftz a;
  rms_scale p s m a scale q;
  axr_empty p ds (atf (rmsq q (SZ.v wo))) 0;
  let mut i = 0sz;
  while (!i <^ 1024sz)
    invariant exists* (iv : sz{iv <= 1024}). i |-> iv
    invariant exists* v. dst |-> v ** pure (Seq.length v == 1024 /\ axr p v (atf (rmsq q (SZ.v wo))) 0 (SZ.v !i))
    decreases (1024 - SZ.v !i)
  {
    let iv = !i;
    axr_get p ss (atf q) 0 1024 (SZ.v iv);
    ax_w p ws (SZ.v wo + SZ.v iv);
    let h = src.(iv);
    let wv = w.(wo +^ iv);
    let w1 = F32.add_rn_ftz (one #F32.t) (wcast wv);
    let m1 = F32.mul_rn_ftz h scale;
    let r = F32.mul_rn_ftz m1 w1;
    rms_elt p q (SZ.v wo) (SZ.v iv) h scale w1 m1 r wv;
    with v. assert dst |-> v;
    axr_snoc p v (atf (rmsq q (SZ.v wo))) 0 (SZ.v iv) (bround r);
    dst.(iv) <- bround r;
    i := iv +^ 1sz;
  }
}

let mul_le_lemma (a b c : nat)
  : Lemma (requires a <= b) (ensures a * c <= b * c)
  = FStar.Math.Lemmas.lemma_mult_le_right c a b

(* rows of [wdot] into dst[0..rows) *)
inline_for_extraction noextract
fn matvec_rows (p : erased pred) (q : erased (env -> S.seq real))
  (#ws : erased (Seq.seq BF.t)) (#xs : erased (Seq.seq F32.t)) (#ds : erased (Seq.seq F32.t))
  (grouped : bool) (w : A.array BF.t) (wo : sz) (x : A.array F32.t) (nm : sz)
  (dst : A.array F32.t) (rows : sz)
  preserves gpu ** w |-> ws ** x |-> xs
  requires dst |-> ds ** pure (SZ.v nm <= 16 /\ SZ.v nm * 256 <= Seq.length xs /\ SZ.v wo + SZ.v rows * (SZ.v nm * 256) <= Seq.length ws /\
    Seq.length ds == SZ.v rows /\ SZ.fits (SZ.v wo + SZ.v rows * (SZ.v nm * 256)) /\
    wok p ws /\ axr p xs (atf q) 0 (SZ.v nm * 256))
  ensures exists* v. dst |-> v ** pure (Seq.length v == SZ.v rows /\
    axr p v (atf (mvq q (SZ.v wo) (SZ.v rows) (SZ.v nm * 256))) 0 (SZ.v rows))
{
  axr_empty p ds (atf (mvq q (SZ.v wo) (SZ.v rows) (SZ.v nm * 256))) 0;
  let mut i = 0sz;
  while (!i <^ rows)
    invariant exists* (iv : sz{iv <= rows}). i |-> iv
    invariant exists* v. dst |-> v ** pure (Seq.length v == SZ.v rows /\
      axr p v (atf (mvq q (SZ.v wo) (SZ.v rows) (SZ.v nm * 256))) 0 (SZ.v !i))
    decreases (SZ.v rows - SZ.v !i)
  {
    let iv = !i;
    mul_le_lemma (SZ.v iv + 1) (SZ.v rows) (SZ.v nm * 256);
    let r = wdot p (atf q) grouped w (wo +^ iv *^ (nm *^ 256sz)) x nm;
    mv_elt p r q (SZ.v wo) (SZ.v rows) (SZ.v nm * 256) (SZ.v iv);
    with v. assert dst |-> v;
    axr_snoc p v (atf (mvq q (SZ.v wo) (SZ.v rows) (SZ.v nm * 256))) 0 (SZ.v iv) r;
    dst.(iv) <- r;
    i := iv +^ 1sz;
  }
}

inline_for_extraction noextract
let mix_off : sz = (c32 11012096ul)

inline_for_extraction noextract
let eps_lit : F32.t = of_literal #F32.t "0.0000009999999974752427078783512115478515625"

let mix_ok (ws : S.seq BF.t) (b : nat) (mix : sz) : prop =
  SZ.v mix == dn_mix b /\ SZ.v mix + 8446112 + 2097152 <= S.length ws /\ SZ.fits (SZ.v mix + 8446112 + 2097152)

(* beta = sigmoid(b . h) *)
inline_for_extraction noextract
fn head_beta (p : erased pred) (b : erased nat) (#ws : erased (Seq.seq BF.t)) (#ns : erased (Seq.seq F32.t))
  (w : A.array BF.t) (mix : sz) (norm : A.array F32.t) (h : sz)
  preserves gpu ** w |-> ws ** norm |-> ns
  requires pure (Seq.length ns == 1024 /\ SZ.v h < 16 /\ mix_ok ws b mix /\ wok p ws /\ axr p ns (atf (hinq b)) 0 1024)
  returns r : F32.t
  ensures pure (ax p r (betaf b (SZ.v h)))
{
  let d = wdot p (atf (hinq b)) true w (mix +^ (c32 8388608ul) +^ h *^ 1024sz) norm 4sz;
  let r = sigmoid32 p (bdot b (SZ.v h)) d;
  beta_fin p r b (SZ.v h);
  r
}

inline_for_extraction noextract
fn head_decay (p : erased pred) (b : erased nat) (#ws : erased (Seq.seq BF.t)) (#ns : erased (Seq.seq F32.t))
  (w : A.array BF.t) (mix : sz) (norm : A.array F32.t) (h : sz)
  preserves gpu ** w |-> ws ** norm |-> ns
  requires pure (Seq.length ns == 1024 /\ SZ.v h < 16 /\ mix_ok ws b mix /\ wok p ws /\ axr p ns (atf (hinq b)) 0 1024)
  returns r : F32.t
  ensures pure (ax p r (decf b (SZ.v h)))
{
  pts_to_len w;
  let d = wdot p (atf (hinq b)) true w (mix +^ (c32 8404992ul) +^ h *^ 1024sz) norm 4sz;
  ax_w p ws (SZ.v mix + 8445968 + SZ.v h);
  ax_w p ws (SZ.v mix + 8445952 + SZ.v h);
  let dt = w.(mix +^ (c32 8445968ul) +^ h);
  let a = F32.add_rn_ftz d (wcast dt);
  let sp = SP.softplus20 a;
  let al = w.(mix +^ (c32 8445952ul) +^ h);
  let ea = Fast.exp (wcast al);
  let m = F32.mul_rn_ftz ea sp;
  let r = Fast.exp (neg m);
  dec_ax p b (SZ.v h) d (wcast dt) a sp (wcast al) ea m r;
  r
}

(* Causal convolution of all 6144 channels: co[c] and the shifted window in cvo. *)
inline_for_extraction noextract
fn conv_all (p : erased pred) (b : erased nat) (#ws : erased (Seq.seq BF.t)) (#cs : erased (Seq.seq F32.t)) (#ps : erased (Seq.seq F32.t))
  (#os : erased (Seq.seq F32.t)) (#ds : erased (Seq.seq F32.t))
  (w : A.array BF.t) (mix : sz) (cv : A.array F32.t) (proj : A.array F32.t)
  (cvo : A.array F32.t) (co : A.array F32.t) (dn : sz)
  preserves gpu ** w |-> ws ** cv |-> cs ** proj |-> ps
  requires cvo |-> os ** co |-> ds ** pure (
    mix_ok ws b mix /\ wok p ws /\ cok p cs /\
    Seq.length cs == 442368 /\ Seq.length ps == 6144 /\ Seq.length ds == 6144 /\
    SZ.v dn < 18 /\ cvo_pre (SZ.v dn) cs os ps 0 /\ axr p ps (atf (qkvq b)) 0 6144)
  ensures exists* o c. cvo |-> o ** co |-> c ** pure (Seq.length c == 6144 /\
    cvo_pre (SZ.v dn) cs o ps 6144 /\ axr p c (atf (coq b (SZ.v dn))) 0 6144)
{
  pts_to_len w;
  axr_empty p ds (atf (coq b (SZ.v dn))) 0;
  let mut i = 0sz;
  while (!i <^ 6144sz)
    invariant exists* (iv : sz{iv <= 6144}). i |-> iv
    invariant exists* o c. cvo |-> o ** co |-> c ** pure (Seq.length c == 6144 /\
      cvo_pre (SZ.v dn) cs o ps (SZ.v !i) /\ axr p c (atf (coq b (SZ.v dn))) 0 (SZ.v !i))
    decreases (6144 - SZ.v !i)
  {
    let iv = !i;
    slot_c4 (SZ.v dn) (SZ.v iv) 0; slot_c4 (SZ.v dn) (SZ.v iv) 1;
    slot_c4 (SZ.v dn) (SZ.v iv) 2; slot_c4 (SZ.v dn) (SZ.v iv) 3;
    let ci = dn *^ 24576sz +^ iv *^ 4sz;
    let wc = mix +^ (c32 8421376ul) +^ iv *^ 4sz;
    ax_c p cs (SZ.v ci + 1); ax_c p cs (SZ.v ci + 2); ax_c p cs (SZ.v ci + 3);
    axr_get p ps (atf (qkvq b)) 0 6144 (SZ.v iv);
    ax_w p ws (SZ.v wc); ax_w p ws (SZ.v wc + 1); ax_w p ws (SZ.v wc + 2); ax_w p ws (SZ.v wc + 3);
    let c0 = cv.(ci +^ 1sz);
    let c1 = cv.(ci +^ 2sz);
    let c2 = cv.(ci +^ 3sz);
    let c3 = proj.(iv);
    with o. assert cvo |-> o;
    cvo_step (SZ.v dn) cs o ps (SZ.v iv) c0 c1 c2 c3;
    cvo.(ci) <- c0;
    cvo.(ci +^ 1sz) <- c1;
    cvo.(ci +^ 2sz) <- c2;
    cvo.(ci +^ 3sz) <- c3;
    let w0 = w.(wc);
    let w1 = w.(wc +^ 1sz);
    let w2 = w.(wc +^ 2sz);
    let w3 = w.(wc +^ 3sz);
    let s0 = F32.fma_rn_ftz c0 (wcast w0) (zero #F32.t #_);
    let s1 = F32.fma_rn_ftz c1 (wcast w1) s0;
    let s2 = F32.fma_rn_ftz c2 (wcast w2) s1;
    let s3 = F32.fma_rn_ftz c3 (wcast w3) s2;
    conv_ax p b (SZ.v dn) (SZ.v iv) c0 c1 c2 c3 (wcast w0) (wcast w1) (wcast w2) (wcast w3) s0 s1 s2 s3;
    let r = silu32 p (cpre b (SZ.v dn) (SZ.v iv)) s3;
    co_fin p b (SZ.v dn) (SZ.v iv) r;
    with c. assert co |-> c;
    axr_snoc p c (atf (coq b (SZ.v dn))) 0 (SZ.v iv) r;
    co.(iv) <- r;
    i := iv +^ 1sz;
  }
}

(* q[i] = co[h*128 + i] * rsqrt(|.|^2 + eps) * scale *)
inline_for_extraction noextract
fn q_fill (p : erased pred) (b d : erased nat) (#cs : erased (Seq.seq F32.t)) (#qs : erased (Seq.seq F32.t))
  (co q : A.array F32.t) (h : sz)
  preserves gpu ** co |-> cs
  requires q |-> qs ** pure (Seq.length cs == 6144 /\ Seq.length qs == 128 /\ SZ.v h < 16 /\
    axr p cs (atf (coq b d)) 0 6144)
  ensures exists* v. q |-> v ** pure (Seq.length v == 128 /\ axr p v (atfo (qq b d) (SZ.v h * 128)) 0 128)
{
  axr_mono p cs (atf (coq b d)) 0 6144 (SZ.v h * 128) (SZ.v h * 128 + 4 * (1 * 32));
  let sq = sqsum p (atf (coq b d)) co (h *^ 128sz) 4sz 1sz true;
  let a = F32.add_rn_ftz sq eps_lit;
  let rs = F32.rsqrt_approx_ftz a;
  nsc_ax p b d (SZ.v h * 128) sq a rs;
  let qsc = F32.mul_rn_ftz rs (of_literal #F32.t "0.088388346135616302490234375");
  qsc_ax p b d (SZ.v h * 128) rs qsc;
  axr_empty p qs (atfo (qq b d) (SZ.v h * 128)) 0;
  let mut i = 0sz;
  while (!i <^ 128sz)
    invariant exists* (iv : sz{iv <= 128}). i |-> iv
    invariant exists* v. q |-> v ** pure (Seq.length v == 128 /\ axr p v (atfo (qq b d) (SZ.v h * 128)) 0 (SZ.v !i))
    decreases (128 - SZ.v !i)
  {
    let iv = !i;
    axr_get p cs (atf (coq b d)) 0 6144 (SZ.v h * 128 + SZ.v iv);
    let e = co.(h *^ 128sz +^ iv);
    let r = F32.mul_rn_ftz e qsc;
    q_elt p b d (SZ.v h) (SZ.v iv) e qsc r;
    with v. assert q |-> v;
    axr_snoc p v (atfo (qq b d) (SZ.v h * 128)) 0 (SZ.v iv) r;
    q.(iv) <- r;
    i := iv +^ 1sz;
  }
}

inline_for_extraction noextract
fn k_fill (p : erased pred) (b d : erased nat) (#cs : erased (Seq.seq F32.t)) (#ks : erased (Seq.seq F32.t))
  (co k : A.array F32.t) (h : sz)
  preserves gpu ** co |-> cs
  requires k |-> ks ** pure (Seq.length cs == 6144 /\ Seq.length ks == 128 /\ SZ.v h < 16 /\
    axr p cs (atf (coq b d)) 0 6144)
  ensures exists* v. k |-> v ** pure (Seq.length v == 128 /\ axr p v (atfo (kq b d) (SZ.v h * 128)) 0 128)
{
  axr_mono p cs (atf (coq b d)) 0 6144 (2048 + SZ.v h * 128) (2048 + SZ.v h * 128 + 4 * (1 * 32));
  let sk = sqsum p (atf (coq b d)) co (2048sz +^ h *^ 128sz) 4sz 1sz true;
  let a = F32.add_rn_ftz sk eps_lit;
  let ksc = F32.rsqrt_approx_ftz a;
  nsc_ax p b d (2048 + SZ.v h * 128) sk a ksc;
  axr_empty p ks (atfo (kq b d) (SZ.v h * 128)) 0;
  let mut i = 0sz;
  while (!i <^ 128sz)
    invariant exists* (iv : sz{iv <= 128}). i |-> iv
    invariant exists* v. k |-> v ** pure (Seq.length v == 128 /\ axr p v (atfo (kq b d) (SZ.v h * 128)) 0 (SZ.v !i))
    decreases (128 - SZ.v !i)
  {
    let iv = !i;
    axr_get p cs (atf (coq b d)) 0 6144 (2048 + SZ.v h * 128 + SZ.v iv);
    let e = co.(2048sz +^ h *^ 128sz +^ iv);
    let r = F32.mul_rn_ftz e ksc;
    k_elt p b d (SZ.v h) (SZ.v iv) e ksc r;
    with v. assert k |-> v;
    axr_snoc p v (atfo (kq b d) (SZ.v h * 128)) 0 (SZ.v iv) r;
    k.(iv) <- r;
    i := iv +^ 1sz;
  }
}

let si_rng (d h j i : nat)
  : Lemma (requires d < 18 /\ h < 16 /\ j < 128 /\ i < 128)
          (ensures d * 262144 + h * 16384 <= si d h j /\ si d h j + 128 <= d * 262144 + h * 16384 + 16384 /\
                   d * 262144 + 262144 <= 4718592 /\ si d h j + i < 4718592 /\
                   (si d h j + i) / 262144 == d /\ si d h j + i == d * 262144 + (h * 16384 + j * 128 + i))
  = FStar.Math.Lemmas.lemma_mult_le_right 262144 (d + 1) 18;
    idx3 h j i;
    idx_slot d (h * 16384 + j * 128 + i)

let hs_inv (p : pred) (b d h : nat) (rs o : S.seq F32.t) (hi : nat) : prop =
  rco_pre d rs o /\ axr p o (atf (rcq b d)) (d * 262144) hi

(* Delta-rule update of head h: core[j] and the state slice. *)
inline_for_extraction noextract
fn head_state (p : erased pred) (b : erased nat)
  (#rs : erased (Seq.seq F32.t)) (#ks : erased (Seq.seq F32.t)) (#qs : erased (Seq.seq F32.t))
  (#vs : erased (Seq.seq F32.t)) (#os : erased (Seq.seq F32.t)) (#cs : erased (Seq.seq F32.t))
  (rc : A.array F32.t) (k q co : A.array F32.t) (rco : A.array F32.t) (core : A.array F32.t)
  (dn h : sz) (decay beta : F32.t)
  preserves gpu ** rc |-> rs ** k |-> ks ** q |-> qs ** co |-> vs
  requires rco |-> os ** core |-> cs ** pure (
    Seq.length rs == 4718592 /\ Seq.length ks == 128 /\ Seq.length qs == 128 /\
    Seq.length vs == 6144 /\ Seq.length cs == 128 /\ SZ.v dn < 18 /\ SZ.v h < 16 /\
    axr p rs rI 0 4718592 /\ axr p ks (atfo (kq b (SZ.v dn)) (SZ.v h * 128)) 0 128 /\
    axr p qs (atfo (qq b (SZ.v dn)) (SZ.v h * 128)) 0 128 /\ axr p vs (atf (coq b (SZ.v dn))) 0 6144 /\
    ax p decay (decf b (SZ.v h)) /\ ax p beta (betaf b (SZ.v h)) /\
    hs_inv p b (SZ.v dn) (SZ.v h) rs os (SZ.v dn * 262144 + SZ.v h * 16384))
  ensures exists* o c. rco |-> o ** core |-> c ** pure (Seq.length c == 128 /\
    hs_inv p b (SZ.v dn) (SZ.v h) rs o (SZ.v dn * 262144 + SZ.v h * 16384 + 16384) /\
    axr p c (atfo (coreq b (SZ.v dn)) (SZ.v h * 128)) 0 128)
{
  let kqv = dot128 p (atfo (kq b (SZ.v dn)) (SZ.v h * 128)) (atfo (qq b (SZ.v dn)) (SZ.v h * 128)) k 0sz q;
  axr_empty p cs (atfo (coreq b (SZ.v dn)) (SZ.v h * 128)) 0;
  let mut j = 0sz;
  while (!j <^ 128sz)
    invariant exists* (jv : sz{jv <= 128}). j |-> jv
    invariant exists* o c. rco |-> o ** core |-> c ** pure (Seq.length c == 128 /\
      hs_inv p b (SZ.v dn) (SZ.v h) rs o (SZ.v dn * 262144 + SZ.v h * 16384 + SZ.v !j * 128) /\
      axr p c (atfo (coreq b (SZ.v dn)) (SZ.v h * 128)) 0 (SZ.v !j))
    decreases (128 - SZ.v !j)
  {
    let jv = !j;
    si_rng (SZ.v dn) (SZ.v h) (SZ.v jv) 0;
    let si_ = dn *^ (c32 262144ul) +^ h *^ 16384sz +^ jv *^ 128sz;
    axr_mono p rs rI 0 4718592 (SZ.v si_) (SZ.v si_ + 128);
    let sk = dot128 p rI (atfo (kq b (SZ.v dn)) (SZ.v h * 128)) rc si_ k;
    let sq = dot128 p rI (atfo (qq b (SZ.v dn)) (SZ.v h * 128)) rc si_ q;
    axr_get p vs (atf (coq b (SZ.v dn))) 0 6144 (4096 + SZ.v h * 128 + SZ.v jv);
    let vj = co.(4096sz +^ h *^ 128sz +^ jv);
    let m1 = F32.fma_rn_ftz (neg decay) sk vj;
    let err = F32.mul_rn_ftz m1 beta;
    err_ax p b (SZ.v dn) (SZ.v h) (SZ.v jv) decay sk vj beta m1 err;
    let m2 = F32.mul_rn_ftz err kqv;
    let cj = F32.fma_rn_ftz decay sq m2;
    core_ax p b (SZ.v dn) (SZ.v h) (SZ.v jv) decay sq err kqv m2 cj;
    with c. assert core |-> c;
    axr_snoc p c (atfo (coreq b (SZ.v dn)) (SZ.v h * 128)) 0 (SZ.v jv) cj;
    core.(jv) <- cj;
    let mut i = 0sz;
    while (!i <^ 128sz)
      invariant exists* (iv : sz{iv <= 128}). i |-> iv
      invariant exists* o. rco |-> o ** pure (hs_inv p b (SZ.v dn) (SZ.v h) rs o (SZ.v si_ + SZ.v !i))
      decreases (128 - SZ.v !i)
    {
      let iv = !i;
      si_rng (SZ.v dn) (SZ.v h) (SZ.v jv) (SZ.v iv);
      axr_get p rs rI 0 4718592 (SZ.v si_ + SZ.v iv);
      axr_get p ks (atfo (kq b (SZ.v dn)) (SZ.v h * 128)) 0 128 (SZ.v iv);
      let s = rc.(si_ +^ iv);
      let ki = k.(iv);
      let m3 = F32.mul_rn_ftz ki err;
      let r = F32.fma_rn_ftz s decay m3;
      st_ax p b (SZ.v dn) (SZ.v h) (SZ.v jv) (SZ.v iv) s decay ki err m3 r;
      with o. assert rco |-> o;
      rco_len (SZ.v dn) rs o;
      rco_upd (SZ.v dn) rs o (SZ.v si_ + SZ.v iv) r;
      axr_snoc p o (atf (rcq b (SZ.v dn))) (SZ.v dn * 262144) (SZ.v si_ + SZ.v iv) r;
      rco.(si_ +^ iv) <- r;
      i := iv +^ 1sz;
    };
    j := jv +^ 1sz;
  }
}

(* Gated RMSNorm of core into out[h*128 ..]. *)
inline_for_extraction noextract
fn head_gate (p : erased pred) (b d : erased nat)
  (#ws : erased (Seq.seq BF.t)) (#cs : erased (Seq.seq F32.t)) (#zs : erased (Seq.seq F32.t))
  (#os : erased (Seq.seq F32.t))
  (w : A.array BF.t) (mix : sz) (core z out : A.array F32.t) (h : sz)
  preserves gpu ** w |-> ws ** core |-> cs ** z |-> zs
  requires out |-> os ** pure (
    mix_ok ws b mix /\ wok p ws /\ d < 18 /\
    Seq.length cs == 128 /\ Seq.length zs == 2048 /\ Seq.length os == 2048 /\ SZ.v h < 16 /\
    axr p cs (atfo (coreq b d) (SZ.v h * 128)) 0 128 /\ axr p zs (atf (zq b)) 0 2048 /\
    axr p os (atf (gatedq b d)) 0 (SZ.v h * 128))
  ensures exists* o. out |-> o ** pure (Seq.length o == 2048 /\ axr p o (atf (gatedq b d)) 0 (SZ.v h * 128 + 128))
{
  pts_to_len w;
  let ss = sqsum p (atfo (coreq b d) (SZ.v h * 128)) core 0sz 1sz 4sz false;
  let m = F32.mul_rn_ftz ss (of_literal #F32.t "0.0078125");
  let a = F32.add_rn_ftz m eps_lit;
  let rstd = F32.rsqrt_approx_ftz a;
  gsc_ax p b d (SZ.v h) ss m a rstd;
  let mut i = 0sz;
  while (!i <^ 128sz)
    invariant exists* (iv : sz{iv <= 128}). i |-> iv
    invariant exists* o. out |-> o ** pure (Seq.length o == 2048 /\ axr p o (atf (gatedq b d)) 0 (SZ.v h * 128 + SZ.v !i))
    decreases (128 - SZ.v !i)
  {
    let iv = !i;
    axr_get p cs (atfo (coreq b d) (SZ.v h * 128)) 0 128 (SZ.v iv);
    ax_w p ws (SZ.v mix + 8445984 + SZ.v iv);
    axr_get p zs (atf (zq b)) 0 2048 (SZ.v h * 128 + SZ.v iv);
    let c = core.(iv);
    let g = w.(mix +^ (c32 8445984ul) +^ iv);
    let zv = z.(h *^ 128sz +^ iv);
    let szv = silu32 p (elt (atf (zq b)) (SZ.v h * 128 + SZ.v iv)) zv;
    let m1 = F32.mul_rn_ftz c rstd;
    let m2 = F32.mul_rn_ftz m1 (wcast g);
    let r = F32.mul_rn_ftz m2 szv;
    gate_elt p b d (SZ.v h) (SZ.v iv) c rstd szv m1 m2 r g;
    with o. assert out |-> o;
    axr_snoc p o (atf (gatedq b d)) 0 (SZ.v h * 128 + SZ.v iv) r;
    out.(h *^ 128sz +^ iv) <- r;
    i := iv +^ 1sz;
  }
}

(* dst[i] = bround(wdot(w, wo + i*(nm*256), x) + res[i]) for i < 1024 *)
inline_for_extraction noextract
fn residual_rows (p : erased pred) (q r : erased (env -> S.seq real))
  (#ws : erased (Seq.seq BF.t)) (#xs : erased (Seq.seq F32.t)) (#rs : erased (Seq.seq F32.t))
  (#ds : erased (Seq.seq F32.t))
  (w : A.array BF.t) (wo : sz) (x : A.array F32.t) (nm : sz) (res : A.array F32.t) (dst : A.array F32.t)
  preserves gpu ** w |-> ws ** x |-> xs ** res |-> rs
  requires dst |-> ds ** pure (SZ.v nm <= 16 /\ SZ.v nm * 256 <= Seq.length xs /\
    SZ.v wo + 1024 * (SZ.v nm * 256) <= Seq.length ws /\ SZ.fits (SZ.v wo + 1024 * (SZ.v nm * 256)) /\
    Seq.length rs == 1024 /\ Seq.length ds == 1024 /\
    wok p ws /\ axr p xs (atf q) 0 (SZ.v nm * 256) /\ axr p rs (atf r) 0 1024)
  ensures exists* v. dst |-> v ** pure (Seq.length v == 1024 /\
    axr p v (atf (resq q r (SZ.v wo) (SZ.v nm * 256))) 0 1024)
{
  axr_empty p ds (atf (resq q r (SZ.v wo) (SZ.v nm * 256))) 0;
  let mut i = 0sz;
  while (!i <^ 1024sz)
    invariant exists* (iv : sz{iv <= 1024}). i |-> iv
    invariant exists* v. dst |-> v ** pure (Seq.length v == 1024 /\
      axr p v (atf (resq q r (SZ.v wo) (SZ.v nm * 256))) 0 (SZ.v !i))
    decreases (1024 - SZ.v !i)
  {
    let iv = !i;
    mul_le_lemma (SZ.v iv + 1) 1024 (SZ.v nm * 256);
    let d = wdot p (atf q) false w (wo +^ iv *^ (nm *^ 256sz)) x nm;
    axr_get p rs (atf r) 0 1024 (SZ.v iv);
    let rv = res.(iv);
    let s = F32.add_rn_ftz d rv;
    res_elt p q r (SZ.v wo) (SZ.v nm * 256) (SZ.v iv) d rv s;
    with v. assert dst |-> v;
    axr_snoc p v (atf (resq q r (SZ.v wo) (SZ.v nm * 256))) 0 (SZ.v iv) (bround s);
    dst.(iv) <- bround s;
    i := iv +^ 1sz;
  }
}

inline_for_extraction noextract
fn mlp_rows (p : erased pred) (q : erased (env -> S.seq real)) (b : erased nat)
  (#ws : erased (Seq.seq BF.t)) (#ns : erased (Seq.seq F32.t)) (#ds : erased (Seq.seq F32.t))
  (w : A.array BF.t) (gw uw : sz) (norm : A.array F32.t) (dst : A.array F32.t)
  preserves gpu ** w |-> ws ** norm |-> ns
  requires dst |-> ds ** pure (Seq.length ns == 1024 /\ Seq.length ds == 3584 /\
    SZ.v gw == b + 2048 /\ SZ.v uw == b + 3672064 /\
    SZ.v gw + 3584 * 1024 <= Seq.length ws /\ SZ.v uw + 3584 * 1024 <= Seq.length ws /\
    SZ.fits (SZ.v gw + 3584 * 1024) /\ SZ.fits (SZ.v uw + 3584 * 1024) /\
    wok p ws /\ axr p ns (atf (rmsq q (b + 1024))) 0 1024)
  ensures exists* v. dst |-> v ** pure (Seq.length v == 3584 /\ axr p v (atf (actq q b)) 0 3584)
{
  axr_empty p ds (atf (actq q b)) 0;
  let mut i = 0sz;
  while (!i <^ 3584sz)
    invariant exists* (iv : sz{iv <= 3584}). i |-> iv
    invariant exists* v. dst |-> v ** pure (Seq.length v == 3584 /\ axr p v (atf (actq q b)) 0 (SZ.v !i))
    decreases (3584 - SZ.v !i)
  {
    let iv = !i;
    mul_le_lemma (SZ.v iv + 1) 3584 1024;
    let g = wdot p (atf (rmsq q (b + 1024))) true w (gw +^ iv *^ 1024sz) norm 4sz;
    let u = wdot p (atf (rmsq q (b + 1024))) true w (uw +^ iv *^ 1024sz) norm 4sz;
    let sg = silu32 p (ssum 1024 (wt (b + 2048 + SZ.v iv * 1024) (atf (rmsq q (b + 1024))))) g;
    let r = F32.mul_rn_ftz sg u;
    act_elt p q b (SZ.v iv) g u sg r;
    with v. assert dst |-> v;
    axr_snoc p v (atf (actq q b)) 0 (SZ.v iv) r;
    dst.(iv) <- r;
    i := iv +^ 1sz;
  }
}

inline_for_extraction noextract
fn rd_kernel (#ws : erased (Seq.seq BF.t)) (#xs : erased (Seq.seq F32.t)) (#rs : erased (Seq.seq F32.t))
  (#cs : erased (Seq.seq F32.t)) (#xo0 : erased (Seq.seq F32.t)) (#ro0 : erased (Seq.seq F32.t))
  (#co0 : erased (Seq.seq F32.t))
  (w : A.array BF.t) (x rc cv xo rco cvo : A.array F32.t) (base dn : sz)
  preserves gpu ** w |-> ws ** x |-> xs ** rc |-> rs ** cv |-> cs
  requires xo |-> xo0 ** rco |-> ro0 ** cvo |-> co0 ** pure (
    SZ.v base + 21555360 <= Seq.length ws /\ SZ.fits (SZ.v base + 21555360) /\
    Seq.length xs == 1024 /\ Seq.length rs == 4718592 /\ Seq.length cs == 442368 /\
    Seq.length xo0 == 1024 /\ Seq.length ro0 == 4718592 /\ Seq.length co0 == 442368 /\ SZ.v dn < 18)
  ensures exists* a b c. xo |-> a ** rco |-> b ** cvo |-> c **
    pure (optimized_recurrent_decoder_relation ws xs rs cs (SZ.v base) (SZ.v dn) a b c)
{
  let p : erased pred = okp ws xs rs cs;
  okp_ok ws xs rs cs;
  let b : erased nat = SZ.v base;
  let d : erased nat = SZ.v dn;
  copy_f32 rc rco (c32 4718592ul);
  copy_f32 cv cvo (c32 442368ul);
  rco_init d rs;
  axr_empty p rs (atf (rcq b d)) (d * 262144);
  xok_axr p xs;
  rok_axr p rs;
  let mut norm = [| zero #F32.t #_; 1024sz |];
  let mut proj = [| zero #F32.t #_; 6144sz |];
  let mut z = [| zero #F32.t #_; 2048sz |];
  let mut out = [| zero #F32.t #_; 2048sz |];
  let mut q = [| zero #F32.t #_; 128sz |];
  let mut k = [| zero #F32.t #_; 128sz |];
  let mut co = [| zero #F32.t #_; 6144sz |];
  let mut core = [| zero #F32.t #_; 128sz |];
  let mut hid = [| zero #F32.t #_; 1024sz |];
  let mut mlp = [| zero #F32.t #_; 3584sz |];
  rmsnorm p xq x w base norm;
  with ns. assert norm |-> ns;
  FStar.Classical.forall_intro (hin_eq b);
  axr_q p ns (rmsq xq b) (hinq b) 0 1024;
  let mix = base +^ mix_off;
  matvec_rows p (hinq b) true w mix norm 4sz proj 6144sz;
  with ps. assert proj |-> ps;
  FStar.Classical.forall_intro (qkv_eq b);
  axr_q p ps (mvq (hinq b) (dn_mix b) 6144 1024) (qkvq b) 0 6144;
  matvec_rows p (hinq b) true w (mix +^ (c32 6291456ul)) norm 4sz z 2048sz;
  with zs. assert z |-> zs;
  FStar.Classical.forall_intro (z_eq b);
  axr_q p zs (mvq (hinq b) (dn_mix b + 6291456) 2048 1024) (zq b) 0 2048;
  cvo_init d cs ps;
  conv_all p b w mix cv proj cvo co dn;
  with os0. assert out |-> os0;
  axr_empty p os0 (atf (gatedq b d)) 0;
  let mut h = 0sz;
  while (!h <^ 16sz)
    invariant exists* (hv : sz{hv <= 16}). h |-> hv
    invariant exists* o. rco |-> o ** pure (hs_inv p b d (SZ.v !h) rs o (d * 262144 + SZ.v !h * 16384))
    invariant exists* s. out |-> s ** pure (Seq.length s == 2048 /\ axr p s (atf (gatedq b d)) 0 (SZ.v !h * 128))
    invariant exists* s. q |-> s ** pure (Seq.length s == 128)
    invariant exists* s. k |-> s ** pure (Seq.length s == 128)
    invariant exists* s. core |-> s ** pure (Seq.length s == 128)
    decreases (16 - SZ.v !h)
  {
    let hv = !h;
    let beta = head_beta p b w mix norm hv;
    let decay = head_decay p b w mix norm hv;
    q_fill p b d co q hv;
    k_fill p b d co k hv;
    head_state p b rc k q co rco core dn hv decay beta;
    head_gate p b d w mix core z out hv;
    h := hv +^ 1sz;
  };
  with gs. assert out |-> gs;
  residual_rows p (gatedq b d) xq w (mix +^ (c32 8446112ul)) out 8sz x hid;
  with hs. assert hid |-> hs;
  FStar.Classical.forall_intro (x1_eq b d);
  axr_q p hs (resq (gatedq b d) xq (dn_mix b + 8446112) 2048) (x1q b d) 0 1024;
  rmsnorm p (x1q b d) hid w (base +^ 1024sz) norm;
  mlp_rows p (x1q b d) b w (base +^ 2048sz) (base +^ (c32 3672064ul)) norm mlp;
  residual_rows p (actq (x1q b d) b) (x1q b d) w (base +^ (c32 7342080ul)) mlp 14sz hid xo;
  with xs'. assert xo |-> xs';
  FStar.Classical.forall_intro (out_eq b d);
  axr_q p xs' (resq (actq (x1q b d) b) (x1q b d) (b + 7342080) 3584) (outq b d) 0 1024;
  with ro. assert rco |-> ro;
  with cvs. assert cvo |-> cvs;
  rel_fin ws xs rs cs xs' ro cvs ps (SZ.v base) (SZ.v dn);
  assert pure (optimized_recurrent_decoder_relation ws xs rs cs (SZ.v base) (SZ.v dn) xs' ro cvs);
}

instance send_global_array_contents
  (#a : Type0)
  (arr : A.array a{is_global arr})
  (#f : perm)
  (contents : Seq.seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

let cast_ok (x : I64.t)
  : Lemma (requires 0 <= I64.v x /\ I64.v x < 4294967296)
          (ensures SZ.v (SZ.uint32_to_sizet (FStar.Int.Cast.int64_to_uint32 x)) == I64.v x)
  = FStar.Math.Lemmas.small_mod (I64.v x) 4294967296

fn kplay_run
  (w : A.array BF.t{is_global w}) (x : A.array F32.t{is_global x})
  (rc : A.array F32.t{is_global rc}) (cv : A.array F32.t{is_global cv})
  (xo : A.array F32.t{is_global xo}) (rco : A.array F32.t{is_global rco})
  (cvo : A.array F32.t{is_global cvo})
  (wn : SZ.t) (base dn : I64.t)
  (#ws : erased (Seq.seq BF.t)) (#xs #rs #cs : erased (Seq.seq F32.t))
  preserves cpu ** on gpu_loc (w |-> ws) ** on gpu_loc (x |-> xs) ** on gpu_loc (rc |-> rs) ** on gpu_loc (cv |-> cs)
  requires (exists* a. on gpu_loc (xo |-> a) ** pure (Seq.length a == 1024)) **
           (exists* b. on gpu_loc (rco |-> b) ** pure (Seq.length b == 4718592)) **
           (exists* c. on gpu_loc (cvo |-> c) ** pure (Seq.length c == 442368)) **
           pure (Seq.length ws == SZ.v wn /\ Seq.length xs == 1024 /\ Seq.length rs == 4718592 /\ Seq.length cs == 442368)
  ensures exists* a b c. on gpu_loc (xo |-> a) ** on gpu_loc (rco |-> b) ** on gpu_loc (cvo |-> c) **
    pure (optimized_recurrent_decoder_relation ws xs rs cs (I64.v base) (I64.v dn) a b c)
{
  if (I64.gte base 0L && I64.lt base 4294967296L && I64.gte dn 0L && I64.lt dn 18L)
  {
    let b = SZ.uint32_to_sizet (FStar.Int.Cast.int64_to_uint32 base);
    let d = SZ.uint32_to_sizet (FStar.Int.Cast.int64_to_uint32 dn);
    cast_ok base;
    cast_ok dn;
    if (wn >=^ (c32 21555360ul) && b <=^ wn -^ (c32 21555360ul))
    {
      with a. assert on gpu_loc (xo |-> a);
      with b0. assert on gpu_loc (rco |-> b0);
      with c. assert on gpu_loc (cvo |-> c);
      launch_kernel_1 (fun () -> rd_kernel #ws #xs #rs #cs #a #b0 #c w x rc cv xo rco cvo b d);
    } else {
      ()
    }
  } else {
    ()
  }
}
