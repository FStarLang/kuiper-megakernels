module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = Kuiper.SizeT
module I64 = FStar.Int64
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32
module Casts = Kuiper.Float.Casts
module Pow = FStar.Math.Pow
module Sqrt = FStar.Math.Sqrt
module Trig = Kuiper.Real.Trigonometry

#set-options "--z3rlimit 50 --fuel 2 --ifuel 1"

(* ------------------------------------------------------------------ *)
(* Real sums (size-generic)                                            *)

(* sum_{k<n} f(a+k), in k order *)
noextract
let rec lsum (f:int -> real) (a:int) (n:nat) : Tot real (decreases n) =
  if n = 0 then 0.0R else lsum f a (n - 1) +. f (a + (n - 1))

(* sum_{k<m} g(l + k*o), in k order *)
noextract
let rec ssum (g:int -> real) (l o:int) (m:nat) : Tot real (decreases m) =
  if m = 0 then 0.0R else ssum g l o (m - 1) +. g (l + (m - 1) * o)

(* descending reduction tree: d levels, current offset o *)
noextract
let rec tree_r (g:int -> real) (d:nat) (o l:int) : Tot real (decreases d) =
  if d = 0 then g l else tree_r g (d - 1) (2 * o) l +. tree_r g (d - 1) (2 * o) (l + o)

noextract
let rec p2 (d:nat) : pos = if d = 0 then 1 else 2 * p2 (d - 1)

noextract
let rec ssum_interleave (g:int -> real) (l o:int) (m:nat)
  : Lemma (ensures ssum g l o (2 * m) == ssum g l (2 * o) m +. ssum g (l + o) (2 * o) m)
          (decreases m)
  = if m = 0 then ()
    else begin
      ssum_interleave g l o (m - 1);
      assert (ssum g l o (2 * m) == ssum g l o (2 * m - 1) +. g (l + (2 * m - 1) * o));
      assert (ssum g l o (2 * m - 1) == ssum g l o (2 * (m - 1)) +. g (l + (2 * m - 2) * o));
      assert (l + (2 * m - 2) * o == l + (m - 1) * (2 * o));
      assert ((l + o) + (m - 1) * (2 * o) == l + (2 * m - 1) * o)
    end

#push-options "--fuel 2"
noextract
let rec tree_ssum (g:int -> real) (d:nat) (o l:int)
  : Lemma (ensures tree_r g d o l == ssum g l o (p2 d)) (decreases d)
  = if d = 0 then ()
    else begin
      tree_ssum g (d - 1) (2 * o) l;
      tree_ssum g (d - 1) (2 * o) (l + o);
      ssum_interleave g l o (p2 (d - 1))
    end
#pop-options

noextract
let rec lsum_split (f:int -> real) (a:int) (n k:nat)
  : Lemma (ensures lsum f a (n + k) == lsum f a n +. lsum f (a + n) k) (decreases k)
  = if k = 0 then () else lsum_split f a n (k - 1)

noextract
let quad (f:int -> real) (a:int) : real =
  ((f a +. f (a + 1)) +. f (a + 2)) +. f (a + 3)

noextract
let lsum_four (f:int -> real) (a:int) (n:nat)
  : Lemma (lsum f a (n + 4) == lsum f a n +. quad f (a + n))
  = lsum_split f a n 4;
    assert (lsum f (a + n) 1 == f (a + n));
    assert (lsum f (a + n) 2 == lsum f (a + n) 1 +. f (a + n + 1));
    assert (lsum f (a + n) 3 == lsum f (a + n) 2 +. f (a + n + 2));
    assert (lsum f (a + n) 4 == lsum f (a + n) 3 +. f (a + n + 3))

noextract
let qf (f:int -> real) (a:int) (l:int) : real = quad f (a + 4 * l)

noextract
let rec ssum_quads (f:int -> real) (a:int) (m:nat)
  : Lemma (ensures ssum (qf f a) 0 1 m == lsum f a (4 * m)) (decreases m)
  = if m = 0 then ()
    else begin
      ssum_quads f a (m - 1);
      lsum_four f a (4 * (m - 1))
    end

(* squares of a row starting at b *)
noextract
let sqf (x:S.seq real) (b:int) (k:int) : real = at x (b + k) *. at x (b + k)

noextract
let rec lsum_sumsq (x:S.seq real) (b:int) (n:nat)
  : Lemma (ensures lsum (sqf x b) 0 n == sumsq n x b) (decreases n)
  = if n = 0 then () else lsum_sumsq x b (n - 1)

noextract
let rec sumsq_cong (n:nat) (x:S.seq real) (ox:int) (y:S.seq real) (oy:int)
  : Lemma (requires forall (k:nat). k < n ==> at x (ox + k) == at y (oy + k))
          (ensures sumsq n x ox == sumsq n y oy) (decreases n)
  = if n = 0 then () else sumsq_cong (n - 1) x ox y oy

(* ------------------------------------------------------------------ *)
(* Native (bit-level) specification of the kernel values               *)

noextract
let ws (s:S.seq bf16) (k:int) : bf16 =
  if 0 <= k && k < S.length s then S.index s k else zero

noextract
let fv (s:S.seq bf16) (k:int) : f32 = Casts.fcast #bf16 #f32 (ws s k)

noextract
let sqv (s:S.seq bf16) (k:int) : f32 = mul (fv s k) (fv s k)

(* sums[j] of lane l of the row starting at b *)
noextract
let lane_n (s:S.seq bf16) (b l j:int) : f32 =
  add (add zero (sqv s (b + 4 * l + j))) (sqv s (b + 4 * l + 512 + j))

noextract
let leaf_n (s:S.seq bf16) (b:int) (l:int) : f32 =
  add (add (add (lane_n s b l 0) (lane_n s b l 1)) (lane_n s b l 2)) (lane_n s b l 3)

noextract
let rec tree_n (g:int -> f32) (d:nat) (o l:int) : Tot f32 (decreases d) =
  if d = 0 then g l else add (tree_n g (d - 1) (2 * o) l) (tree_n g (d - 1) (2 * o) (l + o))

noextract
let c1024 : f32 = div one (of_int 1024L)
noextract
let ceps : f32 = div one (of_int 1000000L)

noextract
let rms_n (s:S.seq bf16) (b:int) : f32 =
  rsqrt (add (mul (tree_n (leaf_n s b) 7 1 0) c1024) ceps)

(* ------------------------------------------------------------------ *)
(* Ideal counterparts                                                  *)

noextract
let lane_r (x:S.seq real) (b l j:int) : real =
  (0.0R +. sqf x b (4 * l + j)) +. sqf x b (4 * l + 512 + j)

noextract
let leaf_r (x:S.seq real) (b:int) (l:int) : real =
  ((lane_r x b l 0 +. lane_r x b l 1) +. lane_r x b l 2) +. lane_r x b l 3

noextract
let fv_approx (s:S.seq bf16) (x:S.seq real) (k:int)
  : Lemma (requires s %~ x) (ensures fv s k %~ at x k)
  = if 0 <= k && k < S.length s then
      Casts.fcast_approx #bf16 #f32 (S.index s k) (S.index x k)
    else begin
      a0 #bf16;
      Casts.fcast_approx #bf16 #f32 (zero #bf16) 0.0R
    end

noextract
let of_int_approx (k:I64.t) : Lemma (of_int #f32 k %~ Real.of_int (I64.v k))
  = of_int_models_pat f32;
    approx_apply (of_int #f32) Kuiper.Float.Realops.r_of_int k k

noextract
let one_approx () : Lemma (one #f32 %~ 1.0R) = a1 #f32

noextract
let div_approx (x y:f32) (r:real) (q:real{q =!= 0.0R})
  : Lemma (requires x %~ r /\ y %~ q) (ensures div x y %~ (r /. q))
  = div_models_pat f32;
    approx_apply2_dom #_ #_ #_ #_ #_ #_ #(fun _ -> True) #(fun (q:real) -> q =!= 0.0R)
      (div #f32) Kuiper.Float.Realops.r_div x y r q

noextract
let rsqrt_approx (x:f32) (r:real{r >. 0.0R})
  : Lemma (requires x %~ r) (ensures rsqrt x %~ Sqrt.rsqrt r)
  = rsqrt_models_pat f32;
    approx_apply_dom #_ #_ #_ #_ #(fun (r:real) -> r >. 0.0R)
      (rsqrt #f32) Kuiper.Float.Realops.r_rsqrt x r

noextract
let sqv_approx (s:S.seq bf16) (x:S.seq real) (b k:int)
  : Lemma (requires s %~ x) (ensures sqv s (b + k) %~ sqf x b k)
  = fv_approx s x (b + k);
    a_mul (fv s (b + k)) (fv s (b + k)) (at x (b + k)) (at x (b + k))

noextract
let lane_approx (s:S.seq bf16) (x:S.seq real) (b l j:int)
  : Lemma (requires s %~ x) (ensures lane_n s b l j %~ lane_r x b l j)
  = sqv_approx s x b (4 * l + j);
    sqv_approx s x b (4 * l + 512 + j);
    a0 #f32;
    a_add (zero #f32) (sqv s (b + 4 * l + j)) 0.0R (sqf x b (4 * l + j));
    a_add (add zero (sqv s (b + 4 * l + j))) (sqv s (b + 4 * l + 512 + j))
          (0.0R +. sqf x b (4 * l + j)) (sqf x b (4 * l + 512 + j))

noextract
let leaf_approx (s:S.seq bf16) (x:S.seq real) (b l:int)
  : Lemma (requires s %~ x) (ensures leaf_n s b l %~ leaf_r x b l)
  = lane_approx s x b l 0; lane_approx s x b l 1;
    lane_approx s x b l 2; lane_approx s x b l 3

noextract
let rec tree_approx (g:int -> f32) (h:int -> real) (d:nat) (o l:int)
  : Lemma (requires forall k. g k %~ h k) (ensures tree_n g d o l %~ tree_r h d o l) (decreases d)
  = if d = 0 then ()
    else begin
      tree_approx g h (d - 1) (2 * o) l;
      tree_approx g h (d - 1) (2 * o) (l + o);
      a_add (tree_n g (d - 1) (2 * o) l) (tree_n g (d - 1) (2 * o) (l + o))
            (tree_r h (d - 1) (2 * o) l) (tree_r h (d - 1) (2 * o) (l + o))
    end

noextract
let rec ssum_add (g g1 g2:int -> real) (l o:int) (m:nat)
  : Lemma (requires forall k. g k == g1 k +. g2 k)
          (ensures ssum g l o m == ssum g1 l o m +. ssum g2 l o m) (decreases m)
  = if m = 0 then () else ssum_add g g1 g2 l o (m - 1)

noextract
let leaf_quads (x:S.seq real) (b l:int)
  : Lemma (leaf_r x b l == qf (sqf x b) 0 l +. qf (sqf x b) 512 l)
  = assert (512 + 4 * l == 4 * l + 512);
    assert (512 + 4 * l + 1 == 4 * l + 512 + 1);
    assert (512 + 4 * l + 2 == 4 * l + 512 + 2);
    assert (512 + 4 * l + 3 == 4 * l + 512 + 3)

noextract
let p2_7 () : Lemma (p2 7 == 128)
  = assert (p2 1 == 2); assert (p2 2 == 4); assert (p2 3 == 8);
    assert (p2 4 == 16); assert (p2 5 == 32); assert (p2 6 == 64)

noextract
let tree_sumsq (x:S.seq real) (b:int)
  : Lemma (tree_r (leaf_r x b) 7 1 0 == sumsq 1024 x b)
  = tree_ssum (leaf_r x b) 7 1 0;
    p2_7 ();
    Classical.forall_intro (leaf_quads x b);
    ssum_add (leaf_r x b) (qf (sqf x b) 0) (qf (sqf x b) 512) 0 1 128;
    ssum_quads (sqf x b) 0 128;
    ssum_quads (sqf x b) 512 128;
    lsum_split (sqf x b) 0 512 512;
    lsum_sumsq x b 1024

noextract
let eps_arith (sq:real)
  : Lemma (requires sq >=. 0.0R)
          (ensures sq *. (1.0R /. 1024.0R) +. 1.0R /. 1000000.0R >. 0.0R /\
                   sq *. (1.0R /. 1024.0R) +. 1.0R /. 1000000.0R == sq /. Real.of_int 1024 +. rms_eps)
  = assert (Real.of_int 1024 == 1024.0R)

noextract
let rms_approx (s:S.seq bf16) (x:S.seq real) (b:int)
  : Lemma (requires s %~ x)
          (ensures rms_n s b %~ Sqrt.rsqrt (sumsq 1024 x b /. Real.of_int 1024 +. rms_eps))
  = Classical.forall_intro (Classical.move_requires (leaf_approx s x b));
    tree_approx (leaf_n s b) (leaf_r x b) 7 1 0;
    tree_sumsq x b;
    let t = tree_n (leaf_n s b) 7 1 0 in
    let sq = sumsq 1024 x b in
    one_approx ();
    of_int_approx 1024L;
    of_int_approx 1000000L;
    div_approx one (of_int 1024L) 1.0R 1024.0R;
    div_approx one (of_int 1000000L) 1.0R 1000000.0R;
    a_mul t c1024 sq (1.0R /. 1024.0R);
    a_add (mul t c1024) ceps (sq *. (1.0R /. 1024.0R)) (1.0R /. 1000000.0R);
    let a = sq *. (1.0R /. 1024.0R) +. 1.0R /. 1000000.0R in
    eps_arith sq;
    rsqrt_approx (add (mul t c1024) ceps) a

(* ------------------------------------------------------------------ *)
(* Native specification of the three outputs                           *)

#set-options "--z3rlimit 50 --fuel 1 --ifuel 1"

module Cast = FStar.Int.Cast

noextract
let ctok (st:S.seq I64.t) (t:int) : int =
  let k = tok_at st t in if 0 <= k && k < 151936 then k else 0

noextract
let rbase (st:S.seq I64.t) (eoff:int) (t:int) : int = eoff + ctok st t * 1024

inline_for_extraction noextract
let norm_elem (v:bf16) (inv:f32) (w:bf16) : bf16 =
  let sc : bf16 = Casts.fcast #f32 #bf16 (mul (Casts.fcast #bf16 #f32 v) inv) in
  Casts.fcast #f32 #bf16 (mul (Casts.fcast #bf16 #f32 sc) (Casts.fcast #bf16 #f32 w))

noextract
let res_n (sw:S.seq bf16) (st:S.seq I64.t) (eoff:int) (i:int) : bf16 =
  ws sw (rbase st eoff (i / 1024) + i % 1024)

noextract
let nrm_n (sw:S.seq bf16) (st:S.seq I64.t) (eoff noff:int) (i:int) : bf16 =
  let b = rbase st eoff (i / 1024) in
  norm_elem (ws sw (b + i % 1024)) (rms_n sw b) (ws sw (noff + i % 1024))

inline_for_extraction noextract
let angle_x (p c:I64.t) : f32 =
  mul (of_int #f32 p) (div one (pow (of_int #f32 1000000L) (div (of_int #f32 c) (of_int #f32 64L))))

noextract
let rang (p c:int) : f32 =
  if 0 <= p && p <= 2147483647 && 0 <= c && c < 64
  then angle_x (I64.int_to_t p) (I64.int_to_t c) else zero

noextract
let rcos (p0 k:int) : bf16 = Casts.fcast #f32 #bf16 (cos (rang (p0 + k / 128) ((k % 128) % 64)))
noextract
let rsin (p0 k:int) : bf16 = Casts.fcast #f32 #bf16 (sin (rang (p0 + k / 128) ((k % 128) % 64)))

noextract
let kfacts (sw:S.seq bf16) (st:S.seq I64.t) (eoff noff p0:int) (r0 r1 r2:S.seq bf16) : prop =
  S.length r0 == 7168 /\ S.length r1 == 7168 /\ S.length r2 == 1792 /\
  (forall (i:nat). i < 7168 ==> S.index r0 i == res_n sw st eoff i /\ S.index r1 i == nrm_n sw st eoff noff i) /\
  (forall (k:nat). k < 896 ==> S.index r2 k == rcos p0 k) /\
  (forall (k:nat). 896 <= k /\ k < 1792 ==> S.index r2 k == rsin p0 (k - 896))

[@@"opaque_to_smt"]
noextract
let kf (sw:S.seq bf16) (st:S.seq I64.t) (eoff noff p0:int) (r0 r1 r2:S.seq bf16) : prop =
  kfacts sw st eoff noff p0 r0 r1 r2

noextract
let kf_intro (sw:S.seq bf16) (st:S.seq I64.t) (eoff noff p0:int) (r0 r1 r2:S.seq bf16)
  : Lemma (requires kfacts sw st eoff noff p0 r0 r1 r2) (ensures kf sw st eoff noff p0 r0 r1 r2)
  = reveal_opaque (`%kf) (kf sw st eoff noff p0 r0 r1 r2)

noextract
let kf_len (sw:S.seq bf16) (st:S.seq I64.t) (eoff noff p0:int) (r0 r1 r2:S.seq bf16)
  : Lemma (requires kf sw st eoff noff p0 r0 r1 r2)
          (ensures S.length r0 == 7168 /\ S.length r1 == 7168 /\ S.length r2 == 1792)
  = reveal_opaque (`%kf) (kf sw st eoff noff p0 r0 r1 r2)

noextract
let kf_row (sw:S.seq bf16) (st:S.seq I64.t) (eoff noff p0:int) (r0 r1 r2:S.seq bf16) (i:nat)
  : Lemma (requires kf sw st eoff noff p0 r0 r1 r2 /\ i < 7168)
          (ensures S.length r0 == 7168 /\ S.length r1 == 7168 /\
                   S.index r0 i == res_n sw st eoff i /\ S.index r1 i == nrm_n sw st eoff noff i)
  = reveal_opaque (`%kf) (kf sw st eoff noff p0 r0 r1 r2)

noextract
let kf_rot (sw:S.seq bf16) (st:S.seq I64.t) (eoff noff p0:int) (r0 r1 r2:S.seq bf16) (i:nat)
  : Lemma (requires kf sw st eoff noff p0 r0 r1 r2 /\ i < 1792)
          (ensures S.length r2 == 1792 /\
                   (if i < 896 then S.index r2 i == rcos p0 i else S.index r2 i == rsin p0 (i - 896)))
  = reveal_opaque (`%kf) (kf sw st eoff noff p0 r0 r1 r2)

(* ------------------------------------------------------------------ *)
(* Relation proof                                                      *)

noextract
let seq_approx_intro (s:S.seq bf16) (r:S.seq real)
  : Lemma (requires S.length s == S.length r /\
                    (forall (i:nat). i < S.length s ==> S.index s i %~ S.index r i))
          (ensures s %~ r)
  = ()

noextract
let seq_approx_elim (s:S.seq bf16) (r:S.seq real) (i:nat)
  : Lemma (requires s %~ r /\ i < S.length s)
          (ensures S.length r == S.length s /\ S.index s i %~ S.index r i)
  = ()

noextract
let divmod (t c:int)
  : Lemma (requires 0 <= c /\ c < 1024) (ensures (t * 1024 + c) / 1024 == t /\ (t * 1024 + c) % 1024 == c)
  = FStar.Math.Lemmas.lemma_div_plus c t 1024;
    FStar.Math.Lemmas.lemma_mod_plus c t 1024;
    FStar.Math.Lemmas.small_div c 1024;
    FStar.Math.Lemmas.small_mod c 1024

noextract
let ctok_ok (st:S.seq I64.t) (t:int)
  : Lemma (requires tokens_ok st /\ S.length st == 7 /\ 0 <= t /\ t < 7)
          (ensures ctok st t == tok_at st t /\ 0 <= tok_at st t /\ tok_at st t < 151936)
  = ()

noextract
let win_at (rw:S.seq real) (o n:int) (k:int)
  : Lemma (requires 0 <= o /\ 0 <= n /\ o + n <= S.length rw /\ 0 <= k /\ k < n)
          (ensures at (window rw o n) k == at rw (o + k))
  = ()

noextract
let embed_idx (we:S.seq real) (st:S.seq I64.t) (i:nat{i < 7168})
  : Lemma (S.index (embed_rows 7 we st) i == at we (tok_at st (i / 1024) * 1024 + i % 1024))
  = reveal_opaque (`%embed_rows) (embed_rows 7 we st)

noextract
let rmsnorm_idx (x w:S.seq real) (i:nat{i < 7168})
  : Lemma (S.index (rmsnorm_rows 7 1024 x w) i ==
           at w (i % 1024) *. (at x i *. rms_inv 1024 x ((i / 1024) * 1024)))
  = reveal_opaque (`%rmsnorm_rows) (rmsnorm_rows 7 1024 x w)

noextract
let norm_elem_approx (v:bf16) (inv:f32) (w:bf16) (rv ri rw:real)
  : Lemma (requires v %~ rv /\ inv %~ ri /\ w %~ rw)
          (ensures norm_elem v inv w %~ (rw *. (rv *. ri)))
  = Casts.fcast_approx #bf16 #f32 v rv;
    let m = mul (Casts.fcast #bf16 #f32 v) inv in
    a_mul (Casts.fcast #bf16 #f32 v) inv rv ri;
    Casts.fcast_approx #f32 #bf16 m (rv *. ri);
    let sc : bf16 = Casts.fcast #f32 #bf16 m in
    Casts.fcast_approx #bf16 #f32 sc (rv *. ri);
    Casts.fcast_approx #bf16 #f32 w rw;
    let m2 = mul (Casts.fcast #bf16 #f32 sc) (Casts.fcast #bf16 #f32 w) in
    a_mul (Casts.fcast #bf16 #f32 sc) (Casts.fcast #bf16 #f32 w) (rv *. ri) rw;
    Casts.fcast_approx #f32 #bf16 m2 ((rv *. ri) *. rw);
    assert ((rv *. ri) *. rw == rw *. (rv *. ri))

noextract
let rang_approx (p c:int)
  : Lemma (requires 0 <= p /\ p <= 2147483647 /\ 0 <= c /\ c < 64)
          (ensures rang p c %~ rope_angle p c)
  = let pi = I64.int_to_t p in
    let ci = I64.int_to_t c in
    of_int_approx pi;
    of_int_approx ci;
    of_int_approx 64L;
    of_int_approx 1000000L;
    let e = div (of_int #f32 ci) (of_int #f32 64L) in
    div_approx (of_int #f32 ci) (of_int #f32 64L) (Real.of_int c) 64.0R;
    F32.pow_approx (of_int #f32 1000000L) e 1000000.0R (Real.of_int c /. 64.0R);
    let pw : Pow.rpos = Pow.pow 1000000.0R (Real.of_int c /. 64.0R) in
    one_approx ();
    let q = pow (of_int #f32 1000000L) e in
    div_approx one q 1.0R pw;
    a_mul (of_int #f32 pi) (div one q) (Real.of_int p) (1.0R /. pw)

noextract
let rope_idx (p0:int) (i:nat{i < 1792})
  : Lemma (S.length (rope_pair p0) == 1792 /\
           S.index (rope_pair p0) i ==
           (if i < 896 then Trig.cos (rope_angle (p0 + i / 128) ((i % 128) % 64))
            else Trig.sin (rope_angle (p0 + (i - 896) / 128) (((i - 896) % 128) % 64))))
  = reveal_opaque (`%rope_cos_table) (rope_cos_table p0 7);
    reveal_opaque (`%rope_sin_table) (rope_sin_table p0 7);
    if i < 896 then S.lemma_index_app1 (rope_cos_table p0 7) (rope_sin_table p0 7) i
    else S.lemma_index_app2 (rope_cos_table p0 7) (rope_sin_table p0 7) i

noextract
let rot_i (sw:S.seq bf16) (st:S.seq I64.t) (a2 a3 a4:int) (r0 r1 r2:S.seq bf16) (i:nat)
  : Lemma (requires kf sw st a2 a3 a4 r0 r1 r2 /\ 0 <= a4 /\ a4 <= 2147483647 - 6 /\ i < 1792)
          (ensures S.length r2 == 1792 /\ ws r2 i %~ at (rope_pair a4) i)
  = rope_idx a4 i;
    kf_rot sw st a2 a3 a4 r0 r1 r2 i;
    if i < 896 then begin
      rang_approx (a4 + i / 128) ((i % 128) % 64);
      F32.cos_approx (rang (a4 + i / 128) ((i % 128) % 64)) (rope_angle (a4 + i / 128) ((i % 128) % 64));
      Casts.fcast_approx #f32 #bf16 (cos (rang (a4 + i / 128) ((i % 128) % 64)))
        (Trig.cos (rope_angle (a4 + i / 128) ((i % 128) % 64)))
    end else begin
      let k = i - 896 in
      rang_approx (a4 + k / 128) ((k % 128) % 64);
      F32.sin_approx (rang (a4 + k / 128) ((k % 128) % 64)) (rope_angle (a4 + k / 128) ((k % 128) % 64));
      Casts.fcast_approx #f32 #bf16 (sin (rang (a4 + k / 128) ((k % 128) % 64)))
        (Trig.sin (rope_angle (a4 + k / 128) ((k % 128) % 64)))
    end

noextract
let premises (sw:S.seq bf16) (st:S.seq I64.t) (a2 a3 a4:int) : prop =
  S.length st == 7 /\ tokens_ok st /\
  0 <= a2 /\ a2 <= int_max - embed_elements /\ a2 + embed_elements <= S.length sw /\
  0 <= a3 /\ a3 <= int_max - 1024 /\ a3 + 1024 <= S.length sw /\
  0 <= a4 /\ a4 <= int_max - 6

noextract
let win_t (sw:S.seq bf16) (st:S.seq I64.t) (a2 a3 a4:int) (r0 r1 r2:S.seq bf16) (t:nat)
  : Lemma (requires premises sw st a2 a3 a4 /\ kf sw st a2 a3 a4 r0 r1 r2 /\ t < 7)
          (ensures window r0 (t * 1024) 1024 == window sw (a2 + tok_at st t * 1024) 1024)
  = ctok_ok st t;
    kf_len sw st a2 a3 a4 r0 r1 r2;
    let b = a2 + tok_at st t * 1024 in
    let aux (k:nat{k < 1024})
      : Lemma (S.index (S.slice r0 (t * 1024) (t * 1024 + 1024)) k == S.index (S.slice sw b (b + 1024)) k)
      = divmod t k;
        kf_row sw st a2 a3 a4 r0 r1 r2 (t * 1024 + k)
    in
    Classical.forall_intro aux;
    S.lemma_eq_intro (S.slice r0 (t * 1024) (t * 1024 + 1024)) (S.slice sw b (b + 1024))

noextract
let xrow (rw:S.seq real) (st:S.seq I64.t) (a2:int) : S.seq real =
  embed_rows 7 (window rw a2 embed_elements) st

noextract
let x_at (sw:S.seq bf16) (st:S.seq I64.t) (a2 a3 a4:int) (rw:S.seq real) (i:nat)
  : Lemma (requires premises sw st a2 a3 a4 /\ S.length rw == S.length sw /\ i < 7168)
          (ensures at (xrow rw st a2) i == at rw (rbase st a2 (i / 1024) + i % 1024) /\
                   0 <= rbase st a2 (i / 1024) /\ rbase st a2 (i / 1024) + 1024 <= S.length sw)
  = embed_idx (window rw a2 embed_elements) st i;
    ctok_ok st (i / 1024);
    win_at rw a2 embed_elements (tok_at st (i / 1024) * 1024 + i % 1024)

noextract
let sumsq_row (sw:S.seq bf16) (st:S.seq I64.t) (a2 a3 a4:int) (rw:S.seq real) (t:nat)
  : Lemma (requires premises sw st a2 a3 a4 /\ S.length rw == S.length sw /\ t < 7)
          (ensures sumsq 1024 (xrow rw st a2) (t * 1024) == sumsq 1024 rw (rbase st a2 t))
  = let aux (k:nat)
      : Lemma (k < 1024 ==> at (xrow rw st a2) (t * 1024 + k) == at rw (rbase st a2 t + k))
      = if k < 1024 then begin
          divmod t k;
          x_at sw st a2 a3 a4 rw (t * 1024 + k)
        end
    in
    Classical.forall_intro aux;
    sumsq_cong 1024 (xrow rw st a2) (t * 1024) rw (rbase st a2 t)

noextract
let ws_approx (sw:S.seq bf16) (rw:S.seq real) (k:int)
  : Lemma (requires sw %~ rw /\ 0 <= k /\ k < S.length sw)
          (ensures ws sw k %~ at rw k)
  = seq_approx_elim sw rw k

noextract
let row_i (sw:S.seq bf16) (st:S.seq I64.t) (a2 a3 a4:int) (r0 r1 r2:S.seq bf16) (rw:S.seq real) (i:nat)
  : Lemma (requires premises sw st a2 a3 a4 /\ kf sw st a2 a3 a4 r0 r1 r2 /\ sw %~ rw /\ i < 7168)
          (ensures S.length r0 == 7168 /\ S.length r1 == 7168 /\
                   ws r0 i %~ at (xrow rw st a2) i /\
                   ws r1 i %~ at (rmsnorm_rows 7 1024 (xrow rw st a2) (window rw a3 1024)) i)
  = seq_approx_elim sw rw 0;
    kf_row sw st a2 a3 a4 r0 r1 r2 i;
    let t = i / 1024 in
    let c = i % 1024 in
    let b = rbase st a2 t in
    x_at sw st a2 a3 a4 rw i;
    ws_approx sw rw (b + c);
    ws_approx sw rw (a3 + c);
    rmsnorm_idx (xrow rw st a2) (window rw a3 1024) i;
    win_at rw a3 1024 c;
    sumsq_row sw st a2 a3 a4 rw t;
    rms_approx sw rw b;
    norm_elem_approx (ws sw (b + c)) (rms_n sw b) (ws sw (a3 + c))
      (at rw (b + c)) (Sqrt.rsqrt (sumsq 1024 rw b /. Real.of_int 1024 +. rms_eps)) (at rw (a3 + c))

noextract
let rw_rel (sw:S.seq bf16) (st:S.seq I64.t) (a2 a3 a4:int) (r0 r1 r2:S.seq bf16) (rw:S.seq real)
  : Lemma (requires premises sw st a2 a3 a4 /\ kf sw st a2 a3 a4 r0 r1 r2 /\ sw %~ rw)
          (ensures r0 %~ xrow rw st a2 /\
                   r1 %~ rmsnorm_rows 7 1024 (xrow rw st a2) (window rw a3 1024))
  = kf_len sw st a2 a3 a4 r0 r1 r2;
    Classical.forall_intro (Classical.move_requires (row_i sw st a2 a3 a4 r0 r1 r2 rw));
    seq_approx_intro r0 (xrow rw st a2);
    seq_approx_intro r1 (rmsnorm_rows 7 1024 (xrow rw st a2) (window rw a3 1024))

noextract
let core_rel (sw:S.seq bf16) (st:S.seq I64.t) (a2 a3 a4:int) (r0 r1 r2:S.seq bf16)
  : Lemma (requires premises sw st a2 a3 a4 /\ kf sw st a2 a3 a4 r0 r1 r2)
          (ensures base_embedding_relation sw st a2 a3 a4 r0 r1 r2)
  = kf_len sw st a2 a3 a4 r0 r1 r2;
    Classical.forall_intro (Classical.move_requires (win_t sw st a2 a3 a4 r0 r1 r2));
    Classical.forall_intro (Classical.move_requires (rot_i sw st a2 a3 a4 r0 r1 r2));
    rope_idx a4 0;
    seq_approx_intro r2 (rope_pair a4);
    Classical.forall_intro (Classical.move_requires (rw_rel sw st a2 a3 a4 r0 r1 r2))

noextract
let final_rel (sw:S.seq bf16) (st:S.seq I64.t) (a2 a3 a4:int) (r0 r1 r2:S.seq bf16)
  : Lemma (requires kfacts sw st a2 a3 a4 r0 r1 r2)
          (ensures base_embedding_relation sw st a2 a3 a4 r0 r1 r2)
  = kf_intro sw st a2 a3 a4 r0 r1 r2;
    Classical.move_requires (core_rel sw st a2 a3 a4 r0 r1) r2

(* ------------------------------------------------------------------ *)
(* Executable kernel                                                   *)

noextract
let lv (o:int) : nat =
  if o >= 128 then 0 else if o >= 64 then 1 else if o >= 32 then 2 else if o >= 16 then 3
  else if o >= 8 then 4 else if o >= 4 then 5 else if o >= 2 then 6 else 7

noextract
let ispow (o:int) : bool =
  o = 1 || o = 2 || o = 4 || o = 8 || o = 16 || o = 32 || o = 64 || o = 128

noextract
let tree_step (g:int -> f32) (o:int) (l:int)
  : Lemma (requires ispow o /\ o > 1)
          (ensures ispow (o / 2) /\ o / 2 >= 1 /\
                   tree_n g (lv (o / 2)) (o / 2) l == add (tree_n g (lv o) o l) (tree_n g (lv o) o (l + o / 2)))
  = assert (lv (o / 2) == lv o + 1 /\ 2 * (o / 2) == o)

inline_for_extraction noextract
let sz_to_i64 (x:SZ.t{SZ.v x < 2147483648}) : y:I64.t{I64.v y == SZ.v x} =
  Cast.uint32_to_int64 (SZ.sizet_to_u32 x)

inline_for_extraction noextract
let i64_to_sz (x:I64.t{0 <= I64.v x /\ I64.v x < 2147483648}) : y:SZ.t{SZ.v y == I64.v x} =
  SZ.uint32_to_sizet (Cast.int64_to_uint32 x)

inline_for_extraction noextract
let sqf32 (v:bf16) : f32 = let x = Casts.fcast #bf16 #f32 v in mul x x

instance send_global_array_contents
  (#a : Type0)
  (arr : array a{is_global arr})
  (#f : perm)
  (contents : S.seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

noextract
let idx_bnd (r c:int)
  : Lemma (requires 0 <= r /\ r < 7 /\ 0 <= c /\ c < 1024)
          (ensures 0 <= r * 1024 + c /\ r * 1024 + c < 7168 /\ r * 1024 + c + 1 == r * 1024 + (c + 1))
  = ()

noextract
let col_step (sw:S.seq bf16) (st:S.seq I64.t) (eoff noff:int) (t c:int)
  : Lemma (requires 0 <= c /\ c < 1024)
          (ensures res_n sw st eoff (t * 1024 + c) == ws sw (rbase st eoff t + c) /\
                   nrm_n sw st eoff noff (t * 1024 + c) ==
                     norm_elem (ws sw (rbase st eoff t + c)) (rms_n sw (rbase st eoff t)) (ws sw (noff + c)))
  = divmod t c

inline_for_extraction noextract
fn rms_row
  (n : SZ.t) (w : larray bf16 (SZ.v n)) (b : SZ.t{SZ.v b + 1024 <= SZ.v n})
  (kd : I64.t{kd == 1024L}) (ke : I64.t{ke == 1000000L})
  (#fw : perm) (#sw : erased (S.seq bf16))
  preserves gpu ** (w |-> Frac fw sw)
  returns r : f32
  ensures pure (r == rms_n sw (SZ.v b))
{
  pts_to_len w;
  let z : f32 = zero;
  let mut partial = [| z; 128sz |];
  let mut lane = 0sz;
  while (!lane <^ 128sz)
    invariant exists* (l:SZ.t{SZ.v l <= 128}) s. lane |-> l ** (partial |-> s) **
      pure (S.length s == 128 /\
            (forall (j:nat). j < SZ.v l ==> S.index s j == leaf_n sw (SZ.v b) j))
    decreases (128 - SZ.v !lane)
  {
    let l = !lane;
    let k = b +^ 4sz *^ l;
    let s0 = add (add zero (sqf32 w.(k))) (sqf32 w.(k +^ 512sz));
    let s1 = add (add zero (sqf32 w.(k +^ 1sz))) (sqf32 w.(k +^ 513sz));
    let s2 = add (add zero (sqf32 w.(k +^ 2sz))) (sqf32 w.(k +^ 514sz));
    let s3 = add (add zero (sqf32 w.(k +^ 3sz))) (sqf32 w.(k +^ 515sz));
    assert pure (s0 == lane_n sw (SZ.v b) (SZ.v l) 0);
    assert pure (s1 == lane_n sw (SZ.v b) (SZ.v l) 1);
    assert pure (s2 == lane_n sw (SZ.v b) (SZ.v l) 2);
    assert pure (s3 == lane_n sw (SZ.v b) (SZ.v l) 3);
    partial.(l) <- add (add (add s0 s1) s2) s3;
    lane := l +^ 1sz;
  };
  let mut off = 128sz;
  while (!off >^ 1sz)
    invariant exists* (o:SZ.t) s. off |-> o ** (partial |-> s) **
      pure (S.length s == 128 /\ ispow (SZ.v o) /\
            (forall (j:nat). j < SZ.v o ==> S.index s j == tree_n (leaf_n sw (SZ.v b)) (lv (SZ.v o)) (SZ.v o) j))
    decreases (SZ.v !off)
  {
    let o = !off;
    let h = o /^ 2sz;
    let mut i = 0sz;
    while (!i <^ h)
      invariant exists* (iv:SZ.t{SZ.v iv <= SZ.v h}) s. i |-> iv ** (partial |-> s) **
        pure (S.length s == 128 /\
              (forall (j:nat). j < SZ.v iv ==> S.index s j == tree_n (leaf_n sw (SZ.v b)) (lv (SZ.v h)) (SZ.v h) j) /\
              (forall (j:nat). SZ.v iv <= j /\ j < SZ.v o ==>
                 S.index s j == tree_n (leaf_n sw (SZ.v b)) (lv (SZ.v o)) (SZ.v o) j))
      decreases (SZ.v h - SZ.v !i)
    {
      let iv = !i;
      let x = partial.(iv);
      let y = partial.(iv +^ h);
      tree_step (leaf_n sw (SZ.v b)) (SZ.v o) (SZ.v iv);
      partial.(iv) <- add x y;
      i := iv +^ 1sz;
    };
    tree_step (leaf_n sw (SZ.v b)) (SZ.v o) 0;
    off := h;
  };
  let t = partial.(0sz);
  let mean = mul t (div one (of_int #f32 kd));
  rsqrt (add mean (div one (of_int #f32 ke)))
}

#push-options "--z3rlimit 200"
inline_for_extraction noextract
fn kernel
  (n : SZ.t)
  (w : larray bf16 (SZ.v n))
  (tokens : larray I64.t 7)
  (eoff : SZ.t) (noff : SZ.t) (pos : I64.t)
  (kd : I64.t{kd == 1024L}) (ke : I64.t{ke == 1000000L})
  (res : larray bf16 7168) (nrm : larray bf16 7168) (rot : larray bf16 1792)
  (#fw #ft : perm) (#sw : erased (S.seq bf16)) (#st : erased (S.seq I64.t))
  (#q0 #q1 #q2 : erased (S.seq bf16))
  (_ : squash (SZ.v eoff + 155582464 <= SZ.v n /\ SZ.v eoff <= 2147483647 - 155582464 /\
               SZ.v noff + 1024 <= SZ.v n /\ 0 <= I64.v pos /\ I64.v pos <= 2147483647 - 6))
  preserves gpu ** (w |-> Frac fw sw) ** (tokens |-> Frac ft st)
  requires (res |-> q0) ** (nrm |-> q1) ** (rot |-> q2)
  ensures exists* r0 r1 r2. (res |-> r0) ** (nrm |-> r1) ** (rot |-> r2) **
    pure (kfacts sw st (SZ.v eoff) (SZ.v noff) (I64.v pos) r0 r1 r2)
{
  pts_to_len rot;
  pts_to_len w;
  pts_to_len tokens;
  let mut i = 0sz;
  while (!i <^ 896sz)
    invariant exists* (iv:SZ.t{SZ.v iv <= 896}) s. i |-> iv ** (rot |-> s) **
      pure (S.length s == 1792 /\
            (forall (k:nat). k < SZ.v iv ==> S.index s k == rcos (I64.v pos) k) /\
            (forall (k:nat). 896 <= k /\ k < 896 + SZ.v iv ==> S.index s k == rsin (I64.v pos) (k - 896)))
    decreases (896 - SZ.v !i)
  {
    let iv = !i;
    let row = iv /^ 128sz;
    let col = iv %^ 128sz;
    let p = I64.add (sz_to_i64 row) pos;
    let c = sz_to_i64 (col %^ 64sz);
    let a = angle_x p c;
    assert pure (a == rang (I64.v pos + SZ.v iv / 128) ((SZ.v iv % 128) % 64));
    rot.(iv) <- Casts.fcast #f32 #bf16 (cos a);
    rot.(iv +^ 896sz) <- Casts.fcast #f32 #bf16 (sin a);
    i := iv +^ 1sz;
  };
  pts_to_len res;
  pts_to_len nrm;
  let mut r = 0sz;
  while (!r <^ 7sz)
    invariant exists* (rv:SZ.t{SZ.v rv <= 7}) s0 s1. r |-> rv ** (res |-> s0) ** (nrm |-> s1) **
      pure (S.length s0 == 7168 /\ S.length s1 == 7168 /\
            (forall (j:nat). j < 7168 /\ j < SZ.v rv * 1024 ==>
               S.index s0 j == res_n sw st (SZ.v eoff) j /\ S.index s1 j == nrm_n sw st (SZ.v eoff) (SZ.v noff) j))
    decreases (7 - SZ.v !r)
  {
    let rv = !r;
    let tk = tokens.(rv);
    let tc : (x:I64.t{0 <= I64.v x /\ I64.v x < 151936}) =
      if I64.gte tk 0L && I64.lt tk 151936L then tk else 0L;
    assert pure (I64.v tc == ctok st (SZ.v rv));
    let b = eoff +^ i64_to_sz tc *^ 1024sz;
    assert pure (SZ.v b == rbase st (SZ.v eoff) (SZ.v rv));
    assert pure (SZ.v b + 1024 <= SZ.v n /\ SZ.fits (SZ.v n));
    let inv = rms_row n w b kd ke;
    let mut c = 0sz;
    while (!c <^ 1024sz)
      invariant exists* (cv:SZ.t{SZ.v cv <= 1024}) s0 s1. c |-> cv ** (res |-> s0) ** (nrm |-> s1) **
        pure (S.length s0 == 7168 /\ S.length s1 == 7168 /\
              (forall (j:nat). j < 7168 /\ j < SZ.v rv * 1024 + SZ.v cv ==>
                 S.index s0 j == res_n sw st (SZ.v eoff) j /\ S.index s1 j == nrm_n sw st (SZ.v eoff) (SZ.v noff) j))
      decreases (1024 - SZ.v !c)
    {
      let cv = !c;
      col_step sw st (SZ.v eoff) (SZ.v noff) (SZ.v rv) (SZ.v cv);
      idx_bnd (SZ.v rv) (SZ.v cv);
      assert pure (SZ.v b + SZ.v cv < SZ.v n);
      let idx = rv *^ 1024sz +^ cv;
      let v = w.(b +^ cv);
      let wv = w.(noff +^ cv);
      assert pure (v == ws sw (SZ.v b + SZ.v cv) /\ wv == ws sw (SZ.v noff + SZ.v cv));
      res.(idx) <- v;
      nrm.(idx) <- norm_elem v inv wv;
      c := cv +^ 1sz;
    };
    assert pure (SZ.v rv * 1024 + 1024 == (SZ.v rv + 1) * 1024);
    r := rv +^ 1sz;
  };
  ()
}
#pop-options

fn kplay_run
  (n : SZ.t)
  (input0 : larray bf16 (SZ.v n){is_global input0})
  (input1 : larray I64.t 7{is_global input1})
  (input2 input3 input4 : I64.t)
  (output0 : larray bf16 7168{is_global output0})
  (output1 : larray bf16 7168{is_global output1})
  (output2 : larray bf16 1792{is_global output2})
  (#f0 #f1 : perm)
  (#s0 : erased (S.seq bf16)) (#s1 : erased (S.seq I64.t))
  (#o0 #o1 #o2 : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (input0 |-> Frac f0 s0) ** on gpu_loc (input1 |-> Frac f1 s1)
  requires on gpu_loc (output0 |-> o0) ** on gpu_loc (output1 |-> o1) ** on gpu_loc (output2 |-> o2)
  ensures exists* r0 r1 r2.
    on gpu_loc (output0 |-> r0) ** on gpu_loc (output1 |-> r1) ** on gpu_loc (output2 |-> r2) **
    pure (base_embedding_relation s0 s1 (I64.v input2) (I64.v input3) (I64.v input4) r0 r1 r2)
{
  map_loc gpu_loc #(input0 |-> Frac f0 s0) #((input0 |-> Frac f0 s0) ** pure (S.length s0 == SZ.v n))
    fn _ { pts_to_len input0 };
  on_star_elim #gpu_loc (input0 |-> Frac f0 s0) (pure (S.length s0 == SZ.v n));
  on_pure_elim gpu_loc (S.length s0 == SZ.v n);
  let n64 = Cast.uint32_to_int64 (SZ.sizet_to_u32 n);
  let ok =
    I64.gte input2 0L && I64.lte input2 1991901183L &&
    I64.lte input2 (I64.sub n64 155582464L) &&
    I64.gte input3 0L && I64.lte input3 2147482623L &&
    I64.lte input3 (I64.sub n64 1024L) &&
    I64.gte input4 0L && I64.lte input4 2147483641L;
  if ok {
    let eoff = i64_to_sz input2;
    let noff = i64_to_sz input3;
    let kd = 1024L;
    let ke = 1000000L;
    launch_kernel_1 (fun () -> kernel n input0 input1 eoff noff input4 kd ke output0 output1 output2
                                 #f0 #f1 #s0 #s1 #o0 #o1 #o2 ());
    with r0 r1 r2. assert (on gpu_loc (output0 |-> r0) ** on gpu_loc (output1 |-> r1) ** on gpu_loc (output2 |-> r2));
    assert pure (kfacts s0 s1 (SZ.v eoff) (SZ.v noff) (I64.v input4) r0 r1 r2);
    final_rel s0 s1 (I64.v input2) (I64.v input3) (I64.v input4) r0 r1 r2;
    ()
  } else {
    ()
  }
}
