module KPlay.Functional

#lang-pulse

open Kuiper
module S = FStar.Seq
module SZ = FStar.SizeT
module I64 = FStar.Int64
module U32 = FStar.UInt32
module Cast = FStar.Int.Cast
module Casts = Kuiper.Float.Casts
module FB = Kuiper.Floating.Base
module F32 = Kuiper.Float32
module Pow = FStar.Math.Pow
module Trig = Kuiper.Real.Trigonometry
module SS = KPlay.SharedSpec
module A = Pulse.Lib.Array

#set-options "--z3rlimit 20 --fuel 1 --ifuel 1"

(* ---------------- conversions ---------------- *)

inline_for_extraction noextract
let sz_to_i64 (x : sz) : (r : I64.t{I64.v r == SZ.v x}) =
  Cast.uint32_to_int64 (SZ.sizet_to_uint32 x)

inline_for_extraction noextract
let i64_to_sz (x : I64.t{0 <= I64.v x /\ I64.v x < 151936}) : (r : sz{SZ.v r == I64.v x}) =
  SZ.uint32_to_sizet (Cast.int64_to_uint32 x)

(* ---------------- residual (embedding copy) ---------------- *)

inline_for_extraction noextract
fn read_emb
  (a0 : A.larray bf16 155582464)
  (#p0 : perm)
  (#s0 : erased (S.seq bf16))
  (t : sz) (c : sz)
  preserves A.pts_to a0 #p0 s0
  requires pure (SZ.v t < 151936 /\ SZ.v c < 1024)
  returns v : bf16
  ensures pure (S.length s0 == 155582464 /\ SZ.v t * 1024 + SZ.v c < 155582464 /\
                v == S.index s0 (SZ.v t * 1024 + SZ.v c))
{
  A.pts_to_len a0;
  let idx = t *^ 1024sz +^ c;
  a0.(idx)
}

noextract
let res_val (s0 : S.seq bf16) (s1 : S.seq I64.t) (j : nat) : bf16 =
  let t = SS.tok_at s1 (j / 1024) in
  if 0 <= t && t < 151936 && t * 1024 + j % 1024 < S.length s0
  then S.index s0 (t * 1024 + j % 1024)
  else zero

inline_for_extraction noextract
fn residual_value
  (a0 : A.larray bf16 155582464)
  (a1 : A.larray I64.t 7)
  (#p0 #p1 : perm)
  (#s0 : erased (S.seq bf16))
  (#s1 : erased (S.seq I64.t))
  (j : sz)
  preserves A.pts_to a0 #p0 s0 ** A.pts_to a1 #p1 s1
  requires pure (SZ.v j < 8192)
  returns v : bf16
  ensures pure (v == res_val s0 s1 (SZ.v j))
{
  A.pts_to_len a0; A.pts_to_len a1;
  let row = j /^ 1024sz;
  let col = j %^ 1024sz;
  let mut tok : I64.t = 0L;
  if (row <^ 7sz) { tok := a1.(row) };
  let tv = !tok;
  assert pure (I64.v tv == SS.tok_at s1 (SZ.v j / 1024));
  if (I64.lte 0L tv && I64.lt tv 151936L) {
    let ti = i64_to_sz tv;
    read_emb a0 ti col
  } else {
    zero #bf16
  }
}

inline_for_extraction noextract
fn prepare_residual
  (a0 : A.larray bf16 155582464)
  (a1 : A.larray I64.t 7)
  (o0 : A.larray bf16 8192)
  (#p0 #p1 : perm)
  (#s0 : erased (S.seq bf16))
  (#s1 : erased (S.seq I64.t))
  (#b0 : erased (S.seq bf16))
  preserves gpu ** A.pts_to a0 #p0 s0 ** A.pts_to a1 #p1 s1
  requires A.pts_to o0 b0
  ensures exists* r. A.pts_to o0 r **
    pure (S.length r == 8192 /\ (forall (k:nat). k < 8192 ==> S.index r k == res_val s0 s1 k))
{
  A.pts_to_len o0;
  let mut i = 0sz;
  while (!i <^ 8192sz)
    invariant exists* (j : sz) r. i |-> j ** A.pts_to o0 r **
      pure (SZ.v j <= 8192 /\ S.length r == 8192 /\
            (forall (k:nat). k < SZ.v j ==> S.index r k == res_val s0 s1 k))
    decreases (8192 - SZ.v !i)
  {
    let j = !i;
    let v = residual_value a0 a1 j;
    A.pts_to_len o0;
    o0.(j) <- v;
    i := j +^ 1sz;
  }
}
(* ---------------- rotary tables ---------------- *)

inline_for_extraction noextract
let rope_angle_f (row col : I64.t) : f32 =
  mul #f32 (of_int #f32 row)
    (div #f32 (one #f32) (FB.pow #f32 (of_int #f32 1000000L) (div #f32 (of_int #f32 col) (of_int #f32 64L))))

noextract
let rope_angle_approx (row col : I64.t)
  : Lemma (rope_angle_f row col %~ SS.rope_angle (I64.v row) (I64.v col))
  = let p = of_int #f32 1000000L in
    let c = of_int #f32 col in
    let w = of_int #f32 64L in
    let e = div #f32 c w in
    let pw = FB.pow #f32 p e in
    let inv = div #f32 (one #f32) pw in
    let er = Real.of_int (I64.v col) /. 64.0R in
    let pr = Pow.pow 1000000.0R er in
    approx_of_int #f32;
    approx_apply (of_int #f32) Kuiper.Float.Realops.r_of_int 1000000L 1000000L;
    approx_apply (of_int #f32) Kuiper.Float.Realops.r_of_int col col;
    approx_apply (of_int #f32) Kuiper.Float.Realops.r_of_int 64L 64L;
    approx_apply (of_int #f32) Kuiper.Float.Realops.r_of_int row row;
    approx_div #f32;
    assert (e %~ er);
    F32.pow_approx p e 1000000.0R er;
    assert (inv %~ (1.0R /. pr));
    a_mul (of_int #f32 row) inv (Real.of_int (I64.v row)) (1.0R /. pr)

inline_for_extraction noextract
let rope_n : (n : sz{SZ.v n == 524288}) = SZ.uint32_to_sizet 524288ul

noextract
let rope_cos_real (k : int) : real = Trig.cos (SS.rope_angle (k / 128) ((k % 128) % 64))
noextract
let rope_sin_real (k : int) : real = Trig.sin (SS.rope_angle (k / 128) ((k % 128) % 64))

noextract
let rope_elem (j : nat) (row col : I64.t)
  : Lemma (requires I64.v row == j / 128 /\ I64.v col == (j % 128) % 64)
          (ensures Casts.fcast #f32 #bf16 (FB.cos #f32 (rope_angle_f row col)) %~ rope_cos_real j /\
                   Casts.fcast #f32 #bf16 (FB.sin #f32 (rope_angle_f row col)) %~ rope_sin_real j)
  = let angle = rope_angle_f row col in
    rope_angle_approx row col;
    let ar = SS.rope_angle (j / 128) ((j % 128) % 64) in
    F32.cos_approx angle ar;
    F32.sin_approx angle ar;
    Casts.fcast_approx #f32 #bf16 (FB.cos #f32 angle) (Trig.cos ar);
    Casts.fcast_approx #f32 #bf16 (FB.sin #f32 angle) (Trig.sin ar)

inline_for_extraction noextract
fn prepare_rope
  (o2 : A.larray bf16 524288)
  (o3 : A.larray bf16 524288)
  (#b2 #b3 : erased (S.seq bf16))
  preserves gpu
  requires A.pts_to o2 b2 ** A.pts_to o3 b3
  ensures exists* r2 r3. A.pts_to o2 r2 ** A.pts_to o3 r3 **
    pure (S.length r2 == 524288 /\ S.length r3 == 524288 /\
          (forall (k:nat). k < 524288 ==> S.index r2 k %~ rope_cos_real k /\ S.index r3 k %~ rope_sin_real k))
{
  A.pts_to_len o2; A.pts_to_len o3;
  let mut i = 0sz;
  while (!i <^ rope_n)
    invariant exists* (j : sz) r2 r3. i |-> j ** A.pts_to o2 r2 ** A.pts_to o3 r3 **
      pure (SZ.v j <= 524288 /\ S.length r2 == 524288 /\ S.length r3 == 524288 /\
            (forall (k:nat). k < SZ.v j ==> S.index r2 k %~ rope_cos_real k /\ S.index r3 k %~ rope_sin_real k))
    decreases (524288 - SZ.v !i)
  {
    let j = !i;
    let row = sz_to_i64 (j /^ 128sz);
    let col = sz_to_i64 ((j %^ 128sz) %^ 64sz);
    let angle = rope_angle_f row col;
    rope_elem (SZ.v j) row col;
    let cv : bf16 = Casts.fcast #f32 #bf16 (FB.cos #f32 angle);
    let sv : bf16 = Casts.fcast #f32 #bf16 (FB.sin #f32 angle);
    A.pts_to_len o2; A.pts_to_len o3;
    o2.(j) <- cv;
    o3.(j) <- sv;
    i := j +^ 1sz;
  }
}

(* ---------------- real sums ---------------- *)

noextract
let sq (x : S.seq real) (i : int) : real = SS.at x i *. SS.at x i

(* fma-order partial sum: sum_{j<m} x[b + j*c]^2, newest term first *)
noextract
let rec psq (x : S.seq real) (b : int) (c : nat) (m : nat) : Tot real (decreases m) =
  if m = 0 then 0.0R else sq x (b + (m - 1) * c) +. psq x b c (m - 1)

noextract
let rec colsum (x : S.seq real) (o : int) (c m : nat) (n : nat) : Tot real (decreases n) =
  if n = 0 then 0.0R else colsum x o c m (n - 1) +. psq x (o + (n - 1)) c m

noextract
let rec sumsq_split (n k : nat) (x : S.seq real) (o : int)
  : Lemma (ensures SS.sumsq (n + k) x o == SS.sumsq n x o +. SS.sumsq k x (o + n)) (decreases k)
  = if k = 0 then () else sumsq_split n (k - 1) x o

noextract
let rec colsum_zero (x : S.seq real) (o : int) (c : nat) (n : nat)
  : Lemma (ensures colsum x o c 0 n == 0.0R) (decreases n)
  = if n = 0 then () else colsum_zero x o c (n - 1)

noextract
let rec colsum_succ (x : S.seq real) (o : int) (c m : nat) (n : nat)
  : Lemma (ensures colsum x o c (m + 1) n == colsum x o c m n +. SS.sumsq n x (o + c * m)) (decreases n)
  = if n = 0 then () else begin
      colsum_succ x o c m (n - 1);
      assert (o + (n - 1) + m * c == o + c * m + (n - 1))
    end

noextract
let rec colsum_total (x : S.seq real) (o : int) (c m : nat)
  : Lemma (ensures colsum x o c m c == SS.sumsq (c * m) x o) (decreases m)
  = if m = 0 then colsum_zero x o c c
    else begin
      colsum_total x o c (m - 1);
      colsum_succ x o c (m - 1) c;
      sumsq_split (c * (m - 1)) c x o
    end

noextract
let rec ssum (s : S.seq real) (lo : int) (n : nat) : Tot real (decreases n) =
  if n = 0 then 0.0R else ssum s lo (n - 1) +. SS.at s (lo + (n - 1))

noextract
let rec ssum_split (s : S.seq real) (lo : int) (n k : nat)
  : Lemma (ensures ssum s lo (n + k) == ssum s lo n +. ssum s (lo + n) k) (decreases k)
  = if k = 0 then () else ssum_split s lo n (k - 1)

noextract
let rec ssum_ext (s t : S.seq real) (lo lt : int) (n : nat)
  : Lemma (requires forall (k:nat). k < n ==> SS.at s (lo + k) == SS.at t (lt + k))
          (ensures ssum s lo n == ssum t lt n) (decreases n)
  = if n = 0 then () else ssum_ext s t lo lt (n - 1)

(* ---------------- generic in-place tree reduction ---------------- *)

noextract
let step (#a : Type) (op : a -> a -> a) (s : S.seq a) (b d : nat) : S.seq a =
  if b + 2 * d <= S.length s then
    S.init (S.length s) (fun (k : nat{k < S.length s}) ->
      if b <= k && k < b + d then op (S.index s k) (S.index s (k + d)) else S.index s k)
  else s

noextract
let rec levels (#a : Type) (op : a -> a -> a) (s : S.seq a) (b : nat) (d : nat)
  : Tot (S.seq a) (decreases d)
  = if d = 0 then s else levels op (step op s b d) b (d / 2)

noextract
let warp_fold (#a : Type) (op : a -> a -> a) (s : S.seq a) (w : nat) : S.seq a =
  let s' = levels op s (32 * w) 16 in
  if 32 * w < S.length s' && w < S.length s' then S.upd s' w (S.index s' (32 * w)) else s'

noextract
let rec warps (#a : Type) (op : a -> a -> a) (s : S.seq a) (w : nat) : Tot (S.seq a) (decreases w) =
  if w = 0 then s else warp_fold op (warps op s (w - 1)) (w - 1)

noextract
let tree (#a : Type) (op : a -> a -> a) (s : S.seq a) : S.seq a =
  levels op (warps op s 8) 0 4

noextract
let step_len (#a : Type) (op : a -> a -> a) (s : S.seq a) (b d : nat)
  : Lemma (S.length (step op s b d) == S.length s) [SMTPat (S.length (step op s b d))]
  = ()

noextract
let rec levels_len (#a : Type) (op : a -> a -> a) (s : S.seq a) (b d : nat)
  : Lemma (ensures S.length (levels op s b d) == S.length s) (decreases d)
          [SMTPat (S.length (levels op s b d))]
  = if d = 0 then () else levels_len op (step op s b d) b (d / 2)

noextract
let rec warps_len (#a : Type) (op : a -> a -> a) (s : S.seq a) (w : nat)
  : Lemma (ensures S.length (warps op s w) == S.length s) (decreases w)
          [SMTPat (S.length (warps op s w))]
  = if w = 0 then () else warps_len op s (w - 1)

noextract
let tree_len (#a : Type) (op : a -> a -> a) (s : S.seq a)
  : Lemma (S.length (tree op s) == S.length s) [SMTPat (S.length (tree op s))]
  = ()

(* frame: entries outside [b, b + 2d) are untouched *)
noextract
let rec levels_frame (#a : Type) (op : a -> a -> a) (s : S.seq a) (b d : nat) (k : nat)
  : Lemma (requires k < S.length s /\ (k < b \/ b + 2 * d <= k))
          (ensures S.index (levels op s b d) k == S.index s k) (decreases d)
  = if d = 0 then () else levels_frame op (step op s b d) b (d / 2) k

noextract
let radd (x y : real) : real = x +. y

noextract
let step_sum (g : S.seq real) (b d : nat) (n : nat)
  : Lemma (requires b + 2 * d <= S.length g /\ n <= d)
          (ensures ssum (step radd g b d) b n == ssum g b n +. ssum g (b + d) n)
  = let rec aux (n : nat{n <= d})
      : Lemma (ensures ssum (step radd g b d) b n == ssum g b n +. ssum g (b + d) n) (decreases n)
      = if n = 0 then () else aux (n - 1)
    in aux n

#push-options "--fuel 2"
noextract
let rec levels_sum (g : S.seq real) (b : nat) (e : nat)
  : Lemma (requires b + 2 * pow2 e <= S.length g)
          (ensures SS.at (levels radd g b (pow2 e)) b == ssum g b (2 * pow2 e)) (decreases e)
  = let d = pow2 e in
    let g' = step radd g b d in
    if e = 0 then begin
      assert (d == 1);
      assert (levels radd g b 1 == g');
      assert (SS.at g' b == SS.at g b +. SS.at g (b + 1));
      assert (ssum g b 1 == SS.at g b);
      assert (ssum g b 2 == SS.at g b +. SS.at g (b + 1))
    end else begin
      assert (pow2 e == 2 * pow2 (e - 1));
      assert (d / 2 == pow2 (e - 1));
      assert (levels radd g b d == levels radd g' b (pow2 (e - 1)));
      levels_sum g' b (e - 1);
      step_sum g b d d;
      ssum_split g b d d
    end
#pop-options

noextract
let pow2_facts () : Lemma (pow2 2 == 4 /\ pow2 4 == 16) =
  assert (pow2 0 == 1);
  assert (pow2 1 == 2);
  assert (pow2 2 == 4);
  assert (pow2 3 == 8);
  assert (pow2 4 == 16)

noextract
let warps_inv (g : S.seq real) (w : nat) : prop =
  (forall (k:nat). 32 * w <= k /\ k < S.length g ==> S.index (warps radd g w) k == S.index g k) /\
  (forall (w':nat). w' < w ==> SS.at (warps radd g w) w' == ssum g (32 * w') 32)

noextract
let rec warps_sum (g : S.seq real) (w : nat)
  : Lemma (requires S.length g == 256 /\ w <= 8) (ensures warps_inv g w) (decreases w)
  = if w = 0 then () else begin
      let v = w - 1 in
      warps_sum g v;
      let s = warps radd g v in
      let s' = levels radd s (32 * v) 16 in
      pow2_facts ();
      levels_sum s (32 * v) 4;
      ssum_ext s g (32 * v) (32 * v) 32;
      assert (SS.at s' (32 * v) == ssum g (32 * v) 32);
      let s'' = warps radd g w in
      assert (s'' == S.upd s' v (S.index s' (32 * v)));
      introduce forall (k:nat). 32 * w <= k /\ k < S.length g ==> S.index s'' k == S.index g k
      with introduce _ ==> _ with levels_frame radd s (32 * v) 16 k;
      introduce forall (w':nat). w' < w ==> SS.at s'' w' == ssum g (32 * w') 32
      with introduce _ ==> _ with (if w' < v then levels_frame radd s (32 * v) 16 w' else ())
    end

noextract
let rec blocks_sum (s g : S.seq real) (c : nat) (m : nat)
  : Lemma (requires forall (w:nat). w < m ==> SS.at s w == ssum g (c * w) c)
          (ensures ssum s 0 m == ssum g 0 (c * m)) (decreases m)
  = if m = 0 then () else begin
      blocks_sum s g c (m - 1);
      ssum_split g 0 (c * (m - 1)) c
    end

(* reals of phase 1: partial[t] = sum_{j<4} x[o + t + 256 j]^2 *)
noextract
let p1_real (x : S.seq real) (o : int) : S.seq real =
  S.init 256 (fun (t : nat{t < 256}) -> psq x (o + t) 256 4)

noextract
let rec p1_colsum (x : S.seq real) (o : int) (n : nat)
  : Lemma (requires n <= 256) (ensures ssum (p1_real x o) 0 n == colsum x o 256 4 n) (decreases n)
  = if n = 0 then () else p1_colsum x o (n - 1)

noextract
let tree_real_total (x : S.seq real) (o : int)
  : Lemma (SS.at (tree radd (p1_real x o)) 0 == SS.sumsq 1024 x o)
  = let g = p1_real x o in
    warps_sum g 8;
    let s = warps radd g 8 in
    pow2_facts ();
    levels_sum s 0 2;
    blocks_sum s g 32 8;
    p1_colsum x o 256;
    colsum_total x o 256 4

(* ---------------- approximation lifting ---------------- *)

noextract
let fadd (x y : f32) : f32 = add #f32 x y

noextract
let step_approx (s : S.seq f32) (g : S.seq real) (b d : nat)
  : Lemma (requires s %~ g) (ensures step fadd s b d %~ step radd g b d)
  = if b + 2 * d <= S.length s then begin
      let s' = step fadd s b d in
      let g' = step radd g b d in
      introduce forall (k:nat). k < S.length s' ==> S.index s' k %~ S.index g' k
      with introduce _ ==> _ with
        (if b <= k && k < b + d then
           a_add (S.index s k) (S.index s (k + d)) (S.index g k) (S.index g (k + d))
         else ())
    end

noextract
let rec levels_approx (s : S.seq f32) (g : S.seq real) (b d : nat)
  : Lemma (requires s %~ g) (ensures levels fadd s b d %~ levels radd g b d) (decreases d)
  = if d = 0 then () else begin
      step_approx s g b d;
      levels_approx (step fadd s b d) (step radd g b d) b (d / 2)
    end

noextract
let rec warps_approx (s : S.seq f32) (g : S.seq real) (w : nat)
  : Lemma (requires s %~ g) (ensures warps fadd s w %~ warps radd g w) (decreases w)
  = if w = 0 then () else begin
      warps_approx s g (w - 1);
      levels_approx (warps fadd s (w - 1)) (warps radd g (w - 1)) (32 * (w - 1)) 16
    end

noextract
let tree_approx (s : S.seq f32) (g : S.seq real)
  : Lemma (requires s %~ g) (ensures tree fadd s %~ tree radd g)
  = warps_approx s g 8;
    levels_approx (warps fadd s 8) (warps radd g 8) 0 4

(* ---------------- normalize: scalar pieces ---------------- *)

module FM = Kuiper.Float32.FastMath
module Sqrt = FStar.Math.Sqrt

noextract
let fma_step (si : S.seq bf16) (idx : nat) (b : int) (j : nat) (sv nv : f32)
  : Lemma
      (requires idx < S.length si /\ idx == b + j * 256 /\
        (forall (xr : S.seq real). si %~ xr ==> sv %~ psq xr b 256 j) /\
        (forall (xr yr zr : real).
          v_approximates (Casts.fcast #bf16 #f32 (S.index si idx)) xr /\
          v_approximates (Casts.fcast #bf16 #f32 (S.index si idx)) yr /\
          v_approximates sv zr ==>
          v_approximates nv (Kuiper.Float.Realops.r_fma xr yr zr)))
      (ensures forall (xr : S.seq real). si %~ xr ==> nv %~ psq xr b 256 (j + 1))
  = introduce forall (xr : S.seq real). si %~ xr ==> nv %~ psq xr b 256 (j + 1)
    with introduce _ ==> _ with begin
      let xi = S.index xr idx in
      Casts.fcast_approx #bf16 #f32 (S.index si idx) xi;
      assert (SS.at xr idx == xi);
      assert (psq xr b 256 (j + 1) == xi *. xi +. psq xr b 256 j)
    end

noextract
let p1_step (si : S.seq bf16) (cur : S.seq f32) (b : int) (t : nat) (s : f32)
  : Lemma
      (requires t < S.length cur /\
        (forall (xr : S.seq real). si %~ xr ==>
          (forall (k:nat). k < t ==> S.index cur k %~ psq xr (b + k) 256 4)) /\
        (forall (xr : S.seq real). si %~ xr ==> s %~ psq xr (b + t) 256 4))
      (ensures
        (forall (xr : S.seq real). si %~ xr ==>
          (forall (k:nat). k < t + 1 ==> S.index (S.upd cur t s) k %~ psq xr (b + k) 256 4)))
  = ()

noextract
let p1_final (si : S.seq bf16) (cur : S.seq f32) (b : int)
  : Lemma
      (requires S.length cur == 256 /\
        (forall (xr : S.seq real). si %~ xr ==>
          (forall (k:nat). k < 256 ==> S.index cur k %~ psq xr (b + k) 256 4)))
      (ensures forall (xr : S.seq real). si %~ xr ==> cur %~ p1_real xr b)
  = ()

noextract
let total_lemma (si : S.seq bf16) (p1 : S.seq f32) (o : int) (tsum : f32)
  : Lemma
      (requires S.length p1 == 256 /\ tsum == S.index (tree fadd p1) 0 /\
        (forall (xr : S.seq real). si %~ xr ==> p1 %~ p1_real xr o))
      (ensures forall (xr : S.seq real). si %~ xr ==> tsum %~ SS.sumsq 1024 xr o)
  = introduce forall (xr : S.seq real). si %~ xr ==> tsum %~ SS.sumsq 1024 xr o
    with introduce _ ==> _ with begin
      tree_approx p1 (p1_real xr o);
      tree_real_total xr o
    end

inline_for_extraction noextract
let inv_f (tsum ov : f32) : f32 =
  FB.rsqrt #f32 (add #f32 (div #f32 tsum (of_int #f32 1024L))
                          (div #f32 ov (of_int #f32 1000000L)))

noextract
let inv_lemma (tsum ov : f32) (ss : real)
  : Lemma (requires tsum %~ ss /\ ss >=. 0.0R /\ ov == one #f32)
          (ensures inv_f tsum ov %~ Sqrt.rsqrt (ss /. Real.of_int 1024 +. SS.rms_eps))
  = approx_of_int #f32;
    approx_apply (of_int #f32) Kuiper.Float.Realops.r_of_int 1024L 1024L;
    approx_apply (of_int #f32) Kuiper.Float.Realops.r_of_int 1000000L 1000000L;
    approx_div #f32;
    let q = div #f32 tsum (of_int #f32 1024L) in
    let e = div #f32 (one #f32) (of_int #f32 1000000L) in
    assert (q %~ (ss /. 1024.0R));
    assert (e %~ (1.0R /. 1000000.0R));
    a_add q e (ss /. 1024.0R) (1.0R /. 1000000.0R);
    approx_rsqrt #f32;
    assert (ss /. 1024.0R +. 1.0R /. 1000000.0R >. 0.0R)

noextract
let inv_q (si : S.seq bf16) (tsum ov : f32) (o : int)
  : Lemma (requires ov == one #f32 /\ forall (xr : S.seq real). si %~ xr ==> tsum %~ SS.sumsq 1024 xr o)
          (ensures forall (xr : S.seq real). si %~ xr ==> inv_f tsum ov %~ SS.rms_inv 1024 xr o)
  = introduce forall (xr : S.seq real). si %~ xr ==> inv_f tsum ov %~ SS.rms_inv 1024 xr o
    with introduce _ ==> _ with inv_lemma tsum ov (SS.sumsq 1024 xr o)

inline_for_extraction noextract
let out_f (x w : bf16) (inv : f32) : bf16 =
  Casts.fcast #f32 #bf16
    (mul #f32 (Casts.fcast #bf16 #f32 x) (mul #f32 inv (Casts.fcast #bf16 #f32 w)))

noextract
let out_lemma (x w : bf16) (inv : f32) (xr wr ir : real)
  : Lemma (requires x %~ xr /\ w %~ wr /\ inv %~ ir)
          (ensures out_f x w inv %~ (wr *. (xr *. ir)))
  = let xf = Casts.fcast #bf16 #f32 x in
    let wf = Casts.fcast #bf16 #f32 w in
    Casts.fcast_approx #bf16 #f32 x xr;
    Casts.fcast_approx #bf16 #f32 w wr;
    a_mul inv wf ir wr;
    a_mul xf (mul #f32 inv wf) xr (ir *. wr);
    assert (xr *. (ir *. wr) == wr *. (xr *. ir));
    Casts.fcast_approx #f32 #bf16 (mul #f32 xf (mul #f32 inv wf)) (wr *. (xr *. ir))

noextract
let norm_real (xr wr : S.seq real) (k : int) : real =
  SS.at wr (k % 1024) *. (SS.at xr k *. SS.rms_inv 1024 xr ((k / 1024) * 1024))

noextract
let norm_val_ok (si sw : S.seq bf16) (v : bf16) (k : int) : prop =
  forall (xr wr : S.seq real). si %~ xr /\ sw %~ wr ==> v %~ norm_real xr wr k

noextract
let elem_lemma (si sw : S.seq bf16) (r c : nat) (inv : f32)
  : Lemma
      (requires S.length si == 8192 /\ S.length sw == 1024 /\ r < 8 /\ c < 1024 /\
        (forall (xr : S.seq real). si %~ xr ==> inv %~ SS.rms_inv 1024 xr (r * 1024)))
      (ensures norm_val_ok si sw (out_f (S.index si (r * 1024 + c)) (S.index sw c) inv) (r * 1024 + c))
  = let k = r * 1024 + c in
    FStar.Math.Lemmas.lemma_div_plus c r 1024;
    FStar.Math.Lemmas.small_div c 1024;
    FStar.Math.Lemmas.lemma_mod_plus c r 1024;
    FStar.Math.Lemmas.small_mod c 1024;
    assert (k / 1024 == r /\ k % 1024 == c);
    introduce forall (xr wr : S.seq real). si %~ xr /\ sw %~ wr ==>
      out_f (S.index si k) (S.index sw c) inv %~ norm_real xr wr k
    with introduce _ ==> _ with
      out_lemma (S.index si k) (S.index sw c) inv (S.index xr k) (S.index wr c) (SS.rms_inv 1024 xr (r * 1024))

noextract
let zero_f32_approx () : Lemma (zero #f32 %~ 0.0R) = a0 #f32

(* ---------------- normalize: Pulse pieces ---------------- *)

inline_for_extraction noextract
fn step_loop (p : A.array f32) (b d : sz) (#s : erased (S.seq f32))
  requires A.pts_to p s ** pure (S.length s == 256 /\ SZ.v b + 2 * SZ.v d <= 256)
  ensures A.pts_to p (step fadd s (SZ.v b) (SZ.v d))
{
  A.pts_to_len p;
  let mut t = 0sz;
  while (!t <^ d)
    invariant exists* (j : sz) r. t |-> j ** A.pts_to p r **
      pure (SZ.v j <= SZ.v d /\ S.length r == 256 /\
        (forall (k:nat). k < 256 ==>
          S.index r k == (if SZ.v b <= k && k < SZ.v b + SZ.v j
                          then fadd (S.index s k) (S.index s (k + SZ.v d))
                          else S.index s k)))
    decreases (SZ.v d - SZ.v !t)
  {
    let j = !t;
    A.pts_to_len p;
    let x = p.(b +^ j);
    let y = p.(b +^ j +^ d);
    p.(b +^ j) <- add #f32 x y;
    t := j +^ 1sz;
  };
  with r. assert A.pts_to p r;
  assert pure (S.equal r (step fadd s (SZ.v b) (SZ.v d)));
  rewrite A.pts_to p r as A.pts_to p (step fadd s (SZ.v b) (SZ.v d));
}

inline_for_extraction noextract
fn levels_loop (p : A.array f32) (b d0 : sz) (#s : erased (S.seq f32))
  requires A.pts_to p s ** pure (S.length s == 256 /\ SZ.v b + 2 * SZ.v d0 <= 256)
  ensures A.pts_to p (levels fadd s (SZ.v b) (SZ.v d0))
{
  let mut d = d0;
  while (!d >^ 0sz)
    invariant exists* (dv : sz) r. d |-> dv ** A.pts_to p r **
      pure (S.length r == 256 /\ SZ.v dv <= SZ.v d0 /\
            levels fadd r (SZ.v b) (SZ.v dv) == levels fadd s (SZ.v b) (SZ.v d0))
    decreases (SZ.v !d)
  {
    let dv = !d;
    step_loop p b dv;
    d := dv /^ 2sz;
  };
  with r. assert A.pts_to p r;
  rewrite A.pts_to p r as A.pts_to p (levels fadd s (SZ.v b) (SZ.v d0));
}

inline_for_extraction noextract
fn phase1
  (inp : A.larray bf16 8192) (p : A.array f32) (r : sz)
  (#pi : perm) (#si : erased (S.seq bf16)) (#s0 : erased (S.seq f32))
  preserves gpu ** A.pts_to inp #pi si
  requires A.pts_to p s0 ** pure (S.length s0 == 256 /\ SZ.v r < 8)
  ensures exists* p1. A.pts_to p p1 ** pure (S.length p1 == 256 /\
    (forall (xr : S.seq real). si %~ xr ==> p1 %~ p1_real xr (SZ.v r * 1024)))
{
  A.pts_to_len inp;
  zero_f32_approx ();
  let mut t = 0sz;
  while (!t <^ 256sz)
    invariant exists* (tv : sz) cur. t |-> tv ** A.pts_to p cur **
      pure (SZ.v tv <= 256 /\ S.length cur == 256 /\
        (forall (xr : S.seq real). si %~ xr ==>
          (forall (k:nat). k < SZ.v tv ==> S.index cur k %~ psq xr (SZ.v r * 1024 + k) 256 4)))
    decreases (256 - SZ.v !t)
  {
    let tv = !t;
    let mut sum : f32 = zero #f32 #_;
    let mut j = 0sz;
    while (!j <^ 4sz)
      invariant exists* (jv : sz) sv. j |-> jv ** sum |-> sv **
        pure (SZ.v jv <= 4 /\
          (forall (xr : S.seq real). si %~ xr ==> sv %~ psq xr (SZ.v r * 1024 + SZ.v tv) 256 (SZ.v jv)))
      decreases (4 - SZ.v !j)
    {
      let jv = !j;
      let idx = r *^ 1024sz +^ tv +^ jv *^ 256sz;
      let x = inp.(idx);
      let xf = Casts.fcast #bf16 #f32 x;
      let sv = !sum;
      let nv = FM.fma_rn xf xf sv;
      fma_step si (SZ.v idx) (SZ.v r * 1024 + SZ.v tv) (SZ.v jv) sv nv;
      sum := nv;
      j := jv +^ 1sz;
    };
    let s = !sum;
    A.pts_to_len p;
    with cur. assert A.pts_to p cur;
    p1_step si cur (SZ.v r * 1024) (SZ.v tv) s;
    p.(tv) <- s;
    t := tv +^ 1sz;
  };
  with cur. assert A.pts_to p cur;
  p1_final si cur (SZ.v r * 1024);
}

inline_for_extraction noextract
fn warp_loop (p : A.array f32) (#p1 : erased (S.seq f32))
  requires A.pts_to p p1 ** pure (S.length p1 == 256)
  ensures A.pts_to p (tree fadd p1)
{
  let mut w = 0sz;
  while (!w <^ 8sz)
    invariant exists* (wv : sz) cur. w |-> wv ** A.pts_to p cur **
      pure (SZ.v wv <= 8 /\ cur == warps fadd p1 (SZ.v wv))
    decreases (8 - SZ.v !w)
  {
    let wv = !w;
    levels_loop p (32sz *^ wv) 16sz;
    A.pts_to_len p;
    let x = p.(32sz *^ wv);
    p.(wv) <- x;
    w := wv +^ 1sz;
  };
  levels_loop p 0sz 4sz;
  with r. assert A.pts_to p r;
  rewrite A.pts_to p r as A.pts_to p (tree fadd p1);
}

inline_for_extraction noextract
fn output_row
  (a2 : A.larray bf16 1024) (inp : A.larray bf16 8192) (o1 : A.larray bf16 8192)
  (rv : sz) (inv : f32)
  (#p2 #pi : perm) (#s2 #si : erased (S.seq bf16)) (#ro : erased (S.seq bf16))
  preserves A.pts_to a2 #p2 s2 ** A.pts_to inp #pi si
  requires A.pts_to o1 ro ** pure (SZ.v rv < 8 /\ S.length ro == 8192 /\
    (forall (k:nat). k < SZ.v rv * 1024 /\ k < 8192 ==> norm_val_ok si s2 (S.index ro k) k) /\
    (forall (xr : S.seq real). si %~ xr ==> inv %~ SS.rms_inv 1024 xr (SZ.v rv * 1024)))
  ensures exists* ro'. A.pts_to o1 ro' ** pure (S.length ro' == 8192 /\
    (forall (k:nat). k < (SZ.v rv + 1) * 1024 /\ k < 8192 ==> norm_val_ok si s2 (S.index ro' k) k))
{
  A.pts_to_len a2; A.pts_to_len inp;
  let mut c = 0sz;
  while (!c <^ 1024sz)
    invariant exists* (cv : sz) rc. c |-> cv ** A.pts_to o1 rc **
      pure (SZ.v cv <= 1024 /\ S.length rc == 8192 /\
        (forall (k:nat). k < SZ.v rv * 1024 + SZ.v cv /\ k < 8192 ==> norm_val_ok si s2 (S.index rc k) k))
    decreases (1024 - SZ.v !c)
  {
    let cv = !c;
    let idx = rv *^ 1024sz +^ cv;
    let x = inp.(idx);
    let w = a2.(cv);
    elem_lemma si s2 (SZ.v rv) (SZ.v cv) inv;
    A.pts_to_len o1;
    o1.(idx) <- out_f x w inv;
    c := cv +^ 1sz;
  };
}

inline_for_extraction noextract
fn normalize
  (a2 : A.larray bf16 1024) (inp : A.larray bf16 8192) (o1 : A.larray bf16 8192)
  (#p2 #pi : perm) (#s2 #si : erased (S.seq bf16)) (#b1 : erased (S.seq bf16))
  preserves gpu ** A.pts_to a2 #p2 s2 ** A.pts_to inp #pi si
  requires A.pts_to o1 b1
  ensures exists* r. A.pts_to o1 r ** pure (S.length r == 8192 /\
    (forall (k:nat). k < 8192 ==> norm_val_ok si s2 (S.index r k) k))
{
  A.pts_to_len o1;
  let mut p = [| zero #f32 #_; 256sz |];
  let mut onec : f32 = one #f32 #_;
  A.pts_to_len p;
  let mut r = 0sz;
  while (!r <^ 8sz)
    invariant exists* (rv : sz) ro ps. r |-> rv ** A.pts_to o1 ro ** A.pts_to p ps **
      pure (SZ.v rv <= 8 /\ S.length ro == 8192 /\ S.length ps == 256 /\
        (forall (k:nat). k < SZ.v rv * 1024 /\ k < 8192 ==> norm_val_ok si s2 (S.index ro k) k))
    decreases (8 - SZ.v !r)
  {
    let rv = !r;
    phase1 inp p rv;
    with p1. assert A.pts_to p p1;
    warp_loop p;
    let tsum = p.(0sz);
    total_lemma si p1 (SZ.v rv * 1024) tsum;
    let ov = !onec;
    inv_q si tsum ov (SZ.v rv * 1024);
    let inv = inv_f tsum ov;
    output_row a2 inp o1 rv inv;
    r := rv +^ 1sz;
  };
}

(* ---------------- composition into the family relation ---------------- *)

noextract
let res_ok (s0 : S.seq bf16) (s1 : S.seq I64.t) (o0 : S.seq bf16) : prop =
  S.length o0 == 8192 /\ (forall (k:nat). k < 8192 ==> S.index o0 k == res_val s0 s1 k)

noextract
let norm_ok (o0 s2 o1 : S.seq bf16) : prop =
  S.length o1 == 8192 /\ (forall (k:nat). k < 8192 ==> norm_val_ok o0 s2 (S.index o1 k) k)

noextract
let rope_ok (o2 o3 : S.seq bf16) : prop =
  S.length o2 == 524288 /\ S.length o3 == 524288 /\
  (forall (k:nat). k < 524288 ==> S.index o2 k %~ rope_cos_real k /\ S.index o3 k %~ rope_sin_real k)

noextract
let dom_ok (s0 : S.seq bf16) (s1 : S.seq I64.t) (s2 : S.seq bf16) : prop =
  S.length s0 == 155582464 /\ S.length s1 == 7 /\ SS.tokens_ok s1 /\ S.length s2 == 1024

noextract
let dm (t c : nat)
  : Lemma (requires c < 1024) (ensures (t * 1024 + c) / 1024 == t /\ (t * 1024 + c) % 1024 == c)
  = FStar.Math.Lemmas.lemma_div_plus c t 1024;
    FStar.Math.Lemmas.small_div c 1024;
    FStar.Math.Lemmas.lemma_mod_plus c t 1024;
    FStar.Math.Lemmas.small_mod c 1024

noextract
let tok_range (s1 : S.seq I64.t) (t : int)
  : Lemma (requires S.length s1 == 7 /\ SS.tokens_ok s1)
          (ensures 0 <= SS.tok_at s1 t /\ SS.tok_at s1 t < 151936)
  = if 0 <= t && t < 7 then assert (0 <= I64.v (S.index s1 t)) else ()

noextract
let res_at (s0 : S.seq bf16) (s1 : S.seq I64.t) (o0 : S.seq bf16) (t c : nat)
  : Lemma (requires S.length s0 == 155582464 /\ S.length s1 == 7 /\ SS.tokens_ok s1 /\
                    res_ok s0 s1 o0 /\ t < 8 /\ c < 1024)
          (ensures SS.tok_at s1 t * 1024 + c < 155582464 /\ 0 <= SS.tok_at s1 t /\
                   S.index o0 (t * 1024 + c) == S.index s0 (SS.tok_at s1 t * 1024 + c))
  = dm t c; tok_range s1 t

noextract
let window_row (s0 : S.seq bf16) (s1 : S.seq I64.t) (o0 : S.seq bf16) (t : nat)
  : Lemma (requires S.length s0 == 155582464 /\ S.length s1 == 7 /\ SS.tokens_ok s1 /\
                    res_ok s0 s1 o0 /\ t < 8)
          (ensures SS.window o0 (t * 1024) 1024 == SS.window s0 (SS.tok_at s1 t * 1024) 1024)
  = let tk = SS.tok_at s1 t in
    tok_range s1 t;
    let w1 = S.slice o0 (t * 1024) (t * 1024 + 1024) in
    let w2 = S.slice s0 (tk * 1024) (tk * 1024 + 1024) in
    introduce forall (c:nat{c < 1024}). S.index w1 c == S.index w2 c
    with res_at s0 s1 o0 t c;
    assert (S.equal w1 w2)

noextract
let embed_at (r0 : S.seq real) (s1 : S.seq I64.t) (rows : nat) (k : nat)
  : Lemma (requires k < rows * 1024)
          (ensures S.index (SS.embed_rows rows r0 s1) k == SS.at r0 (SS.tok_at s1 (k / 1024) * 1024 + k % 1024))
  = reveal_opaque (`%SS.embed_rows) (SS.embed_rows rows r0 s1)

noextract
let o0_approx (s0 : S.seq bf16) (s1 : S.seq I64.t) (o0 : S.seq bf16) (r0 : S.seq real)
  : Lemma (requires S.length s0 == 155582464 /\ S.length s1 == 7 /\ SS.tokens_ok s1 /\
                    res_ok s0 s1 o0 /\ s0 %~ r0)
          (ensures o0 %~ SS.embed_rows 8 r0 s1)
  = introduce forall (k:nat{k < 8192}). S.index o0 k %~ S.index (SS.embed_rows 8 r0 s1) k
    with begin
      let t = k / 1024 in
      let c = k % 1024 in
      assert (k == t * 1024 + c);
      res_at s0 s1 o0 t c;
      embed_at r0 s1 8 k
    end

noextract
let rec sumsq_cong (n : nat) (x y : S.seq real) (ox oy : int)
  : Lemma (requires forall (j:nat). j < n ==> SS.at x (ox + j) == SS.at y (oy + j))
          (ensures SS.sumsq n x ox == SS.sumsq n y oy) (decreases n)
  = if n = 0 then () else sumsq_cong (n - 1) x y ox oy

noextract
let rmsnorm_at (rows : nat) (x w : S.seq real) (k : nat)
  : Lemma (requires k < rows * 1024)
          (ensures S.index (SS.rmsnorm_rows rows 1024 x w) k ==
                   SS.at w (k % 1024) *. (SS.at x k *. SS.rms_inv 1024 x ((k / 1024) * 1024)))
  = reveal_opaque (`%SS.rmsnorm_rows) (SS.rmsnorm_rows rows 1024 x w)

noextract
let e78 (r0 : S.seq real) (s1 : S.seq I64.t) (j : int)
  : Lemma (requires j < 7168)
          (ensures SS.at (SS.embed_rows 7 r0 s1) j == SS.at (SS.embed_rows 8 r0 s1) j)
  = if j >= 0 then (embed_at r0 s1 7 j; embed_at r0 s1 8 j) else ()

noextract
let e8w (r0 : S.seq real) (s1 : S.seq I64.t) (c : nat)
  : Lemma (requires c < 1024 /\ S.length r0 == 155582464 /\ S.length s1 == 7)
          (ensures SS.at (SS.embed_rows 8 r0 s1) (7168 + c) == SS.at (SS.window r0 0 1024) c)
  = dm 7 c; embed_at r0 s1 8 (7168 + c)

noextract
let norm_rows7 (o0 s2 o1 : S.seq bf16) (r0 r2 : S.seq real) (s1 : S.seq I64.t)
  : Lemma (requires norm_ok o0 s2 o1 /\ o0 %~ SS.embed_rows 8 r0 s1 /\ s2 %~ r2)
          (ensures SS.window o1 0 7168 %~ SS.rmsnorm_rows 7 1024 (SS.embed_rows 7 r0 s1) r2)
  = let e7 = SS.embed_rows 7 r0 s1 in
    let e8 = SS.embed_rows 8 r0 s1 in
    introduce forall (k:nat{k < 7168}).
      S.index (SS.window o1 0 7168) k %~ S.index (SS.rmsnorm_rows 7 1024 e7 r2) k
    with begin
      let o = (k / 1024) * 1024 in
      introduce forall (j:nat). j < 1024 ==> SS.at e7 (o + j) == SS.at e8 (o + j)
      with introduce _ ==> _ with e78 r0 s1 (o + j);
      sumsq_cong 1024 e7 e8 o o;
      e78 r0 s1 k;
      rmsnorm_at 7 e7 r2 k;
      assert (S.index o1 k %~ norm_real e8 r2 k)
    end

noextract
let norm_row_pad (o0 s2 o1 : S.seq bf16) (r0 r2 : S.seq real) (s1 : S.seq I64.t)
  : Lemma (requires norm_ok o0 s2 o1 /\ o0 %~ SS.embed_rows 8 r0 s1 /\ s2 %~ r2 /\
                    S.length r0 == 155582464 /\ S.length s1 == 7)
          (ensures SS.window o1 7168 1024 %~ SS.rmsnorm_rows 1 1024 (SS.window r0 0 1024) r2)
  = let e8 = SS.embed_rows 8 r0 s1 in
    let w = SS.window r0 0 1024 in
    introduce forall (j:nat). j < 1024 ==> SS.at e8 (7168 + j) == SS.at w (0 + j)
    with introduce _ ==> _ with e8w r0 s1 j;
    sumsq_cong 1024 e8 w 7168 0;
    introduce forall (c:nat{c < 1024}).
      S.index (SS.window o1 7168 1024) c %~ S.index (SS.rmsnorm_rows 1 1024 w r2) c
    with begin
      dm 7 c; dm 0 c;
      e8w r0 s1 c;
      rmsnorm_at 1 w r2 c;
      assert (S.index o1 (7168 + c) %~ norm_real e8 r2 (7168 + c))
    end

noextract
let embed_windows (s0 : S.seq bf16) (s1 : S.seq I64.t) (o0 : S.seq bf16) (r0 : S.seq real)
  : Lemma (requires S.length s0 == 155582464 /\ S.length s1 == 7 /\ SS.tokens_ok s1 /\
                    res_ok s0 s1 o0 /\ s0 %~ r0)
          (ensures SS.window o0 0 7168 %~ SS.embed_rows 7 r0 s1 /\
                   SS.window o0 7168 1024 %~ SS.window r0 0 1024)
  = o0_approx s0 s1 o0 r0;
    introduce forall (k:nat{k < 7168}). S.index (SS.window o0 0 7168) k %~ S.index (SS.embed_rows 7 r0 s1) k
    with (embed_at r0 s1 7 k; embed_at r0 s1 8 k);
    introduce forall (c:nat{c < 1024}). S.index (SS.window o0 7168 1024) c %~ S.index (SS.window r0 0 1024) c
    with e8w r0 s1 c

noextract
let rope_tables (o2 o3 : S.seq bf16)
  : Lemma (requires rope_ok o2 o3)
          (ensures o2 %~ SS.rope_cos_table 0 4096 /\ o3 %~ SS.rope_sin_table 0 4096)
  = reveal_opaque (`%SS.rope_cos_table) (SS.rope_cos_table 0 4096);
    reveal_opaque (`%SS.rope_sin_table) (SS.rope_sin_table 0 4096);
    assert (S.length (SS.rope_cos_table 0 4096) == 524288);
    introduce forall (k:nat{k < 524288}).
      S.index o2 k %~ S.index (SS.rope_cos_table 0 4096) k /\
      S.index o3 k %~ S.index (SS.rope_sin_table 0 4096) k
    with assert (0 + k / 128 == k / 128)

noextract
let compose (s0 : S.seq bf16) (s1 : S.seq I64.t) (s2 : S.seq bf16) (o0 o1 o2 o3 : S.seq bf16)
  : Lemma (requires res_ok s0 s1 o0 /\ norm_ok o0 s2 o1 /\ rope_ok o2 o3)
          (ensures SS.optimized_embedding_relation s0 s1 s2 o0 o1 o2 o3)
  = introduce dom_ok s0 s1 s2 ==> SS.optimized_embedding_relation s0 s1 s2 o0 o1 o2 o3
    with begin
      introduce forall (t:nat). t < 8 ==>
        SS.window o0 (t * 1024) 1024 == SS.window s0 (SS.tok_at s1 t * 1024) 1024
      with introduce _ ==> _ with window_row s0 s1 o0 t;
      rope_tables o2 o3;
      introduce forall (r0 r2 : S.seq real). s0 %~ r0 /\ s2 %~ r2 ==>
        SS.window o0 0 7168 %~ SS.embed_rows 7 r0 s1 /\
        SS.window o0 7168 1024 %~ SS.window r0 0 1024 /\
        SS.window o1 0 7168 %~ SS.rmsnorm_rows 7 1024 (SS.embed_rows 7 r0 s1) r2 /\
        SS.window o1 7168 1024 %~ SS.rmsnorm_rows 1 1024 (SS.window r0 0 1024) r2
      with introduce _ ==> _ with begin
        embed_windows s0 s1 o0 r0;
        o0_approx s0 s1 o0 r0;
        norm_rows7 o0 s2 o1 r0 r2 s1;
        norm_row_pad o0 s2 o1 r0 r2 s1
      end
    end

(* ---------------- kernels and host entry ---------------- *)

inline_for_extraction noextract
fn prepare_kernel
  (a0 : A.larray bf16 155582464) (a1 : A.larray I64.t 7) (o0 : A.larray bf16 8192)
  (o2 o3 : A.larray bf16 524288)
  (#p0 #p1 : perm) (#s0 : erased (S.seq bf16)) (#s1 : erased (S.seq I64.t))
  (#b0 #b2 #b3 : erased (S.seq bf16))
  preserves gpu ** A.pts_to a0 #p0 s0 ** A.pts_to a1 #p1 s1
  requires A.pts_to o0 b0 ** A.pts_to o2 b2 ** A.pts_to o3 b3
  ensures exists* r0 r2 r3. A.pts_to o0 r0 ** A.pts_to o2 r2 ** A.pts_to o3 r3 **
    pure (res_ok s0 s1 r0 /\ rope_ok r2 r3)
{
  prepare_residual a0 a1 o0;
  prepare_rope o2 o3;
}

inline_for_extraction noextract
fn normalize_kernel
  (a2 : A.larray bf16 1024) (inp : A.larray bf16 8192) (o1 : A.larray bf16 8192)
  (#p2 : perm) (#s2 #si : erased (S.seq bf16)) (#b1 : erased (S.seq bf16))
  preserves gpu ** A.pts_to a2 #p2 s2 ** A.pts_to inp si
  requires A.pts_to o1 b1
  ensures exists* r. A.pts_to o1 r ** pure (norm_ok si s2 r)
{
  normalize a2 inp o1;
}

instance send_global_array_contents
  (#a : Type0)
  (arr : A.array a{is_global arr})
  (#f : perm)
  (contents : S.seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn kplay_run
  (a0 : A.larray bf16 155582464{is_global a0})
  (a1 : A.larray I64.t 7{is_global a1})
  (a2 : A.larray bf16 1024{is_global a2})
  (o0 : A.larray bf16 8192{is_global o0})
  (o1 : A.larray bf16 8192{is_global o1})
  (o2 : A.larray bf16 524288{is_global o2})
  (o3 : A.larray bf16 524288{is_global o3})
  (#p0 #p1 #p2 : perm)
  (#s0 : erased (S.seq bf16))
  (#s1 : erased (S.seq I64.t))
  (#s2 : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (A.pts_to a0 #p0 s0) ** on gpu_loc (A.pts_to a1 #p1 s1) **
            on gpu_loc (A.pts_to a2 #p2 s2)
  requires exists* b0 b1 b2 b3.
    on gpu_loc (A.pts_to o0 b0) ** on gpu_loc (A.pts_to o1 b1) **
    on gpu_loc (A.pts_to o2 b2) ** on gpu_loc (A.pts_to o3 b3)
  ensures exists* r0 r1 r2 r3.
    on gpu_loc (A.pts_to o0 r0) ** on gpu_loc (A.pts_to o1 r1) **
    on gpu_loc (A.pts_to o2 r2) ** on gpu_loc (A.pts_to o3 r3) **
    pure (SS.optimized_embedding_relation s0 s1 s2 r0 r1 r2 r3)
{
  with b0 b1 b2 b3. assert (
    on gpu_loc (A.pts_to o0 b0) ** on gpu_loc (A.pts_to o1 b1) **
    on gpu_loc (A.pts_to o2 b2) ** on gpu_loc (A.pts_to o3 b3));
  launch_kernel_1 (fun () -> prepare_kernel a0 a1 o0 o2 o3 #p0 #p1 #s0 #s1 #b0 #b2 #b3);
  with r0 r2 r3. assert (
    on gpu_loc (A.pts_to o0 r0) ** on gpu_loc (A.pts_to o2 r2) ** on gpu_loc (A.pts_to o3 r3));
  launch_kernel_1 (fun () -> normalize_kernel a2 o0 o1 #p2 #s2 #r0 #b1);
  with r1. assert (on gpu_loc (A.pts_to o1 r1));
  assert pure (res_ok s0 s1 r0);
  assert pure (norm_ok r0 s2 r1);
  assert pure (rope_ok r2 r3);
  compose s0 s1 s2 r0 r1 r2 r3;
  assert pure (SS.optimized_embedding_relation s0 s1 s2 r0 r1 r2 r3);
}
