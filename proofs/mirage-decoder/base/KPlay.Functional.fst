module KPlay.Functional

#lang-pulse

open FStar.Real
open Kuiper
open KPlay.SharedSpec
(**** Sums ****)
open FStar.Real
open KPlay.SharedSpec
module S = FStar.Seq
module Exp = FStar.Math.Exp

#push-options "--fuel 1 --ifuel 0 --z3rlimit 10"

noextract let rec rs (n:nat) (f:nat -> real) : Tot real (decreases n) =
  if n = 0 then 0.0R else rs (n - 1) f +. f (n - 1)

noextract let rec rs_ext (n:nat) (f g:nat -> real)
  : Lemma (requires forall (i:nat). i < n ==> f i == g i) (ensures rs n f == rs n g)
          (decreases n)
  = if n = 0 then () else rs_ext (n - 1) f g

noextract let rec rs_add (n:nat) (f g:nat -> real)
  : Lemma (ensures rs n (fun i -> f i +. g i) == rs n f +. rs n g) (decreases n)
  = if n = 0 then () else rs_add (n - 1) f g

noextract let rec rs_scale (c:real) (n:nat) (f:nat -> real)
  : Lemma (ensures rs n (fun i -> c *. f i) == c *. rs n f) (decreases n)
  = if n = 0 then () else rs_scale c (n - 1) f

noextract let rec rs_split (a b:nat) (f:nat -> real)
  : Lemma (ensures rs (a + b) f == rs a f +. rs b (fun i -> f (a + i))) (decreases b)
  = if b = 0 then () else rs_split a (b - 1) f

noextract let rec rs_block (a b:nat) (f:nat -> real)
  : Lemma (ensures rs (a * b) f == rs a (fun i -> rs b (fun j -> f (i * b + j)))) (decreases a)
  = if a = 0 then ()
    else begin
      let a' : nat = a - 1 in
      let ab' : nat = a' * b in
      assert (a * b == ab' + b);
      let row (i:nat) : real = rs b (fun j -> f (i * b + j)) in
      rs_block a' b f;
      rs_split ab' b f;
      rs_ext b (fun j -> f (ab' + j)) (fun j -> f (a' * b + j));
      assert (rs (a * b) f == rs a' row +. row a');
      rs_ext a row (fun i -> rs b (fun j -> f (i * b + j)))
    end

noextract let rec rs_zero (n:nat)
  : Lemma (ensures rs n (fun _ -> 0.0R) == 0.0R) (decreases n)
  = if n = 0 then () else rs_zero (n - 1)

noextract let rec rs_swap (a b:nat) (f:nat -> nat -> real)
  : Lemma (ensures rs a (fun i -> rs b (fun j -> f i j)) == rs b (fun j -> rs a (fun i -> f i j)))
          (decreases a)
  = if a = 0 then begin
      rs_zero b;
      let aux (j:nat) : Lemma (rs a (fun i -> f i j) == 0.0R) = () in
      FStar.Classical.forall_intro aux;
      rs_ext b (fun j -> 0.0R) (fun j -> rs a (fun i -> f i j))
    end
    else begin
      let a' : nat = a - 1 in
      let g1 (j:nat) : real = rs a' (fun i -> f i j) in
      let g2 (j:nat) : real = f a' j in
      rs_swap a' b f;
      rs_ext a' (fun i -> rs b (fun j -> f i j)) (fun i -> rs b (fun j -> f i j));
      rs_ext b g2 (fun j -> f a' j);
      rs_ext b g1 (fun j -> rs a' (fun i -> f i j));
      rs_add b g1 g2;
      rs_ext b (fun j -> g1 j +. g2 j) (fun j -> rs a (fun i -> f i j))
    end

noextract let rs_strided (a b:nat) (f:nat -> real)
  : Lemma (rs (a * b) f == rs b (fun j -> rs a (fun i -> f (i * b + j))))
  = rs_block a b f;
    rs_swap a b (fun i j -> f (i * b + j))

noextract let rec tree (p:nat) (f:nat -> real) : Tot real (decreases p) =
  if p = 0 then f 0 else tree (p - 1) (fun i -> f i +. f (i + pow2 (p - 1)))

noextract let rec tree_rs (p:nat) (f:nat -> real)
  : Lemma (ensures tree p f == rs (pow2 p) f) (decreases p)
  = if p = 0 then begin
      assert (pow2 0 == 1);
      assert (rs 0 f == 0.0R);
      assert (rs 1 f == rs 0 f +. f 0)
    end
    else begin
      let h : nat = pow2 (p - 1) in
      assert (pow2 p == h + h);
      let g (i:nat) : real = f i +. f (i + h) in
      let f2 (i:nat) : real = f (i + h) in
      tree_rs (p - 1) (fun i -> f i +. f (i + pow2 (p - 1)));
      rs_ext h (fun i -> f i +. f (i + pow2 (p - 1))) (fun i -> f i +. f2 i);
      rs_add h f f2;
      rs_split h h f;
      rs_ext h (fun i -> f (h + i)) f2
    end

noextract let rec sdot_rs (n:nat) (a:S.seq real) (ao sa:int) (b:S.seq real) (bo sb:int)
  : Lemma (ensures sdot n a ao sa b bo sb == rs n (fun k -> at a (ao + k * sa) *. at b (bo + k * sb)))
          (decreases n)
  = if n = 0 then () else sdot_rs (n - 1) a ao sa b bo sb

noextract let rec sumsq_rs (n:nat) (a:S.seq real) (o:int)
  : Lemma (ensures sumsq n a o == rs n (fun k -> at a (o + k) *. at a (o + k))) (decreases n)
  = if n = 0 then () else sumsq_rs (n - 1) a o

noextract let rec sumexp_rs (n:nat) (a:S.seq real) (o:int)
  : Lemma (ensures sumexp n a o == rs n (fun k -> Exp.exp (at a (o + k)))) (decreases n)
  = if n = 0 then () else sumexp_rs (n - 1) a o

noextract let rec rs_pos (n:pos) (f:nat -> real)
  : Lemma (requires forall (i:nat). i < n ==> f i >. 0.0R) (ensures rs n f >. 0.0R)
          (decreases n)
  = if n = 1 then assert (rs 0 f == 0.0R) else rs_pos (n - 1) f

noextract let softmax_shift (n:pos) (s:nat -> real) (m:real) (k:nat)
  : Lemma (rs n (fun i -> Exp.exp (s i -. m)) >. 0.0R /\ rs n (fun i -> Exp.exp (s i)) >. 0.0R /\
           Exp.exp (s k -. m) /. rs n (fun i -> Exp.exp (s i -. m)) ==
           Exp.exp (s k) /. rs n (fun i -> Exp.exp (s i)))
  = let e (i:nat) : real = Exp.exp (s i) in
    let c : real = 1.0R /. Exp.exp m in
    Exp.exp_positive m;
    rs_pos n (fun i -> Exp.exp (s i -. m));
    rs_pos n (fun i -> Exp.exp (s i));
    let aux (i:nat) : Lemma (Exp.exp (s i -. m) == c *. e i) = Exp.exp_sub (s i) m in
    FStar.Classical.forall_intro aux;
    rs_ext n (fun i -> Exp.exp (s i -. m)) (fun i -> c *. e i);
    rs_scale c n e;
    rs_ext n e (fun i -> Exp.exp (s i));
    let r : real = rs n (fun i -> Exp.exp (s i)) in
    let em : real = Exp.exp m in
    let ek : real = Exp.exp (s k) in
    assert (rs n (fun i -> Exp.exp (s i -. m)) == c *. r);
    assert (Exp.exp (s k -. m) == c *. ek);
    assert (c >. 0.0R);
    assert ((c *. ek) /. (c *. r) == ek /. r)

#pop-options

(**** Defs ****)

(* Shared definitions (interfaces between proof pieces). Do not change. *)
open FStar.Real
open Kuiper
open Kuiper.Chest
open Kuiper.Shape
open KPlay.SharedSpec
open Kuiper.EMatrix.Tiling
open Kuiper.TensorCore.MMA

(* Total ghost accessors on 1D chests *)
noextract let bat (#n:nat) (s:chest1 bf16 n) (i:int) : GTot bf16 =
  if 0 <= i && i < n then acc1 s i else zero
noextract let fat (#n:nat) (s:chest1 float n) (i:int) : GTot float =
  if 0 <= i && i < n then acc1 s i else zero

(* Views of the attention scratch: qk is [7][24][128] (16 query heads then 8 key heads),
   qkv is [7][4096] with v at columns 3072..4095. *)
noextract let qpart (qr:S.seq real) : S.seq real =
  S.init 14336 (fun (i:nat{i < 14336}) -> at qr (((i / 2048) * 24 + (i % 2048) / 128) * 128 + i % 128))
noextract let kpart (qr:S.seq real) : S.seq real =
  S.init 7168 (fun (i:nat{i < 7168}) -> at qr (((i / 1024) * 24 + 16 + (i % 1024) / 128) * 128 + i % 128))
noextract let vpart (vr:S.seq real) : S.seq real =
  S.init 7168 (fun (i:nat{i < 7168}) -> at vr ((i / 1024) * 4096 + 3072 + i % 1024))

(* MMA projection model *)
noextract let ktile (#rt #kt : pos) (w : chest2 bf16 (16 * rt) (16 * kt)) (tr : natlt rt) (k : natlt kt)
  : chest2 bf16 16 16 = ematrix_subtile w 16 16 tr k

noextract let xtile (#kt : pos) (x : chest2 bf16 (16 * kt) 8) (k : natlt kt)
  : chest2 bf16 16 8 = ematrix_subtile x 16 8 k 0

noextract let rec kacc (#rt #kt : pos) (w : chest2 bf16 (16 * rt) (16 * kt)) (x : chest2 bf16 (16 * kt) 8)
  (tr : natlt rt) (k0 : nat) (k : nat)
  : Tot (chest2 float 16 8) (decreases k - k0)
  = if k <= k0 || k > kt then const (16 @| 8 @| INil) zero
    else emma (kacc w x tr k0 (k - 1)) (ktile w tr (k - 1)) (xtile x (k - 1))

noextract let rec cmod (#rt #kt : pos) (w : chest2 bf16 (16 * rt) (16 * kt)) (x : chest2 bf16 (16 * kt) 8)
  (c0 : chest2 float (16 * rt) 8) (k0 : nat) (k1 : nat) (n : nat)
  : Tot (chest2 float (16 * rt) 8) (decreases n)
  = if n = 0 || n > rt then c0
    else update_tile (cmod w x c0 k0 k1 (n - 1)) 16 8 (n - 1) 0 (kacc w x (n - 1) k0 k1)

(**** Rms ****)

(* Native model of the CUDA rms_inverse device function and its approximation
   of the real spec rms_inv. *)
open FStar.Real
open Kuiper
open Kuiper.Float.Casts
open KPlay.SharedSpec
module I64 = FStar.Int64

(* width = size == 1024 ? 128 : 32, width = pow2 rlog *)
noextract
let rwidth (size:nat) : pos = if size = 1024 then 128 else 32
noextract
let rlog (size:nat) : nat = if size = 1024 then 7 else 5
(* number of iterations of the start loop *)
noextract
let rnit (size:nat) : nat = size / (4 * rwidth size)

noextract
let i64n (n:nat) : I64.t = if n < pow2 63 then I64.int_to_t n else 0L

(* sums[j] after k iterations of the start loop of lane [lane] (stride st = width*4) *)
noextract
let rec lsum (x:nat -> GTot bf16) (st lane j:nat) (k:nat) : GTot float (decreases k) =
  if k = 0 then zero
  else
    let v : float = fcast (x ((k - 1) * st + (lane * 4 + j))) in
    add (lsum x st lane j (k - 1)) (mul v v)

(* partial[lane] after the first loop *)
noextract
let lpart (x:nat -> GTot bf16) (st nit lane:nat) : GTot float =
  add (add (add (lsum x st lane 0 nit) (lsum x st lane 1 nit)) (lsum x st lane 2 nit))
      (lsum x st lane 3 nit)

(* halving tree over pow2 p values *)
noextract
let rec ftree (p:nat) (f:nat -> GTot float) : GTot float (decreases p) =
  if p = 0 then f 0
  else ftree (p - 1) (fun (i:nat) -> add (f i) (f (i + pow2 (p - 1))))

(* in-place tree contents after k levels (level k uses offset pow2 (p - k)) *)
noextract
let rec hk (p:nat) (f:nat -> GTot float) (k:nat) : Tot (nat -> GTot float) (decreases k) =
  if k = 0 || k > p then f
  else
    let g = hk p f (k - 1) in
    let h : pos = pow2 (p - k) in
    fun (i:nat) -> if i < h then add (g i) (g (i + h)) else g i

noextract
let rms_inv_scale (size:nat) : float = div one (of_int (i64n size))
noextract
let rms_eps_f : float = div one (of_int 1000000L)

noextract
let lanef (x:nat -> GTot bf16) (size:nat) (lane:nat) : GTot float =
  lpart x (4 * rwidth size) (rnit size) lane

noextract
let rms_model (size:nat) (x: nat -> GTot bf16) : GTot float =
  let tot = ftree (rlog size) (lanef x size) in
  let mean = mul tot (rms_inv_scale size) in
  rsqrt (add mean rms_eps_f)

(* ------------------------------------------------------------------ *)
(* Tree lemmas                                                         *)

#push-options "--fuel 1 --ifuel 0 --z3rlimit 10"

noextract
let rec ftree_ext (p:nat) (f g:nat -> GTot float)
  : Lemma (requires forall (i:nat). i < pow2 p ==> f i == g i)
          (ensures ftree p f == ftree p g) (decreases p)
  = if p = 0 then ()
    else ftree_ext (p - 1) (fun (i:nat) -> add (f i) (f (i + pow2 (p - 1))))
                           (fun (i:nat) -> add (g i) (g (i + pow2 (p - 1))))

noextract
let ftree_step (q:pos) (f g:nat -> GTot float)
  : Lemma (requires forall (i:nat). i < pow2 (q - 1) ==> g i == add (f i) (f (i + pow2 (q - 1))))
          (ensures ftree q f == ftree (q - 1) g)
  = ftree_ext (q - 1) (fun (i:nat) -> add (f i) (f (i + pow2 (q - 1)))) g

noextract
let rec ftree_hk (p:nat) (f:nat -> GTot float) (k:nat{k <= p})
  : Lemma (ensures ftree p f == ftree (p - k) (hk p f k)) (decreases k)
  = if k = 0 then ()
    else begin
      ftree_hk p f (k - 1);
      ftree_step (p - (k - 1)) (hk p f (k - 1)) (hk p f k)
    end

noextract
let ftree_hk_full (p:nat) (f:nat -> GTot float)
  : Lemma (ftree p f == hk p f p 0)
  = ftree_hk p f p

noextract
let rec tree_ext (p:nat) (f g:nat -> real)
  : Lemma (requires forall (i:nat). i < pow2 p ==> f i == g i)
          (ensures tree p f == tree p g) (decreases p)
  = if p = 0 then ()
    else tree_ext (p - 1) (fun (i:nat) -> f i +. f (i + pow2 (p - 1)))
                          (fun (i:nat) -> g i +. g (i + pow2 (p - 1)))

noextract
let tree_step (q:pos) (f g:nat -> real)
  : Lemma (requires forall (i:nat). i < pow2 (q - 1) ==> g i == f i +. f (i + pow2 (q - 1)))
          (ensures tree q f == tree (q - 1) g)
  = tree_ext (q - 1) (fun (i:nat) -> f i +. f (i + pow2 (q - 1))) g

noextract
let rec ftree_approx (p:nat) (f:nat -> GTot float) (g:nat -> real)
  : Lemma (requires forall (i:nat). i < pow2 p ==> f i %~ g i)
          (ensures ftree p f %~ tree p g) (decreases p)
  = if p = 0 then ()
    else begin
      let h : pos = pow2 (p - 1) in
      let f' (i:nat) : GTot float = add (f i) (f (i + h)) in
      let g' (i:nat) : real = g i +. g (i + h) in
      let aux (i:nat) : Lemma (i < pow2 (p - 1) ==> f' i %~ g' i) =
        if i < h then (assert (f i %~ g i); assert (f (i + h) %~ g (i + h))) in
      FStar.Classical.forall_intro aux;
      ftree_approx (p - 1) f' g';
      ftree_step p f f';
      tree_step p g g'
    end

(* ------------------------------------------------------------------ *)
(* Lane sums                                                           *)

noextract
let zero_ok () : Lemma ((zero <: float) %~ 0.0R) = Kuiper.Approximates.Scalar.a0 #float
noextract
let one_ok () : Lemma ((one <: float) %~ 1.0R) = Kuiper.Approximates.Scalar.a1 #float

noextract
let sqr (xr:S.seq real) (o:int) (i:nat) : real = at xr (o + i) *. at xr (o + i)

noextract
let lane_real (xr:S.seq real) (o:int) (st nit:nat) (m:nat) : real =
  rs nit (fun (t:nat) -> sqr xr o (t * st + m))

noextract
let rec lsum_approx (size:pos) (x:nat -> GTot bf16) (xr:S.seq real) (o:int)
  (st lane j:nat) (k:nat)
  : Lemma (requires (forall (i:nat). i < size ==> x i %~ at xr (o + i)) /\
                    lane * 4 + j < st /\ k * st <= size)
          (ensures lsum x st lane j k %~ lane_real xr o st k (lane * 4 + j))
          (decreases k)
  = if k = 0 then zero_ok ()
    else begin
      lsum_approx size x xr o st lane j (k - 1);
      let idx : nat = (k - 1) * st + (lane * 4 + j) in
      assert (idx < size);
      let v : float = fcast (x idx) in
      assert (x idx %~ at xr (o + idx));
      assert (v %~ at xr (o + idx));
      assert (mul v v %~ sqr xr o idx)
    end

noextract
let rs4 (f:nat -> real) : Lemma (rs 4 f == ((f 0 +. f 1) +. f 2) +. f 3)
  = assert (rs 0 f == 0.0R);
    assert (rs 1 f == rs 0 f +. f 0);
    assert (rs 2 f == rs 1 f +. f 1);
    assert (rs 3 f == rs 2 f +. f 2);
    assert (rs 4 f == rs 3 f +. f 3)

noextract
let lpart_approx (size:pos) (x:nat -> GTot bf16) (xr:S.seq real) (o:int)
  (st nit lane:nat)
  : Lemma (requires (forall (i:nat). i < size ==> x i %~ at xr (o + i)) /\
                    lane * 4 + 4 <= st /\ nit * st <= size)
          (ensures lpart x st nit lane %~
                   rs 4 (fun (j:nat) -> lane_real xr o st nit (lane * 4 + j)))
  = let r (j:nat) : real = lane_real xr o st nit (lane * 4 + j) in
    let aux (j:nat{j < 4}) : Lemma (lsum x st lane j nit %~ r j) =
      lsum_approx size x xr o st lane j nit
    in
    aux 0; aux 1; aux 2; aux 3;
    rs4 r;
    rs_ext 4 r (fun (j:nat) -> lane_real xr o st nit (lane * 4 + j))

(* sumsq over a block, regrouped as lanes of 4 strided columns *)
noextract
let sumsq_lanes (xr:S.seq real) (o:int) (w:nat) (st:nat{st == w * 4}) (nit:nat)
  : Lemma (sumsq (nit * st) xr o ==
           rs w (fun (lane:nat) -> rs 4 (fun (j:nat) -> lane_real xr o st nit (lane * 4 + j))))
  = sumsq_rs (nit * st) xr o;
    rs_ext (nit * st) (fun k -> at xr (o + k) *. at xr (o + k)) (sqr xr o);
    rs_strided nit st (sqr xr o);
    rs_ext st (fun j -> rs nit (fun i -> sqr xr o (i * st + j))) (lane_real xr o st nit);
    rs_block w 4 (lane_real xr o st nit)

noextract
let pow2_vals () : Lemma (pow2 5 == 32 /\ pow2 7 == 128)
  = assert (pow2 1 == 2); assert (pow2 2 == 4); assert (pow2 3 == 8);
    assert (pow2 4 == 16); assert (pow2 5 == 32); assert (pow2 6 == 64);
    assert (pow2 7 == 128)

noextract
let scale_approx (n:pos{n < pow2 63})
  : Lemma ((div one (of_int (I64.int_to_t n)) <: float) %~ (1.0R /. FStar.Real.of_int n))
  = one_ok ();
    let ni : I64.t = I64.int_to_t n in
    let d : float = of_int ni in
    Kuiper.Approximates.Base.approx_of_int #float;
    Kuiper.Approximates.Core.approx_apply (of_int #float) Kuiper.Float.Realops.r_of_int ni ni;
    assert (d %~ FStar.Real.of_int n);
    Kuiper.Approximates.Base.approx_div #float;
    assert (FStar.Real.of_int n =!= 0.0R);
    assert ((div one d <: float) %~ Kuiper.Float.Realops.r_div 1.0R (FStar.Real.of_int n))

noextract
let mean_real (ss:real) (n:pos)
  : Lemma (requires ss >=. 0.0R)
          (ensures ss *. (1.0R /. FStar.Real.of_int n) == ss /. FStar.Real.of_int n /\ ss /. FStar.Real.of_int n +. rms_eps >. 0.0R)
  = ()

#pop-options

#push-options "--fuel 1 --ifuel 0 --z3rlimit 20"
noextract
let rms_model_approx (size:pos{size == 128 \/ size == 1024}) (x: nat -> GTot bf16) (xr: S.seq real) (o: int)
  : Lemma (requires forall (i:nat). i < size ==> x i %~ at xr (o + i))
          (ensures rms_model size x %~ rms_inv size xr o)
  = let w = rwidth size in
    let p = rlog size in
    let st = 4 * w in
    let nit = rnit size in
    pow2_vals ();
    assert (pow2 p == w);
    assert (nit * (w * 4) == size);
    let f = lanef x size in
    let g (lane:nat) : real = rs 4 (fun (j:nat) -> lane_real xr o st nit (lane * 4 + j)) in
    let aux (lane:nat) : Lemma (lane < pow2 p ==> f lane %~ g lane) =
      if lane < w then lpart_approx size x xr o st nit lane in
    FStar.Classical.forall_intro aux;
    ftree_approx p f g;
    tree_rs p g;
    sumsq_lanes xr o w st nit;
    let tot = ftree p f in
    let ss = sumsq size xr o in
    assert (tot %~ ss);
    assert (i64n size == I64.int_to_t size);
    scale_approx size;
    scale_approx 1000000;
    assert (rms_inv_scale size %~ (1.0R /. FStar.Real.of_int size));
    assert (rms_eps_f %~ rms_eps);
    mean_real ss size;
    let mean = mul tot (rms_inv_scale size) in
    assert (mean %~ (ss *. (1.0R /. FStar.Real.of_int size)));
    assert (add mean rms_eps_f %~ (ss /. FStar.Real.of_int size +. rms_eps));
    assert (ss /. FStar.Real.of_int size +. rms_eps >. 0.0R);
    Kuiper.Approximates.Base.approx_rsqrt #float;
    assert (rms_model size x == rsqrt (add mean rms_eps_f))
#pop-options

(* ------------------------------------------------------------------ *)
(* Extensionality of the model                                         *)

#push-options "--fuel 1 --ifuel 0 --z3rlimit 10"
noextract
let rec lsum_ext (size:pos) (x y:nat -> GTot bf16) (st lane j:nat) (k:nat)
  : Lemma (requires (forall (i:nat). i < size ==> x i == y i) /\
                    lane * 4 + j < st /\ k * st <= size)
          (ensures lsum x st lane j k == lsum y st lane j k)
          (decreases k)
  = if k = 0 then ()
    else begin
      lsum_ext size x y st lane j (k - 1);
      let idx : nat = (k - 1) * st + (lane * 4 + j) in
      assert (idx < size)
    end

noextract
let rms_model_ext (size:nat) (x y: nat -> GTot bf16)
  : Lemma (requires (size == 128 \/ size == 1024) /\ (forall (i:nat). i < size ==> x i == y i))
          (ensures rms_model size x == rms_model size y)
  = let w = rwidth size in
    let st = 4 * w in
    let nit = rnit size in
    pow2_vals ();
    assert (pow2 (rlog size) == w);
    assert (nit * st == size);
    let aux (lane:nat) : Lemma (lane < pow2 (rlog size) ==> lanef x size lane == lanef y size lane) =
      if lane < w then begin
        lsum_ext size x y st lane 0 nit;
        lsum_ext size x y st lane 1 nit;
        lsum_ext size x y st lane 2 nit;
        lsum_ext size x y st lane 3 nit
      end in
    FStar.Classical.forall_intro aux;
    ftree_ext (rlog size) (lanef x size) (lanef y size)
#pop-options

(* ------------------------------------------------------------------ *)
(* Facts used by the device implementation                             *)

#push-options "--fuel 1 --ifuel 0 --z3rlimit 10"
noextract
let hk_step (p:nat) (f:nat -> GTot float) (k:nat{k < p}) (i:nat)
  : Lemma (hk p f (k + 1) i ==
           (if i < pow2 (p - (k + 1)) then add (hk p f k i) (hk p f k (i + pow2 (p - (k + 1))))
            else hk p f k i))
  = ()

noextract
let pow2_half (m:nat)
  : Lemma (pow2 m / 2 == (if m = 0 then 0 else pow2 (m - 1)))
  = if m = 0 then () else ()

noextract
let lsum_step (x:nat -> GTot bf16) (st lane j:nat) (k:nat)
  : Lemma (let v : float = fcast (x (k * st + (lane * 4 + j))) in
           lsum x st lane j (k + 1) == add (lsum x st lane j k) (mul v v))
  = ()

noextract
let start_step (size:pos{size == 128 \/ size == 1024}) (lv k:nat)
  : Lemma (requires lv < rwidth size /\ lv * 4 + k * (4 * rwidth size) < size)
          (ensures k < rnit size /\ lv * 4 + k * (4 * rwidth size) + 4 <= size /\
                   lv * 4 + (k + 1) * (4 * rwidth size) <= size + 4 * rwidth size)
  = ()

noextract
let start_exit (size:pos{size == 128 \/ size == 1024}) (lv k:nat)
  : Lemma (requires lv < rwidth size /\ k <= rnit size /\ lv * 4 + k * (4 * rwidth size) >= size)
          (ensures k == rnit size)
  = ()

noextract
let start_bound (size:pos{size == 128 \/ size == 1024}) (lv k:nat)
  : Lemma (requires lv < rwidth size /\ k <= rnit size)
          (ensures lv * 4 + k * (4 * rwidth size) < size + 4 * rwidth size)
  = ()

noextract
let rms_model_unfold (size:nat) (x:nat -> GTot bf16) (tot:float)
  : Lemma (requires tot == hk (rlog size) (lanef x size) (rlog size) 0)
          (ensures rms_model size x == rsqrt (add (mul tot (rms_inv_scale size)) rms_eps_f))
  = ftree_hk_full (rlog size) (lanef x size)
#pop-options

#push-options "--fuel 1 --ifuel 0 --z3rlimit 10"
noextract
let tree_lvl (p k ov:nat)
  : Lemma (requires k <= p /\ ov == pow2 (p - k) / 2 /\ ov > 0)
          (ensures k < p /\ ov == pow2 (p - (k + 1)) /\ ov + ov == pow2 (p - k) /\
                   ov / 2 == pow2 (p - (k + 1)) / 2)
  = pow2_half (p - k); pow2_half (p - (k + 1))

noextract
let tree_end (p k:nat)
  : Lemma (requires k <= p /\ pow2 (p - k) / 2 == 0) (ensures k == p)
  = pow2_half (p - k)

noextract
let tree_init (p:nat{p <= 7})
  : Lemma (pow2 p <= 128 /\ (forall (k:nat). k <= p ==> pow2 (p - k) <= 128))
  = FStar.Math.Lemmas.pow2_le_compat 7 p;
    pow2_vals ();
    let aux (k:nat) : Lemma (k <= p ==> pow2 (p - k) <= 128) =
      if k <= p then FStar.Math.Lemmas.pow2_le_compat 7 (p - k) in
    FStar.Classical.forall_intro aux
#pop-options

(**** Proj ****)

open FStar.Real
open Kuiper
open Kuiper.Chest
open Kuiper.Shape
open KPlay.SharedSpec
open Kuiper.EMatrix.Tiling
open Kuiper.TensorCore.MMA
open Kuiper.Spec.GEMM

#push-options "--fuel 1 --ifuel 1 --z3rlimit 10"

noextract let approx2_intro (#et:Type0) {| scalar et, real_like et |} (#m #n:nat)
  (c : chest2 et m n) (r : chest2 real m n)
  : Lemma (requires forall (i:natlt m) (j:natlt n). acc2 c i j %~ acc2 r i j)
          (ensures c %~ r)
  = let aux (ix : abs (m @| n @| INil)) : Lemma (acc c ix %~ acc r ix)
      = let (i, (j, ())) = ix in assert (acc2 c i j %~ acc2 r i j)
    in
    FStar.Classical.forall_intro aux

noextract let eq2_intro (#et:Type) (#m #n:nat) (c1 c2 : chest2 et m n)
  : Lemma (requires forall (i:natlt m) (j:natlt n). acc2 c1 i j == acc2 c2 i j)
          (ensures c1 == c2)
  = let aux (ix : abs (m @| n @| INil)) : Lemma (acc c1 ix == acc c2 ix)
      = let (i, (j, ())) = ix in assert (acc2 c1 i j == acc2 c2 i j)
    in
    FStar.Classical.forall_intro aux;
    lemma_equal_intro c1 c2;
    ext c1 c2

noextract let kterm (wr xr : nat -> nat -> real) (row kb j : nat) (kk : nat) : real =
  wr row (16 * kb + kk) *. xr (16 * kb + kk) j

noextract let rec mm_kterm (wr xr : nat -> nat -> real) (row kb : nat) (j : natlt 8)
  (ra : chest2 real 16 16) (rb : chest2 real 16 8) (i : natlt 16) (to : nat{to <= 16})
  : Lemma (requires (forall (a:natlt 16). acc2 ra i a == wr row (16 * kb + a)) /\
                    (forall (a:natlt 16). acc2 rb a j == xr (16 * kb + a) j))
          (ensures __matmul_single ra rb i j to == rs to (kterm wr xr row kb j))
          (decreases to)
  = if to = 0 then () else mm_kterm wr xr row kb j ra rb i (to - 1)

noextract let ksum (wr xr : nat -> nat -> real) (row k0 : nat) (k : nat{k0 <= k}) (j : nat) : real =
  rs (16 * (k - k0)) (kterm wr xr row k0 j)

noextract let ksum_step (wr xr : nat -> nat -> real) (row k0 : nat) (k : pos{k0 < k}) (j : nat)
  : Lemma (ksum wr xr row k0 k j ==
           ksum wr xr row k0 (k - 1) j +. rs 16 (kterm wr xr row (k - 1) j))
  = let m : nat = 16 * (k - 1 - k0) in
    rs_split m 16 (kterm wr xr row k0 j);
    rs_ext 16 (fun b -> kterm wr xr row k0 j (m + b)) (kterm wr xr row (k - 1) j)

noextract let rec kacc_approx (#rt #kt : pos) (w : chest2 bf16 (16 * rt) (16 * kt)) (x : chest2 bf16 (16 * kt) 8)
  (wr xr : nat -> nat -> real) (tr : natlt rt) (k0 : nat) (k : nat)
  : Lemma (requires k0 <= k /\ k <= kt /\
            (forall (i:natlt (16*rt)) (k:natlt (16*kt)). acc2 w i k %~ wr i k) /\
            (forall (k:natlt (16*kt)) (j:natlt 8). acc2 x k j %~ xr k j))
          (ensures forall (i:natlt 16) (j:natlt 8).
                     acc2 (kacc w x tr k0 k) i j %~ ksum wr xr (16 * tr + i) k0 k j)
          (decreases k - k0)
  = if k = k0 then ()
    else begin
      kacc_approx w x wr xr tr k0 (k - 1);
      let kb : natlt kt = k - 1 in
      let rc : chest2 real 16 8 = mk2 (fun i j -> ksum wr xr (16 * tr + i) k0 (k - 1) j) in
      let ra : chest2 real 16 16 = mk2 (fun a b -> wr (16 * tr + a) (16 * kb + b)) in
      let rb : chest2 real 16 8 = mk2 (fun a b -> xr (16 * kb + a) b) in
      let mc = kacc w x tr k0 (k - 1) in
      let ma = ktile w tr kb in
      let mb = xtile x kb in
      approx2_intro mc rc;
      approx2_intro ma ra;
      approx2_intro mb rb;
      emma_approx_lemma mc ma mb rc ra rb;
      let aux (i:natlt 16) (j:natlt 8)
        : Lemma (acc2 (matplus rc (matmul ra rb)) i j == ksum wr xr (16 * tr + i) k0 k j)
        = lemma_matmul_index ra rb i j;
          mm_kterm wr xr (16 * tr + i) kb j ra rb i 16;
          ksum_step wr xr (16 * tr + i) k0 k j
      in
      FStar.Classical.forall_intro_2 aux
    end

noextract let rec cmod_approx_n (#rt #kt:pos) (w: chest2 bf16 (16*rt) (16*kt)) (x: chest2 bf16 (16*kt) 8)
  (c0: chest2 float (16*rt) 8) (k0 k1: nat) (wr xr: nat -> nat -> real) (n:nat)
  (r: natlt (16*rt)) (t: natlt 8)
  : Lemma (requires k0 <= k1 /\ k1 <= kt /\ n <= rt /\ r / 16 < n /\
            (forall (i:natlt (16*rt)) (k:natlt (16*kt)). acc2 w i k %~ wr i k) /\
            (forall (k:natlt (16*kt)) (j:natlt 8). acc2 x k j %~ xr k j))
          (ensures acc2 (cmod w x c0 k0 k1 n) r t %~ ksum wr xr r k0 k1 t)
          (decreases n)
  = let tr : natlt rt = n - 1 in
    FStar.Math.Lemmas.small_div t 8;
    FStar.Math.Lemmas.small_mod t 8;
    if r / 16 = tr then begin
      FStar.Math.Lemmas.euclidean_division_definition r 16;
      kacc_approx w x wr xr tr k0 k1
    end
    else cmod_approx_n w x c0 k0 k1 wr xr (n - 1) r t

noextract let cmod_approx (#rt #kt:pos) (w: chest2 bf16 (16*rt) (16*kt)) (x: chest2 bf16 (16*kt) 8)
  (c0: chest2 float (16*rt) 8) (k0 k1: nat) (wr xr: nat -> nat -> real) (r: natlt (16*rt)) (t: natlt 8)
  : Lemma (requires k0 <= k1 /\ k1 <= kt /\
            (forall (i:natlt (16*rt)) (k:natlt (16*kt)). acc2 w i k %~ wr i k) /\
            (forall (k:natlt (16*kt)) (j:natlt 8). acc2 x k j %~ xr k j))
          (ensures acc2 (cmod w x c0 k0 k1 rt) r t %~
                   rs (16 * (k1 - k0)) (fun k -> wr r (16 * k0 + k) *. xr (16 * k0 + k) t))
  = FStar.Math.Lemmas.euclidean_division_definition r 16;
    cmod_approx_n w x c0 k0 k1 wr xr rt r t;
    rs_ext (16 * (k1 - k0)) (kterm wr xr r k0 t) (fun k -> wr r (16 * k0 + k) *. xr (16 * k0 + k) t)
#pop-options

(**** AttQk ****)

open FStar.Real
open Kuiper
open Kuiper.Chest
open Kuiper.Float.Casts
open KPlay.SharedSpec

noextract let f32 (x:bf16) : float = fcast x
noextract let b16 (x:float) : bf16 = fcast x

noextract let qk_noff (off:nat) (head:nat) : nat = off + 6293504 + (if head < 16 then 0 else 128)

noextract let qk_inv (sv:chest1 bf16 28672) (row head:nat) : GTot float =
  rms_model 128 (fun i -> bat sv (row * 4096 + head * 128 + i))

noextract let nrm_m (#n0:nat) (sw:chest1 bf16 n0) (off:nat) (sv:chest1 bf16 28672) (row head c:nat) : GTot bf16 =
  b16 (mul (f32 (b16 (mul (f32 (bat sv (row * 4096 + head * 128 + c))) (qk_inv sv row head))))
           (f32 (bat sw (qk_noff off head + c))))

noextract let rot_m (x other:float) (cs sn:bf16) : bf16 =
  b16 (add (f32 (b16 (mul x (f32 cs)))) (f32 (b16 (mul other (f32 sn)))))

noextract let qk_m (#n0:nat) (sw:chest1 bf16 n0) (off:nat) (sv:chest1 bf16 28672) (sr:chest1 bf16 1792) (i:nat)
  : GTot bf16 =
  let row = i / 3072 in
  let head = (i / 128) % 24 in
  let c = i % 128 in
  let x = f32 (nrm_m sw off sv row head c) in
  let other : float =
    if c < 64 then neg (f32 (nrm_m sw off sv row head (c + 64)))
    else f32 (nrm_m sw off sv row head (c - 64)) in
  rot_m x other (bat sr (row * 128 + c)) (bat sr (896 + row * 128 + c))

(* real counterpart *)
noextract let nrm_r (vr rw:S.seq real) (off:nat) (row head c:nat) : real =
  let base = row * 4096 + head * 128 in
  (at vr (base + c) *. rms_inv 128 vr base) *. at rw (qk_noff off head + c)

noextract let qk_r (vr rw rr:S.seq real) (off:nat) (i:nat) : real =
  let row = i / 3072 in
  let head = (i / 128) % 24 in
  let c = i % 128 in
  let other = if c < 64 then 0.0R -. nrm_r vr rw off row head (c + 64)
              else nrm_r vr rw off row head (c - 64) in
  nrm_r vr rw off row head c *. at rr (row * 128 + c) +. other *. at rr (896 + row * 128 + c)

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
noextract let nrm_approx (#n0:nat) (sw:chest1 bf16 n0) (off:nat) (sv:chest1 bf16 28672)
  (vr rw:S.seq real) (row:nat{row < 7}) (head:nat{head < 24}) (c:nat{c < 128})
  : Lemma (requires
             off + 6293760 <= n0 /\ S.length rw == n0 /\ S.length vr == 28672 /\
             (forall (i:nat). i < 28672 ==> acc1 sv i %~ S.index vr i) /\
             (forall (i:nat). i < n0 ==> acc1 sw i %~ S.index rw i) /\
             qk_inv sv row head %~ rms_inv 128 vr (row * 4096 + head * 128))
          (ensures nrm_m sw off sv row head c %~ nrm_r vr rw off row head c)
  = let base = row * 4096 + head * 128 in
    assert (base + c < 28672);
    assert (bat sv (base + c) %~ at vr (base + c));
    let x = bat sv (base + c) in
    let xr = at vr (base + c) in
    let w = bat sw (qk_noff off head + c) in
    let wr = at rw (qk_noff off head + c) in
    let inv = qk_inv sv row head in
    let ir = rms_inv 128 vr base in
    assert (w %~ wr);
    assert (f32 x %~ xr);
    assert (f32 w %~ wr);
    assert (mul (f32 x) inv %~ (xr *. ir));
    assert (b16 (mul (f32 x) inv) %~ (xr *. ir));
    assert (f32 (b16 (mul (f32 x) inv)) %~ (xr *. ir));
    assert (mul (f32 (b16 (mul (f32 x) inv))) (f32 w) %~ ((xr *. ir) *. wr));
    assert (b16 (mul (f32 (b16 (mul (f32 x) inv))) (f32 w)) %~ ((xr *. ir) *. wr))

noextract let neg_ap (x:float) (r:real)
  : Lemma (requires x %~ r) (ensures neg x %~ (0.0R -. r))
  = Kuiper.Approximates.Base.neg_models_pat float;
    Kuiper.Approximates.Core.approx_apply (neg #float) Kuiper.Float.Realops.r_neg x r

noextract let rot_approx (x other:float) (cs sn:bf16) (xr otr cr sr:real)
  : Lemma (requires x %~ xr /\ other %~ otr /\ cs %~ cr /\ sn %~ sr)
          (ensures rot_m x other cs sn %~ (xr *. cr +. otr *. sr))
  = assert (f32 cs %~ cr);
    assert (f32 sn %~ sr);
    assert (mul x (f32 cs) %~ (xr *. cr));
    assert (f32 (b16 (mul x (f32 cs))) %~ (xr *. cr));
    assert (mul other (f32 sn) %~ (otr *. sr));
    assert (f32 (b16 (mul other (f32 sn))) %~ (otr *. sr));
    assert (add (f32 (b16 (mul x (f32 cs)))) (f32 (b16 (mul other (f32 sn)))) %~ (xr *. cr +. otr *. sr))

noextract let qk_approx (#n0:nat) (sw:chest1 bf16 n0) (off:nat) (sv:chest1 bf16 28672) (sr:chest1 bf16 1792)
  (vr rw rr:S.seq real) (i:nat{i < 21504})
  : Lemma (requires
             off + 6293760 <= n0 /\ S.length rw == n0 /\ S.length vr == 28672 /\ S.length rr == 1792 /\
             (forall (i:nat). i < 28672 ==> acc1 sv i %~ S.index vr i) /\
             (forall (i:nat). i < n0 ==> acc1 sw i %~ S.index rw i) /\
             (forall (i:nat). i < 1792 ==> acc1 sr i %~ S.index rr i) /\
             (forall (row:nat) (head:nat). row < 7 /\ head < 24 ==>
                qk_inv sv row head %~ rms_inv 128 vr (row * 4096 + head * 128)))
          (ensures qk_m sw off sv sr i %~ qk_r vr rw rr off i)
  = let row = i / 3072 in
    let head = (i / 128) % 24 in
    let c = i % 128 in
    assert (row < 7);
    nrm_approx sw off sv vr rw row head c;
    (if c < 64 then nrm_approx sw off sv vr rw row head (c + 64)
     else nrm_approx sw off sv vr rw row head (c - 64));
    assert (row * 128 + c < 896);
    assert (bat sr (row * 128 + c) %~ at rr (row * 128 + c));
    let cs = bat sr (row * 128 + c) in
    let sn = bat sr (896 + row * 128 + c) in
    assert (cs %~ at rr (row * 128 + c));
    assert (sn %~ at rr (896 + row * 128 + c));
    let x = f32 (nrm_m sw off sv row head c) in
    let xr = nrm_r vr rw off row head c in
    assert (x %~ xr);
    let other : float =
      if c < 64 then neg (f32 (nrm_m sw off sv row head (c + 64)))
      else f32 (nrm_m sw off sv row head (c - 64)) in
    let otr = if c < 64 then 0.0R -. nrm_r vr rw off row head (c + 64)
              else nrm_r vr rw off row head (c - 64) in
    (if c < 64 then neg_ap (f32 (nrm_m sw off sv row head (c + 64))) (nrm_r vr rw off row head (c + 64)));
    assert (other %~ otr);
    rot_approx x other cs sn xr otr (at rr (row * 128 + c)) (at rr (896 + row * 128 + c))
#pop-options

(**** Chain ****)

open FStar.Real
open Kuiper
open Kuiper.Chest
open Kuiper.Shape
open Kuiper.Float.Casts
open KPlay.SharedSpec

noextract let xrf (#kt:pos) (xc:chest2 bf16 (16 * kt) 8) (xr:S.seq real) (k j:nat) : GTot real =
  if j < 7 then at xr (j * (16 * kt) + k)
  else if k < 16 * kt && j < 8 then to_real (acc2 xc k j) else 0.0R

noextract let pterm (rw xr:S.seq real) (wo kk f t b:nat) (k:nat) : real =
  at rw (wo + f * kk + (b + k)) *. at xr (t * kk + (b + k))

noextract let lterm (w x:S.seq real) (kk f t:nat) (k:nat) : real =
  at w (f * kk + k) *. at x (t * kk + k)

noextract let wrf (rw:S.seq real) (wo kk:nat) (i k:nat) : real = at rw (wo + i * kk + k)

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
noextract let proj_chain (#rt #kt:pos) (#n0:nat) (s0:chest1 bf16 n0) (wo:nat) (sx:chest1 bf16 (7 * (16 * kt)))
  (wc:chest2 bf16 (16 * rt) (16 * kt)) (xc:chest2 bf16 (16 * kt) 8) (c0:chest2 float (16 * rt) 8)
  (k0 k1:nat) (rw xr:S.seq real) (f:natlt (16 * rt)) (t:nat{t < 7})
  : Lemma (requires k0 <= k1 /\ k1 <= kt /\ wo + (16 * rt) * (16 * kt) <= n0 /\
             S.length rw == n0 /\ S.length xr == 7 * (16 * kt) /\
             (forall (i:natlt (16 * rt)) (k:natlt (16 * kt)). acc2 wc i k == bat s0 (wo + i * (16 * kt) + k)) /\
             (forall (k:natlt (16 * kt)) (t:nat{t < 7}). acc2 xc k t == bat sx (t * (16 * kt) + k)) /\
             (forall (i:nat). i < n0 ==> acc1 s0 i %~ S.index rw i) /\
             (forall (i:nat). i < 7 * (16 * kt) ==> acc1 sx i %~ S.index xr i))
          (ensures acc2 (cmod wc xc c0 k0 k1 rt) f t %~
                   rs (16 * (k1 - k0)) (pterm rw xr wo (16 * kt) f t (16 * k0)))
  = let kk = 16 * kt in
    let aw (i:natlt (16 * rt)) (k:natlt (16 * kt)) : Lemma (acc2 wc i k %~ wrf rw wo kk i k) =
      FStar.Math.Lemmas.lemma_mult_le_right kk i (16 * rt - 1);
      assert (wo + i * kk + k < n0) in
    let ax (k:natlt (16 * kt)) (j:natlt 8) : Lemma (acc2 xc k j %~ xrf xc xr k j) =
      if j < 7 then begin
        FStar.Math.Lemmas.lemma_mult_le_right kk j 6;
        assert (j * kk + k < 7 * kk)
      end else () in
    Classical.forall_intro_2 aw;
    Classical.forall_intro_2 ax;
    cmod_approx wc xc c0 k0 k1 (wrf rw wo kk) (xrf xc xr) f t;
    rs_ext (16 * (k1 - k0)) (fun k -> wrf rw wo kk f (16 * k0 + k) *. xrf xc xr (16 * k0 + k) t)
                            (pterm rw xr wo kk f t (16 * k0))
#pop-options

(* entry of a linear layer as a sum *)
#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
noextract let linear_entry (rows outs inner:nat) (w x:S.seq real) (t f:nat)
  : Lemma (requires t < rows /\ f < outs)
          (ensures at (linear rows outs inner w x) (t * outs + f) ==
                   rs inner (lterm w x inner f t))
  = reveal_opaque (`%linear) linear;
    let i = t * outs + f in
    FStar.Math.Lemmas.lemma_mult_le_right outs t (rows - 1);
    FStar.Math.Lemmas.lemma_div_plus f t outs;
    FStar.Math.Lemmas.small_div f outs;
    FStar.Math.Lemmas.lemma_mod_plus f t outs;
    FStar.Math.Lemmas.small_mod f outs;
    assert (i / outs == t /\ i % outs == f);
    sdot_rs inner w (f * inner) 1 x (t * inner) 1;
    rs_ext inner (fun k -> at w (f * inner + k * 1) *. at x (t * inner + k * 1))
                 (lterm w x inner f t)
#pop-options

noextract let at_window (s:S.seq real) (o n j:int)
  : Lemma (requires 0 <= o /\ 0 <= n /\ o + n <= S.length s /\ 0 <= j /\ j < n)
          (ensures at (window s o n) j == at s (o + j))
  = ()

noextract let seq_ap (#n:nat) (c:chest1 bf16 n) (r:S.seq real)
  : Lemma (requires S.length r == n /\ (forall (i:nat). i < n ==> acc1 c i %~ at r i))
          (ensures chest1_to_seq c %~ r)
  = ()

noextract let seq_ap_elim (#n:nat) (c:chest1 bf16 n) (r:S.seq real)
  : Lemma (requires chest1_to_seq c %~ r)
          (ensures S.length r == n /\ (forall (i:nat). i < n ==> acc1 c i %~ S.index r i))
  = assert (forall (i:nat). i < n ==> S.index (chest1_to_seq c) i == acc1 c i)

noextract let divmod (t f r:nat)
  : Lemma (requires f < r) (ensures (t * r + f) / r == t /\ (t * r + f) % r == f)
  = FStar.Math.Lemmas.lemma_div_plus f t r;
    FStar.Math.Lemmas.small_div f r;
    FStar.Math.Lemmas.lemma_mod_plus f t r;
    FStar.Math.Lemmas.small_mod f r

noextract let win_row (rw:S.seq real) (wo r kk f:nat)
  : Lemma (requires wo + r * kk <= S.length rw /\ f < r)
          (ensures forall (k:nat). k < kk ==> at (window rw wo (r * kk)) (f * kk + k) == at rw (wo + f * kk + k))
  = FStar.Math.Lemmas.lemma_mult_le_right kk f (r - 1);
    let aux (k:nat{k < kk}) : Lemma (at (window rw wo (r * kk)) (f * kk + k) == at rw (wo + f * kk + k)) =
      at_window rw wo (r * kk) (f * kk + k) in
    Classical.forall_intro aux

#push-options "--fuel 0 --ifuel 0 --z3rlimit 60"
noextract let proj_full (#rt #kt:pos) (#n0:nat) (s0:chest1 bf16 n0) (wo:nat) (sx:chest1 bf16 (7 * (16 * kt)))
  (wc:chest2 bf16 (16 * rt) (16 * kt)) (xc:chest2 bf16 (16 * kt) 8) (c0:chest2 float (16 * rt) 8)
  (out:chest1 bf16 (7 * (16 * rt))) (rw xr:S.seq real) (i:natlt (7 * (16 * rt)))
  : Lemma (requires wo + (16 * rt) * (16 * kt) <= n0 /\
             S.length rw == n0 /\ S.length xr == 7 * (16 * kt) /\
             (forall (i:natlt (16 * rt)) (k:natlt (16 * kt)). acc2 wc i k == bat s0 (wo + i * (16 * kt) + k)) /\
             (forall (k:natlt (16 * kt)) (t:nat{t < 7}). acc2 xc k t == bat sx (t * (16 * kt) + k)) /\
             (forall (i:nat). i < n0 ==> acc1 s0 i %~ S.index rw i) /\
             (forall (i:nat). i < 7 * (16 * kt) ==> acc1 sx i %~ S.index xr i) /\
             (forall (i:natlt (7 * (16 * rt))).
                acc1 out i == (fcast (acc2 (cmod wc xc c0 0 kt rt) (i % (16 * rt)) (i / (16 * rt))) <: bf16)))
          (ensures acc1 out i %~
                   at (linear 7 (16 * rt) (16 * kt) (window rw wo ((16 * rt) * (16 * kt))) xr) i)
  = let r = 16 * rt in
    let kk = 16 * kt in
    let t = i / r in
    let f = i % r in
    FStar.Math.Lemmas.euclidean_division_definition i r;
    proj_chain s0 wo sx wc xc c0 0 kt rw xr f t;
    linear_entry 7 r kk (window rw wo (r * kk)) xr t f;
    win_row rw wo r kk f;
    assert (16 * (kt - 0) == kk);
    let c = acc2 (cmod wc xc c0 0 kt rt) f t in
    assert (acc1 out i == (fcast c <: bf16));
    rs_ext kk (pterm rw xr wo kk f t 0) (lterm (window rw wo (r * kk)) xr kk f t)

noextract let proj_split (#rt #kt:pos) (#n0:nat) (s0:chest1 bf16 n0) (wo:nat) (sx:chest1 bf16 (7 * (16 * kt)))
  (wc:chest2 bf16 (16 * rt) (16 * kt)) (xc:chest2 bf16 (16 * kt) 8) (c0 c0':chest2 float (16 * rt) 8)
  (kh:nat{kh <= kt}) (out:chest1 bf16 (7 * (16 * rt))) (rw xr:S.seq real) (i:natlt (7 * (16 * rt)))
  : Lemma (requires wo + (16 * rt) * (16 * kt) <= n0 /\
             S.length rw == n0 /\ S.length xr == 7 * (16 * kt) /\
             (forall (i:natlt (16 * rt)) (k:natlt (16 * kt)). acc2 wc i k == bat s0 (wo + i * (16 * kt) + k)) /\
             (forall (k:natlt (16 * kt)) (t:nat{t < 7}). acc2 xc k t == bat sx (t * (16 * kt) + k)) /\
             (forall (i:nat). i < n0 ==> acc1 s0 i %~ S.index rw i) /\
             (forall (i:nat). i < 7 * (16 * kt) ==> acc1 sx i %~ S.index xr i) /\
             (forall (i:natlt (7 * (16 * rt))).
                acc1 out i == (fcast (add (acc2 (cmod wc xc c0 0 kh rt) (i % (16 * rt)) (i / (16 * rt)))
                                          (acc2 (cmod wc xc c0' kh kt rt) (i % (16 * rt)) (i / (16 * rt)))) <: bf16)))
          (ensures acc1 out i %~
                   at (linear 7 (16 * rt) (16 * kt) (window rw wo ((16 * rt) * (16 * kt))) xr) i)
  = let r = 16 * rt in
    let kk = 16 * kt in
    let t = i / r in
    let f = i % r in
    FStar.Math.Lemmas.euclidean_division_definition i r;
    proj_chain s0 wo sx wc xc c0 0 kh rw xr f t;
    proj_chain s0 wo sx wc xc c0' kh kt rw xr f t;
    linear_entry 7 r kk (window rw wo (r * kk)) xr t f;
    win_row rw wo r kk f;
    assert (16 * kh + 16 * (kt - kh) == kk);
    let g = lterm (window rw wo (r * kk)) xr kk f t in
    rs_split (16 * kh) (16 * (kt - kh)) g;
    rs_ext (16 * kh) (pterm rw xr wo kk f t 0) g;
    rs_ext (16 * (kt - kh)) (pterm rw xr wo kk f t (16 * kh)) (fun k -> g (16 * kh + k))
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 60"
noextract let proj_stage (#rt #kt:pos) (#n0:nat) (s0:chest1 bf16 n0) (wo:nat) (sx:chest1 bf16 (7 * (16 * kt)))
  (wc:chest2 bf16 (16 * rt) (16 * kt)) (xc:chest2 bf16 (16 * kt) 8) (c0:chest2 float (16 * rt) 8)
  (out:chest1 bf16 (7 * (16 * rt))) (rw xr:S.seq real)
  : Lemma (requires wo + (16 * rt) * (16 * kt) <= n0 /\
             chest1_to_seq s0 %~ rw /\ chest1_to_seq sx %~ xr /\
             (forall (i:natlt (16 * rt)) (k:natlt (16 * kt)). acc2 wc i k == bat s0 (wo + i * (16 * kt) + k)) /\
             (forall (k:natlt (16 * kt)) (t:nat{t < 7}). acc2 xc k t == bat sx (t * (16 * kt) + k)) /\
             (forall (i:natlt (7 * (16 * rt))).
                acc1 out i == (fcast (acc2 (cmod wc xc c0 0 kt rt) (i % (16 * rt)) (i / (16 * rt))) <: bf16)))
          (ensures chest1_to_seq out %~
                   linear 7 (16 * rt) (16 * kt) (window rw wo ((16 * rt) * (16 * kt))) xr)
  = seq_ap_elim s0 rw;
    seq_ap_elim sx xr;
    let aux (i:natlt (7 * (16 * rt)))
      : Lemma (acc1 out i %~ at (linear 7 (16 * rt) (16 * kt) (window rw wo ((16 * rt) * (16 * kt))) xr) i)
      = proj_full s0 wo sx wc xc c0 out rw xr i in
    Classical.forall_intro aux;
    seq_ap out (linear 7 (16 * rt) (16 * kt) (window rw wo ((16 * rt) * (16 * kt))) xr)

noextract let proj_stage_split (#rt #kt:pos) (#n0:nat) (s0:chest1 bf16 n0) (wo:nat) (sx:chest1 bf16 (7 * (16 * kt)))
  (wc:chest2 bf16 (16 * rt) (16 * kt)) (xc:chest2 bf16 (16 * kt) 8) (c0 c0':chest2 float (16 * rt) 8)
  (kh:nat{kh <= kt}) (out:chest1 bf16 (7 * (16 * rt))) (rw xr:S.seq real)
  : Lemma (requires wo + (16 * rt) * (16 * kt) <= n0 /\
             chest1_to_seq s0 %~ rw /\ chest1_to_seq sx %~ xr /\
             (forall (i:natlt (16 * rt)) (k:natlt (16 * kt)). acc2 wc i k == bat s0 (wo + i * (16 * kt) + k)) /\
             (forall (k:natlt (16 * kt)) (t:nat{t < 7}). acc2 xc k t == bat sx (t * (16 * kt) + k)) /\
             (forall (i:natlt (7 * (16 * rt))).
                acc1 out i == (fcast (add (acc2 (cmod wc xc c0 0 kh rt) (i % (16 * rt)) (i / (16 * rt)))
                                          (acc2 (cmod wc xc c0' kh kt rt) (i % (16 * rt)) (i / (16 * rt)))) <: bf16)))
          (ensures chest1_to_seq out %~
                   linear 7 (16 * rt) (16 * kt) (window rw wo ((16 * rt) * (16 * kt))) xr)
  = seq_ap_elim s0 rw;
    seq_ap_elim sx xr;
    let aux (i:natlt (7 * (16 * rt)))
      : Lemma (acc1 out i %~ at (linear 7 (16 * rt) (16 * kt) (window rw wo ((16 * rt) * (16 * kt))) xr) i)
      = proj_split s0 wo sx wc xc c0 c0' kh out rw xr i in
    Classical.forall_intro aux;
    seq_ap out (linear 7 (16 * rt) (16 * kt) (window rw wo ((16 * rt) * (16 * kt))) xr)
#pop-options

(**** ResNorm ****)

open FStar.Real
open Kuiper
open Kuiper.Chest
open Kuiper.Float.Casts
open KPlay.SharedSpec

noextract let sum_m (sr su:chest1 bf16 7168) (i:nat) : GTot bf16 =
  b16 (add (f32 (bat sr i)) (f32 (bat su i)))

noextract let rn_inv (sr su:chest1 bf16 7168) (row:nat) : GTot float =
  rms_model 1024 (fun c -> sum_m sr su (row * 1024 + c))

noextract let rn_m (#n0:nat) (sw:chest1 bf16 n0) (noff:nat) (sr su:chest1 bf16 7168) (i:nat) : GTot bf16 =
  b16 (mul (f32 (b16 (mul (f32 (sum_m sr su i)) (rn_inv sr su (i / 1024)))))
           (f32 (bat sw (noff + i % 1024))))

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
noextract let sum_approx (sr su:chest1 bf16 7168) (rr ur:S.seq real) (i:nat{i < 7168})
  : Lemma (requires S.length rr == 7168 /\ S.length ur == 7168 /\
             (forall (i:nat). i < 7168 ==> acc1 sr i %~ S.index rr i) /\
             (forall (i:nat). i < 7168 ==> acc1 su i %~ S.index ur i))
          (ensures sum_m sr su i %~ at (vadd rr ur) i)
  = assert (f32 (bat sr i) %~ S.index rr i);
    assert (f32 (bat su i) %~ S.index ur i);
    assert (add (f32 (bat sr i)) (f32 (bat su i)) %~ (S.index rr i +. S.index ur i))

noextract let rn_approx (#n0:nat) (sw:chest1 bf16 n0) (noff:nat) (sr su:chest1 bf16 7168)
  (rw rr ur:S.seq real) (i:nat{i < 7168})
  : Lemma (requires S.length rr == 7168 /\ S.length ur == 7168 /\ S.length rw == n0 /\ noff + 1024 <= n0 /\
             (forall (i:nat). i < 7168 ==> acc1 sr i %~ S.index rr i) /\
             (forall (i:nat). i < 7168 ==> acc1 su i %~ S.index ur i) /\
             (forall (i:nat). i < n0 ==> acc1 sw i %~ S.index rw i) /\
             rn_inv sr su (i / 1024) %~ rms_inv 1024 (vadd rr ur) ((i / 1024) * 1024))
          (ensures rn_m sw noff sr su i %~ at (rmsnorm_rows 7 1024 (vadd rr ur) (window rw noff 1024)) i)
  = reveal_opaque (`%rmsnorm_rows) rmsnorm_rows;
    sum_approx sr su rr ur i;
    let c = i % 1024 in
    let x = sum_m sr su i in
    let xr = at (vadd rr ur) i in
    let inv = rn_inv sr su (i / 1024) in
    let ir = rms_inv 1024 (vadd rr ur) ((i / 1024) * 1024) in
    let w = bat sw (noff + c) in
    let wr = at (window rw noff 1024) c in
    assert (w %~ wr);
    assert (f32 x %~ xr);
    assert (f32 w %~ wr);
    assert (mul (f32 x) inv %~ (xr *. ir));
    assert (f32 (b16 (mul (f32 x) inv)) %~ (xr *. ir));
    assert (mul (f32 (b16 (mul (f32 x) inv))) (f32 w) %~ ((xr *. ir) *. wr));
    assert (b16 (mul (f32 (b16 (mul (f32 x) inv))) (f32 w)) %~ ((xr *. ir) *. wr))
#pop-options

(**** Compose ****)

open FStar.Real
open Kuiper
open Kuiper.Chest
open KPlay.SharedSpec
module ML = FStar.Math.Lemmas

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
noextract let stack_entry (w x:S.seq real) (r kk f0 r' t f:nat)
  : Lemma (requires f0 + r' <= r /\ f0 <= f /\ f < f0 + r' /\ t < 7 /\ r * kk <= S.length w)
          (ensures at (linear 7 r kk w x) (t * r + f) ==
                   at (linear 7 r' kk (window w (f0 * kk) (r' * kk)) x) (t * r' + (f - f0)))
  = linear_entry 7 r kk w x t f;
    linear_entry 7 r' kk (window w (f0 * kk) (r' * kk)) x t (f - f0);
    ML.lemma_mult_le_right kk (f0 + r') r;
    let aux (k:nat{k < kk})
      : Lemma (lterm w x kk f t k == lterm (window w (f0 * kk) (r' * kk)) x kk (f - f0) t k)
      = ML.lemma_mult_le_right kk (f - f0) (r' - 1);
        ML.distributivity_sub_left f f0 kk;
        at_window w (f0 * kk) (r' * kk) ((f - f0) * kk + k) in
    Classical.forall_intro aux;
    rs_ext kk (lterm w x kk f t) (lterm (window w (f0 * kk) (r' * kk)) x kk (f - f0) t)

noextract let sqterm (a:S.seq real) (o:int) (k:nat) : real = at a (o + k) *. at a (o + k)

noextract let rms_inv_ext (d:pos) (a b:S.seq real) (oa ob:int)
  : Lemma (requires forall (k:nat). k < d ==> at a (oa + k) == at b (ob + k))
          (ensures rms_inv d a oa == rms_inv d b ob)
  = sumsq_rs d a oa;
    sumsq_rs d b ob;
    rs_ext d (fun k -> at a (oa + k) *. at a (oa + k)) (fun k -> at b (ob + k) *. at b (ob + k))
#pop-options

noextract let qk_idx_gen (m c:nat)
  : Lemma (requires c < 128)
          (ensures (m * 128 + c) / 128 == m /\ (m * 128 + c) % 128 == c)
  = divmod m c 128

(* Rows of the stacked qkv real result; head block hoff (0 = q, 16 = k), nh heads, f0 = row offset *)
#push-options "--fuel 0 --ifuel 0 --z3rlimit 60"
noextract let norm_entry (rw rn:S.seq real) (off:nat) (nh:pos) (hoff f0:nat) (t h c:nat)
  : Lemma (requires off + 6293760 <= S.length rw /\ t < 7 /\ h < nh /\ c < 128 /\
             f0 == hoff * 128 /\ hoff + nh <= 24 /\ (hoff == 0 \/ hoff == 16) /\ (hoff == 0 ==> nh <= 16))
          (ensures (let qkvr = linear 7 4096 1024 (window rw (off + 2048) 4194304) rn in
                    let lin = linear 7 (nh * 128) 1024 (window rw (off + 2048 + f0 * 1024) (nh * 128 * 1024)) rn in
                    at (rmsnorm_rows (7 * nh) 128 lin (window rw (qk_noff off hoff) 128))
                       (t * (nh * 128) + h * 128 + c) ==
                    nrm_r qkvr rw off t (hoff + h) c))
  = let qkvr = linear 7 4096 1024 (window rw (off + 2048) 4194304) rn in
    let wsub = window rw (off + 2048 + f0 * 1024) (nh * 128 * 1024) in
    let lin = linear 7 (nh * 128) 1024 wsub rn in
    window_window rw (off + 2048) 4194304 (f0 * 1024) (nh * 128 * 1024);
    let j = t * (nh * 128) + h * 128 + c in
    reveal_opaque (`%rmsnorm_rows) rmsnorm_rows;
    assert (j == (t * nh + h) * 128 + c);
    ML.lemma_mult_le_right 128 (t * nh + h) (7 * nh - 1);
    qk_idx_gen (t * nh + h) c;
    let ent (c':nat{c' < 128})
      : Lemma (at lin (t * (nh * 128) + h * 128 + c') == at qkvr (t * 4096 + (hoff + h) * 128 + c'))
      = stack_entry (window rw (off + 2048) 4194304) rn 4096 1024 f0 (nh * 128) t (f0 + h * 128 + c') in
    Classical.forall_intro ent;
    rms_inv_ext 128 lin qkvr (t * (nh * 128) + h * 128) (t * 4096 + (hoff + h) * 128);
    at_window rw (qk_noff off hoff) 128 c
#pop-options

noextract let qk_idx (r h c:nat)
  : Lemma (requires r < 7 /\ h < 24 /\ c < 128)
          (ensures (let i = (r * 24 + h) * 128 + c in
                    i < 21504 /\ i / 3072 == r /\ (i / 128) % 24 == h /\ i % 128 == c))
  = let i = (r * 24 + h) * 128 + c in
    divmod (r * 24 + h) c 128;
    divmod r h 24;
    ML.division_multiplication_lemma i 128 24

noextract let qr_seq (qkvr rw rr:S.seq real) (off:nat) : S.seq real =
  S.init 21504 (fun (i:nat{i < 21504}) -> qk_r qkvr rw rr off i)

#push-options "--fuel 0 --ifuel 0 --z3rlimit 80"
noextract let jdec (nh:pos) (j:nat)
  : Lemma (j == (j / (nh * 128)) * (nh * 128) + ((j % (nh * 128)) / 128) * 128 + j % 128 /\
           (j % (nh * 128)) / 128 < nh)
  = let m = j % (nh * 128) in
    ML.euclidean_division_definition j (nh * 128);
    ML.euclidean_division_definition m 128;
    ML.modulo_modulo_lemma j 128 nh;
    assert (128 * nh == nh * 128);
    ML.lemma_div_lt_nat m 7 128 |> (fun _ -> ());
    ML.lemma_div_le m (nh * 128 - 1) 128;
    ML.lemma_div_plus 127 (nh - 1) 128;
    ML.small_div 127 128

noextract let part_entry (rw rn rr:S.seq real) (off:nat) (nh:pos) (hoff f0:nat) (j:nat)
  : Lemma (requires off + 6293760 <= S.length rw /\ S.length rr == 1792 /\ j < 7 * nh * 128 /\
             f0 == hoff * 128 /\ hoff + nh <= 24 /\ (hoff == 0 \/ hoff == 16) /\ (hoff == 0 ==> nh <= 16))
          (ensures (let qkvr = linear 7 4096 1024 (window rw (off + 2048) 4194304) rn in
                    let lin = linear 7 (nh * 128) 1024 (window rw (off + 2048 + f0 * 1024) (nh * 128 * 1024)) rn in
                    let rn' = rmsnorm_rows (7 * nh) 128 lin (window rw (qk_noff off hoff) 128) in
                    at (rope nh rn' (window rr 0 896) (window rr 896 896)) j ==
                    at (qr_seq qkvr rw rr off)
                       (((j / (nh * 128)) * 24 + hoff + (j % (nh * 128)) / 128) * 128 + j % 128)))
  = let qkvr = linear 7 4096 1024 (window rw (off + 2048) 4194304) rn in
    reveal_opaque (`%rope) rope;
    jdec nh j;
    let t = j / (nh * 128) in
    let h = (j % (nh * 128)) / 128 in
    let c = j % 128 in
    ML.lemma_div_lt_nat j 7 (nh * 128) |> (fun _ -> ());
    assert (t < 7);
    norm_entry rw rn off nh hoff f0 t h c;
    (if c < 64 then norm_entry rw rn off nh hoff f0 t h (c + 64)
     else norm_entry rw rn off nh hoff f0 t h (c - 64));
    qk_idx t (hoff + h) c;
    at_window rr 0 896 (t * 128 + c);
    at_window rr 896 896 (t * 128 + c)
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 80"
noextract let qpart_eq (rw rn rr:S.seq real) (off:nat)
  : Lemma (requires off + 6293760 <= S.length rw /\ S.length rr == 1792)
          (ensures (let qkvr = linear 7 4096 1024 (window rw (off + 2048) 4194304) rn in
                    qpart (qr_seq qkvr rw rr off) ==
                    queries (packed_layer rw off) (window rr 0 896) (window rr 896 896) rn))
  = let qkvr = linear 7 4096 1024 (window rw (off + 2048) 4194304) rn in
    let a = qpart (qr_seq qkvr rw rr off) in
    let b = queries (packed_layer rw off) (window rr 0 896) (window rr 896 896) rn in
    let aux (j:nat{j < 14336}) : Lemma (S.index a j == S.index b j) =
      part_entry rw rn rr off 16 0 0 j in
    Classical.forall_intro aux;
    S.lemma_eq_intro a b

noextract let kpart_eq (rw rn rr:S.seq real) (off:nat)
  : Lemma (requires off + 6293760 <= S.length rw /\ S.length rr == 1792)
          (ensures (let qkvr = linear 7 4096 1024 (window rw (off + 2048) 4194304) rn in
                    kpart (qr_seq qkvr rw rr off) ==
                    keys (packed_layer rw off) (window rr 0 896) (window rr 896 896) rn))
  = let qkvr = linear 7 4096 1024 (window rw (off + 2048) 4194304) rn in
    let a = kpart (qr_seq qkvr rw rr off) in
    let b = keys (packed_layer rw off) (window rr 0 896) (window rr 896 896) rn in
    let aux (j:nat{j < 7168}) : Lemma (S.index a j == S.index b j) =
      part_entry rw rn rr off 8 16 2048 j in
    Classical.forall_intro aux;
    S.lemma_eq_intro a b

noextract let vpart_eq (rw rn:S.seq real) (off:nat)
  : Lemma (requires off + 6293760 <= S.length rw)
          (ensures vpart (linear 7 4096 1024 (window rw (off + 2048) 4194304) rn) ==
                   linear 7 1024 1024 (packed_layer rw off).wv rn)
  = let w = window rw (off + 2048) 4194304 in
    let a = vpart (linear 7 4096 1024 w rn) in
    let b = linear 7 1024 1024 (packed_layer rw off).wv rn in
    window_window rw (off + 2048) 4194304 (3072 * 1024) (1024 * 1024);
    let aux (j:nat{j < 7168}) : Lemma (S.index a j == S.index b j) =
      divmod (j / 1024) (j % 1024) 1024;
      ML.euclidean_division_definition j 1024;
      stack_entry w rn 4096 1024 3072 1024 (j / 1024) (3072 + j % 1024) in
    Classical.forall_intro aux;
    S.lemma_eq_intro a b
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
noextract let qk_inv_ok (sv:chest1 bf16 28672) (vr:S.seq real) (row head:nat)
  : Lemma (requires row < 7 /\ head < 24 /\ chest1_to_seq sv %~ vr)
          (ensures qk_inv sv row head %~ rms_inv 128 vr (row * 4096 + head * 128))
  = seq_ap_elim sv vr;
    FStar.Math.Lemmas.lemma_mult_le_right 4096 row 6;
    FStar.Math.Lemmas.lemma_mult_le_right 128 head 23;
    let b : nat = row * 4096 + head * 128 in
    assert (b + 128 <= 28672);
    let e (i:nat) : Lemma (requires i < 128) (ensures bat sv (b + i) %~ at vr (b + i)) =
      assert (b + i < 28672) in
    Classical.forall_intro (Classical.move_requires e);
    rms_model_approx 128 (fun i -> bat sv (row * 4096 + head * 128 + i)) vr b

noextract let qk_stage (#n0:nat) (s0:chest1 bf16 n0) (off:nat) (sv:chest1 bf16 28672) (s3:chest1 bf16 1792)
  (sq:chest1 bf16 21504) (rw vr rr:S.seq real)
  : Lemma (requires off + 6293760 <= n0 /\
             chest1_to_seq s0 %~ rw /\ chest1_to_seq sv %~ vr /\ chest1_to_seq s3 %~ rr /\
             (forall (i:nat). i < 21504 ==> acc1 sq i == qk_m s0 off sv s3 i))
          (ensures chest1_to_seq sq %~ qr_seq vr rw rr off)
  = let inv' (row:nat) : Lemma (forall (head:nat). row < 7 /\ head < 24 ==>
                                  qk_inv sv row head %~ rms_inv 128 vr (row * 4096 + head * 128)) =
      Classical.forall_intro (Classical.move_requires (qk_inv_ok sv vr row)) in
    Classical.forall_intro inv';
    seq_ap_elim s0 rw;
    seq_ap_elim sv vr;
    seq_ap_elim s3 rr;
    let aux (i:nat{i < 21504}) : Lemma (acc1 sq i %~ at (qr_seq vr rw rr off) i) =
      qk_approx s0 off sv s3 vr rw rr i in
    Classical.forall_intro aux;
    seq_ap sq (qr_seq vr rw rr off)
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
noextract let rn_inv_ok (sr su:chest1 bf16 7168) (rr ur:S.seq real) (row:nat)
  : Lemma (requires row < 7 /\ chest1_to_seq sr %~ rr /\ chest1_to_seq su %~ ur)
          (ensures rn_inv sr su row %~ rms_inv 1024 (vadd rr ur) (row * 1024))
  = seq_ap_elim sr rr;
    seq_ap_elim su ur;
    FStar.Math.Lemmas.lemma_mult_le_right 1024 row 6;
    let b : nat = row * 1024 in
    assert (b + 1024 <= 7168);
    let e (c:nat) : Lemma (requires c < 1024) (ensures sum_m sr su (row * 1024 + c) %~ at (vadd rr ur) (b + c)) =
      sum_approx sr su rr ur (b + c) in
    Classical.forall_intro (Classical.move_requires e);
    rms_model_approx 1024 (fun c -> sum_m sr su (row * 1024 + c)) (vadd rr ur) b

noextract let rn_stage (#n0:nat) (sw:chest1 bf16 n0) (noff:nat) (sr su s1 s2:chest1 bf16 7168)
  (rw rr ur:S.seq real)
  : Lemma (requires noff + 1024 <= n0 /\
             chest1_to_seq sw %~ rw /\ chest1_to_seq sr %~ rr /\ chest1_to_seq su %~ ur /\
             (forall (i:nat). i < 7168 ==> acc1 s1 i == sum_m sr su i /\ acc1 s2 i == rn_m sw noff sr su i))
          (ensures chest1_to_seq s1 %~ vadd rr ur /\
                   chest1_to_seq s2 %~ rmsnorm_rows 7 1024 (vadd rr ur) (window rw noff 1024))
  = Classical.forall_intro (Classical.move_requires (rn_inv_ok sr su rr ur));
    seq_ap_elim sw rw;
    seq_ap_elim sr rr;
    seq_ap_elim su ur;
    let a1 (i:nat{i < 7168}) : Lemma (acc1 s1 i %~ at (vadd rr ur) i) =
      sum_approx sr su rr ur i in
    let a2 (i:nat{i < 7168}) : Lemma (acc1 s2 i %~ at (rmsnorm_rows 7 1024 (vadd rr ur) (window rw noff 1024)) i) =
      assert (i / 1024 < 7);
      rn_approx sw noff sr su rw rr ur i in
    Classical.forall_intro a1;
    Classical.forall_intro a2;
    seq_ap s1 (vadd rr ur);
    seq_ap s2 (rmsnorm_rows 7 1024 (vadd rr ur) (window rw noff 1024))
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 60"
noextract let gu_idx (i:nat{i < 21504})
  : Lemma (let t = i / 3072 in let f = i % 3072 in
           t < 7 /\ i == t * 3072 + f /\
           (i / 3072) * 6144 + i % 3072 == t * 6144 + f /\
           (i / 3072) * 6144 + 3072 + i % 3072 == t * 6144 + (f + 3072) /\
           t * 6144 + (f + 3072) < 43008)
  = ML.euclidean_division_definition i 3072;
    ML.lemma_mult_le_right 6144 (i / 3072) 6

noextract let silu_stage (sa:chest1 bf16 21504) (w n:S.seq real)
  : Lemma (requires S.length w == 6291456 /\
             (forall (i:nat). i < 21504 ==>
                acc1 sa i %~ (silu (at (linear 7 6144 1024 w n) ((i / 3072) * 6144 + i % 3072)) *.
                              at (linear 7 6144 1024 w n) ((i / 3072) * 6144 + 3072 + i % 3072))))
          (ensures chest1_to_seq sa %~
                   silu_product (linear 7 3072 1024 (window w 0 3145728) n)
                                (linear 7 3072 1024 (window w 3145728 3145728) n))
  = reveal_opaque (`%silu_product) silu_product;
    let g = linear 7 3072 1024 (window w 0 3145728) n in
    let u = linear 7 3072 1024 (window w 3145728 3145728) n in
    let aux (i:nat{i < 21504}) : Lemma (acc1 sa i %~ at (silu_product g u) i) =
      gu_idx i;
      let t = i / 3072 in
      let f = i % 3072 in
      stack_entry w n 6144 1024 0 3072 t f;
      stack_entry w n 6144 1024 3072 3072 t (f + 3072);
      assert (at g i == S.index g i);
      assert (at (silu_product g u) i == silu (S.index g i) *. at u i) in
    Classical.forall_intro aux;
    seq_ap sa (silu_product g u)
#pop-options

(**** RmsDev ****)


open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Float.Casts
module SZ = Kuiper.SizeT
module U32 = FStar.UInt32
module GR = Pulse.Lib.GhostReference

noextract
let xof (#n:nat) (s:chest1 bf16 n) (o:nat) (i:nat) : GTot bf16 = bat s (o + i)

noextract
let size_i64 (size:SZ.t) : Lemma (requires SZ.v size == 128 \/ SZ.v size == 1024)
  (ensures FStar.Int.Cast.uint32_to_int64 (SZ.sizet_to_u32 size) == i64n (SZ.v size))
  = I64.v_inj (FStar.Int.Cast.uint32_to_int64 (SZ.sizet_to_u32 size)) (i64n (SZ.v size))

noextract
let xof_lam (#n:nat) (s:chest1 bf16 n) (o:nat) (size:nat)
  : Lemma (requires size == 128 \/ size == 1024)
          (ensures rms_model size (xof s o) == rms_model size (fun i -> bat s (o + i)))
  = rms_model_ext size (xof s o) (fun i -> bat s (o + i))

noextract
let acc1_upd (#et:Type) (#d:nat) (c:chest1 et d) (j:natlt d) (v:et) (i:natlt d)
  : Lemma (acc1 (Kuiper.Chest.upd c (j, ()) v) i == (if i = j then v else acc1 c i))
          [SMTPat (acc1 (Kuiper.Chest.upd c (j, ()) v) i)]
  = ()

#push-options "--z3rlimit 40 --fuel 1 --ifuel 1"
(* partial[lane] of one lane: four strided sums, combined ((s0+s1)+s2)+s3 *)
inline_for_extraction noextract
fn lane_dev
  (#n : nat)
  (src : array1 bf16 (l1_forward n))
  (o size lv st : SZ.t)
  (#ps : perm) (#s : chest1 bf16 n)
  requires pure ((SZ.v size == 128 \/ SZ.v size == 1024) /\ SZ.v o + SZ.v size <= n /\ SZ.fits n /\
                 SZ.v lv < rwidth (SZ.v size) /\ SZ.v st == 4 * rwidth (SZ.v size))
  preserves gpu ** src |-> Frac ps s
  returns r : float
  ensures pure (r == lanef (xof s (SZ.v o)) (SZ.v size) (SZ.v lv))
{
  let sz : erased nat = hide #nat (SZ.v size);
  let mut s0 : float = zero;
  let mut s1 : float = zero;
  let mut s2 : float = zero;
  let mut s3 : float = zero;
  let kg = GR.alloc #nat 0;
  let mut start : SZ.t = lv *^ 4sz;
  while (!start <^ size)
    invariant exists* (sv : SZ.t) (a0 a1 a2 a3 : float) (k : nat).
      start |-> sv ** s0 |-> a0 ** s1 |-> a1 ** s2 |-> a2 ** s3 |-> a3 ** GR.pts_to kg k **
      pure (SZ.v sv == SZ.v lv * 4 + k * (4 * rwidth sz) /\ k <= rnit sz /\
            SZ.v sv < SZ.v size + 4 * rwidth sz /\
            a0 == lsum (xof s (SZ.v o)) (4 * rwidth sz) (SZ.v lv) 0 k /\
            a1 == lsum (xof s (SZ.v o)) (4 * rwidth sz) (SZ.v lv) 1 k /\
            a2 == lsum (xof s (SZ.v o)) (4 * rwidth sz) (SZ.v lv) 2 k /\
            a3 == lsum (xof s (SZ.v o)) (4 * rwidth sz) (SZ.v lv) 3 k)
    decreases (SZ.v size + 1024 - SZ.v !start)
  {
    let sv = !start;
    let k = GR.read kg;
    start_step sz (SZ.v lv) k;
    lsum_step (xof s (SZ.v o)) (4 * rwidth sz) (SZ.v lv) 0 k;
    lsum_step (xof s (SZ.v o)) (4 * rwidth sz) (SZ.v lv) 1 k;
    lsum_step (xof s (SZ.v o)) (4 * rwidth sz) (SZ.v lv) 2 k;
    lsum_step (xof s (SZ.v o)) (4 * rwidth sz) (SZ.v lv) 3 k;
    let b = o +^ sv;
    let v0 : float = fcast (Kuiper.Tensor.read src (((b +^ 0sz) <: szlt n), ()) #ps);
    let c0 = !s0;
    s0 := add c0 (mul v0 v0);
    let v1 : float = fcast (Kuiper.Tensor.read src (((b +^ 1sz) <: szlt n), ()) #ps);
    let c1 = !s1;
    s1 := add c1 (mul v1 v1);
    let v2 : float = fcast (Kuiper.Tensor.read src (((b +^ 2sz) <: szlt n), ()) #ps);
    let c2 = !s2;
    s2 := add c2 (mul v2 v2);
    let v3 : float = fcast (Kuiper.Tensor.read src (((b +^ 3sz) <: szlt n), ()) #ps);
    let c3 = !s3;
    s3 := add c3 (mul v3 v3);
    GR.write kg (k + 1);
    start := sv +^ st;
  };
  let kf = GR.read kg;
  start_exit sz (SZ.v lv) kf;
  GR.free kg;
  let t0 = !s0;
  let t1 = !s1;
  let t2 = !s2;
  let t3 = !s3;
  add (add (add t0 t1) t2) t3
}

(* in-place halving tree over partial[0 .. pow2 p) *)
inline_for_extraction noextract
fn tree_dev
  (p : erased nat)
  (f : erased (nat -> GTot float))
  (width : SZ.t)
  (partial : array1 float (l1_forward 128))
  (#sp : chest1 float 128)
  requires partial |-> sp **
    pure (p <= 7 /\ SZ.v width == pow2 p /\
          (forall (i:nat). i < pow2 p ==> i < 128 /\ acc1 sp i == reveal f i))
  returns r : float
  ensures exists* sp'. partial |-> sp' ** pure (r == hk p f p 0)
{
  tree_init p;
  pow2_half p;
  let kt = GR.alloc #nat 0;
  let mut off : SZ.t = width /^ 2sz;
  while (!off >^ 0sz)
    invariant exists* (ov : SZ.t) (pv : chest1 float 128) (k : nat).
      off |-> ov ** partial |-> pv ** GR.pts_to kt k **
      pure (k <= p /\ SZ.v ov == pow2 (p - k) / 2 /\
            (forall (i:nat). i < pow2 (p - k) ==> i < 128 /\ acc1 pv i == hk p f k i))
    decreases (SZ.v !off)
  {
    let ov = !off;
    let k = GR.read kt;
    tree_lvl p k (SZ.v ov);
    let mut l2 : SZ.t = 0sz;
    while (!l2 <^ ov)
      invariant exists* (lv : SZ.t) (pv : chest1 float 128).
        l2 |-> lv ** partial |-> pv **
        pure (SZ.v lv <= SZ.v ov /\
              (forall (i:nat). i < pow2 (p - k) ==>
                 i < 128 /\ acc1 pv i == (if i < SZ.v lv then hk p f (k + 1) i else hk p f k i)))
      decreases (SZ.v ov - SZ.v !l2)
    {
      let lv = !l2;
      hk_step p f k (SZ.v lv);
      let a = Kuiper.Tensor.read partial ((lv <: szlt 128), ());
      let b = Kuiper.Tensor.read partial (((lv +^ ov) <: szlt 128), ());
      Kuiper.Tensor.write partial ((lv <: szlt 128), ()) (add a b);
      l2 := lv +^ 1sz;
    };
    GR.write kt (k + 1);
    off := ov /^ 2sz;
  };
  let kf = GR.read kt;
  tree_end p kf;
  GR.free kt;
  Kuiper.Tensor.read partial ((0sz <: szlt 128), ())
}

inline_for_extraction noextract
fn rms_dev
  (#n : nat)
  (src : array1 bf16 (l1_forward n))
  (o size : SZ.t)
  (partial : array1 float (l1_forward 128))
  (#ps : perm) (#s : chest1 bf16 n) (#sp : chest1 float 128)
  requires pure ((SZ.v size == 128 \/ SZ.v size == 1024) /\ SZ.v o + SZ.v size <= n)
  preserves gpu ** src |-> Frac ps s
  requires partial |-> sp
  returns r : float
  ensures exists* sp'. partial |-> sp' ** pure (r == rms_model (SZ.v size) (fun i -> bat s (SZ.v o + i)))
{
  tensor_pts_to_ref src;
  let sz : erased nat = hide #nat (SZ.v size);
  let width : SZ.t = (if size = 1024sz then 128sz else 32sz);
  assert (pure (SZ.v width == rwidth sz));
  pow2_vals ();
  let st : SZ.t = width *^ 4sz;
  let mut lane : SZ.t = 0sz;
  while (!lane <^ width)
    invariant exists* (lv : SZ.t) (pv : chest1 float 128).
      lane |-> lv ** partial |-> pv **
      pure (SZ.v lv <= SZ.v width /\
            (forall (i:nat). i < SZ.v lv ==> acc1 pv i == lanef (xof s (SZ.v o)) sz i))
    decreases (SZ.v width - SZ.v !lane)
  {
    let lv = !lane;
    let pl = lane_dev src o size lv st #ps #s;
    Kuiper.Tensor.write partial ((lv <: szlt 128), ()) pl;
    lane := lv +^ 1sz;
  };
  let tot = tree_dev (hide (rlog sz)) (hide (lanef (xof s (SZ.v o)) sz)) width partial;
  size_i64 size;
  rms_model_unfold sz (xof s (SZ.v o)) tot;
  xof_lam s (SZ.v o) sz;
  (* held in locals so that karamel does not constant-fold float ops on literals (it crashes) *)
  let mut szc : I64.t = FStar.Int.Cast.uint32_to_int64 (SZ.sizet_to_u32 size);
  let mut epc : I64.t = 1000000L;
  let szv = !szc;
  let epv = !epc;
  let mean = mul tot (div one (of_int szv));
  rsqrt (add mean (div one (of_int epv)))
}
#pop-options

(**** Kern ****)


open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Float.Casts
open KPlay.SharedSpec

noextract let kq_idx (r h c:nat)
  : Lemma (requires r < 7 /\ h < 24 /\ c < 128)
          (ensures (let i = (r * 24 + h) * 128 + c in
                    i < 21504 /\ i / 3072 == r /\ (i / 128) % 24 == h /\ i % 128 == c))
  = let i = (r * 24 + h) * 128 + c in
    FStar.Math.Lemmas.lemma_div_plus c (r * 24 + h) 128;
    FStar.Math.Lemmas.small_div c 128;
    FStar.Math.Lemmas.lemma_mod_plus c (r * 24 + h) 128;
    FStar.Math.Lemmas.small_mod c 128;
    FStar.Math.Lemmas.lemma_mod_plus h r 24;
    FStar.Math.Lemmas.small_mod h 24;
    FStar.Math.Lemmas.division_multiplication_lemma i 128 24;
    FStar.Math.Lemmas.lemma_div_plus h r 24;
    FStar.Math.Lemmas.small_div h 24

noextract let acc1_upd_k (#et : Type) (#m : nat) (d : chest1 et m) (i : natlt m) (v : et) (i' : natlt m)
  : Lemma (acc1 (upd d (i, ()) v) i' == (if i' = i then v else acc1 d i'))
          [SMTPat (acc1 (upd d (i, ()) v) i')]
  = ()

[@@"opaque_to_smt"]
noextract let qk_done (#n0:nat) (sw:chest1 bf16 n0) (off:nat) (sv:chest1 bf16 28672) (sr:chest1 bf16 1792)
  (sq:chest1 bf16 21504) (n:nat) : prop =
  forall (i:nat). i < n /\ i < 21504 ==> acc1 sq i == qk_m sw off sv sr i

[@@"opaque_to_smt"]
noextract let rn_done (#n0:nat) (sw:chest1 bf16 n0) (noff:nat) (sr su:chest1 bf16 7168)
  (s1 s2:chest1 bf16 7168) (n:nat) : prop =
  forall (i:nat). i < n /\ i < 7168 ==> acc1 s1 i == sum_m sr su i /\ acc1 s2 i == rn_m sw noff sr su i

noextract let row_idx2 (r c:nat)
  : Lemma (requires r < 7 /\ c < 1024)
          (ensures r * 1024 + c < 7168 /\ (r * 1024 + c) / 1024 == r /\ (r * 1024 + c) % 1024 == c)
  = FStar.Math.Lemmas.lemma_div_plus c r 1024;
    FStar.Math.Lemmas.small_div c 1024;
    FStar.Math.Lemmas.lemma_mod_plus c r 1024;
    FStar.Math.Lemmas.small_mod c 1024


#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
noextract let bat_in (#n:nat) (s:chest1 bf16 n) (i:nat)
  : Lemma (requires i < n) (ensures bat s i == acc1 s i) = ()

#push-options "--fuel 1 --ifuel 1 --z3rlimit 40"
noextract let done_upd (#m:nat) (q1:chest1 bf16 m) (f:nat -> GTot bf16) (i:nat{i < m}) (y:bf16)
  : Lemma (requires (forall (j:nat). j < i /\ j < m ==> acc1 q1 j == f j) /\ y == f i)
          (ensures (forall (j:nat). j < i + 1 /\ j < m ==> acc1 (upd q1 (i, ()) y) j == f j))
  = let aux (j:nat{j < i + 1 /\ j < m}) : Lemma (acc1 (upd q1 (i, ()) y) j == f j) =
      acc1_upd_k q1 i y j in
    Classical.forall_intro aux

noextract let done2_upd (#m:nat) (a1 a2:chest1 bf16 m) (f g:nat -> GTot bf16) (i:nat{i < m}) (y:bf16)
  : Lemma (requires (forall (j:nat). j < i /\ j < m ==> acc1 a1 j == f j /\ acc1 a2 j == g j) /\
                    acc1 a1 i == f i /\ y == g i)
          (ensures (forall (j:nat). j < i + 1 /\ j < m ==> acc1 a1 j == f j /\ acc1 (upd a2 (i, ()) y) j == g j))
  = let aux (j:nat{j < i + 1 /\ j < m}) : Lemma (acc1 a1 j == f j /\ acc1 (upd a2 (i, ()) y) j == g j) =
      acc1_upd_k a2 i y j in
    Classical.forall_intro aux

noextract let frame2_upd (#m:nat) (a1 a2:chest1 bf16 m) (f g:nat -> GTot bf16) (n:nat) (k:nat{k < m /\ n <= k}) (y:bf16)
  : Lemma (requires (forall (j:nat). j < n /\ j < m ==> acc1 a1 j == f j /\ acc1 a2 j == g j))
          (ensures (forall (j:nat). j < n /\ j < m ==> acc1 (upd a1 (k, ()) y) j == f j /\ acc1 a2 j == g j))
  = let aux (j:nat{j < n /\ j < m}) : Lemma (acc1 (upd a1 (k, ()) y) j == f j /\ acc1 a2 j == g j) =
      acc1_upd_k a1 k y j in
    Classical.forall_intro aux
#pop-options

noextract let qk_done_upd (#n0:nat) (sw:chest1 bf16 n0) (off:nat) (sv:chest1 bf16 28672) (sr:chest1 bf16 1792)
  (q1:chest1 bf16 21504) (i:nat{i < 21504}) (y:bf16)
  : Lemma (requires qk_done sw off sv sr q1 i /\ y == qk_m sw off sv sr i)
          (ensures qk_done sw off sv sr (upd q1 (i, ()) y) (i + 1))
  = reveal_opaque (`%qk_done) (qk_done sw off sv sr q1 i);
    reveal_opaque (`%qk_done) (qk_done sw off sv sr (upd q1 (i, ()) y) (i + 1));
    done_upd q1 (qk_m sw off sv sr) i y

noextract let qk_inv_eq (sv:chest1 bf16 28672) (r h:nat) (o:SZ.t)
  : Lemma (requires SZ.v o == r * 4096 + h * 128)
          (ensures rms_model (SZ.v 128sz) (fun i -> bat sv (SZ.v o + i)) == qk_inv sv r h)
  = rms_model_ext 128 (fun i -> bat sv (SZ.v o + i)) (fun i -> bat sv (r * 4096 + h * 128 + i))

noextract let rn_inv_eq (sr su a1:chest1 bf16 7168) (r:nat) (o:SZ.t)
  : Lemma (requires r < 7 /\ SZ.v o == r * 1024 /\
                    (forall (c:nat). c < 1024 ==> acc1 a1 (r * 1024 + c) == sum_m sr su (r * 1024 + c)))
          (ensures rms_model (SZ.v 1024sz) (fun i -> bat a1 (SZ.v o + i)) == rn_inv sr su r)
  = let aux (i:nat) : Lemma (i < 1024 ==> bat a1 (SZ.v o + i) == sum_m sr su (r * 1024 + i)) =
      if i < 1024 then bat_in a1 (SZ.v o + i) else () in
    Classical.forall_intro aux;
    rms_model_ext 1024 (fun i -> bat a1 (SZ.v o + i)) (fun c -> sum_m sr su (r * 1024 + c))

noextract let qk_done0 (#n0:nat) (sw:chest1 bf16 n0) (off:nat) (sv:chest1 bf16 28672) (sr:chest1 bf16 1792)
  (q1:chest1 bf16 21504)
  : Lemma (qk_done sw off sv sr q1 0)
  = reveal_opaque (`%qk_done) (qk_done sw off sv sr q1 0)

noextract let rn_done0 (#n0:nat) (sw:chest1 bf16 n0) (noff:nat) (sr su:chest1 bf16 7168) (a1 a2:chest1 bf16 7168)
  : Lemma (rn_done sw noff sr su a1 a2 0)
  = reveal_opaque (`%rn_done) (rn_done sw noff sr su a1 a2 0)

noextract let nrm_step (#n0:nat) (sw:chest1 bf16 n0) (off:nat) (sv:chest1 bf16 28672) (r h:nat)
  (n1:chest1 bf16 128) (cv:nat{cv < 128}) (y:bf16)
  : Lemma (requires (forall (c:nat). c < cv ==> acc1 n1 c == nrm_m sw off sv r h c) /\
                    y == nrm_m sw off sv r h cv)
          (ensures forall (c:nat). c < cv + 1 ==> acc1 (upd n1 (cv, ()) y) c == nrm_m sw off sv r h c)
  = ()

noextract let qk_step (#n0:nat) (sw:chest1 bf16 n0) (off:nat) (sv:chest1 bf16 28672) (sr:chest1 bf16 1792)
  (r h:nat) (n1:chest1 bf16 128) (q1:chest1 bf16 21504) (cv:nat{cv < 128}) (xn other:float) (cs sn y:bf16)
  : Lemma (requires r < 7 /\ h < 24 /\
             (forall (c:nat). c < 128 ==> acc1 n1 c == nrm_m sw off sv r h c) /\
             qk_done sw off sv sr q1 ((r * 24 + h) * 128 + cv) /\
             xn == f32 (acc1 n1 cv) /\
             other == (if cv < 64 then neg (f32 (acc1 n1 (cv + 64))) else f32 (acc1 n1 (cv - 64))) /\
             r * 128 + cv < 896 /\
             cs == acc1 sr (r * 128 + cv) /\ sn == acc1 sr (896 + r * 128 + cv) /\
             y == rot_m xn other cs sn)
          (ensures (r * 24 + h) * 128 + cv < 21504 /\
                   qk_done sw off sv sr (upd q1 ((r * 24 + h) * 128 + cv, ()) y) ((r * 24 + h) * 128 + cv + 1))
  = kq_idx r h cv;
    let i = (r * 24 + h) * 128 + cv in
    assert (i / 3072 == r /\ (i / 128) % 24 == h /\ i % 128 == cv);
    assert (xn == f32 (nrm_m sw off sv r h cv));
    let oth : float =
      if cv < 64 then neg (f32 (nrm_m sw off sv r h (cv + 64)))
      else f32 (nrm_m sw off sv r h (cv - 64)) in
    assert (other == oth);
    bat_in sr (r * 128 + cv);
    bat_in sr (896 + r * 128 + cv);
    assert (qk_m sw off sv sr i == rot_m (f32 (nrm_m sw off sv r h cv)) oth
              (bat sr (r * 128 + cv)) (bat sr (896 + r * 128 + cv)));
    assert (y == qk_m sw off sv sr i);
    qk_done_upd sw off sv sr q1 i y

noextract let sum_step (#n0:nat) (sw:chest1 bf16 n0) (noff:nat) (sr su:chest1 bf16 7168) (a1 a2:chest1 bf16 7168) (r cv:nat) (y:bf16)
  : Lemma (requires r < 7 /\ cv < 1024 /\ rn_done sw noff sr su a1 a2 (r * 1024) /\
             (forall (c:nat). c < cv ==> acc1 a1 (r * 1024 + c) == sum_m sr su (r * 1024 + c)) /\
             y == sum_m sr su (r * 1024 + cv))
          (ensures r * 1024 + cv < 7168 /\
                   (forall (c:nat). c < cv + 1 ==>
                      acc1 (upd a1 (r * 1024 + cv, ()) y) (r * 1024 + c) == sum_m sr su (r * 1024 + c)) /\
                   rn_done sw noff sr su (upd a1 (r * 1024 + cv, ()) y) a2 (r * 1024))
  = row_idx2 r cv;
    reveal_opaque (`%rn_done) (rn_done sw noff sr su a1 a2 (r * 1024));
    reveal_opaque (`%rn_done) (rn_done sw noff sr su (upd a1 (r * 1024 + cv, ()) y) a2 (r * 1024));
    frame2_upd a1 a2 (sum_m sr su) (rn_m sw noff sr su) (r * 1024) (r * 1024 + cv) y

noextract let rn_step (#n0:nat) (sw:chest1 bf16 n0) (noff:nat) (sr su:chest1 bf16 7168) (a1 a2:chest1 bf16 7168)
  (r cv:nat) (y:bf16)
  : Lemma (requires r < 7 /\ cv < 1024 /\
             (forall (c:nat). c < 1024 ==> acc1 a1 (r * 1024 + c) == sum_m sr su (r * 1024 + c)) /\
             rn_done sw noff sr su a1 a2 (r * 1024 + cv) /\
             y == rn_m sw noff sr su (r * 1024 + cv))
          (ensures r * 1024 + cv < 7168 /\
                   rn_done sw noff sr su a1 (upd a2 (r * 1024 + cv, ()) y) (r * 1024 + cv + 1))
  = row_idx2 r cv;
    assert (acc1 a1 (r * 1024 + cv) == sum_m sr su (r * 1024 + cv));
    reveal_opaque (`%rn_done) (rn_done sw noff sr su a1 a2 (r * 1024 + cv));
    reveal_opaque (`%rn_done) (rn_done sw noff sr su a1 (upd a2 (r * 1024 + cv, ()) y) (r * 1024 + cv + 1));
    done2_upd a1 a2 (sum_m sr su) (rn_m sw noff sr su) (r * 1024 + cv) y

noextract let qk_rows (#n0:nat) (sw:chest1 bf16 n0) (off:nat) (sv:chest1 bf16 28672) (sr:chest1 bf16 1792)
  (q1:chest1 bf16 21504) (r:nat)
  : Lemma (requires r < 7 /\ qk_done sw off sv sr q1 ((r * 24 + 24) * 128))
          (ensures qk_done sw off sv sr q1 ((r + 1) * 3072))
  = assert ((r * 24 + 24) * 128 == (r + 1) * 3072)

noextract let rn_rows (#n0:nat) (sw:chest1 bf16 n0) (noff:nat) (sr su:chest1 bf16 7168) (a1 a2:chest1 bf16 7168) (r:nat)
  : Lemma (requires r < 7 /\ rn_done sw noff sr su a1 a2 (r * 1024 + 1024))
          (ensures rn_done sw noff sr su a1 a2 ((r + 1) * 1024))
  = assert (r * 1024 + 1024 == (r + 1) * 1024)
#pop-options

#push-options "--z3rlimit 80 --fuel 0 --ifuel 0"
inline_for_extraction noextract
fn att_qk
  (#n0 : erased nat{SZ.fits n0})
  (w : array1 bf16 (l1_forward n0))
  (off : SZ.t{SZ.v off + 6293760 <= n0})
  (qkv : array1 bf16 (l1_forward 28672))
  (rot : array1 bf16 (l1_forward 1792))
  (qk : array1 bf16 (l1_forward 21504))
  (nrm : array1 bf16 (l1_forward 128))
  (partial : array1 float (l1_forward 128))
  (#pw #pv #pr : perm)
  (#sw : chest1 bf16 n0) (#sv : chest1 bf16 28672) (#sr : chest1 bf16 1792)
  (#sq : chest1 bf16 21504) (#sn : chest1 bf16 128) (#sp : chest1 float 128)
  preserves gpu
  preserves w |-> Frac pw sw
  preserves qkv |-> Frac pv sv
  preserves rot |-> Frac pr sr
  requires qk |-> sq
  requires nrm |-> sn
  requires partial |-> sp
  ensures exists* (sq' : chest1 bf16 21504) (sn' : chest1 bf16 128) (sp' : chest1 float 128).
    qk |-> sq' ** nrm |-> sn' ** partial |-> sp' **
    pure (qk_done sw (SZ.v off) sv sr sq' 21504)
{
  qk_done0 sw (SZ.v off) sv sr sq;
  let mut row : sz = 0sz;
  while (!row <^ 7sz)
    invariant exists* (rv : sz) (q1 : chest1 bf16 21504) (n1 : chest1 bf16 128) (p1 : chest1 float 128).
      row |-> rv ** qk |-> q1 ** nrm |-> n1 ** partial |-> p1 **
      pure (SZ.v rv <= 7 /\ qk_done sw (SZ.v off) sv sr q1 (SZ.v rv * 3072))
    decreases (7 - SZ.v !row)
  {
    let r = !row;
    let mut head : sz = 0sz;
    while (!head <^ 24sz)
      invariant exists* (hv : sz) (q1 : chest1 bf16 21504) (n1 : chest1 bf16 128) (p1 : chest1 float 128).
        head |-> hv ** qk |-> q1 ** nrm |-> n1 ** partial |-> p1 **
        pure (SZ.v hv <= 24 /\ qk_done sw (SZ.v off) sv sr q1 ((SZ.v r * 24 + SZ.v hv) * 128))
      decreases (24 - SZ.v !head)
    {
      let h = !head;
      FStar.Math.Lemmas.lemma_mult_le_right 4096 (SZ.v r) 6;
      FStar.Math.Lemmas.lemma_mult_le_right 128 (SZ.v h) 23;
      FStar.Math.Lemmas.lemma_mult_le_right 128 (SZ.v r) 6;
      FStar.Math.Lemmas.lemma_mult_le_right 24 (SZ.v r) 6;
      kq_idx (SZ.v r) (SZ.v h) 0;
      let base : sz = r *^ 4096sz +^ h *^ 128sz;
      let noff : sz = off +^ SZ.uint32_to_sizet 6293504ul +^ (if h <^ 16sz then 0sz else 128sz);
      assert pure (SZ.v noff == qk_noff (SZ.v off) (SZ.v h));
      let inv = rms_dev qkv base 128sz partial;
      qk_inv_eq sv (SZ.v r) (SZ.v h) base;
      assert pure (inv == qk_inv sv (SZ.v r) (SZ.v h));
      let mut col : sz = 0sz;
      while (!col <^ 128sz)
        invariant exists* (cv : sz) (n1 : chest1 bf16 128).
          col |-> cv ** nrm |-> n1 **
          pure (SZ.v cv <= 128 /\
            (forall (c:nat). c < SZ.v cv ==> acc1 n1 c == nrm_m sw (SZ.v off) sv (SZ.v r) (SZ.v h) c))
        decreases (128 - SZ.v !col)
      {
        let c = !col;
        let x : float = fcast (Kuiper.Tensor.read qkv (((base +^ c) <: szlt 28672), ()) #pv);
        let sc : bf16 = fcast (mul x inv);
        let wv : float = fcast (Kuiper.Tensor.read w (((noff +^ c) <: szlt n0), ()) #pw);
        let y : bf16 = fcast (mul (fcast sc <: float) wv);
        assert pure (SZ.v base + SZ.v c < 28672 /\ SZ.v noff + SZ.v c < n0);
        assert pure (y == nrm_m sw (SZ.v off) sv (SZ.v r) (SZ.v h) (SZ.v c));
        with n1. assert (nrm |-> n1);
        nrm_step sw (SZ.v off) sv (SZ.v r) (SZ.v h) n1 (SZ.v c) y;
        Kuiper.Tensor.write nrm ((c <: szlt 128), ()) y;
        col := c +^ 1sz;
      };
      let qbase : sz = (r *^ 24sz +^ h) *^ 128sz;
      col := 0sz;
      while (!col <^ 128sz)
        invariant exists* (cv : sz) (q1 : chest1 bf16 21504) (n1 : chest1 bf16 128).
          col |-> cv ** qk |-> q1 ** nrm |-> n1 **
          pure (SZ.v cv <= 128 /\
            (forall (c:nat). c < 128 ==> acc1 n1 c == nrm_m sw (SZ.v off) sv (SZ.v r) (SZ.v h) c) /\
            qk_done sw (SZ.v off) sv sr q1 ((SZ.v r * 24 + SZ.v h) * 128 + SZ.v cv))
        decreases (128 - SZ.v !col)
      {
        let c = !col;
        kq_idx (SZ.v r) (SZ.v h) (SZ.v c);
        let xn : float = fcast (Kuiper.Tensor.read nrm ((c <: szlt 128), ()));
        let other : float =
          if (c <^ 64sz) {
            let t0 = Kuiper.Tensor.read nrm (((c +^ 64sz) <: szlt 128), ());
            let t : float = fcast t0;
            neg t
          } else {
            let t0 = Kuiper.Tensor.read nrm (((c -^ 64sz) <: szlt 128), ());
            let t : float = fcast t0;
            t
          };
        let cs = Kuiper.Tensor.read rot (((r *^ 128sz +^ c) <: szlt 1792), ()) #pr;
        let sn = Kuiper.Tensor.read rot (((896sz +^ r *^ 128sz +^ c) <: szlt 1792), ()) #pr;
        let a : bf16 = fcast (mul xn (fcast cs <: float));
        let b : bf16 = fcast (mul other (fcast sn <: float));
        let y : bf16 = fcast (add (fcast a <: float) (fcast b <: float));
        assert pure (y == rot_m xn other cs sn);
        with n1 q1. assert (nrm |-> n1 ** qk |-> q1);
        qk_step sw (SZ.v off) sv sr (SZ.v r) (SZ.v h) n1 q1 (SZ.v c) xn other cs sn y;
        Kuiper.Tensor.write qk (((qbase +^ c) <: szlt 21504), ()) y;
        col := c +^ 1sz;
      };
      head := h +^ 1sz;
    };
    with q1. assert (qk |-> q1);
    qk_rows sw (SZ.v off) sv sr q1 (SZ.v r);
    row := r +^ 1sz;
  };
}
#pop-options

#push-options "--z3rlimit 80 --fuel 0 --ifuel 0"
inline_for_extraction noextract
fn resnorm
  (#n0 : erased nat{SZ.fits n0})
  (w : array1 bf16 (l1_forward n0))
  (noff : SZ.t{SZ.v noff + 1024 <= n0})
  (res : array1 bf16 (l1_forward 7168))
  (ua : array1 bf16 (l1_forward 7168))
  (summed : array1 bf16 (l1_forward 7168))
  (normd : array1 bf16 (l1_forward 7168))
  (partial : array1 float (l1_forward 128))
  (#pw #p1 #p2 : perm)
  (#sw : chest1 bf16 n0) (#sr #su : chest1 bf16 7168)
  (#s1 #s2 : chest1 bf16 7168) (#sp : chest1 float 128)
  preserves gpu
  preserves w |-> Frac pw sw
  preserves res |-> Frac p1 sr
  preserves ua |-> Frac p2 su
  requires summed |-> s1
  requires normd |-> s2
  requires partial |-> sp
  ensures exists* (s1' s2' : chest1 bf16 7168) (sp' : chest1 float 128).
    summed |-> s1' ** normd |-> s2' ** partial |-> sp' **
    pure (rn_done sw (SZ.v noff) sr su s1' s2' 7168)
{
  rn_done0 sw (SZ.v noff) sr su s1 s2;
  let mut row : sz = 0sz;
  while (!row <^ 7sz)
    invariant exists* (rv : sz) (a1 a2 : chest1 bf16 7168) (q1 : chest1 float 128).
      row |-> rv ** summed |-> a1 ** normd |-> a2 ** partial |-> q1 **
      pure (SZ.v rv <= 7 /\ rn_done sw (SZ.v noff) sr su a1 a2 (SZ.v rv * 1024))
    decreases (7 - SZ.v !row)
  {
    let r = !row;
    let base : sz = r *^ 1024sz;
    let mut col : sz = 0sz;
    while (!col <^ 1024sz)
      invariant exists* (cv : sz) (a1 a2 : chest1 bf16 7168).
        col |-> cv ** summed |-> a1 ** normd |-> a2 **
        pure (SZ.v cv <= 1024 /\ rn_done sw (SZ.v noff) sr su a1 a2 (SZ.v r * 1024) /\
          (forall (c:nat). c < SZ.v cv ==> acc1 a1 (SZ.v r * 1024 + c) == sum_m sr su (SZ.v r * 1024 + c)))
      decreases (1024 - SZ.v !col)
    {
      let c = !col;
      row_idx2 (SZ.v r) (SZ.v c);
      let x : float = fcast (Kuiper.Tensor.read res (((base +^ c) <: szlt 7168), ()) #p1);
      let u : float = fcast (Kuiper.Tensor.read ua (((base +^ c) <: szlt 7168), ()) #p2);
      let y : bf16 = fcast (add x u);
      with a1. assert (summed |-> a1);
      with a2. assert (normd |-> a2);
      sum_step sw (SZ.v noff) sr su a1 a2 (SZ.v r) (SZ.v c) y;
      Kuiper.Tensor.write summed (((base +^ c) <: szlt 7168), ()) y;
      col := c +^ 1sz;
    };
    with a1 a2. assert (summed |-> a1 ** normd |-> a2);
    let inv = rms_dev summed base 1024sz partial;
    rn_inv_eq sr su a1 (SZ.v r) base;
    assert pure (inv == rn_inv sr su (SZ.v r));
    col := 0sz;
    while (!col <^ 1024sz)
      invariant exists* (cv : sz) (a2 : chest1 bf16 7168).
        col |-> cv ** normd |-> a2 **
        pure (SZ.v cv <= 1024 /\ rn_done sw (SZ.v noff) sr su a1 a2 (SZ.v r * 1024 + SZ.v cv))
      decreases (1024 - SZ.v !col)
    {
      let c = !col;
      row_idx2 (SZ.v r) (SZ.v c);
      let x : float = fcast (Kuiper.Tensor.read summed (((base +^ c) <: szlt 7168), ()));
      let sc : bf16 = fcast (mul x inv);
      let wv : float = fcast (Kuiper.Tensor.read w (((noff +^ c) <: szlt n0), ()) #pw);
      let y : bf16 = fcast (mul (fcast sc <: float) wv);
      assert pure (y == rn_m sw (SZ.v noff) sr su (SZ.v r * 1024 + SZ.v c));
      with a2. assert (normd |-> a2);
      rn_step sw (SZ.v noff) sr su a1 a2 (SZ.v r) (SZ.v c) y;
      Kuiper.Tensor.write normd (((base +^ c) <: szlt 7168), ()) y;
      col := c +^ 1sz;
    };
    with a2. assert (normd |-> a2);
    rn_rows sw (SZ.v noff) sr su a1 a2 (SZ.v r);
    row := r +^ 1sz;
  };
}
#pop-options

(**** Softmax ****)

(* Pure models and lemmas for the score / softmax / value phases of attention. *)
open FStar.Real
open Kuiper
open Kuiper.Chest
open Kuiper.Float.Casts
open KPlay.SharedSpec
module Sqrt = FStar.Math.Sqrt
module Cl = FStar.Classical

#push-options "--fuel 1 --ifuel 0 --z3rlimit 10"

(* ------------------------------------------------------------------ *)
(* Deterministic float building blocks (shared with the Pulse code)   *)

inline_for_extraction noextract
let mx (a b : float) : float = if lt a b then b else a

inline_for_extraction noextract
let smax7 (s0 s1 s2 s3 s4 s5 s6 : float) : float =
  mx (mx (mx s0 s4) (mx s2 s6)) (mx (mx s1 s5) s3)

inline_for_extraction noextract
let ssum7 (e0 e1 e2 e3 e4 e5 e6 : float) : float =
  add (add (add e0 e4) (add e2 e6)) (add (add e1 e5) e3)

inline_for_extraction noextract
let score_of (sum scale mask : float) (masked : bool) : float =
  let dot : bf16 = fcast sum in
  let scaled : bf16 = fcast (mul (fcast dot) scale) in
  let ms : float = if masked then mask else zero in
  let r : bf16 = fcast (add (fcast scaled) ms) in
  fcast r

noextract let sm_zero_ok () : Lemma (v_approximates (zero #float) 0.0R) = a0 #float
noextract let zero_ok_b () : Lemma (v_approximates (zero #bf16) 0.0R) = a0 #bf16

noextract let cast_bf (x : bf16) (r : real)
  : Lemma (requires v_approximates x r) (ensures v_approximates (fcast x <: float) r)
  = fcast_approx #bf16 #float x r

noextract let cast_fb (x : float) (r : real)
  : Lemma (requires v_approximates x r) (ensures v_approximates (fcast x <: bf16) r)
  = fcast_approx #float #bf16 x r

noextract let add_ok (x y : float) (r s : real)
  : Lemma (requires v_approximates x r /\ v_approximates y s)
          (ensures v_approximates (add x y) (r +. s))
  = a_add x y r s

noextract let mul_ok (x y : float) (r s : real)
  : Lemma (requires v_approximates x r /\ v_approximates y s)
          (ensures v_approximates (mul x y) (r *. s))
  = a_mul x y r s

noextract let sub_ok (x y : float) (r s : real)
  : Lemma (requires v_approximates x r /\ v_approximates y s)
          (ensures v_approximates (sub x y) (r -. s))
  = approx_sub #float; approx_apply2 (sub #float) Kuiper.Float.Realops.r_sub x y r s

noextract let exp_ok (x : float) (r : real)
  : Lemma (requires v_approximates x r)
          (ensures v_approximates (fexp x) (Exp.exp r))
  = approx_fexp #float; approx_apply (fexp #float) Kuiper.Float.Realops.r_exp x r

noextract let div_ok (x y : float) (r : real) (s : real{s =!= 0.0R})
  : Lemma (requires v_approximates x r /\ v_approximates y s)
          (ensures v_approximates (div x y) (r /. s))
  = approx_div #float

noextract let neg_ok (x : float) (r : real)
  : Lemma (requires v_approximates x r)
          (ensures v_approximates (neg x) (0.0R -. r))
  = approx_neg #float; approx_apply (neg #float) Kuiper.Float.Realops.r_neg x r

noextract let of_int_ok (n : FStar.Int64.t)
  : Lemma (v_approximates (of_int #float n) (of_int (FStar.Int64.v n)))
  = approx_of_int #float; approx_apply (of_int #float) Kuiper.Float.Realops.r_of_int n n

noextract let sm_one_ok () : Lemma (v_approximates (one #float) 1.0R) = a1 #float

(* ------------------------------------------------------------------ *)
(* Constants                                                          *)

noextract let consts_ok (_:unit)
  : Lemma (v_approximates (div #float one (sqrt (of_int 128L))) attn_scale /\
           (let p30 : float = of_int 1073741824L in
            v_approximates (neg (mul #float (of_int 255L) (mul (mul p30 p30) (mul p30 p30)))) mask_min))
  = let c128 : float = of_int 128L in
    of_int_ok 128L;
    let sq : float = sqrt c128 in
    approx_sqrt #float;
    assert (v_approximates sq (Sqrt.sqrt 128.0R));
    Sqrt.sqrt_positive 128.0R;
    sm_one_ok ();
    div_ok one sq 1.0R (Sqrt.sqrt 128.0R);
    let p30 : float = of_int 1073741824L in
    of_int_ok 1073741824L;
    let k30 = 1073741824.0R in
    let p60 = mul p30 p30 in
    mul_ok p30 p30 k30 k30;
    let p120 = mul p60 p60 in
    mul_ok p60 p60 (k30 *. k30) (k30 *. k30);
    let c255 : float = of_int 255L in
    of_int_ok 255L;
    let m = mul c255 p120 in
    mul_ok c255 p120 255.0R ((k30 *. k30) *. (k30 *. k30));
    neg_ok m (255.0R *. ((k30 *. k30) *. (k30 *. k30)));
    assert (0.0R -. 255.0R *. ((k30 *. k30) *. (k30 *. k30)) == mask_min)

(* ------------------------------------------------------------------ *)
(* Reads                                                              *)

noextract let qok (sq : chest1 bf16 21504) (qr : S.seq real) : prop = chest1_to_seq sq %~ qr
noextract let vok (sv : chest1 bf16 28672) (vr : S.seq real) : prop = chest1_to_seq sv %~ vr

noextract let qread (sq : chest1 bf16 21504) (qr : S.seq real) (i : nat{i < 21504})
  : Lemma (requires qok sq qr) (ensures v_approximates (fcast (acc1 sq i) <: float) (at qr i))
  = assert (S.index (chest1_to_seq sq) i == acc1 sq i);
    cast_bf (acc1 sq i) (at qr i)

noextract let vread (sv : chest1 bf16 28672) (vr : S.seq real) (i : nat{i < 28672})
  : Lemma (requires vok sv vr) (ensures v_approximates (fcast (acc1 sv i) <: float) (at vr i))
  = assert (S.index (chest1_to_seq sv) i == acc1 sv i);
    cast_bf (acc1 sv i) (at vr i)

(* ------------------------------------------------------------------ *)
(* Scores                                                             *)

noextract let sm_lterm (qr : S.seq real) (qb kb lane : int) (i : nat) : real =
  at qr (qb + (i * 16 + lane)) *. at qr (kb + (i * 16 + lane))
noextract let sm_lsum (qr : S.seq real) (qb kb lane : int) (k : nat) : real = rs k (sm_lterm qr qb kb lane)
noextract let ltot (qr : S.seq real) (qb kb : int) (l : nat) : real = sm_lsum qr qb kb l 8
noextract let ssum (qr : S.seq real) (qb kb : int) (n : nat) : real = rs n (ltot qr qb kb)

noextract let lane_init (sq : chest1 bf16 21504) (qb kb lane : int)
  : Lemma (forall qr. qok sq qr ==> v_approximates (zero #float) (sm_lsum qr qb kb lane 0))
  = sm_zero_ok ()

noextract let lane_step (sq : chest1 bf16 21504) (qb kb lane : nat) (k : nat) (x y : bf16) (p p' : float)
  : Lemma (requires qb + (k * 16 + lane) < 21504 /\ kb + (k * 16 + lane) < 21504 /\
                    x == acc1 sq (qb + (k * 16 + lane)) /\ y == acc1 sq (kb + (k * 16 + lane)) /\
                    (forall qr. qok sq qr ==> v_approximates p (sm_lsum qr qb kb lane k)) /\
                    (forall (xr yr zr : real).
                       v_approximates (fcast x <: float) xr /\ v_approximates (fcast y <: float) yr /\
                       v_approximates p zr ==> v_approximates p' (xr *. yr +. zr)))
          (ensures forall qr. qok sq qr ==> v_approximates p' (sm_lsum qr qb kb lane (k + 1)))
  = let aux (qr : S.seq real)
      : Lemma (requires qok sq qr) (ensures v_approximates p' (sm_lsum qr qb kb lane (k + 1)))
      = qread sq qr (qb + (k * 16 + lane));
        qread sq qr (kb + (k * 16 + lane));
        assert (sm_lsum qr qb kb lane (k + 1) == sm_lsum qr qb kb lane k +. sm_lterm qr qb kb lane k);
        let xr = at qr (qb + (k * 16 + lane)) in
        let yr = at qr (kb + (k * 16 + lane)) in
        let zr = sm_lsum qr qb kb lane k in
        assert (v_approximates p zr);
        assert (v_approximates p' (xr *. yr +. zr))
    in
    Cl.forall_intro (Cl.move_requires aux)

noextract let sum_init (sq : chest1 bf16 21504) (qb kb : int)
  : Lemma (forall qr. qok sq qr ==> v_approximates (zero #float) (ssum qr qb kb 0))
  = sm_zero_ok ()

noextract let sm_sum_step (sq : chest1 bf16 21504) (qb kb : int) (l : nat) (sm p : float)
  : Lemma (requires (forall qr. qok sq qr ==> v_approximates sm (ssum qr qb kb l)) /\
                    (forall qr. qok sq qr ==> v_approximates p (sm_lsum qr qb kb l 8)))
          (ensures forall qr. qok sq qr ==> v_approximates (add sm p) (ssum qr qb kb (l + 1)))
  = let aux (qr : S.seq real)
      : Lemma (requires qok sq qr) (ensures v_approximates (add sm p) (ssum qr qb kb (l + 1)))
      = add_ok sm p (ssum qr qb kb l) (sm_lsum qr qb kb l 8)
    in
    Cl.forall_intro (Cl.move_requires aux)

noextract let qpart_at (qr : S.seq real) (t h k : nat)
  : Lemma (requires t < 7 /\ h < 16 /\ k < 128)
          (ensures at (qpart qr) ((t * 16 + h) * 128 + k) == at qr ((t * 24 + h) * 128 + k))
  = let i = (t * 16 + h) * 128 + k in
    assert (i == t * 2048 + (h * 128 + k));
    assert (i / 2048 == t /\ i % 2048 == h * 128 + k);
    assert ((h * 128 + k) / 128 == h /\ (h * 128 + k) % 128 == k);
    assert (i % 128 == k)

noextract let kpart_at (qr : S.seq real) (j hh k : nat)
  : Lemma (requires j < 7 /\ hh < 8 /\ k < 128)
          (ensures at (kpart qr) ((j * 8 + hh) * 128 + k) == at qr ((j * 24 + 16 + hh) * 128 + k))
  = let i = (j * 8 + hh) * 128 + k in
    assert (i == j * 1024 + (hh * 128 + k));
    assert (i / 1024 == j /\ i % 1024 == hh * 128 + k);
    assert ((hh * 128 + k) / 128 == hh /\ (hh * 128 + k) % 128 == k);
    assert (i % 128 == k)

noextract let ssum_eq (qr : S.seq real) (t h j : nat)
  : Lemma (requires t < 7 /\ h < 16 /\ j < 7)
          (ensures ssum qr ((t * 24 + h) * 128) ((j * 24 + 16 + h / 2) * 128) 16 ==
                   sdot 128 (qpart qr) ((t * 16 + h) * 128) 1 (kpart qr) ((j * 8 + h / 2) * 128) 1)
  = let qb = (t * 24 + h) * 128 in
    let kb = (j * 24 + 16 + h / 2) * 128 in
    let g (c : nat) : real = at qr (qb + c) *. at qr (kb + c) in
    sdot_rs 128 (qpart qr) ((t * 16 + h) * 128) 1 (kpart qr) ((j * 8 + h / 2) * 128) 1;
    let aux (c : nat)
      : Lemma (c < 128 ==>
               at (qpart qr) ((t * 16 + h) * 128 + c * 1) *. at (kpart qr) ((j * 8 + h / 2) * 128 + c * 1) == g c)
      = if c < 128 then (qpart_at qr t h c; kpart_at qr j (h / 2) c)
    in
    Cl.forall_intro aux;
    rs_ext 128 (fun c -> at (qpart qr) ((t * 16 + h) * 128 + c * 1) *. at (kpart qr) ((j * 8 + h / 2) * 128 + c * 1)) g;
    rs_strided 8 16 g;
    let aux2 (l : nat) : Lemma (ltot qr qb kb l == rs 8 (fun i -> g (i * 16 + l)))
      = rs_ext 8 (sm_lterm qr qb kb l) (fun i -> g (i * 16 + l))
    in
    Cl.forall_intro aux2;
    rs_ext 16 (ltot qr qb kb) (fun l -> rs 8 (fun i -> g (i * 16 + l)))

noextract let ascore (qr : S.seq real) : S.seq real = attn_scores (qpart qr) (kpart qr)
noextract let sreal (qr : S.seq real) (t h j : nat) : real = at (ascore qr) ((t * 16 + h) * 7 + j)

noextract let sreal_eq (qr : S.seq real) (t h j : nat)
  : Lemma (requires t < 7 /\ h < 16 /\ j < 7)
          (ensures sreal qr t h j ==
                   sdot 128 (qpart qr) ((t * 16 + h) * 128) 1 (kpart qr) ((j * 8 + h / 2) * 128) 1 *. attn_scale +.
                   (if j > t then mask_min else 0.0R))
  = reveal_opaque (`%attn_scores) attn_scores;
    let i = (t * 16 + h) * 7 + j in
    assert (i == t * 112 + (h * 7 + j));
    assert (i / 112 == t /\ i % 112 == h * 7 + j);
    assert (i / 7 == t * 16 + h);
    assert ((i / 7) % 16 == h /\ i % 7 == j)

noextract let score_real (sum scale mask : float) (masked : bool) (d : real)
  : Lemma (requires v_approximates sum d /\ v_approximates scale attn_scale /\ v_approximates mask mask_min)
          (ensures v_approximates (score_of sum scale mask masked)
                     (d *. attn_scale +. (if masked then mask_min else 0.0R)))
  = let dot : bf16 = fcast sum in
    cast_fb sum d;
    cast_bf dot d;
    let x : float = mul (fcast dot) scale in
    mul_ok (fcast dot) scale d attn_scale;
    let scaled : bf16 = fcast x in
    cast_fb x (d *. attn_scale);
    cast_bf scaled (d *. attn_scale);
    let ms : float = if masked then mask else zero in
    sm_zero_ok ();
    let mr : real = if masked then mask_min else 0.0R in
    assert (v_approximates ms mr);
    add_ok (fcast scaled) ms (d *. attn_scale) mr;
    let r : bf16 = fcast (add (fcast scaled) ms) in
    cast_fb (add (fcast scaled) ms) (d *. attn_scale +. mr);
    cast_bf r (d *. attn_scale +. mr)

noextract let score_lemma (sq : chest1 bf16 21504) (sum scale mask : float) (t h j : nat)
  : Lemma (requires t < 7 /\ h < 16 /\ j < 7 /\
                    v_approximates scale attn_scale /\ v_approximates mask mask_min /\
                    (forall qr. qok sq qr ==>
                       v_approximates sum (ssum qr ((t * 24 + h) * 128) ((j * 24 + 16 + h / 2) * 128) 16)))
          (ensures forall qr. qok sq qr ==> v_approximates (score_of sum scale mask (t < j)) (sreal qr t h j))
  = let aux (qr : S.seq real)
      : Lemma (requires qok sq qr) (ensures v_approximates (score_of sum scale mask (t < j)) (sreal qr t h j))
      = ssum_eq qr t h j;
        sreal_eq qr t h j;
        score_real sum scale mask (t < j) (ssum qr ((t * 24 + h) * 128) ((j * 24 + 16 + h / 2) * 128) 16)
    in
    Cl.forall_intro (Cl.move_requires aux)

(* ------------------------------------------------------------------ *)
(* Softmax                                                            *)

noextract let gm (ss : chest1 float 8) : GTot float =
  smax7 (acc1 ss 0) (acc1 ss 1) (acc1 ss 2) (acc1 ss 3) (acc1 ss 4) (acc1 ss 5) (acc1 ss 6)
noextract let ge (ss : chest1 float 8) (k : natlt 8) : GTot float = fexp (sub (acc1 ss k) (gm ss))
noextract let gtot (ss : chest1 float 8) : GTot float =
  ssum7 (ge ss 0) (ge ss 1) (ge ss 2) (ge ss 3) (ge ss 4) (ge ss 5) (ge ss 6)
noextract let gp (ss : chest1 float 8) (k : natlt 8) : GTot bf16 = fcast (div (ge ss k) (gtot ss))

#push-options "--fuel 8"
noextract let rs7 (f : nat -> real)
  : Lemma (rs 7 f == ((f 0 +. f 4) +. (f 2 +. f 6)) +. ((f 1 +. f 5) +. f 3))
  = ()
#pop-options

noextract let ge_ok (ss : chest1 float 8) (a : S.seq real) (b : nat) (mr : real) (j : nat{j < 7})
  : Lemma (requires v_approximates (gm ss) mr /\ v_approximates (acc1 ss j) (at a (b * 7 + j)))
          (ensures v_approximates (ge ss j) (Exp.exp (at a (b * 7 + j) -. mr)))
  = sub_ok (acc1 ss j) (gm ss) (at a (b * 7 + j)) mr;
    exp_ok (sub (acc1 ss j) (gm ss)) (at a (b * 7 + j) -. mr)

noextract let tot_ok (e0 e1 e2 e3 e4 e5 e6 : float) (r0 r1 r2 r3 r4 r5 r6 : real)
  : Lemma (requires v_approximates e0 r0 /\ v_approximates e1 r1 /\ v_approximates e2 r2 /\
                    v_approximates e3 r3 /\ v_approximates e4 r4 /\ v_approximates e5 r5 /\
                    v_approximates e6 r6)
          (ensures v_approximates (ssum7 e0 e1 e2 e3 e4 e5 e6)
                     (((r0 +. r4) +. (r2 +. r6)) +. ((r1 +. r5) +. r3)))
  = add_ok e0 e4 r0 r4;
    add_ok e2 e6 r2 r6;
    add_ok (add e0 e4) (add e2 e6) (r0 +. r4) (r2 +. r6);
    add_ok e1 e5 r1 r5;
    add_ok (add e1 e5) e3 (r1 +. r5) r3;
    add_ok (add (add e0 e4) (add e2 e6)) (add (add e1 e5) e3)
           ((r0 +. r4) +. (r2 +. r6)) ((r1 +. r5) +. r3)

noextract let softmax7_at (a : S.seq real) (b : nat) (k : nat)
  : Lemma (requires b < 112 /\ k < 7 /\ S.length a == 784)
          (ensures at (softmax7 a) (b * 7 + k) == Exp.exp (at a (b * 7 + k)) /. sumexp 7 a (b * 7))
  = reveal_opaque (`%softmax7) softmax7;
    assert ((b * 7 + k) / 7 == b)

noextract let prob_lemma (ss : chest1 float 8) (a : S.seq real) (b : nat) (k : nat{k < 7})
  : Lemma (requires b < 112 /\ S.length a == 784 /\
                    (forall (j : nat). j < 7 ==> v_approximates (acc1 ss j) (at a (b * 7 + j))))
          (ensures v_approximates (fcast (gp ss k) <: float) (at (softmax7 a) (b * 7 + k)))
  = let mr = to_real (gm ss) in
    to_real_ok (gm ss);
    let s (i : nat) : real = at a (b * 7 + i) in
    let e (i : nat) : real = Exp.exp (s i -. mr) in
    ge_ok ss a b mr 0; ge_ok ss a b mr 1; ge_ok ss a b mr 2; ge_ok ss a b mr 3;
    ge_ok ss a b mr 4; ge_ok ss a b mr 5; ge_ok ss a b mr 6;
    tot_ok (ge ss 0) (ge ss 1) (ge ss 2) (ge ss 3) (ge ss 4) (ge ss 5) (ge ss 6)
           (e 0) (e 1) (e 2) (e 3) (e 4) (e 5) (e 6);
    rs7 e;
    softmax_shift 7 s mr k;
    rs_ext 7 e (fun i -> Exp.exp (s i -. mr));
    let t = rs 7 e in
    assert (v_approximates (gtot ss) t);
    ge_ok ss a b mr k;
    div_ok (ge ss k) (gtot ss) (e k) t;
    cast_fb (div (ge ss k) (gtot ss)) (e k /. t);
    cast_bf (gp ss k) (e k /. t);
    sumexp_rs 7 a (b * 7);
    rs_ext 7 (fun i -> Exp.exp (s i)) (fun k -> Exp.exp (at a (b * 7 + k)));
    softmax7_at a b k

noextract let pr (qr : S.seq real) (t h k : nat) : real = at (softmax7 (ascore qr)) ((t * 16 + h) * 7 + k)

noextract let sc_good (sq : chest1 bf16 21504) (ss : chest1 float 8) (t h : nat) (j : natlt 8) : prop =
  forall qr. qok sq qr ==> v_approximates (acc1 ss j) (sreal qr t h j)

[@@"opaque_to_smt"]
noextract let scok (sq : chest1 bf16 21504) (ss : chest1 float 8) (t h : nat) (n : nat) : prop =
  forall (j : natlt 8). j < n ==> sc_good sq ss t h j

noextract let scok_init (sq : chest1 bf16 21504) (ss : chest1 float 8) (t h : nat)
  : Lemma (scok sq ss t h 0)
  = reveal_opaque (`%scok) scok

noextract let scok_step (sq : chest1 bf16 21504) (ss : chest1 float 8) (t h : nat) (n : natlt 8) (x : float)
  : Lemma (requires scok sq ss t h n /\ (forall qr. qok sq qr ==> v_approximates x (sreal qr t h n)))
          (ensures scok sq (upd1 ss n x) t h (n + 1))
  = reveal_opaque (`%scok) scok;
    let ss' = upd1 ss n x in
    let aux (j : natlt 8) : Lemma (j < n + 1 ==> sc_good sq ss' t h j)
      = assert (acc1 ss' j == (if j = n then x else acc1 ss j))
    in
    Cl.forall_intro aux

noextract let probs_lift (sq : chest1 bf16 21504) (ss : chest1 float 8) (t h : nat) (k : nat{k < 7})
  : Lemma (requires t < 7 /\ h < 16 /\ scok sq ss t h 7)
          (ensures forall qr. qok sq qr ==> v_approximates (fcast (gp ss k) <: float) (pr qr t h k))
  = reveal_opaque (`%scok) scok;
    let aux (qr : S.seq real)
      : Lemma (requires qok sq qr) (ensures v_approximates (fcast (gp ss k) <: float) (pr qr t h k))
      = let a = ascore qr in
        let b = t * 16 + h in
        let aux2 (j : nat) : Lemma (j < 7 ==> v_approximates (acc1 ss j) (at a (b * 7 + j)))
          = if j < 7 then assert (sc_good sq ss t h j)
        in
        Cl.forall_intro aux2;
        prob_lemma ss a b k
    in
    Cl.forall_intro (Cl.move_requires aux)

(* ------------------------------------------------------------------ *)
(* Values                                                             *)

noextract let vidx (hh col k : nat) : nat = k * 4096 + 3072 + hh * 128 + col

noextract let vterm (qr vr : S.seq real) (t h col k0 : nat) (i : nat) : real =
  pr qr t h (k0 + i) *. at vr (vidx (h / 2) col (k0 + i))
noextract let vsum (qr vr : S.seq real) (t h col k0 : nat) (n : nat) : real = rs n (vterm qr vr t h col k0)

noextract let vinit (sq : chest1 bf16 21504) (sv : chest1 bf16 28672) (t h col k0 : nat)
  : Lemma (forall qr vr. qok sq qr /\ vok sv vr ==> v_approximates (zero #float) (vsum qr vr t h col k0 0))
  = sm_zero_ok ()

noextract let vstep (sq : chest1 bf16 21504) (sv : chest1 bf16 28672) (t h col k0 n : nat)
  (pk v : bf16) (z z' : float)
  : Lemma (requires k0 + n < 7 /\ h < 16 /\ col < 128 /\
                    (forall qr. qok sq qr ==> v_approximates (fcast pk <: float) (pr qr t h (k0 + n))) /\
                    v == acc1 sv (vidx (h / 2) col (k0 + n)) /\
                    (forall qr vr. qok sq qr /\ vok sv vr ==> v_approximates z (vsum qr vr t h col k0 n)) /\
                    (forall (xr yr zr : real).
                       v_approximates (fcast pk <: float) xr /\ v_approximates (fcast v <: float) yr /\
                       v_approximates z zr ==> v_approximates z' (xr *. yr +. zr)))
          (ensures forall qr vr. qok sq qr /\ vok sv vr ==> v_approximates z' (vsum qr vr t h col k0 (n + 1)))
  = let aux (qr vr : S.seq real)
      : Lemma (requires qok sq qr /\ vok sv vr)
              (ensures v_approximates z' (vsum qr vr t h col k0 (n + 1)))
      = vread sv vr (vidx (h / 2) col (k0 + n));
        let xr = pr qr t h (k0 + n) in
        let yr = at vr (vidx (h / 2) col (k0 + n)) in
        let zr = vsum qr vr t h col k0 n in
        assert (v_approximates (fcast pk <: float) xr);
        assert (v_approximates z zr);
        assert (v_approximates z' (xr *. yr +. zr));
        assert (vsum qr vr t h col k0 (n + 1) == zr +. vterm qr vr t h col k0 n)
    in
    Cl.forall_intro_2 (Cl.move_requires_2 aux)

noextract let avf (qr vr : S.seq real) : S.seq real = attn_values (softmax7 (ascore qr)) (vpart vr)
noextract let oidx (t h c : nat) : nat = t * 2048 + h * 128 + c

noextract let vpart_at (vr : S.seq real) (hh col k : nat)
  : Lemma (requires hh < 8 /\ col < 128 /\ k < 7)
          (ensures at (vpart vr) (hh * 128 + col + k * 1024) == at vr (vidx hh col k))
  = let i = hh * 128 + col + k * 1024 in
    assert (i == k * 1024 + (hh * 128 + col));
    assert (i / 1024 == k /\ i % 1024 == hh * 128 + col)

noextract let avf_at (qr vr : S.seq real) (t h col : nat)
  : Lemma (requires t < 7 /\ h < 16 /\ col < 128)
          (ensures at (avf qr vr) (oidx t h col) == vsum qr vr t h col 0 7)
  = reveal_opaque (`%attn_values) attn_values;
    let i = oidx t h col in
    assert (i / 2048 == t /\ i % 2048 == h * 128 + col);
    assert ((h * 128 + col) / 128 == h /\ (h * 128 + col) % 128 == col);
    assert (i % 128 == col);
    let p = softmax7 (ascore qr) in
    let v = vpart vr in
    sdot_rs 7 p ((t * 16 + h) * 7) 1 v ((h / 2) * 128 + col) 1024;
    let aux (k : nat)
      : Lemma (k < 7 ==> at p ((t * 16 + h) * 7 + k * 1) *. at v ((h / 2) * 128 + col + k * 1024) ==
                         vterm qr vr t h col 0 k)
      = if k < 7 then vpart_at vr (h / 2) col k
    in
    Cl.forall_intro aux;
    rs_ext 7 (fun k -> at p ((t * 16 + h) * 7 + k * 1) *. at v ((h / 2) * 128 + col + k * 1024))
             (vterm qr vr t h col 0)

noextract let vfinal (sq : chest1 bf16 21504) (sv : chest1 bf16 28672) (t h col : nat) (a b : float)
  : Lemma (requires t < 7 /\ h < 16 /\ col < 128 /\
                    (forall qr vr. qok sq qr /\ vok sv vr ==> v_approximates a (vsum qr vr t h col 0 4)) /\
                    (forall qr vr. qok sq qr /\ vok sv vr ==> v_approximates b (vsum qr vr t h col 4 3)))
          (ensures forall qr vr. qok sq qr /\ vok sv vr ==>
                     v_approximates (fcast (add a b) <: bf16) (at (avf qr vr) (oidx t h col)))
  = let aux (qr vr : S.seq real)
      : Lemma (requires qok sq qr /\ vok sv vr)
              (ensures v_approximates (fcast (add a b) <: bf16) (at (avf qr vr) (oidx t h col)))
      = let ra = vsum qr vr t h col 0 4 in
        let rb = vsum qr vr t h col 4 3 in
        add_ok a b ra rb;
        cast_fb (add a b) (ra +. rb);
        rs_split 4 3 (vterm qr vr t h col 0);
        rs_ext 3 (fun i -> vterm qr vr t h col 0 (4 + i)) (vterm qr vr t h col 4);
        avf_at qr vr t h col
    in
    Cl.forall_intro_2 (Cl.move_requires_2 aux)

(* ------------------------------------------------------------------ *)
(* Output progress                                                    *)

noextract let ogood (sq : chest1 bf16 21504) (sv : chest1 bf16 28672) (so : chest1 bf16 14336) (i : nat) : prop =
  i < 14336 ==>
  (forall qr vr. qok sq qr /\ vok sv vr ==> v_approximates (acc1 so i) (at (avf qr vr) i))

noextract let before (h t c h' t' c' : nat) : bool = h' < h || (h' = h && (t' < t || (t' = t && c' < c)))

[@@"opaque_to_smt"]
noextract let odone (sq : chest1 bf16 21504) (sv : chest1 bf16 28672) (so : chest1 bf16 14336) (h t c : nat) : prop =
  forall (h' t' c' : nat). h' < 16 /\ t' < 7 /\ c' < 128 /\ before h t c h' t' c' ==>
    ogood sq sv so (oidx t' h' c')

noextract let odone_init (sq : chest1 bf16 21504) (sv : chest1 bf16 28672) (so : chest1 bf16 14336)
  : Lemma (odone sq sv so 0 0 0)
  = reveal_opaque (`%odone) odone

noextract let oidx_inj (t h c t' h' c' : nat)
  : Lemma (requires t < 7 /\ h < 16 /\ c < 128 /\ t' < 7 /\ h' < 16 /\ c' < 128 /\
                    oidx t h c == oidx t' h' c')
          (ensures t == t' /\ h == h' /\ c == c')
  = let i = oidx t h c in
    assert (i / 2048 == t /\ i % 2048 == h * 128 + c);
    assert (i / 2048 == t' /\ i % 2048 == h' * 128 + c');
    assert ((h * 128 + c) / 128 == h /\ (h * 128 + c) % 128 == c);
    assert ((h' * 128 + c') / 128 == h' /\ (h' * 128 + c') % 128 == c')

noextract let odone_step (sq : chest1 bf16 21504) (sv : chest1 bf16 28672) (so : chest1 bf16 14336)
  (h t c : nat) (o : bf16)
  : Lemma (requires h < 16 /\ t < 7 /\ c < 128 /\ odone sq sv so h t c /\
                    (forall qr vr. qok sq qr /\ vok sv vr ==> v_approximates o (at (avf qr vr) (oidx t h c))))
          (ensures oidx t h c < 14336 /\ odone sq sv (upd1 so (oidx t h c) o) h t (c + 1))
  = reveal_opaque (`%odone) odone;
    let so' = upd1 so (oidx t h c) o in
    let aux (h' t' c' : nat)
      : Lemma (h' < 16 /\ t' < 7 /\ c' < 128 /\ before h t (c + 1) h' t' c' ==> ogood sq sv so' (oidx t' h' c'))
      = if h' < 16 && t' < 7 && c' < 128 && before h t (c + 1) h' t' c' then begin
          if h' = h && t' = t && c' = c then ()
          else begin
            assert (before h t c h' t' c');
            assert (ogood sq sv so (oidx t' h' c'));
            Cl.move_requires (oidx_inj t h c t' h') c';
            assert (acc1 so' (oidx t' h' c') == acc1 so (oidx t' h' c'))
          end
        end
    in
    Cl.forall_intro_3 aux

noextract let odone_row (sq : chest1 bf16 21504) (sv : chest1 bf16 28672) (so : chest1 bf16 14336) (h t : nat)
  : Lemma (requires odone sq sv so h t 128) (ensures odone sq sv so h (t + 1) 0)
  = reveal_opaque (`%odone) odone

noextract let odone_head (sq : chest1 bf16 21504) (sv : chest1 bf16 28672) (so : chest1 bf16 14336) (h : nat)
  : Lemma (requires odone sq sv so h 7 0) (ensures odone sq sv so (h + 1) 0 0)
  = reveal_opaque (`%odone) odone

noextract let odone_final (sq : chest1 bf16 21504) (sv : chest1 bf16 28672) (so : chest1 bf16 14336)
  : Lemma (requires odone sq sv so 16 0 0)
          (ensures forall (qr vr : S.seq real). chest1_to_seq sq %~ qr /\ chest1_to_seq sv %~ vr ==>
                     chest1_to_seq so %~ attn_values (softmax7 (attn_scores (qpart qr) (kpart qr))) (vpart vr))
  = reveal_opaque (`%odone) odone;
    let aux (qr vr : S.seq real)
      : Lemma (requires chest1_to_seq sq %~ qr /\ chest1_to_seq sv %~ vr)
              (ensures chest1_to_seq so %~ attn_values (softmax7 (attn_scores (qpart qr) (kpart qr))) (vpart vr))
      = assert (qok sq qr /\ vok sv vr);
        let aux2 (i : nat)
          : Lemma (i < 14336 ==> v_approximates (S.index (chest1_to_seq so) i) (S.index (avf qr vr) i))
          = if i < 14336 then begin
              let t = i / 2048 in
              let h = (i % 2048) / 128 in
              let c = i % 128 in
              assert (oidx t h c == i);
              assert (before 16 0 0 h t c);
              assert (ogood sq sv so (oidx t h c))
            end
        in
        Cl.forall_intro aux2;
        assert (seq_approximates (chest1_to_seq so) (avf qr vr))
    in
    Cl.forall_intro_2 (Cl.move_requires_2 aux)

#pop-options

(**** AttSv ****)


open FStar.Real
open Kuiper
open Kuiper.Chest
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Float.Casts
open KPlay.SharedSpec
module FM = Kuiper.Float32.FastMath

#push-options "--fuel 2 --ifuel 1 --z3rlimit 20"

(* One lane partial of the score dot product: fma over col = lane, lane+16, ..., < 128. *)
inline_for_extraction noextract
fn lane_partial
  (qk : array1 bf16 (l1_forward 21504))
  (qb kb : sz) (lane : szlt 16)
  (#fq : perm) (#sq : chest1 bf16 21504)
  preserves gpu ** qk |-> Frac fq sq
  requires pure (SZ.v qb + 128 <= 21504 /\ SZ.v kb + 128 <= 21504)
  returns p : float
  ensures pure (forall qr. qok sq qr ==> v_approximates p (sm_lsum qr (SZ.v qb) (SZ.v kb) (SZ.v lane) 8))
{
  let mut partial : float = zero;
  let mut col : sz = lane;
  lane_init sq (SZ.v qb) (SZ.v kb) (SZ.v lane);
  while (!col <^ 128sz)
    invariant exists* (c : sz) (pv : float).
      col |-> c ** partial |-> pv **
      pure (SZ.v c % 16 == SZ.v lane /\ SZ.v c < 144 /\
            (forall qr. qok sq qr ==> v_approximates pv (sm_lsum qr (SZ.v qb) (SZ.v kb) (SZ.v lane) (SZ.v c / 16))))
    decreases (144 - SZ.v !col)
  {
    let c = !col;
    let x = Kuiper.Tensor.read qk ((qb +^ c <: szlt 21504), ()) #fq;
    let y = Kuiper.Tensor.read qk ((kb +^ c <: szlt 21504), ()) #fq;
    let pv = !partial;
    let pv' = FM.fma_rn (fcast x) (fcast y) pv;
    lane_step sq (SZ.v qb) (SZ.v kb) (SZ.v lane) (SZ.v c / 16) x y pv pv';
    partial := pv';
    col := c +^ 16sz;
  };
  !partial
}

(* Masked, scaled score of (row t, head h, key j). *)
inline_for_extraction noextract
fn score1
  (qk : array1 bf16 (l1_forward 21504))
  (t h j : sz) (scale mask : float)
  (#fq : perm) (#sq : chest1 bf16 21504)
  preserves gpu ** qk |-> Frac fq sq
  requires pure (SZ.v t < 7 /\ SZ.v h < 16 /\ SZ.v j < 7 /\
                 v_approximates scale attn_scale /\ v_approximates mask mask_min)
  returns s : float
  ensures pure (forall qr. qok sq qr ==> v_approximates s (sreal qr (SZ.v t) (SZ.v h) (SZ.v j)))
{
  let qb = (t *^ 24sz +^ h) *^ 128sz;
  let kb = (j *^ 24sz +^ 16sz +^ h /^ 2sz) *^ 128sz;
  let mut sum : float = zero;
  let mut lane : sz = 0sz;
  sum_init sq (SZ.v qb) (SZ.v kb);
  while (!lane <^ 16sz)
    invariant exists* (l : sz) (sm : float).
      lane |-> l ** sum |-> sm **
      pure (SZ.v l <= 16 /\
            (forall qr. qok sq qr ==> v_approximates sm (ssum qr (SZ.v qb) (SZ.v kb) (SZ.v l))))
    decreases (16 - SZ.v !lane)
  {
    let l = !lane;
    let p = lane_partial qk qb kb l #fq #sq;
    let sm = !sum;
    sm_sum_step sq (SZ.v qb) (SZ.v kb) (SZ.v l) sm p;
    sum := add sm p;
    lane := l +^ 1sz;
  };
  let sm = !sum;
  score_lemma sq sm scale mask (SZ.v t) (SZ.v h) (SZ.v j);
  score_of sm scale mask (t <^ j)
}

(* Scores of the 7 keys into sc[0..6]. *)
inline_for_extraction noextract
fn scores7
  (qk : array1 bf16 (l1_forward 21504))
  (sc : array1 float (l1_forward 8))
  (t h : sz) (scale mask : float)
  (#fq : perm) (#sq : chest1 bf16 21504) (#ss : chest1 float 8)
  preserves gpu ** qk |-> Frac fq sq
  requires sc |-> ss **
    pure (SZ.v t < 7 /\ SZ.v h < 16 /\ v_approximates scale attn_scale /\ v_approximates mask mask_min)
  ensures exists* ss'. sc |-> ss' ** pure (scok sq ss' (SZ.v t) (SZ.v h) 7)
{
  let mut key : sz = 0sz;
  scok_init sq ss (SZ.v t) (SZ.v h);
  while (!key <^ 7sz)
    invariant exists* (k : sz) (ssk : chest1 float 8).
      key |-> k ** sc |-> ssk **
      pure (SZ.v k <= 7 /\ scok sq ssk (SZ.v t) (SZ.v h) (SZ.v k))
    decreases (7 - SZ.v !key)
  {
    let k = !key;
    let s = score1 qk t h k scale mask #fq #sq;
    with ssk. assert (sc |-> ssk);
    Kuiper.Tensor.write sc ((k <: szlt 8), ()) s;
    scok_step sq ssk (SZ.v t) (SZ.v h) (SZ.v k) s;
    key := k +^ 1sz;
  };
}

inline_for_extraction noextract
fn att_row
  (qk : array1 bf16 (l1_forward 21504))
  (qkv : array1 bf16 (l1_forward 28672))
  (out : array1 bf16 (l1_forward 14336))
  (sc : array1 float (l1_forward 8))
  (t h : sz) (scale mask : float)
  (#fq #fv : perm) (#sq : chest1 bf16 21504) (#sv : chest1 bf16 28672)
  (#so : chest1 bf16 14336) (#ss : chest1 float 8)
  preserves gpu ** qk |-> Frac fq sq ** qkv |-> Frac fv sv
  requires out |-> so ** sc |-> ss **
    pure (SZ.v t < 7 /\ SZ.v h < 16 /\ v_approximates scale attn_scale /\ v_approximates mask mask_min /\
          odone sq sv so (SZ.v h) (SZ.v t) 0)
  ensures exists* so' ss'. out |-> so' ** sc |-> ss' ** pure (odone sq sv so' (SZ.v h) (SZ.v t) 128)
{
  scores7 qk sc t h scale mask #fq #sq #ss;
  with ss1. assert (sc |-> ss1);
  let s0 = Kuiper.Tensor.read sc ((0sz <: szlt 8), ());
  let s1 = Kuiper.Tensor.read sc ((1sz <: szlt 8), ());
  let s2 = Kuiper.Tensor.read sc ((2sz <: szlt 8), ());
  let s3 = Kuiper.Tensor.read sc ((3sz <: szlt 8), ());
  let s4 = Kuiper.Tensor.read sc ((4sz <: szlt 8), ());
  let s5 = Kuiper.Tensor.read sc ((5sz <: szlt 8), ());
  let s6 = Kuiper.Tensor.read sc ((6sz <: szlt 8), ());
  let m = smax7 s0 s1 s2 s3 s4 s5 s6;
  let e0 = fexp (sub s0 m);
  let e1 = fexp (sub s1 m);
  let e2 = fexp (sub s2 m);
  let e3 = fexp (sub s3 m);
  let e4 = fexp (sub s4 m);
  let e5 = fexp (sub s5 m);
  let e6 = fexp (sub s6 m);
  let tot = ssum7 e0 e1 e2 e3 e4 e5 e6;
  let p0 : bf16 = fcast (div e0 tot);
  let p1 : bf16 = fcast (div e1 tot);
  let p2 : bf16 = fcast (div e2 tot);
  let p3 : bf16 = fcast (div e3 tot);
  let p4 : bf16 = fcast (div e4 tot);
  let p5 : bf16 = fcast (div e5 tot);
  let p6 : bf16 = fcast (div e6 tot);
  assert (pure (p0 == gp ss1 0 /\ p1 == gp ss1 1 /\ p2 == gp ss1 2 /\ p3 == gp ss1 3 /\
                p4 == gp ss1 4 /\ p5 == gp ss1 5 /\ p6 == gp ss1 6));
  probs_lift sq ss1 (SZ.v t) (SZ.v h) 0;
  probs_lift sq ss1 (SZ.v t) (SZ.v h) 1;
  probs_lift sq ss1 (SZ.v t) (SZ.v h) 2;
  probs_lift sq ss1 (SZ.v t) (SZ.v h) 3;
  probs_lift sq ss1 (SZ.v t) (SZ.v h) 4;
  probs_lift sq ss1 (SZ.v t) (SZ.v h) 5;
  probs_lift sq ss1 (SZ.v t) (SZ.v h) 6;
  let hh = h /^ 2sz;
  let mut col : sz = 0sz;
  while (!col <^ 128sz)
    invariant exists* (c : sz) (so1 : chest1 bf16 14336).
      col |-> c ** out |-> so1 **
      pure (SZ.v c <= 128 /\ odone sq sv so1 (SZ.v h) (SZ.v t) (SZ.v c))
    decreases (128 - SZ.v !col)
  {
    let c = !col;
    let vb = 3072sz +^ hh *^ 128sz +^ c;
    let v0 = Kuiper.Tensor.read qkv ((vb <: szlt 28672), ()) #fv;
    let v1 = Kuiper.Tensor.read qkv ((4096sz +^ vb <: szlt 28672), ()) #fv;
    let v2 = Kuiper.Tensor.read qkv ((8192sz +^ vb <: szlt 28672), ()) #fv;
    let v3 = Kuiper.Tensor.read qkv ((12288sz +^ vb <: szlt 28672), ()) #fv;
    let v4 = Kuiper.Tensor.read qkv ((16384sz +^ vb <: szlt 28672), ()) #fv;
    let v5 = Kuiper.Tensor.read qkv ((20480sz +^ vb <: szlt 28672), ()) #fv;
    let v6 = Kuiper.Tensor.read qkv ((24576sz +^ vb <: szlt 28672), ()) #fv;
    vinit sq sv (SZ.v t) (SZ.v h) (SZ.v c) 0;
    let a0 = FM.fma_rn (fcast p0) (fcast v0) zero;
    vstep sq sv (SZ.v t) (SZ.v h) (SZ.v c) 0 0 p0 v0 zero a0;
    let a1 = FM.fma_rn (fcast p1) (fcast v1) a0;
    vstep sq sv (SZ.v t) (SZ.v h) (SZ.v c) 0 1 p1 v1 a0 a1;
    let a2 = FM.fma_rn (fcast p2) (fcast v2) a1;
    vstep sq sv (SZ.v t) (SZ.v h) (SZ.v c) 0 2 p2 v2 a1 a2;
    let a3 = FM.fma_rn (fcast p3) (fcast v3) a2;
    vstep sq sv (SZ.v t) (SZ.v h) (SZ.v c) 0 3 p3 v3 a2 a3;
    vinit sq sv (SZ.v t) (SZ.v h) (SZ.v c) 4;
    let b0 = FM.fma_rn (fcast p4) (fcast v4) zero;
    vstep sq sv (SZ.v t) (SZ.v h) (SZ.v c) 4 0 p4 v4 zero b0;
    let b1 = FM.fma_rn (fcast p5) (fcast v5) b0;
    vstep sq sv (SZ.v t) (SZ.v h) (SZ.v c) 4 1 p5 v5 b0 b1;
    let b2 = FM.fma_rn (fcast p6) (fcast v6) b1;
    vstep sq sv (SZ.v t) (SZ.v h) (SZ.v c) 4 2 p6 v6 b1 b2;
    vfinal sq sv (SZ.v t) (SZ.v h) (SZ.v c) a3 b2;
    let o : bf16 = fcast (add a3 b2);
    with so1. assert (out |-> so1);
    Kuiper.Tensor.write out ((t *^ 2048sz +^ h *^ 128sz +^ c <: szlt 14336), ()) o;
    odone_step sq sv so1 (SZ.v h) (SZ.v t) (SZ.v c) o;
    col := c +^ 1sz;
  };
}

inline_for_extraction noextract
fn att_sv
  (qk : array1 bf16 (l1_forward 21504))
  (qkv : array1 bf16 (l1_forward 28672))
  (out : array1 bf16 (l1_forward 14336))
  (sc : array1 float (l1_forward 8))
  (#fq #fv : perm) (#sq : chest1 bf16 21504) (#sv : chest1 bf16 28672) (#so : chest1 bf16 14336) (#ss : chest1 float 8)
  preserves gpu ** qk |-> Frac fq sq ** qkv |-> Frac fv sv
  requires out |-> so ** sc |-> ss
  ensures exists* so' ss'. out |-> so' ** sc |-> ss' **
    pure (forall (qr vr : S.seq real). chest1_to_seq sq %~ qr /\ chest1_to_seq sv %~ vr ==>
            chest1_to_seq so' %~ attn_values (softmax7 (attn_scores (qpart qr) (kpart qr))) (vpart vr))
{
  let scale : float = div one (sqrt (of_int 128L));
  (* p30 goes through a mutable cell so that krml does not constant-fold the float products
     (its folder cannot parse the resulting literal). *)
  let mut p30c : float = of_int 1073741824L;
  let p30 : float = !p30c;
  let mask : float = neg (mul (of_int 255L) (mul (mul p30 p30) (mul p30 p30)));
  consts_ok ();
  odone_init sq sv so;
  let mut head : sz = 0sz;
  while (!head <^ 16sz)
    invariant exists* (h : sz) (so1 : chest1 bf16 14336) (ss1 : chest1 float 8).
      head |-> h ** out |-> so1 ** sc |-> ss1 **
      pure (SZ.v h <= 16 /\ odone sq sv so1 (SZ.v h) 0 0)
    decreases (16 - SZ.v !head)
  {
    let h = !head;
    let mut row : sz = 0sz;
    while (!row <^ 7sz)
      invariant exists* (t : sz) (so1 : chest1 bf16 14336) (ss1 : chest1 float 8).
        row |-> t ** out |-> so1 ** sc |-> ss1 **
        pure (SZ.v t <= 7 /\ odone sq sv so1 (SZ.v h) (SZ.v t) 0)
      decreases (7 - SZ.v !row)
    {
      let t = !row;
      att_row qk qkv out sc t h scale mask;
      with so2. assert (out |-> so2);
      odone_row sq sv so2 (SZ.v h) (SZ.v t);
      row := t +^ 1sz;
    };
    with so2. assert (out |-> so2);
    odone_head sq sv so2 (SZ.v h);
    head := h +^ 1sz;
  };
  with h so2. assert (head |-> h ** out |-> so2);
  assert pure (SZ.v h == 16);
  assert pure (odone sq sv so2 16 0 0);
  odone_final sq sv so2;
  assert pure (forall (qr vr : S.seq real). chest1_to_seq sq %~ qr /\ chest1_to_seq sv %~ vr ==>
            chest1_to_seq so2 %~ attn_values (softmax7 (attn_scores (qpart qr) (kpart qr))) (vpart vr));
}

#pop-options

(**** Simple ****)


open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major, l2_col_major, l1_forward, c_l2_row_major, c_l2_col_major }
open Kuiper.Float.Casts
open KPlay.SharedSpec

(* Layout instances whose static dimension is given by an erased nat, realized at runtime. *)
inline_for_extraction noextract
let crm (m : erased nat{SZ.fits m}) (n : erased nat) (nz : SZ.t{SZ.v nz == reveal n /\ SZ.fits (m * n)})
  : ctlayout (l2_row_major m n)
  = coerce_eq () (c_l2_row_major m nz)

inline_for_extraction noextract
let ccm (m : erased nat) (mz : SZ.t{SZ.v mz == reveal m}) (n : erased nat{SZ.fits n /\ SZ.fits (m * n)})
  : ctlayout (l2_col_major m n)
  = coerce_eq () (c_l2_col_major mz n)

noextract
let acc2_upd (#et : Type) (#m #n : nat) (d : chest2 et m n) (i : natlt m) (k : natlt n) (v : et)
  (i' : natlt m) (k' : natlt n)
  : Lemma (acc2 (upd d (i, (k, ())) v) i' k' == (if i' = i && k' = k then v else acc2 d i' k'))
          [SMTPat (acc2 (upd d (i, (k, ())) v) i' k')]
  = ()

noextract
let acc1_upd_s (#et : Type) (#m : nat) (d : chest1 et m) (i : natlt m) (v : et) (i' : natlt m)
  : Lemma (acc1 (upd d (i, ()) v) i' == (if i' = i then v else acc1 d i'))
          [SMTPat (acc1 (upd d (i, ()) v) i')]
  = ()

noextract
let row_idx (off iv k rr kk n : nat)
  : Lemma (requires iv < rr /\ k < kk /\ off + rr * kk <= n)
          (ensures off + iv * kk + k < n /\ off + (iv + 1) * kk <= n /\
                   (iv + 1) * kk == iv * kk + kk)
  = ML.lemma_mult_le_right kk (iv + 1) rr;
    ML.distributivity_add_left iv 1 kk


noextract
let divmod_idx (t r rr : nat)
  : Lemma (requires r < rr)
          (ensures (t * rr + r) / rr == t /\ (t * rr + r) % rr == r)
  = ML.lemma_div_plus r t rr;
    ML.lemma_mod_plus r t rr;
    ML.small_div r rr;
    ML.small_mod r rr

noextract
let div7 (i : nat) (r : pos)
  : Lemma (requires i < 7 * r) (ensures i / r < 7) [SMTPat (i / r); SMTPat (7 * r)] =
  if i / r >= 7 then (ML.lemma_div_mod i r; ML.lemma_mult_le_right r 7 (i / r))

(* ---------------- wcopy ---------------- *)

noextract
let wc_inv (#n #m #k : nat) (s : chest1 bf16 n) (off : nat) (d : chest2 bf16 m k) (iv kv : nat) : prop =
  forall (i : natlt m) (j : natlt k). (i < iv \/ (i == iv /\ j < kv)) ==>
    acc2 d i j == bat s (off + i * k + j)

noextract
let wc_step (#n #m #k : nat) (s : chest1 bf16 n) (off : nat) (d : chest2 bf16 m k)
  (iv : natlt m) (kv : natlt k) (x : bf16)
  : Lemma (requires wc_inv s off d iv kv /\ off + iv * k + kv < n /\ x == acc1 s (off + iv * k + kv))
          (ensures wc_inv s off (upd d (iv, (kv, ())) x) iv (kv + 1))
  = ()

noextract
let wc_row (#n #m #k : nat) (s : chest1 bf16 n) (off : nat) (d : chest2 bf16 m k) (iv : nat)
  : Lemma (requires wc_inv s off d iv k) (ensures wc_inv s off d (iv + 1) 0)
  = ()

noextract
let wc_fin (#n #m #k : nat) (s : chest1 bf16 n) (off : nat) (d : chest2 bf16 m k)
  : Lemma (requires wc_inv s off d m 0)
          (ensures forall (i : natlt m) (j : natlt k). acc2 d i j == bat s (off + i * k + j))
  = ()

#push-options "--z3rlimit 40 --fuel 0 --ifuel 0"
inline_for_extraction noextract
fn wcopy
  (rt kt : szp)
  (#n : erased nat{SZ.fits n})
  (src : array1 bf16 (l1_forward n))
  (off : SZ.t{SZ.v off + 16 * rt * 16 * kt <= n})
  (dst : array2 bf16 (l2_row_major (16 * rt) (16 * kt)))
  (#f : perm)
  (#s : chest1 bf16 n)
  (#d : chest2 bf16 (16 * rt) (16 * kt))
  preserves gpu
  preserves src |-> Frac f s
  requires dst |-> d
  ensures exists* (d' : chest2 bf16 (16 * rt) (16 * kt)).
    dst |-> d' **
    pure (forall (i : natlt (16 * rt)) (k : natlt (16 * kt)).
      acc2 d' i k == bat s (SZ.v off + i * (16 * kt) + k))
{
  assert pure ((16 * rt) * (16 * kt) == 16 * rt * 16 * kt);
  let kk : SZ.t = 16sz *^ kt;
  let rr : SZ.t = 16sz *^ rt;
  let inst = crm (16 * rt) (16 * kt) kk;
  let mut i : sz = 0sz;
  let mut base : sz = off;
  while (!i <^ rr)
    invariant exists* (iv bv : sz) (dv : chest2 bf16 (16 * rt) (16 * kt)).
      i |-> iv ** base |-> bv ** dst |-> dv **
      pure (SZ.v iv <= 16 * rt /\ SZ.v bv == SZ.v off + SZ.v iv * (16 * kt) /\
        wc_inv s (SZ.v off) dv (SZ.v iv) 0)
    decreases (16 * rt - SZ.v !i)
  {
    let iv = !i;
    let bv = !base;
    let mut k : sz = 0sz;
    while (!k <^ kk)
      invariant exists* (kv : sz) (dv : chest2 bf16 (16 * rt) (16 * kt)).
        k |-> kv ** dst |-> dv **
        pure (SZ.v kv <= 16 * kt /\ wc_inv s (SZ.v off) dv (SZ.v iv) (SZ.v kv))
      decreases (16 * kt - SZ.v !k)
    {
      let kv = !k;
      with dv. assert (dst |-> dv);
      row_idx (SZ.v off) (SZ.v iv) (SZ.v kv) (16 * rt) (16 * kt) n;
      let x = Kuiper.Tensor.read src (((bv +^ kv) <: szlt n), ()) #f;
      wc_step s (SZ.v off) dv (SZ.v iv) (SZ.v kv) x;
      Kuiper.Tensor.write #_ #_ #_ #_ #inst dst (cidx2 (iv <: szlt (16 * rt)) (kv <: szlt (16 * kt))) x;
      k := kv +^ 1sz;
    };
    with dv. assert (dst |-> dv);
    wc_row s (SZ.v off) dv (SZ.v iv);
    row_idx (SZ.v off) (SZ.v iv) 0 (16 * rt) (16 * kt) n;
    base := bv +^ kk;
    i := iv +^ 1sz;
  };
  with dv. assert (dst |-> dv);
  wc_fin s (SZ.v off) dv;
}
#pop-options

(* ---------------- pad ---------------- *)

noextract
let pd_inv (#kk : nat) (s : chest1 bf16 (7 * kk)) (d0 d : chest2 bf16 kk 8) (tv kv : nat) : prop =
  (forall (k : natlt kk) (t : natlt 8). (t < tv \/ (t == tv /\ k < kv)) ==>
    acc2 d k t == bat s (t * kk + k)) /\
  (forall (k : natlt kk) (t : natlt 8). (t > tv \/ (t == tv /\ k >= kv)) ==>
    acc2 d k t == acc2 d0 k t)

noextract
let pd_step (#kk : nat) (s : chest1 bf16 (7 * kk)) (d0 d : chest2 bf16 kk 8)
  (tv : natlt 7) (kv : natlt kk) (x : bf16)
  : Lemma (requires pd_inv s d0 d tv kv /\ tv * kk + kv < 7 * kk /\ x == acc1 s (tv * kk + kv))
          (ensures pd_inv s d0 (upd d (kv, (tv, ())) x) tv (kv + 1))
  = ()

noextract
let pd_row (#kk : nat) (s : chest1 bf16 (7 * kk)) (d0 d : chest2 bf16 kk 8) (tv : nat)
  : Lemma (requires pd_inv s d0 d tv kk) (ensures pd_inv s d0 d (tv + 1) 0)
  = ()

noextract
let pd_fin (#kk : nat) (s : chest1 bf16 (7 * kk)) (d0 d : chest2 bf16 kk 8)
  : Lemma (requires pd_inv s d0 d 7 0)
          (ensures (forall (k : natlt kk) (t : nat{t < 7}). acc2 d k t == bat s (t * kk + k)) /\
                   (forall (k : natlt kk). acc2 d k 7 == acc2 d0 k 7))
  = ()

#push-options "--z3rlimit 40 --fuel 0 --ifuel 0"
inline_for_extraction noextract
fn pad
  (kt : szp)
  (src : array1 bf16 (l1_forward (7 * (16 * kt))))
  (dst : array2 bf16 (l2_col_major (16 * kt) 8))
  (#f : perm)
  (#s : chest1 bf16 (7 * (16 * kt)))
  (#d : chest2 bf16 (16 * kt) 8)
  preserves gpu
  preserves src |-> Frac f s
  requires dst |-> d
  requires pure (SZ.fits (16 * kt * 8))
  ensures exists* (d' : chest2 bf16 (16 * kt) 8).
    dst |-> d' **
    pure ((forall (k : natlt (16 * kt)) (t : nat{t < 7}).
      acc2 d' k t == bat s (t * (16 * kt) + k)) /\
      (forall (k : natlt (16 * kt)). acc2 d' k 7 == acc2 d k 7))
{
  let kk : SZ.t = 16sz *^ kt;
  let inst = ccm (16 * kt) kk 8;
  let mut t : sz = 0sz;
  let mut base : sz = 0sz;
  while (!t <^ 7sz)
    invariant exists* (tv bv : sz) (dv : chest2 bf16 (16 * kt) 8).
      t |-> tv ** base |-> bv ** dst |-> dv **
      pure (SZ.v tv <= 7 /\ SZ.v bv == SZ.v tv * (16 * kt) /\ pd_inv s d dv (SZ.v tv) 0)
    decreases (7 - SZ.v !t)
  {
    let tv = !t;
    let bv = !base;
    let mut k : sz = 0sz;
    while (!k <^ kk)
      invariant exists* (kv : sz) (dv : chest2 bf16 (16 * kt) 8).
        k |-> kv ** dst |-> dv **
        pure (SZ.v kv <= 16 * kt /\ pd_inv s d dv (SZ.v tv) (SZ.v kv))
      decreases (16 * kt - SZ.v !k)
    {
      let kv = !k;
      with dv. assert (dst |-> dv);
      row_idx 0 (SZ.v tv) (SZ.v kv) 7 (16 * kt) (7 * (16 * kt));
      let x = Kuiper.Tensor.read src (((bv +^ kv) <: szlt (7 * (16 * kt))), ()) #f;
      pd_step s d dv (SZ.v tv) (SZ.v kv) x;
      Kuiper.Tensor.write #_ #_ #_ #_ #inst dst (cidx2 (kv <: szlt (16 * kt)) (tv <: szlt 8)) x;
      k := kv +^ 1sz;
    };
    with dv. assert (dst |-> dv);
    pd_row s d dv (SZ.v tv);
    row_idx 0 (SZ.v tv) 0 7 (16 * kt) (7 * (16 * kt));
    base := bv +^ kk;
    t := tv +^ 1sz;
  };
  with dv. assert (dst |-> dv);
  pd_fin s d dv;
}
#pop-options

(* ---------------- convert ---------------- *)

noextract
let upto (#m : nat) (o : chest1 bf16 m) (f : natlt m -> GTot bf16) (n : nat) : prop =
  forall (i : natlt m). i < n ==> acc1 o i == f i

noextract
let upto_step (#m : nat) (o : chest1 bf16 m) (f : natlt m -> GTot bf16) (n : natlt m) (y : bf16)
  : Lemma (requires upto o f n /\ y == f n) (ensures upto (upd o (n, ()) y) f (n + 1))
  = introduce forall (i : natlt m). i < n + 1 ==> acc1 (upd o (n, ()) y) i == f i
    with introduce _ ==> _ with acc1_upd_s o n y i

[@@"opaque_to_smt"]
noextract
let cvt_f (#r : pos) (ec : chest2 float r 8) (i : natlt (7 * r)) : GTot bf16 =
  div7 i r; fcast (acc2 ec (i % r) (i / r))

noextract
let cvt_upto (#r : pos) (ec : chest2 float r 8) (o : chest1 bf16 (7 * r)) (n : nat) : prop =
  upto o (cvt_f ec) n

noextract
let cvt_step (#r : pos) (ec : chest2 float r 8) (o : chest1 bf16 (7 * r))
  (t : natlt 7) (rv : natlt r) (y : bf16)
  : Lemma (requires cvt_upto ec o (t * r + rv) /\ t * r + rv < 7 * r /\ y == fcast (acc2 ec rv t))
          (ensures cvt_upto ec (upd o (t * r + rv, ()) y) (t * r + rv + 1))
  = divmod_idx t rv r;
    reveal_opaque (`%cvt_f) (cvt_f ec (t * r + rv));
    upto_step o (cvt_f ec) (t * r + rv) y

noextract
let cvt_fin (#r : pos) (ec : chest2 float r 8) (o : chest1 bf16 (7 * r))
  : Lemma (requires cvt_upto ec o (7 * r))
          (ensures forall (i : natlt (7 * r)). acc1 o i == (fcast (acc2 ec (i % r) (i / r)) <: bf16))
  = introduce forall (i : natlt (7 * r)). acc1 o i == (fcast (acc2 ec (i % r) (i / r)) <: bf16)
    with reveal_opaque (`%cvt_f) (cvt_f ec i)

#push-options "--z3rlimit 40 --fuel 0 --ifuel 0"
inline_for_extraction noextract
fn convert
  (rt : szp)
  (c : array2 float (l2_row_major (16 * rt) 8))
  (out : array1 bf16 (l1_forward (7 * (16 * rt))))
  (#f : perm)
  (#ec : chest2 float (16 * rt) 8)
  (#o : chest1 bf16 (7 * (16 * rt)))
  preserves gpu
  preserves c |-> Frac f ec
  requires out |-> o
  requires pure (SZ.fits (16 * rt * 8))
  ensures exists* (o' : chest1 bf16 (7 * (16 * rt))).
    out |-> o' **
    pure (forall (i : natlt (7 * (16 * rt))).
      acc1 o' i == (fcast (acc2 ec (i % (16 * rt)) (i / (16 * rt))) <: bf16))
{
  let rr : SZ.t = 16sz *^ rt;
  let inst = crm (16 * rt) 8 8sz;
  let mut t : sz = 0sz;
  let mut base : sz = 0sz;
  while (!t <^ 7sz)
    invariant exists* (tv bv : sz) (ov : chest1 bf16 (7 * (16 * rt))).
      t |-> tv ** base |-> bv ** out |-> ov **
      pure (SZ.v tv <= 7 /\ SZ.v bv == SZ.v tv * (16 * rt) /\ cvt_upto ec ov (SZ.v bv))
    decreases (7 - SZ.v !t)
  {
    let tv = !t;
    let bv = !base;
    let mut r : sz = 0sz;
    while (!r <^ rr)
      invariant exists* (rv : sz) (ov : chest1 bf16 (7 * (16 * rt))).
        r |-> rv ** out |-> ov **
        pure (SZ.v rv <= 16 * rt /\ cvt_upto ec ov (SZ.v bv + SZ.v rv))
      decreases (16 * rt - SZ.v !r)
    {
      let rv = !r;
      with ov. assert (out |-> ov);
      row_idx 0 (SZ.v tv) (SZ.v rv) 7 (16 * rt) (7 * (16 * rt));
      let x = Kuiper.Tensor.read #_ #_ #_ #_ #inst c (cidx2 (rv <: szlt (16 * rt)) (tv <: szlt 8)) #f;
      let y : bf16 = fcast x;
      cvt_step ec ov (SZ.v tv) (SZ.v rv) y;
      Kuiper.Tensor.write out (((bv +^ rv) <: szlt (7 * (16 * rt))), ()) y;
      r := rv +^ 1sz;
    };
    row_idx 0 (SZ.v tv) 0 7 (16 * rt) (7 * (16 * rt));
    base := bv +^ rr;
    t := tv +^ 1sz;
  };
  with ov. assert (out |-> ov);
  assert pure (cvt_upto ec ov (7 * (16 * rt)));
  cvt_fin ec ov;
  assert pure (forall (i : natlt (7 * (16 * rt))).
      acc1 ov i == (fcast (acc2 ec (i % (16 * rt)) (i / (16 * rt))) <: bf16));
}
#pop-options

(* ---------------- convert_add (split-K) ---------------- *)

[@@"opaque_to_smt"]
noextract
let cvta_f (#r : pos) (e1 e2 : chest2 float r 8) (i : natlt (7 * r)) : GTot bf16 =
  div7 i r; fcast (add (acc2 e1 (i % r) (i / r)) (acc2 e2 (i % r) (i / r)))

noextract
let cvta_upto (#r : pos) (e1 e2 : chest2 float r 8) (o : chest1 bf16 (7 * r)) (n : nat) : prop =
  upto o (cvta_f e1 e2) n

noextract
let cvta_step (#r : pos) (e1 e2 : chest2 float r 8) (o : chest1 bf16 (7 * r))
  (t : natlt 7) (rv : natlt r) (y : bf16)
  : Lemma (requires cvta_upto e1 e2 o (t * r + rv) /\ t * r + rv < 7 * r /\
                    y == fcast (add (acc2 e1 rv t) (acc2 e2 rv t)))
          (ensures cvta_upto e1 e2 (upd o (t * r + rv, ()) y) (t * r + rv + 1))
  = divmod_idx t rv r;
    reveal_opaque (`%cvta_f) (cvta_f e1 e2 (t * r + rv));
    upto_step o (cvta_f e1 e2) (t * r + rv) y

noextract
let cvta_fin (#r : pos) (e1 e2 : chest2 float r 8) (o : chest1 bf16 (7 * r))
  : Lemma (requires cvta_upto e1 e2 o (7 * r))
          (ensures forall (i : natlt (7 * r)).
            acc1 o i == (fcast (add (acc2 e1 (i % r) (i / r)) (acc2 e2 (i % r) (i / r))) <: bf16))
  = introduce forall (i : natlt (7 * r)).
      acc1 o i == (fcast (add (acc2 e1 (i % r) (i / r)) (acc2 e2 (i % r) (i / r))) <: bf16)
    with reveal_opaque (`%cvta_f) (cvta_f e1 e2 i)

#push-options "--z3rlimit 40 --fuel 0 --ifuel 0"
inline_for_extraction noextract
fn convert_add
  (rt : szp)
  (c1 : array2 float (l2_row_major (16 * rt) 8))
  (c2 : array2 float (l2_row_major (16 * rt) 8))
  (out : array1 bf16 (l1_forward (7 * (16 * rt))))
  (#f1 #f2 : perm)
  (#e1 #e2 : chest2 float (16 * rt) 8)
  (#o : chest1 bf16 (7 * (16 * rt)))
  preserves gpu
  preserves c1 |-> Frac f1 e1
  preserves c2 |-> Frac f2 e2
  requires out |-> o
  requires pure (SZ.fits (16 * rt * 8))
  ensures exists* (o' : chest1 bf16 (7 * (16 * rt))).
    out |-> o' **
    pure (forall (i : natlt (7 * (16 * rt))).
      acc1 o' i == (fcast (add (acc2 e1 (i % (16 * rt)) (i / (16 * rt)))
                               (acc2 e2 (i % (16 * rt)) (i / (16 * rt)))) <: bf16))
{
  let rr : SZ.t = 16sz *^ rt;
  let inst = crm (16 * rt) 8 8sz;
  let mut t : sz = 0sz;
  let mut base : sz = 0sz;
  while (!t <^ 7sz)
    invariant exists* (tv bv : sz) (ov : chest1 bf16 (7 * (16 * rt))).
      t |-> tv ** base |-> bv ** out |-> ov **
      pure (SZ.v tv <= 7 /\ SZ.v bv == SZ.v tv * (16 * rt) /\ cvta_upto e1 e2 ov (SZ.v bv))
    decreases (7 - SZ.v !t)
  {
    let tv = !t;
    let bv = !base;
    let mut r : sz = 0sz;
    while (!r <^ rr)
      invariant exists* (rv : sz) (ov : chest1 bf16 (7 * (16 * rt))).
        r |-> rv ** out |-> ov **
        pure (SZ.v rv <= 16 * rt /\ cvta_upto e1 e2 ov (SZ.v bv + SZ.v rv))
      decreases (16 * rt - SZ.v !r)
    {
      let rv = !r;
      with ov. assert (out |-> ov);
      row_idx 0 (SZ.v tv) (SZ.v rv) 7 (16 * rt) (7 * (16 * rt));
      let x1 = Kuiper.Tensor.read #_ #_ #_ #_ #inst c1 (cidx2 (rv <: szlt (16 * rt)) (tv <: szlt 8)) #f1;
      let x2 = Kuiper.Tensor.read #_ #_ #_ #_ #inst c2 (cidx2 (rv <: szlt (16 * rt)) (tv <: szlt 8)) #f2;
      let y : bf16 = fcast (add x1 x2);
      cvta_step e1 e2 ov (SZ.v tv) (SZ.v rv) y;
      Kuiper.Tensor.write out (((bv +^ rv) <: szlt (7 * (16 * rt))), ()) y;
      r := rv +^ 1sz;
    };
    row_idx 0 (SZ.v tv) 0 7 (16 * rt) (7 * (16 * rt));
    base := bv +^ rr;
    t := tv +^ 1sz;
  };
  with ov. assert (out |-> ov);
  assert pure (cvta_upto e1 e2 ov (7 * (16 * rt)));
  cvta_fin e1 e2 ov;
  assert pure (forall (i : natlt (7 * (16 * rt))).
      acc1 ov i == (fcast (add (acc2 e1 (i % (16 * rt)) (i / (16 * rt)))
                               (acc2 e2 (i % (16 * rt)) (i / (16 * rt)))) <: bf16));
}
#pop-options

(* ---------------- silu ---------------- *)

noextract
let silu_f (g u : bf16) : bf16 =
  let gate : float = fcast g in
  let act : bf16 = fcast (div gate (add one (fexp (neg gate)))) in
  fcast (mul (fcast act <: float) (fcast u <: float))

noextract
let silu_f_approx (g u : bf16) (gr ur : real)
  : Lemma (requires g %~ gr /\ u %~ ur)
          (ensures silu_f g u %~ (silu gr *. ur))
  = let gate : float = fcast g in
    assert (gate %~ gr);
    let ng = neg gate in
    assert (has_type Kuiper.Float32.is_floating_real_like (floating_real_like float));
    assert (ng %~ (0.0R -. gr));
    let e = fexp ng in
    assert (e %~ FStar.Math.Exp.exp (0.0R -. gr));
    let den = add one e in
    a_add (one #float) e 1.0R (FStar.Math.Exp.exp (0.0R -. gr));
    FStar.Math.Exp.exp_positive (0.0R -. gr);
    let dr = 1.0R +. FStar.Math.Exp.exp (0.0R -. gr) in
    assert (den %~ dr);
    assert (div gate den %~ (gr /. dr));
    assert (gr /. dr == silu gr);
    let act : bf16 = fcast (div gate den) in
    assert (act %~ silu gr);
    let af : float = fcast act in
    let uf : float = fcast u in
    assert (af %~ silu gr);
    assert (uf %~ ur);
    a_mul af uf (silu gr) ur

[@@"opaque_to_smt"]
noextract
let sl_f (s : chest1 bf16 43008) (i : natlt 21504) : GTot bf16 =
  silu_f (bat s ((i / 3072) * 6144 + i % 3072)) (bat s ((i / 3072) * 6144 + 3072 + i % 3072))

noextract
let sl_upto (s : chest1 bf16 43008) (o : chest1 bf16 21504) (n : nat) : prop =
  upto o (sl_f s) n

noextract
let sl_step (s : chest1 bf16 43008) (o : chest1 bf16 21504) (row : natlt 7) (col : natlt 3072) (y : bf16)
  : Lemma (requires sl_upto s o (row * 3072 + col) /\ row * 6144 + 3072 + col < 43008 /\
                    row * 3072 + col < 21504 /\
                    y == silu_f (acc1 s (row * 6144 + col)) (acc1 s (row * 6144 + 3072 + col)))
          (ensures sl_upto s (upd o (row * 3072 + col, ()) y) (row * 3072 + col + 1))
  = divmod_idx row col 3072;
    reveal_opaque (`%sl_f) (sl_f s (row * 3072 + col));
    upto_step o (sl_f s) (row * 3072 + col) y

noextract
let sl_fin (s : chest1 bf16 43008) (o : chest1 bf16 21504)
  : Lemma (requires sl_upto s o 21504)
          (ensures forall (i : natlt 21504).
            acc1 o i == silu_f (bat s ((i / 3072) * 6144 + i % 3072))
                               (bat s ((i / 3072) * 6144 + 3072 + i % 3072)))
  = introduce forall (i : natlt 21504).
      acc1 o i == silu_f (bat s ((i / 3072) * 6144 + i % 3072))
                         (bat s ((i / 3072) * 6144 + 3072 + i % 3072))
    with reveal_opaque (`%sl_f) (sl_f s i)

#push-options "--z3rlimit 40 --fuel 0 --ifuel 0"
inline_for_extraction noextract
fn silu_k
  (gu : array1 bf16 (l1_forward 43008))
  (out : array1 bf16 (l1_forward 21504))
  (#f : perm)
  (#s : chest1 bf16 43008)
  (#o : chest1 bf16 21504)
  preserves gpu
  preserves gu |-> Frac f s
  requires out |-> o
  ensures exists* (o' : chest1 bf16 21504).
    out |-> o' **
    pure (forall (i : natlt 21504).
      acc1 o' i == silu_f (bat s ((i / 3072) * 6144 + i % 3072))
                          (bat s ((i / 3072) * 6144 + 3072 + i % 3072)))
{
  let mut row : sz = 0sz;
  let mut gbase : sz = 0sz;
  let mut obase : sz = 0sz;
  while (!row <^ 7sz)
    invariant exists* (rv gb ob : sz) (ov : chest1 bf16 21504).
      row |-> rv ** gbase |-> gb ** obase |-> ob ** out |-> ov **
      pure (SZ.v rv <= 7 /\ SZ.v gb == SZ.v rv * 6144 /\ SZ.v ob == SZ.v rv * 3072 /\
        sl_upto s ov (SZ.v ob))
    decreases (7 - SZ.v !row)
  {
    let rv = !row;
    let gb = !gbase;
    let ob = !obase;
    let mut col : sz = 0sz;
    while (!col <^ 3072sz)
      invariant exists* (cv : sz) (ov : chest1 bf16 21504).
        col |-> cv ** out |-> ov **
        pure (SZ.v cv <= 3072 /\ sl_upto s ov (SZ.v ob + SZ.v cv))
      decreases (3072 - SZ.v !col)
    {
      let cv = !col;
      with ov. assert (out |-> ov);
      row_idx 0 (SZ.v rv) (SZ.v cv) 7 3072 21504;
      row_idx 0 (SZ.v rv) (SZ.v cv + 3072) 7 6144 43008;
      let gi = gb +^ cv;
      let g = Kuiper.Tensor.read gu ((gi <: szlt 43008), ()) #f;
      let gate : float = fcast g;
      let act : bf16 = fcast (div gate (add one (fexp (neg gate))));
      let u = Kuiper.Tensor.read gu (((gi +^ 3072sz) <: szlt 43008), ()) #f;
      let y : bf16 = fcast (mul (fcast act <: float) (fcast u <: float));
      sl_step s ov (SZ.v rv) (SZ.v cv) y;
      Kuiper.Tensor.write out (((ob +^ cv) <: szlt 21504), ()) y;
      col := cv +^ 1sz;
    };
    row_idx 0 (SZ.v rv) 0 7 3072 21504;
    row_idx 0 (SZ.v rv) 0 7 6144 43008;
    gbase := gb +^ 6144sz;
    obase := ob +^ 3072sz;
    row := rv +^ 1sz;
  };
  with ov. assert (out |-> ov);
  assert pure (sl_upto s ov 21504);
  sl_fin s ov;
  assert pure (forall (i : natlt 21504).
      acc1 ov i == silu_f (bat s ((i / 3072) * 6144 + i % 3072))
                          (bat s ((i / 3072) * 6144 + 3072 + i % 3072)));
}
#pop-options

(**** Final ****)

open FStar.Real
open Kuiper
open Kuiper.Chest
open Kuiper.Float.Casts
open KPlay.SharedSpec

[@@"opaque_to_smt"]
noextract let wc_ok (#rt #kt:pos) (#n0:nat) (s0:chest1 bf16 n0) (wo:nat) (wc:chest2 bf16 (16 * rt) (16 * kt)) : prop =
  forall (i:natlt (16 * rt)) (k:natlt (16 * kt)). acc2 wc i k == bat s0 (wo + i * (16 * kt) + k)

[@@"opaque_to_smt"]
noextract let pad_ok (#kt:pos) (sx:chest1 bf16 (7 * (16 * kt))) (xc:chest2 bf16 (16 * kt) 8) : prop =
  forall (k:natlt (16 * kt)) (t:nat{t < 7}). acc2 xc k t == bat sx (t * (16 * kt) + k)

[@@"opaque_to_smt"]
noextract let cv_ok (#rt #kt:pos) (wc:chest2 bf16 (16 * rt) (16 * kt)) (xc:chest2 bf16 (16 * kt) 8)
  (c0:chest2 float (16 * rt) 8) (out:chest1 bf16 (7 * (16 * rt))) : prop =
  forall (i:natlt (7 * (16 * rt))).
    acc1 out i == (fcast (acc2 (cmod wc xc c0 0 kt rt) (i % (16 * rt)) (i / (16 * rt))) <: bf16)

[@@"opaque_to_smt"]
noextract let cva_ok (#rt #kt:pos) (wc:chest2 bf16 (16 * rt) (16 * kt)) (xc:chest2 bf16 (16 * kt) 8)
  (c0 c0':chest2 float (16 * rt) 8) (kh:nat) (out:chest1 bf16 (7 * (16 * rt))) : prop =
  forall (i:natlt (7 * (16 * rt))).
    acc1 out i == (fcast (add (acc2 (cmod wc xc c0 0 kh rt) (i % (16 * rt)) (i / (16 * rt)))
                              (acc2 (cmod wc xc c0' kh kt rt) (i % (16 * rt)) (i / (16 * rt)))) <: bf16)

noextract let att_ok (sq:chest1 bf16 21504) (sv:chest1 bf16 28672) (so:chest1 bf16 14336) : prop =
  forall (qr vr : S.seq real). chest1_to_seq sq %~ qr /\ chest1_to_seq sv %~ vr ==>
    chest1_to_seq so %~ attn_values (softmax7 (attn_scores (qpart qr) (kpart qr))) (vpart vr)

[@@"opaque_to_smt"]
noextract let silu_ok (s:chest1 bf16 43008) (o:chest1 bf16 21504) : prop =
  forall (i:nat). i < 21504 ==>
    acc1 o i == silu_f (bat s ((i / 3072) * 6144 + i % 3072))
                       (bat s ((i / 3072) * 6144 + 3072 + i % 3072))

#push-options "--fuel 0 --ifuel 0 --z3rlimit 60"
noextract let silu_step (sg:chest1 bf16 43008) (sa:chest1 bf16 21504) (w n:S.seq real)
  : Lemma (requires S.length w == 6291456 /\ silu_ok sg sa /\ chest1_to_seq sg %~ linear 7 6144 1024 w n)
          (ensures chest1_to_seq sa %~
                   silu_product (linear 7 3072 1024 (window w 0 3145728) n)
                                (linear 7 3072 1024 (window w 3145728 3145728) n))
  = reveal_opaque (`%silu_ok) silu_ok;
    let gr = linear 7 6144 1024 w n in
    seq_ap_elim sg gr;
    let aux (i:nat{i < 21504}) : Lemma (acc1 sa i %~ (silu (at gr ((i / 3072) * 6144 + i % 3072)) *.
                              at gr ((i / 3072) * 6144 + 3072 + i % 3072))) =
      gu_idx i;
      let a = (i / 3072) * 6144 + i % 3072 in
      let b = (i / 3072) * 6144 + 3072 + i % 3072 in
      assert (bat sg a %~ at gr a);
      assert (bat sg b %~ at gr b);
      silu_f_approx (bat sg a) (bat sg b) (at gr a) (at gr b) in
    Classical.forall_intro aux;
    silu_stage sa w n
#pop-options


[@@"opaque_to_smt"]
noextract let qk_ok (#n0:nat) (s0:chest1 bf16 n0) (off:nat) (sqkv:chest1 bf16 28672) (s3:chest1 bf16 1792)
  (sqk:chest1 bf16 21504) : prop =
  forall (i:nat). i < 21504 ==> acc1 sqk i == qk_m s0 off sqkv s3 i

[@@"opaque_to_smt"]
noextract let rn_ok (#n0:nat) (s0:chest1 bf16 n0) (noff:nat) (sr su s1 s2:chest1 bf16 7168) : prop =
  forall (i:nat). i < 7168 ==> acc1 s1 i == sum_m sr su i /\ acc1 s2 i == rn_m s0 noff sr su i

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
noextract let pstage (#rt #kt:pos) (#n0:nat) (s0:chest1 bf16 n0) (wo:nat) (sx:chest1 bf16 (7 * (16 * kt)))
  (wc:chest2 bf16 (16 * rt) (16 * kt)) (xc:chest2 bf16 (16 * kt) 8) (c0:chest2 float (16 * rt) 8)
  (out:chest1 bf16 (7 * (16 * rt))) (rw xr:S.seq real)
  : Lemma (requires wo + (16 * rt) * (16 * kt) <= n0 /\
             chest1_to_seq s0 %~ rw /\ chest1_to_seq sx %~ xr /\
             wc_ok #rt #kt s0 wo wc /\ pad_ok #kt sx xc /\ cv_ok #rt #kt wc xc c0 out)
          (ensures chest1_to_seq out %~
                   linear 7 (16 * rt) (16 * kt) (window rw wo ((16 * rt) * (16 * kt))) xr)
  = reveal_opaque (`%wc_ok) (wc_ok #rt #kt #n0);
    reveal_opaque (`%pad_ok) (pad_ok #kt);
    reveal_opaque (`%cv_ok) (cv_ok #rt #kt);
    proj_stage #rt #kt s0 wo sx wc xc c0 out rw xr

noextract let pstage_split (#rt #kt:pos) (#n0:nat) (s0:chest1 bf16 n0) (wo:nat) (sx:chest1 bf16 (7 * (16 * kt)))
  (wc:chest2 bf16 (16 * rt) (16 * kt)) (xc:chest2 bf16 (16 * kt) 8) (c0 c0':chest2 float (16 * rt) 8)
  (kh:nat{kh <= kt}) (out:chest1 bf16 (7 * (16 * rt))) (rw xr:S.seq real)
  : Lemma (requires wo + (16 * rt) * (16 * kt) <= n0 /\
             chest1_to_seq s0 %~ rw /\ chest1_to_seq sx %~ xr /\
             wc_ok #rt #kt s0 wo wc /\ pad_ok #kt sx xc /\ cva_ok #rt #kt wc xc c0 c0' kh out)
          (ensures chest1_to_seq out %~
                   linear 7 (16 * rt) (16 * kt) (window rw wo ((16 * rt) * (16 * kt))) xr)
  = reveal_opaque (`%wc_ok) (wc_ok #rt #kt #n0);
    reveal_opaque (`%pad_ok) (pad_ok #kt);
    reveal_opaque (`%cva_ok) (cva_ok #rt #kt);
    proj_stage_split #rt #kt s0 wo sx wc xc c0 c0' kh out rw xr

noextract let qstage (#n0:nat) (s0:chest1 bf16 n0) (off:nat) (sv:chest1 bf16 28672) (s3:chest1 bf16 1792)
  (sq:chest1 bf16 21504) (rw vr rr:S.seq real)
  : Lemma (requires off + 6293760 <= n0 /\
             chest1_to_seq s0 %~ rw /\ chest1_to_seq sv %~ vr /\ chest1_to_seq s3 %~ rr /\
             qk_ok s0 off sv s3 sq)
          (ensures chest1_to_seq sq %~ qr_seq vr rw rr off)
  = reveal_opaque (`%qk_ok) (qk_ok #n0);
    qk_stage s0 off sv s3 sq rw vr rr

noextract let rstage (#n0:nat) (sw:chest1 bf16 n0) (noff:nat) (sr su s1 s2:chest1 bf16 7168)
  (rw rr ur:S.seq real)
  : Lemma (requires noff + 1024 <= n0 /\
             chest1_to_seq sw %~ rw /\ chest1_to_seq sr %~ rr /\ chest1_to_seq su %~ ur /\
             rn_ok sw noff sr su s1 s2)
          (ensures chest1_to_seq s1 %~ vadd rr ur /\
                   chest1_to_seq s2 %~ rmsnorm_rows 7 1024 (vadd rr ur) (window rw noff 1024))
  = reveal_opaque (`%rn_ok) (rn_ok #n0);
    rn_stage sw noff sr su s1 s2 rw rr ur

noextract let astage (sq:chest1 bf16 21504) (sv:chest1 bf16 28672) (so:chest1 bf16 14336) (qr vr:S.seq real)
  : Lemma (requires att_ok sq sv so /\ chest1_to_seq sq %~ qr /\ chest1_to_seq sv %~ vr)
          (ensures chest1_to_seq so %~ attn_values (softmax7 (attn_scores (qpart qr) (kpart qr))) (vpart vr))
  = ()
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 200"
noextract let decoder_real (#n0:nat) (s0:chest1 bf16 n0) (s1 s2:chest1 bf16 7168) (s3:chest1 bf16 1792)
  (off a5:nat)
  (wqkv:chest2 bf16 (16 * 256) (16 * 64)) (x0:chest2 bf16 (16 * 64) 8) (c0:chest2 float (16 * 256) 8)
  (sqkv:chest1 bf16 28672) (sqk:chest1 bf16 21504) (satt:chest1 bf16 14336)
  (wo:chest2 bf16 (16 * 64) (16 * 128)) (x1:chest2 bf16 (16 * 128) 8) (c1:chest2 float (16 * 64) 8)
  (sproj:chest1 bf16 7168) (sres:chest1 bf16 7168) (snorm:chest1 bf16 7168)
  (wgu:chest2 bf16 (16 * 384) (16 * 64)) (x2:chest2 bf16 (16 * 64) 8) (c2:chest2 float (16 * 384) 8)
  (sgu:chest1 bf16 43008) (sact:chest1 bf16 21504)
  (wd:chest2 bf16 (16 * 64) (16 * 192)) (x3:chest2 bf16 (16 * 192) 8) (c3 c3':chest2 float (16 * 64) 8)
  (sproj2:chest1 bf16 7168) (so0 so1:chest1 bf16 7168)
  (rw rr rn rrot:S.seq real)
  : Lemma (requires off + layer_elements <= n0 /\ a5 + 1024 <= n0 /\
             wc_ok #256 #64 s0 (off + 2048) wqkv /\ pad_ok #64 s2 x0 /\ cv_ok #256 #64 wqkv x0 c0 sqkv /\
             qk_ok s0 off sqkv s3 sqk /\
             att_ok sqk sqkv satt /\
             wc_ok #64 #128 s0 (off + 4196352) wo /\ pad_ok #128 satt x1 /\ cv_ok #64 #128 wo x1 c1 sproj /\
             rn_ok s0 (off + 1024) s1 sproj sres snorm /\
             wc_ok #384 #64 s0 (off + 6293760) wgu /\ pad_ok #64 snorm x2 /\ cv_ok #384 #64 wgu x2 c2 sgu /\
             silu_ok sgu sact /\
             wc_ok #64 #192 s0 (off + 12585216) wd /\ pad_ok #192 sact x3 /\ cva_ok #64 #192 wd x3 c3 c3' 96 sproj2 /\
             rn_ok s0 a5 sres sproj2 so0 so1 /\
             chest1_to_seq s0 %~ rw /\ chest1_to_seq s1 %~ rr /\
             chest1_to_seq s2 %~ rn /\ chest1_to_seq s3 %~ rrot)
          (ensures (let res = layer_residual (packed_layer rw off) (window rrot 0 896) (window rrot 896 896) rr rn in
                    chest1_to_seq so0 %~ res /\
                    chest1_to_seq so1 %~ rmsnorm_rows 7 1024 res (window rw a5 1024)))
  = let lw = packed_layer rw off in
    let cos = window rrot 0 896 in
    let sin = window rrot 896 896 in
    seq_ap_elim s0 rw;
    seq_ap_elim s3 rrot;
    assert (S.length rw == n0);
    assert (S.length rrot == 1792);
    (* QKV projection *)
    let qkvr = linear 7 4096 1024 (window rw (off + 2048) 4194304) rn in
    pstage #256 #64 s0 (off + 2048) s2 wqkv x0 c0 sqkv rw rn;
    (* q/k norm + rope *)
    let qr = qr_seq qkvr rw rrot off in
    qstage s0 off sqkv s3 sqk rw qkvr rrot;
    astage sqk sqkv satt qr qkvr;
    (* attention *)
    qpart_eq rw rn rrot off;
    kpart_eq rw rn rrot off;
    vpart_eq rw rn off;
    let ar = attn_values (softmax7 (attn_scores (queries lw cos sin rn) (keys lw cos sin rn)))
                         (linear 7 1024 1024 lw.wv rn) in
    assert (chest1_to_seq satt %~ ar);
    (* o_proj *)
    pstage #64 #128 s0 (off + 4196352) satt wo x1 c1 sproj rw ar;
    let attn = attention lw cos sin rn in
    assert (chest1_to_seq sproj %~ attn);
    (* residual + post norm *)
    rstage s0 (off + 1024) s1 sproj sres snorm rw rr attn;
    let h = vadd rr attn in
    let nr = rmsnorm_rows 7 1024 h lw.post_norm in
    assert (chest1_to_seq snorm %~ nr);
    (* gate/up *)
    let wg = window rw (off + 6293760) 6291456 in
    assert (off + 6293760 + 6291456 <= S.length rw);
    assert (S.length wg == 6291456);
    pstage #384 #64 s0 (off + 6293760) snorm wgu x2 c2 sgu rw nr;
    assert (chest1_to_seq sgu %~ linear 7 6144 1024 wg nr);
    silu_step sgu sact wg nr;
    window_window rw (off + 6293760) 6291456 0 3145728;
    window_window rw (off + 6293760) 6291456 3145728 3145728;
    let act = silu_product (linear 7 3072 1024 lw.w_gate nr) (linear 7 3072 1024 lw.w_up nr) in
    assert (chest1_to_seq sact %~ act);
    (* down *)
    pstage_split #64 #192 s0 (off + 12585216) sact wd x3 c3 c3' 96 sproj2 rw act;
    let m = mlp lw nr in
    assert (chest1_to_seq sproj2 %~ m);
    rstage s0 a5 sres sproj2 so0 so1 rw h m
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 60"
noextract let decoder_ok (#n0:nat) (s0:chest1 bf16 n0) (s1 s2:chest1 bf16 7168) (s3:chest1 bf16 1792)
  (off a5:nat)
  (wqkv:chest2 bf16 (16 * 256) (16 * 64)) (x0:chest2 bf16 (16 * 64) 8) (c0:chest2 float (16 * 256) 8)
  (sqkv:chest1 bf16 28672) (sqk:chest1 bf16 21504) (satt:chest1 bf16 14336)
  (wo:chest2 bf16 (16 * 64) (16 * 128)) (x1:chest2 bf16 (16 * 128) 8) (c1:chest2 float (16 * 64) 8)
  (sproj:chest1 bf16 7168) (sres:chest1 bf16 7168) (snorm:chest1 bf16 7168)
  (wgu:chest2 bf16 (16 * 384) (16 * 64)) (x2:chest2 bf16 (16 * 64) 8) (c2:chest2 float (16 * 384) 8)
  (sgu:chest1 bf16 43008) (sact:chest1 bf16 21504)
  (wd:chest2 bf16 (16 * 64) (16 * 192)) (x3:chest2 bf16 (16 * 192) 8) (c3 c3':chest2 float (16 * 64) 8)
  (sproj2:chest1 bf16 7168) (so0 so1:chest1 bf16 7168)
  : Lemma (requires off + layer_elements <= n0 /\ a5 + 1024 <= n0 /\
             wc_ok #256 #64 s0 (off + 2048) wqkv /\ pad_ok #64 s2 x0 /\ cv_ok #256 #64 wqkv x0 c0 sqkv /\
             qk_ok s0 off sqkv s3 sqk /\
             att_ok sqk sqkv satt /\
             wc_ok #64 #128 s0 (off + 4196352) wo /\ pad_ok #128 satt x1 /\ cv_ok #64 #128 wo x1 c1 sproj /\
             rn_ok s0 (off + 1024) s1 sproj sres snorm /\
             wc_ok #384 #64 s0 (off + 6293760) wgu /\ pad_ok #64 snorm x2 /\ cv_ok #384 #64 wgu x2 c2 sgu /\
             silu_ok sgu sact /\
             wc_ok #64 #192 s0 (off + 12585216) wd /\ pad_ok #192 sact x3 /\ cva_ok #64 #192 wd x3 c3 c3' 96 sproj2 /\
             rn_ok s0 a5 sres sproj2 so0 so1)
          (ensures base_decoder_relation (chest1_to_seq s0) (chest1_to_seq s1) (chest1_to_seq s2)
                     (chest1_to_seq s3) off a5 (chest1_to_seq so0) (chest1_to_seq so1))
  = let aux (rw rr rn rrot:S.seq real)
      : Lemma (requires chest1_to_seq s0 %~ rw /\ chest1_to_seq s1 %~ rr /\
                        chest1_to_seq s2 %~ rn /\ chest1_to_seq s3 %~ rrot)
              (ensures (let res = layer_residual (packed_layer rw off) (window rrot 0 896) (window rrot 896 896) rr rn in
                        chest1_to_seq so0 %~ res /\
                        chest1_to_seq so1 %~ rmsnorm_rows 7 1024 res (window rw a5 1024)))
      = decoder_real s0 s1 s2 s3 off a5 wqkv x0 c0 sqkv sqk satt wo x1 c1 sproj sres snorm
          wgu x2 c2 sgu sact wd x3 c3 c3' sproj2 so0 so1 rw rr rn rrot in
    let aux4 (rw rr rn rrot:S.seq real)
      : Lemma (chest1_to_seq s0 %~ rw /\ chest1_to_seq s1 %~ rr /\
               chest1_to_seq s2 %~ rn /\ chest1_to_seq s3 %~ rrot ==>
               (let res = layer_residual (packed_layer rw off) (window rrot 0 896) (window rrot 896 896) rr rn in
                chest1_to_seq so0 %~ res /\
                chest1_to_seq so1 %~ rmsnorm_rows 7 1024 res (window rw a5 1024)))
      = Classical.move_requires (aux rw rr rn) rrot in
    Classical.forall_intro_4 aux4
#pop-options

(**** Mma ****)


open Kuiper
open Kuiper.Tensor
open Kuiper.TensorCore.MMA
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major, l2_col_major, l1_forward }
open Kuiper.Tensor.Tiling
open Kuiper.Float.Casts

noextract unfold
let pkpre (#rt #kt : pos)
  (w : array2 bf16 (l2_row_major (16 * rt) (16 * kt))) (fw : perm) (ew : chest2 bf16 (16 * rt) (16 * kt))
  (x : array2 bf16 (l2_col_major (16 * kt) 8)) (fx : perm) (ex : chest2 bf16 (16 * kt) 8)
  (c : array2 float (l2_row_major (16 * rt) 8)) (ec : chest2 float (16 * rt) 8)
  (tid : nat) : slprop =
  w |-> Frac (fw /. 32) ew ** x |-> Frac (fx /. 32) ex ** c |-> Frac (1.0R /. 32) ec

#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn pthread
  (rt kt : szp)
  (k0 : sz) (k1 : sz{k0 <= k1 /\ k1 <= kt})
  (#_ : squash (SZ.fits (16 * rt * 16 * kt) /\ SZ.fits (16 * kt * 8) /\ SZ.fits (16 * rt * 8)))
  (w : array2 bf16 (l2_row_major (16 * rt) (16 * kt)))
  (x : array2 bf16 (l2_col_major (16 * kt) 8))
  (c : array2 float (l2_row_major (16 * rt) 8))
  (#fw #fx : perm)
  (#ew : chest2 bf16 (16 * rt) (16 * kt))
  (#ex : chest2 bf16 (16 * kt) 8)
  (#ec : chest2 float (16 * rt) 8)
  preserves gpu
  preserves w |-> Frac fw ew
  preserves x |-> Frac fx ex
  requires c |-> Frac (1.0R /. 32) ec
  ensures c |-> Frac (1.0R /. 32) (cmod ew ex ec k0 k1 rt)
{
  let fr = alloc_fragment ();
  let mut i : sz = 0sz;
  while (!i <^ rt)
    invariant exists* (iv : sz) (vf : chest2 float 16 8).
      i |-> iv ** fr |-> vf ** pure (iv <= rt) **
      c |-> Frac (1.0R /. 32) (cmod ew ex ec k0 k1 iv)
    decreases (rt - !i)
  {
    let tr = !i;
    fill fr zero;
    let mut k : sz = k0;
    while (!k <^ k1)
      invariant exists* (kv : sz).
        k |-> kv ** pure (k0 <= kv /\ kv <= k1) **
        fr |-> kacc ew ex tr k0 kv
      decreases (k1 - !k)
    {
      let kv = !k;
      let ta = array2_extract_tile_ro' w 16 16 (SZ.v tr) (SZ.v kv);
      let tb = array2_extract_tile_ro' x 16 8 (SZ.v kv) 0;
      mma_sync ta tb fr;
      Pulse.Lib.Trade.elim_trade _ (w |-> Frac fw ew);
      Pulse.Lib.Trade.elim_trade _ (x |-> Frac fx ex);
      k := kv +^ 1sz;
    };
    let tc = array2_extract_tile_st c 16 8 (SZ.v tr) 0;
    store fr tc;
    Pulse.Lib.Forall.elim_forall (kacc ew ex tr k0 k1);
    Pulse.Lib.Trade.elim_trade _ _;
    i := tr +^ 1sz;
  };
  drop_ (fr |-> _);
}
#pop-options

inline_for_extraction noextract
fn pkf
  (rt kt : szp)
  (k0 : sz) (k1 : sz{k0 <= k1 /\ k1 <= kt})
  (#_ : squash (SZ.fits (16 * rt * 16 * kt) /\ SZ.fits (16 * kt * 8) /\ SZ.fits (16 * rt * 8)))
  (w : array2 bf16 (l2_row_major (16 * rt) (16 * kt)))
  (x : array2 bf16 (l2_col_major (16 * kt) 8))
  (c : array2 float (l2_row_major (16 * rt) 8))
  (fw fx : perm)
  (ew : chest2 bf16 (16 * rt) (16 * kt))
  (ex : chest2 bf16 (16 * kt) 8)
  (ec : chest2 float (16 * rt) 8)
  (tid : szlt 32sz)
  ()
  norewrite
  requires gpu ** pkpre w fw ew x fx ex c ec tid ** thread_id 32sz tid
  ensures gpu ** pkpre w fw ew x fx ex c (cmod ew ex ec k0 k1 rt) tid ** thread_id 32sz tid
{
  pthread rt kt k0 k1 w x c #(fw /. 32) #(fx /. 32) #ew #ex #ec;
}

ghost
fn psetup
  (#rt #kt : pos)
  (w : array2 bf16 (l2_row_major (16 * rt) (16 * kt))) (fw : perm) (ew : chest2 bf16 (16 * rt) (16 * kt))
  (x : array2 bf16 (l2_col_major (16 * kt) 8)) (fx : perm) (ex : chest2 bf16 (16 * kt) 8)
  (c : array2 float (l2_row_major (16 * rt) 8)) (ec : chest2 float (16 * rt) 8)
  ()
  norewrite
  requires w |-> Frac fw ew ** x |-> Frac fx ex ** c |-> ec
  ensures (forall+ (tid : natlt 32). pkpre w fw ew x fx ex c ec tid) ** emp
{
  tensor_share_n w 32;
  tensor_share_n x 32;
  tensor_share_n c 32;
  forevery_zip3 #(natlt 32) (fun _ -> w |-> Frac (fw /. 32) ew) (fun _ -> x |-> Frac (fx /. 32) ex)
    (fun _ -> c |-> Frac (1.0R /. 32) ec);
}

ghost
fn pteardown
  (#rt #kt : pos)
  (w : array2 bf16 (l2_row_major (16 * rt) (16 * kt))) (fw : perm) (ew : chest2 bf16 (16 * rt) (16 * kt))
  (x : array2 bf16 (l2_col_major (16 * kt) 8)) (fx : perm) (ex : chest2 bf16 (16 * kt) 8)
  (c : array2 float (l2_row_major (16 * rt) 8)) (ec : chest2 float (16 * rt) 8)
  ()
  norewrite
  requires (forall+ (tid : natlt 32). pkpre w fw ew x fx ex c ec tid) ** emp
  ensures w |-> Frac fw ew ** x |-> Frac fx ex ** c |-> ec
{
  forevery_unzip3 #(natlt 32) (fun _ -> w |-> Frac (fw /. 32) ew) (fun _ -> x |-> Frac (fx /. 32) ex)
    (fun _ -> c |-> Frac (1.0R /. 32) ec);
  tensor_gather_n w 32;
  tensor_gather_n x 32;
  tensor_gather_n c 32;
}

inline_for_extraction noextract
let pdesc
  (rt kt : szp)
  (k0 : sz) (k1 : sz{k0 <= k1 /\ k1 <= kt})
  (#_ : squash (SZ.fits (16 * rt * 16 * kt) /\ SZ.fits (16 * kt * 8) /\ SZ.fits (16 * rt * 8)))
  (w : array2 bf16 (l2_row_major (16 * rt) (16 * kt)) { is_global w })
  (x : array2 bf16 (l2_col_major (16 * kt) 8) { is_global x })
  (c : array2 float (l2_row_major (16 * rt) 8) { is_global c })
  (fw fx : perm)
  (ew : chest2 bf16 (16 * rt) (16 * kt))
  (ex : chest2 bf16 (16 * kt) 8)
  (ec : chest2 float (16 * rt) 8)
  : kernel_desc
      (w |-> Frac fw ew ** x |-> Frac fx ex ** c |-> ec)
      (w |-> Frac fw ew ** x |-> Frac fx ex ** c |-> cmod ew ex ec k0 k1 rt)
= {
  nthr = 32sz;
  frame = emp;
  kpre = pkpre w fw ew x fx ex c ec;
  kpost = pkpre w fw ew x fx ex c (cmod ew ex ec k0 k1 rt);
  block_setup = psetup w fw ew x fx ex c ec;
  block_teardown = pteardown w fw ew x fx ex c (cmod ew ex ec k0 k1 rt);
  f = pkf rt kt k0 k1 w x c fw fx ew ex ec;
  kpost_sendable = solve;
  kpre_sendable = solve;
  full_post_sendable = solve;
  full_pre_sendable = solve;
} <: kernel_desc_1_n _ _


(**** HostLem ****)

open FStar.Real
open Kuiper
open Kuiper.Chest
open Kuiper.Float.Casts
open KPlay.SharedSpec

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
noextract let wc_intro (#rt #kt:pos) (#n0:nat) (s0:chest1 bf16 n0) (wo:nat) (wc:chest2 bf16 (16 * rt) (16 * kt))
  : Lemma (requires forall (i:natlt (16 * rt)) (k:natlt (16 * kt)). acc2 wc i k == bat s0 (wo + i * (16 * kt) + k))
          (ensures wc_ok #rt #kt s0 wo wc)
  = reveal_opaque (`%wc_ok) (wc_ok #rt #kt s0 wo wc)

noextract let pad_intro (#kt:pos) (sx:chest1 bf16 (7 * (16 * kt))) (xc:chest2 bf16 (16 * kt) 8)
  : Lemma (requires forall (k:natlt (16 * kt)) (t:nat{t < 7}). acc2 xc k t == bat sx (t * (16 * kt) + k))
          (ensures pad_ok #kt sx xc)
  = reveal_opaque (`%pad_ok) (pad_ok #kt sx xc)

noextract let cv_intro (#rt #kt:pos) (wc:chest2 bf16 (16 * rt) (16 * kt)) (xc:chest2 bf16 (16 * kt) 8)
  (c0:chest2 float (16 * rt) 8) (out:chest1 bf16 (7 * (16 * rt)))
  : Lemma (requires forall (i:natlt (7 * (16 * rt))).
             acc1 out i == (fcast (acc2 (cmod wc xc c0 0 kt rt) (i % (16 * rt)) (i / (16 * rt))) <: bf16))
          (ensures cv_ok #rt #kt wc xc c0 out)
  = reveal_opaque (`%cv_ok) (cv_ok #rt #kt wc xc c0 out)

noextract let cva_intro (#rt #kt:pos) (wc:chest2 bf16 (16 * rt) (16 * kt)) (xc:chest2 bf16 (16 * kt) 8)
  (c0 c0':chest2 float (16 * rt) 8) (kh:nat) (out:chest1 bf16 (7 * (16 * rt)))
  : Lemma (requires forall (i:natlt (7 * (16 * rt))).
             acc1 out i == (fcast (add (acc2 (cmod wc xc c0 0 kh rt) (i % (16 * rt)) (i / (16 * rt)))
                                       (acc2 (cmod wc xc c0' kh kt rt) (i % (16 * rt)) (i / (16 * rt)))) <: bf16))
          (ensures cva_ok #rt #kt wc xc c0 c0' kh out)
  = reveal_opaque (`%cva_ok) (cva_ok #rt #kt wc xc c0 c0' kh out)

noextract let silu_intro (sg:chest1 bf16 43008) (sa:chest1 bf16 21504)
  : Lemma (requires forall (i:natlt 21504).
             acc1 sa i == silu_f (bat sg ((i / 3072) * 6144 + i % 3072))
                                 (bat sg ((i / 3072) * 6144 + 3072 + i % 3072)))
          (ensures silu_ok sg sa)
  = reveal_opaque (`%silu_ok) (silu_ok sg sa)

noextract let qk_intro (#n0:nat) (s0:chest1 bf16 n0) (off:nat) (sv:chest1 bf16 28672) (s3:chest1 bf16 1792)
  (sq:chest1 bf16 21504)
  : Lemma (requires qk_done s0 off sv s3 sq 21504) (ensures qk_ok s0 off sv s3 sq)
  = reveal_opaque (`%qk_done) (qk_done s0 off sv s3 sq 21504);
    reveal_opaque (`%qk_ok) (qk_ok s0 off sv s3 sq)

noextract let rn_intro (#n0:nat) (s0:chest1 bf16 n0) (noff:nat) (sr su s1 s2:chest1 bf16 7168)
  : Lemma (requires rn_done s0 noff sr su s1 s2 7168) (ensures rn_ok s0 noff sr su s1 s2)
  = reveal_opaque (`%rn_done) (rn_done s0 noff sr su s1 s2 7168);
    reveal_opaque (`%rn_ok) (rn_ok s0 noff sr su s1 s2)

noextract let att_intro (sq:chest1 bf16 21504) (sv:chest1 bf16 28672) (so:chest1 bf16 14336)
  : Lemma (requires forall (qr vr : S.seq real). chest1_to_seq sq %~ qr /\ chest1_to_seq sv %~ vr ==>
             chest1_to_seq so %~ attn_values (softmax7 (attn_scores (qpart qr) (kpart qr))) (vpart vr))
          (ensures att_ok sq sv so)
  = ()

noextract let rel_vacuous (#n0:nat) (s0:chest1 bf16 n0) (s1 s2:chest1 bf16 7168) (s3:chest1 bf16 1792)
  (a4 a5:int) (o0 o1:chest1 bf16 7168)
  : Lemma (requires ~(0 <= a4 /\ a4 <= int_max - layer_elements /\ a4 + layer_elements <= n0 /\
                      0 <= a5 /\ a5 <= int_max - 1024 /\ a5 + 1024 <= n0))
          (ensures base_decoder_relation (chest1_to_seq s0) (chest1_to_seq s1) (chest1_to_seq s2)
                     (chest1_to_seq s3) a4 a5 (chest1_to_seq o0) (chest1_to_seq o1))
  = ()
#pop-options

(**** Host ****)


open FStar.Real
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major, l2_col_major, l1_forward }
open Kuiper.Float.Casts
open KPlay.SharedSpec

#push-options "--z3rlimit 40"
fn h_wcopy
  (rt kt : szp)
  (#n : erased nat{SZ.fits n})
  (src : array1 bf16 (l1_forward n) { is_global src })
  (off : SZ.t{SZ.v off + 16 * rt * 16 * kt <= n})
  (dst : array2 bf16 (l2_row_major (16 * rt) (16 * kt)) { is_global dst })
  (#f : perm) (#s : chest1 bf16 n) (#d : chest2 bf16 (16 * rt) (16 * kt))
  preserves cpu ** on gpu_loc (src |-> Frac f s)
  requires on gpu_loc (dst |-> d)
  ensures exists* (d' : chest2 bf16 (16 * rt) (16 * kt)).
    on gpu_loc (dst |-> d') ** pure (wc_ok #(SZ.v rt) #(SZ.v kt) s (SZ.v off) d')
{
  launch_kernel_1 (fun () -> wcopy rt kt #n src off dst #f #s #d);
  with d'. assert on gpu_loc (dst |-> d');
  wc_intro #(SZ.v rt) #(SZ.v kt) #n s (SZ.v off) d';
  assert pure (wc_ok #(SZ.v rt) #(SZ.v kt) #n s (SZ.v off) d');
}

fn h_pad
  (kt : szp)
  (src : array1 bf16 (l1_forward (7 * (16 * kt))) { is_global src })
  (dst : array2 bf16 (l2_col_major (16 * kt) 8) { is_global dst })
  (#f : perm) (#s : chest1 bf16 (7 * (16 * kt))) (#d : chest2 bf16 (16 * kt) 8)
  preserves cpu ** on gpu_loc (src |-> Frac f s)
  requires on gpu_loc (dst |-> d) ** pure (SZ.fits (16 * kt * 8))
  ensures exists* (d' : chest2 bf16 (16 * kt) 8).
    on gpu_loc (dst |-> d') ** pure (pad_ok #(SZ.v kt) s d')
{
  launch_kernel_1 (fun () -> pad kt src dst #f #s #d);
  with d'. assert on gpu_loc (dst |-> d');
  pad_intro #(SZ.v kt) s d';
  assert pure (pad_ok #(SZ.v kt) s d');
}

fn h_mma
  (rt kt : szp)
  (k0 : sz) (k1 : sz{k0 <= k1 /\ k1 <= kt})
  (#_ : squash (SZ.fits (16 * rt * 16 * kt) /\ SZ.fits (16 * kt * 8) /\ SZ.fits (16 * rt * 8)))
  (w : array2 bf16 (l2_row_major (16 * rt) (16 * kt)) { is_global w })
  (x : array2 bf16 (l2_col_major (16 * kt) 8) { is_global x })
  (c : array2 float (l2_row_major (16 * rt) 8) { is_global c })
  (#fw #fx : perm)
  (#ew : chest2 bf16 (16 * rt) (16 * kt))
  (#ex : chest2 bf16 (16 * kt) 8)
  (#ec : chest2 float (16 * rt) 8)
  preserves cpu ** on gpu_loc (w |-> Frac fw ew) ** on gpu_loc (x |-> Frac fx ex)
  requires on gpu_loc (c |-> ec)
  ensures on gpu_loc (c |-> cmod ew ex ec (SZ.v k0) (SZ.v k1) (SZ.v rt))
{
  on_star_intro #gpu_loc (w |-> Frac fw ew) (x |-> Frac fx ex);
  on_star_intro #gpu_loc ((w |-> Frac fw ew) ** (x |-> Frac fx ex)) (c |-> ec);
  launch_kernel_full_sync (pdesc rt kt k0 k1 #() w x c fw fx ew ex ec);
  on_star_elim #gpu_loc ((w |-> Frac fw ew) ** (x |-> Frac fx ex)) (c |-> cmod ew ex ec (SZ.v k0) (SZ.v k1) (SZ.v rt));
  on_star_elim #gpu_loc (w |-> Frac fw ew) (x |-> Frac fx ex);
}

fn h_proj
  (rt kt : szp)
  (#_ : squash (SZ.fits (16 * rt * 16 * kt) /\ SZ.fits (16 * kt * 8) /\ SZ.fits (16 * rt * 8)))
  (w : array2 bf16 (l2_row_major (16 * rt) (16 * kt)) { is_global w })
  (x : array2 bf16 (l2_col_major (16 * kt) 8) { is_global x })
  (c : array2 float (l2_row_major (16 * rt) 8) { is_global c })
  (out : array1 bf16 (l1_forward (7 * (16 * rt))) { is_global out })
  (#fw #fx : perm)
  (#ew : chest2 bf16 (16 * rt) (16 * kt))
  (#ex : chest2 bf16 (16 * kt) 8)
  (#ec : chest2 float (16 * rt) 8)
  (#o : chest1 bf16 (7 * (16 * rt)))
  preserves cpu ** on gpu_loc (w |-> Frac fw ew) ** on gpu_loc (x |-> Frac fx ex)
  requires on gpu_loc (c |-> ec) ** on gpu_loc (out |-> o)
  ensures exists* (ec' : chest2 float (16 * rt) 8) (o' : chest1 bf16 (7 * (16 * rt))).
    on gpu_loc (c |-> ec') ** on gpu_loc (out |-> o') **
    pure (cv_ok #(SZ.v rt) #(SZ.v kt) ew ex ec o')
{
  h_mma rt kt 0sz kt #() w x c #fw #fx #ew #ex #ec;
  launch_kernel_1 (fun () -> convert rt c out #1.0R #(cmod ew ex ec 0 (SZ.v kt) (SZ.v rt)) #o);
  with o'. assert on gpu_loc (out |-> o');
  cv_intro #(SZ.v rt) #(SZ.v kt) ew ex ec o';
  assert pure (cv_ok #(SZ.v rt) #(SZ.v kt) ew ex ec o');
}

fn h_proj_split
  (rt kt : szp) (kh : sz{kh <= kt})
  (#_ : squash (SZ.fits (16 * rt * 16 * kt) /\ SZ.fits (16 * kt * 8) /\ SZ.fits (16 * rt * 8)))
  (w : array2 bf16 (l2_row_major (16 * rt) (16 * kt)) { is_global w })
  (x : array2 bf16 (l2_col_major (16 * kt) 8) { is_global x })
  (c : array2 float (l2_row_major (16 * rt) 8) { is_global c })
  (c' : array2 float (l2_row_major (16 * rt) 8) { is_global c' })
  (out : array1 bf16 (l1_forward (7 * (16 * rt))) { is_global out })
  (#fw #fx : perm)
  (#ew : chest2 bf16 (16 * rt) (16 * kt))
  (#ex : chest2 bf16 (16 * kt) 8)
  (#ec #ec2 : chest2 float (16 * rt) 8)
  (#o : chest1 bf16 (7 * (16 * rt)))
  preserves cpu ** on gpu_loc (w |-> Frac fw ew) ** on gpu_loc (x |-> Frac fx ex)
  requires on gpu_loc (c |-> ec) ** on gpu_loc (c' |-> ec2) ** on gpu_loc (out |-> o)
  ensures exists* (e1 e2 : chest2 float (16 * rt) 8) (o' : chest1 bf16 (7 * (16 * rt))).
    on gpu_loc (c |-> e1) ** on gpu_loc (c' |-> e2) ** on gpu_loc (out |-> o') **
    pure (cva_ok #(SZ.v rt) #(SZ.v kt) ew ex ec ec2 (SZ.v kh) o')
{
  h_mma rt kt 0sz kh #() w x c #fw #fx #ew #ex #ec;
  h_mma rt kt kh kt #() w x c' #fw #fx #ew #ex #ec2;
  launch_kernel_1 (fun () -> convert_add rt c c' out #1.0R #1.0R
    #(cmod ew ex ec 0 (SZ.v kh) (SZ.v rt)) #(cmod ew ex ec2 (SZ.v kh) (SZ.v kt) (SZ.v rt)) #o);
  with o'. assert on gpu_loc (out |-> o');
  cva_intro #(SZ.v rt) #(SZ.v kt) ew ex ec ec2 (SZ.v kh) o';
  assert pure (cva_ok #(SZ.v rt) #(SZ.v kt) ew ex ec ec2 (SZ.v kh) o');
}

fn h_silu
  (gu : array1 bf16 (l1_forward 43008) { is_global gu })
  (out : array1 bf16 (l1_forward 21504) { is_global out })
  (#f : perm) (#s : chest1 bf16 43008) (#o : chest1 bf16 21504)
  preserves cpu ** on gpu_loc (gu |-> Frac f s)
  requires on gpu_loc (out |-> o)
  ensures exists* (o' : chest1 bf16 21504). on gpu_loc (out |-> o') ** pure (silu_ok s o')
{
  launch_kernel_1 (fun () -> silu_k gu out #f #s #o);
  with o'. assert on gpu_loc (out |-> o');
  silu_intro s o';
  assert pure (silu_ok s o');
}

fn h_attqk
  (#n0 : erased nat{SZ.fits n0})
  (w : array1 bf16 (l1_forward n0) { is_global w })
  (off : SZ.t{SZ.v off + 6293760 <= n0})
  (qkv : array1 bf16 (l1_forward 28672) { is_global qkv })
  (rot : array1 bf16 (l1_forward 1792) { is_global rot })
  (qk : array1 bf16 (l1_forward 21504) { is_global qk })
  (nrm : array1 bf16 (l1_forward 128) { is_global nrm })
  (partial : array1 float (l1_forward 128) { is_global partial })
  (#pw #pv #pr : perm)
  (#sw : chest1 bf16 n0) (#sv : chest1 bf16 28672) (#sr : chest1 bf16 1792)
  (#sq : chest1 bf16 21504) (#sn : chest1 bf16 128) (#sp : chest1 float 128)
  preserves cpu ** on gpu_loc (w |-> Frac pw sw) ** on gpu_loc (qkv |-> Frac pv sv) ** on gpu_loc (rot |-> Frac pr sr)
  requires on gpu_loc (qk |-> sq) ** on gpu_loc (nrm |-> sn) ** on gpu_loc (partial |-> sp)
  ensures exists* (sq' : chest1 bf16 21504) (sn' : chest1 bf16 128) (sp' : chest1 float 128).
    on gpu_loc (qk |-> sq') ** on gpu_loc (nrm |-> sn') ** on gpu_loc (partial |-> sp') **
    pure (qk_ok sw (SZ.v off) sv sr sq')
{
  launch_kernel_1 (fun () -> att_qk #n0 w off qkv rot qk nrm partial #pw #pv #pr #sw #sv #sr #sq #sn #sp);
  with sq'. assert on gpu_loc (qk |-> sq');
  qk_intro #n0 sw (SZ.v off) sv sr sq';
  assert pure (qk_ok #n0 sw (SZ.v off) sv sr sq');
}

fn h_attsv
  (qk : array1 bf16 (l1_forward 21504) { is_global qk })
  (qkv : array1 bf16 (l1_forward 28672) { is_global qkv })
  (out : array1 bf16 (l1_forward 14336) { is_global out })
  (sc : array1 float (l1_forward 8) { is_global sc })
  (#fq #fv : perm) (#sq : chest1 bf16 21504) (#sv : chest1 bf16 28672) (#so : chest1 bf16 14336) (#ss : chest1 float 8)
  preserves cpu ** on gpu_loc (qk |-> Frac fq sq) ** on gpu_loc (qkv |-> Frac fv sv)
  requires on gpu_loc (out |-> so) ** on gpu_loc (sc |-> ss)
  ensures exists* (so' : chest1 bf16 14336) (ss' : chest1 float 8).
    on gpu_loc (out |-> so') ** on gpu_loc (sc |-> ss') ** pure (att_ok sq sv so')
{
  launch_kernel_1 (fun () -> att_sv qk qkv out sc #fq #fv #sq #sv #so #ss);
  with so'. assert on gpu_loc (out |-> so');
  att_intro sq sv so';
  assert pure (att_ok sq sv so');
}

fn h_resnorm
  (#n0 : erased nat{SZ.fits n0})
  (w : array1 bf16 (l1_forward n0) { is_global w })
  (noff : SZ.t{SZ.v noff + 1024 <= n0})
  (res : array1 bf16 (l1_forward 7168) { is_global res })
  (ua : array1 bf16 (l1_forward 7168) { is_global ua })
  (summed : array1 bf16 (l1_forward 7168) { is_global summed })
  (normd : array1 bf16 (l1_forward 7168) { is_global normd })
  (partial : array1 float (l1_forward 128) { is_global partial })
  (#pw #p1 #p2 : perm)
  (#sw : chest1 bf16 n0) (#sr #su : chest1 bf16 7168)
  (#s1 #s2 : chest1 bf16 7168) (#sp : chest1 float 128)
  preserves cpu ** on gpu_loc (w |-> Frac pw sw) ** on gpu_loc (res |-> Frac p1 sr) ** on gpu_loc (ua |-> Frac p2 su)
  requires on gpu_loc (summed |-> s1) ** on gpu_loc (normd |-> s2) ** on gpu_loc (partial |-> sp)
  ensures exists* (s1' s2' : chest1 bf16 7168) (sp' : chest1 float 128).
    on gpu_loc (summed |-> s1') ** on gpu_loc (normd |-> s2') ** on gpu_loc (partial |-> sp') **
    pure (rn_ok sw (SZ.v noff) sr su s1' s2')
{
  launch_kernel_1 (fun () -> resnorm #n0 w noff res ua summed normd partial #pw #p1 #p2 #sw #sr #su #s1 #s2 #sp);
  with s1' s2'. assert (on gpu_loc (summed |-> s1') ** on gpu_loc (normd |-> s2'));
  rn_intro #n0 sw (SZ.v noff) sr su s1' s2';
  assert pure (rn_ok #n0 sw (SZ.v noff) sr su s1' s2');
}
#pop-options

inline_for_extraction noextract
let i64_to_sz (x:I64.t{0 <= I64.v x /\ I64.v x < 2147483648}) : y:SZ.t{SZ.v y == I64.v x} =
  SZ.uint32_to_sizet (FStar.Int.Cast.int64_to_uint32 x)

#push-options "--z3rlimit 80"
fn kplay_run
  (n0 : SZ.t)
  (input0 : array1 bf16 (l1_forward n0) { is_global input0 })
  (input1 : array1 bf16 (l1_forward 7168) { is_global input1 })
  (input2 : array1 bf16 (l1_forward 7168) { is_global input2 })
  (input3 : array1 bf16 (l1_forward 1792) { is_global input3 })
  (input4 : I64.t)
  (input5 : I64.t)
  (output0 : array1 bf16 (l1_forward 7168) { is_global output0 })
  (output1 : array1 bf16 (l1_forward 7168) { is_global output1 })
  (#p0 #p1 #p2 #p3 : perm)
  (#s0 : chest1 bf16 n0)
  (#s1 #s2 : chest1 bf16 7168)
  (#s3 : chest1 bf16 1792)
  (#o0 #o1 : chest1 bf16 7168)
  preserves
    cpu **
    on gpu_loc (input0 |-> Frac p0 s0) **
    on gpu_loc (input1 |-> Frac p1 s1) **
    on gpu_loc (input2 |-> Frac p2 s2) **
    on gpu_loc (input3 |-> Frac p3 s3)
  requires
    on gpu_loc (output0 |-> o0) **
    on gpu_loc (output1 |-> o1)
  ensures
    exists* (o0' o1' : chest1 bf16 7168).
      on gpu_loc (output0 |-> o0') **
      on gpu_loc (output1 |-> o1') **
      pure (base_decoder_relation (chest1_to_seq s0) (chest1_to_seq s1) (chest1_to_seq s2) (chest1_to_seq s3)
              (I64.v input4) (I64.v input5) (chest1_to_seq o0') (chest1_to_seq o1'))
{
  let ok =
    I64.gte input4 0L && I64.lte input4 2131752703L &&
    I64.gte input5 0L && I64.lte input5 2147482623L;
  if ok {
    let off = i64_to_sz input4;
    let a5 = i64_to_sz input5;
    let le : SZ.t = SZ.uint32_to_sizet 15730944ul;
    if (le <=^ n0 && off <=^ n0 -^ le && 1024sz <=^ n0 && a5 <=^ n0 -^ 1024sz) {
      (* weights *)
      let w1 = alloc0 #bf16 (SZ.uint32_to_sizet 4194304ul) (l2_row_major (16 * 256) (16 * 64));
      h_wcopy 256sz 64sz #(SZ.v n0) input0 (off +^ 2048sz) w1;
      let w2 = alloc0 #bf16 (SZ.uint32_to_sizet 2097152ul) (l2_row_major (16 * 64) (16 * 128));
      h_wcopy 64sz 128sz #(SZ.v n0) input0 (off +^ SZ.uint32_to_sizet 4196352ul) w2;
      let w3 = alloc0 #bf16 (SZ.uint32_to_sizet 6291456ul) (l2_row_major (16 * 384) (16 * 64));
      h_wcopy 384sz 64sz #(SZ.v n0) input0 (off +^ SZ.uint32_to_sizet 6293760ul) w3;
      let w4 = alloc0 #bf16 (SZ.uint32_to_sizet 3145728ul) (l2_row_major (16 * 64) (16 * 192));
      h_wcopy 64sz 192sz #(SZ.v n0) input0 (off +^ SZ.uint32_to_sizet 12585216ul) w4;
      with ew1. assert on gpu_loc (w1 |-> ew1);
      with ew2. assert on gpu_loc (w2 |-> ew2);
      with ew3. assert on gpu_loc (w3 |-> ew3);
      with ew4. assert on gpu_loc (w4 |-> ew4);
      (* QKV projection *)
      let xa = alloc0 #bf16 8192sz (l2_col_major (16 * 64) 8);
      h_pad 64sz input2 xa;
      with exa. assert on gpu_loc (xa |-> exa);
      let ca = alloc0 #float 32768sz (l2_row_major (16 * 256) 8);
      with eca. assert on gpu_loc (ca |-> eca);
      let qkv = alloc0 #bf16 28672sz (l1_forward 28672);
      h_proj 256sz 64sz #() w1 xa ca qkv;
      with sqkv. assert on gpu_loc (qkv |-> sqkv);
      (* q/k norm + rope *)
      let qk = alloc0 #bf16 21504sz (l1_forward 21504);
      let nrm = alloc0 #bf16 128sz (l1_forward 128);
      let part = alloc0 #float 128sz (l1_forward 128);
      h_attqk #(SZ.v n0) input0 off qkv input3 qk nrm part;
      with sqk. assert on gpu_loc (qk |-> sqk);
      (* attention *)
      let att = alloc0 #bf16 14336sz (l1_forward 14336);
      let sc = alloc0 #float 8sz (l1_forward 8);
      h_attsv qk qkv att sc;
      with satt. assert on gpu_loc (att |-> satt);
      (* o_proj *)
      let xb = alloc0 #bf16 16384sz (l2_col_major (16 * 128) 8);
      h_pad 128sz att xb;
      with exb. assert on gpu_loc (xb |-> exb);
      let cb = alloc0 #float 8192sz (l2_row_major (16 * 64) 8);
      with ecb. assert on gpu_loc (cb |-> ecb);
      let proj = alloc0 #bf16 7168sz (l1_forward 7168);
      h_proj 64sz 128sz #() w2 xb cb proj;
      with sproj. assert on gpu_loc (proj |-> sproj);
      (* residual + post-attention norm *)
      let res = alloc0 #bf16 7168sz (l1_forward 7168);
      let norm = alloc0 #bf16 7168sz (l1_forward 7168);
      h_resnorm #(SZ.v n0) input0 (off +^ 1024sz) input1 proj res norm part;
      with sres. assert on gpu_loc (res |-> sres);
      with snorm. assert on gpu_loc (norm |-> snorm);
      (* gate/up *)
      let xc = alloc0 #bf16 8192sz (l2_col_major (16 * 64) 8);
      h_pad 64sz norm xc;
      with exc. assert on gpu_loc (xc |-> exc);
      let cc = alloc0 #float 49152sz (l2_row_major (16 * 384) 8);
      with ecc. assert on gpu_loc (cc |-> ecc);
      let gu = alloc0 #bf16 43008sz (l1_forward 43008);
      h_proj 384sz 64sz #() w3 xc cc gu;
      with sgu. assert on gpu_loc (gu |-> sgu);
      let act = alloc0 #bf16 21504sz (l1_forward 21504);
      h_silu gu act;
      with sact. assert on gpu_loc (act |-> sact);
      (* down *)
      let xd = alloc0 #bf16 24576sz (l2_col_major (16 * 192) 8);
      h_pad 192sz act xd;
      with exd. assert on gpu_loc (xd |-> exd);
      let cd = alloc0 #float 8192sz (l2_row_major (16 * 64) 8);
      with ecd. assert on gpu_loc (cd |-> ecd);
      let cd2 = alloc0 #float 8192sz (l2_row_major (16 * 64) 8);
      with ecd2. assert on gpu_loc (cd2 |-> ecd2);
      let proj2 = alloc0 #bf16 7168sz (l1_forward 7168);
      h_proj_split 64sz 192sz 96sz #() w4 xd cd cd2 proj2;
      with sproj2. assert on gpu_loc (proj2 |-> sproj2);
      (* residual + final norm *)
      h_resnorm #(SZ.v n0) input0 a5 res proj2 output0 output1 part;
      with so0. assert on gpu_loc (output0 |-> so0);
      with so1. assert on gpu_loc (output1 |-> so1);
      decoder_ok #(SZ.v n0) s0 s1 s2 s3 (SZ.v off) (SZ.v a5)
        ew1 exa eca sqkv sqk satt ew2 exb ecb sproj sres snorm
        ew3 exc ecc sgu sact ew4 exd ecd ecd2 sproj2 so0 so1;
      free w1; free w2; free w3; free w4;
      free xa; free xb; free xc; free xd;
      free ca; free cb; free cc; free cd; free cd2;
      free qkv; free qk; free nrm; free part; free att; free sc;
      free proj; free res; free norm; free gu; free act; free proj2;
      ()
    } else {
      rel_vacuous #(SZ.v n0) s0 s1 s2 s3 (I64.v input4) (I64.v input5) o0 o1;
      ()
    }
  } else {
    rel_vacuous #(SZ.v n0) s0 s1 s2 s3 (I64.v input4) (I64.v input5) o0 o1;
    ()
  }
}
#pop-options
