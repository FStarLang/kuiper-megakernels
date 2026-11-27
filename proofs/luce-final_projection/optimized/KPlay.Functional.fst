module KPlay.Functional

#lang-pulse

(* Proof of the optimized final_projection family: one-thread CUDA kernel
   (square sum, RMSNorm(1+w), continuous-dot vocabulary projection) plus the
   history copy with row d overwritten by bf16(hidden). *)


(*** Sums *)
open FStar.Real
open Kuiper.Real
open Kuiper.Seq.Common
module S = FStar.Seq
open KPlay.SharedSpec

#set-options "--z3rlimit 20 --fuel 1 --ifuel 0"

(*** Left-to-right real sums *)

noextract let rec rs (n:nat) (f:nat -> real) : Tot real (decreases n) =
  if n = 0 then 0.0R else rs (n - 1) f +. f (n - 1)

noextract let rec rs_ext (n:nat) (f g:nat -> real)
  : Lemma (requires forall (k:nat). k < n ==> f k == g k)
          (ensures rs n f == rs n g) (decreases n)
  = if n = 0 then () else rs_ext (n - 1) f g

noextract let rs_one (f:nat -> real) : Lemma (rs 1 f == f 0) =
  assert (rs 0 f == 0.0R)

noextract let shift (f:nat -> real) (p:nat) : nat -> real = fun k -> f (p + k)

noextract let rec rs_split (a b:nat) (f:nat -> real)
  : Lemma (ensures rs (a + b) f == rs a f +. rs b (shift f a)) (decreases b)
  = if b = 0 then () else rs_split a (b - 1) f

noextract let add_f (f g:nat -> real) : nat -> real = fun k -> f k +. g k

noextract let rec rs_add (n:nat) (f g:nat -> real)
  : Lemma (ensures rs n (add_f f g) == rs n f +. rs n g) (decreases n)
  = if n = 0 then () else rs_add (n - 1) f g

noextract let rec rs_zero (n:nat) (f:nat -> real)
  : Lemma (requires forall (k:nat). k < n ==> f k == 0.0R)
          (ensures rs n f == 0.0R) (decreases n)
  = if n = 0 then () else rs_zero (n - 1) f

noextract let rec rs_nonneg (n:nat) (f:nat -> real)
  : Lemma (requires forall (k:nat). k < n ==> f k >=. 0.0R)
          (ensures rs n f >=. 0.0R) (decreases n)
  = if n = 0 then () else rs_nonneg (n - 1) f

noextract let row (b:nat) (g:nat -> real) (i:nat) : nat -> real = fun j -> g (i * b + j)
let rows (b:nat) (g:nat -> real) : nat -> real = fun i -> rs b (row b g i)

noextract let rec rs_block (a b:nat) (g:nat -> real)
  : Lemma (ensures rs (a * b) g == rs a (rows b g)) (decreases a)
  = if a = 0 then ()
    else begin
      rs_block (a - 1) b g;
      let p : nat = (a - 1) * b in
      assert (p + b == a * b);
      rs_split p b g;
      rs_ext b (shift g p) (row b g (a - 1))
    end

noextract let col (h:nat -> nat -> real) (j:nat) : nat -> real = fun i -> h i j
let rowf (h:nat -> nat -> real) (i:nat) : nat -> real = fun j -> h i j
noextract let rsr (b:nat) (h:nat -> nat -> real) : nat -> real = fun i -> rs b (rowf h i)
let rsc (a:nat) (h:nat -> nat -> real) : nat -> real = fun j -> rs a (col h j)

noextract let rec rs_swap (a b:nat) (h:nat -> nat -> real)
  : Lemma (ensures rs a (rsr b h) == rs b (rsc a h)) (decreases a)
  = if a = 0 then rs_zero b (rsc 0 h)
    else begin
      rs_swap (a - 1) b h;
      rs_add b (rsc (a - 1) h) (rowf h (a - 1));
      rs_ext b (add_f (rsc (a - 1) h) (rowf h (a - 1))) (rsc a h)
    end

(*** vsum agrees with rs *)

noextract let rec rsum_rs (n:nat) (s:S.seq real) (f:nat -> real)
  : Lemma (requires S.length s == n /\ (forall (k:nat). k < n ==> S.index s k == f k))
          (ensures rsum s == rs n f) (decreases n)
  = if n = 0 then ()
    else begin
      let s' = S.slice s 0 (n - 1) in
      rsum_rs (n - 1) s' f;
      S.lemma_eq_intro (S.slice s 0 n) s;
      lemma_seq_fold_left_slice 0.0R (+.) s 0 (n - 1)
    end

noextract let vsum_rs (n:nat) (f:nat -> real)
  : Lemma (ensures vsum n f == rs n f)
  = rsum_rs n (S.init n (fun (k:nat{k < n}) -> f k)) f

(*** Lits *)
open FStar.Real
module Lit = Kuiper.Float32.Literals
module Cert = Kuiper.Float32.ConstantFold.Certificates
module CF = Kuiper.Float32.ConstantFold
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module A = Kuiper.Approximates.Base
open KPlay.SharedSpec

noextract let pos_mul (a b:int) : Lemma (requires a > 0 /\ b > 0) (ensures a * b > 0) = ()

noextract let value_pos (s:string) (m:nat) (e:int)
  : Lemma (requires Cert.exact_normal s m e) (ensures Lit.value s >. 0.0R)
  = match Lit.integral (FStar.String.list_of_string s) 0 false with
    | Some (n, d) ->
      let (un, ud) = Cert.unit_fraction e in
      pos_mul m un; pos_mul (m * un) d;
      assert (n * ud > 0);
      assert (n > 0)
    | None -> ()

noextract let eps_s : string = "0.0000009999999974752427078783512115478515625"

noextract let eps_cert () : Lemma (Cert.exact_normal eps_s 8796093 (-43)) =
  assert_norm (Cert.exact_normal eps_s 8796093 (-43))

noextract let eps_facts ()
  : Lemma (A.v_approximates (F.of_literal #FP.t eps_s) eps_f32 /\ eps_f32 >. 0.0R)
  = eps_cert ();
    CF.literal_approx eps_s 8796093 (-43);
    value_pos eps_s 8796093 (-43)

(*** Tree *)
open Kuiper
module S = FStar.Seq
module SZ = Kuiper.SizeT
module F32 = Kuiper.Float32

#set-options "--z3rlimit 20 --fuel 1 --ifuel 0"

noextract let arr_ok (#e:Type) (p:e -> prop) (s:S.seq F32.t) (n:nat) (g:e -> nat -> real) : prop =
  n <= S.length s /\ (forall (r:e) (k:nat). p r /\ k < n ==> S.index s k %~ g r k)

noextract let stepf (#e:Type) (g:e -> nat -> real) (d:nat) : e -> nat -> real =
  fun r k -> g r k +. g r (k + d)

noextract let step_eq (#e:Type) (g:e -> nat -> real) (d:nat) (r:e)
  : Lemma (rs (d + d) (g r) == rs d (stepf g d r))
  = rs_split d d (g r);
    rs_add d (g r) (shift (g r) d);
    rs_ext d (add_f (g r) (shift (g r) d)) (stepf g d r)

noextract let tree_g (#e:Type) (g:e -> nat -> real) : e -> nat -> real =
  stepf (stepf (stepf (stepf (stepf g 16) 8) 4) 2) 1

noextract let tree_eq (#e:Type) (g:e -> nat -> real) (r:e)
  : Lemma (rs 32 (g r) == tree_g g r 0)
  = step_eq g 16 r;
    step_eq (stepf g 16) 8 r;
    step_eq (stepf (stepf g 16) 8) 4 r;
    step_eq (stepf (stepf (stepf g 16) 8) 4) 2 r;
    step_eq (stepf (stepf (stepf (stepf g 16) 8) 4) 2) 1 r;
    rs_one (stepf (stepf (stepf (stepf (stepf g 16) 8) 4) 2) 1 r);
    assert (rs 32 (g r) == rs 1 (stepf (stepf (stepf (stepf (stepf g 16) 8) 4) 2) 1 r));
    assert (rs 32 (g r) == stepf (stepf (stepf (stepf (stepf g 16) 8) 4) 2) 1 r 0)

noextract let add_step (#e:Type) (p:e -> prop) (g:e -> nat -> real) (d i:nat) (u w res:F32.t)
  : Lemma (requires (forall (r:e). p r ==> u %~ g r i /\ w %~ g r (i + d)) /\
                    (forall (xr yr:real). u %~ xr /\ w %~ yr ==> res %~ (xr +. yr)))
          (ensures forall (r:e). p r ==> res %~ stepf g d r i)
  = ()

inline_for_extraction noextract
fn tree_step (#e:Type0) (p:erased (e -> prop)) (g:erased (e -> nat -> real))
  (a:larray F32.t 32) (d:SZ.t{0 < SZ.v d /\ SZ.v d <= 16}) (#s0:erased (S.seq F32.t))
  preserves gpu
  requires a |-> s0 ** pure (arr_ok (reveal p) s0 (SZ.v d + SZ.v d) (reveal g))
  ensures exists* s1. a |-> s1 ** pure (arr_ok (reveal p) s1 (SZ.v d) (stepf (reveal g) (SZ.v d)))
{
  pts_to_len a;
  let mut i = 0sz;
  while (!i <^ d)
    invariant exists* (vi:SZ.t) (s:S.seq F32.t). i |-> vi ** a |-> s ** pure (
      SZ.v vi <= SZ.v d /\ S.length s == 32 /\
      (forall (k:nat). SZ.v vi <= k /\ k < 32 ==> S.index s k == S.index s0 k) /\
      arr_ok (reveal p) s (SZ.v vi) (stepf (reveal g) (SZ.v d)))
    decreases (SZ.v d - SZ.v !i)
  {
    let vi = !i;
    let u = a.(vi);
    let w = a.(vi +^ d);
    let res = F32.add_rn_ftz u w;
    add_step (reveal p) (reveal g) (SZ.v d) (SZ.v vi) u w res;
    a.(vi) <- res;
    i := vi +^ 1sz;
  }
}

inline_for_extraction noextract
fn tree32 (#e:Type0) (p:erased (e -> prop)) (g:erased (e -> nat -> real))
  (a:larray F32.t 32) (#s0:erased (S.seq F32.t))
  preserves gpu
  requires a |-> s0 ** pure (arr_ok (reveal p) s0 32 (reveal g))
  returns res:F32.t
  ensures exists* s1. a |-> s1 ** pure (forall (r:e). reveal p r ==> res %~ rs 32 (reveal g r))
{
  tree_step p g a 16sz;
  tree_step p (stepf (reveal g) 16) a 8sz;
  tree_step p (stepf (stepf (reveal g) 16) 8) a 4sz;
  tree_step p (stepf (stepf (stepf (reveal g) 16) 8) 4) a 2sz;
  tree_step p (stepf (stepf (stepf (stepf (reveal g) 16) 8) 4) 2) a 1sz;
  pts_to_len a;
  let res = a.(0sz);
  FStar.Classical.forall_intro (tree_eq (reveal g));
  res
}

(*** SqSum *)
open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = Kuiper.SizeT
module F32 = Kuiper.Float32

#set-options "--z3rlimit 20 --fuel 1 --ifuel 0"

noextract let sqf (v:S.seq real) : nat -> real = fun k -> at v k *. at v k

noextract let lane_v (v:S.seq real) (w:nat) : nat -> real =
  fun l -> sqf v (32 * w + l + 512) +. sqf v (32 * w + l)

noextract let warp_v (v:S.seq real) : nat -> real =
  fun w -> if w < 16 then rs 32 (lane_v v w) else 0.0R

noextract let lane_eq (v:S.seq real) (w:nat)
  : Lemma (rs 32 (lane_v v w) == add_f (rows 32 (shift (sqf v) 512)) (rows 32 (sqf v)) w)
  = rs_ext 32 (lane_v v w) (add_f (row 32 (shift (sqf v) 512) w) (row 32 (sqf v) w));
    rs_add 32 (row 32 (shift (sqf v) 512) w) (row 32 (sqf v) w)

noextract let sq_sum_rs_o (v:S.seq real) (o:int) (n:nat)
  : Lemma (sq_sum v o n == rs n (fun k -> at v (o + k) *. at v (o + k)))
  = vsum_rs n (fun k -> at v (o + k) *. at v (o + k))

noextract let sq_sum_rs (v:S.seq real) (n:nat)
  : Lemma (sq_sum v 0 n == rs n (sqf v))
  = sq_sum_rs_o v 0 n;
    rs_ext n (fun k -> at v (0 + k) *. at v (0 + k)) (sqf v)

noextract let warp_eq (v:S.seq real)
  : Lemma (rs 32 (warp_v v) == sq_sum v 0 1024)
  = rs_split 16 16 (warp_v v);
    rs_zero 16 (shift (warp_v v) 16);
    introduce forall (w:nat). w < 16 ==>
        warp_v v w == add_f (rows 32 (shift (sqf v) 512)) (rows 32 (sqf v)) w
      with introduce _ ==> _ with lane_eq v w;
    rs_ext 16 (warp_v v) (add_f (rows 32 (shift (sqf v) 512)) (rows 32 (sqf v)));
    rs_add 16 (rows 32 (shift (sqf v) 512)) (rows 32 (sqf v));
    rs_block 16 32 (shift (sqf v) 512);
    rs_block 16 32 (sqf v);
    rs_split 512 512 (sqf v);
    sq_sum_rs v 1024

noextract let sq_nonneg (v:S.seq real) (n:nat)
  : Lemma (sq_sum v 0 n >=. 0.0R)
  = sq_sum_rs v n;
    introduce forall (k:nat). k < n ==> sqf v k >=. 0.0R
      with introduce _ ==> _ with (
        let a = at v k in
        assert (a *. a >=. 0.0R));
    rs_nonneg n (sqf v)

noextract let zero_approx () : Lemma (zero #F32.t %~ 0.0R) = Kuiper.Approximates.Scalar.a0 #F32.t

noextract let lane_e (#e:Type) (xr:e -> S.seq real) (w:nat) : e -> nat -> real = fun r -> lane_v (xr r) w
let warp_e (#e:Type) (xr:e -> S.seq real) : e -> nat -> real = fun r -> warp_v (xr r)

noextract let x_at (#e:Type) (p:e -> prop) (xr:e -> S.seq real) (xs:S.seq F32.t) (i:nat)
  : Lemma (requires (forall (r:e). p r ==> xs %~ xr r) /\ i < S.length xs)
          (ensures forall (r:e). p r ==> S.index xs i %~ at (xr r) i)
  = ()

noextract let fma_inst (a b c res:F32.t) (x y z:real)
  : Lemma (requires (forall (xr yr zr:real). a %~ xr /\ b %~ yr /\ c %~ zr ==> res %~ (xr *. yr +. zr)) /\
                    a %~ x /\ b %~ y /\ c %~ z)
          (ensures res %~ (x *. y +. z))
  = ()

noextract let approx_eq (a:F32.t) (x y:real) : Lemma (requires a %~ x /\ x == y) (ensures a %~ y) = ()

noextract let lane_step (#e:Type) (p:e -> prop) (xr:e -> S.seq real) (w l:nat) (a b s1 s2:F32.t)
  : Lemma (requires (forall (r:e). p r ==> a %~ at (xr r) (32 * w + l) /\
                                        b %~ at (xr r) (32 * w + l + 512)) /\
                    (forall (xr' yr zr:real). a %~ xr' /\ a %~ yr /\ zero #F32.t %~ zr ==>
                       s1 %~ (xr' *. yr +. zr)) /\
                    (forall (xr' yr zr:real). b %~ xr' /\ b %~ yr /\ s1 %~ zr ==>
                       s2 %~ (xr' *. yr +. zr)))
          (ensures forall (r:e). p r ==> s2 %~ lane_e xr w r l)
  = zero_approx ();
    introduce forall (r:e). p r ==> s2 %~ lane_e xr w r l
      with introduce _ ==> _ with (
        let ar = at (xr r) (32 * w + l) in
        let br = at (xr r) (32 * w + l + 512) in
        fma_inst a a (zero #F32.t) s1 ar ar 0.0R;
        approx_eq s1 (ar *. ar +. 0.0R) (ar *. ar);
        fma_inst b b s1 s2 br br (ar *. ar))

noextract let warp_upd (#e:Type) (p:e -> prop) (xr:e -> S.seq real) (ws:S.seq F32.t) (vw:nat) (t:F32.t)
  : Lemma (requires S.length ws == 32 /\ vw < 16 /\ arr_ok p ws vw (warp_e xr) /\
                    (forall (r:e). p r ==> t %~ rs 32 (lane_e xr vw r)))
          (ensures arr_ok p (S.upd ws vw t) (vw + 1) (warp_e xr))
  = introduce forall (r:e). p r ==> t %~ warp_e xr r vw
      with introduce _ ==> _ with (
        rs_ext 32 (lane_e xr vw r) (lane_v (xr r) vw))

noextract let warp_final (#e:Type) (p:e -> prop) (xr:e -> S.seq real) (ws:S.seq F32.t)
  : Lemma (requires S.length ws == 32 /\ arr_ok p ws 16 (warp_e xr) /\
                    (forall (k:nat). 16 <= k /\ k < 32 ==> S.index ws k == zero #F32.t))
          (ensures arr_ok p ws 32 (warp_e xr))
  = zero_approx ()

noextract let ss_final (#e:Type) (p:e -> prop) (xr:e -> S.seq real) (ss:F32.t)
  : Lemma (requires forall (r:e). p r ==> ss %~ rs 32 (warp_e xr r))
          (ensures forall (r:e). p r ==> ss %~ sq_sum (xr r) 0 1024)
  = introduce forall (r:e). p r ==> ss %~ sq_sum (xr r) 0 1024
      with introduce _ ==> _ with (
        rs_ext 32 (warp_e xr r) (warp_v (xr r));
        warp_eq (xr r))

inline_for_extraction noextract
fn square_sum (#e:Type0) (p:erased (e -> prop)) (xr:erased (e -> S.seq real))
  (x:larray F32.t 1024) (#f:perm) (#xs:erased (S.seq F32.t))
  preserves gpu ** x |-> Frac f xs
  requires pure (forall (r:e). reveal p r ==> reveal xs %~ reveal xr r)
  returns ss:F32.t
  ensures pure (forall (r:e). reveal p r ==> ss %~ sq_sum (reveal xr r) 0 1024)
{
  pts_to_len x;
  let mut warps : array F32.t = [| zero #F32.t; 32sz |];
  let mut lanes : array F32.t = [| zero #F32.t; 32sz |];
  zero_approx ();
  let mut w = 0sz;
  while (!w <^ 16sz)
    invariant exists* (vw:SZ.t) (ws ls:S.seq F32.t). w |-> vw ** warps |-> ws ** lanes |-> ls ** pure (
      SZ.v vw <= 16 /\ S.length ws == 32 /\ S.length ls == 32 /\
      (forall (k:nat). SZ.v vw <= k /\ k < 32 ==> S.index ws k == zero #F32.t) /\
      arr_ok (reveal p) ws (SZ.v vw) (warp_e (reveal xr)))
    decreases (16 - SZ.v !w)
  {
    let vw = !w;
    let mut l = 0sz;
    while (!l <^ 32sz)
      invariant exists* (vl:SZ.t) (ls:S.seq F32.t). l |-> vl ** lanes |-> ls ** pure (
        SZ.v vl <= 32 /\ S.length ls == 32 /\
        arr_ok (reveal p) ls (SZ.v vl) (lane_e (reveal xr) (SZ.v vw)))
      decreases (32 - SZ.v !l)
    {
      let vl = !l;
      let i0 = 32sz *^ vw +^ vl;
      let a = x.(i0);
      let s1 = F32.fma_rn_ftz a a (zero #F32.t);
      let b = x.(i0 +^ 512sz);
      let s2 = F32.fma_rn_ftz b b s1;
      x_at (reveal p) (reveal xr) xs (SZ.v i0);
      x_at (reveal p) (reveal xr) xs (SZ.v i0 + 512);
      lane_step (reveal p) (reveal xr) (SZ.v vw) (SZ.v vl) a b s1 s2;
      lanes.(vl) <- s2;
      l := vl +^ 1sz;
    };
    let t = tree32 p (lane_e (reveal xr) (SZ.v vw)) lanes;
    pts_to_len lanes;
    with ws. assert (warps |-> ws);
    warp_upd (reveal p) (reveal xr) ws (SZ.v vw) t;
    warps.(vw) <- t;
    w := vw +^ 1sz;
  };
  with ws. assert (warps |-> ws);
  warp_final (reveal p) (reveal xr) ws;
  let ss = tree32 p (warp_e (reveal xr)) warps;
  ss_final (reveal p) (reveal xr) ss;
  ss
}

(*** Dot *)
open Kuiper
open Kuiper.Float.Casts
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = Kuiper.SizeT
module F32 = Kuiper.Float32
module BF16 = Kuiper.BFloat16

#set-options "--z3rlimit 20 --fuel 1 --ifuel 0"

noextract let prodf (w:S.seq real) (wo:int) (x:S.seq real) : nat -> real = fun k -> at w (wo + k) *. at x k

noextract let lane_g (g:nat -> real) (l:nat) : nat -> real = fun t -> g (8 * l + 256 * (t / 8) + t % 8)

noextract let lanesum (g:nat -> real) : nat -> real = fun l -> rs 32 (lane_g g l)

noextract let dh (g:nat -> real) : nat -> nat -> real = fun l c -> rows 8 (shift g (256 * c)) l

noextract let divmod8 (c j:nat) : Lemma (requires j < 8) (ensures (c * 8 + j) / 8 == c /\ (c * 8 + j) % 8 == j)
  = FStar.Math.Lemmas.lemma_div_plus j c 8;
    FStar.Math.Lemmas.lemma_mod_plus j c 8

noextract let lane_rows (g:nat -> real) (l c:nat)
  : Lemma (requires c < 4) (ensures rows 8 (lane_g g l) c == rowf (dh g) l c)
  = introduce forall (j:nat). j < 8 ==> row 8 (lane_g g l) c j == row 8 (shift g (256 * c)) l j
      with introduce _ ==> _ with divmod8 c j;
    rs_ext 8 (row 8 (lane_g g l) c) (row 8 (shift g (256 * c)) l)

noextract let lane_split (g:nat -> real) (l:nat)
  : Lemma (lanesum g l == rsr 4 (dh g) l)
  = rs_block 4 8 (lane_g g l);
    introduce forall (c:nat). c < 4 ==> rows 8 (lane_g g l) c == rowf (dh g) l c
      with introduce _ ==> _ with lane_rows g l c;
    rs_ext 4 (rows 8 (lane_g g l)) (rowf (dh g) l)

noextract let col_eq (g:nat -> real) (c:nat)
  : Lemma (rsc 32 (dh g) c == rows 256 g c)
  = rs_ext 32 (col (dh g) c) (rows 8 (shift g (256 * c)));
    rs_block 32 8 (shift g (256 * c));
    rs_ext 256 (shift g (256 * c)) (row 256 g c)

noextract let dot_eq (g:nat -> real)
  : Lemma (rs 32 (lanesum g) == rs 1024 g)
  = FStar.Classical.forall_intro (lane_split g);
    rs_ext 32 (lanesum g) (rsr 4 (dh g));
    rs_swap 32 4 (dh g);
    FStar.Classical.forall_intro (col_eq g);
    rs_ext 4 (rsc 32 (dh g)) (rows 256 g);
    rs_block 4 256 g

noextract let dot_rs_o (w:S.seq real) (wo:int) (x:S.seq real) (n:nat)
  : Lemma (dot w wo x n == rs n (fun k -> at w (wo + k) *. at x k))
  = vsum_rs n (fun k -> at w (wo + k) *. at x k)

noextract let dot_rs (w:S.seq real) (wo:int) (x:S.seq real)
  : Lemma (dot w wo x 1024 == rs 32 (lanesum (prodf w wo x)))
  = dot_rs_o w wo x 1024;
    rs_ext 1024 (fun k -> at w (wo + k) *. at x k) (prodf w wo x);
    dot_eq (prodf w wo x)

noextract let dlane_e (#e:Type) (tr hr:e -> S.seq real) (base:nat) : e -> nat -> real =
  fun r l -> lanesum (prodf (tr r) base (hr r)) l

noextract let dot_step (#e:Type) (p:e -> prop) (tr hr:e -> S.seq real) (base l c j:nat) (a b s s':F32.t)
  : Lemma (requires j < 8 /\
                    (forall (r:e). p r ==> a %~ at (tr r) (base + (8 * l + 256 * c + j)) /\
                                        b %~ at (hr r) (8 * l + 256 * c + j) /\
                                        s %~ rs (c * 8 + j) (lane_g (prodf (tr r) base (hr r)) l)) /\
                    (forall (xr yr zr:real). a %~ xr /\ b %~ yr /\ s %~ zr ==> s' %~ (xr *. yr +. zr)))
          (ensures forall (r:e). p r ==> s' %~ rs (c * 8 + j + 1) (lane_g (prodf (tr r) base (hr r)) l))
  = divmod8 c j;
    introduce forall (r:e). p r ==> s' %~ rs (c * 8 + j + 1) (lane_g (prodf (tr r) base (hr r)) l)
      with introduce _ ==> _ with (
        let g = prodf (tr r) base (hr r) in
        let t : nat = c * 8 + j in
        let ar = at (tr r) (base + (8 * l + 256 * c + j)) in
        let br = at (hr r) (8 * l + 256 * c + j) in
        fma_inst a b s s' ar br (rs t (lane_g g l));
        assert (lane_g g l t == ar *. br);
        assert (rs (t + 1) (lane_g g l) == rs t (lane_g g l) +. ar *. br))

noextract let dot_fin (#e:Type) (p:e -> prop) (tr hr:e -> S.seq real) (base:nat) (res:F32.t)
  : Lemma (requires forall (r:e). p r ==> res %~ rs 32 (dlane_e tr hr base r))
          (ensures forall (r:e). p r ==> res %~ dot (tr r) base (hr r) 1024)
  = introduce forall (r:e). p r ==> res %~ dot (tr r) base (hr r) 1024
      with introduce _ ==> _ with (
        rs_ext 32 (dlane_e tr hr base r) (lanesum (prodf (tr r) base (hr r)));
        dot_rs (tr r) base (hr r))

noextract let bf_at (#e:Type) (p:e -> prop) (tr:e -> S.seq real) (ts:S.seq BF16.t) (i:nat)
  : Lemma (requires (forall (r:e). p r ==> ts %~ tr r) /\ i < S.length ts)
          (ensures forall (r:e). p r ==> c_bf16_32.fcast (S.index ts i) %~ at (tr r) i)
  = ()

inline_for_extraction noextract
fn cdot (#e:Type0) (p:erased (e -> prop)) (tr hr:erased (e -> S.seq real))
  (tab:array BF16.t) (base:SZ.t) (x:larray F32.t 1024)
  (#ft #fx:perm) (#ts:erased (S.seq BF16.t)) (#xs:erased (S.seq F32.t))
  preserves gpu ** tab |-> Frac ft ts ** x |-> Frac fx xs
  requires pure (SZ.v base + 1024 <= S.length ts /\ SZ.fits (SZ.v base + 1024) /\
                 (forall (r:e). reveal p r ==> reveal ts %~ reveal tr r /\ reveal xs %~ reveal hr r))
  returns res:F32.t
  ensures pure (forall (r:e). reveal p r ==> res %~ dot (reveal tr r) (SZ.v base) (reveal hr r) 1024)
{
  pts_to_len x;
  let mut lanes : array F32.t = [| zero #F32.t; 32sz |];
  zero_approx ();
  let mut l = 0sz;
  while (!l <^ 32sz)
    invariant exists* (vl:SZ.t) (ls:S.seq F32.t). l |-> vl ** lanes |-> ls ** pure (
      SZ.v vl <= 32 /\ S.length ls == 32 /\
      arr_ok (reveal p) ls (SZ.v vl) (dlane_e (reveal tr) (reveal hr) (SZ.v base)))
    decreases (32 - SZ.v !l)
  {
    let vl = !l;
    let z0 : F32.t = zero;
    let mut s : F32.t = z0;
    let mut c = 0sz;
    while (!c <^ 4sz)
      invariant exists* (vc:SZ.t) (vs:F32.t). c |-> vc ** s |-> vs ** pure (
        SZ.v vc <= 4 /\
        (forall (r:e). reveal p r ==>
           vs %~ rs (SZ.v vc * 8) (lane_g (prodf (reveal tr r) (SZ.v base) (reveal hr r)) (SZ.v vl))))
      decreases (4 - SZ.v !c)
    {
      let vc = !c;
      let mut j = 0sz;
      while (!j <^ 8sz)
        invariant exists* (vj:SZ.t) (vs:F32.t). j |-> vj ** s |-> vs ** pure (
          SZ.v vj <= 8 /\
          (forall (r:e). reveal p r ==>
             vs %~ rs (SZ.v vc * 8 + SZ.v vj) (lane_g (prodf (reveal tr r) (SZ.v base) (reveal hr r)) (SZ.v vl))))
        decreases (8 - SZ.v !j)
      {
        let vj = !j;
        let k = 8sz *^ vl +^ 256sz *^ vc +^ vj;
        let a = c_bf16_32.fcast tab.(base +^ k);
        let b = x.(k);
        let vs = !s;
        let s2 = F32.fma_rn_ftz a b vs;
        bf_at (reveal p) (reveal tr) ts (SZ.v base + SZ.v k);
        x_at (reveal p) (reveal hr) xs (SZ.v k);
        dot_step (reveal p) (reveal tr) (reveal hr) (SZ.v base) (SZ.v vl) (SZ.v vc) (SZ.v vj) a b vs s2;
        s := s2;
        j := vj +^ 1sz;
      };
      c := vc +^ 1sz;
    };
    let vs = !s;
    lanes.(vl) <- vs;
    l := vl +^ 1sz;
  };
  let res = tree32 p (dlane_e (reveal tr) (reveal hr) (SZ.v base)) lanes;
  dot_fin (reveal p) (reveal tr) (reveal hr) (SZ.v base) res;
  res
}

(*** Kern *)
open Kuiper
open Kuiper.Float.Casts
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = Kuiper.SizeT
module F32 = Kuiper.Float32
module BF16 = Kuiper.BFloat16
module FB = Kuiper.Floating.Base
module FM = Kuiper.Float32.FastMath
module I64 = FStar.Int64
module U32 = FStar.UInt32
module Cast = FStar.Int.Cast
module SS = KPlay.SharedSpec
module Sqrt = FStar.Math.Sqrt

#set-options "--z3rlimit 20 --fuel 1 --ifuel 0"

noextract noeq type env = { rt : S.seq real; rn : S.seq real; rx : S.seq real }

noextract let pe (ts ns:erased (S.seq BF16.t)) (xs:erased (S.seq F32.t)) : env -> prop =
  fun r -> reveal ts %~ r.rt /\ reveal ns %~ r.rn /\ reveal xs %~ r.rx

noextract let ex : env -> S.seq real = fun r -> r.rx
let et : env -> S.seq real = fun r -> r.rt

noextract let inv_r (r:env) : real = rsqrt_t (rdiv (sq_sum r.rx 0 1024) (of_int 1024) +. eps_f32)

noextract let hn_r : env -> S.seq real = fun r -> rmsnorm1p eps_f32 r.rx 1024 r.rn 0

noextract let c1024_approx () : Lemma (FB.of_int #F32.t 1024L %~ of_int 1024) =
  Kuiper.Approximates.Base.approx_of_int #F32.t;
  assert (Kuiper.Approximates.Core.int64_can_approximate.approximates 1024L 1024L)

noextract let one_approx () : Lemma (one #F32.t %~ 1.0R) = Kuiper.Approximates.Scalar.a1 #F32.t

noextract let scale_ok (p:env -> prop) (ss c q ev a sc:F32.t)
  : Lemma (requires (forall (r:env). p r ==> ss %~ sq_sum r.rx 0 1024) /\
                    c %~ of_int 1024 /\ ev %~ eps_f32 /\ eps_f32 >. 0.0R /\
                    (forall (xr:real) (yr:real{yr =!= 0.0R}). ss %~ xr /\ c %~ yr ==> q %~ (xr /. yr)) /\
                    (forall (xr yr:real). q %~ xr /\ ev %~ yr ==> a %~ (xr +. yr)) /\
                    (forall (xr:Sqrt.rpos). a %~ xr ==> sc %~ Sqrt.rsqrt xr))
          (ensures forall (r:env). p r ==> sc %~ inv_r r)
  = introduce forall (r:env). p r ==> sc %~ inv_r r
      with introduce _ ==> _ with (
        let s = sq_sum r.rx 0 1024 in
        sq_nonneg r.rx 1024;
        let n : real = of_int 1024 in
        assert (n =!= 0.0R);
        let qd : real = s /. n in
        assert (qd >=. 0.0R);
        assert (q %~ qd);
        let ad : real = qd +. eps_f32 in
        assert (ad >. 0.0R);
        assert (a %~ ad);
        assert (rdiv s n == qd);
        assert (inv_r r == Sqrt.rsqrt ad))

noextract let norm_ok (p:env -> prop) (k:nat) (xi wi sc t1 t2 t3:F32.t)
  : Lemma (requires k < 1024 /\
                    (forall (r:env). p r ==> xi %~ at r.rx k /\ wi %~ at r.rn k /\ sc %~ inv_r r) /\
                    one #F32.t %~ 1.0R /\
                    (forall (xr yr:real). xi %~ xr /\ sc %~ yr ==> t1 %~ (xr *. yr)) /\
                    (forall (xr yr:real). one #F32.t %~ xr /\ wi %~ yr ==> t2 %~ (xr +. yr)) /\
                    (forall (xr yr:real). t1 %~ xr /\ t2 %~ yr ==> t3 %~ (xr *. yr)))
          (ensures forall (r:env). p r ==> t3 %~ S.index (hn_r r) k)
  = introduce forall (r:env). p r ==> t3 %~ S.index (hn_r r) k
      with introduce _ ==> _ with (
        let v1 = at r.rx k *. inv_r r in
        let v2 = 1.0R +. at r.rn k in
        assert (t1 %~ v1);
        assert (t2 %~ v2);
        assert (t3 %~ (v1 *. v2));
        assert (S.index (hn_r r) k == v1 *. v2))

noextract let norm_fin (p:env -> prop) (nv:S.seq F32.t)
  : Lemma (requires S.length nv == 1024 /\
                    (forall (r:env) (k:nat). p r /\ k < 1024 ==> S.index nv k %~ S.index (hn_r r) k))
          (ensures forall (r:env). p r ==> nv %~ hn_r r)
  = ()

noextract let row_out (k d:nat) : Lemma (requires k / 1024 <> d) (ensures k < d * 1024 \/ d * 1024 + 1024 <= k)
  = if d * 1024 <= k && k < d * 1024 + 1024 then begin
      FStar.Math.Lemmas.lemma_div_plus (k - d * 1024) d 1024;
      FStar.Math.Lemmas.small_div (k - d * 1024) 1024
    end

noextract let kpost (ts ns:S.seq BF16.t) (xs:S.seq F32.t) (h0:S.seq BF16.t) (valid:bool) (du:nat)
          (sc:S.seq F32.t) (h1:S.seq BF16.t) : prop =
  S.length sc == 248320 /\ S.length h1 == S.length h0 /\
  (forall (k:nat). k < S.length h0 /\ (valid ==> k / 1024 <> du) ==> S.index h1 k == S.index h0 k) /\
  (forall (rt rn rx:S.seq real). ts %~ rt /\ ns %~ rn /\ xs %~ rx ==>
     sc %~ logits opt_numerics rt 0 rn 0 rx /\ (valid ==> bf_row h1 du %~ rx))

noextract let scores_fin (ts ns:S.seq BF16.t) (xs:S.seq F32.t) (sv:S.seq F32.t) (rt rn rx:S.seq real)
  : Lemma (requires S.length sv == 248320 /\
                    (forall (r:env) (k:nat). pe ts ns xs r /\ k < 248320 ==>
                        S.index sv k %~ dot r.rt (k * 1024) (hn_r r) 1024) /\
                    ts %~ rt /\ ns %~ rn /\ xs %~ rx)
          (ensures sv %~ logits opt_numerics rt 0 rn 0 rx)
  = let r = { rt = rt; rn = rn; rx = rx } in
    assert (pe ts ns xs r);
    assert (opt_numerics.eps == eps_f32);
    introduce forall (k:nat). k < 248320 ==> S.index sv k %~ S.index (logits opt_numerics rt 0 rn 0 rx) k
      with introduce _ ==> _ with (
        assert (S.index (logits opt_numerics rt 0 rn 0 rx) k == dot rt (0 + k * 1024) (hn_r r) 1024))

noextract let hist_fin (xs:S.seq F32.t) (h1:S.seq BF16.t) (du:nat) (rx:S.seq real)
  : Lemma (requires S.length xs == 1024 /\ du * 1024 + 1024 <= S.length h1 /\
                    (forall (i:nat). i < 1024 ==> S.index h1 (du * 1024 + i) == c_32_bf16.fcast (S.index xs i)) /\
                    xs %~ rx)
          (ensures bf_row h1 du %~ rx)
  = introduce forall (i:nat). i < 1024 ==> S.index (bf_row h1 du) i %~ S.index rx i
      with introduce _ ==> _ with (
        assert (S.index (bf_row h1 du) i == S.index h1 (du * 1024 + i));
        assert (S.index xs i %~ S.index rx i))

noextract let kfin_pre (ts ns:S.seq BF16.t) (xs:S.seq F32.t) (sv:S.seq F32.t) : prop =
  S.length sv == 248320 /\
  (forall (r:env) (k:nat). pe ts ns xs r /\ k < 248320 ==>
     S.index sv k %~ dot r.rt (k * 1024) (hn_r r) 1024)

noextract let hpost (xs:S.seq F32.t) (h0:S.seq BF16.t) (valid:bool) (du:nat) (h1:S.seq BF16.t) : prop =
  S.length h1 == S.length h0 /\
  (forall (k:nat). k < S.length h0 /\ (valid ==> k / 1024 <> du) ==> S.index h1 k == S.index h0 k) /\
  (forall (rx:S.seq real). xs %~ rx /\ valid ==> bf_row h1 du %~ rx)

noextract let hist_post (xs:S.seq F32.t) (h0 h:S.seq BF16.t) (du:nat)
  : Lemma (requires S.length xs == 1024 /\ S.length h == S.length h0 /\ du * 1024 + 1024 <= S.length h0 /\
                    (forall (k:nat). k < S.length h0 /\ (k < du * 1024 \/ du * 1024 + 1024 <= k) ==>
                        S.index h k == S.index h0 k) /\
                    (forall (k:nat). k < 1024 ==> S.index h (du * 1024 + k) == c_32_bf16.fcast (S.index xs k)))
          (ensures hpost xs h0 true du h)
  = introduce forall (k:nat). k < S.length h0 /\ k / 1024 <> du ==> S.index h k == S.index h0 k
      with introduce _ ==> _ with row_out k du;
    introduce forall (rx:S.seq real). xs %~ rx ==> bf_row h du %~ rx
      with introduce _ ==> _ with hist_fin xs h du rx

noextract let kfin (ts ns:S.seq BF16.t) (xs:S.seq F32.t) (h0:S.seq BF16.t) (valid:bool) (du:nat)
         (sv:S.seq F32.t) (h1:S.seq BF16.t)
  : Lemma (requires kfin_pre ts ns xs sv /\ hpost xs h0 valid du h1)
          (ensures kpost ts ns xs h0 valid du sv h1)
  = introduce forall (rt rn rx:S.seq real). ts %~ rt /\ ns %~ rn /\ xs %~ rx ==>
        sv %~ logits opt_numerics rt 0 rn 0 rx
      with introduce _ ==> _ with scores_fin ts ns xs sv rt rn rx

inline_for_extraction noextract
fn hist_write (x:larray F32.t 1024) (ho:array BF16.t) (du:SZ.t) (valid:bool)
  (#fx:perm) (#xs:erased (S.seq F32.t)) (#h0:erased (S.seq BF16.t))
  preserves gpu ** x |-> Frac fx xs
  requires ho |-> h0 ** pure (valid ==> SZ.v du * 1024 + 1024 <= S.length h0 /\ SZ.fits (S.length h0))
  ensures exists* h1. ho |-> h1 ** pure (hpost xs h0 valid (SZ.v du) h1)
{
  pts_to_len x; pts_to_len ho;
  if valid {
    let mut j = 0sz;
    while (!j <^ 1024sz)
      invariant exists* (vj:SZ.t) (h:S.seq BF16.t). j |-> vj ** ho |-> h ** pure (
        SZ.v vj <= 1024 /\ S.length h == S.length h0 /\
        (forall (k:nat). k < S.length h0 /\ (k < SZ.v du * 1024 \/ SZ.v du * 1024 + SZ.v vj <= k) ==>
            S.index h k == S.index h0 k) /\
        (forall (k:nat). k < SZ.v vj ==> S.index h (SZ.v du * 1024 + k) == c_32_bf16.fcast (S.index xs k)))
      decreases (1024 - SZ.v !j)
    {
      let vj = !j;
      let xv = x.(vj);
      ho.(du *^ 1024sz +^ vj) <- c_32_bf16.fcast xv;
      j := vj +^ 1sz;
    };
    with h. assert (ho |-> h);
    hist_post xs h0 h (SZ.v du);
  } else {
    ()
  }
}

#push-options "--z3rlimit 100"
inline_for_extraction noextract
fn fp_kernel
  (tab:array BF16.t) (nw:larray BF16.t 1024) (x:larray F32.t 1024)
  (sc:larray F32.t 248320) (ho:array BF16.t) (hn:SZ.t) (du:SZ.t) (valid:bool)
  (#ft #fw #fx:perm) (#ts #ns:erased (S.seq BF16.t)) (#xs:erased (S.seq F32.t))
  (#sc0:erased (S.seq F32.t)) (#h0:erased (S.seq BF16.t))
  preserves gpu ** tab |-> Frac ft ts ** nw |-> Frac fw ns ** x |-> Frac fx xs
  requires sc |-> sc0 ** ho |-> h0 ** pure (
    S.length ts == table_len /\ S.length h0 == SZ.v hn /\
    (valid ==> SZ.v du * 1024 + 1024 <= SZ.v hn))
  ensures exists* (sc1:S.seq F32.t) (h1:S.seq BF16.t). sc |-> sc1 ** ho |-> h1 **
    pure (kpost ts ns xs h0 valid (SZ.v du) sc1 h1)
{
  pts_to_len x; pts_to_len nw; pts_to_len sc; pts_to_len ho;
  let ss = square_sum (pe ts ns xs) ex x;
  let c = FB.of_int #F32.t 1024L;
  let q = FM.divide ss c;
  let ev = FB.of_literal #F32.t "0.0000009999999974752427078783512115478515625";
  let a = F32.add_rn_ftz q ev;
  let scale = F32.rsqrt_approx_ftz a;
  c1024_approx ();
  eps_facts ();
  scale_ok (pe ts ns xs) ss c q ev a scale;
  one_approx ();
  let mut norm : array F32.t = [| zero #F32.t; 1024sz |];
  let mut i = 0sz;
  while (!i <^ 1024sz)
    invariant exists* (vi:SZ.t) (nv:S.seq F32.t). i |-> vi ** norm |-> nv ** pure (
      SZ.v vi <= 1024 /\ S.length nv == 1024 /\
      (forall (r:env) (k:nat). pe ts ns xs r /\ k < SZ.v vi ==> S.index nv k %~ S.index (hn_r r) k))
    decreases (1024 - SZ.v !i)
  {
    let vi = !i;
    let xi = x.(vi);
    let t1 = F32.mul_rn_ftz xi scale;
    let wi = c_bf16_32.fcast nw.(vi);
    let t2 = F32.add_rn_ftz (one #F32.t) wi;
    let t3 = F32.mul_rn_ftz t1 t2;
    norm_ok (pe ts ns xs) (SZ.v vi) xi wi scale t1 t2 t3;
    norm.(vi) <- t3;
    i := vi +^ 1sz;
  };
  with nv. assert (norm |-> nv);
  norm_fin (pe ts ns xs) nv;
  hist_write x ho du valid;
  with (h1:S.seq BF16.t). assert (ho |-> h1);
  let mut m = 0sz;
  let nsc = SZ.uint32_to_sizet 248320ul;
  while (!m <^ nsc)
    invariant exists* (vm:SZ.t) (sv:S.seq F32.t). m |-> vm ** sc |-> sv ** pure (
      SZ.v vm <= 248320 /\ S.length sv == 248320 /\
      (forall (r:env) (k:nat). pe ts ns xs r /\ k < SZ.v vm ==>
         S.index sv k %~ dot r.rt (k * 1024) (hn_r r) 1024))
    decreases (248320 - SZ.v !m)
  {
    let vm = !m;
    let res = cdot (pe ts ns xs) et hn_r tab (vm *^ 1024sz) norm;
    sc.(vm) <- res;
    m := vm +^ 1sz;
  };
  with (sv:S.seq F32.t). assert (sc |-> sv);
  assert pure (kfin_pre ts ns xs sv);
  assert pure (hpost xs h0 valid (SZ.v du) h1);
  kfin ts ns xs h0 valid (SZ.v du) sv h1;
  assert pure (kpost ts ns xs h0 valid (SZ.v du) sv h1);
  ()
}
#pop-options

noextract let du_ok (d:I64.t) (du hn:SZ.t) (ok valid:bool)
  : Lemma (requires (ok == (I64.v d >= 0 && I64.v d < 4194304)) /\
                    (ok ==> SZ.v du == I64.v d % pow2 32) /\
                    (valid == (ok && SZ.v du < SZ.v hn / 1024)))
          (ensures (valid ==> SZ.v du == I64.v d /\ SZ.v du * 1024 + 1024 <= SZ.v hn) /\
                   ((0 <= I64.v d /\ I64.v d < SZ.v hn / 1024) ==> valid))
  = if ok then begin
      FStar.Math.Lemmas.small_mod (I64.v d) (pow2 32);
      if valid then begin
        let q : nat = SZ.v hn / 1024 in
        FStar.Math.Lemmas.lemma_div_mod (SZ.v hn) 1024;
        FStar.Math.Lemmas.lemma_mult_le_right 1024 (SZ.v du + 1) q
      end
    end

noextract let rel_ok (ts ns:S.seq BF16.t) (xs:S.seq F32.t) (hs:S.seq BF16.t) (d:int) (valid:bool) (du:nat)
           (sv:S.seq F32.t) (hv:S.seq BF16.t)
  : Lemma (requires kpost ts ns xs hs valid du sv hv /\
                    (valid ==> du == d) /\
                    ((0 <= d /\ d < S.length hs / 1024) ==> valid))
          (ensures SS.optimized_final_projection_relation ts ns xs hs d sv hv)
  = ()

instance send_global_array_contents
  (#a : Type0) (arr : array a{is_global arr}) (#f : perm) (contents : S.seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

ghost
fn gpu_len (#a:Type0) (arr:array a) (#f:perm) (#s:erased (S.seq a))
  requires on gpu_loc (arr |-> Frac f s)
  ensures on gpu_loc (arr |-> Frac f s) ** pure (S.length s == Pulse.Lib.Array.length arr)
{
  ghost_impersonate gpu_loc (on gpu_loc (arr |-> Frac f s))
    (on gpu_loc (arr |-> Frac f s) ** pure (S.length s == Pulse.Lib.Array.length arr)) fn _ {
    on_elim (arr |-> Frac f s);
    pts_to_len arr;
    on_intro (arr |-> Frac f s);
  }
}

fn kplay_run
  (hn:SZ.t)
  (tab:larray BF16.t 254279680{is_global tab})
  (nw:larray BF16.t 1024{is_global nw})
  (x:larray F32.t 1024{is_global x})
  (hin:larray BF16.t (SZ.v hn){is_global hin})
  (sc:larray F32.t 248320{is_global sc})
  (hout:larray BF16.t (SZ.v hn){is_global hout})
  (d:I64.t)
  (#ft #fw #fx #fh:perm)
  (#ts #ns:erased (S.seq BF16.t)) (#xs:erased (S.seq F32.t)) (#hs:erased (S.seq BF16.t))
  (#sc0:erased (S.seq F32.t)) (#ho0:erased (S.seq BF16.t))
  preserves cpu ** on gpu_loc (tab |-> Frac ft ts) ** on gpu_loc (nw |-> Frac fw ns) **
            on gpu_loc (x |-> Frac fx xs) ** on gpu_loc (hin |-> Frac fh hs)
  requires on gpu_loc (sc |-> sc0) ** on gpu_loc (hout |-> ho0)
  ensures exists* (sv:S.seq F32.t) (hv:S.seq BF16.t).
    on gpu_loc (sc |-> sv) ** on gpu_loc (hout |-> hv) **
    pure (SS.optimized_final_projection_relation ts ns xs hs (I64.v d) sv hv)
{
  gpu_len tab; gpu_len hin; gpu_len hout;
  memcpy_device_to_device hout hin hn;
  let ok = (I64.gte d 0L && I64.lt d 4194304L);
  let du : SZ.t = (if ok then SZ.uint32_to_sizet (Cast.int64_to_uint32 d) else 0sz);
  let valid = (ok && SZ.lt du (hn /^ 1024sz));
  du_ok d du hn ok valid;
  launch_kernel_1 (fun () -> fp_kernel tab nw x sc hout hn du valid #ft #fw #fx #ts #ns #xs #sc0 #hs);
  with (sv:S.seq F32.t) (hv:S.seq BF16.t). assert (on gpu_loc (sc |-> sv) ** on gpu_loc (hout |-> hv));
  rel_ok ts ns xs hs (I64.v d) valid (SZ.v du) sv hv;
}
