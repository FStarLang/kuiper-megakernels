module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Approximates
open Kuiper.Float.Realops
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = FStar.SizeT
module I64 = FStar.Int64
module U32 = FStar.UInt32
module Cast = FStar.Int.Cast
module Fast = Kuiper.Float32.FastMath
module FB = Kuiper.Floating.Base
module F32 = Kuiper.Float32

#set-options "--z3rlimit 30"

(* ------------------------------------------------------------------ *)
(* Native view of the kernel's maximum recurrence                        *)

noextract
let nidx (s:S.seq f32) (i:nat) : f32 =
  if i < S.length s then S.index s i else zero

(* maximum after visiting experts 0..k-1 of row t (fmaxf from -inf). *)
noextract
let rec smax (s:S.seq f32) (t:nat) (k:nat) : Tot f32 (decreases k) =
  if k = 0 then ninf else fmax (smax s t (k - 1)) (nidx s (8 * t + (k - 1)))

(* ------------------------------------------------------------------ *)
(* Real view                                                              *)

(* sum_{j<k} exp rs[8t+j] *)
noextract
let rec esum (rs:S.seq real) (t:nat) (k:nat) : Tot real (decreases k) =
  if k = 0 then 0.0R else esum rs t (k - 1) +. exp (ridx rs (8 * t + (k - 1)))

let rec esum_pos (rs:S.seq real) (t:nat) (k:pos)
  : Lemma (ensures esum rs t k >. 0.0R) (decreases k)
  = if k = 1 then () else esum_pos rs t (k - 1)

(* The ideal the running denominator approximates after k experts, with the
   (arbitrary) ideal c of the running maximum: sum_{j<k} exp (r_j - c). *)
noextract
let den_ideal (rs:S.seq real) (t:nat) (k:nat) (c:real) : real =
  esum rs t k /. exp c

noextract
let den_ok (s:S.seq f32) (t:nat) (k:nat) (d:f32) : prop =
  forall (rs:S.seq real). s %~ rs ==>
    d %~ den_ideal rs t k (to_real (smax s t k))

noextract
let out_ideal (rs:S.seq real) (t:nat) (j:nat) : real =
  esum_pos rs t 8;
  exp (ridx rs (8 * t + j)) /. esum rs t 8

noextract
let out_ok (s:S.seq f32) (t:nat) (j:nat) (o:f32) : prop =
  forall (rs:S.seq real). s %~ rs ==> o %~ out_ideal rs t j

(* ------------------------------------------------------------------ *)
(* Real algebra                                                           *)

let real_step (e r c c':real)
  : Lemma ((e /. exp c) *. exp (c -. c') +. exp (r -. c') == (e +. exp r) /. exp c')
  = exp_sub c c';
    exp_sub r c';
    let ec = exp c in
    let ec' = exp c' in
    let er = exp r in
    assert ((e /. ec) *. (ec /. ec') == e /. ec');
    assert (e /. ec' +. er /. ec' == (e +. er) /. ec')

let real_out (r c:real) (e:real{e >. 0.0R})
  : Lemma (exp (r -. c) /. (e /. exp c) == exp r /. e)
  = exp_sub r c;
    let ec = exp c in
    let er = exp r in
    assert ((er /. ec) /. (e /. ec) == er /. e)

(* ------------------------------------------------------------------ *)
(* Loop lemmas                                                            *)

let den_init (s:S.seq f32) (t:nat)
  : Lemma (den_ok s t 0 zero)
  = ()

let score_approx (s:S.seq f32) (rs:S.seq real) (i:nat)
  : Lemma (requires s %~ rs /\ i < S.length s)
          (ensures nidx s i %~ ridx rs i)
  = ()

let den_step (s:S.seq f32) (t:nat) (k:nat) (d a ea b eb d':f32)
  : Lemma
    (requires
      8 * t + k < S.length s /\ den_ok s t k d /\
      a == Fast.pure_sub_rn (smax s t k) (smax s t (k + 1)) /\
      b == Fast.pure_sub_rn (nidx s (8 * t + k)) (smax s t (k + 1)) /\
      (forall (xr:real). v_approximates a xr ==> v_approximates ea (r_exp xr)) /\
      (forall (xr:real). v_approximates b xr ==> v_approximates eb (r_exp xr)) /\
      (forall (xr yr zr:real).
        v_approximates d xr /\ v_approximates ea yr /\ v_approximates eb zr ==>
        v_approximates d' (r_fma xr yr zr)))
    (ensures den_ok s t (k + 1) d')
  = let m = smax s t k in
    let m' = smax s t (k + 1) in
    let x = nidx s (8 * t + k) in
    let aux (rs:S.seq real)
      : Lemma (requires s %~ rs)
              (ensures d' %~ den_ideal rs t (k + 1) (to_real m'))
      = let c = to_real m in
        let c' = to_real m' in
        let r = ridx rs (8 * t + k) in
        to_real_ok m;
        to_real_ok m';
        score_approx s rs (8 * t + k);
        assert (v_approximates a (c -. c'));
        assert (v_approximates b (r -. c'));
        assert (v_approximates ea (exp (c -. c')));
        assert (v_approximates eb (exp (r -. c')));
        assert (v_approximates d (den_ideal rs t k c));
        real_step (esum rs t k) r c c';
        assert (esum rs t (k + 1) == esum rs t k +. exp r)
    in
    Classical.forall_intro (Classical.move_requires aux)

let out_step (s:S.seq f32) (t:nat) (j:nat) (d b eb o:f32)
  : Lemma
    (requires
      8 * t + 8 <= S.length s /\ j < 8 /\ den_ok s t 8 d /\
      b == Fast.pure_sub_rn (nidx s (8 * t + j)) (smax s t 8) /\
      (forall (xr:real). v_approximates b xr ==> v_approximates eb (r_exp xr)) /\
      (forall (xr:real) (yr:real{yr =!= 0.0R}).
        v_approximates eb xr /\ v_approximates d yr ==>
        v_approximates o (r_div xr yr)))
    (ensures out_ok s t j o)
  = let m = smax s t 8 in
    let aux (rs:S.seq real)
      : Lemma (requires s %~ rs)
              (ensures o %~ out_ideal rs t j)
      = let c = to_real m in
        let r = ridx rs (8 * t + j) in
        to_real_ok m;
        score_approx s rs (8 * t + j);
        esum_pos rs t 8;
        assert (v_approximates b (r -. c));
        assert (v_approximates eb (exp (r -. c)));
        assert (v_approximates d (den_ideal rs t 8 c));
        real_out r c (esum rs t 8)
    in
    Classical.forall_intro (Classical.move_requires aux)

(* ------------------------------------------------------------------ *)
(* From the per-element facts to the shared relation                      *)

let rec rsum_prefix (rs:S.seq real) (t:nat) (k:nat)
  : Lemma
    (requires 8 * t + 8 <= S.length rs /\ k <= 8)
    (ensures rsum (S.slice (exp_row (row_of rs t)) 0 k) == esum rs t k)
    (decreases k)
  = let er = exp_row (row_of rs t) in
    if k = 0 then begin
      assert (S.length (S.slice er 0 0) == 0);
      assert_spinoff (rsum (S.slice er 0 0) == 0.0R)
    end else begin
      rsum_prefix rs t (k - 1);
      Kuiper.Seq.Common.lemma_seq_fold_left_slice 0.0R (+.) er 0 (k - 1)
    end

let rsum_row (rs:S.seq real) (t:nat)
  : Lemma (requires 8 * t + 8 <= S.length rs)
          (ensures rsum (exp_row (row_of rs t)) == esum rs t 8)
  = rsum_prefix rs t 8;
    assert (S.equal (S.slice (exp_row (row_of rs t)) 0 8) (exp_row (row_of rs t)))

let final_relation (s:S.seq f32) (t:nat) (v:S.seq f32)
  : Lemma
    (requires S.length s == tokens * experts /\ t < tokens /\ S.length v == 8 /\
              (forall (j:nat). j < 8 ==> out_ok s t j (S.index v j)))
    (ensures optimized_softmax_relation s t v)
  = let aux (rs:S.seq real)
      : Lemma (requires s %~ rs)
              (ensures v %~ ref_softmax (row_of rs t))
      = rsum_row rs t;
        let rr = row_of rs t in
        let p = ref_softmax rr in
        let elt (j:nat{j < 8})
          : Lemma (S.index v j %~ S.index p j)
          = assert (out_ok s t j (S.index v j));
            assert (S.index rr j == ridx rs (8 * t + j))
        in
        Classical.forall_intro elt
    in
    Classical.forall_intro (Classical.move_requires aux)

(* ------------------------------------------------------------------ *)
(* Kernel: softmax_row<<<1, 1>>> of the CUDA source                      *)

let smax_step (s:S.seq f32) (t:nat) (k:nat)
  : Lemma (smax s t (k + 1) == fmax (smax s t k) (nidx s (8 * t + k)))
  = ()

inline_for_extraction noextract
fn softmax_row
  (scores : larray f32 2048)
  (t : SZ.t{SZ.v t < 256})
  (out : larray f32 8)
  (#s : erased (lseq f32 2048))
  (#before : erased (S.seq f32))
  preserves gpu ** scores |-> s
  requires out |-> before
  ensures exists* after.
    out |-> after ** pure (optimized_softmax_relation s (SZ.v t) after)
{
  pts_to_len out;
  let neg_inf : f32 = FB.neg FB.infinity;
  assert (pure (neg_inf == smax s (SZ.v t) 0));
  den_init s (SZ.v t);
  let mut maximum : f32 = neg_inf;
  let mut denominator : f32 = zero;
  let mut e = 0sz;
  while (!e <^ 8sz)
    invariant exists* (k : SZ.t) (m d : f32).
      e |-> k ** maximum |-> m ** denominator |-> d **
      pure (SZ.v k <= 8 /\ m == smax s (SZ.v t) (SZ.v k) /\ den_ok s (SZ.v t) (SZ.v k) d)
    decreases (8 - SZ.v !e)
  {
    let k = !e;
    let x = scores.(8sz *^ t +^ k);
    let previous = !maximum;
    let mx = fmax previous x;
    maximum := mx;
    let a = Fast.sub_rn previous mx;
    let ea = Fast.exp a;
    let b = Fast.sub_rn x mx;
    let eb = Fast.exp b;
    let dprev = !denominator;
    let dn = Fast.fma_rn dprev ea eb;
    denominator := dn;
    smax_step s (SZ.v t) (SZ.v k);
    den_step s (SZ.v t) (SZ.v k) dprev a ea b eb dn;
    e := k +^ 1sz;
  };
  let mfin = !maximum;
  let dfin = !denominator;
  let mut j = 0sz;
  while (!j <^ 8sz)
    invariant exists* (k : SZ.t) (v : S.seq f32).
      j |-> k ** out |-> v **
      pure (SZ.v k <= 8 /\ S.length v == 8 /\
            (forall (i:nat). i < SZ.v k ==> out_ok s (SZ.v t) i (S.index v i)))
    decreases (8 - SZ.v !j)
  {
    let k = !j;
    let x = scores.(8sz *^ t +^ k);
    let b = Fast.sub_rn x mfin;
    let eb = Fast.exp b;
    let o = Fast.divide eb dfin;
    out_step s (SZ.v t) (SZ.v k) dfin b eb o;
    out.(k) <- o;
    j := k +^ 1sz;
  };
  with v. assert out |-> v;
  assert (pure (S.length v == 8));
  assert (pure (forall (i:nat). i < 8 ==> out_ok s (SZ.v t) i (S.index v i)));
  assert (pure (S.length s == tokens * experts));
  final_relation s (SZ.v t) v;
  assert (pure (optimized_softmax_relation s (SZ.v t) v));
}

(* launch_kernel_1 needs gpu_of sendability for global array contents. *)
instance send_global_array_contents
  (#a : Type0)
  (arr : array a{is_global arr})
  (#f : perm)
  (contents : S.seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn kplay_run
  (scores : larray f32 2048 {is_global scores})
  (token : i64)
  (out : larray f32 8 {is_global out})
  (#s : erased (lseq f32 2048))
  preserves cpu ** on gpu_loc (scores |-> s)
  requires exists* before. on gpu_loc (out |-> before)
  ensures exists* after.
    on gpu_loc (out |-> after) **
    pure (optimized_softmax_relation s (I64.v token) after)
{
  with before. assert on gpu_loc (out |-> before);
  if (I64.gte token 0L && I64.lt token 256L) {
    let t = SZ.uint32_to_sizet (Cast.int64_to_uint32 token);
    assert (pure (SZ.v t == I64.v token));
    launch_kernel_1 (fun () -> softmax_row scores t out #s #before);
    ()
  } else {
    ()
  }
}
