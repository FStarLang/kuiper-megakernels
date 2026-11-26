module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Approximates
open Kuiper.Float.Realops
open Kuiper.Floating.Base
open Kuiper.Seq.Common
open KPlay.SharedSpec
module S = FStar.Seq
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32
module I64 = FStar.Int64
module SZ = Kuiper.SizeT
module Cast = Kuiper.Float.Casts.Base
module Fast = Kuiper.Float32.FastMath
module CE = Kuiper.KPlay.CpuExp

(* ---------------- Real targets ---------------- *)

noextract
let rsc (rx rw:S.seq real) (t e:nat) : real = ref_score rx rw t e

noextract
let rec rmx (rx rw:S.seq real) (t:nat) (k:pos) : Tot real (decreases k) =
  if k = 1 then rsc rx rw t 0
  else rmax (rmx rx rw t (k - 1)) (rsc rx rw t (k - 1))

noextract
let rmaxrow (rx rw:S.seq real) (t:nat) : real = rmx rx rw t experts

noextract
let rex (rx rw:S.seq real) (t e:nat) : real = exp (rsc rx rw t e -. rmaxrow rx rw t)

noextract
let rec rden (rx rw:S.seq real) (t:nat) (k:pos) : Tot real (decreases k) =
  if k = 1 then rex rx rw t 0
  else rden rx rw t (k - 1) +. rex rx rw t (k - 1)

noextract
let rec rden_pos (rx rw:S.seq real) (t:nat) (k:pos)
  : Lemma (ensures rden rx rw t k >. 0.0R) (decreases k)
  = if k = 1 then () else rden_pos rx rw t (k - 1)

(* ---------------- Native predicates ---------------- *)

noextract
let ins_ok (vx vw:S.seq BF.t) (rx rw:S.seq real) : prop = vx %~ rx /\ vw %~ rw

noextract
let acc_ok (vx vw:S.seq BF.t) (t e n:nat) (acc:F32.t) : prop =
  forall (rx rw:S.seq real). ins_ok vx vw rx rw ==>
    acc %~ dot_prefix hidden experts rx rw t e n

noextract
let score_cell (vx vw:S.seq BF.t) (t e:nat) (v:F32.t) : prop =
  forall (rx rw:S.seq real). ins_ok vx vw rx rw ==> v %~ rsc rx rw t e

noextract
let max_ok (vx vw:S.seq BF.t) (t:nat) (k:pos) (m:F32.t) : prop =
  forall (rx rw:S.seq real). ins_ok vx vw rx rw ==> m %~ rmx rx rw t k

noextract
let exp_cell (vx vw:S.seq BF.t) (t e:nat) (v:F32.t) : prop =
  forall (rx rw:S.seq real). ins_ok vx vw rx rw ==> v %~ rex rx rw t e

noextract
let den_ok (vx vw:S.seq BF.t) (t:nat) (k:pos) (d:F32.t) : prop =
  forall (rx rw:S.seq real). ins_ok vx vw rx rw ==> d %~ rden rx rw t k

noextract
let inv_ok (vx vw:S.seq BF.t) (t:nat) (d:F32.t) : prop =
  forall (rx rw:S.seq real). ins_ok vx vw rx rw ==>
    (rden_pos rx rw t experts; d %~ (1.0R /. rden rx rw t experts))

noextract
let row_index (rx rw:S.seq real) (t i:nat)
  : Lemma (requires t < tokens /\ i < experts)
          (ensures S.length (row_of (ref_scores rx rw) t) == experts /\
                   S.length (ref_softmax (row_of (ref_scores rx rw) t)) == experts /\
                   S.index (row_of (ref_scores rx rw) t) i == rsc rx rw t i)
  = assert ((8 * t + i) / 8 == t);
    assert ((8 * t + i) % 8 == i)

noextract
let final_cell (vx vw:S.seq BF.t) (t:nat{t < tokens}) (e:nat{e < experts}) (v:F32.t) : prop =
  forall (rx rw:S.seq real). ins_ok vx vw rx rw ==>
    (row_index rx rw t e; v %~ S.index (ref_softmax (row_of (ref_scores rx rw) t)) e)

(* ---------------- Step lemmas ---------------- *)

noextract
let acc_init (vx vw:S.seq BF.t) (t e:nat)
  : Lemma (acc_ok vx vw t e 0 zero)
  = ()

noextract
let fma_post (a b c res:F32.t) : prop =
  forall (xr yr zr:real).
    v_approximates a xr /\ v_approximates b yr /\ v_approximates c zr ==>
    v_approximates res (xr *. yr +. zr)

noextract
let acc_step (vx vw:S.seq BF.t) (t e n:nat) (acc res:F32.t)
  : Lemma (requires S.length vx == tokens * hidden /\ S.length vw == hidden * experts /\
                    t < tokens /\ e < experts /\ n < hidden /\ acc_ok vx vw t e n acc /\
                    fma_post (Cast.cast_bf16_to_f32 (S.index vx (t * hidden + n)))
                             (Cast.cast_bf16_to_f32 (S.index vw (n * experts + e))) acc res)
          (ensures acc_ok vx vw t e (n + 1) res)
  = let aux (rx rw:S.seq real)
      : Lemma (requires ins_ok vx vw rx rw)
              (ensures res %~ dot_prefix hidden experts rx rw t e (n + 1))
      = let i = t * hidden + n in
        let j = n * experts + e in
        assert (S.index vx i %~ S.index rx i);
        assert (S.index vw j %~ S.index rw j);
        Cast.cast_bf16_to_f32_ok (S.index vx i) (S.index rx i);
        Cast.cast_bf16_to_f32_ok (S.index vw j) (S.index rw j);
        let dn = dot_prefix hidden experts rx rw t e n in
        assert (ridx rx i == S.index rx i);
        assert (ridx rw j == S.index rw j);
        assert (dot_prefix hidden experts rx rw t e (n + 1) == dn +. S.index rx i *. S.index rw j);
        assert (acc %~ dn);
        assert (res %~ (S.index rx i *. S.index rw j +. dn))
    in
    Classical.forall_intro_2 (fun rx rw -> Classical.move_requires (aux rx) rw)

noextract
let acc_done (vx vw:S.seq BF.t) (t e:nat) (acc:F32.t)
  : Lemma (requires acc_ok vx vw t e hidden acc)
          (ensures score_cell vx vw t e acc)
  = let aux (rx rw:S.seq real)
      : Lemma (requires ins_ok vx vw rx rw) (ensures acc %~ rsc rx rw t e)
      = ()
    in
    Classical.forall_intro_2 (fun rx rw -> Classical.move_requires (aux rx) rw)

noextract
let max_init (vx vw:S.seq BF.t) (t:nat) (v:F32.t)
  : Lemma (requires score_cell vx vw t 0 v) (ensures max_ok vx vw t 1 v)
  = ()

noextract
let max_step (vx vw:S.seq BF.t) (t:nat) (k:pos) (m v:F32.t)
  : Lemma (requires max_ok vx vw t k m /\ score_cell vx vw t k v)
          (ensures max_ok vx vw t (k + 1) (fmax m v))
  = let aux (rx rw:S.seq real)
      : Lemma (requires ins_ok vx vw rx rw)
              (ensures fmax m v %~ rmx rx rw t (k + 1))
      = fmax_models_pat F32.t;
        approx_apply2 (fmax #F32.t) r_fmax m v (rmx rx rw t k) (rsc rx rw t k)
    in
    Classical.forall_intro_2 (fun rx rw -> Classical.move_requires (aux rx) rw)

noextract
let sub_post (a b res:F32.t) : prop =
  forall (xr yr:real).
    v_approximates a xr /\ v_approximates b yr ==> v_approximates res (xr -. yr)

noextract
let exp_step (vx vw:S.seq BF.t) (t e:nat) (m v d:F32.t)
  : Lemma (requires max_ok vx vw t experts m /\ score_cell vx vw t e v /\ sub_post v m d)
          (ensures exp_cell vx vw t e (CE.cpu_exp_f32 d))
  = let aux (rx rw:S.seq real)
      : Lemma (requires ins_ok vx vw rx rw)
              (ensures CE.cpu_exp_f32 d %~ rex rx rw t e)
      = CE.cpu_exp_f32_approx d (rsc rx rw t e -. rmaxrow rx rw t)
    in
    Classical.forall_intro_2 (fun rx rw -> Classical.move_requires (aux rx) rw)

noextract
let den_init (vx vw:S.seq BF.t) (t:nat) (v:F32.t)
  : Lemma (requires exp_cell vx vw t 0 v) (ensures den_ok vx vw t 1 v)
  = ()

noextract
let den_step (vx vw:S.seq BF.t) (t:nat) (k:pos) (d v:F32.t)
  : Lemma (requires den_ok vx vw t k d /\ exp_cell vx vw t k v)
          (ensures den_ok vx vw t (k + 1) (add d v))
  = let aux (rx rw:S.seq real)
      : Lemma (requires ins_ok vx vw rx rw)
              (ensures add d v %~ rden rx rw t (k + 1))
      = a_add d v (rden rx rw t k) (rex rx rw t k)
    in
    Classical.forall_intro_2 (fun rx rw -> Classical.move_requires (aux rx) rw)

noextract
let inv_step (vx vw:S.seq BF.t) (t:nat) (d:F32.t)
  : Lemma (requires den_ok vx vw t experts d)
          (ensures inv_ok vx vw t (div one d))
  = let aux (rx rw:S.seq real)
      : Lemma (requires ins_ok vx vw rx rw)
              (ensures (rden_pos rx rw t experts; div one d %~ (1.0R /. rden rx rw t experts)))
      = rden_pos rx rw t experts;
        let dr : (y:real{y =!= 0.0R}) = rden rx rw t experts in
        div_models_pat F32.t;
        approx_apply2_dom #_ #_ #_ #_ #_ #_ #(fun _ -> True) #(fun (y:real) -> y =!= 0.0R) (div #F32.t) r_div one d 1.0R dr
    in
    Classical.forall_intro_2 (fun rx rw -> Classical.move_requires (aux rx) rw)

(* ---------------- Real identity ---------------- *)


noextract
let div_add (a b e:real)
  : Lemma (requires e >. 0.0R) (ensures a /. e +. b /. e == (a +. b) /. e)
  = ()

noextract
let rec rsum_prefix (rx rw:S.seq real) (t:nat) (k:pos)
  : Lemma (requires t < tokens /\ k <= experts)
          (ensures (let ex = exp_row (row_of (ref_scores rx rw) t) in
                    S.length ex == experts /\
                    rden rx rw t k == rsum (S.slice ex 0 k) /. exp (rmaxrow rx rw t)))
          (decreases k)
  = let ex = exp_row (row_of (ref_scores rx rw) t) in
    row_index rx rw t 0;
    row_index rx rw t (k - 1);
    let em = exp (rmaxrow rx rw t) in
    if k = 1 then begin
      lemma_seq_fold_left_slice #real #real 0.0R (+.) ex 0 0;
      assert (S.equal (S.slice ex 0 0) S.empty);
      assert (rsum (S.slice ex 0 0) == 0.0R)
    end else begin
      rsum_prefix rx rw t (k - 1);
      lemma_seq_fold_left_slice #real #real 0.0R (+.) ex 0 (k - 1);
      let a = rsum (S.slice ex 0 (k - 1)) in
      let b = exp (rsc rx rw t (k - 1)) in
      assert (S.index ex (k - 1) == b);
      assert (rsum (S.slice ex 0 k) == a +. b);
      exp_sub (rsc rx rw t (k - 1)) (rmaxrow rx rw t);
      assert (rex rx rw t (k - 1) == b /. em);
      assert (rden rx rw t k == a /. em +. b /. em);
      div_add a b em
    end

noextract
let real_alg (a e p:real)
  : Lemma (requires e >. 0.0R /\ p >. 0.0R)
          (ensures (a /. e) *. (1.0R /. (p /. e)) == a /. p)
  = ()

noextract
let final_identity (rx rw:S.seq real) (t e:nat)
  : Lemma (requires t < tokens /\ e < experts)
          (ensures (rden_pos rx rw t experts;
                    rex rx rw t e *. (1.0R /. rden rx rw t experts) ==
                    S.index (ref_softmax (row_of (ref_scores rx rw) t)) e))
  = rden_pos rx rw t experts;
    row_index rx rw t e;
    rsum_prefix rx rw t experts;
    let row = row_of (ref_scores rx rw) t in
    let ex = exp_row row in
    assert (S.equal (S.slice ex 0 experts) ex);
    let p = rsum ex in
    let em = exp (rmaxrow rx rw t) in
    assert (rden rx rw t experts == p /. em);
    assert (p >. 0.0R);
    real_alg (exp (rsc rx rw t e)) em p

noextract
let final_step (vx vw:S.seq BF.t) (t:nat{t < tokens}) (e:nat{e < experts}) (v d:F32.t)
  : Lemma (requires exp_cell vx vw t e v /\ inv_ok vx vw t d)
          (ensures final_cell vx vw t e (mul v d))
  = let aux (rx rw:S.seq real)
      : Lemma (requires ins_ok vx vw rx rw)
              (ensures (row_index rx rw t e;
                        mul v d %~ S.index (ref_softmax (row_of (ref_scores rx rw) t)) e))
      = rden_pos rx rw t experts;
        final_identity rx rw t e;
        a_mul v d (rex rx rw t e) (1.0R /. rden rx rw t experts)
    in
    Classical.forall_intro_2 (fun rx rw -> Classical.move_requires (aux rx) rw)

noextract
let all_final (vx vw:S.seq BF.t) (s:S.seq F32.t) : prop =
  S.length s == tokens * experts /\
  (forall (t:nat{t < tokens}) (e:nat{e < experts}).
     final_cell vx vw t e (S.index s (experts * t + e)))

noextract
let rows_ok (vx vw:S.seq BF.t) (s:S.seq F32.t)
  : Lemma (requires all_final vx vw s) (ensures softmax_rows_ok vx vw s)
  = let aux (t:nat) (rx rw:S.seq real)
      : Lemma (requires t < tokens /\ ins_ok vx vw rx rw)
              (ensures row_of s t %~ ref_softmax (row_of (ref_scores rx rw) t))
      = let row = row_of s t in
        let rr = ref_softmax (row_of (ref_scores rx rw) t) in
        row_index rx rw t 0;
        assert (S.length row == experts);
        assert (S.length rr == experts);
        let cell (i:nat)
          : Lemma (requires i < experts /\ S.length row == experts /\ S.length rr == experts)
                  (ensures S.index row i %~ S.index rr i)
          = row_index rx rw t i;
            assert (S.index row i == S.index s (experts * t + i));
            assert (final_cell vx vw t i (S.index s (experts * t + i)))
        in
        introduce forall (i:nat). i < S.length row ==> S.index row i %~ S.index rr i
        with introduce _ ==> _ with cell i
    in
    let aux2 (t:nat)
      : Lemma (requires t < tokens) (ensures softmax_row_ok vx vw s t)
      = Classical.forall_intro_2 (fun rx rw -> Classical.move_requires (aux t rx) rw)
    in
    Classical.forall_intro (Classical.move_requires aux2)

(* ---------------- Row-phase invariants ---------------- *)

#push-options "--z3rlimit 30"

noextract
let at (s:S.seq F32.t) (t e:nat) : F32.t =
  if experts * t + e < S.length s then S.index s (experts * t + e) else zero

noextract
let at_upd (s:S.seq F32.t) (t e:nat) (v:F32.t) (t' e':nat)
  : Lemma (requires e < experts /\ e' < experts /\ experts * t + e < S.length s)
          (ensures at (S.upd s (experts * t + e) v) t' e' ==
                   (if t' = t && e' = e then v else at s t' e'))
  = ()

noextract
let done_rows (vx vw:S.seq BF.t) (s:S.seq F32.t) (t:nat{t <= tokens}) : prop =
  S.length s == tokens * experts /\
  (forall (t':nat) (e:nat). t' < t /\ e < experts ==> final_cell vx vw t' e (at s t' e))

noextract
let p1 (vx vw:S.seq BF.t) (s:S.seq F32.t) (t:nat{t < tokens}) (e:nat) : prop =
  done_rows vx vw s t /\
  (forall (e':nat). e' < e ==> score_cell vx vw t e' (at s t e'))

noextract
let p2 (vx vw:S.seq BF.t) (s:S.seq F32.t) (t:nat{t < tokens}) (e:nat) : prop =
  done_rows vx vw s t /\
  (forall (e':nat). e' < e ==> exp_cell vx vw t e' (at s t e')) /\
  (forall (e':nat). e <= e' /\ e' < experts ==> score_cell vx vw t e' (at s t e'))

noextract
let p3 (vx vw:S.seq BF.t) (s:S.seq F32.t) (t:nat{t < tokens}) (e:nat) : prop =
  done_rows vx vw s t /\
  (forall (e':nat). e' < e /\ e' < experts ==> final_cell vx vw t e' (at s t e')) /\
  (forall (e':nat). e <= e' /\ e' < experts ==> exp_cell vx vw t e' (at s t e'))

noextract
let done_rows_upd (vx vw:S.seq BF.t) (s:S.seq F32.t) (t:nat{t < tokens}) (e:nat{e < experts}) (v:F32.t)
  : Lemma (requires done_rows vx vw s t)
          (ensures done_rows vx vw (S.upd s (experts * t + e) v) t)
  = let s' = S.upd s (experts * t + e) v in
    introduce forall (t':nat) (e':nat). t' < t /\ e' < experts ==> final_cell vx vw t' e' (at s' t' e')
    with introduce _ ==> _ with at_upd s t e v t' e'

noextract
let p1_step (vx vw:S.seq BF.t) (s:S.seq F32.t) (t:nat{t < tokens}) (e:nat{e < experts}) (v:F32.t)
  : Lemma (requires p1 vx vw s t e /\ score_cell vx vw t e v)
          (ensures p1 vx vw (S.upd s (experts * t + e) v) t (e + 1))
  = let s' = S.upd s (experts * t + e) v in
    done_rows_upd vx vw s t e v;
    introduce forall (e':nat). e' < e + 1 ==> score_cell vx vw t e' (at s' t e')
    with introduce _ ==> _ with at_upd s t e v t e'

noextract
let p1_p2 (vx vw:S.seq BF.t) (s:S.seq F32.t) (t:nat{t < tokens})
  : Lemma (requires p1 vx vw s t experts) (ensures p2 vx vw s t 0)
  = ()

noextract
let p2_step (vx vw:S.seq BF.t) (s:S.seq F32.t) (t:nat{t < tokens}) (e:nat{e < experts}) (v:F32.t)
  : Lemma (requires p2 vx vw s t e /\ exp_cell vx vw t e v)
          (ensures p2 vx vw (S.upd s (experts * t + e) v) t (e + 1))
  = let s' = S.upd s (experts * t + e) v in
    done_rows_upd vx vw s t e v;
    introduce forall (e':nat). e' < e + 1 ==> exp_cell vx vw t e' (at s' t e')
    with introduce _ ==> _ with at_upd s t e v t e';
    introduce forall (e':nat). e + 1 <= e' /\ e' < experts ==> score_cell vx vw t e' (at s' t e')
    with introduce _ ==> _ with at_upd s t e v t e'

noextract
let p2_p3 (vx vw:S.seq BF.t) (s:S.seq F32.t) (t:nat{t < tokens})
  : Lemma (requires p2 vx vw s t experts) (ensures p3 vx vw s t 0)
  = ()

noextract
let p3_step (vx vw:S.seq BF.t) (s:S.seq F32.t) (t:nat{t < tokens}) (e:nat{e < experts}) (v:F32.t)
  : Lemma (requires p3 vx vw s t e /\ final_cell vx vw t e v)
          (ensures p3 vx vw (S.upd s (experts * t + e) v) t (e + 1))
  = let s' = S.upd s (experts * t + e) v in
    done_rows_upd vx vw s t e v;
    introduce forall (e':nat). e' < e + 1 /\ e' < experts ==> final_cell vx vw t e' (at s' t e')
    with introduce _ ==> _ with begin
      at_upd s t e v t e';
      if e' = e then assert (at s' t e' == v)
      else begin
        assert (at s' t e' == at s t e');
        assert (final_cell vx vw t e' (at s t e'))
      end
    end;
    introduce forall (e':nat). e + 1 <= e' /\ e' < experts ==> exp_cell vx vw t e' (at s' t e')
    with introduce _ ==> _ with at_upd s t e v t e'

noextract
let p3_done (vx vw:S.seq BF.t) (s:S.seq F32.t) (t:nat{t < tokens})
  : Lemma (requires p3 vx vw s t experts) (ensures done_rows vx vw s (t + 1))
  = ()

noextract
let done_final (vx vw:S.seq BF.t) (s:S.seq F32.t)
  : Lemma (requires done_rows vx vw s tokens) (ensures softmax_rows_ok vx vw s)
  = introduce forall (t:nat{t < tokens}) (e:nat{e < experts}).
      final_cell vx vw t e (S.index s (experts * t + e))
    with assert (at s t e == S.index s (experts * t + e));
    rows_ok vx vw s

#pop-options

(* ---------------- Kernel ---------------- *)

inline_for_extraction noextract
fn dot
  (x : larray bf16 x_len)
  (w : larray bf16 w_len)
  (#vx : erased (lseq bf16 x_len))
  (#vw : erased (lseq bf16 w_len))
  (t : sz{SZ.v t < tokens})
  (e : sz{SZ.v e < experts})
  preserves gpu ** x |-> vx ** w |-> vw
  returns r : f32
  ensures pure (score_cell vx vw (SZ.v t) (SZ.v e) r)
{
  let mut acc : f32 = zero;
  let mut h = 0sz;
  while (!h <^ 512sz)
    invariant exists* (hv : sz{SZ.v hv <= hidden}) (a : f32).
      h |-> hv ** acc |-> a ** pure (acc_ok vx vw (SZ.v t) (SZ.v e) (SZ.v hv) a)
    decreases (512sz - !h)
  {
    let hv = !h;
    let xv = x.(t *^ 512sz +^ hv);
    let wv = w.(hv *^ 8sz +^ e);
    let a = !acc;
    let r = Fast.fma_rn (Cast.cast_bf16_to_f32 xv) (Cast.cast_bf16_to_f32 wv) a;
    acc_step vx vw (SZ.v t) (SZ.v e) (SZ.v hv) a r;
    acc := r;
    h := hv +^ 1sz;
  };
  !acc
}

#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn softmax_row
  (x : larray bf16 x_len)
  (w : larray bf16 w_len)
  (sm : larray f32 sm_len)
  (#vx : erased (lseq bf16 x_len))
  (#vw : erased (lseq bf16 w_len))
  (#s0 : erased (seq f32))
  (t : sz{SZ.v t < tokens})
  preserves gpu ** x |-> vx ** w |-> vw
  requires sm |-> s0 ** pure (done_rows vx vw s0 (SZ.v t))
  ensures exists* (s1 : seq f32). sm |-> s1 ** pure (done_rows vx vw s1 (SZ.v t + 1))
{
  let base = t *^ 8sz;
  (* scores *)
  let mut e = 0sz;
  while (!e <^ 8sz)
    invariant exists* (ev : sz{SZ.v ev <= experts}) (s : seq f32).
      e |-> ev ** sm |-> s ** pure (p1 vx vw s (SZ.v t) (SZ.v ev))
    decreases (8sz - !e)
  {
    let ev = !e;
    let v = dot x w #vx #vw t ev;
    with s. assert sm |-> s;
    assert pure (SZ.v (base +^ ev) == experts * SZ.v t + SZ.v ev);
    sm.(base +^ ev) <- v;
    p1_step vx vw s (SZ.v t) (SZ.v ev) v;
    e := ev +^ 1sz;
  };
  with s1. assert sm |-> s1;
  p1_p2 vx vw s1 (SZ.v t);
  (* maximum *)
  let m0 = sm.(base);
  max_init vx vw (SZ.v t) m0;
  let mut mx = m0;
  e := 1sz;
  while (!e <^ 8sz)
    invariant exists* (ev : sz{1 <= SZ.v ev /\ SZ.v ev <= experts}) (m : f32).
      e |-> ev ** mx |-> m ** sm |-> s1 ** pure (max_ok vx vw (SZ.v t) (SZ.v ev) m)
    decreases (8sz - !e)
  {
    let ev = !e;
    let v = sm.(base +^ ev);
    let m = !mx;
    max_step vx vw (SZ.v t) (SZ.v ev) m v;
    mx := fmax m v;
    e := ev +^ 1sz;
  };
  let m = !mx;
  (* exponentials *)
  e := 0sz;
  while (!e <^ 8sz)
    invariant exists* (ev : sz{SZ.v ev <= experts}) (s : seq f32).
      e |-> ev ** sm |-> s ** pure (p2 vx vw s (SZ.v t) (SZ.v ev))
    decreases (8sz - !e)
  {
    let ev = !e;
    with s. assert sm |-> s;
    let v = sm.(base +^ ev);
    let d = Fast.sub_rn v m;
    exp_step vx vw (SZ.v t) (SZ.v ev) m v d;
    let ex = CE.cpu_exp_f32 d;
    assert pure (SZ.v (base +^ ev) == experts * SZ.v t + SZ.v ev);
    sm.(base +^ ev) <- ex;
    p2_step vx vw s (SZ.v t) (SZ.v ev) ex;
    e := ev +^ 1sz;
  };
  with s2. assert sm |-> s2;
  p2_p3 vx vw s2 (SZ.v t);
  (* denominator *)
  let d0 = sm.(base);
  den_init vx vw (SZ.v t) d0;
  let mut den = d0;
  e := 1sz;
  while (!e <^ 8sz)
    invariant exists* (ev : sz{1 <= SZ.v ev /\ SZ.v ev <= experts}) (d : f32).
      e |-> ev ** den |-> d ** sm |-> s2 ** pure (den_ok vx vw (SZ.v t) (SZ.v ev) d)
    decreases (8sz - !e)
  {
    let ev = !e;
    let v = sm.(base +^ ev);
    let d = !den;
    den_step vx vw (SZ.v t) (SZ.v ev) d v;
    den := add d v;
    e := ev +^ 1sz;
  };
  let dn = !den;
  let inv = div one dn;
  inv_step vx vw (SZ.v t) dn;
  (* normalize *)
  e := 0sz;
  while (!e <^ 8sz)
    invariant exists* (ev : sz{SZ.v ev <= experts}) (s : seq f32).
      e |-> ev ** sm |-> s ** pure (p3 vx vw s (SZ.v t) (SZ.v ev))
    decreases (8sz - !e)
  {
    let ev = !e;
    with s. assert sm |-> s;
    let v = sm.(base +^ ev);
    final_step vx vw (SZ.v t) (SZ.v ev) v inv;
    assert pure (SZ.v (base +^ ev) == experts * SZ.v t + SZ.v ev);
    sm.(base +^ ev) <- mul v inv;
    p3_step vx vw s (SZ.v t) (SZ.v ev) (mul v inv);
    e := ev +^ 1sz;
  };
  with s3. assert sm |-> s3;
  assert pure (p3 vx vw s3 (SZ.v t) experts);
  p3_done vx vw s3 (SZ.v t);
}
#pop-options

inline_for_extraction noextract
fn init_state
  (prefix : larray i64 prefix_len)
  (idx : larray i64 slot_len)
  (probs : larray f32 slot_len)
  (#p0 : erased (seq i64))
  (#i0 : erased (seq i64))
  (#q0 : erased (seq f32))
  preserves gpu
  requires prefix |-> p0 ** idx |-> i0 ** probs |-> q0
  ensures exists* (p : seq i64) (i : seq i64) (q : seq f32).
    prefix |-> p ** idx |-> i ** probs |-> q ** pure (initial_routing_state p i q)
{
  pts_to_len prefix;
  pts_to_len idx;
  pts_to_len probs;
  let mut k = 0sz;
  while (!k <^ 256sz)
    invariant exists* (kv : sz{SZ.v kv <= slots}) (i : seq i64) (q : seq f32).
      k |-> kv ** idx |-> i ** probs |-> q **
      pure (S.length i == slots /\ S.length q == slots /\
            (forall (j:nat). j < SZ.v kv ==> I64.v (S.index i j) == -1 /\ S.index q j == zero))
    decreases (256sz - !k)
  {
    let kv = !k;
    idx.(kv) <- (-1L);
    probs.(kv) <- zero;
    k := kv +^ 1sz;
  };
  let mut e = 0sz;
  while (!e <^ 8sz)
    invariant exists* (ev : sz{SZ.v ev <= experts}) (p : seq i64).
      e |-> ev ** prefix |-> p **
      pure (S.length p == experts /\
            (forall (j:nat). j < SZ.v ev ==> I64.v (S.index p j) == 0))
    decreases (8sz - !e)
  {
    let ev = !e;
    prefix.(ev) <- 0L;
    e := ev +^ 1sz;
  };
}

inline_for_extraction noextract
fn kernel
  (x : larray bf16 x_len)
  (w : larray bf16 w_len)
  (sm : larray f32 sm_len)
  (prefix : larray i64 prefix_len)
  (idx : larray i64 slot_len)
  (probs : larray f32 slot_len)
  (#vx : erased (lseq bf16 x_len))
  (#vw : erased (lseq bf16 w_len))
  (#s0 : erased (seq f32))
  (#p0 : erased (seq i64))
  (#i0 : erased (seq i64))
  (#q0 : erased (seq f32))
  preserves gpu ** x |-> vx ** w |-> vw
  requires sm |-> s0 ** prefix |-> p0 ** idx |-> i0 ** probs |-> q0
  ensures exists* (s : seq f32) (p : seq i64) (i : seq i64) (q : seq f32).
    sm |-> s ** prefix |-> p ** idx |-> i ** probs |-> q **
    pure (base_projection_softmax_relation vx vw s p i q S.empty)
{
  pts_to_len sm;
  let mut t = 0sz;
  while (!t <^ 256sz)
    invariant exists* (tv : sz{SZ.v tv <= tokens}) (s : seq f32).
      t |-> tv ** sm |-> s ** pure (done_rows vx vw s (SZ.v tv))
    decreases (256sz - !t)
  {
    let tv = !t;
    softmax_row x w sm #vx #vw tv;
    t := tv +^ 1sz;
  };
  with s. assert sm |-> s;
  done_final vx vw s;
  init_state prefix idx probs;
}

(* The generic instance targets visibility_of arr; launch_kernel_1 needs gpu_of. *)
instance send_global_array_contents
  (#a : Type0)
  (arr : array a{is_global arr})
  (#f : perm)
  (contents : seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn kplay_run
  (x : larray bf16 x_len{is_global x})
  (w : larray bf16 w_len{is_global w})
  (softmax : larray f32 sm_len{is_global softmax})
  (prefix : larray i64 prefix_len{is_global prefix})
  (token_indices : larray i64 slot_len{is_global token_indices})
  (probabilities : larray f32 slot_len{is_global probabilities})
  (survival : larray i32 surv_len{is_global survival})
  (#vx : erased (lseq bf16 x_len))
  (#vw : erased (lseq bf16 w_len))
  (#vsurv : erased (lseq i32 surv_len))
  preserves cpu ** on gpu_loc (x |-> vx) ** on gpu_loc (w |-> vw) **
            on gpu_loc (survival |-> vsurv)
  requires exists* (s0 : seq f32) (p0 : seq i64) (i0 : seq i64) (q0 : seq f32).
    on gpu_loc (softmax |-> s0) ** on gpu_loc (prefix |-> p0) **
    on gpu_loc (token_indices |-> i0) ** on gpu_loc (probabilities |-> q0)
  ensures exists* (s : seq f32) (p : seq i64) (i : seq i64) (q : seq f32).
    on gpu_loc (softmax |-> s) ** on gpu_loc (prefix |-> p) **
    on gpu_loc (token_indices |-> i) ** on gpu_loc (probabilities |-> q) **
    pure (base_projection_softmax_relation vx vw s p i q vsurv)
{
  with s0. assert on gpu_loc (softmax |-> s0);
  with p0. assert on gpu_loc (prefix |-> p0);
  with i0. assert on gpu_loc (token_indices |-> i0);
  with q0. assert on gpu_loc (probabilities |-> q0);
  launch_kernel_1 (fun () ->
    kernel x w softmax prefix token_indices probabilities #vx #vw #s0 #p0 #i0 #q0);
  with s p i q. assert
    on gpu_loc (softmax |-> s) ** on gpu_loc (prefix |-> p) **
    on gpu_loc (token_indices |-> i) ** on gpu_loc (probabilities |-> q);
  assert pure (S.equal (reveal vsurv) S.empty);
}
