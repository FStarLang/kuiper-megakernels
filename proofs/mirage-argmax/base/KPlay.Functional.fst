module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = FStar.SizeT
module I64 = FStar.Int64
module FB = Kuiper.Floating.Base
module FL = Kuiper.Floating
module F32 = Kuiper.Float32
module C = Kuiper.Float.Casts
module SS = KPlay.SharedSpec

(* ------------------------------------------------------------------ *)
(* Order facts: BF16 comparisons agree with the comparisons of the     *)
(* exact f32 widening (the kernel compares float(logits[i])).          *)

inline_for_extraction noextract
let widen (x : bf16) : f32 = C.fcast #bf16 #f32 x

let widen_kind (x : bf16)
  : Lemma (FB.kind (widen x) == FB.kind x)
  = C.bf16_to_f32_kind x

let f32_nan_test (x : f32)
  : Lemma (FB.eq x x <==> ~(FB.NaN? (FB.kind x)))
  = FL.ieee_eq_refl x

let bf16_lte_refl (x : bf16)
  : Lemma (requires ~(SS.is_nan x)) (ensures FB.lte x x)
  = FL.ieee_eq_refl x

let lt_widen (x y : bf16)
  : Lemma (requires ~(SS.is_nan x) /\ ~(SS.is_nan y))
          (ensures FB.lt x y <==> FB.lt (widen x) (widen y))
  = widen_kind x; widen_kind y;
    C.bf16_to_f32_lte y x;
    FB.negate_lt_is_lte x y;
    FB.negate_lt_is_lte (widen x) (widen y)

(* On non-NaN f32: x <= y < z ==> x < z, from transitivity of <, the
   lt/lte duality and the bit-level definition of IEEE equality. *)
let f32_lte_lt (x y z : f32)
  : Lemma (requires ~(FB.NaN? (FB.kind x)) /\ ~(FB.NaN? (FB.kind y)) /\ ~(FB.NaN? (FB.kind z)) /\
                    FB.lte x y /\ FB.lt y z)
          (ensures FB.lt x z)
  = FB.lte_is_lt_or_eq x y;
    if FB.lt x y then F32.lt_transitive x y z
    else begin
      FB.negate_lt_is_lte x z;
      FB.lte_is_lt_or_eq z x;
      if FB.lt x z then ()
      else if FB.lt z x then begin
        F32.lt_transitive y z x;
        FB.negate_lt_is_lte y x
      end else begin
        FB.eq_spec x y; FB.eq_spec z x; FB.eq_spec z y;
        FB.bit_eq_spec x y; FB.bit_eq_spec z x; FB.bit_eq_spec z y;
        FB.bits_spec x y; FB.bits_spec z x; FB.bits_spec z y;
        assert (FB.eq z y);
        FB.lte_is_lt_or_eq z y;
        FB.negate_lt_is_lte y z
      end
    end

let bf16_lte_lt (x y z : bf16)
  : Lemma (requires ~(SS.is_nan x) /\ ~(SS.is_nan y) /\ ~(SS.is_nan z) /\
                    FB.lte x y /\ FB.lt y z)
          (ensures FB.lt x z)
  = widen_kind x; widen_kind y; widen_kind z;
    C.bf16_to_f32_lte x y;
    lt_widen y z;
    f32_lte_lt (widen x) (widen y) (widen z);
    lt_widen x z

(* ------------------------------------------------------------------ *)
(* Serial select_token loop invariant                                  *)

let nan_before (s : S.seq bf16) (c : nat) : prop =
  exists (i:nat). i < c /\ i < S.length s /\ SS.is_nan (S.index s i)

(* After visiting columns [0, c), j is the first maximum (or first NaN) of the prefix. *)
let inv (s : S.seq bf16) (c : nat) (j : nat) : prop =
  j < c /\ c <= S.length s /\
  (nan_before s c ==>
     SS.is_nan (S.index s j) /\ (forall (i:nat). i < j ==> ~(SS.is_nan (S.index s i)))) /\
  (~(nan_before s c) ==>
     (forall (i:nat). i < c ==> FB.lte (S.index s i) (S.index s j)) /\
     (forall (i:nat). i < j ==> FB.lt (S.index s i) (S.index s j)))

(* The kernel's update test: value > best || (isnan(value) && !isnan(best)). *)
inline_for_extraction noextract
let better (fb fv : f32) : bool =
  FB.lt fb fv || (not (FB.eq fv fv) && FB.eq fb fb)

let inv_init (s : S.seq bf16)
  : Lemma (requires 0 < S.length s) (ensures inv s 1 0)
  = if SS.is_nan (S.index s 0) then assert (nan_before s 1)
    else bf16_lte_refl (S.index s 0)

let inv_step (s : S.seq bf16) (c j : nat)
  : Lemma (requires inv s c j /\ c < S.length s)
          (ensures inv s (c + 1)
                     (if better (widen (S.index s j)) (widen (S.index s c)) then c else j))
  = let b = S.index s j in
    let v = S.index s c in
    widen_kind b; widen_kind v;
    f32_nan_test (widen b); f32_nan_test (widen v);
    if FB.lt (widen b) (widen v) then F32.lt_ordered (widen b) (widen v);
    if SS.is_nan v then assert (nan_before s (c + 1));
    if nan_before s c then begin
      assert (nan_before s (c + 1))
    end else if SS.is_nan v then ()
    else begin
      assert (~(nan_before s (c + 1)));
      lt_widen b v;
      if FB.lt b v then begin
        bf16_lte_refl v;
        let aux (i:nat{i < c}) : Lemma (FB.lt (S.index s i) v) =
          bf16_lte_lt (S.index s i) b v in
        Classical.forall_intro aux
      end else
        FB.negate_lt_is_lte b v
    end

let inv_final (s : S.seq bf16) (j : nat)
  : Lemma (requires inv s (S.length s) j) (ensures first_argmax s j)
  = assert (nan_before s (S.length s) <==> has_nan s)

(* ------------------------------------------------------------------ *)
(* Kernel and host entry point                                         *)

instance send_global_array_contents
  (#a : Type0)
  (arr : array a{is_global arr})
  (#f : perm)
  (contents : seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

inline_for_extraction noextract
fn select_token
  (n0 : sz)
  (n1 : sz)
  (input0 : larray i64 (SZ.v n0))
  (input1 : larray bf16 (SZ.v n1))
  (input2 : i64)
  (output0 : larray i64 (SZ.v n0 + 1))
  (output1 : larray i64 1)
  (#p0 #p1 : perm)
  (#s0 : erased (seq i64))
  (#s1 : erased (seq bf16))
  (#v0 #v1 : erased (seq i64))
  preserves
    gpu ** input0 |-> Frac p0 s0 ** input1 |-> Frac p1 s1
  requires
    output0 |-> v0 ** output1 |-> v1
  ensures
    exists* (r0 r1 : seq i64).
      output0 |-> r0 ** output1 |-> r1 **
      pure (base_argmax_relation s0 s1 (I64.v input2) r0 r1)
{
  pts_to_len input0;
  pts_to_len input1;
  pts_to_len output0;
  pts_to_len output1;
  (* for (i = 0; i < prompt_length; ++i) output[i] = prompt[i]; *)
  let mut i : sz = 0sz;
  while (!i <^ n0)
    invariant exists* (vi : sz) (r0 : seq i64).
      i |-> vi ** output0 |-> r0 **
      pure (SZ.v vi <= SZ.v n0 /\ S.length r0 == SZ.v n0 + 1 /\
            (forall (k:nat). k < SZ.v vi ==> S.index r0 k == S.index s0 k))
    decreases (SZ.v n0 - SZ.v !i)
  {
    let iv = !i;
    let x = input0.(iv);
    output0.(iv) <- x;
    i := iv +^ 1sz;
  };
  if (0sz <^ n1) {
    let first = input1.(0sz);
    inv_init s1;
    let mut best : f32 = widen first;
    let mut idx : sz = 0sz;
    let mut col : sz = 1sz;
    while (!col <^ n1)
      invariant exists* (c : sz) (j : sz) (b : f32).
        col |-> c ** idx |-> j ** best |-> b **
        pure (SZ.v j < SZ.v n1 /\ inv s1 (SZ.v c) (SZ.v j) /\
              b == widen (S.index s1 (SZ.v j)))
      decreases (SZ.v n1 - SZ.v !col)
    {
      let c = !col;
      let j = !idx;
      let v = input1.(c);
      let fv = widen v;
      let fb = !best;
      inv_step s1 (SZ.v c) (SZ.v j);
      if (better fb fv) {
        best := fv;
        idx := c;
      };
      col := c +^ 1sz;
    };
    let j = !idx;
    inv_final s1 (SZ.v j);
    let r = FStar.Int.Cast.uint32_to_int64 (sizet_to_u32 j);
    output0.(n0) <- r;
    output1.(0sz) <- r;
  } else {
    output0.(n0) <- 0L;
    output1.(0sz) <- 0L;
  }
}

fn kplay_run
  (n0 : sz)
  (n1 : sz)
  (input0 : larray i64 (SZ.v n0) { is_global input0 })
  (input1 : larray bf16 (SZ.v n1) { is_global input1 })
  (input2 : i64)
  (output0 : larray i64 (SZ.v n0 + 1) { is_global output0 })
  (output1 : larray i64 1 { is_global output1 })
  (#p0 #p1 : perm)
  (#s0 : erased (seq i64))
  (#s1 : erased (seq bf16))
  preserves
    cpu **
    on gpu_loc (input0 |-> Frac p0 s0) **
    on gpu_loc (input1 |-> Frac p1 s1)
  requires
    exists* (v0 v1 : seq i64).
      on gpu_loc (output0 |-> v0) ** on gpu_loc (output1 |-> v1)
  ensures
    exists* (r0 r1 : seq i64).
      on gpu_loc (output0 |-> r0) ** on gpu_loc (output1 |-> r1) **
      pure (base_argmax_relation s0 s1 (I64.v input2) r0 r1)
{
  with v0 v1. assert (on gpu_loc (output0 |-> v0) ** on gpu_loc (output1 |-> v1));
  launch_kernel_1
    (fun () ->
      select_token n0 n1 input0 input1 input2 output0 output1
        #p0 #p1 #s0 #s1 #v0 #v1);
}
