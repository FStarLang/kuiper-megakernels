module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Float.Casts
module S = FStar.Seq
module I64 = FStar.Int64
module SZ = FStar.SizeT
module SS = KPlay.SharedSpec
module F = Kuiper.Floating
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32

(*** Native operations of select_token *)

(* float(logits[i]): exact BF16 -> F32 widening. *)
inline_for_extraction noextract
let w (x:BF.t) : F32.t = c_bf16_32.fcast x

(* isnan(x), computed as x != x. *)
inline_for_extraction noextract
let kp_isnan (x:F32.t) : bool = not (eq x x)

(* The update test: !isnan(value) && (isnan(next) || next > value). *)
inline_for_extraction noextract
let kp_take (v nx:F32.t) : bool = not (kp_isnan v) && (kp_isnan nx || lt v nx)

noextract
let row_t = r:S.seq BF.t{S.length r == 248320}

(* Ghost model of the serial scan over indices i..248319. *)
noextract
let rec scan (row:row_t) (i:nat) (v:F32.t) (b:nat{b < 248320})
  : Tot (r:nat{r < 248320}) (decreases (if i < 248320 then 248320 - i else 0))
  = if i >= 248320 then b
    else let nx = w (S.index row i) in
         if kp_take v nx then scan row (i + 1) nx i else scan row (i + 1) v b

noextract
let scan_step (row:row_t) (i:nat{i < 248320}) (v:F32.t) (b:nat{b < 248320})
  : Lemma (scan row i v b ==
           (if kp_take v (w (S.index row i)) then scan row (i + 1) (w (S.index row i)) i
            else scan row (i + 1) v b))
  = ()

(* Destination test, guarded by the domain ranges so int64 arithmetic cannot overflow. *)
noextract
let kp_active (p s t:int) : bool =
  1 <= p && p <= 12 && 1 <= s && s <= 12 && 0 <= t && t < 12 &&
  p <= t + 1 && t + 1 < s + p

noextract
let kp_best (row:row_t) : r:nat{r < 248320} = scan row 1 (w (S.index row 0)) 0

noextract
let kp_out (row:row_t) (p s t:int) : I64.t =
  if kp_active p s t then I64.int_to_t (kp_best row) else 0L

inline_for_extraction noextract
let kp_active_exec (p s tt:I64.t)
  : r:bool{r == kp_active (I64.v p) (I64.v s) (I64.v tt)}
  = let open FStar.Int64 in
    if p >=^ 1L && p <=^ 12L && s >=^ 1L && s <=^ 12L && tt >=^ 0L && tt <^ 12L
    then p <=^ (tt +^ 1L) && (tt +^ 1L) <^ (s +^ p)
    else false

(*** Order facts through the widening *)

noextract
let w_nan (x:BF.t)
  : Lemma (kp_isnan (w x) <==> SS.is_nan x)
  = bf16_to_f32_kind x; F.ieee_eq_refl (w x)

noextract
let w_lt (x y:BF.t)
  : Lemma (requires ~(SS.is_nan x) /\ ~(SS.is_nan y))
          (ensures lt (w x) (w y) <==> lt x y)
  = bf16_to_f32_kind x; bf16_to_f32_kind y; bf16_to_f32_lte y x;
    F.negate_lt_is_lte (w x) (w y); F.negate_lt_is_lte x y

noextract
let lt_irrefl (x:BF.t)
  : Lemma (requires ~(SS.is_nan x)) (ensures ~(lt x x))
  = F.ieee_eq_refl x; F.lte_is_lt_or_eq x x; F.negate_lt_is_lte x x

noextract
let bf_lt_trans (x y z:BF.t)
  : Lemma (requires ~(SS.is_nan x) /\ ~(SS.is_nan y) /\ ~(SS.is_nan z) /\ lt x y /\ lt y z)
          (ensures lt x z)
  = w_lt x y; w_lt y z; w_lt x z; F32.lt_transitive (w x) (w y) (w z)

(* IEEE equality is transitive on non-NaN values. *)
noextract
let f32_eq_trans (a b c:F32.t)
  : Lemma (requires ~(SS.is_nan a) /\ ~(SS.is_nan b) /\ ~(SS.is_nan c) /\ eq a b /\ eq b c)
          (ensures eq a c)
  = F.eq_spec a b; F.eq_spec b c; F.eq_spec a c;
    F.bit_eq_spec a b; F.bit_eq_spec b c; F.bit_eq_spec a c;
    if F.bit_eq a b then F.bits_spec a b;
    if F.bit_eq b c then F.bits_spec b c

(* not (b < a) and b < c imply a < c (non-NaN). *)
noextract
let f32_mix (a b c:F32.t)
  : Lemma (requires ~(SS.is_nan a) /\ ~(SS.is_nan b) /\ ~(SS.is_nan c) /\ ~(lt b a) /\ lt b c)
          (ensures lt a c)
  = F.negate_lt_is_lte a c; F.negate_lt_is_lte b a; F.negate_lt_is_lte b c;
    F.lte_is_lt_or_eq c a; F.lte_is_lt_or_eq a b; F.lte_is_lt_or_eq c b;
    if lt a c then ()
    else begin
      (* c <= a *)
      if lt c a then F32.lt_transitive b c a
      else begin
        (* c == a; a <= b *)
        if lt a b then F32.lt_transitive a b c
        else f32_eq_trans c a b
      end
    end

noextract
let bf_mix (a b c:BF.t)
  : Lemma (requires ~(SS.is_nan a) /\ ~(SS.is_nan b) /\ ~(SS.is_nan c) /\ ~(lt b a) /\ lt b c)
          (ensures lt a c)
  = w_lt b a; w_lt b c; w_lt a c;
    bf16_to_f32_kind a; bf16_to_f32_kind b; bf16_to_f32_kind c;
    f32_mix (w a) (w b) (w c)

(*** Scan invariant: first NaN, else first maximum, over the consumed row *)

noextract
let nan_at (row:row_t) (k:nat{k < 248320}) : prop = SS.is_nan (S.index row k)

(* row[b] is the first NaN *)
noextract
let case_nan (row:row_t) (b:nat{b < 248320}) : prop =
  nan_at row b /\ (forall (k:nat). k < b ==> ~(nan_at row k))

(* no NaN in [0,i); row[b] is maximal there and strictly above every earlier entry *)
noextract
let case_max (row:row_t) (i:nat{i <= 248320}) (b:nat{b < 248320}) : prop =
  (forall (k:nat). k < i ==> ~(nan_at row k)) /\
  (forall (k:nat). k < i ==> ~(lt (S.index row b) (S.index row k))) /\
  (forall (k:nat). k < b ==> lt (S.index row k) (S.index row b))

noextract
let inv (row:row_t) (i:nat) (b:nat{b < 248320}) : prop =
  b < i /\ i <= 248320 /\ (case_nan row b \/ case_max row i b)

noextract
let inv_init (row:row_t)
  : Lemma (inv row 1 0)
  = if F.NaN? (F.kind (S.index row 0)) then ()
    else lt_irrefl (S.index row 0)

noextract
let step_take_max (row:row_t) (i:nat{i < 248320}) (b:nat{b < i})
  : Lemma (requires case_max row i b /\ ~(nan_at row i) /\ lt (S.index row b) (S.index row i))
          (ensures case_max row (i + 1) i)
  = let y = S.index row i in
    lt_irrefl y;
    let aux (k:nat)
      : Lemma (requires k < i)
              (ensures k < 248320 /\ lt (S.index row k) y /\ ~(lt y (S.index row k)))
      = let x = S.index row k in
        bf_mix x (S.index row b) y;
        if lt y x then (bf_lt_trans y x y)
    in
    Classical.forall_intro (Classical.move_requires aux)

noextract
let step (row:row_t) (i:nat{i < 248320}) (v:F32.t) (b:nat{b < 248320})
  : Lemma (requires inv row i b /\ v == w (S.index row b))
          (ensures (let nx = w (S.index row i) in
                    if kp_take v nx then inv row (i + 1) i else inv row (i + 1) b))
  = let x = S.index row b in
    let y = S.index row i in
    w_nan x; w_nan y;
    if F.NaN? (F.kind x) then ()
    else begin
      assert (case_max row i b);
      if F.NaN? (F.kind y) then ()
      else begin
        w_lt x y;
        if lt x y then step_take_max row i b
        else ()
      end
    end

noextract
let rec scan_inv (row:row_t) (i:nat) (v:F32.t) (b:nat{b < 248320})
  : Lemma (requires inv row i b /\ v == w (S.index row b))
          (ensures inv row 248320 (scan row i v b))
          (decreases (if i < 248320 then 248320 - i else 0))
  = if i >= 248320 then ()
    else begin
      step row i v b;
      let nx = w (S.index row i) in
      if kp_take v nx then scan_inv row (i + 1) nx i else scan_inv row (i + 1) v b
    end

noextract
let inv_final (row:row_t) (b:nat{b < 248320})
  : Lemma (requires inv row 248320 b) (ensures SS.first_allowed_choice row b)
  = if F.NaN? (F.kind (S.index row b)) then begin
      assert (case_nan row b);
      assert (SS.allowed_choice row b)
    end else begin
      assert (case_max row 248320 b);
      assert (SS.allowed_choice row b)
    end

noextract
let kp_best_ok (row:row_t)
  : Lemma (SS.first_allowed_choice row (kp_best row))
  = inv_init row;
    scan_inv row 1 (w (S.index row 0)) 0;
    inv_final row (kp_best row)

noextract
let kp_out_ok (row:row_t) (p s t:int) (sel:S.seq I64.t)
  : Lemma (requires S.length sel == 1 /\ S.index sel 0 == kp_out row p s t)
          (ensures SS.base_argmax_relation row p s t sel)
  = kp_best_ok row

(*** Kernel *)

noextract
let row_of (s:chest1 bf16 kplay_vocab) : GTot row_t = chest1_to_seq s

noextract
let row_of_index (s:chest1 bf16 kplay_vocab) (i:nat{i < 248320})
  : Lemma (S.index (row_of s) i == acc s (i, ()))
  = ()

noextract
let out_upd (c:chest1 i64 kplay_one) (j:abs (kplay_one @| INil)) (v:I64.t)
  : Lemma (requires j == (0, ()))
          (ensures S.index (chest1_to_seq (upd c j v)) 0 == v)
  = ()

inline_for_extraction noextract
fn kp_scan
  (input0 : array1 bf16 (l1_forward kplay_vocab))
  (#pm : perm)
  (#s : chest1 bf16 kplay_vocab)
  preserves
    input0 |-> Frac pm s
  returns
    r : I64.t
  ensures
    pure (I64.v r == kp_best (row_of s))
{
  let x0 = w (Kuiper.Tensor.read input0 ((0sz <: szlt kplay_vocab), ()) #pm);
  row_of_index s 0;
  let mut value : F32.t = x0;
  let mut best : I64.t = 0L;
  let mut i : szle kplay_vocab = 1sz;
  let mut ii : I64.t = 1L;
  while (!i <^ kplay_vocab)
    invariant exists* (vi : szle kplay_vocab) (vii : I64.t) (vv : F32.t) (vb : I64.t).
      (i |-> vi) ** (ii |-> vii) ** (value |-> vv) ** (best |-> vb) **
      pure (1 <= SZ.v vi /\ I64.v vii == SZ.v vi /\ 0 <= I64.v vb /\ I64.v vb < SZ.v vi /\
            scan (row_of s) (SZ.v vi) vv (I64.v vb) == kp_best (row_of s))
    decreases (SZ.v kplay_vocab - SZ.v !i)
  {
    let iv = !i;
    let nx = w (Kuiper.Tensor.read input0 ((iv <: szlt kplay_vocab), ()) #pm);
    row_of_index s (SZ.v iv);
    let v = !value;
    scan_step (row_of s) (SZ.v iv) v (I64.v !best);
    if (kp_take v nx) {
      value := nx;
      best := !ii;
    };
    i := iv +^ 1sz;
    ii := FStar.Int64.add !ii 1L;
  };
  !best
}

inline_for_extraction noextract
fn kf
  (input0 : array1 bf16 (l1_forward kplay_vocab) { is_global input0 })
  (input1 : I64.t)
  (input2 : I64.t)
  (input3 : I64.t)
  (output0 : array1 i64 (l1_forward kplay_one) { is_global output0 })
  (#input0_perm : perm)
  (#input0_state : chest1 bf16 kplay_vocab)
  (#output0_state : chest1 i64 kplay_one)
  preserves
    gpu **
    input0 |-> Frac input0_perm input0_state
  requires
    output0 |-> output0_state
  ensures
    exists* (s' : chest1 i64 kplay_one).
      (output0 |-> s') **
      pure (S.index (chest1_to_seq s') 0 ==
            kp_out (row_of input0_state) (I64.v input1) (I64.v input2) (I64.v input3))
{
  let act = kp_active_exec input1 input2 input3;
  let res : I64.t =
    if act { kp_scan input0 #input0_perm #input0_state } else { 0L };
  Kuiper.Tensor.write output0 ((0sz <: szlt kplay_one), ()) res;
  assert (pure (up #1 #(kplay_one @| INil) ((0sz <: szlt kplay_one), ()) == (0, ())));
  out_upd output0_state (up #1 #(kplay_one @| INil) ((0sz <: szlt kplay_one), ())) res;
}

fn kplay_run
  (input0 : array1 bf16 (l1_forward kplay_vocab) { is_global input0 })
  (input1 : I64.t)
  (input2 : I64.t)
  (input3 : I64.t)
  (output0 : array1 i64 (l1_forward kplay_one) { is_global output0 })
  (#input0_perm : perm)
  (#input0_state : chest1 bf16 kplay_vocab)
  (#output0_state : chest1 i64 kplay_one)
  preserves
    cpu **
    on gpu_loc (input0 |-> Frac input0_perm input0_state)
  requires
    on gpu_loc (output0 |-> output0_state)
  ensures
    exists* (output0_state' : chest1 i64 kplay_one).
      on gpu_loc (output0 |-> output0_state') **
      pure (SS.base_argmax_relation (chest1_to_seq input0_state)
              (I64.v input1) (I64.v input2) (I64.v input3)
              (chest1_to_seq output0_state'))
{
  launch_kernel_1
    (fun () ->
      kf input0 input1 input2 input3 output0
        #input0_perm #input0_state #output0_state);
  with s'. assert (on gpu_loc ((output0 |-> s') **
    pure (S.index (chest1_to_seq s') 0 ==
          kp_out (row_of input0_state) (I64.v input1) (I64.v input2) (I64.v input3))));
  assert (pure (S.length (chest1_to_seq s') == 1));
  assert (pure (S.index (chest1_to_seq s') 0 ==
          kp_out (row_of input0_state) (I64.v input1) (I64.v input2) (I64.v input3)));
  kp_out_ok (row_of input0_state) (I64.v input1) (I64.v input2) (I64.v input3)
    (chest1_to_seq s');
  assert (pure (SS.base_argmax_relation (chest1_to_seq input0_state)
              (I64.v input1) (I64.v input2) (I64.v input3) (chest1_to_seq s')));
}
