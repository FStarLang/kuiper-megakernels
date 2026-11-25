module Kuiper.KPlay.SwiGLU

open Kuiper
open Kuiper.Chest

module Cast = Kuiper.Float.Casts.Base
module MS = Kuiper.Spec.GEMM

(* Abstract real refinement, not IEEE bit equivalence or an error bound.
   The existing Float16/Float32 real_like and floating_real_like instances,
   and Float.Casts.Base conversion/refinement primitives, remain trusted.
   No tensor-core, direct-MMA, split-K, or CUDA/library equivalence is supplied
   here. Each matrix lemma requires BOTH projection refinements from its caller.

   The concrete programs below deliberately have different cast boundaries.
   All use division of the gate by the denominator, NOT gate times a rounded
   reciprocal. They are not asserted to be physically equal to each other or to
   a reciprocal-multiply program. In particular, these
   ordinary floating operations do not select a fast/FTZ extraction bridge. *)

(* Exact-real SiLU, defined locally so this module needs only released Kuiper
   APIs, not the project-owned transcendental overlay. At the real level this
   is the same equation as g / (1 + exp(-g)); the physical programs below
   still use division rather than multiplication by a rounded reciprocal. *)
inline_for_extraction noextract
let silu_real (gate : real) : real =
  exp_positive (0.0R -. gate);
  gate *. (1.0R /. (1.0R +. exp (0.0R -. gate)))

inline_for_extraction noextract
let swiglu_real (gate up : real) : real =
  silu_real gate *. up

inline_for_extraction noextract
let swiglu_matmul_real
  (#m #k #n : nat)
  (a : chest2 real m k)
  (w_gate w_up : chest2 real k n)
  : GTot (chest2 real m n)
  = chest_comb swiglu_real (MS.matmul a w_gate) (MS.matmul a w_up)

inline_for_extraction noextract
let silu_div_f32 (gate : f32) : f32 =
  div #f32 gate (add #f32 one (fexp #f32 (sub #f32 zero gate)))

(* Half gate/up: float SiLU, round SiLU to half, then half multiplication. *)
inline_for_extraction noextract
let swiglu_f16_div (gate up : half) : half =
  let gate_f32 = Cast.cast_f16_to_f32 gate in
  let silu_f32 = silu_div_f32 gate_f32 in
  let silu_f16 = Cast.cast_f32_to_f16 silu_f32 in
  mul #half silu_f16 up

(* Float gate/up: float SiLU and multiplication, then one final half cast. *)
inline_for_extraction noextract
let swiglu_f32_div_to_f16 (gate up : f32) : half =
  Cast.cast_f32_to_f16 (mul #f32 (silu_div_f32 gate) up)

(* Confirmed base cell: retain the half SiLU boundary, then promote both
   half operands for a float product and round the product back to half.
   This is a separate program, not a physical equality with swiglu_f16_div. *)
inline_for_extraction noextract
let swiglu_f16_div_promoted_mul (gate up : half) : half =
  let gate_f32 = Cast.cast_f16_to_f32 gate in
  let silu_f32 = silu_div_f32 gate_f32 in
  let silu_f16 = Cast.cast_f32_to_f16 silu_f32 in
  let activated_f32 = Cast.cast_f16_to_f32 silu_f16 in
  let up_f32 = Cast.cast_f16_to_f32 up in
  Cast.cast_f32_to_f16 (mul #f32 activated_f32 up_f32)

val silu_div_f32_approx (gate : f32) (g : real)
  : Lemma
      (requires gate %~ g)
      (ensures silu_div_f32 gate %~ silu_real g)

val swiglu_f16_div_approx (gate up : half) (g u : real)
  : Lemma
      (requires gate %~ g /\ up %~ u)
      (ensures swiglu_f16_div gate up %~ swiglu_real g u)

val swiglu_f32_div_to_f16_approx (gate up : f32) (g u : real)
  : Lemma
      (requires gate %~ g /\ up %~ u)
      (ensures swiglu_f32_div_to_f16 gate up %~ swiglu_real g u)

(* The exact per-cell output premise is established by the calling witness.
   In particular, knowing just the two projection refinements does not prove
   that an arbitrary output buffer contains the epilogue's result. *)
val swiglu_f16_div_matmul_approx
  (#m #k #n : nat)
  (a : chest2 real m k)
  (w_gate w_up : chest2 real k n)
  (gate up out : chest2 half m n)
  : Lemma
      (requires
        gate %~ MS.matmul a w_gate /\
        up %~ MS.matmul a w_up /\
        (forall (i : natlt m) (j : natlt n).
          acc2 out i j == swiglu_f16_div (acc2 gate i j) (acc2 up i j)))
      (ensures out %~ swiglu_matmul_real a w_gate w_up)

val swiglu_f32_div_to_f16_matmul_approx
  (#m #k #n : nat)
  (a : chest2 real m k)
  (w_gate w_up : chest2 real k n)
  (gate up : chest2 f32 m n)
  (out : chest2 half m n)
  : Lemma
      (requires
        gate %~ MS.matmul a w_gate /\
        up %~ MS.matmul a w_up /\
        (forall (i : natlt m) (j : natlt n).
          acc2 out i j ==
            swiglu_f32_div_to_f16 (acc2 gate i j) (acc2 up i j)))
      (ensures out %~ swiglu_matmul_real a w_gate w_up)

val swiglu_f16_div_promoted_mul_approx (gate up : half) (g u : real)
  : Lemma
      (requires gate %~ g /\ up %~ u)
      (ensures swiglu_f16_div_promoted_mul gate up %~ swiglu_real g u)

val swiglu_f16_div_promoted_mul_matmul_approx
  (#m #k #n : nat)
  (a : chest2 real m k)
  (w_gate w_up : chest2 real k n)
  (gate up out : chest2 half m n)
  : Lemma
      (requires
        gate %~ MS.matmul a w_gate /\
        up %~ MS.matmul a w_up /\
        (forall (i : natlt m) (j : natlt n).
          acc2 out i j ==
            swiglu_f16_div_promoted_mul (acc2 gate i j) (acc2 up i j)))
      (ensures out %~ swiglu_matmul_real a w_gate w_up)
