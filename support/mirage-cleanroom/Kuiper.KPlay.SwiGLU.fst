module Kuiper.KPlay.SwiGLU

open Kuiper
open Kuiper.Chest

module Cast = Kuiper.Float.Casts.Base
module MS = Kuiper.Spec.GEMM
module EM = Kuiper.EMatrix

(* Attribution: the scalar-then-matrix refinement structure follows
   deepopt-kuiper/examples/Kuiper.Kernel.UGS.Epilogue.{fst,fsti} and
   Kuiper.Kernel.UGS.Round.{fst,fsti} (commit 15a1d1b).
   The physical division programs and their distinct cast boundaries are
   specified in this module's interface, rather than reusing UGS's physical
   reciprocal-multiply epilogue. See the interface for the inherited trust
   boundary; these proofs add no physical equivalence primitive. *)

let silu_div_f32_approx (gate : f32) (g : real)
  : Lemma
      (requires gate %~ g)
      (ensures silu_div_f32 gate %~ silu_real g)
  =
  assert (has_type Kuiper.Float32.is_floating_real_like (floating_real_like f32));
  let neg_gate = sub #f32 zero gate in
  assert (neg_gate %~ (0.0R -. g));
  let exp_gate = fexp #f32 neg_gate in
  assert (exp_gate %~ exp (0.0R -. g));
  let denominator = add #f32 one exp_gate in
  a_add (one #f32) exp_gate 1.0R (exp (0.0R -. g));
  exp_positive (0.0R -. g);
  assert (div #f32 gate denominator %~ (g /. (1.0R +. exp (0.0R -. g))));
  assert (g /. (1.0R +. exp (0.0R -. g)) == silu_real g)

let swiglu_f16_div_approx (gate up : half) (g u : real)
  : Lemma
      (requires gate %~ g /\ up %~ u)
      (ensures swiglu_f16_div gate up %~ swiglu_real g u)
  =
  let gate_f32 = Cast.cast_f16_to_f32 gate in
  Cast.cast_f16_to_f32_ok gate g;
  silu_div_f32_approx gate_f32 g;
  let silu_f32 = silu_div_f32 gate_f32 in
  Cast.cast_f32_to_f16_ok silu_f32 (silu_real g);
  a_mul (Cast.cast_f32_to_f16 silu_f32) up (silu_real g) u

let swiglu_f32_div_to_f16_approx (gate up : f32) (g u : real)
  : Lemma
      (requires gate %~ g /\ up %~ u)
      (ensures swiglu_f32_div_to_f16 gate up %~ swiglu_real g u)
  =
  silu_div_f32_approx gate g;
  let silu_f32 = silu_div_f32 gate in
  a_mul silu_f32 up (silu_real g) u;
  Cast.cast_f32_to_f16_ok (mul #f32 silu_f32 up) (swiglu_real g u)

let swiglu_f16_div_matmul_approx
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
  =
  introduce forall (i : natlt m) (j : natlt n).
    acc2 out i j %~ acc2 (swiglu_matmul_real a w_gate w_up) i j
  with (
    assert (acc2 gate i j %~ acc2 (MS.matmul a w_gate) i j);
    assert (acc2 up i j %~ acc2 (MS.matmul a w_up) i j);
    swiglu_f16_div_approx (acc2 gate i j) (acc2 up i j)
      (acc2 (MS.matmul a w_gate) i j) (acc2 (MS.matmul a w_up) i j)
  );
  EM.lemma_approximates_intro out (swiglu_matmul_real a w_gate w_up)

let swiglu_f32_div_to_f16_matmul_approx
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
  =
  introduce forall (i : natlt m) (j : natlt n).
    acc2 out i j %~ acc2 (swiglu_matmul_real a w_gate w_up) i j
  with (
    assert (acc2 gate i j %~ acc2 (MS.matmul a w_gate) i j);
    assert (acc2 up i j %~ acc2 (MS.matmul a w_up) i j);
    swiglu_f32_div_to_f16_approx (acc2 gate i j) (acc2 up i j)
      (acc2 (MS.matmul a w_gate) i j) (acc2 (MS.matmul a w_up) i j)
  );
  EM.lemma_approximates_intro out (swiglu_matmul_real a w_gate w_up)

let swiglu_f16_div_promoted_mul_approx (gate up : half) (g u : real)
  : Lemma
      (requires gate %~ g /\ up %~ u)
      (ensures swiglu_f16_div_promoted_mul gate up %~ swiglu_real g u)
  =
  let gate_f32 = Cast.cast_f16_to_f32 gate in
  Cast.cast_f16_to_f32_ok gate g;
  silu_div_f32_approx gate_f32 g;
  let silu_f32 = silu_div_f32 gate_f32 in
  Cast.cast_f32_to_f16_ok silu_f32 (silu_real g);
  let silu_f16 = Cast.cast_f32_to_f16 silu_f32 in
  Cast.cast_f16_to_f32_ok silu_f16 (silu_real g);
  Cast.cast_f16_to_f32_ok up u;
  let activated_f32 = Cast.cast_f16_to_f32 silu_f16 in
  let up_f32 = Cast.cast_f16_to_f32 up in
  a_mul activated_f32 up_f32 (silu_real g) u;
  Cast.cast_f32_to_f16_ok (mul #f32 activated_f32 up_f32) (swiglu_real g u)

let swiglu_f16_div_promoted_mul_matmul_approx
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
  =
  introduce forall (i : natlt m) (j : natlt n).
    acc2 out i j %~ acc2 (swiglu_matmul_real a w_gate w_up) i j
  with (
    assert (acc2 gate i j %~ acc2 (MS.matmul a w_gate) i j);
    assert (acc2 up i j %~ acc2 (MS.matmul a w_up) i j);
    swiglu_f16_div_promoted_mul_approx (acc2 gate i j) (acc2 up i j)
      (acc2 (MS.matmul a w_gate) i j) (acc2 (MS.matmul a w_up) i j)
  );
  EM.lemma_approximates_intro out (swiglu_matmul_real a w_gate w_up)
