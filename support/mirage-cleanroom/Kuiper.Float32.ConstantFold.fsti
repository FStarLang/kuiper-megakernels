module Kuiper.Float32.ConstantFold

module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module A = Kuiper.Approximates.Base
module Lit = Kuiper.Float32.Literals
module Cert = Kuiper.Float32.ConstantFold.Certificates

inline_for_extraction let () = ()

(* Trusted literal interpretation has one fixed target per source spelling.
   A rounding-cell certificate cannot choose another ideal for the same value. *)
val literal_approx (s:string) (mantissa:nat) (exponent:int)
  : Lemma
      (requires Cert.exact_normal s mantissa exponent)
      (ensures A.v_approximates (F.of_literal #FP.t s) (Lit.value s))

(* Trusted correspondence for correctly rounded division, not fast reciprocal.
   Bit equality deliberately does not transport approximation facts. *)
val reciprocal_literal_bit_eq
  (input output:string) (im:nat) (ie:int) (om:nat) (oe:int)
  : Lemma
      (requires Cert.reciprocal input output im ie om oe)
      (ensures F.bit_eq (F.of_literal #FP.t output)
        (F.div (F.one #FP.t) (F.of_literal #FP.t input)))
