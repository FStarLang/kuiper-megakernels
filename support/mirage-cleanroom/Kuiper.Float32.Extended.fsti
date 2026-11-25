module Kuiper.Float32.Extended

#lang-pulse

open Pulse.Lib.Core
open Kuiper.Locs
open Kuiper.Real
module A = Kuiper.Approximates.Base
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module Fast = Kuiper.Float32.FastMath
module Raw = Kuiper.Float32.Raw
module RawFast = Kuiper.Float32.FastMath.Raw

noeq type ideal =
  | Real : real -> ideal
  | NegInfinity : ideal

let subtract_ideal (x:ideal) (y:real) : ideal =
  match x with
  | Real r -> Real (r -. y)
  | NegInfinity -> NegInfinity

let exponential_ideal (x:ideal) : real =
  match x with
  | Real r -> exp r
  | NegInfinity -> 0.0R

(* A separate type keeps sentinels out of the total real-valued dictionary.
   Its representation and ideal cannot be constructed or reassigned by clients. *)
val t : Type0
val meaning : t -> GTot ideal
val raw : t -> GTot Raw.t
val raw32 : FP.t -> GTot Raw.t

val negative_infinity : x:t{
  meaning x == NegInfinity /\ raw x == Raw.neg Raw.infinity}

val lift (x:FP.t) : Tot (y:t{
  meaning y == Real (A.to_real x) /\ raw y == raw32 x})

val pure_sub_rn (x:t) (y:FP.t) : GTot (z:t{
  meaning z == subtract_ideal (meaning x) (A.to_real y) /\
  raw z == RawFast.sub_rn (raw x) (raw32 y)})

val pure_exp (x:t) : GTot (y:FP.t{
  A.v_approximates y (exponential_ideal (meaning x)) /\
  raw32 y == Raw.fexp (raw x)})

noextract
fn sub_rn (x:t) (y:FP.t)
  preserves gpu
  returns z:t
  ensures pure (z == pure_sub_rn x y)

noextract
fn fexp (x:t)
  preserves gpu
  returns y:FP.t
  ensures pure (y == pure_exp x)

val real_subtraction (x y:FP.t)
  : Lemma (pure_sub_rn (lift x) y == lift (Fast.pure_sub_rn x y))

val real_exponential (x:FP.t)
  : Lemma (pure_exp (lift x) == F.fexp x)

(* Representation equality is a source-binding fact, not approximation transfer. *)
val negative_infinity_native (m:FP.t)
  : Lemma (
    F.bit_eq (pure_exp (pure_sub_rn negative_infinity m))
      (F.fexp (Fast.pure_sub_rn (F.neg F.infinity) m)))

inline_for_extraction let () = ()
