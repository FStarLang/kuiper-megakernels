module Kuiper.Float32.Extended

friend Kuiper.Float32.Base
friend Kuiper.Float32
friend Kuiper.Float32.FastMath

#lang-pulse

open Pulse.Lib.Core
open Kuiper.Locs
open Kuiper.Real
open Kuiper.Float.Model
module A = Kuiper.Approximates.Base
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module Fast = Kuiper.Float32.FastMath
module Raw = Kuiper.Float32.Raw
module RawFast = Kuiper.Float32.FastMath.Raw

noeq type t = {
  physical : Raw.t;
  interpretation : ideal;
}

let meaning (x:t) = x.interpretation
let raw (x:t) = x.physical
let raw32 (x:FP.t) = x.repr

let negative_infinity : t = {
  physical = Raw.neg Raw.infinity;
  interpretation = NegInfinity;
}

let lift (x:FP.t) : t = {
  physical = x.repr;
  interpretation = Real x.ideal;
}

let pure_sub_rn (x:t) (y:FP.t) : t = {
  physical = RawFast.sub_rn x.physical y.repr;
  interpretation = subtract_ideal x.interpretation y.ideal;
}

let pure_exp (x:t) : FP.t = {
  repr = Raw.fexp x.physical;
  ideal = exponential_ideal x.interpretation;
}

noextract
fn sub_rn (x:t) (y:FP.t)
  preserves gpu
  returns z:t
  ensures pure (z == pure_sub_rn x y)
{
  let z = {
    physical = RawFast.sub_rn x.physical y.repr;
    interpretation = subtract_ideal x.interpretation y.ideal
  };
  z
}

noextract
fn fexp (x:t)
  preserves gpu
  returns y:FP.t
  ensures pure (y == pure_exp x)
{
  let y:FP.t = {
    repr = Raw.fexp x.physical;
    ideal = exponential_ideal x.interpretation
  };
  y
}

let real_subtraction (x y:FP.t) = ()
let real_exponential (x:FP.t) = ()

let negative_infinity_native (m:FP.t) =
  let p = Raw.fexp (RawFast.sub_rn (Raw.neg Raw.infinity) m.repr) in
  Raw.bit_eq_spec p p
