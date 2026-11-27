module KPlay.Spec.Review

open KPlay.Spec
module Br = KPlay.Spec.Bridge
module B = KPlay.Proof.Wiring.Base
module O = KPlay.Proof.Wiring.Optimized
module R = KPlay.Review

let base r =
  R.base r;
  Br.program_postcondition_typed r.B.input0 r.B.input1 r.B.input2 (B.output0 r) (B.output1 r)

let optimized r =
  R.optimized r;
  Br.program_postcondition_typed r.O.input0 r.O.input1 r.O.input2 (O.output0 r) (O.output1 r)

let same_guarantee rb ro =
  base rb;
  optimized ro
