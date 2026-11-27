module KPlay.Spec.Review

open KPlay.Spec
open KPlay.Spec.Bridge
module B = KPlay.Proof.Wiring.Base
module O = KPlay.Proof.Wiring.Optimized
module R = KPlay.Review

let base r =
  R.base r;
  program_postcondition_implies r.B.input0 r.B.input1
    (B.output0 r) (B.output1 r) (B.output2 r) (B.output3 r)

let optimized r =
  R.optimized r;
  program_postcondition_implies r.O.input0 r.O.input1
    (O.output0 r) (O.output1 r) (O.output2 r) (O.output3 r)
