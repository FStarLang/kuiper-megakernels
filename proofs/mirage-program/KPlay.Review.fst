module KPlay.Review

open KPlay.SharedSpec
module B = KPlay.Proof.Wiring.Base
module O = KPlay.Proof.Wiring.Optimized
module PB = KPlay.Proof.Program.Base
module PO = KPlay.Proof.Program.Optimized

let base r = PB.refines r

let optimized r = PO.refines r

let same_guarantee rb ro =
  base rb;
  optimized ro
