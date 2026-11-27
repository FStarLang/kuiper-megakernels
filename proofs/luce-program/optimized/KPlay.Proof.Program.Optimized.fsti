module KPlay.Proof.Program.Optimized

(* GENERATED; do not edit. The composition theorem for the optimized program:
   the family relations at every host call imply the program postcondition.
   Prove it in KPlay.Proof.Program.Optimized.fst. *)

open KPlay.SharedSpec
open KPlay.Proof.Wiring.Optimized

val refines (r:run)
  : Lemma (requires input_lengths r /\ family_ok r)
          (ensures program_postcondition r.input0 r.input1 r.input2 (output0 r) (output1 r))
