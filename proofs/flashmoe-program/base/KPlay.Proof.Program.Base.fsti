module KPlay.Proof.Program.Base

(* GENERATED; do not edit. The composition theorem for the base program:
   the family relations at every host call imply the program postcondition.
   Prove it in KPlay.Proof.Program.Base.fst. *)

open KPlay.SharedSpec
open KPlay.Proof.Wiring.Base

val refines (r:run)
  : Lemma (requires input_lengths r /\ family_ok r)
          (ensures program_postcondition r.input0 r.input1 (output0 r) (output1 r) (output2 r) (output3 r))
