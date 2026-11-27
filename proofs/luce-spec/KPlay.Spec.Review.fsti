module KPlay.Spec.Review

(* Top-level review statement for the LUCE Qwen3.5-0.8B greedy BF16 decode,
   phrased in the typed specification KPlay.Spec.

   For each role (base: pinned HF transformers reference; optimized: Luce
   megakernel), a run whose caller inputs have the expected lengths and whose
   kernel families satisfy their relations produces outputs that satisfy
   KPlay.Spec.postcondition, read through the layout-only views of
   KPlay.Spec.Bridge. Proved from KPlay.Review (which concludes the flat
   KPlay.SharedSpec.program_postcondition) and Bridge.program_to_typed. The
   converse, Bridge.typed_to_program, shows the typed statement is not weaker
   than the flat one. *)

module B = KPlay.Proof.Wiring.Base
module O = KPlay.Proof.Wiring.Optimized
module T = KPlay.Spec
module V = KPlay.Spec.Bridge

val base (r:B.run)
  : Lemma (requires B.input_lengths r /\ B.family_ok r)
          (ensures T.postcondition (V.inputs_of r.B.input0 r.B.input1 r.B.input2)
                                   (V.outputs_of (B.output0 r) (B.output1 r)))

val optimized (r:O.run)
  : Lemma (requires O.input_lengths r /\ O.family_ok r)
          (ensures T.postcondition (V.inputs_of r.O.input0 r.O.input1 r.O.input2)
                                   (V.outputs_of (O.output0 r) (O.output1 r)))
