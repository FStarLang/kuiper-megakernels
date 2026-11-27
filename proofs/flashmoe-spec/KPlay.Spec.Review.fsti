module KPlay.Spec.Review

(* The top statement for FlashMoE, stated against the typed spec.

   Both reduced programs (base and optimized), given caller inputs of the right
   shape and satisfying their per-role family relations, produce outputs that
   satisfy KPlay.Spec.postcondition: there is a routing run whose router
   probabilities approximate softmax(tokens @ router_weights), whose selected
   experts are an allowed top-1 choice, whose combine weights approximate p/p
   for kept tokens, and whose expert_demand, token_indices, probabilities and
   survival are exactly the capacity-pruned routing of that run. The views
   inputs_of/outputs_of only fix the row-major layout of the flat buffers. *)

open KPlay.Spec
open KPlay.Spec.Bridge
module B = KPlay.Proof.Wiring.Base
module O = KPlay.Proof.Wiring.Optimized

val base (r:B.run)
  : Lemma (requires B.input_lengths r /\ B.family_ok r)
          (ensures postcondition (inputs_of r.B.input0 r.B.input1)
                     (outputs_of (B.output0 r) (B.output1 r) (B.output2 r) (B.output3 r)))

val optimized (r:O.run)
  : Lemma (requires O.input_lengths r /\ O.family_ok r)
          (ensures postcondition (inputs_of r.O.input0 r.O.input1)
                     (outputs_of (O.output0 r) (O.output1 r) (O.output2 r) (O.output3 r)))
