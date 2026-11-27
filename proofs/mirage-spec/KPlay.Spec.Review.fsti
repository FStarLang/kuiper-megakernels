module KPlay.Spec.Review

(* Base/optimized review against the typed specification KPlay.Spec.

   KPlay.Review proves that both reduced Mirage programs satisfy the flat
   KPlay.SharedSpec.program_postcondition. KPlay.Spec.Bridge proves that, for
   caller buffers of the program's lengths, the flat postcondition implies
   the typed KPlay.Spec.postcondition read through the views inputs_of and
   outputs_of. This module composes the two: both roles meet the typed
   specification on the same caller inputs. *)

open KPlay.Spec
module Br = KPlay.Spec.Bridge
module B = KPlay.Proof.Wiring.Base
module O = KPlay.Proof.Wiring.Optimized

let base_input (r:B.run) : inputs = Br.inputs_of r.B.input0 r.B.input1 r.B.input2

let optimized_input (r:O.run) : inputs = Br.inputs_of r.O.input0 r.O.input1 r.O.input2

val base (r:B.run)
  : Lemma (requires B.input_lengths r /\ B.family_ok r)
          (ensures postcondition (base_input r) (Br.outputs_of (B.output0 r) (B.output1 r)))

val optimized (r:O.run)
  : Lemma (requires O.input_lengths r /\ O.family_ok r)
          (ensures postcondition (optimized_input r) (Br.outputs_of (O.output0 r) (O.output1 r)))

(* Run on the same caller buffers, both roles satisfy the same typed guarantee. *)
val same_guarantee (rb:B.run) (ro:O.run)
  : Lemma (requires B.input_lengths rb /\ B.family_ok rb /\
                    O.input_lengths ro /\ O.family_ok ro /\
                    rb.B.input0 == ro.O.input0 /\ rb.B.input1 == ro.O.input1 /\
                    rb.B.input2 == ro.O.input2)
          (ensures (let x = base_input rb in
                    postcondition x (Br.outputs_of (B.output0 rb) (B.output1 rb)) /\
                    postcondition x (Br.outputs_of (O.output0 ro) (O.output1 ro))))
