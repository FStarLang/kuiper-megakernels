module KPlay.Review

(* Base/optimized review of the Mirage programs.

   Both reduced programs are related to the same role-independent
   KPlay.SharedSpec.program_postcondition. The two roles differ only in their
   family decomposition (KPlay.Proof.Wiring.<Role>): base calls embedding,
   28 decoder layers, projection and first-maximum argmax; optimized calls a
   padded eight-row embedding, 28 split-weight decoder layers, a padded
   projection and a tree argmax. Their family relations are stated in
   KPlay.SharedSpec; this module only checks that both role theorems conclude
   the same predicate on the same caller inputs.

   This is common-task refinement, not output equality: on tied maximal
   scores the two roles may return different tokens, and their hidden rows
   need not be bitwise equal. *)

open KPlay.SharedSpec
module S = FStar.Seq
module BF = Kuiper.BFloat16
module I64 = FStar.Int64
module B = KPlay.Proof.Wiring.Base
module O = KPlay.Proof.Wiring.Optimized

(* The common caller inputs: packed weights, prompt, decode steps. *)
noeq type inputs = {
  weights : S.seq BF.t;
  prompt : S.seq I64.t;
  decode_steps : int;
}

let base_inputs (r:B.run) : inputs =
  { weights = r.B.input0; prompt = r.B.input1; decode_steps = r.B.input2 }

let optimized_inputs (r:O.run) : inputs =
  { weights = r.O.input0; prompt = r.O.input1; decode_steps = r.O.input2 }

(* The common guarantee about returned tokens and hidden rows. *)
let after_call (x:inputs) (tokens:S.seq I64.t) (hidden:S.seq BF.t) : prop =
  program_postcondition x.weights x.prompt x.decode_steps tokens hidden

val base (r:B.run)
  : Lemma (requires B.input_lengths r /\ B.family_ok r)
          (ensures after_call (base_inputs r) (B.output0 r) (B.output1 r))

val optimized (r:O.run)
  : Lemma (requires O.input_lengths r /\ O.family_ok r)
          (ensures after_call (optimized_inputs r) (O.output0 r) (O.output1 r))

(* Run on the same caller inputs, both roles satisfy the same guarantee. *)
val same_guarantee (rb:B.run) (ro:O.run)
  : Lemma (requires B.input_lengths rb /\ B.family_ok rb /\
                    O.input_lengths ro /\ O.family_ok ro /\
                    base_inputs rb == optimized_inputs ro)
          (ensures (let x = base_inputs rb in
                    after_call x (B.output0 rb) (B.output1 rb) /\
                    after_call x (O.output0 ro) (O.output1 ro)))
