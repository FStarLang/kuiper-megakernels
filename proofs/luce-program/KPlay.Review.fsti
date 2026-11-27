module KPlay.Review

(* Base/optimized review of the LUCE Qwen3.5-0.8B greedy BF16 decode.

   Both reduced programs are related to the same role-independent
   KPlay.SharedSpec.program_postcondition. The roles differ in their family
   decomposition and in how they pad the run to the fixed 12-position cache
   extent (KPlay.Proof.Wiring.<Role>, a translation of <role>/plan.json).
   Base (pinned HF transformers 5.3.0 reference) uses initialize,
   embedding_rope, recurrent_decoder (18 Gated DeltaNet layers),
   full_attention_decoder (6 layers), final_norm_projection, argmax and
   token_history_update. Optimized (Luce-Org/lucebox megakernel) uses
   initialize, embedding, recurrent_decoder, attention_decoder,
   final_projection, argmax and inactive_decode (padding positions). Their family
   relations are stated in KPlay.SharedSpec; this module only checks that both
   role theorems conclude the same predicate on the same caller inputs.

   This is common-task refinement, not output equality: hidden rows and score
   rows need only approximate (%~) the same real reference, so the generated
   tokens and hidden states may differ between roles. *)

open KPlay.SharedSpec
module S = FStar.Seq
module BF = Kuiper.BFloat16
module I64 = FStar.Int64
module B = KPlay.Proof.Wiring.Base
module O = KPlay.Proof.Wiring.Optimized

(* The common caller inputs: packed BF16 weights [752393024], prompt tokens [P]
   and the number of decode steps S. *)
noeq type inputs = {
  weights : S.seq BF.t;
  prompt : S.seq I64.t;
  steps : int;
}

let base_inputs (r:B.run) : inputs = { weights = r.B.input0; prompt = r.B.input1; steps = r.B.input2 }

let optimized_inputs (r:O.run) : inputs = { weights = r.O.input0; prompt = r.O.input1; steps = r.O.input2 }

(* The common guarantee about the generated tokens [S] and hidden rows [S,1024]. *)
let after_call (x:inputs) (tokens:S.seq I64.t) (hidden:S.seq BF.t) : prop =
  program_postcondition x.weights x.prompt x.steps tokens hidden

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
