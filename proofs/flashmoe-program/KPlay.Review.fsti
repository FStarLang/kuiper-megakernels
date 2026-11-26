module KPlay.Review

(* Base/optimized review of the FlashMoE capacity-pruned top-1 router.

   Both reduced programs are related to the same role-independent
   KPlay.SharedSpec.program_postcondition. The two roles differ only in their
   family decomposition (KPlay.Proof.Wiring.<Role>): base calls one fused
   projection/softmax, then per token a first-maximum argmax and one
   capacity/routing update with int64 prefix counts, and a final int32
   expert_demand conversion; optimized calls one tensor-core projection, then
   per token an online softmax, an argmax and one routing update with int32
   counts. Their family relations are stated in KPlay.SharedSpec; this module
   only checks that both role theorems conclude the same predicate on the
   same caller inputs.

   This is common-task refinement, not output equality: the softmax rows of
   the two roles need only approximate the same real softmax (%~), so the
   selected experts, and hence the routing outputs, may differ between roles. *)

open KPlay.SharedSpec
module S = FStar.Seq
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32
module I32 = FStar.Int32
module I64 = FStar.Int64
module B = KPlay.Proof.Wiring.Base
module O = KPlay.Proof.Wiring.Optimized

(* The common caller inputs: BF16 tokens [256,512] and router weights [512,8]. *)
noeq type inputs = {
  tokens : S.seq BF.t;
  weights : S.seq BF.t;
}

let base_inputs (r:B.run) : inputs = { tokens = r.B.input0; weights = r.B.input1 }

let optimized_inputs (r:O.run) : inputs = { tokens = r.O.input0; weights = r.O.input1 }

(* The common guarantee about demand, token indices, probabilities and survival. *)
let after_call (x:inputs) (demand:S.seq I32.t) (idx:S.seq I64.t) (probs:S.seq F32.t)
  (surv:S.seq I32.t) : prop =
  program_postcondition x.tokens x.weights demand idx probs surv

val base (r:B.run)
  : Lemma (requires B.input_lengths r /\ B.family_ok r)
          (ensures after_call (base_inputs r) (B.output0 r) (B.output1 r) (B.output2 r) (B.output3 r))

val optimized (r:O.run)
  : Lemma (requires O.input_lengths r /\ O.family_ok r)
          (ensures after_call (optimized_inputs r) (O.output0 r) (O.output1 r) (O.output2 r) (O.output3 r))

(* Run on the same caller inputs, both roles satisfy the same guarantee. *)
val same_guarantee (rb:B.run) (ro:O.run)
  : Lemma (requires B.input_lengths rb /\ B.family_ok rb /\
                    O.input_lengths ro /\ O.family_ok ro /\
                    base_inputs rb == optimized_inputs ro)
          (ensures (let x = base_inputs rb in
                    after_call x (B.output0 rb) (B.output1 rb) (B.output2 rb) (B.output3 rb) /\
                    after_call x (O.output0 ro) (O.output1 ro) (O.output2 ro) (O.output3 ro)))
