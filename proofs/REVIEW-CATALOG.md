# Base/optimized review catalog

**Start here:** `proofs/<project>-spec/KPlay.Spec.Review.fsti` states, for both
roles, the guarantee over the typed specification `KPlay.Spec` (records,
named reference equations, typed `postcondition`, in the form of
`review/checked-kernel-specs`). `KPlay.Spec.Bridge.program_postcondition_iff`
proves that typed postcondition equivalent to the flat `program_postcondition`
used below, given buffer lengths. The flat reviews below are what the typed ones
are built from.

All proofs in this repository come from the clean-room RESOLVE proof pipeline
(see the README sections on
[Mirage](../README.md#mirage-clean-room-resolve-pipeline-proof),
[FlashMoE](../README.md#flashmoe-clean-room-resolve-pipeline-proof) and
[LUCE](../README.md#luce-clean-room-resolve-pipeline-proof)). Each project has
one review, `proofs/<project>-program/KPlay.Review.fsti`. In each, both roles
take identical inputs and their theorems `KPlay.Proof.Program.<Role>.refines`
conclude the **identical** role-independent `program_postcondition` of the
project's `KPlay.SharedSpec`, so there is no representation difference to
assess at the reviewed boundary. Each manifest review case includes **all
roots of both implementation cases**, not merely their interfaces.

The roles may use different family decompositions; this is visible only in
their wiring (`KPlay.Proof.Wiring.<Role>`), not in the reviewed postcondition.
No project has tightness cases.

| Review | Common inputs | Common guarantee (`program_postcondition`) | What it does not establish |
| --- | --- | --- | --- |
| [mirage-program](mirage-program/KPlay.Review.fsti) | BF16 weights, seven prompt tokens | Prompt copied before the token; final hidden rows approximate the reference; the token is an allowed maximum of a score row approximating the reference logits | Equal tokens or hidden rows between roles; an error bound |
| [flashmoe-program](flashmoe-program/KPlay.Review.fsti) | BF16 tokens `[256,512]`, router weights `[512,8]` | Softmax rows approximate the real softmax of `x @ w`; first-maximum selections over them; demand, indices, probabilities and survival exactly the token-order capacity routing of those selections | Equal selected experts between roles; top-1 stability against rounding |
| [luce-program](luce-program/KPlay.Review.fsti) | BF16 weights `[752393024]`, prompt `[P]`, decode steps `S` | For `1 <= P <= 8`, `1 <= S <= 5`: each hidden row approximates the reference residual at position `P-1+j`; each token is an allowed maximum of a BF16 or FP32 score row approximating the reference logits there | Equal tokens or hidden rows between roles (RESOLVE observed bitwise differences); an error bound; anything outside the 12-position profile |

All statements are relative to the trusted numerical/support theory listed in
[CLAIMS.md](../CLAIMS.md). A common specification could still describe the
wrong intended task: reference fidelity is a separate assessment. **No row
establishes equality of outputs from independently executed base and
optimized programs.**
