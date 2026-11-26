# Base/optimized contract catalog

Start with the linked `KPlay.Review.fsti`: it contains the input/output
representations, both wrapper signatures, a detailed comparison immediately
above `switcher`, and the **actual dispatch-only implementation** of `switcher`.
The corresponding `.fst` proves the adaptations and calls the native verified
implementations. Each manifest review case includes **all roots of both native
implementation cases**, not merely their interfaces.

The Mirage proofs in this repository come from the clean-room RESOLVE proof
pipeline (see the [README](../README.md#mirage-clean-room-resolve-pipeline-proof)).
Their one review, [`mirage-program`](mirage-program/KPlay.Review.fsti), is simpler than
the ones below: both roles take identical inputs and satisfy the identical shared
`program_postcondition`, so there is no representation difference to assess. Mirage
has no tightness cases. The detailed assessments below cover LUCE and FlashMoE.

## How to read the assessments

The judgments below concern **substitutability at the reviewed boundary**, not
measured performance or approval of a numerical policy. A role-indexed type is
not a conversion routine: dispatch does not repack buffers, execute missing
prefix steps, initialize missing auxiliary state, or discharge numerical
premises.

| Significance | Meaning |
| --- | --- |
| None for the task / low operational | The mathematical guarantee is unchanged; an explicit scheduling or indexing distinction still matters to callers. |
| Moderate | Representation, precision, batching, or auxiliary-state differences require caller/composition work. A common logical target does not establish equal native results. |
| High | The available functional/validity guarantee changes, permitted choices can change observable behavior, or an output/API mismatch materially prevents direct substitution. |

**Tight functional specification** needs a stated observation boundary:

- **Exact functional:** fixed specified inputs determine the common observable
  result, not merely its shape or a safety invariant.
- **Exact allowed-choice relation:** every allowed result is characterized, but
  deliberate ties/choices mean the specification is not a single-valued function.
- **Relative functional target:** the logical result is determined, and native
  cells are related to it through the declared approximation/representation.
  This is substantive functional correctness, but **not** uniqueness of native
  floating-point outputs or a quantitative error bound.
- **Conditional:** the functional conclusion holds only under the displayed
  representation/numerical premises. A uniqueness theorem for the shared model
  does not discharge those premises. In particular, a universal implication over
  representations does not prove that arbitrary native inputs have one.

The eight family `KPlay.Proof.Tightness.fst` files provide witnesses and
uniqueness or allowed-result characterizations at their **shared-specification**
boundary. They do not automatically certify every padding cell, every auxiliary
native output, or a composed whole program. The program reviews provide checked
contract correspondence, not additional whole-program tightness certificates.
Allocation failures and completion epochs also remain observable behavior; a
claim about successful task outputs does not erase them.

Here, **unconditional** always means subject to the function's existing
precondition, not a guarantee for arbitrary input bytes.

All statements are relative to the existing trusted numerical/support theory;
strict checking does not remove that theory's assumptions. See
[claims and trust boundary](../CLAIMS.md). A tight specification could still
describe the wrong intended task: reference fidelity is a separate assessment.
Unless explicitly stated for a fixed selector input, **no row below establishes
equality of outputs from independently executed base and optimized programs**.

## Complete programs

| Review | Signature relation and what it means | Significance judgment | Is the proven specification tight functional? |
| --- | --- | --- | --- |
| [flashmoe-program](flashmoe-program/KPlay.Review.fsti) | Identical `before_call`, outcome, ownership, frame recovery and `Shared.task_relation`; `after_call` differs only by `completion`: **one epoch versus three**. Allocation failure keeps the original epoch. Once enqueued, construction failure and success use the role's completion epoch. The `.fst` proves failure equality and return-predicate equality after aligning epochs. | **No functional-task difference; low operational significance.** A consumer must wait for the correct queue tail. The epoch count is not a runtime ratio. Scope is the retained router, not expert MLP execution. | **Complete relative task, not a single-valued native function or separately certified whole-program tightness.** Every routing output is connected to the actual returned cells, but permitted projection/softmax observations and selections can differ between executions. Epoch alignment equates predicates on one result, not two computed results. |
| [luce-program](luce-program/KPlay.Review.fsti) | `input_views_agree` equates packed values, addresses and prompt storage. Both have the same precondition, output shapes, allocation-failure contract and **next-epoch** completion. `observed_payload` gives common complete safety/history; base numerical prefix is unconditional, optimized requires **`family_dependencies /\ numerical_scope` of its actual execution**. Observation types include BF16 versus FP32 convolution/normalization. | **High semantic significance.** The same caller ABI and epoch do not make the unconditional numerical guarantees interchangeable. Dispatch does not establish the optimized premises. | **Complete relative execution/history relation; conditional numerical functionality for optimized, not whole-program uniqueness.** Actual token/hidden cells, feedback and caller state are linked to execution witnesses. Numerical/selection freedom remains; family tightness results do not imply a uniquely determined generated history here. |

## FlashMoE families

| Review | Signature relation and what it means | Significance judgment | Is the proven specification tight functional? |
| --- | --- | --- | --- |
| [flashmoe-projection_softmax](flashmoe-projection_softmax/KPlay.Review.fsti) | `rows Base = 1`, versus a full profile matrix for optimized. Base initializes/copies routing auxiliaries; optimized has none. Both approximate the same row-wise projection/softmax target; optimized additionally covers **every represented input model**. `base_batch_refines` lifts collected row guarantees, but `switcher` does not execute that batch. | **Moderate:** batching, auxiliary state and the additional quantified guarantee prevent literal one-call substitution. The underlying row-wise mathematical task is unchanged. | **Relative functional target, not unique FP32 outputs.** [Tightness `witness`/`unique`](flashmoe-projection_softmax/KPlay.Proof.Tightness.fst) fixes the shared real matrix and preserved inputs. The review's `%~` relation still allows native numerical freedom and does not establish equal score comparisons. |
| [flashmoe-argmax](flashmoe-argmax/KPlay.Review.fsti) | Base permits only one row and copies routing-frame/row buffers; optimized permits a batch and materializes gathered-best cells. Both prove the **same stable maximum/gather relation on the supplied finite FP32 probabilities**. `base_batch_refines` is a composition proof, not a new executor. | **Moderate API significance; no selector-task difference on identical rows.** Batch size and auxiliary ownership differ. This does not equate the outputs of upstream probability producers. | **Yes: exact functional selector on fixed native input rows.** [Tightness `witness`/`unique`](flashmoe-argmax/KPlay.Proof.Tightness.fst) fixes IDs and gathered denotations; `after` also fixes actual optimized gathered cells. Stable ties are specified rather than left arbitrary. |
| [flashmoe-normalization_capacity](flashmoe-normalization_capacity/KPlay.Review.fsti) | Base advances one represented prefix; optimized returns the full route. Counts are I64 versus I32. Base emits **eight I32 demand cells only at token 255, even in Full**; final Full demand for all experts is a logical view of I64 counts, not a returned 16-cell I32 buffer. `completed_refines_shared` connects completion to `route_post`. | **High integration/output significance.** A full-route or Full-demand consumer cannot replace optimized with one base call or assume a missing physical demand array. The completed logical routing task remains common. | **Exact discrete routing; relative numerical weights.** [Tightness `full_contract`, `exact_outputs_unique`, `all_targets_unique`](flashmoe-normalization_capacity/KPlay.Proof.Tightness.fst) fixes demand, owners and survival for the selected boundary and characterizes weight targets. Occupied float weights retain approximation freedom; the review additionally fixes unused weights to native zero. |

## LUCE families

| Review | Signature relation and what it means | Significance judgment | Is the proven specification tight functional? |
| --- | --- | --- | --- |
| [luce-embedding](luce-embedding/KPlay.Review.fsti) | Identical exact BF16 embedding-row `post` and token-selection domain. Base additionally produces rotary data and initial zero state/convolution buffers; optimized `extra` is unit. Half-permission inputs allow the previous-token/prompt alias. | **No common-row difference; moderate integration significance.** Optimized does not replace base's decoder initialization or rotary-output obligations. | **Yes for the common hidden row.** [Tightness `witness`/`unique`](luce-embedding/KPlay.Proof.Tightness.fst) determines the exact BF16 datum sequence, not merely a real approximation. Base auxiliary cells have separate explicit guarantees; the two complete multi-output APIs are not identical. |
| [luce-recurrent_decoder](luce-recurrent_decoder/KPlay.Review.fsti) | BF16 versus FP32 convolution cells; transposed recurrent key/value axes; base always needs materialized old state while optimized has empty initial-state ownership. `post` gives the same complete shared layer/cache transition **for every represented logical input**, via role-specific input/output representations. | **Moderate:** state ABI, initial allocation and precision affect integration and possibly later numerical behavior. No different logical recurrence is introduced. | **Relative, representation-conditional functionality.** [Tightness `witness`/`unique`](luce-recurrent_decoder/KPlay.Proof.Tightness.fst) determines hidden, recurrent, convolution and framed KV state for fixed logical inputs. It does not prove arbitrary native inputs have a representation or uniquely determine native floating cells. |
| [luce-full_attention_decoder](luce-full_attention_decoder/KPlay.Review.fsti) | Compact time-major versus full head-major KV arrays; explicit base rotary input versus internal optimized rotary computation. Initial source buffers are null. Both retain old physical cache cells, but `numerical_premise` is **True for base versus finiteness of actual optimized live scores** before `semantic_post` follows. | **High semantic significance:** the optimized functional conclusion is conditional where base's is not. Cache/layout changes also require moderate ABI adaptation. Unconditional ownership/retention is not unconditional decoder correctness. | **Conditional relative functionality.** [Tightness `witness`/`unique`](luce-full_attention_decoder/KPlay.Proof.Tightness.fst) fixes the shared hidden/cache result. Native approximation freedom remains; outside optimized finiteness, the review guarantees retained cells/ownership but not the full numerical transition. |
| [luce-final_norm_projection](luce-final_norm_projection/KPlay.Review.fsti) | Identical inputs and `targets`; returned score type is **BF16 versus FP32**. Both exactly append the original, unnormalized hidden row on emitting calls and preserve input/history permissions. | **Moderate numerical/ABI significance.** The mathematical logits and history operation agree, but the contracts do not ensure comparison preservation or equal next-token choices. | **Relative functional logits plus exact history.** [Tightness `unique`/`full_contract`](luce-final_norm_projection/KPlay.Proof.Tightness.fst) determines logical logits, hidden and state and characterizes scalar approximation. It does not turn the two native score types into a unique floating-output function. |
| [luce-argmax](luce-argmax/KPlay.Review.fsti) | BF16 versus FP32 inputs with explicit category/order/tie correspondence, including NaNs/infinities and signed-zero keys. Base promises **first maximum**; optimized permits **any maximum**. `base_post_refines_optimized` proves one-way implication; base now exposes the common fact so dispatch itself stays trivial. Both update history exactly. | **High semantic significance:** tied maxima can select different tokens and alter later decoding. Any-maximum does not imply first-maximum; upstream score correspondence is a premise, not a consequence of precision conversion. | **Base: exact first-choice function. Common/optimized: exact allowed-choice relation, not single-valued on ties.** [Tightness `full_contract`/`fixed_choices_unique`](luce-argmax/KPlay.Proof.Tightness.fst) characterizes permitted model/choice histories and uniqueness after choices are fixed; it does not remove those choices. |
