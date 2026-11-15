# Specifications, tightness, and trust

The authoritative claims are the checked declarations, not the names of the
upstream projects. A proof relates a supplied Pulse implementation to a
specification under its explicit premises. Source review and finite native
comparisons connected these translations to retained CUDA historically; the
replay script is not a verified CUDA parser, compiler, or reduction algorithm.

## Complete programs

Complete proof sources: Mirage [base](proofs/mirage-program/base/) /
[optimized](proofs/mirage-program/optimized/), LUCE
[base](proofs/luce-program/base/) / [optimized](proofs/luce-program/optimized/),
and FlashMoE [base](proofs/flashmoe-program/base/) /
[optimized](proofs/flashmoe-program/optimized/).
Each implementation directory includes its contract and complete implementation;
helper modules are checked too. Each program's shared specification and common
`KPlay.Proof.Shared.*` helpers live once at `proofs/<project>-program/`.
The shared-check case IDs remain `<project>-shared`; Mirage's additional
shared-check implementation is in `proofs/mirage-program/shared/`.

| Program | Covered workload and observations | Important boundary |
| --- | --- | --- |
| Mirage | Qwen3-0.6B; all 28 layers, prompt length 7, one decode step, hidden width 1024, vocabulary 151936. Complete prompt-plus-generated tokens and seven returned hidden rows, with the modeled caller/cache/resource chain. | Repaired/authorized retained programs; finite-score selection and bilateral numerical premises. Not arbitrary prompt/decode lengths. |
| LUCE | Qwen3.5-0.8B; all 24 layers: 18 recurrent and six full-attention layers. Prompt lengths 1..520, decode counts 1/5/30/128, and `P+S-1<=2048`. Complete generated-token and pre-final-normalization BF16 hidden histories, recurrent/convolution/KV state, feedback, and caller ownership/lifetimes. | Repaired optimized scalar loads; stable, non-materializing inputs, including ordinary contiguous offset views. FP32 final-normalization observations are distinct from BF16 hidden/history observations. |
| FlashMoE | Retained router only: `S=256,H=512,E=8,C=32,top_k=1`, finite BF16 tokens bounded by 32 and expert-major weights bounded by 4. Complete demand, packed indices, normalized probabilities, and survival. | No expert MLP or larger-profile claim. The exact original optimized candidate and its original-to-retained binding are unavailable. |

The base and optimized roles refine a common **relative algebraic task**.
This is not bitwise equivalence, an IEEE error bound, proof of numerical
stability, or a guarantee of equal selected tokens/experts. Each execution
follows its own permitted numerical observations and selection history.
Satisfaction of the numerical premises is not newly established by replay.

FlashMoE's [`KPlay.Review.fsti`](proofs/flashmoe-program/KPlay.Review.fsti) is a
compact checked review surface for this claim. Its `base` and `optimized` functions
call the distinctly named verified entry modules, with a common precondition and
postcondition parameterized only by the one-versus-three completion epochs.
The corresponding `.fst` proves the adapters from the original role-specific postconditions
to that common form. Allocation and result-construction failures, output
ownership, input preservation and pending frame recovery remain explicit.
The review does not equate native numerical observations or prove the two
executions choose the same experts. The `flashmoe-program-review` case rechecks
both implementation bodies and their helpers along with the review.

Mirage's shared check includes its mixed-observation/legacy compatibility
bridge. Both complete Mirage programs and that check import one
[canonical specification module](proofs/mirage-program/KPlay.SharedSpec.fst);
there are no case-local copies of the complete-program specification.
This deduplication does not change either program's contract or assumptions.
LUCE's shared model similarly separates normalization and hidden
types. These representation changes do not identify unequal native types or
change the real-valued equations. FlashMoE retains actual `q / (+0 + q)`
normalization and the first capacity-eligible tokens in ascending token order;
it does not replace the computation with a literal one.

## Family contracts and their tightness proofs

Each `proofs/<project>-<family>/` directory contains the single shared source
used by its `base/` and `optimized/` implementations and a
`KPlay.Proof.Tightness.fst` module. The shared source retains the exact bytes
bound by both family acceptances.
`witness` constructs a permitted result. The subsequent lemmas establish the
appropriate uniqueness or complete allowed-result characterization.

| Project | Family | Determined observations or retained freedom |
| --- | --- | --- |
| Mirage | [embedding_position](proofs/mirage-embedding_position/) | Embedding values and specified positional/rotary observations |
| Mirage | [decoder](proofs/mirage-decoder/) | Complete hidden result and every modeled cache slot, including contents, length, and initialization |
| Mirage | [final_norm_projection](proofs/mirage-final_norm_projection/) | Normalized values, projected scores, and the specified preserved state |
| Mirage | [argmax](proofs/mirage-argmax/) | Prompt/output relation and allowed maxima of the same actual finite scores; uniqueness after fixing the permitted choice |
| LUCE | [embedding](proofs/luce-embedding/) | Complete embedding vector |
| LUCE | [recurrent_decoder](proofs/luce-recurrent_decoder/) | Hidden values and complete convolution/recurrent state |
| LUCE | [full_attention_decoder](proofs/luce-full_attention_decoder/) | Hidden values and complete specified KV state |
| LUCE | [final_norm_projection](proofs/luce-final_norm_projection/) | Mathematical normalization/projection targets and preserved state; native values remain subject to the declared relation |
| LUCE | [argmax](proofs/luce-argmax/) | Complete output/history relation; NaN priority, infinities, and authorized any-maximum ties remain. Fixing choices determines the state history |
| FlashMoE | [projection_softmax](proofs/flashmoe-projection_softmax/) | Complete mathematical projection/softmax targets under the declared native observation relation |
| FlashMoE | [argmax](proofs/flashmoe-argmax/) | Selected IDs and gathered probabilities, including the native tie rule |
| FlashMoE | [normalization_capacity](proofs/flashmoe-normalization_capacity/) | Exact demand, owners/indices and survival; complete mathematical probability targets with the existing approximation freedom |

Inspect `unique`, `full_contract`, `no_extra_results`, `fixed_choice_unique`,
`fixed_choices_unique`, and `all_targets_unique` where present. Not every
family has the same theorem shape. In particular, float-to-real approximation
does **not** make two allowed floating-point bit patterns identical.

Mirage's [embedding/position review](proofs/mirage-embedding_position/KPlay.Review.fsti)
and [argmax review](proofs/mirage-argmax/KPlay.Review.fsti) additionally provide
checked `base`, `optimized`, and implementation-indexed `switcher` contracts.
Embedding compares the seven logical prompt positions, not optimized padding or
unused rotary-table entries. Argmax exposes the different physical row strides
and requires finite, comparison-faithful logical score representations; its
common postcondition permits the shared specification's tie choices.
Neither review asserts equality of results from separate executions.

The [Mirage final norm/projection review](proofs/mirage-final_norm_projection/KPlay.Review.fsti)
uses one logical input for both roles and displays its physical representation:
packed-weight offsets and residual-plus-delta for base, versus separate gamma,
embedding, and prepared hidden buffers for optimized. Both prove the complete
shared normalization/projection postcondition and projection of all seven rows.
The optimized eighth input row is outside the logical task, and padded score
columns represent zero. This does not establish bitwise equality or preservation
of score comparisons between executions.

The [LUCE argmax review](proofs/luce-argmax/KPlay.Review.fsti) preserves the
stronger first-maximum base guarantee and proves it implies the optimized
any-maximum guarantee for the same logical scores and output. The dispatcher
establishes both its role's guarantee and the common any-maximum guarantee.
Score representation preserves categories (including NaNs and infinities),
signed-zero keys, ordering, and ties across the BF16/FP32 physical interfaces.
Both contracts retain exact history updates and inactive empty histories.

The [LUCE final norm/projection review](proofs/luce-final_norm_projection/KPlay.Review.fsti)
uses identical input ownership and mathematical targets for both implementations,
while exposing the BF16/FP32 score representation difference. Both append the
exact unnormalized input hidden row on emitting calls, preserve prior history,
and preserve input permissions. The common approximation target does not
establish comparison preservation or bitwise equality between their scores.

The [LUCE embedding review](proofs/luce-embedding/KPlay.Review.fsti) states the
same exact BF16 embedding-row postcondition for both implementations. It also
exposes the base-only rotary and initial-state outputs without implying that
the optimized embedding call produces them. Half-permission input ownership
supports the initial previous-token/prompt alias. The common domain requires
valid prompt tokens and an explicitly identified selected token.

The [LUCE recurrent-decoder review](proofs/luce-recurrent_decoder/KPlay.Review.fsti)
exposes the BF16/FP32 convolution-state difference, transposed recurrent
key/value axes, and optimized allocation-free initial state. Both return
represented layer values satisfying the same complete shared decoder/cache
postcondition, including installation of the layer state and cache framing.
The base requires materialized buffers representing zero state on first use.
This is common-task refinement, not equality of separate numerical executions.

The [LUCE full-attention review](proofs/luce-full_attention_decoder/KPlay.Review.fsti)
exposes compact time-major base caches versus full head-major optimized caches,
null initial source buffers, and the base-only rotary input. Both return
represented layer values satisfying the complete shared decoder/cache transition,
but the optimized numerical guarantee requires finiteness of the actual returned
live-score observation. This premise is visible in `numerical_premise`, not
assumed or discharged by the dispatcher. Physical cache retention and ownership
are unconditional. The review does not claim unconditional equivalence of the
native numerical contracts or equality of independent executions.

The [FlashMoE argmax review](proofs/flashmoe-argmax/KPlay.Review.fsti) gives
both entry points the same stable-maximum/gather postcondition on finite FP32
probabilities in [0,1]. It exposes that a base call processes one row and copies
its routing frame, whereas an optimized call processes a batch and materializes
the gathered probabilities. `base_batch_refines` proves that the row contracts
compose into the batched shared postcondition; it is not a new batch executor.
The dispatcher can choose either implementation for one row, and only optimized
for a larger batch. All frame copies and actual best-buffer cells remain explicit.

The [FlashMoE projection/softmax review](proofs/flashmoe-projection_softmax/KPlay.Review.fsti)
gives both roles the same row-wise real projection/softmax target, related to
every actual returned FP32 probability. Base processes one row and initializes
or copies the routing frame, extending survival by a zero row; optimized
processes the complete profile matrix without those auxiliary buffers.
`base_batch_refines` connects collected base-row guarantees to the complete
shared postcondition. The optimized postcondition additionally covers every
represented input model, not just the canonical input denotation.
Dispatch selects a role-specific call shape, not two interchangeable batch
executors, and does not assert equality of floating-point outputs.

The [FlashMoE normalization/capacity review](proofs/flashmoe-normalization_capacity/KPlay.Review.fsti)
exposes one base prefix transition versus complete optimized routing. Both use
the same native selection boundary, with exact selected-row/gather correspondence.
The common postcondition represents the complete shared state at the processed
prefix and the full route on completion; `completed_refines_shared` makes that
connection explicit for either representation. Base counts are I64, and its
separate I32 demand buffer has eight cells only at token 255, including in Full.
Thus final Full demand is a logical view of computed base counts, not a returned
16-cell I32 buffer. Optimized returns all expert demands as actual I32 cells.
Owners, normalized weights, unused zero slots, and every survival cell remain
connected to the shared specification.

These are **family-level** tightness proofs. They are not additional
whole-program tightness certificates, task-correctness proofs, hardware
models, or native termination proofs. A determinate but wrong task could still
be tight; reference fidelity and the approved observation/freedom policies
remain separate obligations. An inconsistent trusted theory would also defeat
the intended interpretation.

## Trusted boundary

* F*, Pulse, Z3, their foundational logic, and the mathematical/numerical model.
  The qualitative approximation relation is not a quantitative error model.
  Existing `assume`/`admit` declarations in the support libraries are visible
  and retained, not silently discharged by strict checking of an example.
* Native scalar operations, BF16/FP32 casts, transcendental and fast/FTZ
  operations, comparison/classification, guarded range/finiteness contracts,
  and BF16 MMA/register-layout contracts.
* CUDA allocation/alignment/residency, arrays and native views, current-stream
  launches and epochs, PyTorch input borrowing and owning results, and the
  specified success/failure/cleanup behavior. These contracts are not proofs
  of CUDA hardware, the C++ standard library, or PyTorch internals.
* Reviewed task interpretation and translation/source correspondence. Fresh
  proof checking establishes the supplied source theorems; hashes establish
  identity, not semantic correspondence.

`support/mirage` and `support/luce` contain only changed/new source dependencies
relative to the pinned public Kuiper package. FlashMoE layers its changes over
the LUCE support snapshot. `support/foundation` contains the five changed
F*/Pulse source files. Unrelated sparse/GEMM patches were excluded using the
actual dependency closures. The historical compiler and the public replay
compiler have the same F* commit but were built with OCaml 5.4.0 and 5.3.0,
respectively; no historical binary-installation fingerprint is claimed.

## Native checks in this release

The complete-program smoke tests use synthetic, non-secret inputs. FlashMoE
checks all four outputs for both a tie/overflow case and a balanced-routing
case. Mirage and LUCE check complete token/hidden output shapes, token range,
zero-weight hidden results, and input preservation; LUCE also uses offset-one
BF16 weights. These are finite diagnostics, not the universal theorem.
Compilation of a family entry point is not execution or numerical validation.

Proof receipts from the original work are represented by their hashes, not
redistributed private logs or review transcripts. Those hashes are provenance
identifiers, not standalone proof of historical acceptance. New replay
results are generated from the source files included here.

The original Mirage optimized-decoder source returned SMT `unknown` on the
real-division rewrite in `rms_radicand_at` during public-toolchain replay.
[replay-adjustments.patch](replay-adjustments.patch) makes that algebraic step
explicit with a proved helper. The original source SHA-256 remains recorded in
`manifest.json`. This changes proof presentation only: the theorem statement,
specification, native operations, and trusted assumptions are unchanged.
