# Specifications, tightness, and trust

The authoritative claims are the checked declarations, not the names of the
upstream projects. A proof relates a supplied Pulse implementation to a
specification under its explicit premises. Source review and finite native
comparisons connected these translations to retained CUDA historically; the
replay script is not a verified CUDA parser, compiler, or reduction algorithm.

## Complete programs

Complete proof sources: Mirage [base](proofs/mirage-program/base/) /
[optimized](proofs/mirage-program/optimized/) and FlashMoE
[base](proofs/flashmoe-program/base/) / [optimized](proofs/flashmoe-program/optimized/)
(clean-room pipeline; see [Mirage](#mirage-clean-room-pipeline-proof) and
[FlashMoE](#flashmoe-clean-room-pipeline-proof) below), and LUCE
[base](proofs/luce-program/base/) / [optimized](proofs/luce-program/optimized/).
Each implementation directory includes its contract and complete implementation;
helper modules are checked too. The LUCE program's shared specification
and common `KPlay.Proof.Shared.*` helpers live once at `proofs/luce-program/`;
Mirage and FlashMoE each have one `proofs/<project>-shared/KPlay.SharedSpec.fst`.
The shared-check case IDs are `<project>-shared`.

| Program | Covered workload and observations | Important boundary |
| --- | --- | --- |
| Mirage | Qwen3-0.6B; all 28 layers, one forward pass over a seven-token prompt plus the greedy next token, hidden width 1024, vocabulary 151936. Prompt copied before the generated token; seven final normalized hidden rows; the token is an allowed maximum of a score row approximating the reference logits. | Composition over the generated host wiring of the reduced program; family kernels linked to the reduced CUDA by finite bitwise replay. Not arbitrary prompt/decode lengths. |
| LUCE | Qwen3.5-0.8B; all 24 layers: 18 recurrent and six full-attention layers. Prompt lengths 1..520, decode counts 1/5/30/128, and `P+S-1<=2048`. Complete generated-token and pre-final-normalization BF16 hidden histories, recurrent/convolution/KV state, feedback, and caller ownership/lifetimes. | Repaired optimized scalar loads; stable, non-materializing inputs, including ordinary contiguous offset views. FP32 final-normalization observations are distinct from BF16 hidden/history observations. |
| FlashMoE | Capacity-pruned top-1 router only, smoke profile `S=256,H=512,E=8,C=32,top_k=1`; finite BF16 tokens bounded by 32 and `[H,E]` weights bounded by 4. Complete demand, kept token indices, combine probabilities and survival: exact token-order capacity routing of first-maximum selections over softmax rows that approximate the real softmax of `x @ w`. | Optimized original is FlashMoE **with PR #32** (deterministic router); gate weights `[E,H]`. Trusts `cpu_exp_f32_approx`; top-1 stability against rounding not proved. Composition over generated host wiring; links to reduced/original CUDA are finite bitwise checks. No expert MLP or larger profile. |

The base and optimized roles refine a common **relative algebraic task**.
This is not bitwise equivalence, an IEEE error bound, proof of numerical
stability, or a guarantee of equal selected tokens/experts. Each execution
follows its own permitted numerical observations and selection history.
Satisfaction of the numerical premises is not newly established by replay.

The Mirage and FlashMoE `KPlay.Review` modules state that both roles satisfy
the identical role-independent `program_postcondition` on the same inputs.

The [LUCE program review](proofs/luce-program/KPlay.Review.fsti) proves that both
packing maps describe the same 752393024 BF16 weights, parameter addresses and
prompt cells. Both wrappers retain allocation failure, owning returned tensors,
and the complete queued safety/history postcondition at the next epoch.
Their actual execution witnesses remain distinct: base convolution/normalization
observations are BF16, optimized observations FP32. Base numerical correctness
is unconditional; the optimized numerical prefix is conditional on its
`family_dependencies` and `numerical_scope` for that returned execution.
The dispatcher does not claim unconditional numerical equivalence or equal
generated tokens.

LUCE's shared model separates normalization and hidden
types. These representation changes do not identify unequal native types or
change the real-valued equations.

## Family contracts and their tightness proofs

Each LUCE `proofs/luce-<family>/` directory contains the single shared source
used by its `base/` and `optimized/` implementations and a
`KPlay.Proof.Tightness.fst` module. The shared source retains the exact bytes
bound by both family acceptances.
`witness` constructs a permitted result. The subsequent lemmas establish the
appropriate uniqueness or complete allowed-result characterization.

| Project | Family | Determined observations or retained freedom |
| --- | --- | --- |
| LUCE | [embedding](proofs/luce-embedding/) | Complete embedding vector |
| LUCE | [recurrent_decoder](proofs/luce-recurrent_decoder/) | Hidden values and complete convolution/recurrent state |
| LUCE | [full_attention_decoder](proofs/luce-full_attention_decoder/) | Hidden values and complete specified KV state |
| LUCE | [final_norm_projection](proofs/luce-final_norm_projection/) | Mathematical normalization/projection targets and preserved state; native values remain subject to the declared relation |
| LUCE | [argmax](proofs/luce-argmax/) | Complete output/history relation; NaN priority, infinities, and authorized any-maximum ties remain. Fixing choices determines the state history |

Inspect `unique`, `full_contract`, `no_extra_results`, `fixed_choice_unique`,
`fixed_choices_unique`, and `all_targets_unique` where present. Not every
family has the same theorem shape. In particular, float-to-real approximation
does **not** make two allowed floating-point bit patterns identical.

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

`support/luce` contains only changed/new source dependencies relative to the
default pinned public Kuiper package. `support/mirage-cleanroom` contains the
66 `src/lib` modules that differ from or are absent in Kuiper
`nightly-2026-11-18`; it is the library of the RESOLVE installation that
checked the Mirage proofs. `support/flashmoe-cleanroom` adds the two
`Kuiper.KPlay.CpuExp` modules; with `support/mirage-cleanroom` it is exactly the
library of the RESOLVE installation that checked the FlashMoE proofs. `support/foundation` contains the five changed
F*/Pulse source files; it applies only to the default toolchain. Unrelated sparse/GEMM patches were excluded using the
actual dependency closures. The historical compiler and the public replay
compiler have the same F* commit but were built with OCaml 5.4.0 and 5.3.0,
respectively; no historical binary-installation fingerprint is claimed.

## Native checks in this release

The complete-program smoke tests use synthetic, non-secret inputs. FlashMoE
checks all four outputs for both a tie/overflow case and a balanced-routing
case (weights in the `[512,8]` layout). Mirage and LUCE check complete token/hidden output shapes, token range,
zero-weight hidden results, and input preservation; LUCE also uses offset-one
BF16 weights. These are finite diagnostics, not the universal theorem.
Compilation of a family entry point is not execution or numerical validation.

Proof receipts from the original work are represented by their hashes, not
redistributed private logs or review transcripts. Those hashes are provenance
identifiers, not standalone proof of historical acceptance. New replay
results are generated from the source files included here.

## Mirage clean-room pipeline proof

The Mirage cases come from the clean-room RESOLVE proof pipeline. Its agents
had no access to earlier Mirage proofs, so they were proved fresh from the
reduced CUDA and the PyTorch reference.
One specification, [`proofs/mirage-shared/KPlay.SharedSpec.fst`](proofs/mirage-shared/KPlay.SharedSpec.fst),
states:

1. **Reference math over reals.** Each decoder layer is cut where the
   implementation cuts it: `layer_residual`, then `rmsnorm_rows`.
2. **`program_postcondition`.** Role independent. It states three things:
   - the prompt is copied in front of the generated token;
   - for every real weight vector that the BF16 weights approximate, the returned
     hidden rows approximate `final_hidden`;
   - some score row approximating `next_token_logits` exists, and the token is an
     allowed maximum of that row (`argmax_choice`). That means the first NaN if the row
     has one; otherwise any index whose value is `>=` every value.
3. **One relation per role and family.** Each holds for all real inputs that
   the actual inputs approximate; none is stated over the actual native values.

`KPlay.Proof.Program.<Role>.refines` proves that the family relations at every
host call of the reduced program imply `program_postcondition`. Each family's
Kuiper implementation proves its relation.

`KPlay.Review` (case `mirage-program-review`) checks that both roles, run on
the same inputs, satisfy this same postcondition. Mirage has no tightness cases.

This guarantee is relational, not an error bound and not equality of base and
optimized outputs. `x %~ r` means the machine value `x` stands for the exact real
result `r` with rounding abstracted away; it bounds no distance between them.

The specification cites the reference as `inputs/reference/*`; those files are
shipped byte-identical in [`reference/mirage/`](reference/mirage/). The host
call plans behind the generated wiring are shipped as
`proofs/mirage-program/<role>/plan.json`. `mirage_check.py` replays these plans
on a GPU. It checks each extracted family kernel bitwise against the reduced
family CUDA, and it checks the composed plan bitwise against the reduced program
(results in `benchmarks/mirage-plan-replay.json`). Those are finite observations
that make the trusted links inspectable; they do not replace them.

It is conditional on the trusted assumptions listed prominently in the
[README](README.md#trusted-assumptions-mirage):
- the BF16 widening and literal laws added by RESOLVE;
- the open Kuiper PR #181 `bits_spec` law;
- the generated host wiring (a translation of `plan.json`);
- the finite bitwise replays that connect extracted kernels, reduced CUDA, and
  the original megakernel.

## FlashMoE clean-room pipeline proof

The FlashMoE cases also come from the clean-room RESOLVE proof pipeline; its
agents had no access to the former FlashMoE proofs. The subject is the
capacity-pruned top-1 router. **The optimized original is FlashMoE with
[PR #32](https://github.com/osayamenja/FlashMoE/pull/32) applied**
(`tyler-utah/FlashMoE@96ad271` = `osayamenja/FlashMoE@9cc0c32` + PR #32),
which makes capacity assignment deterministic in input-token order. Its gate
takes weights as `[E,H]`; the reference and both reduced programs take the
logical `[H,E] = [512,8]` layout. One specification,
[`proofs/flashmoe-shared/KPlay.SharedSpec.fst`](proofs/flashmoe-shared/KPlay.SharedSpec.fst),
states:

1. **Reference math over reals** (`reference/flashmoe/base.py` `reference_route`):
   scores `x @ w`, row softmax, first-maximum top-1, the `p / p` combine
   weight, and capacity in input-token order as closed forms (`count_upto`,
   `rank`, `kept`, `slot_token`).
2. **`program_postcondition`.** Role independent. For inputs of the stated
   lengths there exist softmax rows, selections and combine weights such that:
   - every row approximates, for all reals the inputs approximate, the real
     softmax of that row of `x @ w`;
   - each selection is an allowed first maximum of its native row
     (`allowed_choice`);
   - each kept token's combine weight approximates `p / p` of its selected probability;
   - demand, token indices, probabilities and survival are **exactly** the
     token-order capacity routing of those selections (`routing_state`,
     `survival_rows`), with unfilled slots `-1` / `+0.0`.
3. **One relation per role and family.** The roles use different families:
   base `projection_softmax`, `argmax`, `capacity_routing` (I64 counts),
   `expert_demand`; optimized `projection`, `softmax`, `argmax`, `routing`
   (I32 counts).

`KPlay.Proof.Program.<Role>.refines` proves that the family relations at every
host call imply `program_postcondition`; each family's Kuiper implementation
proves its relation. `KPlay.Review` (case `flashmoe-program-review`) checks
that both roles, on the same inputs, satisfy this same postcondition. There
are no tightness cases.

The guarantee is relational. Selections are over the native rows each role
computes, so base and optimized may select different experts where scores are
close. The routing outputs are then exact functions of those selections.

It is conditional on the trusted assumptions listed in the
[README](README.md#trusted-assumptions-flashmoe):
- **`cpu_exp_f32_approx`**, the single trusted law of the opaque
  `Kuiper.KPlay.CpuExp.cpu_exp_f32` (`support/flashmoe-cleanroom/`). The base
  `softmax_exp` is bit-identical to this routine (the scalar CUDA form of
  PyTorch's CPU SLEEF AVX512 `expf`);
- the Kuiper library laws of `support/mirage-cleanroom` (BF16→F32 cast,
  literal approximation, float bit specifications, fast-math intrinsics);
- the generated host wiring (a translation of `plan.json`);
- the finite bitwise links between extracted kernels, reduced CUDA and the
  original programs. RESOLVE's original-to-reduced link is a same-role bitwise
  translation check on 8 samples, not a universal equivalence.

Stability of the top-1 choice against rounding (native argmax = real argmax)
is **not** proved; it is recorded as a separate obligation. RESOLVE's testing
(8 samples, DTester 8/8 comparable runs per sample under unstructured
global/shared delays) is finite evidence about the original and reduced
programs; the proof is about the reduced programs on every input in the domain.
