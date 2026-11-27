# Specifications, tightness, and trust

The authoritative claims are the checked declarations, not the names of the
upstream projects. A proof relates a supplied Pulse implementation to a
specification under its explicit premises. Source review and finite native
comparisons connected these translations to retained CUDA historically; the
replay script is not a verified CUDA parser, compiler, or reduction algorithm.

## Complete programs

Complete proof sources: Mirage [base](proofs/mirage-program/base/) /
[optimized](proofs/mirage-program/optimized/) and FlashMoE
[base](proofs/flashmoe-program/base/) / [optimized](proofs/flashmoe-program/optimized/),
and LUCE [base](proofs/luce-program/base/) / [optimized](proofs/luce-program/optimized/),
all from the clean-room pipeline (see [Mirage](#mirage-clean-room-pipeline-proof),
[FlashMoE](#flashmoe-clean-room-pipeline-proof) and
[LUCE](#luce-clean-room-pipeline-proof) below).
Each implementation directory includes its contract and complete implementation;
helper modules are checked too. Each project has one
`proofs/<project>-shared/KPlay.SharedSpec.fst`.
The shared-check case IDs are `<project>-shared`.

| Program | Covered workload and observations | Important boundary |
| --- | --- | --- |
| Mirage | Qwen3-0.6B; all 28 layers, one forward pass over a seven-token prompt plus the greedy next token, hidden width 1024, vocabulary 151936. Prompt copied before the generated token; seven final normalized hidden rows; the token is an allowed maximum of a score row approximating the reference logits. | Composition over the generated host wiring of the reduced program; family kernels linked to the reduced CUDA by finite bitwise replay. Not arbitrary prompt/decode lengths. |
| LUCE | Qwen3.5-0.8B greedy BF16 decode; all 24 layers: 18 Gated DeltaNet and six gated full-attention layers. Prompt lengths 1..8, decode steps 1..5 (tested: 1 and 5), cache extent `P+S-1 <= 12`. Returned tokens `[S]` and pre-final-norm BF16 hidden rows `[S,1024]`: each hidden row approximates the reference residual at position `P-1+j`, and each token is an allowed maximum of a BF16 or FP32 score row approximating the reference logits there. | Optimized original is the official **Luce-Org/lucebox** megakernel (`0f212ac`); base is the pinned Transformers 5.3.0 reference. Trusts two softplus laws (`Luce.Softplus*`) and the listed Kuiper laws. Composition over generated host wiring; links to reduced/original CUDA are finite bitwise checks. Relations are vacuous outside the profile's integer domain. |
| FlashMoE | Capacity-pruned top-1 router only, smoke profile `S=256,H=512,E=8,C=32,top_k=1`; finite BF16 tokens bounded by 32 and `[H,E]` weights bounded by 4. Complete demand, kept token indices, combine probabilities and survival: exact token-order capacity routing of first-maximum selections over softmax rows that approximate the real softmax of `x @ w`. | Optimized original is FlashMoE **with PR #32** (deterministic router); gate weights `[E,H]`. Trusts `cpu_exp_f32_approx`; top-1 stability against rounding not proved. Composition over generated host wiring; links to reduced/original CUDA are finite bitwise checks. No expert MLP or larger profile. |

The base and optimized roles refine a common **relative algebraic task**.
This is not bitwise equivalence, an IEEE error bound, proof of numerical
stability, or a guarantee of equal selected tokens/experts. Each execution
follows its own permitted numerical observations and selection history.
Satisfaction of the numerical premises is not newly established by replay.

Each project's `KPlay.Review` module (`<project>-program-review`) states
that both roles satisfy the identical role-independent `program_postcondition`
on the same inputs.

## Family contracts

Every family case proves that its Kuiper/Pulse implementation ensures its
family relation in the project's `KPlay.SharedSpec`, for **all** reals that the
inputs approximate. No case in this release has a tightness proof: the five
LUCE family-tightness proofs and five family reviews belonged to the former
migrated LUCE artifacts, which the clean-room LUCE proof replaces. A
determinate but wrong task could still be tight, and a loose relation could
still be correct; reference fidelity remains a separate, reviewed obligation.
An inconsistent trusted theory would also defeat the intended interpretation.

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

`support/mirage-cleanroom` contains the
66 `src/lib` modules that differ from or are absent in Kuiper
`nightly-2026-11-18`; it is the library of the RESOLVE installation that
checked the Mirage proofs. `support/flashmoe-cleanroom` adds the two
`Kuiper.KPlay.CpuExp` modules; with `support/mirage-cleanroom` it is exactly the
library of the RESOLVE installation that checked the FlashMoE proofs.
`support/luce-cleanroom` adds `Luce.Softplus` and `Luce.Softplus.Fast`; the
three overlays together are exactly the 70 differing `src/lib` modules of the
RESOLVE installation that checked the LUCE proofs. `support/foundation`
contains the five changed F*/Pulse source files; it applies only to the
`default` toolchain, which no current case uses.

## Native checks in this release

The complete-program smoke tests use synthetic, non-secret inputs. FlashMoE
checks all four outputs for both a tie/overflow case and a balanced-routing
case (weights in the `[512,8]` layout). Mirage and LUCE check complete token/hidden output shapes, token range,
zero-weight hidden results, and input preservation; LUCE uses an offset-one
(contiguous) BF16 weight view. These are finite diagnostics, not the universal theorem.
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
   - each kept token's combine weight approximates `p / p` of its selected
     probability. This holds on every kept token, with no premise on the
     selected probability; we edited this part of the spec by hand after the
     RESOLVE run. To prove it, the optimized argmax seeds its scan from
     `row[0]`
     ([`patches/flashmoe-argmax-start-row0.patch`](patches/flashmoe-argmax-start-row0.patch),
     a local patch to the original);
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

## LUCE clean-room pipeline proof

The LUCE cases also come from the clean-room RESOLVE proof pipeline; its
agents had no access to the former LUCE proofs. **The optimized original is
the official [Luce-Org/lucebox](https://github.com/Luce-Org/lucebox) megakernel**
at `0f212acacb284cf77f29f4fb4f706b1f4fdc3f88`, `optimizations/megakernel`
(BF16 Qwen3.5-0.8B decode megakernel; includes the merged #761, #804 and
#1026), with the local RoPE patch below. **The base** is the pinned Hugging Face Transformers 5.3.0 Qwen3.5
reference (`Qwen/Qwen3.5-0.8B` revision `2fc0636`). Hardware: H100, `sm_90`.

**Local patch and hand edits.** The optimized original carries one local,
non-upstream patch, [`patches/luce-rope-pytorch-table.patch`](patches/luce-rope-pytorch-table.patch):
the RoPE frequency is `float(position) * ROPE_INV_FREQ[i]` with PyTorch's
float32 `inv_freq` table instead of `float(position) / powf(1e7, 2i/64)`. The
reduced optimized kernels (`embedding`, program) and the optimized `embedding`
proof compute the same product. The model constants in `KPlay.SharedSpec` and
`KPlay.Spec` are pinned to PyTorch's float32 values (`numerics_ok nm` is
`nm == model_numerics`). These specs, that proof and the reduced kernels were
edited by hand after the RESOLVE run, and the base `full_attention_decoder`
proof gained one helper lemma (`cp_probs_at`) to re-check under the new spec. On the four RESOLVE samples the patched
original and the reduced patched program agree bitwise (tokens and hidden
states); this is a test, not part of the proof.

One specification,
[`proofs/luce-shared/KPlay.SharedSpec.fst`](proofs/luce-shared/KPlay.SharedSpec.fst),
states:

1. **Reference math over reals** (`reference/luce/`): embedding, Qwen3.5
   `(1 + w)` RMSNorm, Gated DeltaNet layers (causal depthwise convolution,
   l2-normalized scaled queries/keys, threshold-20 softplus decay, delta-rule
   state update, gated RMSNorm), gated full-attention layers (per-head norms,
   partial RoPE, grouped-query causal softmax), the SwiGLU MLP, the final norm
   and the tied LM head, all from fresh zero state (`ref_hidden`, `ref_logits`).
2. **`program_postcondition`.** Role independent. Under `program_premise`
   (752393024 weights, `1 <= P <= 8`, `1 <= S <= 5`, valid prompt tokens):
   there are `S` valid tokens and `S` hidden rows, and, with the model
   constants pinned to PyTorch's float32 values (`numerics_ok`: eps `1e-6f`,
   DeltaNet query scale `1/sqrt(128)` rounded to f32, and the 32 float32 RoPE
   `inv_freq` values of the reference), for every real weight vector the BF16
   weights approximate and every step `j < S`:
   - hidden row `j` approximates `ref_hidden` at position `P-1+j`, with the
     prompt followed by the previously generated tokens as input;
   - token `j` is an allowed maximum (`allowed_choice`: NaN first, otherwise
     no strictly greater element) of some BF16 or FP32 score row that
     approximates `ref_logits` of that hidden state.
3. **One relation per role and family.** Base `initialize`, `embedding_rope`,
   `recurrent_decoder`, `full_attention_decoder`, `final_norm_projection`,
   `argmax`, `token_history_update`; optimized `initialize`, `embedding`,
   `recurrent_decoder`, `attention_decoder`, `final_projection`, `argmax`,
   `inactive_decode`.

`KPlay.Proof.Program.<Role>.refines` proves that the family relations at every
host call imply `program_postcondition`; each family's Kuiper implementation
proves its relation. `KPlay.Review` (case `luce-program-review`) checks that
both roles, on the same inputs, satisfy this same postcondition. There are no
tightness cases.

**Profile.** All 24 layers (18 Gated DeltaNet + 6 gated full attention),
prompt lengths 1..8, decode steps {1, 5} (the premise admits 1..5), cache
extent `P+S-1 <= 12` positions. Both reduced programs pad the run to the
12-position extent: after the last computed position, `13-P-S` iterations run
`argmax` and `token_history_update` (base) or `inactive_decode` (optimized). Family relations hold
vacuously for integer arguments outside this domain (for example a cache
extent above 12).

**FMA contraction.** Every Kuiper implementation is compiled with
`--fmad=false` (`validation.json`: `cuda_flags ["--fmad=false"]`,
`preserve_float_rounding`), so the compiler never contracts the extracted
code. The reduced optimized attention decoder's RoPE line
`sub32(mul32(x0, cos), mul32(x1, sin))` uses PTX `mul.ftz`/`sub.ftz` without
`.rn`; under nvcc's default `--fmad=true`, the build in which RESOLVE recorded
the reduced kernels, ptxas contracts it into `fma(x0, cos, -(x1*sin))`. The
optimized `attention_decoder` implementation models this contraction
explicitly as `F32.fma_rn_ftz u0 c (neg (mul_rn_ftz u1 s))` and matched the
recorded traces bitwise. The family's `M.json` states this numeric model
("nvcc --fmad=false, so no compiler contraction ... the RoPE rotation is
written explicitly ... as fma.rn.ftz(x0, c, -(x1*s))"); its `--fmad=false`
refers to the Kuiper build. `luce_check.py` likewise builds the reduced
kernels with default flags and the extracted kernels with `--fmad=false`.

The guarantee is relational (`%~`), not an error bound. RESOLVE's testing
found base and optimized outputs **not** bitwise equal on any of its 4
samples (unlike FlashMoE), so the two roles agree only through this common
specification; they may choose different tokens where scores are close.

**Trusted assumptions.** All proofs are F*. In addition to F*, Pulse and Z3,
the guarantee trusts every item of `proof/program/result.json`
`trusted_assumptions`: the accepted specification's `M.json` trust list
(items 1-4) followed by the families' `M.json` trust items, verbatim:

1. Luce.Softplus.softplus20_approx: LUCE-local numerical assumption that the base threshold-20 softplus (a > 20 ? a : log1pf(expf(a))) approximates Luce.Softplus.model (= KPlay.SharedSpec.softplus).
2. Luce.Softplus.Fast.softplus20: LUCE-local assumption for the optimized ex2/lg2 softplus composite against the same model.
3. Kuiper.Float.Casts.Raw: the two trusted BF16/F32 widening laws (via bf16_to_f32_kind, bf16_to_f32_lte).
4. Kuiper library r_* ideal laws of the floating operations (Kuiper.Float.Realops, Kuiper.Float32.FastMath) and the exact-literal certificates of Kuiper.Float32.ConstantFold.
5. CUDA completion and typed tensor allocation/access by the fixed RESOLVE wrapper (synchronous-stable-buffers-v1)
6. Kuiper/Pulse extraction to CUDA and the Kuiper assumption that size_t values fit in 32 bits
7. Kuiper library laws: exactness and order preservation of the BF16->FP32 widening (bf16_to_f32_kind, bf16_to_f32_lte), FP32 lt ordering/transitivity, and the IEEE eq/bits class laws
8. The wrapper checks contiguity and the logits shape [248320]
9. Kuiper library approximation laws for of_int, FP32 mul, cosf/sinf and FP32->BF16 conversion
10. The wrapper passes numel(input0..2) as the buffer lengths and checks contiguity and the table shape [248320,1024]
11. Kuiper.Float32 add/mul/rsqrt real-approximation contracts and their extraction to __fadd_rn/__fmul_rn/rsqrtf (compiled with --fmad=false)
12. Kuiper.Float32.ConstantFold literal certificates for the f32 literals 1e-6f and 1/1024
13. Kuiper.Float.Casts bf16 <-> f32 conversion contracts (fcast_approx)
14. Kuiper.TensorCore.MMA m16n8k16 model (emma, emma_approx_lemma) (fill, mma_sync from global-memory tiles, store) and its extraction to PTX mma.sync
15. Kuiper launch_kernel_1, launch_sync (kernel_desc_1_n), gpu_array_alloc/free and the global-array send-across instance (on/on_pure_elim)
16. Kuiper.SizeT assumption that size_t values fit in 32 bits (size_t extracted as uint32_t)
17. Host wrapper glue: contiguous typed storage, shape checks, fresh output allocation, CUDA completion
18. F*/Pulse/Karamel toolchain
19. Kuiper tensor-core MMA model (Kuiper.TensorCore.MMA mma_sync / emma and emma_approx_lemma)
20. Kuiper float32/bfloat16 operation models and their approximation laws (add, mul, div, neg, sub, fma_rn, conversions)
21. Kuiper.Float32 rsqrt and exp approximation laws (rsqrtf, expf)
22. Kuiper/Pulse GPU launch, memcpy and memory model; F* and Pulse extraction to CUDA
23. CUDA kernel launch and completion (launch_kernel_1, single thread)
24. typed PyTorch tensor allocation and data_ptr access in the wrapper
25. Kuiper extraction of FStar.SizeT to 32-bit size and BF16/FP32 zero constants
26. CUDA completion and typed tensor allocation/access by the fixed RESOLVE wrapper (synchronous-stable-buffers-v1) and the wrapper.json argument/output binding
27. Kuiper floating-point approximation laws for native F32/BF16 operations (Kuiper.Float.Realops, Kuiper.Float32.FastMath.fma_rn) and the two trusted cast laws in Kuiper.Float.Casts.Raw
28. Kuiper.TensorCore.MMA emma_approx_lemma (tensor-core m16n8k16 BF16 MMA approximates the real dot product)
29. Kuiper.Float32.ConstantFold literal_approx / Kuiper.Float32.Literals exact-normal literal values
30. Luce.Softplus softplus20_approx (thresholded softplus approximates the real softplus)
31. The wrapper passes numel(input0) as S and checks input shapes [S], [S,1024], [1024], [] and contiguity
32. Kuiper/Pulse verified extraction to CUDA and the KaRaMeL toolchain
33. Kuiper native binary32 comparison axioms (Kuiper.Float32 / Kuiper.Floating laws)
34. Fixed host wrapper glue generated from wrapper.json (allocation, shape checks, argument passing)
35. Callers supply 0 <= decision < S; out-of-range decisions are outside the relation premise
36. Kuiper/Pulse toolchain, extraction to CUDA, and the Kuiper float operation contracts (ftz ops, approx rsqrt/rcp, __expf, bf16 casts)
37. The generated host wrapper (wrapper.json): pointer binding, contiguity/shape checks and passing numel(weights) as uint32 length
38. The frozen KPlay.SharedSpec relation as the statement of correctness
39. One assert_norm certificate establishing the decimal literal of eps (1e-6 as float32) as an exact float
40. Kuiper.Float32 GPU intrinsic contracts (mul_rn_ftz, rcp_approx_ftz, exp2_approx_ftz) and FastMath cos/sin, flog, div, of_int real-approximation contracts and their extraction to PTX/CUDA
41. Kuiper.Float.Casts bf16->f32 widening contract (fcast_approx)
42. Kuiper launch_kernel_1 and global-array send-across instance (Kuiper.Array.Core.is_send_pts_to)
43. Host wrapper glue: contiguous typed storage, table shape check, fresh output allocation, CUDA completion
44. Kuiper/Pulse array model, launch_kernel_1 / memcpy_device_to_device semantics and Karamel extraction to CUDA
45. Kuiper float32/bfloat16 primitive models (fma/add/mul ftz, FastMath divide as __fdividef, rsqrt approximation, casts) and their approximation lemmas
46. CUDA completion and typed tensor allocation/access in the fixed host wrapper
47. Float32 literal parsing of the epsilon constant via FB.of_literal
48. Kuiper.Array.Core.memcpy_device_to_device specification and its extraction to cudaMemcpy(cudaMemcpyDeviceToDevice)
49. Local sized instance for FStar.Int64.t (8 bytes, matches sizeof(int64_t))
50. Kuiper/Pulse array model, launch_kernel_1 semantics and Karamel extraction to CUDA
51. Kuiper zero constants for float32 and bfloat16 extract to positive-zero literals
52. Luce.Softplus.Fast.softplus20: LUCE-local assumed approximation law of the thresholded fast softplus composite (Luce.Softplus.model).
53. Kuiper float32 primitive contracts (add/mul/fma round-to-nearest flush-to-zero, approximate exp2/rcp/rsqrt, fast-math exp/log) and literal interpretation (Kuiper.Float32.ConstantFold.literal_approx).
54. Kuiper bfloat16/float32 cast approximation laws (Kuiper.Float.Casts).

The two `Luce.Softplus*` laws are new, interface-only `assume`s in
[`support/luce-cleanroom/`](support/luce-cleanroom/); the cast, literal and
bit laws are those of `support/mirage-cleanroom` (see the
[README](README.md#trusted-assumptions-mirage)). Outside F* the guarantee also
trusts the generated host wiring `KPlay.Proof.Wiring.<Role>` (a translation of
`proofs/luce-program/<role>/plan.json`), and the finite bitwise links from the
extracted family kernels to the reduced CUDA and from the reduced programs to
the original sources (RESOLVE's same-role bitwise translation check,
`universal_original_source_equivalence: false`).

**Testing versus proof.** RESOLVE tested 4 input samples (categories V0, V1,
V7, V8; `(P,S)` = (3,5), (1,1), (8,5), (2,1)) with the actual checkpoint
converted to BF16. The base was stable (5 runs, one distinct result per
sample). DTester ran each sample 8 times: 4 baseline runs plus 2 runs under
each of two perturbations (unstructured scoped delays on global and on shared
memory, 5000 cycles, p=0.8, no prepass). All runs were **comparable (8/8 per
sample)** with no mismatches, crashes or infrastructure errors ("no result
nondeterminism detected"). The final RESOLVE check approved the candidate
(`dtester_approved = 1`, `kuiper_approved = 1`; "DTester and Kuiper approved
the candidate"). Testing covers only these inputs and schedules; the proof
covers every input in the stated domain for the reduced programs, under the
trusted assumptions above.
