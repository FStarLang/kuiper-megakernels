# Kuiper megakernel proof artifacts

Standalone artifacts from the RESOLVE megakernel case studies: **six complete
program refinements, 30 family refinements and three shared-specification
checks**, plus three checked base/optimized program reviews and three typed
specifications (`proofs/<project>-spec`) proved equivalent to the flat ones.
Each typed specification is also checked to **define a function**: one answer
per input, up to argmax ties (cases `<project>-tightness`; see
[CLAIMS.md](CLAIMS.md#tightness-each-spec-defines-a-function)).
No RESOLVE installation, agent, DTester,
model weights, private checkout, or historical checked cache is required.

This standalone repository was migrated from `FStarLang/kuiper`, branch
`examples/megakernel-proof-artifacts`, commit
`756939b576b262523c8a45e92316ea97b895aa1f`.
The support-library and CUDA payloads and historical timing records are
unchanged. Proofs are now grouped by kernel with one shared specification,
as described below. See [NOTICE.md](NOTICE.md) for attribution and licensing
boundaries.
On this branch the Mirage, FlashMoE and LUCE artifacts are all replaced by
proofs from the clean-room RESOLVE proof pipeline (sections below).

## Mirage: clean-room RESOLVE pipeline proof

On this branch the Mirage artifacts are replaced by the proof produced by the
clean-room RESOLVE proof pipeline
([`src/KPlay/proof_pipeline`](https://github.com/msr-central/resolve/tree/main/src/KPlay/proof_pipeline)).
The agents saw only the reduced CUDA, the PyTorch reference and the Kuiper
library. They had no access to earlier Mirage proofs, including the former
Mirage artifacts here. The selected Qwen3-0.6B kernel is proved for both roles,
and the result then passed RESOLVE's normal final check. The kernel runs one
forward pass over a seven-token prompt and appends the greedy (argmax) next
token; it returns the eight tokens and the seven final hidden rows.

| Case | What F* checks |
| --- | --- |
| `mirage-shared` | [`KPlay.SharedSpec`](proofs/mirage-shared/KPlay.SharedSpec.fst): exact real reference math, the role-independent `program_postcondition`, and one quantified relation per role and family |
| `mirage-{argmax,decoder,embedding,projection}-{base,optimized}` | Each Kuiper/Pulse implementation in `KPlay.Functional` ensures its family relation: for **all** reals that the inputs approximate, the outputs approximate the reference of those reals |
| `mirage-program-{base,optimized}` | `KPlay.Proof.Program.<Role>.refines`: the family relations at every host call of the reduced program imply `program_postcondition`. That postcondition covers the prompt copied in front of the token, the final hidden rows approximating `final_hidden`, and the token being an allowed maximum of a score row that approximates `next_token_logits` |
| `mirage-program-review` | [`KPlay.Review`](proofs/mirage-program/KPlay.Review.fsti): the short statement to read. On the same inputs, **both** roles' final state satisfies the same `program_postcondition` |

**Reading `%~`.** `x %~ r` relates a machine value `x` (BF16/F32) to a real
number `r` in Kuiper's abstract floating-point model. The specification is
written over exact real arithmetic, and `%~` says the computed value stands
for that real result with rounding abstracted away. It is a relation, not a
numeric error bound: the proofs do not say how far `x` is from `r`.

**Typed specification.** [`proofs/mirage-spec`](proofs/mirage-spec) restates
the guarantee for reading. [`KPlay.Spec`](proofs/mirage-spec/KPlay.Spec.fst)
models the reference over named dimensions, typed vectors and matrices, and
records for weights, inputs and outputs. It has one real-valued equation per
reference operation, each citing its source line, and a typed `postcondition`
that is structural, numerical and selection. `KPlay.Spec.Bridge` proves that
`program_postcondition` is equivalent to that `postcondition` on the
program's buffer lengths, read through layout-only views
(`program_postcondition_iff`). [`KPlay.Spec.Review`](proofs/mirage-spec/KPlay.Spec.Review.fsti)
then states that both roles meet the typed `postcondition`. Check it with
`python3 replay.py verify --case mirage-spec --work <work> --jobs 8`; this also
re-checks the program proofs.

### Mirage files

| Path | Contents |
| --- | --- |
| `reference/mirage/` | The PyTorch reference that `KPlay.SharedSpec` cites as `inputs/reference/*`: `base.py`, the checkpoint ABI `model_abi.py`, `model_config.json`, and the unmodified Transformers 4.57.1 Qwen3 sources it runs. [`provenance.json`](reference/mirage/provenance.json) lists their hashes and origins |
| `kernels/mirage-program-<role>.cu` | The reduced complete program for each role (the CUDA whose behavior is proved) |
| `kernels/mirage-<family>-<role>.cu` | The reduced family kernels, one per host call kind |
| `kernels/mirage-extracted/<family>-<role>/` | The CUDA that Kuiper extracted from each verified implementation (`KPlay_Functional.{cu,h}`) plus a small PyTorch binding (`binding.cu`). `kernels/mirage-extracted/include/` holds the three RESOLVE install headers that differ from or are absent in the Kuiper release |
| `proofs/mirage-program/<role>/plan.json` | The recorded host call plan of the reduced program: each family call, its arguments (prompt, weight views, earlier outputs) and the program outputs. `KPlay.Proof.Wiring.<Role>` is the translation of this plan, and `accepted_cuda_sha256` binds it to the program CUDA |

### Observing the reduced kernels (GPU)

[`mirage_check.py`](mirage_check.py) makes the bitwise links concrete. For each
role it runs the complete reduced program and then replays `plan.json` call by
call. Each family call runs both the reduced family kernel and the
Kuiper-extracted kernel on the same arguments, and the two must match bitwise.
Composing the reduced family outputs per the plan must then reproduce the
complete program's outputs bitwise. It downloads the pinned Kuiper release
headers, and compiles the extracted code with `--fmad=false` as RESOLVE did
(the extracted code is the F* model, so the compiler must not contract it).

```bash
# GPU environment as in "Inspect and run the reduced kernels" below.
.venv-gpu/bin/python mirage_check.py                # synthetic seeded weights
# Real Qwen3-0.6B weights plus the Transformers reference:
.venv-gpu/bin/pip install transformers==4.57.1 safetensors
CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python mirage_check.py \
    --checkpoint model.safetensors --reference \
    --prompt 785,3974,13876,38835,34208,916,279   # "The quick brown fox jumps over the"
```

[`benchmarks/mirage-plan-replay.json`](benchmarks/mirage-plan-replay.json)
records both runs on an H100. All 31 base and 31 optimized calls matched
bitwise, and both compositions reproduced their programs. With the real
checkpoint both roles and the reference chose `" lazy"`. The base hidden rows
were bitwise equal to the reference's, and the optimized rows differed by at
most 3.0 (mean 0.13; values up to 60.75). These are observations on one input,
not proofs.

These cases use profile `mirage-cleanroom`. That profile is pinned to Kuiper
`nightly-2026-11-18` (F* `049c07a`, Z3 5.3.0) plus the RESOLVE library overlay in
[`support/mirage-cleanroom/`](support/mirage-cleanroom/). The overlay holds the
66 `src/lib` modules that differ from or are absent in the stock package.

> [!WARNING]
> ### Trusted assumptions (Mirage)
>
> The Mirage proofs rely on the following **unproved** laws, in addition to
> Kuiper's own IEEE/CUDA model. Each one is new compared with stock
> `nightly-2026-11-18`; the diff is reproducible from `support/mirage-cleanroom/`.
>
> | Law (interface-only `val`) | Source | Statement |
> | --- | --- | --- |
> | `cast_bf16_to_f32_kind` (`Kuiper.Float.Casts.Raw`) | RESOLVE | BF16→F32 widening preserves the floating class (Finite / Infinite / NaN). Exact widening per PTX `cvt` and IEEE 754-2019 §5.4.2, §6.2; RESOLVE tested all 65,536 BF16 values on the GPU |
> | `cast_bf16_to_f32_lte` (`Kuiper.Float.Casts.Raw`) | RESOLVE | BF16→F32 widening preserves native `<=`; tested on all 65,536² pairs against `__hle` |
> | `literal_approx` (`Kuiper.Float32.ConstantFold`) | RESOLVE | A source literal certified as an exactly representable normal F32 value approximates its parsed real value |
> | `reciprocal_literal_bit_eq` (`Kuiper.Float32.ConstantFold`) | RESOLVE | A certified correctly rounded reciprocal literal is bit-equal to `1 / input` |
> | `bits_spec` (`Kuiper.{BFloat16,Float16,Float32,Float64}.Raw`) | Kuiper PR #181 (open) | Equal bit patterns have equal floating class and zero-ness |
>
> The remaining library changes in the overlay weaken stock laws or narrow their
> scope; none strengthens a stock law:
> - Kuiper PR #181: `bit_eq` no longer implies `==`.
> - Kuiper PR #135: the tensor-core `emma_approx_lemma` is specialized to BF16 16×16×8.
>
> Outside F*, the guarantee also trusts:
> - the generated host wiring `KPlay.Proof.Wiring.<Role>`, a mechanical
>   translation of the reduced program's recorded host call plan
>   (`proofs/mirage-program/<role>/plan.json`);
> - the finite bitwise replays linking the extracted family kernels to the
>   reduced CUDA, and the reduced program to the original Mirage megakernel.
>
> The guarantee is relational (`%~`), not an error bound or bitwise equality
> between roles. Optimized and base may choose different maximal tokens on ties.

## FlashMoE: clean-room RESOLVE pipeline proof

The FlashMoE artifacts are likewise replaced by a proof from the clean-room
RESOLVE proof pipeline. Its agents saw only the reduced CUDA, the PyTorch
reference and the Kuiper library, not the former FlashMoE proofs. The subject
is FlashMoE's capacity-pruned top-1 **router** in the smoke profile: BF16
tokens `[256,512]` and router weights `[512,8]`, 8 experts, capacity 32. It
returns the expert demand `[8]` (I32), the kept token indices `[8,32]` (I64),
their combine probabilities `[8,32]` (F32) and the survival mask `[256,8]`
(I32). Both roles are proved, and the result passed RESOLVE's normal final
check (DTester and Kuiper both approved). The expert MLPs are not covered.

> [!IMPORTANT]
> ### What the optimized original is
>
> The optimized original is **FlashMoE with
> [PR #32](https://github.com/osayamenja/FlashMoE/pull/32) applied**
> (deterministic router; open at the time of writing):
> `tyler-utah/FlashMoE@96ad271` = upstream `osayamenja/FlashMoE@9cc0c32` + PR #32.
> Upstream reserves capacity slots in atomic arrival order, so its routing
> outputs are not deterministic. PR #32 assigns capacity in input-token order.
> The FlashMoE gate takes its weights as **`[E,H]`**; the case harness passes
> `router_weights.t().contiguous()`. The reference and both reduced programs
> take the logical `[H,E] = [512,8]` layout, and the reduced optimized program
> folds the transpose into its indexing
> ([`reference/flashmoe/LAYOUT.md`](reference/flashmoe/LAYOUT.md)).
> See `sources.json`.

Base and optimized decompose the router into **different families**, so
family directories exist only for the roles that use them:

| Role | Families (host calls per run) |
| --- | --- |
| base | `projection_softmax` (1), then per token `argmax` and `capacity_routing` with I64 prefix counts (256 each), then `expert_demand` (1, I64→I32) |
| optimized | tensor-core `projection` (1), then per token `softmax`, `argmax` and `routing` with I32 counts (256 each) |

| Case | What F* checks |
| --- | --- |
| `flashmoe-shared` | [`KPlay.SharedSpec`](proofs/flashmoe-shared/KPlay.SharedSpec.fst): exact real reference math (scores `x @ w`, softmax, first-maximum top-1, `p / p` combine weight, token-order capacity), the role-independent `program_postcondition`, and one relation per role and family |
| `flashmoe-{projection_softmax,argmax,capacity_routing,expert_demand}-base`, `flashmoe-{projection,softmax,argmax,routing}-optimized` | Each Kuiper/Pulse implementation in `KPlay.Functional` ensures its family relation, for **all** reals that the inputs approximate |
| `flashmoe-program-{base,optimized}` | `KPlay.Proof.Program.<Role>.refines`: the family relations at every host call imply `program_postcondition`. There exist softmax rows approximating the real softmax of `x @ w`; each token's expert is an allowed first maximum of its row; every kept token's combine weight approximates `p / p`; and demand, indices, probabilities and survival are **exactly** the token-order capacity routing of those selections |
| `flashmoe-program-review` | [`KPlay.Review`](proofs/flashmoe-program/KPlay.Review.fsti): on the same inputs, **both** roles' outputs satisfy the same `program_postcondition` |

Selection is over the native rows actually computed. The two roles may
therefore pick different experts where real-valued scores are close, and their
outputs need not be equal. `%~` is read as for Mirage above.

The combine weight is constrained on every kept token, with no premise on the
selected probability. We edited this part of the spec by hand after the
RESOLVE run, which had required the selected probability to be above −infinity.
To prove it, the optimized argmax now seeds its scan from `row[0]` instead of
−infinity, as base does. This is a local patch to the original,
[`patches/flashmoe-argmax-start-row0.patch`](patches/flashmoe-argmax-start-row0.patch),
and it is also applied to the reduced kernels.

Review case `flashmoe-spec` ([`proofs/flashmoe-spec/`](proofs/flashmoe-spec/))
restates the same guarantee as a typed spec that people can read.
[`KPlay.Spec`](proofs/flashmoe-spec/KPlay.Spec.fst) has named dimensions,
records for `inputs`/`outputs`, one real equation per reference operation, and
`postcondition input output` = structural ∧ numerical ∧ selection over a
routing run. It never mentions flat buffers or `KPlay.SharedSpec`.
`KPlay.Spec.Bridge` gives the row-major views `inputs_of`/`outputs_of` and
proves `program_postcondition_iff`: for inputs of the right length, the flat
`program_postcondition` holds iff the output lengths and the typed
`postcondition` both hold. `KPlay.Spec.Review` concludes the typed
`postcondition` for both roles. To check:
`python3 replay.py verify --case flashmoe-spec --work <work> --jobs 8`.

These cases use profile `flashmoe-cleanroom`, pinned to the same Kuiper
`nightly-2026-11-18` release as Mirage. RESOLVE's FlashMoE installation was
built from that release (`archive_sha256` matches the pin); its library
overlay is `support/mirage-cleanroom/` plus `Kuiper.KPlay.CpuExp`.

> [!WARNING]
> ### Trusted assumptions (FlashMoE)
>
> All proofs are F*. In addition to Kuiper's IEEE/CUDA model and the shared
> overlay laws listed under [Mirage](#trusted-assumptions-mirage) (FlashMoE
> uses the same library plus one module), the FlashMoE guarantee relies on the
> following, recorded in RESOLVE's accepted specification (`M.json`, `trust`):
>
> - **`cpu_exp_f32_approx`** (`Kuiper.KPlay.CpuExp`, new,
>   [`support/flashmoe-cleanroom/`](support/flashmoe-cleanroom/)): **TRUSTED,
>   unproved**. `x %~ r ==> cpu_exp_f32 x %~ exp r`. `cpu_exp_f32` is an
>   opaque Kuiper routine (`include/kplay_cpu_exp.cuh`), the scalar CUDA form
>   of the AVX512 SLEEF `expf` used by PyTorch's CPU softmax, returning +0.0
>   below −104. The base kernel's `softmax_exp` is source-identical to it
>   (bit-identical), and `base_projection_softmax_relation` is proved with this
>   law. It plays the role that Kuiper's trusted CUDA `expf` law plays for optimized.
> - The base `int32_t(prefix[e])` demand conversion is exact for
>   `0 <= prefix[e] <= 256`; this premise is discharged in the proof.
> - **Not proved:** stability of the top-1 choice against rounding, i.e. that
>   the native argmax equals the argmax of the real reference softmax. The
>   postcondition selects over the native rows, which approximate the real softmax.
> - Trusted Kuiper library laws (BF16→F32 cast, literal approximation, float
>   bit specifications, fast-math intrinsics). `%~` is qualitative; there is no error bound.
>
> Outside F*, the guarantee also trusts:
> - the generated host wiring `KPlay.Proof.Wiring.<Role>`, a mechanical
>   translation of `proofs/flashmoe-program/<role>/plan.json`;
> - the finite bitwise links from the extracted family kernels to the reduced
>   CUDA, and from the reduced programs to the original sources. The latter is
>   RESOLVE's same-role bitwise translation check on 8 samples
>   (`universal_original_source_equivalence: false`), not an equivalence theorem.

**Testing versus proof.** RESOLVE tested on **8 input samples**, one per
category V0–V7 of the case's input generator. On each, the candidate was
bit-equivalent to the original program. DTester then ran each sample 8 times:
4 baseline runs plus 2 runs under each of two perturbations (unstructured
scoped delays on global and on shared memory, 5000 cycles, p=0.8). All runs
were **comparable (8/8 per sample)**, with no result mismatches, races or crashes.
DTester reported 0 scope-instrumented sites, and there was no count/policy
prepass and no structured scope, so this is weak evidence about schedules.
Testing covers only these inputs and schedules. The proof covers **every**
input in the stated domain (finite BF16, |tokens| ≤ 32, |weights| ≤ 4) for
the reduced programs, under the trusted assumptions above. Neither covers the
full-size FlashMoE profile or the expert computation.

### FlashMoE files

| Path | Contents |
| --- | --- |
| `reference/flashmoe/` | The PyTorch reference that `KPlay.SharedSpec` cites: `base.py` (`reference_route`), `input-domain.txt` and `LAYOUT.md`. [`provenance.json`](reference/flashmoe/provenance.json) lists their hashes. `base.py` (unmodified) imports RESOLVE's case adapter, which is not published; `E8_mega_kernel_h100/` is a standalone shim written from `input-domain.txt` (profiles, input checks and this repository's own spread-expert `make_inputs`). Run with `PYTHONPATH=reference/flashmoe` |
| `kernels/flashmoe-program-<role>.cu` | The reduced complete program for each role |
| `kernels/flashmoe-<family>-<role>.cu` | The reduced family kernels of that role |
| `kernels/flashmoe-extracted/<family>-<role>/` | Kuiper's extraction of each verified implementation (`KPlay_Functional.{cu,h}`) plus a PyTorch binding (`binding.cu`). `include/` holds the four RESOLVE install headers that differ from or are absent in the release, including `kplay_cpu_exp.cuh` |
| `proofs/flashmoe-program/<role>/plan.json` | The recorded host call plan (8 samples) behind `KPlay.Proof.Wiring.<Role>`; `accepted_cuda_sha256` binds it to the program CUDA |
| `support/flashmoe-cleanroom/` | `Kuiper.KPlay.CpuExp.{fst,fsti}`. Profile `flashmoe-cleanroom` = `support/mirage-cleanroom` + this directory; together they are exactly the 68 `src/lib` modules of RESOLVE's install that differ from Kuiper `nightly-2026-11-18` |

### Observing the FlashMoE kernels (GPU)

[`flashmoe_check.py`](flashmoe_check.py) does for FlashMoE what
`mirage_check.py` does for Mirage. It replays each role's plan call by call,
running the reduced and the extracted family kernel on the same arguments and
requiring bitwise equality. It then requires that the composition reproduce the
complete program bitwise. Finally it compares the program outputs with
`reference_route`.

```bash
.venv-gpu/bin/python flashmoe_check.py --seed 0 --seed 1 --seed 2
```

Results from 3 seeds on one H100 PCIe are in
[benchmarks/flashmoe-plan-replay.json](benchmarks/flashmoe-plan-replay.json).
For each seed, every recorded plan call was replayed against the extracted
Kuiper kernel. Base has 514 calls (projection_softmax 1, argmax 256,
capacity_routing 256, expert_demand 1) and optimized has 769 (projection 1,
softmax 256, argmax 256, routing 256). Every call was bitwise equal. The
composed extracted calls reproduced each complete program's outputs exactly.
Both roles also matched the reference: same expert demand, token indices and
survival, and a maximum absolute probability difference of 0.0. The default
generator draws dyadic tokens and router weights whose dot products are exact
in FP32, so tokens spread over the experts and some experts overflow the
capacity of 32. Per-seed expert demand (survivors) was:

| Seed | Expert demand | Survivors (capacity 32) |
| --- | --- | --- |
| 0 | 41, 24, 35, 30, 27, 39, 28, 32 | 32, 24, 32, 30, 27, 32, 28, 32 |
| 1 | 28, 30, 39, 47, 30, 20, 36, 26 | 28, 30, 32, 32, 30, 20, 32, 26 |
| 2 | 26, 30, 44, 35, 30, 34, 23, 34 | 26, 30, 32, 32, 30, 32, 23, 32 |

The generator is `make_inputs` of the reference shim. RESOLVE's DTester input
categories V0-V7 are not published. These are finite observations, not proofs.

## LUCE: clean-room RESOLVE pipeline proof

The LUCE artifacts are likewise replaced by a proof from the clean-room
RESOLVE proof pipeline. Its agents saw only the reduced CUDA, the PyTorch
reference and the Kuiper library, not the former LUCE proofs. The subject is
greedy BF16 decode of Qwen3.5-0.8B: each call takes the packed BF16 weights
`[752393024]`, prompt tokens `[P]` and a decode-step count `S`, starts from zero
state, and returns `S` generated tokens (I64) and the `S` pre-final-norm hidden
rows `[S,1024]` (BF16). Both roles are proved, and the result passed RESOLVE's
normal final check (DTester and Kuiper both approved).

> [!IMPORTANT]
> ### What the originals are
>
> - **Optimized:** the official
>   [Luce-Org/lucebox](https://github.com/Luce-Org/lucebox) megakernel at
>   `0f212acacb284cf77f29f4fb4f706b1f4fdc3f88`, `optimizations/megakernel`
>   (BF16 Qwen3.5-0.8B decode megakernel; includes the merged #761, #804 and
>   #1026), decode path only, with one local (not upstream) patch,
>   [`patches/luce-rope-pytorch-table.patch`](patches/luce-rope-pytorch-table.patch):
>   RoPE frequencies are `position * inv_freq[i]` with PyTorch's float32
>   `inv_freq` table instead of `position / powf(1e7, 2i/64)`.
> - **Base:** the pinned Hugging Face Transformers **5.3.0** Qwen3.5 reference
>   (`Qwen/Qwen3.5-0.8B` revision `2fc0636`, BF16, eager attention).
> - **Hardware:** H100, `sm_90`.
> - **Profile:** all 24 layers (18 Gated DeltaNet + 6 gated full attention),
>   prompt lengths 1..8, decode steps {1, 5}; cache extent `P+S-1 <= 12`
>   positions. The relations hold **vacuously** for integer arguments outside
>   this domain (for example a cache extent above 12).
>
> See `sources.json` and [`reference/luce/`](reference/luce/).

| Role | Families (host calls for `P=3, S=5`) |
| --- | --- |
| base | `initialize` (1); per position `embedding_rope` (7), 18 `recurrent_decoder` (126) and 6 `full_attention_decoder` (42); per generated position `final_norm_projection` (7); `argmax` and `token_history_update` (12 each, including padding to 12 positions) |
| optimized | `initialize` (1); per position `embedding` (7), 18 `recurrent_decoder` (126) and 6 `attention_decoder` (42); per decode step `final_projection` and `argmax` (5 each); `inactive_decode` for the padding positions (5) |

| Case | What F* checks |
| --- | --- |
| `luce-shared` | [`KPlay.SharedSpec`](proofs/luce-shared/KPlay.SharedSpec.fst): exact real reference math (embedding, `(1 + w)` RMSNorm, Gated DeltaNet with threshold-20 softplus, gated attention with partial RoPE, SwiGLU MLP, final norm and tied LM head from zero state), the role-independent `program_postcondition`, and one relation per role and family |
| `luce-<family>-{base,optimized}` (14 cases) | Each Kuiper/Pulse implementation in `KPlay.Functional` ensures its family relation, for **all** reals that the inputs approximate |
| `luce-program-{base,optimized}` | `KPlay.Proof.Program.<Role>.refines`: the family relations at every host call imply `program_postcondition`. For every step `j < S`, hidden row `j` approximates the reference residual at position `P-1+j` (prompt, then the generated tokens, as input), and token `j` is an allowed maximum of a BF16 or FP32 score row approximating the reference logits there. Model constants (eps, DeltaNet query scale, RoPE inverse frequencies) are pinned to PyTorch's float32 values |
| `luce-program-review` | [`KPlay.Review`](proofs/luce-program/KPlay.Review.fsti): on the same inputs, **both** roles' outputs satisfy the same `program_postcondition` |

**Typed specification (`luce-spec`).** [`proofs/luce-spec/KPlay.Spec.fst`](proofs/luce-spec/KPlay.Spec.fst)
restates the guarantee over named dimensions and weight/input/output records, with one real
equation per reference operation (citing `reference/luce/`) and
`postcondition input output = structural /\ numerical /\ selection`. It does not mention flat
offsets. [`KPlay.Spec.Bridge`](proofs/luce-spec/KPlay.Spec.Bridge.fst) gives the layout-only views
`inputs_of`/`outputs_of`. It proves `program_postcondition_iff`: for 752393024 packed weights and
1024 hidden values per generated token, `program_postcondition` holds exactly when the typed
`postcondition` holds. [`KPlay.Spec.Review`](proofs/luce-spec/KPlay.Spec.Review.fsti) concludes
the typed `postcondition` for both roles. Check with
`python3 replay.py verify --case luce-spec --jobs 8`. This case re-verifies the program proofs.

**Constants and hand edits.** The model constants (eps, DeltaNet query scale,
RoPE inverse frequencies) are PyTorch's float32 values in both
`KPlay.SharedSpec` and `KPlay.Spec` (`model_numerics`, the only constants). The
specs, the optimized `embedding` proof and the reduced optimized kernels were
edited by hand after the RESOLVE run to match the patched original; the base
`full_attention_decoder` proof gained one helper lemma to re-check.
On the four RESOLVE samples the patched original and the reduced patched
program produce bitwise-identical tokens and hidden states. The patch changes
the hidden states on the two multi-token samples (tokens unchanged), and
the outputs remain not bitwise equal to PyTorch.

RESOLVE's testing found base and optimized outputs **not bitwise equal** on
any sample, so the roles agree only through this common specification. They
may choose different tokens where scores are close. `%~` is read as for Mirage.

**FMA contraction.** The reduced optimized attention decoder computes the
first RoPE half as `sub32(mul32(x0, cos), mul32(x1, sin))` with PTX
`mul.ftz`/`sub.ftz` (no `.rn`). Under nvcc's default `--fmad=true`, in which
RESOLVE recorded the reduced kernels, ptxas contracts this into
`fma(x0, cos, -(x1*sin))`. The optimized `attention_decoder` implementation
models the contraction **explicitly** (`F32.fma_rn_ftz u0 c (neg (mul_rn_ftz u1 s))`).
Like every Kuiper implementation here it is compiled with `--fmad=false`
(`validation.json`; the family's `M.json` numeric model says the same), so the
extracted code is never contracted, and it matched the recorded traces bitwise.

These cases use profile `luce-cleanroom` =
`support/mirage-cleanroom` + `support/flashmoe-cleanroom` +
[`support/luce-cleanroom/`](support/luce-cleanroom/) (`Luce.Softplus`,
`Luce.Softplus.Fast`), pinned to the same Kuiper `nightly-2026-11-18` release.
Together they are exactly the 70 `src/lib` modules of RESOLVE's LUCE
installation that differ from the stock package (`archive_sha256` matches the pin).

> [!WARNING]
> ### Trusted assumptions (LUCE)
>
> All proofs are F*. RESOLVE's `proof/program/result.json` lists 54
> `trusted_assumptions`; [CLAIMS.md](CLAIMS.md#luce-clean-room-pipeline-proof)
> reproduces them all verbatim. In summary:
>
> - **`Luce.Softplus.softplus20_approx`** (new, **TRUSTED, unproved**): the
>   base threshold-20 softplus `a > 20 ? a : log1pf(expf(a))` approximates
>   `Luce.Softplus.model` (= `KPlay.SharedSpec.softplus`).
> - **`Luce.Softplus.Fast.softplus20`** (new, **TRUSTED, unproved**): the same
>   for the optimized `ex2`/`lg2` softplus composite.
> - The two BF16→F32 widening laws (`Kuiper.Float.Casts.Raw`
>   `bf16_to_f32_kind`, `bf16_to_f32_lte`), the Kuiper `r_*` ideal laws of the
>   floating operations (`Kuiper.Float.Realops`, `Kuiper.Float32.FastMath`),
>   and the `Kuiper.Float32.ConstantFold` exact-literal certificates (e.g.
>   `1e-6f`, `1/1024`), as listed under [Mirage](#trusted-assumptions-mirage).
> - Kuiper's models of FP32/BF16 operations, casts, GPU intrinsics
>   (`rsqrt`/`rcp`/`exp2` approximations, `__expf`, `cosf`/`sinf`), comparisons,
>   the BF16 tensor-core MMA (`emma_approx_lemma`), launches, `memcpy`, arrays,
>   the 32-bit `size_t` extraction assumption, and F*/Pulse/KaRaMeL extraction to CUDA.
> - The fixed RESOLVE host wrappers (`wrapper.json`): contiguity and shape
>   checks, typed allocation, argument passing and CUDA completion.
> - `%~` is qualitative; there is no error bound. Stability of the greedy token
>   against rounding is not proved.
>
> Outside F*, the guarantee also trusts:
> - the generated host wiring `KPlay.Proof.Wiring.<Role>`, a mechanical
>   translation of `proofs/luce-program/<role>/plan.json`;
> - the finite bitwise links from the extracted family kernels to the reduced
>   CUDA, and from the reduced programs to the original sources
>   (`universal_original_source_equivalence: false`).

**Testing versus proof.** RESOLVE tested **4 input samples** (categories V0,
V1, V7, V8; `(P,S)` = (3,5), (1,1), (8,5), (2,1)) with the actual checkpoint
in BF16. The base was repeatable (5 runs, one distinct result per sample).
DTester ran each sample 8 times: 4 baseline runs plus 2 runs under each of two
perturbations (unstructured scoped delays on global and on shared memory, 5000
cycles, p=0.8, no prepass). All runs were **comparable (8/8 per sample)**, with
no mismatches, crashes or infrastructure errors. The final RESOLVE check
approved the candidate (`dtester_approved = 1`, `kuiper_approved = 1`).
Testing covers only these inputs and schedules. The proof covers **every**
input in the stated domain for the reduced programs, under the trusted
assumptions above.

### LUCE files

| Path | Contents |
| --- | --- |
| `reference/luce/` | The reference that `KPlay.SharedSpec` cites as `inputs/reference/*`, byte-identical: verbatim Transformers v5.3.0 excerpts `modeling_qwen3_5.py` and `configuration_qwen3_5.py`, the task's `base.py` (an excerpt; not runnable), `LAYOUT.md`, `weight_layout.json`, `model_config.json.txt` and `input-domain.txt`. [`provenance.json`](reference/luce/provenance.json) lists their hashes and origins |
| `kernels/luce-program-<role>.cu` | The reduced complete program for each role |
| `kernels/luce-<family>-<role>.cu` | The reduced family kernels of that role |
| `kernels/luce-extracted/<family>-<role>/` | Kuiper's extraction of each verified implementation (`KPlay_Functional.{cu,h}`) plus a PyTorch binding (`binding.cu`); `include/` holds the same four RESOLVE install headers as FlashMoE's |
| `proofs/luce-program/<role>/plan.json` | The recorded host call plan (4 samples) behind `KPlay.Proof.Wiring.<Role>`; `accepted_cuda_sha256` binds it to the program CUDA |
| `support/luce-cleanroom/` | `Luce.Softplus.fst` and `Luce.Softplus.Fast.fst` |
| `patches/` | The local RoPE patch to the Lucebox original (`luce-rope-pytorch-table.patch`) and a `README` |

### Observing the LUCE kernels (GPU)

[`luce_check.py`](luce_check.py) replays each role's plan for each of the four
recorded samples (with RESOLVE's testing prompts), running the reduced and the
extracted family kernel on the same arguments and requiring bitwise equality,
and then requires that the composition reproduce the complete program bitwise.
It builds the reduced kernels with nvcc's default flags and the extracted ones
with `--fmad=false`. It finally records how far base and optimized agree.

```bash
.venv-gpu/bin/python luce_check.py                                  # synthetic seeded weights
.venv-gpu/bin/python luce_check.py --weights model.safetensors      # real Qwen3.5-0.8B weights
```

With the actual Qwen3.5-0.8B checkpoint packed as BF16 on an H100
([`benchmarks/luce-plan-replay.json`](benchmarks/luce-plan-replay.json)), all
**1,276** family calls (base 672, optimized 604, across the four samples)
matched bitwise between the extracted and reduced kernels, and both roles'
compositions reproduced their complete programs bitwise on every sample. Base
and optimized generated the **same tokens** on all four samples, but their
hidden rows were not bitwise equal (max absolute difference 0.03125--0.0625 at
values up to 6.2). With the default synthetic weights (seed 0, recorded in
the same file) every bitwise check also passed; there the roles chose
different tokens on sample-0000 from the first step, as near-tied random
scores allow. These are finite observations, not proofs.

## Quick start (CPU only)

Use **Linux x86_64 and Python 3.12+**; Ubuntu 24.04 is the tested environment.
Proof checking needs no GPU, CUDA toolkit, Python packages, OCaml installation,
or local Kuiper/F* build. The pinned binaries need the GMP, Zstandard, and C++
runtime libraries. Internet access to GitHub is required for the initial
toolchain download.

On Ubuntu 24.04:

```sh
sudo apt-get update
sudo apt-get install -y git python3 make ca-certificates libgmp10 libzstd1 libstdc++6
git clone https://github.com/FStarLang/kuiper-megakernels.git
cd kuiper-megakernels
make -j2 integrity test
python3 replay.py verify --case luce-shared --jobs 2
```

The last command checks a small supplied proof (the LUCE shared specification). Its first run also prepares
and checks the support libraries, which can take several minutes.
All commands below are run from the repository root.

## Check the proofs

The full suite includes a successful check exceeding **20 GiB peak child
memory** and takes hours, not seconds. Allow for multiple concurrent compiler
processes when choosing `--jobs`; peak child memory is not total memory.
Use a suitably large host, or select `shared`.

```sh
python3 replay.py list
python3 replay.py verify --kind shared --jobs 2
python3 replay.py verify --case 'flashmoe-*' --jobs 8
python3 replay.py verify --case 'luce-*' --jobs 16
python3 replay.py verify --kind tightness --jobs 3   # about 30 s each
python3 tightness.py --check mirage flashmoe luce  # generated theorems unchanged
python3 replay.py verify --kind family --jobs 4
python3 replay.py verify --kind program --jobs 4
python3 replay.py verify --jobs 4
```

The first verification downloads the checksum-pinned public toolchain in
`toolchain.json`, stages the historical source overlays, and checks their
interfaces and implementation bodies. Dependency checking is timed separately;
download and extraction are not included. Subsequent invocations reuse that
preparation, **not example proof results**. Every requested `.fst` and `.fsti`
is checked afresh with
`--force --cache_off --admit_smt_queries false`. All current cases use the
`kuiper-nightly-2026-11-18` pin (F* `049c07a0d9902cd39bf45a47a736be891297b79f`,
Z3 5.3.0). The manifest's `toolchains` map assigns profiles to pins; any other
profile would use `default` (F* `da21c39`, Z3 4.13.3), which the former LUCE
artifacts used and no current case does.

Results and per-module logs go to `.work/proofs/runs/<run>/`. JSON records wall
time, child CPU time, peak child RSS, source hashes, tool versions, and worker
count. This measures **checking the supplied proof**, not finding it. Existing
native/numerical axioms are still trusted; disabling SMT admission does not
turn those axioms into hardware proofs. See [CLAIMS.md](CLAIMS.md).

Each project uses one specification for all its cases,
`proofs/<project>-shared/KPlay.SharedSpec.fst`, and one directory per family
and role, `proofs/<project>-<family>/<role>/`. Family directories exist only
for the roles that have that family. The complete programs and the review are
under `proofs/<project>-program/`:

```text
proofs/luce-shared/KPlay.SharedSpec.fst
proofs/luce-argmax/{base,optimized}/KPlay.Functional.{fst,fsti}
proofs/luce-embedding_rope/base/KPlay.Functional.{fst,fsti}
proofs/luce-program/KPlay.Review.{fst,fsti}
proofs/luce-program/<role>/KPlay.Proof.{Program,Wiring}.<Role>.*, plan.json
```

Every role imports the project's single `KPlay.SharedSpec` module. The manifest
lists the common sources as roots of each case, and the runner includes their
directory and checks them afresh with each selected case.

`--case` accepts a quoted pattern and is repeatable. `--kind` selects `family`,
`program`, `shared`, `review`, or `tightness`; the default is `all`. A changed support
profile requires a new `--work` directory. The old compiler's packaged checked
caches produced verification failures during preparation, so replay deliberately
retains the historical cache-free checking policy.

Verification exits nonzero on a failed check. Inspect the printed results path
and the adjacent module logs. Interrupted runs are not reported as successful.
To start over after an interrupted toolchain preparation, use a fresh directory:

```sh
python3 replay.py verify --case luce-shared --jobs 2 --work .work/retry
```

## Review the base/optimized correspondence

The three reviews (`mirage-program-review`, `flashmoe-program-review`,
`luce-program-review`) are described in their sections above: in each, both
roles satisfy the identical `program_postcondition` on the same inputs. The
[review catalog](proofs/REVIEW-CATALOG.md) summarizes what each one states.

Each `*-review` case checks the review **and all roots of both implementation cases**,
including their bodies, not just their interfaces. The manifest validator rejects
a review that omits a referenced implementation's roots. It is also included in
`--kind review` and the default full suite; the existing `shared` CI suite is
unchanged.

## Continuous integration

The [workflow](.github/workflows/megakernel-artifacts.yml) checks frozen source
hashes and runner unit tests on pushes to `main` and on pull requests.
Proof checks are opt-in: maintainers can open **Actions**, select
**Megakernel artifacts**, choose **Run workflow**, and select `shared`, or use:

```sh
gh workflow run megakernel-artifacts.yml --repo FStarLang/kuiper-megakernels -f suite=shared
```

The manual proof job uses two workers and uploads timing records and diagnostic
logs as the `megakernel-proof-checks` artifact, including on failure.
Full program/family suites remain local commands for suitably provisioned hosts;
they are intentionally not mandatory hosted-runner checks.
See [benchmarks/](benchmarks/) for measured costs and the CI rationale.

## Inspect and run the reduced kernels

`kernels/` contains the exact retained CUDA, including repaired revisions.
`manifest.json` binds every proof/support/kernel file by SHA-256, records the
role-specific historical receipt identities, and lists each CUDA entry's ABI.
The `KPlay.*` names are historical F* module names, not a Python dependency.

```sh
# Optional full upstream source checkouts; no builds or model downloads.
python3 fetch_sources.py

# In a separate environment with the CUDA 13.0 toolkit and an NVIDIA GPU:
python3 -m venv .venv-gpu
. .venv-gpu/bin/activate
python -m pip install -r requirements-gpu.txt
python run_kernels.py --project all --device 0 --jobs 2
python run_kernels.py --project all --kind family --compile-only --jobs 2
python mirage_check.py   # Mirage plan replay and extracted-kernel links (see above)
python flashmoe_check.py # FlashMoE plan replay, extracted-kernel links, reference
python luce_check.py     # LUCE plan replay and extracted-kernel links (see above)
```

The smoke runner builds **all six complete reduced programs** and checks their
complete outputs on synthetic fixtures, including FlashMoE ties/capacity and
LUCE's offset-one weight view. It does not demand identical tokens between
implementations when their specifications permit different choices.
The 30 family entry points can be compiled independently; execution requires
their documented caller state, rather than arbitrary zero-filled arguments.
Apart from `mirage_check.py`, `flashmoe_check.py` and `luce_check.py`, these are reduced-CUDA checks, **not new extraction
of the Pulse witnesses** or replays of the original megakernels. No weights or tensors are downloaded.

Full upstream pins, kernel/reference entry paths, model identities, and
provenance limitations are in `sources.json`. An upstream checkout is a
reference, **not a substitute for the exact repaired code in `kernels/`**.
The optional `--submodules` flag fetches
pinned submodules too. Inspect [NOTICE.md](NOTICE.md) before redistribution.
The model revisions identify models, not an external Transformers code
revision. Inspecting the upstream entry paths does not reconstruct the original
run or close its source-correspondence obligations.
Native extension builds may be reused on subsequent invocations; select a new
`--work` directory to measure fresh compilation.

## Where this fits in RESOLVE

| Stage | What is released here | Evidence boundary |
| --- | --- | --- |
| Original workload and task | Pinned upstream references and explicit mathematical specifications | Reference/task interpretation was reviewed, not proved by checking a Git hash |
| Reduction and decomposition | Exact complete and family CUDA sources | Original/reduced correspondence is historical finite/reviewed evidence, not a universal equivalence theorem |
| Specification and implementation | Shared contracts, interfaces, complete Pulse implementations and helper proofs | F* checks refinement under the stated numerical and native contracts |
| Tightness | `<p>-tightness`: each typed spec defines a function (one answer per input, up to argmax ties); theorem written by `tightness.py`, proof checked by F* | The question file (outputs listed, reference values, allowed ties) is reviewed |
| Replay | Small standalone proof and CUDA runners | No discovery, agent orchestration, fuzzing pipeline, or DTester implementation |

This repository is independent of Kuiper's source tree: most cases reuse module
names and must have isolated include paths. The replay runner provides the required
isolation and obtains its compiler from the pinned public release. No Kuiper
checkout or submodule is required. Frozen payloads preserve their original
whitespace and are excluded from formatting/whitespace cleanup.
[CLAIMS.md](CLAIMS.md) describes
the exact theorem scopes. See [benchmarks/](benchmarks/) for measured costs and
the CI recommendation; source integrity and runner tests are also available through
`make -j4 integrity test`.
