# Kuiper megakernel proof artifacts

Standalone artifacts from the RESOLVE megakernel case studies: **six complete
program refinements, 24 family refinements, three shared-specification checks,
and eight family-tightness proofs**, plus ten checked contract reviews.
No RESOLVE installation, agent, DTester,
model weights, private checkout, or historical checked cache is required.

This standalone repository was migrated from `FStarLang/kuiper`, branch
`examples/megakernel-proof-artifacts`, commit
`756939b576b262523c8a45e92316ea97b895aa1f`.
The support-library and CUDA payloads and historical timing records are
unchanged. Proofs are now grouped by kernel with one shared specification,
as described below. See [NOTICE.md](NOTICE.md) for attribution and licensing
boundaries.
FlashMoE still has an explicitly unresolved original-candidate provenance gap.

## Mirage: clean-room RESOLVE pipeline proof

On this branch the Mirage artifacts are replaced by the proof produced by the
clean-room RESOLVE proof pipeline
([`src/KPlay/proof_pipeline`](https://github.com/msr-central/resolve/tree/main/src/KPlay/proof_pipeline)).
The agents saw only the reduced CUDA, the PyTorch reference and the Kuiper
library. They had no access to earlier Mirage proofs, including the former
Mirage artifacts here. The selected Qwen3-0.6B kernel (seven-token prefill plus one greedy decode
step) is proved for both roles, and the result then passed RESOLVE's normal
final check.

| Case | What F* checks |
| --- | --- |
| `mirage-shared` | [`KPlay.SharedSpec`](proofs/mirage-shared/KPlay.SharedSpec.fst): exact real reference math, the role-independent `program_postcondition`, and one quantified relation per role and family |
| `mirage-{argmax,decoder,embedding,projection}-{base,optimized}` | Each Kuiper/Pulse implementation in `KPlay.Functional` ensures its family relation: for **all** reals that the inputs approximate, the outputs approximate the reference of those reals |
| `mirage-program-{base,optimized}` | `KPlay.Proof.Program.<Role>.refines`: the family relations at every host call of the reduced program imply `program_postcondition`. That postcondition covers the prompt copied in front of the token, the final hidden rows approximating `final_hidden`, and the token being an allowed maximum of a score row that approximates `next_token_logits` |

`kernels/mirage-<family>-<role>.cu` and `kernels/mirage-program-<role>.cu` are
the reduced family and complete programs. `*.extracted.cu` is the CUDA that
Kuiper extracted from each verified implementation. RESOLVE replayed it bitwise
against the reduced family CUDA on recorded inputs.
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
>   translation of the reduced program's recorded host call plan;
> - the finite bitwise replays linking the extracted family kernels to the
>   reduced CUDA, and the reduced program to the original Mirage megakernel.
>
> The guarantee is relational (`%~`), not an error bound or bitwise equality
> between roles. Optimized and base may choose different maximal tokens on ties.

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
python3 replay.py verify --case luce-argmax-tightness --jobs 2
```

The last command checks a small supplied proof. Its first run also prepares
and checks the support libraries, which can take several minutes.
All commands below are run from the repository root.

## Check the proofs

The full suite includes a successful check exceeding **20 GiB peak child
memory** and takes hours, not seconds. Allow for multiple concurrent compiler
processes when choosing `--jobs`; peak child memory is not total memory.
Use a suitably large host, or select `tightness`/`shared`.

```sh
python3 replay.py list
python3 replay.py verify --kind tightness --jobs 2
python3 replay.py verify --kind shared --jobs 2
python3 replay.py verify --case flashmoe-program-base --jobs 4
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
`--force --cache_off --admit_smt_queries false`. LUCE and FlashMoE use F* commit
`da21c3925bf09a0a1d9e7558170be82da5edb37c` and Z3 4.13.3. Mirage uses the
`kuiper-nightly-2026-11-18` pin (F* `049c07a0d9902cd39bf45a47a736be891297b79f`,
Z3 5.3.0). The manifest's `toolchains` map assigns profiles to pins, and every
other profile uses `default`.

Results and per-module logs go to `.work/proofs/runs/<run>/`. JSON records wall
time, child CPU time, peak child RSS, source hashes, tool versions, and worker
count. This measures **checking the supplied proof**, not finding it. Existing
native/numerical axioms are still trusted; disabling SMT admission does not
turn those axioms into hardware proofs. See [CLAIMS.md](CLAIMS.md).

The reviewed kernels' entry modules additionally have distinct
module names so the checked review can import both at once; only their filenames,
module declarations, and qualified tactic self-references changed, except for
explicit proof steps added to two recurrent-decoder lemmas and a full-attention
cache-loop exit after renaming exposed solver sensitivity. Contracts and executable operations are unchanged. Historical hashes are retained
in `manifest.json` under `historical_entry_modules`. Other retained proof sources
preserve their historical bytes.

Each kernel has one directory containing its shared specification, its
family-tightness proof where present, and separate implementations. For example:

```text
proofs/flashmoe-argmax/
  KPlay.SharedSpec.fst
  KPlay.Proof.Tightness.fst
  KPlay.Review.{fst,fsti}
  base/KPlay.Base.Functional.{fst,fsti}
  optimized/KPlay.Optimized.Functional.{fst,fsti}
```

The LUCE and FlashMoE programs use the same layout under
`proofs/<project>-program/`, without whole-program tightness proofs.
Their common `KPlay.Proof.Shared.*` helper modules also live at the group root.
Mirage instead uses one specification for all its cases,
`proofs/mirage-shared/KPlay.SharedSpec.fst`, and `proofs/mirage-<family>/<role>/`.
Every role imports the group's single `KPlay.SharedSpec` module. The manifest
lists the common sources as roots of each case, and the runner includes their
directory and checks them afresh with each selected case.

The family specifications and the LUCE/FlashMoE program specifications were
already byte-identical across roles. Only paths and duplicate files changed;
theorem statements, numerical assumptions, and implementation bodies are unchanged.
Existing case IDs and replay commands are unchanged.

`--case` accepts a quoted pattern and is repeatable. `--kind` selects `family`,
`program`, `shared`, `tightness`, or `review`; the default is `all`. A changed support
profile requires a new `--work` directory. The old compiler's packaged checked
caches produced verification failures during preparation, so replay deliberately
retains the historical cache-free checking policy.

Verification exits nonzero on a failed check. Inspect the printed results path
and the adjacent module logs. Interrupted runs are not reported as successful.
To start over after an interrupted toolchain preparation, use a fresh directory:

```sh
python3 replay.py verify --case luce-argmax-tightness --jobs 2 --work .work/retry
```

## Review the base/optimized correspondence

Start with [`proofs/flashmoe-program/KPlay.Review.fsti`](proofs/flashmoe-program/KPlay.Review.fsti).
It exposes the common caller precondition, the complete success/failure
postcondition, the completion-epoch definition, and the `base`, `optimized`,
and the implementation-indexed, dispatch-only `switcher` definition. The corresponding `.fst` keeps the adaptation
proofs and wrappers that call the verified implementations directly. Their
signatures are identical except for `BaseRole` versus `OptimizedRole`: completion
is after one launch versus three. `input_models_agree` proves the input adapters
identical; `allocation_failure_contracts_agree` and `returned_contracts_agree`
prove postcondition equality, with completion epochs aligned for the latter.
These are equalities of predicates on the same result, not of two executions'
results. Both wrappers preserve the caller's input references,
input cells and frame; successful returned cells satisfy the same
`Shared.task_relation`. This is common-task refinement, not output equality.
`switcher i` dispatches to the chosen implementation under that same precondition
and proves `after_call i`; compatibility is checked at the dispatch site.

```sh
python3 replay.py verify --case flashmoe-program-review --jobs 8
```

The [LUCE program review](proofs/luce-program/KPlay.Review.fsti) uses the same
dispatcher style. It proves equality of the packed input views and exposes the
same complete safety/history postcondition for both implementations. Base numerical
correctness is unconditional; optimized numerical correctness retains its
family-dependency and numerical-scope premises on the actual returned execution.
Both enqueue one completion epoch and preserve allocation-failure behavior.

The [review catalog](proofs/REVIEW-CATALOG.md) compares all ten groups,
assesses the significance of each contract difference, and distinguishes
exact functionality, allowed choices, and conditional/relative numerical
tightness. Each interface includes a more detailed comment at its visible
`switcher` implementation.

All eight family reviews use the same interface-and-dispatcher style:

| Review interface | Explicit representation difference | Common guarantee |
| --- | --- | --- |
| [LUCE argmax](proofs/luce-argmax/KPlay.Review.fsti) | BF16 versus FP32 scores; first versus any maximum | Explicit proof that the base postcondition implies the optimized postcondition on the same represented logical scores and output; exact history update |
| [LUCE final norm/projection](proofs/luce-final_norm_projection/KPlay.Review.fsti) | BF16 versus FP32 returned logits | Identical shared mathematical logit targets, exact hidden-history append, and preserved input ownership |
| [LUCE embedding](proofs/luce-embedding/KPlay.Review.fsti) | Base additionally returns rotary data and initial recurrent/convolution state | Identical exact BF16 embedding-row postcondition; auxiliary outputs and initial-token alias ownership are explicit |
| [LUCE recurrent decoder](proofs/luce-recurrent_decoder/KPlay.Review.fsti) | BF16 versus FP32 convolution state, transposed recurrent axes, and optimized allocation-free initial state | Same complete shared decoder/cache transition under explicit input/output representations |
| [LUCE full-attention decoder](proofs/luce-full_attention_decoder/KPlay.Review.fsti) | Compact time-major versus full head-major caches; external versus internal rotary computation | Complete shared decoder/cache transition; optimized numerical guarantee requires finite actual live scores, while physical cache retention is unconditional |
| [FlashMoE argmax](proofs/flashmoe-argmax/KPlay.Review.fsti) | Base copies one row and routing frame; optimized selects a batch and returns gathered best probabilities | Identical stable-maximum/gather relation; explicit lemma lifts base row contracts to the batched shared postcondition |
| [FlashMoE projection/softmax](proofs/flashmoe-projection_softmax/KPlay.Review.fsti) | One row plus routing-frame initialization/copy versus a complete profile matrix | Same projection/softmax target for every returned cell; base batch-lifting proof and optimized guarantee for every represented input model |
| [FlashMoE normalization/capacity](proofs/flashmoe-normalization_capacity/KPlay.Review.fsti) | One prefix step with I64 counts versus complete routing with I32 demand; base emits eight I32 demand cells only at token 255 | Same represented shared prefix state, complete route postcondition on completion, and an explicit common full-route refinement lemma |

Each `*-review` case checks the review **and all roots of both implementation cases**,
including their bodies, not just their interfaces. The manifest validator rejects
a review that omits a referenced implementation's roots. It is also included in
`--kind review` and the default full suite; the existing `shared` CI suite is
unchanged.

## Continuous integration

The [workflow](.github/workflows/megakernel-artifacts.yml) checks frozen source
hashes and runner unit tests on pushes to `main` and on pull requests.
Proof checks are opt-in: maintainers can open **Actions**, select
**Megakernel artifacts**, choose **Run workflow**, and select `tightness` or
`shared`, or use:

```sh
gh workflow run megakernel-artifacts.yml --repo FStarLang/kuiper-megakernels -f suite=tightness
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
```

The smoke runner builds **all six complete reduced programs** and checks their
complete outputs on synthetic fixtures, including FlashMoE ties/capacity and
LUCE's offset-one input repair. It does not demand identical tokens between
implementations when their specifications permit different choices.
The 24 family entry points can be compiled independently; execution requires
their documented caller state, rather than arbitrary zero-filled arguments.
These are reduced-CUDA checks, **not new extraction of the Pulse witnesses**
or replays of the original megakernels. No weights or tensors are downloaded.

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
| Tightness | Eight constructive LUCE/FlashMoE family-level proofs; Mirage has none | Existence and full-observation determination, or the stated choice/approximation policy; not IEEE uniqueness |
| Replay | Small standalone proof and CUDA runners | No discovery, agent orchestration, fuzzing pipeline, or DTester implementation |

This repository is independent of Kuiper's source tree: most cases reuse module
names and must have isolated include paths. Reviewed kernels' entry
modules are named `KPlay.Base.Functional` and `KPlay.Optimized.Functional` to
allow their joint review. The replay runner provides the required
isolation and obtains its compiler from the pinned public release. No Kuiper
checkout or submodule is required. Frozen payloads preserve their original
whitespace and are excluded from formatting/whitespace cleanup.
[CLAIMS.md](CLAIMS.md) describes
the exact theorem scopes. See [benchmarks/](benchmarks/) for measured costs and
the CI recommendation; source integrity and runner tests are also available through
`make -j4 integrity test`.
