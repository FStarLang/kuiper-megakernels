# Kuiper megakernel proof artifacts

Standalone artifacts from the RESOLVE megakernel case studies: **six complete
program refinements, 24 family refinements, three shared-specification checks,
and 12 family-tightness proofs**, plus eight checked contract reviews.
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
python3 replay.py verify --case mirage-argmax-tightness --jobs 2
```

The last command checks a small supplied proof. Its first run also prepares
and checks the Mirage support library, which can take several minutes.
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
`--force --cache_off --admit_smt_queries false`, F* commit
`da21c3925bf09a0a1d9e7558170be82da5edb37c`, and Z3 4.13.3.

Results and per-module logs go to `.work/proofs/runs/<run>/`. JSON records wall
time, child CPU time, peak child RSS, source hashes, tool versions, and worker
count. This measures **checking the supplied proof**, not finding it. Existing
native/numerical axioms are still trusted; disabling SMT admission does not
turn those axioms into hardware proofs. See [CLAIMS.md](CLAIMS.md).

One proof-only replay adjustment is disclosed in
[replay-adjustments.patch](replay-adjustments.patch): an explicit, proved
real-division identity in Mirage's optimized decoder. Its original source hash
is retained in `manifest.json`; contracts, executable operations, and CUDA are
unchanged. The reviewed kernels' entry modules additionally have distinct
module names so the checked review can import both at once; only their filenames,
module declarations, and qualified tactic self-references changed, except for
explicit proof steps added to two recurrent-decoder lemmas after renaming exposed
solver sensitivity. Contracts and executable operations are unchanged. Historical hashes are retained
in `manifest.json` under `historical_entry_modules`. Other retained proof sources
preserve their historical bytes.

Each kernel has one directory containing its shared specification, its
family-tightness proof where present, and separate implementations. For example:

```text
proofs/mirage-decoder/
  KPlay.SharedSpec.fst
  KPlay.Proof.Tightness.fst
  base/KPlay.Functional.{fst,fsti}
  optimized/KPlay.Functional.{fst,fsti}
```

The three complete programs use the same layout under
`proofs/<project>-program/`, without whole-program tightness proofs.
Their common `KPlay.Proof.Shared.*` helper modules also live at the group root;
Mirage's additional shared-check implementation lives in `shared/`.
Every role imports the group's single `KPlay.SharedSpec` module. The manifest
lists the common sources as roots of each case, and the runner includes their
directory and checks them afresh with each selected case.

The family specifications and the LUCE/FlashMoE program specifications were
already byte-identical across roles. The complete Mirage programs use the
unchanged former optimized/shared specification, whose prefix contains the
entire former base specification. Only paths and duplicate files changed;
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
python3 replay.py verify --case mirage-argmax-tightness --jobs 2 --work .work/retry
```

## Review the base/optimized correspondence

Start with [`proofs/flashmoe-program/KPlay.Review.fsti`](proofs/flashmoe-program/KPlay.Review.fsti).
It exposes the common caller precondition, the complete success/failure
postcondition, the completion-epoch definition, and the `base`, `optimized`,
and implementation-indexed `switcher` signatures. The corresponding `.fst` keeps the adaptation
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

Seven family reviews use the same interface-and-dispatcher style:

| Review interface | Explicit representation difference | Common guarantee |
| --- | --- | --- |
| [Mirage embedding/position](proofs/mirage-embedding_position/KPlay.Review.fsti) | Seven versus eight hidden rows; seven versus 4096 rotary positions | Shared embedding/rotary postcondition on all seven prompt positions |
| [Mirage argmax](proofs/mirage-argmax/KPlay.Review.fsti) | Vocabulary stride 151936 versus padded stride 153600 | Shared prompt/output and allowed-maximum relation, under explicit finite comparison-faithful input representation |
| [LUCE argmax](proofs/luce-argmax/KPlay.Review.fsti) | BF16 versus FP32 scores; first versus any maximum | Explicit proof that the base postcondition implies the optimized postcondition on the same represented logical scores and output; exact history update |
| [LUCE final norm/projection](proofs/luce-final_norm_projection/KPlay.Review.fsti) | BF16 versus FP32 returned logits | Identical shared mathematical logit targets, exact hidden-history append, and preserved input ownership |
| [LUCE embedding](proofs/luce-embedding/KPlay.Review.fsti) | Base additionally returns rotary data and initial recurrent/convolution state | Identical exact BF16 embedding-row postcondition; auxiliary outputs and initial-token alias ownership are explicit |
| [LUCE recurrent decoder](proofs/luce-recurrent_decoder/KPlay.Review.fsti) | BF16 versus FP32 convolution state, transposed recurrent axes, and optimized allocation-free initial state | Same complete shared decoder/cache transition under explicit input/output representations |
| [FlashMoE argmax](proofs/flashmoe-argmax/KPlay.Review.fsti) | Base copies one row and routing frame; optimized selects a batch and returns gathered best probabilities | Identical stable-maximum/gather relation; explicit lemma lifts base row contracts to the batched shared postcondition |

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
| Tightness | Twelve constructive family-level proofs | Existence and full-observation determination, or the stated choice/approximation policy; not IEEE uniqueness |
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
