# Attribution and release status

These artifacts were migrated from the Kuiper artifact branch to a standalone
repository at the release owner's request on **November 13, 2026**.
The migration preserves existing source notices and dependency license texts;
it does not introduce an additional outbound license.

| Material | Source and status |
| --- | --- |
| Mirage upstream | `mirage-project/mirage`, pinned in `sources.json`; Apache-2.0 text retained in `licenses/Mirage-Apache-2.0.txt` |
| LUCE upstream | Official `Luce-Org/lucebox` at `0f212acacb284cf77f29f4fb4f706b1f4fdc3f88` (`optimizations/megakernel`), pinned in `sources.json`; Apache-2.0, copyright 2026 Lucebox; license text retained in `licenses/Lucebox-Apache-2.0.txt`. The included CUDA consists of RESOLVE's reduced programs, not a bundled upstream checkout |
| FlashMoE upstream | `osayamenja/FlashMoE` with PR #32 applied (`tyler-utah/FlashMoE`), pinned in `sources.json`; BSD-3-Clause, copyright 2026 Osayamen Jonathan Aimuyo; original notice and conditions retained in `licenses/FlashMoE-BSD-3-Clause.txt` |
| F*/Pulse support | Source snapshots and changes relative to the pinned compiler package; applicable Apache-2.0 texts retained |
| Reference semantics | Qwen3 and Qwen3.5 Transformers model implementations, with model identities pinned in `sources.json`; the Mirage Qwen3 sources are included in `reference/mirage/`, the FlashMoE router reference in `reference/flashmoe/` and the LUCE Qwen3.5 reference excerpts in `reference/luce/`; Transformers and PyTorch license texts retained. No model weights, tokenizers, or input tensors are included |
| Proofs and new replay code | Authored during the proof-artifact work. No additional outbound license is assigned by this migration; dependency notices do not automatically license this material |

Existing notices inside source files are preserved. Dependency license texts
are not a blanket grant for every file in this directory. A public repository
without a license is not automatically permission to redistribute derivative
code, and a public draft branch/PR would already disclose it. See GitHub's
[licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository);
the release decision requires the appropriate rightsholders' approval.

## Retained changes and provenance

The CUDA files are **reductions/translations and repaired variants**, not
verbatim claims about the upstream revisions. Their exact bytes and
role-specific historical source/acceptance identities are in `manifest.json`.
No runtime source rewriting is performed by the smoke runner; its generated
C++ file only adds a Python binding to the existing `kplay_run` function.

The Mirage proofs, kernels and `support/mirage-cleanroom` are output of the
clean-room RESOLVE proof pipeline, which replaced the earlier Mirage
artifacts. `kernels/mirage-*.cu` are RESOLVE's reduced Mirage CUDA.
`kernels/mirage-extracted/*/KPlay_Functional.{cu,h}` are unmodified Kuiper
extractions of the verified implementations; each `binding.cu` is the RESOLVE
replay binding with its include paths made relative. `kernels/mirage-extracted/include/`
holds three headers from RESOLVE's Kuiper installation. `proofs/mirage-program/*/plan.json`
are RESOLVE's recorded host call plans.
`reference/mirage/` is the PyTorch reference used by RESOLVE: `base.py`,
`model_abi.py` and `model_config.json` come from the RESOLVE Mirage task, and
the remaining Python files are unmodified Hugging Face Transformers 4.57.1
sources (Apache License 2.0; see `licenses/Transformers-Apache-2.0.txt`).
`mirage_check.py` is new.
`support/mirage-cleanroom` is Kuiper library source from `nightly-2026-11-18`,
including open Kuiper PRs #135 and #181 and RESOLVE's numeric additions.

The FlashMoE proofs, kernels and `support/flashmoe-cleanroom` are likewise
output of the clean-room RESOLVE proof pipeline and replace the earlier
FlashMoE artifacts. `kernels/flashmoe-*.cu` are RESOLVE's reduced FlashMoE
router CUDA; the optimized original is upstream FlashMoE with PR #32 applied
(see `sources.json`). `kernels/flashmoe-extracted/` follows the Mirage layout;
its `include/` holds four headers from RESOLVE's Kuiper installation,
including the new `kplay_cpu_exp.cuh`. `proofs/flashmoe-program/*/plan.json`
are RESOLVE's recorded host call plans. `reference/flashmoe/` holds the
RESOLVE FlashMoE task reference (`base.py`, `LAYOUT.md`) and `input-domain.txt`,
a plain-language restatement of its trusted input domain written for this
repository.
The RESOLVE case adapter that `base.py` imports is not published;
`reference/flashmoe/E8_mega_kernel_h100/` is a new standalone shim, written
for this repository from `input-domain.txt`, so that `base.py` runs unmodified.
`support/flashmoe-cleanroom` (`Kuiper.KPlay.CpuExp`) is RESOLVE Kuiper library
source. `flashmoe_check.py` is new. The link from the reduced programs to the
original source is RESOLVE's finite same-role bitwise translation check, not
an original-megakernel equivalence claim.

The LUCE proofs, kernels and `support/luce-cleanroom` are likewise output of
the clean-room RESOLVE proof pipeline and replace the earlier, migrated LUCE
artifacts (whose upstream was `PixelML/luce-megakernel`). `kernels/luce-*.cu`
are RESOLVE's reduced LUCE CUDA; the optimized original is the official
`Luce-Org/lucebox` megakernel and the base is the pinned Transformers 5.3.0
Qwen3.5 reference (see `sources.json`). `kernels/luce-extracted/` follows the
Mirage layout; its `include/` holds the same four RESOLVE install headers as
FlashMoE's. `proofs/luce-program/*/plan.json` are RESOLVE's recorded host call
plans. `reference/luce/` holds the reference files RESOLVE used, byte-identical:
`modeling_qwen3_5.py` and `configuration_qwen3_5.py` are verbatim excerpts of
Hugging Face Transformers v5.3.0 (Apache License 2.0; see
`licenses/Transformers-Apache-2.0.txt`); `base.py` is an excerpt of the RESOLVE
LUCE task's base role and is not runnable; `LAYOUT.md`, `weight_layout.json`,
`model_config.json.txt` (the model's `config.json`) and `input-domain.txt`
describe the contract, layout and input domain. `support/luce-cleanroom`
(`Luce.Softplus`, `Luce.Softplus.Fast`) is RESOLVE Kuiper library source.
`luce_check.py` is new. The link from the reduced programs to the original
source is RESOLVE's finite same-role bitwise translation check, not an
original-megakernel equivalence claim.

Publication approval does not remove applicable licensing conditions or the
documented scientific/provenance limits.
No RESOLVE/DTester implementation, private agent transcript, environment
dictionary, historical executable/cache, captured tensor, or model weight is
part of this package.
