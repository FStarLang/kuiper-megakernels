# Attribution and release status

These artifacts were migrated from the Kuiper artifact branch to a standalone
repository at the release owner's request on **November 13, 2026**.
The migration preserves existing source notices and dependency license texts;
it does not introduce an additional outbound license.
Publication authorization is not presented here as evidence of a separate
permission grant from LUCE's authors.

| Material | Source and status |
| --- | --- |
| Mirage upstream | `mirage-project/mirage`, pinned in `sources.json`; Apache-2.0 text retained in `licenses/Mirage-Apache-2.0.txt` |
| LUCE upstream | **`PixelML/luce-megakernel`**, not `Luce-Org/lucebox`; at `bab084443f14e6ab3ba76f8e8e3f721ad2e56144`, the complete tree has no LICENSE, COPYING, or NOTICE file. The included CUDA consists of reduced implementations produced during the case study, not a bundled upstream checkout. No upstream license or separately documented permission grant is asserted |
| FlashMoE upstream | `osayamenja/FlashMoE` with PR #32 applied (`tyler-utah/FlashMoE`), pinned in `sources.json`; BSD-3-Clause, copyright 2026 Osayamen Jonathan Aimuyo; original notice and conditions retained in `licenses/FlashMoE-BSD-3-Clause.txt` |
| F*/Pulse support | Source snapshots and changes relative to the pinned compiler package; applicable Apache-2.0 texts retained |
| Reference semantics | Qwen3 and Qwen3.5 Transformers model implementations, with model identities pinned in `sources.json`; the Mirage Qwen3 sources are included in `reference/mirage/` and the FlashMoE router reference in `reference/flashmoe/`; Transformers and PyTorch license texts retained. No model weights, tokenizers, or input tensors are included |
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
The LUCE
complete-program, embedding, argmax, final norm/projection, recurrent-decoder, and full-attention-decoder entry
modules were subsequently renamed to
`KPlay.Base.Functional` and `KPlay.Optimized.Functional` for their joint checked
review. Filenames, module declarations, and qualified tactic self-references
(in LUCE final norm/projection, recurrent decoder, and full attention) changed; their
historical hashes are retained as `historical_entry_modules` in `manifest.json`.
Recurrent decoder additionally has an explicit erased product witness and
intermediate approximation facts in base `gate_head`, and a pointwise sum proof in optimized
`convolution_sum_shared`, resolving solver sensitivity exposed by renaming.
Full attention additionally has an explicit ghost loop-exit index and equality
in optimized `store_cache`. These proof stabilizations do not alter contracts
or executable operations.
The `KPlay.Review.{fst,fsti}` files are new contract-adaptation proofs and review
interfaces, not historical artifacts.
LUCE complete-program's byte-identical `KPlay.Proof.NativeInput.fst` copies were
also consolidated into its group directory; both implementation cases check
that same unchanged source.
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
LUCE's complete optimized program
replaces alignment-dependent `uint4`/`float4` reinterpretation reads with typed
scalar reads while preserving indices, casts, FMA/FTZ operations and
accumulation order. Its shared normalization/hidden type separation is a
representation correction. Earlier family artifacts keep their own accepted
source revisions; they must not be silently relabeled as the later complete
program repair.

The FlashMoE proofs, kernels and `support/flashmoe-cleanroom` are likewise
output of the clean-room RESOLVE proof pipeline and replace the earlier
FlashMoE artifacts. `kernels/flashmoe-*.cu` are RESOLVE's reduced FlashMoE
router CUDA; the optimized original is upstream FlashMoE with PR #32 applied
(see `sources.json`). `kernels/flashmoe-extracted/` follows the Mirage layout;
its `include/` holds four headers from RESOLVE's Kuiper installation,
including the new `kplay_cpu_exp.cuh`. `proofs/flashmoe-program/*/plan.json`
are RESOLVE's recorded host call plans. `reference/flashmoe/` holds the
RESOLVE FlashMoE task reference (`base.py`, `input-domain.txt`, `LAYOUT.md`).
`support/flashmoe-cleanroom` (`Kuiper.KPlay.CpuExp`) is RESOLVE Kuiper library
source. `flashmoe_check.py` is new. The link from the reduced programs to the
original source is RESOLVE's finite same-role bitwise translation check, not
an original-megakernel equivalence claim.

Publication approval does not remove applicable licensing conditions or the
documented scientific/provenance limits.
No RESOLVE/DTester implementation, private agent transcript, environment
dictionary, historical executable/cache, captured tensor, or model weight is
part of this package.
