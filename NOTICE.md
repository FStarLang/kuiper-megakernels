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
| FlashMoE upstream | `osayamenja/FlashMoE`, pinned in `sources.json`; BSD-3-Clause, copyright 2026 Osayamen Jonathan Aimuyo; original notice and conditions retained in `licenses/FlashMoE-BSD-3-Clause.txt` |
| F*/Pulse support | Source snapshots and changes relative to the pinned compiler package; applicable Apache-2.0 texts retained |
| Reference semantics | Qwen3 and Qwen3.5 Transformers model implementations, with model identities pinned in `sources.json`; Transformers and PyTorch license texts retained. No model weights, tokenizers, or input tensors are included |
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
The single proof-only replay adjustment is recorded separately with its
original source hash and reversible diff; it is not misidentified as the
original receipt-bound proof bytes.
The FlashMoE complete-program and argmax, Mirage embedding/position and argmax, and LUCE
embedding, argmax, final norm/projection, and recurrent-decoder entry
modules were subsequently renamed to
`KPlay.Base.Functional` and `KPlay.Optimized.Functional` for their joint checked
review. Filenames, module declarations, and qualified tactic self-references
(in LUCE final norm/projection and recurrent decoder) changed; their
historical hashes are retained as `historical_entry_modules` in `manifest.json`.
Recurrent decoder additionally has an explicit erased product witness and
intermediate approximation facts in base `gate_head`, and a pointwise sum proof in optimized
`convolution_sum_shared`, resolving solver sensitivity exposed by renaming.
Neither change alters a contract or an executable operation.
The `KPlay.Review.{fst,fsti}` files are new contract-adaptation proofs and review
interfaces, not historical artifacts.
No runtime source rewriting is performed by the smoke runner; its generated
C++ file only adds a Python binding to the existing `kplay_run` function.

Mirage includes its authorized finite-score/selection and reference/task
repairs. Its current shared specification carries the checked
mixed-observation compatibility bridge. LUCE's complete optimized program
replaces alignment-dependent `uint4`/`float4` reinterpretation reads with typed
scalar reads while preserving indices, casts, FMA/FTZ operations and
accumulation order. Its shared normalization/hidden type separation is a
representation correction. Earlier family artifacts keep their own accepted
source revisions; they must not be silently relabeled as the later complete
program repair.

FlashMoE's retained router is fully identified, but **the original optimized
candidate binding remains unavailable**. The optional upstream checkout is
only a source reference; it does not close that gap or authorize an
original-megakernel equivalence claim.

Publication approval does not remove applicable licensing conditions or the
documented scientific/provenance limits.
No RESOLVE/DTester implementation, private agent transcript, environment
dictionary, historical executable/cache, captured tensor, or model weight is
part of this package.
