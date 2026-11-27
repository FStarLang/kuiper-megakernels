# Local patches

- `flashmoe-argmax-start-row0.patch` (FlashMoE @ 96ad271, `csrc/include/flashmoe/gate.cuh`, single-block top-k path only): seed the top-k scan with the first unselected expert (`row[0]` for k = 1) instead of -infinity, keeping strict `>`.
  Why: the selected probability is then always the selected row entry, so the combine weight is provably `row[s] / row[s]` on every row, including all-NaN rows. Outputs do not change, because an all-NaN row gives NaN either way.
  This is a local patch for the proof, not an upstream change.

- `luce-rope-pytorch-table.patch` (Lucebox @ 0f212acacb284cf77f29f4fb4f706b1f4fdc3f88, `optimizations/megakernel/kernel.cu`; apply with `patch -p1` from the Lucebox root): replace the runtime `powf` RoPE frequencies with PyTorch's float32 `inv_freq` table, so the optimized program computes `position * inv_freq[i]` as the reference does.
  Why: the spec pins the RoPE frequencies to PyTorch's float32 values, so there is one answer per input.
  This is a local patch for the proof, not an upstream change; the LUCE specs and reduced optimized kernels assume it.
