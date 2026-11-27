# Local patches

- `flashmoe-argmax-start-row0.patch` (FlashMoE @ 96ad271, `csrc/include/flashmoe/gate.cuh`, single-block top-k path only): seed the top-k scan with the first unselected expert (`row[0]` for k = 1) instead of -infinity, keeping strict `>`.
  Why: the selected probability is then always the selected row entry, so the combine weight is provably `row[s] / row[s]` on every row, including all-NaN rows. Outputs do not change, because an all-NaN row gives NaN either way.
  This is a local patch for the proof, not an upstream change.
