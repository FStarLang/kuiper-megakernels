# Input layout note

`base.py` (`reference_route`) takes logical router weights `[hidden, experts]`
(`[512, 8]` in the smoke profile), and `scores = tokens @ router_weights`.
Both reduced CUDA programs take the weights in that same logical layout:
CUDA `weights[h*8 + e]` is reference `router_weights[h, e]`. (The FlashMoE
gate itself consumes `[experts, hidden]`; the case harness passes
`router_weights.t().contiguous()` to it, and the reduced optimized program
folds that transpose into its indexing.) Inputs: BF16 tokens `[256, 512]` and
weights, finite, |tokens| <= 32, |weights| <= 4, contiguous, same device (see
`input-domain.txt`). Smoke profile: 256 tokens, hidden 512, 8 experts,
capacity 32, top_k 1.
