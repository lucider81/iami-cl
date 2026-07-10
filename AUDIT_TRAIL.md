# Audit Trail

## 2026-07-10: Independent Audit

Critical bugs found in v1.5.3a-v1.6b:

| Finding | Status |
|---|---|
| v1.5.3a crashes at task 0 | Confirmed |
| A-GEM erases current-task gradient | Confirmed |
| Commitment alpha is causally inert | Confirmed |
| v1.6 is a renamed copy (14 lines changed) | Confirmed |
| v1.6b has no experiment (prints two lines) | Confirmed |
| One-token transformer, dead attention | Confirmed (fixed: 16-patch) |
| "50% replay" != 50% gradient influence | Confirmed (fixed: sample-weighted) |
| 1,051K params, not 810K | Confirmed (honest: 808K) |

## 2026-07-11: v1.7 Rebuild

**All previous results INVALIDATED.** Root cause: `torch.is_grad_enabled()` returned
False in the IPython environment, making every `backward()` a no-op.

v1.7: explicit `torch.set_grad_enabled(True)`, sample-weighted loss, per-task forward,
16-patch input, asserted disjointness, honest parameter counts.

## Results

| Setting | 5-Task Permuted MNIST (2 epochs, 500 samples) |
|---|---|
| Baseline | 40-57% (directional range) |
| Replay 50% | 99-100% (directional range) |
| **Improvement** | **+50-60pp** |

## Gate 8c

Original v1.5.3a self-healing (with scoping bug patched):
- 30 boolean circuits over 8-bit strings
- Mean retention: **0.817** (corrected from 0.805)
- Verdict: **FAILED** (threshold 0.99)
- 7/30 tasks had val_ba < 0.70 (marginal)
- 0/30 tasks hit abandon path
