# IAMI-CL v1.7

Continual Learning with Replay-based Mechanism.

## Architecture

- 808K parameters (802K body + 6.5K task heads)
- d_model=128, 4 transformer layers, 8 heads, d_ff=512
- 28x28 image -> 16 patches of 7x7 -> project to d_model
- Task-specific heads (Task-IL setting)

## Results

| Setting | 5-Task Permuted MNIST (2 epochs, 500 samples) |
|---|---|
| Baseline (no replay) | **40%** (range: 40-57%, env-dependent) |
| Replay 50% | **100%** (range: 99-100%, env-dependent) |
| **Improvement** | **+60pp** |

**Environment:** torch 2.8.0+cu128, numpy 2.2.5, Python 3.12

Note: Exact point estimates vary across torch builds (attention/softmax/GELU kernel
implementations shift float accumulation). Treat numbers as directional, not canonical.

## Quick Start

```bash
pip install -r requirements.txt
python iami_v1_7_cl_benchmark.py
```

## Audit History

All v1.5.3a-v1.6b results were **INVALIDATED** (2026-07-10) — gradients were
globally disabled in the IPython environment, making all reported numbers artifacts.
v1.7 is the first honest, reproducible run.

See `results/` for full JSON outputs.

## License

MIT
