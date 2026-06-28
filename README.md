# Neuro-Symbolic Learning: Data Efficiency, Reasoning Shortcuts & Compositional Generalization

A controlled study on **MNIST-Addition** of what neural-symbolic *integration* actually buys —
and when it silently breaks. We wire a fixed symbolic `+` into a shared CNN digit classifier,
train on **sums only** (no digit labels), and measure not just task accuracy but the *latent symbol
grounding* and *compositional generalization*.

## Key findings (5 seeds/condition, 95% CIs, Welch t-tests + Holm correction)

- **Integration is a large, significant capability + data-efficiency win.** NeSy reaches **85.4%**
  sum accuracy at 1,000 training pairs vs the neural baseline's **22.0%** (Cohen's *d*=56,
  *p*=2.9e-13), matching a fully digit-supervised oracle.
- **Grounding emerges for free — when the task identifies it.** With *no* digit labels, NeSy
  recovers the latent digits at **99.0%** on standard addition.
- **Flip one operator and the mind breaks.** Changing `+` → `(a+b) mod 10` (same data, same net)
  makes correct grounding **never emerge**: among seeds that learn the task, raw grounding is
  **≈0.1%** — the net reads digits systematically wrong via a global relabeling (sometimes an exact
  `(d+5)` cyclic shift) that leaves `(a+b) mod 10` invariant. A digit-supervised *oracle* grounds at
  99% on the *same* task, so distant supervision under a symmetric operator is the cause. A textbook
  *reasoning shortcut*, invisible to task accuracy.
- **A 10-label cure.** Just **10 labeled images** (1/class) restore grounding to **97.7% in every
  seed** (and stabilize training) exactly when a shortcut exists; unsupervised regularizers mostly cannot.
- **Correct grounding unlocks a new capability.** Trained on single-digit sums only, NeSy solves
  **2-digit (95.7%)** and **3-digit (93.7%)** addition zero-shot via symbolic place-value
  composition; the neural baseline scores **0%** (structurally incapable).

See **[REPORT.md](REPORT.md)** for the full writeup, tables, statistics, and discussion.

## Figures (`figures/`)
- `fig1_data_efficiency.png` — learning curves + grounding-without-labels.
- `fig2_shortcut.png` — standard vs modular: same accuracy, collapsed grounding.
- `fig3_confusion.png` — grounding confusion matrices for all 5 modular seeds (the shortcut).
- `fig4_compositional_ood.png` — zero-shot multi-digit addition (neural=0).
- `fig5_mitigations.png` — 10 labels fix the shortcut.

## Reproduce
```bash
uv venv && source .venv/bin/activate      # Python 3.12
uv add torch torchvision numpy pandas scipy scikit-learn matplotlib

# data integrity / module smoke tests
python src/data.py

# run experiments (writes results/raw/*.json) — ~6 min on 3x A6000
python -m src.run_experiments --exp all   # E1/E2, E3, E4, E5

# statistics + figures (writes results/stats.json, figures/*.png)
python -m src.analyze
```
Config (sizes, seeds, hyperparameters) lives in `src/config.py`. GPUs used are set in
`src/run_experiments.py` (`GPUS = [0, 1, 3]`).

## Structure
```
planning.md              # Phase 0/1: motivation, novelty, design, pre-registered analysis plan
REPORT.md                # primary deliverable: full research report
src/
  data.py                # MNIST + MNIST-Addition loaders, multi-digit OOD generator
  models.py              # shared CNN; NeuralSum; differentiable symbolic sum & place-value compose
  train.py               # train loops (neural/nesy/oracle), eval, grounding + Hungarian metrics
  config.py              # all hyperparameters and experiment grids
  run_experiments.py     # GPU-parallel runner for E1-E5
  analyze.py             # statistics, hypothesis tests, figures
results/
  raw/*.json             # per-run results        results/stats.json  # aggregates + tests
  summary_table.csv
figures/*.png
datasets/ code/ papers/  # pre-gathered resources (literature_review.md, resources.md)
```
NeSy model = compact differentiable marginalization `P(sum=s)=Σ_{a+b=s}P(a)P(b)` (the
DeepProbLog/Scallop idea). Digit labels are used **only** for evaluation and the oracle/mitigation
conditions; NeSy and the neural baseline train on sums alone.
