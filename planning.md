# Research Plan: Reasoning Shortcuts & Compositional Generalization in Neuro-Symbolic Learning

**Research domain:** Artificial Intelligence (Neuro-Symbolic learning)
**Primary benchmark:** MNIST-Addition (`datasets/mnist/mnist_addition/`)
**Date:** 2026-06-28

---

## Motivation & Novelty Assessment

### Why This Research Matters
Neuro-symbolic (NeSy) AI promises systems that *perceive* with neural nets and *reason* with
symbolic structure. The headline claims in the literature are (a) **data efficiency** and (b)
**interpretable, correct symbol grounding**. But a subtle and under-reported failure mode —
**reasoning shortcuts** — means a NeSy system can achieve near-perfect *task* accuracy while
learning a *systematically wrong* mapping from perception to symbols (e.g. a consistent
permutation of the digits). If the latent symbols are wrong, every downstream promise of NeSy
(interpretability, knowledge reuse, compositional generalization) collapses silently. Practitioners
who only watch task accuracy will not notice. Quantifying *when* shortcuts appear and *what fixes
them* is therefore directly important for anyone deploying NeSy systems.

### Gap in Existing Work
From `literature_review.md` (§6 Gaps): the "NeSy is more data-efficient" claim is widely *asserted*
but rarely measured head-to-head on identical splits with confidence intervals; and **grounding
accuracy is under-reported** — most papers report only task accuracy, so reasoning shortcuts are
invisible. Softened Symbol Grounding (2024) and Deep Symbolic Learning (2022) raise the shortcut
problem but a clean, controlled, multi-seed *measurement* of the task-accuracy–vs–grounding-accuracy
gap, the conditions under which shortcuts emerge, and a comparison of cheap mitigations is missing.

### Our Novel Contribution
A controlled empirical study on MNIST-Addition that, on identical splits and across 5 random seeds,
measures **four** things side-by-side that the literature usually reports in isolation:
1. **Capability / data efficiency** — NeSy vs. neural-only vs. digit-supervised oracle.
2. **The grounding gap** — task (sum) accuracy *minus* latent digit-grounding accuracy, including a
   permutation-aware analysis (Hungarian alignment) that explicitly detects *systematic* shortcuts.
3. **Mitigations** — three cheap interventions (a few labeled digits; a marginal-uniformity prior;
   prediction-entropy regularization) and how much grounding each restores.
4. **Compositional OOD generalization (capstone)** — reuse the single-digit classifier, composed
   with a *symbolic place-value sum*, to solve **multi-digit** addition never seen in training.
   This is the sharpest test of "integration → a new capability": the neural-only model is
   structurally incapable of it; the NeSy model should do it zero-shot *iff* its grounding is correct
   — tying the capstone directly back to the grounding-gap finding.

### Experiment Justification
- **E1 (Data efficiency):** Establishes the core NeSy capability claim with statistical rigor —
  needed as the foundation and to locate the low-data regime where shortcuts are most likely.
- **E2 (Grounding gap / shortcuts):** The scientific heart — measures whether high sum accuracy
  implies correct symbols. Needed because it is exactly the under-reported quantity.
- **E3 (Mitigations):** Turns the diagnosis into actionable guidance — *which* cheap fix best
  restores grounding. Needed to make the finding useful, not just cautionary.
- **E4 (Compositional OOD):** Demonstrates the *payoff* of correct grounding and the unique
  capability that integration unlocks. Needed to show why the grounding gap is not academic.

---

## Research Question
On the canonical MNIST-Addition benchmark, does integrating a fixed symbolic addition operation
with a shared neural digit classifier (trained on **sums only**, distant supervision) (i) match a
fully digit-supervised oracle and beat a neural-only baseline — especially with little data; (ii)
recover the *correct* latent digit grounding, or does it exploit reasoning shortcuts; (iii) admit a
cheap fix when grounding fails; and (iv) generalize compositionally to multi-digit addition it never
saw?

## Background and Motivation
See Motivation & Novelty above and `literature_review.md`. NeSy approach used here is the compact
**differentiable marginalization** form of DeepProbLog/Scallop: the network outputs per-image digit
distributions and P(sum=s)=Σ_{a+b=s} P(a)P(b) is computed by convolving the two distributions; the
symbolic operation (+) is fixed and exact, only perception is learned.

## Hypothesis Decomposition
- **H1 (capability):** NeSy test sum-accuracy > neural-only, with the gap largest at small train
  sizes; NeSy ≈ oracle at moderate+ data.
- **H2 (grounding gap):** Test sum-accuracy overstates latent digit-grounding accuracy; the gap is
  larger in the low-data regime; some seeds exhibit *systematic* shortcuts (permutation/collapse)
  detectable by Hungarian-aligned grounding > raw grounding.
- **H3 (mitigation):** A small grounding signal (few labeled digits) and/or a marginal-uniformity
  prior raises grounding accuracy toward the oracle without hurting sum accuracy.
- **H4 (compositional OOD):** NeSy (with correct grounding) solves 2-digit addition zero-shot via
  symbolic place-value composition; neural-only ≈ chance/zero; OOD success tracks grounding accuracy.

Independent variables: model type, #train pairs, mitigation type/strength, #digits at test.
Dependent variables: test sum-accuracy, digit-grounding accuracy (raw + Hungarian-aligned),
multi-digit accuracy, training time.

## Proposed Methodology

### Approach
Implement all three models on a **shared CNN architecture** so the only difference is the
learning signal / output head — a clean ablation isolating the symbolic structure as the variable.
- **Neural-only:** shared CNN encodes each image; features concatenated → MLP → 19-way softmax over
  sums {0..18}. No knowledge that sum=d1+d2.
- **NeSy:** shared CNN → 10-way digit logits per image; P(sum)=conv(P(d1),P(d2)); NLL on true sum.
  Symbolic addition fixed and exact. Distant supervision (sum only).
- **Oracle:** same 10-way CNN trained with digit cross-entropy on true labels (upper bound on
  grounding); sum via argmax. Bounds achievable performance.

### Experimental Steps
1. **E1 Data efficiency:** train sizes {100,500,1000,5000,15000,30000} × {neural,nesy,oracle} × 5
   seeds. Metric: test sum-accuracy. Learning curves + per-size Welch t-tests (NeSy vs neural) and
   Cohen's d.
2. **E2 Grounding gap:** for nesy & oracle runs in E1, compute digit-grounding accuracy on test
   images (argmax 10-way vs true label), plus **Hungarian-aligned** grounding (best label
   permutation) to flag systematic shortcuts; report sum-acc − grounding-acc gap vs train size.
3. **E3 Mitigations** (at a low-data size where the gap is largest): (a) +k labeled digits
   (k∈{0,10,50,200}) via auxiliary digit CE; (b) marginal-uniformity prior (KL of batch-mean digit
   distribution to uniform, weight λ); (c) entropy regularization. 5 seeds each. Metric: grounding
   accuracy + sum accuracy.
4. **E4 Compositional OOD:** build a 2-digit-number addition test set from MNIST (two 2-digit
   numbers → sum 0..198). Evaluate the *single-digit-trained* classifiers composed with symbolic
   place-value sum (NeSy, oracle); neural-only has no compatible output → report its structural
   failure. Correlate OOD accuracy with single-digit grounding accuracy across seeds.

### Baselines
Neural-only CNN sum-classifier (the key contrast), and a digit-supervised oracle (upper bound), per
`literature_review.md` §4. NeSy is the differentiable-marginalization DeepProbLog/Scallop family.

### Evaluation Metrics
- **Test sum-accuracy** (primary task metric).
- **Digit-grounding accuracy** (raw and Hungarian-aligned) — detects shortcuts.
- **Multi-digit (OOD) accuracy** — compositional generalization.
- **Training wall-clock** — secondary.
Justification: directly from `literature_review.md` §5; grounding + permutation-alignment is the
under-reported axis this study adds.

### Statistical Analysis Plan
5 seeds per configuration. Report mean ± 95% CI (t-based; bootstrap as robustness check). Primary
test: **Welch's two-sample t-test** (unequal variance) for NeSy vs neural-only at each train size;
report **Cohen's d** effect sizes. Significance α=0.05; **Holm–Bonferroni** correction across the 6
train-size comparisons. For H4, Pearson/Spearman correlation between grounding accuracy and OOD
accuracy across seeds.

## Expected Outcomes
- **Supports H1** if NeSy ≫ neural-only at small N and ≈ oracle at large N (p<0.05, large d).
- **Supports H2** if grounding-acc < sum-acc with a widening gap at low data, and ≥1 seed shows
  Hungarian-aligned ≫ raw grounding (systematic shortcut).
- **Supports H3** if a mitigation moves grounding toward oracle at fixed sum accuracy.
- **Supports H4** if NeSy OOD accuracy ≫ 0 and correlates with grounding; neural-only ≈ 0.
- **Refutes** correspondingly (e.g., if grounding is already near-perfect everywhere → MNIST-Addition
  is shortcut-robust at single digit, itself a clean, honest, reportable result).

## Timeline and Milestones
- Phase 0–1 planning: done here. (~30 min)
- Phase 2 data/code: ~40 min (data loaders, 3 models, trainer, metrics).
- Phase 4 experiments E1–E4: ~60 min (CNNs train in ~10–30 s each on A6000; ~150 runs, GPU-parallel).
- Phase 5 analysis + figures: ~30 min. Phase 6 report: ~25 min. ~20% debug buffer included.

## Potential Challenges
- **Reasoning shortcuts may be rare at single-digit** (boundary sums 0,18 strongly pin grounding):
  if so, induce/observe them in the low-data regime and report honestly — either outcome is a result.
- **Inference cost / scale:** kept tiny by design (single-digit marginalization is a 10×10 conv).
- **Seed variance:** addressed by 5 seeds + CIs.
- **OOD test fairness:** neural-only genuinely cannot emit sums>18; we report this as a structural
  (not tuning) limitation and also give it the most charitable possible scoring.

## Success Criteria
The study succeeds if it delivers, with statistical support, a clear answer to each of H1–H4 and a
quantified grounding gap with at least one concrete mitigation — regardless of which direction the
hypotheses resolve. Reproducibility (fixed seeds, saved configs/results, runnable scripts) is
required.

## Reproducibility
Fixed seeds; all hyperparameters in `src/config.py`; raw per-run results saved to `results/raw/`;
aggregate to `results/`; figures to `figures/`; environment pinned via `pyproject.toml`/`uv.lock`.
