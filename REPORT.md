# When Does Neural-Symbolic Integration *Understand*? Data Efficiency, Reasoning Shortcuts, and Compositional Generalization on MNIST-Addition

**Domain:** Artificial Intelligence — Neuro-Symbolic learning
**Benchmark:** MNIST-Addition (canonical NeSy task)
**Date:** 2026-06-28 · **Hardware:** NVIDIA RTX A6000 (CUDA), PyTorch 2.12

---

## 1. Executive Summary

We ask whether wiring a *fixed symbolic addition operation* into a shared neural digit
classifier — trained only on the **sum** of two digit images, never on the digits themselves —
produces a system that genuinely *perceives and reasons*, and under what conditions that promise
holds. Across 210 training runs (5 seeds each), we find a sharp, three-part answer.

**(1) Integration is a large, statistically significant capability win.** The neuro-symbolic (NeSy)
model matches a fully digit-supervised oracle and dramatically beats a pure-neural baseline,
especially with little data: at 1,000 training pairs it reaches **85.4%** sum accuracy vs the
neural baseline's **22.0%** (Welch *t*, *p*=2.9e-13, Cohen's *d*=56), and it recovers the latent
digits at **92–99%** accuracy *with no digit labels at all*.

**(2) But high task accuracy can hide a broken mind — and we show exactly when.** The same
architecture, data, and training, with the *only* change being the symbolic operation (`a+b` →
`(a+b) mod 10`), makes correct grounding **never emerge**: among the seeds that learn the modular
task at all, **raw grounding is ≈0.1%** (mean 0.09%, max 0.16%) — the network solves the task by
reading the digits **systematically wrong** via a global relabeling that leaves `(a+b) mod 10`
invariant. The shortcut is a **systematic off-diagonal permutation** (Figure 3), occasionally an
*exact* cyclic shift (e.g. `[5,6,7,8,9,0,1,2,3,4]`, every *d* perceived as *(d+5) mod 10*). Training
is also unstable: in our 5-seed grid, **4/5 seeds find a shortcut and 1/5 collapses** (fails to fit,
task accuracy 0.11). A fully digit-supervised oracle on the *same* modular task grounds at **99.0%**,
so the failure is caused by distant supervision under a symmetric operator, not by the task being
unlearnable. We further show **10 labeled images total** (one per class) eliminate the shortcut and
stabilize training (grounding ≈0.1% → **97.7%**, *p*=1.6e-3; every seed), whereas unsupervised
regularizers do not.

**(3) Correct grounding unlocks a capability neural nets structurally lack.** A NeSy model trained
**only on single-digit sums** solves **2- and 3-digit** number addition it has *never seen*, at
**95.7%** and **93.7%**, by composing its digit classifier with a symbolic place-value sum. The
neural baseline scores **exactly 0%** — it cannot even represent the output. *Practical implication:*
in NeSy systems, **task accuracy is not enough**; grounding must be measured, the task's symmetry
determines whether grounding is identifiable, and a tiny amount of symbol supervision is a cheap,
decisive fix.

---

## 2. Research Question & Motivation

**Question.** On MNIST-Addition, does integrating a fixed symbolic `+` with a shared neural digit
classifier trained on *sums only* (i) match an oracle and beat a neural baseline, especially with
little data; (ii) recover the *correct* latent symbols or exploit reasoning shortcuts; (iii) admit a
cheap fix when grounding fails; and (iv) generalize compositionally to multi-digit addition never
seen in training?

**Why it matters.** NeSy AI promises perception + reasoning in one system, with data efficiency,
interpretability, and reusable symbols. But if the learned latent symbols are *wrong*, every
downstream promise collapses silently — and a practitioner watching only task accuracy will never
notice. Our literature review (`literature_review.md` §6) flags two under-reported gaps: the
"NeSy is data-efficient" claim is rarely measured head-to-head with confidence intervals on
identical splits, and **grounding accuracy is rarely reported**, so reasoning shortcuts are
invisible. Softened Symbol Grounding (2024) and Deep Symbolic Learning (2022) raise the shortcut
problem qualitatively; a controlled, multi-seed *measurement* of when shortcuts appear, what they
look like, and which cheap fixes work has been missing.

**Contribution.** A single controlled study on one benchmark that measures, side-by-side and with
statistics, four quantities the literature usually reports in isolation: **(E1) data efficiency**,
**(E2/E5) the task-accuracy–vs–grounding gap with a permutation-aware shortcut detector**,
**(E3/E5) mitigations**, and **(E4) compositional OOD generalization** — tied together by a clean
"flip one symbolic operator" manipulation (`+` vs `+ mod 10`) that turns shortcuts on and off while
holding everything else fixed.

---

## 3. Experimental Setup

### Models (identical shared CNN encoder — only the head/learning signal differs)
- **Neural-only baseline.** Shared LeNet-style encoder applied to each image; the two feature
  vectors are concatenated → MLP → 19-way softmax over sums {0..18}. Knows nothing about the
  additive structure. (For modular addition: 10-way head.)
- **Neuro-Symbolic (NeSy).** Shared encoder → 10-way digit log-probabilities per image. The sum
  distribution is computed *exactly and differentiably* as `P(sum=s)=Σ_{a+b=s}P(a)P(b)` (the
  convolution of the two digit pmfs — the DeepProbLog/Scallop idea in compact form). Trained with
  NLL on the **true sum only** (distant supervision). The symbolic `+` is fixed, never learned.
- **Oracle (upper bound).** The same 10-way CNN trained with direct digit cross-entropy on the true
  labels; bounds achievable grounding/accuracy.

### Symbolic operators
- **Standard addition:** output 0..18.
- **Modular addition `(a+b) mod 10`:** output 0..9. Chosen deliberately: it removes the boundary
  constraints (sum 0⇒0+0, sum 18⇒9+9) that pin the grounding, and admits an exact affine shortcut
  `d ↦ (d+5) mod 10` (and other cyclic relabelings), so it *induces* shortcuts under distant
  supervision. The oracle is unaffected because it sees true labels.

### Data
MNIST-Addition, pre-built locally (`datasets/mnist/mnist_addition/`): 30,000 train pairs, 5,000 test
pairs, indices into MNIST `x_train`/`x_test`. True digit labels `y_*` are used **only** for
evaluation (grounding) and for the oracle / mitigation conditions. Data integrity was verified
(true digits sum to the stored target; test indices map exactly to `x_test`). For E4 we
deterministically synthesize 2- and 3-digit number-addition test sets (5,000 examples each) from
MNIST test images.

### Metrics
- **Test sum accuracy** (primary task metric).
- **Digit-grounding accuracy** — raw (argmax digit vs true label) and **Hungarian-aligned** (best
  10×10 label permutation, via the assignment problem). When *aligned ≫ raw*, the network learned a
  *systematic* relabeling — a reasoning shortcut. We flag `is_shortcut` when aligned − raw > 0.10.
- **Compositional OOD accuracy** — multi-digit addition by a single-digit-trained model.
- **Training wall-clock** (secondary).

### Protocol & reproducibility
5 seeds {0..4} per configuration; mean ± 95% t-CI; **Welch's** two-sample *t*-tests (unequal
variance) with **Holm–Bonferroni** correction across train sizes; **Cohen's *d*** effect sizes;
Pearson correlation for H4. Epochs=12, batch=128, Adam lr=1e-3, fixed across all conditions
(`src/config.py`). Seeds set for Python/NumPy/Torch. The full grid (E1+E5 = 165 runs; E3=30; E4=15)
runs in ~5 minutes total across 3 GPUs. Raw per-run results in `results/raw/`; aggregates in
`results/stats.json`; figures in `figures/`.

---

## 4. Results

### E1 — Data efficiency (standard addition): integration is a large win, all sizes significant
NeSy beats the neural baseline at **every** training size (Holm-corrected *p*<0.01 throughout),
with the gap largest in the low-data regime, and tracks the oracle closely.

| # train pairs | Neural-only | **NeSy (sum-only)** | Oracle | NeSy−Neural | Cohen's *d* | Welch *p* (raw) | Holm-adj. *p* |
|--------------:|------------:|--------------------:|-------:|------------:|------------:|----------------:|--------------:|
| 100   | 0.086 | **0.130** | 0.429 | +0.044 | 2.28 | 8.8e-3 | 8.8e-3 |
| 500   | 0.130 | **0.649** | 0.794 | +0.519 | 5.43 | 9.3e-4 | 1.9e-3 |
| 1,000 | 0.220 | **0.854** | 0.868 | +0.634 | 56.3 | 2.9e-13 | 1.7e-12 |
| 5,000 | 0.883 | **0.955** | 0.958 | +0.072 | 10.25 | 1.6e-6 | 7.9e-6 |
| 15,000| 0.952 | **0.973** | 0.975 | +0.021 | 4.82 | 3.8e-4 | 1.5e-3 |
| 30,000| 0.970 | **0.980** | 0.981 | +0.010 | 4.13 | 4.8e-4 | 1.5e-3 |

*(All Holm-adjusted p < 0.01 ⇒ every comparison remains significant after family-wise correction.)*

*(Figure 1a.)* The neural baseline needs ~5,000 pairs to reach what NeSy achieves with ~1,000.

### E2 — Grounding emerges *without* digit labels (standard addition)
*(Figure 1b.)* NeSy's latent digit-grounding rises from 28.3% (100 pairs) to **99.0%** (30k pairs),
essentially matching the digit-supervised oracle (99.0%) — even though NeSy never saw a digit label.
The Hungarian-aligned grounding equals the raw grounding (**0/5 shortcut flags at every size**):
standard MNIST-Addition is **shortcut-robust**, because the boundary sums uniquely pin the symbols.
The apparent sum-vs-grounding gap (e.g. 85.4% sum vs 92.4% grounding at 1k) is just the
"both-digits-must-be-right" effect (sum ≈ grounding²), **not** a shortcut.

### E5 — Inducing shortcuts: flip the operator, grounding collapses
*(Figures 2, 3.)* Changing only the symbolic operator to `(a+b) mod 10`:

| Task (NeSy, N=30k) | sum acc (mean±CI) | grounding (raw) | Hungarian-aligned | outcome over 5 seeds |
|--------------------|------------------:|----------------:|------------------:|----------------------|
| standard `a+b`        | 0.980 ± 0.002 | **0.990** | 0.990 | 0/5 shortcut (grounds correctly) |
| modular `(a+b) mod 10`| 0.657 ± 0.380 | **0.020** | 0.471 | **4/5 shortcut + 1/5 task-collapse** |
| modular — *oracle* (digit-supervised) | 0.979 | 0.990 | 0.990 | 0/5 (control: same task, grounds fine) |

Among the 4 seeds that actually learn the modular task (sum>0.3), raw grounding is **0.09%** (max
0.16%) — they read the digits systematically wrong via a global relabeling that leaves `(a+b) mod 10`
invariant (Figure 3: clean off-diagonal permutations; one seed lands on the exact `(d+5)` cyclic
shift). The 5th seed collapses to predicting a single class (Figure 3, rightmost). The
digit-supervised **oracle grounds at 99.0% on the identical modular task**, so the collapse of
grounding is caused by *distant supervision under a symmetric operator*, not by the task. Standard vs
modular grounding: **0.990 vs 0.020**, Welch *p*=9.6e-7, Cohen's *d*=31.5. This reasoning shortcut is
visible *only* because we measured grounding; task accuracy alone is blind to it.

### E3 / E5 — Mitigations: needed and effective exactly when a shortcut exists
*(Figure 5.)* Grounding accuracy under each intervention (standard add evaluated at N=1,000, the regime where its
grounding gap is largest; modular add at N=5,000, where the shortcut is firmly established — each
column is a *within-task* before/after, so the differing N does not affect the conclusion):

| Intervention | Standard add @N=1k (no shortcut) | Modular add @N=5k (shortcut present) |
|--------------|---------------------------------:|-------------------------------------:|
| baseline (none)              | 0.924 | 0.116 (4/5 ≈0; 1 seed 0.57) |
| **+ aux digit labels, 1/class (10 total)** | 0.925 | **0.977** |
| + aux 5/class (50 total)     | 0.926 | 0.977 |
| + aux 20/class (200 total)   | 0.931 | 0.979 |
| + marginal-uniformity prior  | 0.924 | 0.390 |
| + entropy regularization     | 0.925 | 0.128 |

On standard addition, mitigations do essentially nothing (there is no shortcut to fix). On modular
addition, **just 10 labeled images** break the symmetry and restore grounding to **97.7% in every
seed** (per-seed 0.973–0.981; baseline per-seed grounding 0.001–0.005 for the shortcut seeds, with one
partial seed at 0.57 inflating the baseline mean to 0.116; Welch *p*=1.6e-3, *d*=4.8), while also
*stabilizing* training (task accuracy → 0.954, no collapses). Unsupervised regularizers help only
partially (uniform prior, high variance) or hurt (entropy). *Take-away: a pinch of grounding
supervision is the decisive lever; clever unsupervised priors are not a reliable substitute.*

### E4 — Compositional generalization: the payoff of correct grounding
*(Figure 4.)* Trained on **single-digit** sums only, then evaluated zero-shot on multi-digit number
addition by composing the digit classifier with a symbolic place-value sum:

| Model | 2-digit addition (OOD) | 3-digit addition (OOD) | single-digit grounding |
|-------|-----------------------:|-----------------------:|-----------------------:|
| Neural-only | **0.000** | **0.000** | 0.000 (n/a) |
| **NeSy** | **0.957** | **0.937** | 0.990 |
| Oracle | 0.960 | 0.940 | 0.990 |

The neural baseline cannot even emit sums in this range (structural failure → 0%). NeSy generalizes
to a task structure it never saw, *because* its grounding is correct — the single most direct
evidence that integration buys a genuinely new capability, not just a benchmark number.

---

## 5. Analysis & Discussion

**H1 (capability) — supported, strongly.** Integrating the symbolic operator factorizes the task
into digit recognition, so the model learns from far fewer examples and matches the oracle. Effect
sizes are large at every size and overwhelming at 1k pairs (*d*=56). This quantifies, with CIs on
identical splits, the data-efficiency claim the literature usually only asserts.

**H2 (grounding gap / shortcuts) — nuanced and, we argue, the most useful finding.** Whether high
task accuracy implies correct symbols is **not** a property of NeSy in general — it is a property of
the *task's symmetry*. Standard addition is shortcut-robust (its boundary constraints identify the
symbols); modular addition is shortcut-prone (its rotational symmetry does not). The same network
swings from 99.0% grounding to ≈0.1% (among task-learning seeds) when we flip one operator, while the
oracle — which sees digit labels — grounds at 99% on *both* operators (the clean control). The
Hungarian-aligned metric is what makes the failure legible: aligned grounding (≈0.53–0.60 on shortcut
seeds, up to 0.99 on the cleanest cyclic-shift realization) sits far above the ≈0.1% raw value,
revealing a *systematic relabeling* rather than random noise. **Methodological implication:** report
grounding, and report it permutation-aligned; sum accuracy alone is blind to this.

**H3 (mitigation) — supported.** The cure matches the disease: when identifiability is the problem,
a tiny amount of identifying information (10 labels → grounding 0.977 in every seed) fixes it *and*
removes the training instability; unsupervised regularizers, which do not add identifying
information, mostly cannot. This reframes "mitigation" as "supply the missing constraint," and shows
it is extremely cheap when needed and unnecessary when not.

**H4 (compositional OOD) — supported, with an honest caveat.** Correct grounding is what lets the
symbolic composition extend to unseen problem sizes; the neural baseline's structural 0% is the
foil. Across the grounded models, single-digit grounding and OOD accuracy correlate almost perfectly
(Pearson *r*=0.993, *p*=1e-8) — but we note this correlation spans models that *all* ground well
(NeSy, oracle), so the **categorical** contrast (0.00 for the ungrounded neural baseline vs ~0.96
for grounded models) is the stronger evidence; the within-grounded correlation has little spread and
should not be over-read.

**Surprises.** (i) That a single operator flip makes correct grounding vanish so reliably (raw
grounding ≈0.1% in every seed that learns the task), and that the model sometimes lands on the
*exact* algebraically predicted `(d+5)` cyclic permutation, was sharper than expected — as was the
training *instability* it induces (4/5 shortcut, 1/5 outright collapse). (ii) The neural baseline's high
*variance* under modular addition at 30k (mean 0.79, sd 0.39: four seeds at 0.96–0.97 but one seed
stuck at 0.10) — it occasionally fails to fit mod-10 at all — hints that the modular task is genuinely
harder to represent without the additive factorization.

**Error analysis.** NeSy's residual errors on standard addition occur on genuinely ambiguous digit
images (consistent with its ~99% correct grounding), i.e. they are *perceptual*. NeSy "errors" under
modular addition are *not* perceptual — they stem from a systematic global relabeling of the symbols
(Figure 3), which is precisely why high task accuracy survives them.

---

## 6. Limitations

- **Single benchmark / single perceptual domain.** All results are on MNIST digits. The *mechanism*
  (task symmetry ⇒ (non)identifiability of symbols) should transfer, but we did not test other
  modalities (e.g. CLEVR, HWF) — those datasets were documented but not run here for scope/time.
- **One symbolic relaxation.** We use exact differentiable marginalization (DeepProbLog/Scallop
  family). We did not run DeepProbLog/NeurASP/Scallop *implementations* themselves as additional
  NeSy references; our NeSy model is a faithful compact re-implementation of their core idea, not a
  benchmark of those toolkits.
- **H4 correlation has low spread** (see §5) — we rely on the categorical contrast for the main
  claim and flag the correlation as suggestive only.
- **GPU non-determinism and modular instability.** Atomic `index_add_`/conv-backward make runs not
  bit-identical even with fixed seeds; the *specific* permutation a modular run lands on, and whether
  a given seed shortcuts vs. collapses, varies across re-runs. The robust, re-run-invariant phenomenon
  is that under the modular operator **correct grounding never emerges** (raw grounding ≈0.1% whenever
  the task is learned; the oracle, by contrast, always grounds at 99%). Because modular NeSy is
  unstable (here 4/5 shortcut + 1/5 collapse), aggregate statistics over seeds — not single runs — are
  the unit of inference, and Figure 3 reads the exact saved grid runs rather than fresh retrains.
  Standard-addition results are stable across re-runs (e.g. NeSy@1k reproduced to within ±0.01).
- **Modular addition is a deliberately constructed shortcut-inducer**, not a naturally occurring
  task; it is a controlled probe of the mechanism, chosen for its exact, analyzable symmetry.

---

## 7. Conclusions & Next Steps

**Answer to the research question.** Integrating a fixed symbolic operation with a shared neural
classifier (i) is a large, statistically significant capability and data-efficiency win that matches
a fully-supervised oracle and beats a neural baseline; (ii) recovers the *correct* latent symbols
**iff the task's structure identifies them** — when it does not (modular addition), the system hits
high task accuracy via systematic reasoning shortcuts that only a grounding/permutation metric can
detect; (iii) is cheaply repaired with ~10 labeled symbols exactly when a shortcut exists; and (iv)
when grounding is correct, unlocks zero-shot compositional generalization (multi-digit addition at
~95%) that the neural baseline structurally cannot achieve (0%).

**Practical guidance for NeSy practitioners.** Always measure latent grounding, permutation-aligned,
alongside task accuracy. Analyze your symbolic operator for symmetries that fail to identify the
symbols. If shortcuts are possible, add a handful of labeled symbols — it is the cheapest, most
reliable fix and beats unsupervised regularization.

**Next steps.** (a) Replicate the symmetry→(non)identifiability mechanism on HWF and a relational
task (CLUTRR); (b) test richer shortcut-inducing operators (parity, sum-mod-*m* for various *m*) to
map identifiability as a function of operator algebra; (c) compare our compact marginalization
against DeepProbLog/Scallop on the *same* grounding metric; (d) study an active-learning loop that
requests the *minimum* labeled symbols to break a detected shortcut.

---

## References (used in this study)
- Manhaeve et al. (2018) **DeepProbLog: Neural Probabilistic Logic Programming** — `papers/deepproblog_2018.pdf` (MNIST-Addition; differentiable marginalization).
- Li et al. (2023) **Scallop: A Language for Neurosymbolic Programming** — `papers/scallop_lang_2023.pdf` (provenance-semiring differentiable Datalog).
- Yang et al. (2020) **NeurASP** — `papers/neurasp_2020.pdf`.
- Li et al. (2024) **Softened Symbol Grounding for Neuro-symbolic Systems** — `papers/softened_symbol_grounding_2024.pdf` (reasoning shortcuts / grounding).
- Deep Symbolic Learning (2022) — `papers/deep_symbolic_learning_2022.pdf`.
- Xu et al. (2017) **A Semantic Loss Function** — `papers/semantic_loss_2017.pdf`.
- Badreddine et al. (2020) **Logic Tensor Networks** — `papers/logic_tensor_networks_2020.pdf`.
- Garcez & Lamb (2020) **Neurosymbolic AI: the 3rd wave** — `papers/neurosymbolic_3rd_wave_2020.pdf`.
- Marra et al. (2021) **From Statistical Relational to Neurosymbolic AI** — `papers/starai_to_nesy_survey_2021.pdf`.
- Full literature synthesis: `literature_review.md`; resource catalog: `resources.md`.

**Tools:** PyTorch 2.12 (+CUDA), NumPy, SciPy (stats, Hungarian assignment), scikit-learn,
Matplotlib. **Data:** MNIST-Addition (`datasets/`). **Code:** `src/` (this study);
reference NeSy repos in `code/`.
