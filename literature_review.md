# Literature Review: Neural-Symbolic Learning

**Research hypothesis.** *Investigate the integration of neural and symbolic learning
approaches to achieve novel or improved capabilities.*

**Scope.** This review synthesizes 20 downloaded papers (`papers/`) selected from a
relevance-ranked set of 121 (paper-finder, diligent mode). It targets the practical question
the experiment runner must answer: *what concrete, runnable setup best demonstrates that
combining neural perception with symbolic reasoning improves a measurable capability?*

---

## 1. Research Area Overview

Neural networks excel at perception and pattern recognition from raw data but are weak at
systematic, compositional, data-efficient reasoning and offer little interpretability or a
clean way to inject prior knowledge. Symbolic systems (logic programs, ASP, constraint
solvers) are the mirror image: strong at compositional reasoning, knowledge injection, and
explanation, but brittle and unable to learn from raw perceptual input. **Neuro-Symbolic
(NeSy) AI** seeks to combine the two so that a single system perceives *and* reasons
(Garcez & Lamb, *Neurosymbolic AI: the 3rd wave*, 2020; Yu et al. survey, 2021).

Garcez et al. (*Neural-Symbolic Computing*, 2019) and the StarAI↔NeSy survey (Marra et al.,
2021) frame the design space along recurring dimensions: (1) **how logic and learning are
coupled** — logic compiled *into* the network vs. a differentiable logic layer vs. a neural
front-end feeding a discrete solver; (2) **inference style** — proof-based vs. model-based;
(3) **what is learned** — network parameters only, or also the symbolic rules/structure; and
(4) **how tightly the original logical/probabilistic semantics are preserved**. A structured
NLP review (Hamilton et al., 2022) reports an empirical regularity: systems where **logic is
compiled into the neural network** tend to satisfy the most NeSy goals (reasoning, OOD
generalization, interpretability, low-data learning).

A central recurring obstacle is the **symbol grounding problem** with only distant
supervision: when only the final answer (e.g. a sum) is supervised, the mapping from
sub-symbolic input to symbols is under-constrained and training can collapse to spurious
solutions (Li et al., *Softened Symbol Grounding*, 2024; *Deep Symbolic Learning*, 2022).

---

## 2. Taxonomy of Approaches (with representative downloaded papers)

**A. Probabilistic logic + neural predicates (compile logic into a differentiable circuit).**
- **DeepProbLog** (Manhaeve et al., 2018/2019) — adds a *neural predicate* to ProbLog;
  probabilities of atoms are produced by a NN, inference compiled to SDDs, gradients flow end
  to end. Introduces the **MNIST-Addition** benchmark. Exact inference → does not scale to
  large symbol spaces.
- **NeurASP** (Yang et al., 2020) — NN softmax outputs become a probability distribution over
  atoms in an Answer Set Program; ASP rules both correct perception at inference and supervise
  training via `clingo`.
- **Scalable Neural-Probabilistic ASP / SLASH** (Skryagin et al., 2021/2023) — addresses
  DeepProbLog/NeurASP scalability using probabilistic circuits and approximate solving.
- **NeuPSL** (Pryor et al., 2022) — Neural Probabilistic Soft Logic; combines neural nets with
  PSL's continuous (hinge-loss MRF) relational reasoning for scalable collective inference.
- **Scallop** (Li et al., 2021/2023) — differentiable Datalog with **provenance semirings**;
  a tunable top-k provenance trades exactness for scalability; exposed to PyTorch via `scallopy`.

**B. Differentiable logic as a loss / representation layer.**
- **Logic Tensor Networks** (Badreddine et al., 2020) — "Real Logic", a many-valued
  differentiable first-order logic; formulas become satisfiability objectives. Unifies
  classification, clustering, relational learning, regression.
- **Semantic Loss** (Xu et al., 2017) — a differentiable loss measuring distance from
  satisfying logical constraints over network outputs; strong for semi-supervised learning and
  structured prediction. Knowledge-as-constraint rather than knowledge-as-architecture.

**C. Neural front-end + symbolic program executor (modular).**
- **NS-VQA** (Yi et al., 2018) — scene parser → symbolic scene; question parser → program;
  deterministic executor. ~99.8% on CLEVR with little supervision; robust to long programs and
  highly data-efficient.
- **Neuro-Symbolic Concept Learner (NS-CL)** (Mao et al., 2019) — learns visual concepts,
  words, and semantic parsing jointly from images+QA with **no** explicit concept/program
  labels, via a curriculum and a differentiable program executor.

**D. Learning the symbols / rules themselves (structure learning).**
- **∂ILP — Learning Explanatory Rules from Noisy Data** (Evans & Grefenstette, 2017) — a
  differentiable relaxation of Inductive Logic Programming; induces interpretable first-order
  rules robust to noise.
- **pix2rule** (Cingillioglu & Russo, 2021) — end-to-end learning of rules directly from pixels.
- **Deep Symbolic Learning** (2022) — jointly *discovers* the discrete symbols and the rules
  relating them from perception, instead of assuming a fixed symbol vocabulary.

**E. LLM-era neurosymbolic reasoning (autoformalization + external solver).**
- **LINC** (Olausson et al., 2023) — the LLM translates natural-language problems into
  first-order logic, and an external theorem prover (Prover9) decides; substantially more
  reliable than chain-of-thought on FOLIO / ProofWriter, especially for hard, multi-step
  deduction.
- **Improving Rule-based Reasoning in LLMs using Neurosymbolic Representations** (2025) —
  injects symbolic representations to improve LLM rule-following.

**F. Sequencing / planning.**
- **Neurosymbolic RL and Planning: A Survey** (2023) — maps symbolic structure into RL for
  sample-efficiency, safety, and interpretability.

---

## 3. Standard Benchmarks & Datasets in the Literature

| Benchmark | Task | Used by | Notes |
|-----------|------|---------|-------|
| **MNIST-Addition** | sum two digit-images; only the sum is supervised | DeepProbLog, NeurASP, Scallop, NeuPSL, SLASH, Softened Symbol Grounding | **De-facto standard**; tiny, CPU-friendly; the canonical "perception via symbolic constraint" test. *Pre-built locally in `datasets/mnist/mnist_addition/`.* |
| **Multi-digit / multi-addend addition** | sum of N k-digit numbers | DeepProbLog, SLASH, Scallop | Stresses scalability of inference. |
| **Visual Sudoku** | solve Sudoku from images of digits | NeurASP, Scallop, Softened Symbol Grounding | Combines perception + hard constraints. |
| **HWF (Hand-Written Formula)** | evaluate a handwritten arithmetic expression | DeepProbLog, NGS, Scallop | Larger symbol space; grammar constraints. |
| **CLEVR** | compositional visual question answering | NS-VQA, NS-CL | ~18 GB; download instructions in `datasets/README.md`. |
| **FOLIO / ProofWriter** | NL/FOL deductive reasoning | LINC, LLM-NeSy | For LLM-based neurosymbolic reasoning. |
| **Knowledge graphs (FB15k, WN18, CLUTRR)** | link prediction / relational reasoning | LTN, DeepProbLog (CLUTRR), KG-NeSy surveys | Relational reasoning. |

---

## 4. Common Baselines

- **Neural-only baseline.** The same perception network trained to predict the final label
  directly (e.g., a CNN reading two concatenated images and predicting the 19-way sum). This is
  the key contrast: NeSy methods reach far higher accuracy and far better data efficiency on
  MNIST-Addition because the symbolic structure factorizes the task into digit recognition.
- **Perception-then-symbolic with full digit supervision** (an *oracle/upper-bound*): train the
  CNN on digit labels, then add. Bounds what perfect grounding could achieve.
- **Other NeSy systems** on the same task (DeepProbLog vs. NeurASP vs. Scallop vs. NeuPSL) —
  most papers cross-compare on MNIST-Addition and its multi-digit variants.
- For LLM reasoning: **chain-of-thought / direct prompting** vs. the neurosymbolic
  (autoformalize-then-prove) pipeline (LINC).

---

## 5. Evaluation Metrics

- **Task accuracy** (e.g., exact-match accuracy of the predicted sum / answer). Primary metric.
- **Data efficiency / learning curves** — accuracy vs. number of training examples; NeSy's
  headline advantage. Strongly recommended as a second axis.
- **Generalization to unseen structure** — e.g., train on N-digit, test on (N+k)-digit addition;
  OOD/compositional generalization.
- **Grounding accuracy** — does the latent digit predictor actually recover digits (not just the
  sum)? Detects spurious "reasoning shortcuts."
- **Wall-clock / inference cost** — exact probabilistic inference is the scalability bottleneck;
  several papers (SLASH, Scallop top-k) are evaluated explicitly on time/scaling.
- **Interpretability** (qualitative) — for rule-learners (∂ILP, pix2rule), whether the induced
  rules are correct and human-readable.

---

## 6. Gaps and Opportunities

1. **Scalability of exact inference.** DeepProbLog/NeurASP do exact probabilistic inference,
   which blows up with more digits/objects. Approximate methods (SLASH, Scallop top-k provenance,
   NeuPSL) trade exactness for scale — quantifying that trade-off on a controlled task is a clean,
   tractable experiment.
2. **Reasoning shortcuts / symbol grounding.** With only distant supervision, models can hit high
   task accuracy while learning *wrong* digit groundings (e.g., a consistent permutation). Softened
   Symbol Grounding (2024) and Deep Symbolic Learning (2022) tackle this. Measuring grounding
   accuracy alongside task accuracy is under-reported and easy to add.
3. **Data efficiency quantification.** The "NeSy is more data-efficient" claim is widely asserted;
   a clean head-to-head learning-curve comparison (NeSy vs. CNN baseline) on identical MNIST-Addition
   splits is a high-value, low-cost contribution.
4. **LLM + solver integration.** LINC shows autoformalization+prover beats CoT, but coverage of
   formalization failures and where the LLM mis-translates remains open.
5. **Unified benchmarking.** The field lacks standardized protocols (cf. "How to Think About
   Benchmarking Neurosymbolic AI?", #114 in the ranked list); inconsistent splits/metrics make
   cross-paper comparison hard.

---

## 7. Recommendations for Our Experiment

Given local resources and compute realism, the highest-value, most-defensible experiment:

- **Primary task:** **MNIST-Addition** (pre-built in `datasets/mnist/mnist_addition/`). It is the
  canonical NeSy benchmark, trains in minutes on CPU, and directly tests the hypothesis.
- **Core comparison (the "improved capability"):**
  1. **Neural-only baseline** — CNN over the two concatenated images → 19-way sum classifier.
  2. **Neural-symbolic model** — shared CNN digit classifier (10-way) whose two predictions are
     combined by the *fixed symbolic operation* `sum = d1 + d2`, trained end-to-end with only the
     sum supervised (a differentiable marginalization over digit pairs, i.e., the DeepProbLog/Scallop
     idea implemented compactly). Optionally also run DeepProbLog itself as a reference.
- **Metrics:** sum-accuracy (primary), **learning curves / data efficiency** (vary train size:
  e.g., 100 / 1k / 5k / 30k pairs), **latent digit-grounding accuracy** (using held-out digit labels
  *only for evaluation*), and OOD generalization (train on the standard 0–9 single-digit pairs, test
  on multi-digit addition if time permits).
- **Expected result (from the literature):** the neural-symbolic model should match a fully
  digit-supervised CNN and dramatically beat the neural-only baseline — especially in the low-data
  regime — while also recovering correct digit groundings. This is exactly the "novel/improved
  capability from integration" the hypothesis predicts.
- **Reference implementations to borrow from:** `code/deepproblog/.../examples/MNIST/` (canonical
  loaders + Prolog program), `code/NeurASP/examples/mnistAdd/`, `code/scallop/` (`scallopy`).
- **Watch-outs:** keep an eye on reasoning shortcuts (report grounding accuracy, not just sum
  accuracy); exact-inference methods may be slow if you scale digit count — start small.

---

### Appendix: Surveys for deeper context
- Garcez & Lamb (2020) *Neurosymbolic AI: the 3rd wave* — manifesto & roadmap.
- Marra et al. (2021) *From Statistical Relational to Neurosymbolic AI* — 7-dimension taxonomy.
- Yu et al. (2021) *A survey on neural-symbolic learning systems* — challenges/methods/apps.
- Wang & Yang (2022) *Towards Data-and Knowledge-Driven AI: A Survey on Neuro-Symbolic Computing*.
- *Neurosymbolic RL and Planning: A Survey* (2023) — sequential decision-making.
- Hamilton et al. (2022) — structured review of NeSy in NLP (what actually meets its promise).
