# Resources Catalog — Neural-Symbolic Learning

## Summary
Resources gathered for the research project *"Integration of neural and symbolic learning
approaches to achieve novel or improved capabilities."*

- **Papers downloaded:** 20 PDFs (`papers/`) — selected from 121 relevance-ranked results.
- **Datasets prepared:** MNIST + the canonical **MNIST-Addition** NeSy benchmark (`datasets/`).
- **Code repositories cloned:** 6 reference NeSy frameworks (`code/`).
- **No user-specified resources** were provided in the research spec, so selection was driven by
  the relevance ranking, foundational importance, and runnability.

---

## Papers
Total: **20** (see `papers/README.md` for abstracts; full ranked list in
`paper_search_results/ranked_papers.json`).

| # | Title | Year | File |
|---|-------|------|------|
| 1 | The Neuro-Symbolic Concept Learner (NS-CL) | 2019 | `papers/nscl_concept_learner_2019.pdf` |
| 2 | DeepProbLog: Neural Probabilistic Logic Programming | 2018 | `papers/deepproblog_2018.pdf` |
| 3 | Learning Explanatory Rules from Noisy Data (∂ILP) | 2017 | `papers/dilp_explanatory_rules_2017.pdf` |
| 4 | Neurosymbolic AI: the 3rd wave | 2020 | `papers/neurosymbolic_3rd_wave_2020.pdf` |
| 5 | Neural-Symbolic Computing: An Effective Methodology | 2019 | `papers/nesy_computing_methodology_2019.pdf` |
| 6 | Logic Tensor Networks | 2020 | `papers/logic_tensor_networks_2020.pdf` |
| 7 | NeurASP: Embracing NNs into Answer Set Programming | 2020 | `papers/neurasp_2020.pdf` |
| 8 | From Statistical Relational to Neurosymbolic AI (survey) | 2021 | `papers/starai_to_nesy_survey_2021.pdf` |
| 9 | Neural-Symbolic VQA (NS-VQA) | 2018 | `papers/ns_vqa_2018.pdf` |
| 10 | A Semantic Loss Function for DL with Symbolic Knowledge | 2017 | `papers/semantic_loss_2017.pdf` |
| 11 | Scallop: A Language for Neurosymbolic Programming | 2023 | `papers/scallop_lang_2023.pdf` |
| 12 | NeuPSL: Neural Probabilistic Soft Logic | 2022 | `papers/neupsl_2022.pdf` |
| 13 | Deep Symbolic Learning: Discovering Symbols and Rules | 2022 | `papers/deep_symbolic_learning_2022.pdf` |
| 14 | Softened Symbol Grounding for Neuro-symbolic Systems | 2024 | `papers/softened_symbol_grounding_2024.pdf` |
| 15 | LINC: Neurosymbolic Logical Reasoning with LMs + Provers | 2023 | `papers/linc_2023.pdf` |
| 16 | Towards Data-and Knowledge-Driven AI (NeSy survey) | 2022 | `papers/data_knowledge_driven_survey_2022.pdf` |
| 17 | pix2rule: End-to-end Neuro-symbolic Rule Learning | 2021 | `papers/pix2rule_2021.pdf` |
| 18 | Neurosymbolic RL and Planning: A Survey | 2023 | `papers/nesy_rl_planning_survey_2023.pdf` |
| 19 | Improving Rule-based Reasoning in LLMs (Neurosymbolic) | 2025 | `papers/improving_rule_reasoning_llm_2025.pdf` |
| 20 | Scalable Neural-Probabilistic Answer Set Programming | 2023 | `papers/scalable_neural_prob_asp_2023.pdf` |

---

## Datasets
Total prepared: **MNIST** (+ derived **MNIST-Addition**). See `datasets/README.md` for full
download/build/load instructions. Large data excluded from git via `datasets/.gitignore`.

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| MNIST | LeCun / deepproblog repo | 70k imgs, ~11 MB | digit classification (perception) | `datasets/mnist/` | raw IDX + `mnist.npz` |
| MNIST-Addition | derived from MNIST | 30k train / 5k test pairs | sum-of-two-digits (distant supervision) | `datasets/mnist/mnist_addition/` | **primary NeSy benchmark**; CSV index files kept in git |
| CLEVR | Stanford / FAIR | ~18 GB | compositional VQA | (instructions only) | for NS-VQA / NS-CL |
| FOLIO | yale-nlp (HF) | ~1.4k, small | NL→FOL reasoning | (instructions only) | for LINC-style LLM NeSy |
| ProofWriter | AllenAI | medium | deductive reasoning | (instructions only) | for LINC-style LLM NeSy |

---

## Code Repositories
Total cloned: **6** (shallow). Excluded from git via `code/.gitignore`; see `code/README.md`
for per-repo entry points, requirements, and run commands.

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| DeepProbLog | github.com/ML-KULeuven/deepproblog | Prob. logic + neural predicate; ships MNIST-Addition | `code/deepproblog/` |
| NeurASP | github.com/azreasoners/NeurASP | NN outputs as ASP facts; Sudoku, MNIST-Add | `code/NeurASP/` |
| Logic Tensor Networks | github.com/logictensornetworks/logictensornetworks | Differentiable first-order fuzzy logic | `code/logictensornetworks/` |
| Scallop | github.com/scallop-lang/scallop | Differentiable Datalog (`scallopy`) | `code/scallop/` |
| Semantic-Loss | github.com/UCLA-StarAI/Semantic-Loss | Constraint-aware semantic loss | `code/Semantic-Loss/` |
| NS-VQA | github.com/kexinyi/ns-vqa | Neural-symbolic VQA on CLEVR | `code/ns-vqa/` |

---

## Resource Gathering Notes

### Search Strategy
- Used the **paper-finder** service (diligent mode, query "neural-symbolic integration learning
  reasoning") → 121 relevance-ranked papers with abstracts (cached in `paper_search_results/`).
- Selected 20 PDFs spanning four pillars: (a) foundational methods (DeepProbLog, NeurASP, LTN,
  Semantic Loss, ∂ILP, NS-VQA, NS-CL), (b) scalability/modern systems (Scallop, NeuPSL, SLASH,
  Deep Symbolic Learning, Softened Symbol Grounding, pix2rule), (c) surveys, (d) LLM-era NeSy
  (LINC, LLM rule-reasoning).
- Cloned the canonical reference implementations for the top methods.

### Selection Criteria
Relevance score (focused on rel=3), citation count (foundational works), recency (2023–2025 for
SOTA), code availability/runnability, and coverage of the design-space taxonomy.

### Challenges Encountered
- The installed `arxiv` library's `download_pdf` API differed; switched to resolving arXiv IDs via
  search then downloading PDFs with `requests`.
- The fuzzy title→arXiv matcher initially mis-grabbed several unrelated (physics) papers; these
  were detected by extracting page-1 titles with `pypdf` and re-downloaded using **verified arXiv
  IDs** (e.g., ∂ILP 1711.04574, 3rd-wave 2012.05876, StaRAI survey 2108.11451, LINC 2310.15164,
  Semantic Loss 1711.11157, Scallop 2304.04812). All 20 final PDFs were title-validated.
- "Neurosymbolic Programming" (Chaudhuri et al.) is a Foundations & Trends monograph not on arXiv —
  substituted the closely related Scallop language paper.

### Gaps and Workarounds
- Large datasets (CLEVR ~18 GB) are documented with download instructions rather than fetched.
- MNIST-Addition is not distributed as a standalone dataset; it was **generated deterministically**
  from MNIST (seeds 42/7) and saved as small CSV index files so the experiment runner has an
  immediately usable, git-friendly benchmark.

---

## Recommendations for Experiment Design

1. **Primary dataset:** MNIST-Addition (`datasets/mnist/mnist_addition/`) — local, fast, canonical.
2. **Baselines:** (i) neural-only CNN sum-classifier; (ii) digit-supervised CNN oracle (upper bound);
   (iii) optionally DeepProbLog / NeurASP as published NeSy references.
3. **Metrics:** sum-accuracy (primary), **learning curves / data efficiency** (vary train size),
   **latent digit-grounding accuracy**, and OOD/compositional generalization.
4. **Code to adapt/reuse:** `code/deepproblog/.../examples/MNIST/` for loaders + the symbolic
   program; `scallopy` (in `code/scallop/`) or a compact custom differentiable-marginalization
   implementation for the neural-symbolic model.
5. **Headline question to answer:** does integrating a fixed symbolic `sum` operation with a shared
   neural digit classifier (trained on *sums only*) match a fully digit-supervised CNN and beat the
   neural-only baseline — particularly in the low-data regime — while recovering correct groundings?
   The literature strongly predicts yes; verifying it on identical splits is the experiment.
