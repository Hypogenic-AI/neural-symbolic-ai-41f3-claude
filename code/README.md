# Cloned Repositories

Reference implementations of major Neural-Symbolic (NeSy) frameworks. These are third-party
repos cloned (shallow, `--depth 1`) for use as baselines / building blocks by the experiment
runner. They are **excluded from git** (`.gitignore`) — re-clone with the commands below.

| Repo | URL | Role | Local path |
|------|-----|------|-----------|
| DeepProbLog | https://github.com/ML-KULeuven/deepproblog | Probabilistic logic + NN (neural predicate); ships MNIST-Addition | `code/deepproblog/` |
| NeurASP | https://github.com/azreasoners/NeurASP | NN outputs as ASP probabilistic facts; Sudoku, MNIST-Add | `code/NeurASP/` |
| Logic Tensor Networks | https://github.com/logictensornetworks/logictensornetworks | Differentiable first-order fuzzy logic (Real Logic) | `code/logictensornetworks/` |
| Scallop | https://github.com/scallop-lang/scallop | Differentiable Datalog / provenance semirings (`scallopy`) | `code/scallop/` |
| Semantic-Loss | https://github.com/UCLA-StarAI/Semantic-Loss | Semantic loss for constraint-aware (semi-)supervised learning | `code/Semantic-Loss/` |
| NS-VQA | https://github.com/kexinyi/ns-vqa | Neural-symbolic VQA on CLEVR (scene parse + program exec) | `code/ns-vqa/` |

### Re-clone
```bash
cd code
git clone --depth 1 https://github.com/ML-KULeuven/deepproblog.git deepproblog
git clone --depth 1 https://github.com/azreasoners/NeurASP.git NeurASP
git clone --depth 1 https://github.com/logictensornetworks/logictensornetworks.git logictensornetworks
git clone --depth 1 https://github.com/scallop-lang/scallop.git scallop
git clone --depth 1 https://github.com/UCLA-StarAI/Semantic-Loss.git Semantic-Loss
git clone --depth 1 https://github.com/kexinyi/ns-vqa.git ns-vqa
```

---

## DeepProbLog  (recommended starting point)
- **Purpose**: Extends ProbLog with a *neural predicate* — probabilistic facts whose
  probabilities come from a neural net. End-to-end training via differentiable inference
  (compiled to SDDs / arithmetic circuits).
- **Key files**:
  - `src/deepproblog/examples/MNIST/addition.py` — the canonical **MNIST-Addition** experiment.
  - `src/deepproblog/examples/MNIST/models/addition.pl` — the symbolic program (`addition(X,Y,Z) :- digit(X,A), digit(Y,B), Z is A+B`).
  - `src/deepproblog/examples/MNIST/network.py` — the CNN digit classifier.
  - Other examples: `HWF/` (hand-written formulas), `Forth/`, `CLUTRR/`, `Coins/`, `Poker/`.
  - **Bundled data**: raw MNIST is already present under `examples/MNIST/models/data/MNIST/`.
- **Requirements** (`requirements.txt`): `problog`, `PySDD`, `torch`, `torchvision`. Approximate
  inference additionally needs SWI-Prolog (<9.0) + `pyswip`. Install:
  `uv pip install problog pysdd torch torchvision`.
- **Run**: `cd code/deepproblog/src && python -m deepproblog.examples.MNIST.addition`
- **Why it matters**: defines the MNIST-Addition benchmark used everywhere; the cleanest
  reference for "train perception through a symbolic constraint."

## NeurASP
- **Purpose**: Treats NN softmax outputs as a probability distribution over atoms in an
  Answer Set Program; symbolic constraints (ASP rules) both *correct* perception at inference
  and *supervise* training. Solver: `clingo`.
- **Key files**: `neurasp.py` (core), `examples/` (`sudoku/`, `mnistAdd/`, `shortestPath/`,
  `top_k/`, `toy_car/`, …), each with its own README.
- **Requirements**: `clingo>=5.5`, `torch`, `torchvision`, `tqdm`. (`uv pip install clingo torch torchvision tqdm`).
- **Run**: `cd code/NeurASP/examples/mnistAdd && python train.py` (per-example READMEs).

## Logic Tensor Networks (LTN)
- **Purpose**: "Real Logic" — a differentiable, many-valued first-order logic. Predicates/functions
  are neural nets; logical formulas become loss terms maximizing satisfiability. Good for
  relational learning, constraint injection, semi-supervised classification.
- **Key files**: `ltn/` (core), `examples/` (binary/multiclass classification, clustering,
  regression, `smokes_friends_cancer`, `parent_ancestor`), `tutorials/` (notebooks), `examples/mnist/`.
- **Requirements**: `pyproject.toml` / `requirements.txt` — TensorFlow 2.x. (`uv pip install ltn`).

## Scallop
- **Purpose**: A declarative *neurosymbolic programming language* — differentiable Datalog with
  provenance semirings; Python bindings (`scallopy`) plug logic into PyTorch as a layer.
- **Key files**: `readme.md`, `etc/scallopy/` (Python bindings), `examples/`. The repo is a Rust
  workspace; the Python package `scallopy` is the practical entry point (`pip install scallopy`).
- **Requirements**: Rust toolchain to build from source, or `pip install scallopy` for the
  prebuilt binding. Used for MNIST-Addition, Visual-Sudoku, pathfinding, VQA.

## Semantic-Loss
- **Purpose**: Reference code for the semantic loss function (#10) — a differentiable penalty
  measuring how far a network's outputs are from satisfying logical constraints; boosts
  semi-supervised accuracy and structured-output validity.
- **Key files**: `semi_supervised/` (MNIST/FASHION/CIFAR semi-supervised experiments),
  `complex_constraints/`. TensorFlow-based; constraints compiled to circuits.

## NS-VQA
- **Purpose**: Neural-Symbolic Visual Question Answering on CLEVR — a Mask-R-CNN scene parser
  produces a symbolic scene representation, an LSTM parses the question into a program, and a
  deterministic executor runs the program. Achieves ~99.8% on CLEVR with minimal supervision.
- **Key files**: `scene_parse/`, `reason/` (question→program + executor), `requirements.txt`.
  Requires the CLEVR dataset (see `datasets/README.md`).

---

## Notes for the experiment runner
- **Fastest path to a result**: DeepProbLog MNIST-Addition, or a from-scratch reimplementation
  using the local `datasets/mnist/mnist_addition/` splits (compare a CNN sum-classifier baseline
  vs. a neural-symbolic model that predicts per-digit then sums).
- Several repos pin **TensorFlow 1.x/2.x vs PyTorch** and specific solver versions (`clingo`,
  `PySDD`, SWI-Prolog). Create per-experiment environments to avoid conflicts; do not install all
  simultaneously into the shared `.venv`.
- For LLM-era neurosymbolic reasoning (LINC, #15), no heavy repo is needed — the pattern is
  "LLM autoformalizes NL → FOL, then an external prover (Prover9/Z3) decides." Datasets: FOLIO,
  ProofWriter (see `datasets/README.md`).
