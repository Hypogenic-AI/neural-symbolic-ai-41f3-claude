# Datasets

Datasets for Neural-Symbolic (NeSy) learning experiments. Large/raw data files are
excluded from git via `.gitignore`; small index files and samples are kept. Follow the
instructions below to (re)materialize the data locally.

The **primary, ready-to-use benchmark is MNIST-Addition** — the canonical NeSy task
(introduced by DeepProbLog). It is small, fast to train, and directly exercises the
neural-perception + symbolic-reasoning integration that this project investigates.

---

## Dataset 1: MNIST  (primary — perception backbone)

### Overview
- **Source**: Yann LeCun's MNIST (http://yann.lecun.com/exdb/mnist/), redistributed in
  `code/deepproblog/.../examples/MNIST/models/data/MNIST/raw/`.
- **Size**: 70,000 grayscale 28×28 digit images (60k train / 10k test). ~11 MB packed.
- **Format**: raw IDX `ubyte` files **and** a convenience `mnist.npz` (numpy arrays).
- **Task**: digit classification (the neural perception component of NeSy tasks).
- **Local location**: `datasets/mnist/`
- **License**: CC BY-SA 3.0 (MNIST is freely redistributable).

### Files (after build)
```
datasets/mnist/
├── train-images-idx3-ubyte, train-labels-idx1-ubyte   # raw MNIST
├── t10k-images-idx3-ubyte,  t10k-labels-idx1-ubyte
├── mnist.npz                                           # x_train,y_train,x_test,y_test (uint8)
└── mnist_addition/                                     # see Dataset 2
```

### Download / build instructions
The raw files already ship inside the cloned `deepproblog` repo. To rebuild `datasets/mnist`
from scratch:
```bash
# (a) copy raw ubyte files from the deepproblog example
cp code/deepproblog/src/deepproblog/examples/MNIST/models/data/MNIST/raw/*ubyte datasets/mnist/

# (b) OR download fresh via torchvision (requires: uv pip install torchvision)
python -c "from torchvision.datasets import MNIST; MNIST('datasets/mnist', download=True)"

# (c) OR via HuggingFace (requires: uv pip install datasets)
python -c "from datasets import load_dataset; load_dataset('mnist').save_to_disk('datasets/mnist_hf')"
```

### Loading
```python
import numpy as np
d = np.load("datasets/mnist/mnist.npz")
x_train, y_train, x_test, y_test = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
# x_* : uint8 (N,28,28) in [0,255]; y_* : uint8 labels 0..9
```

---

## Dataset 2: MNIST-Addition  (primary NeSy benchmark)

### Overview
- **Task**: given **two** MNIST images, predict the **sum** of the two depicted digits
  (label ∈ {0,…,18}). The digit labels are *never* given during training — only the sum —
  so the model must learn digit perception purely through the symbolic addition constraint.
  This is the standard weakly/distantly-supervised NeSy benchmark.
- **Splits**: 30,000 train pairs (from 60k MNIST train images), 5,000 test pairs (from 10k test).
- **Format**: CSV index files referencing MNIST image indices.
- **Local location**: `datasets/mnist/mnist_addition/`
- **Reference**: Manhaeve et al., *DeepProbLog: Neural Probabilistic Logic Programming*, NeurIPS 2018.

### Files (kept in git — small)
```
mnist_addition/
├── train_pairs.csv   # columns: img1_idx,img2_idx,sum   (30,000 rows)
├── test_pairs.csv    # columns: img1_idx,img2_idx,sum   (5,000 rows)
└── samples.json      # 10 example records with digit labels for inspection
```

### Build instructions
Generated deterministically from MNIST (seeds 42/7). To regenerate:
```bash
python - <<'PY'
import numpy as np
d=np.load("datasets/mnist/mnist.npz"); ytr,yte=d["y_train"],d["y_test"]
def pairs(y,n,seed):
    r=np.random.RandomState(seed); idx=r.permutation(len(y)); n=min(n,len(idx)//2)
    a,b=idx[:2*n:2],idx[1:2*n:2]; return a,b,y[a]+y[b]
# ... write CSVs (see datasets build script / git history)
PY
```

### Loading
```python
import numpy as np, pandas as pd
d = np.load("datasets/mnist/mnist.npz")
X, Y = d["x_train"], d["y_train"]                       # use x_test/y_test for the test split
pairs = pd.read_csv("datasets/mnist/mnist_addition/train_pairs.csv")
i, j, s = pairs.iloc[0]                                  # img1_idx, img2_idx, sum
img1, img2, target = X[i], X[j], s                       # two 28x28 images -> sum target
```

### Sample (`mnist_addition/samples.json`)
```json
[{"img1_idx": ..., "img2_idx": ..., "digit1": 7, "digit2": 2, "sum": 9}, ...]
```

### Notes
- A DeepProbLog-ready version (Prolog facts + loaders) is bundled in
  `code/deepproblog/src/deepproblog/examples/MNIST/`.
- Standard strong baseline: a CNN that takes the two images concatenated and predicts the
  19-way sum directly (no symbolic component) — see literature_review.md for typical numbers.

---

## Additional benchmarks (download-on-demand — NOT pre-downloaded due to size)

These are widely used in the NeSy literature; instructions only (large downloads).

### CLEVR  (for NS-VQA / NS-CL visual reasoning)
- ~18 GB images + questions. Used by NS-VQA (#9) and the Neuro-Symbolic Concept Learner (#1).
- Download: https://cs.stanford.edu/people/jcjohns/clevr/
  ```bash
  wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -O datasets/CLEVR_v1.0.zip
  unzip datasets/CLEVR_v1.0.zip -d datasets/clevr/
  ```

### FOLIO  (for LLM-based neurosymbolic logical reasoning — LINC #15)
- First-order-logic natural-language reasoning, ~1.4k examples. Small.
  ```bash
  pip install datasets
  python -c "from datasets import load_dataset; load_dataset('yale-nlp/FOLIO').save_to_disk('datasets/folio')"
  ```

### ProofWriter  (deductive reasoning over rules — LINC #15)
- Synthetic rule-based deductive reasoning. https://allenai.org/data/proofwriter
  ```bash
  wget https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip -O datasets/proofwriter.zip
  ```

### Visual Sudoku / HWF (Hand-Written Formula)
- Visual-Sudoku and HWF are used by Scallop (#11), SLASH, and Softened Symbol Grounding (#14).
- HWF loader ships in `code/deepproblog/.../examples/HWF/`.

---

## Recommendation for the experiment runner
Start with **MNIST-Addition** (`datasets/mnist/`): it is fully local, trains in minutes on CPU,
and is the de-facto benchmark every NeSy method (DeepProbLog, NeurASP, Scallop, NeuPSL,
Softened Symbol Grounding, Scalable Neural-Probabilistic ASP) reports on. It lets you compare a
neural-only CNN baseline against neural-symbolic integration on identical data.
