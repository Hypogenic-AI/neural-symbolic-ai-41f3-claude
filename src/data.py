"""Data loading for MNIST-Addition (neuro-symbolic benchmark).

Verified facts about the local data (see planning.md / session diagnostics):
- datasets/mnist/mnist.npz holds x_train/y_train (60k) and x_test/y_test (10k),
  identical ordering to the raw IDX files.
- train_pairs.csv columns (img1_idx,img2_idx,sum) index into x_train/y_train.
- test_pairs.csv indexes into x_test/y_test.
- y_* are TRUE digit labels; we use them ONLY for evaluation/grounding analysis
  and (optionally) for the oracle / mitigation experiments. The NeSy and
  neural-only models are trained on the `sum` column alone (distant supervision).
"""
import os
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MNIST_DIR = os.path.join(ROOT, "datasets", "mnist")
ADD_DIR = os.path.join(MNIST_DIR, "mnist_addition")


def load_mnist():
    """Return (x_train, y_train, x_test, y_test) as float32 tensors in [0,1].

    Images shaped (N,1,28,28); labels int64.
    """
    d = np.load(os.path.join(MNIST_DIR, "mnist.npz"))
    def prep_x(x):
        return torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(1)
    def prep_y(y):
        return torch.from_numpy(y.astype(np.int64))
    return (prep_x(d["x_train"]), prep_y(d["y_train"]),
            prep_x(d["x_test"]), prep_y(d["y_test"]))


def load_pairs(split):
    """Load addition pairs for 'train' or 'test'. Returns a DataFrame."""
    fn = os.path.join(ADD_DIR, f"{split}_pairs.csv")
    return pd.read_csv(fn)


class AdditionPairs(torch.utils.data.Dataset):
    """Pairs of MNIST images with their sum (and true digits for evaluation).

    Each item: (img1, img2, sum, digit1, digit2).
    Digits are carried through only so eval code can measure grounding; the
    training losses for NeSy / neural-only never touch them.
    """

    def __init__(self, x, y, pairs_df, limit=None, seed=0):
        df = pairs_df
        if limit is not None and limit < len(df):
            # deterministic subsample for data-efficiency curves
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(df), size=limit, replace=False)
            df = df.iloc[idx].reset_index(drop=True)
        self.i1 = torch.from_numpy(df["img1_idx"].to_numpy().copy())
        self.i2 = torch.from_numpy(df["img2_idx"].to_numpy().copy())
        self.sum = torch.from_numpy(df["sum"].to_numpy().astype(np.int64))
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.sum)

    def __getitem__(self, k):
        a, b = self.i1[k], self.i2[k]
        return self.x[a], self.x[b], self.sum[k], self.y[a], self.y[b]


def make_multidigit_test(x, y, n_pairs=5000, n_digits=2, seed=123):
    """Build an out-of-distribution multi-digit addition test set from MNIST.

    Each example: two `n_digits`-digit numbers (each digit an MNIST image),
    target = their integer sum. Used to test compositional generalization of a
    SINGLE-digit-trained classifier composed with a symbolic place-value sum.

    Returns a dict of tensors:
      imgs: (n_pairs, 2, n_digits, 1, 28, 28)  -- [example, operand, position]
      digits: (n_pairs, 2, n_digits)           -- true digit at each position
      numbers: (n_pairs, 2)                    -- the two true integers
      target: (n_pairs,)                       -- true sum
    """
    rng = np.random.default_rng(seed)
    N = x.shape[0]
    img_idx = rng.integers(0, N, size=(n_pairs, 2, n_digits))
    digits = y[torch.from_numpy(img_idx)]              # (n_pairs,2,n_digits)
    imgs = x[torch.from_numpy(img_idx)]                # (n_pairs,2,n_digits,1,28,28)
    place = torch.tensor([10 ** (n_digits - 1 - p) for p in range(n_digits)])
    numbers = (digits * place).sum(dim=2)              # (n_pairs,2)
    target = numbers.sum(dim=1)                         # (n_pairs,)
    return {"imgs": imgs, "digits": digits, "numbers": numbers, "target": target}


if __name__ == "__main__":
    xtr, ytr, xte, yte = load_mnist()
    print("x_train", xtr.shape, "range", float(xtr.min()), float(xtr.max()))
    tr = load_pairs("train"); te = load_pairs("test")
    print("train pairs", len(tr), "test pairs", len(te))
    ds = AdditionPairs(xtr, ytr, tr, limit=100, seed=0)
    i1, i2, s, d1, d2 = ds[0]
    print("sample shapes", i1.shape, "sum", int(s), "digits", int(d1), int(d2),
          "check", int(d1) + int(d2) == int(s))
    md = make_multidigit_test(xte, yte, n_pairs=8, n_digits=2)
    print("multidigit imgs", md["imgs"].shape, "numbers[0]", md["numbers"][0].tolist(),
          "target[0]", int(md["target"][0]))
