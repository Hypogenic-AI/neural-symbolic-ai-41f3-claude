"""Run experiments E1-E4 and save raw results to results/raw/.

Parallelizes the grid across multiple GPUs with a process pool. Each worker
process pins itself to one GPU and loads MNIST once (via the pool initializer).

Usage:
    python -m src.run_experiments --exp e1   # data efficiency + grounding (E1/E2)
    python -m src.run_experiments --exp e3   # mitigations
    python -m src.run_experiments --exp e4   # compositional OOD
    python -m src.run_experiments --exp all
"""
import os
import json
import time
import argparse
import multiprocessing as mp

import numpy as np
import torch

from . import config as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")
os.makedirs(RAW, exist_ok=True)

# GPUs with ample free memory (GPU 2 is busy on this shared box -> excluded)
GPUS = [0, 1, 3]

# per-worker globals
_G = {}


def _init_worker(gpu_list):
    """Pool initializer: pick a GPU by worker identity and load data once."""
    from .data import load_mnist, load_pairs, AdditionPairs
    wid = mp.current_process()._identity[0] - 1
    dev = f"cuda:{gpu_list[wid % len(gpu_list)]}"
    xtr, ytr, xte, yte = load_mnist()
    trp = load_pairs("train")
    tep = load_pairs("test")
    _G["dev"] = dev
    _G["xtr"], _G["ytr"], _G["xte"], _G["yte"] = xtr, ytr, xte, yte
    _G["trp"] = trp
    _G["test_ds"] = AdditionPairs(xte, yte, tep)


def _run_one(job):
    """Train+eval one (E1/E2 or E3) configuration. Returns a result dict."""
    from .train import train_model
    dev = _G["dev"]
    ev = train_model(
        job["kind"], _G["xtr"], _G["ytr"], _G["trp"], _G["test_ds"], dev,
        limit=job.get("limit"), seed=job["seed"], epochs=C.EPOCHS,
        batch_size=C.BATCH_SIZE, lr=C.LR,
        aux_digit_k=job.get("aux_digit_k", 0),
        uniform_prior=job.get("uniform_prior", 0.0),
        entropy_reg=job.get("entropy_reg", 0.0),
        modulo=job.get("modulo"),
    )[2]
    # Keep the confusion matrix only for the E5 modular curve NeSy runs (needed to
    # render Figure 3 faithfully from the SAME run that produced the statistics);
    # drop it elsewhere to keep result files small.
    keep_conf = (job.get("exp") == "e5" and job.get("phase") == "curve"
                 and job["kind"] == "nesy" and job.get("limit") == max(C.E5_TRAIN_SIZES))
    if not keep_conf:
        ev.pop("confusion", None)
    out = {**{k: v for k, v in job.items()}, **ev, "device": dev}
    return out


def _run_e4(job):
    """Train a single-digit model at full data, then evaluate compositional OOD."""
    from .train import train_model
    from .models import compose_number_logprob, nesy_sum_logprob
    from .data import make_multidigit_test
    dev = _G["dev"]
    model = train_model(job["kind"], _G["xtr"], _G["ytr"], _G["trp"],
                        _G["test_ds"], dev, limit=C.E4_TRAIN_SIZE, seed=job["seed"],
                        epochs=C.EPOCHS, batch_size=C.BATCH_SIZE, lr=C.LR)[0]
    model.eval()
    # in-distribution single-digit eval (grounding) for correlation analysis
    from .train import evaluate
    single = evaluate(model, "neural" if job["kind"] == "neural" else "nesy",
                      _G["test_ds"], dev)
    res = {"kind": job["kind"], "seed": job["seed"], "device": dev,
           "single_sum_acc": single["sum_acc"],
           "single_grounding_acc": single.get("grounding_acc")}
    for nd in C.E4_N_DIGITS:
        md = make_multidigit_test(_G["xte"], _G["yte"], n_pairs=C.E4_N_PAIRS,
                                  n_digits=nd, seed=100 + nd)
        acc = _eval_multidigit(model, job["kind"], md, dev, nd)
        res[f"ood_{nd}digit_acc"] = acc
    return res


@torch.no_grad()
def _eval_multidigit(model, kind, md, dev, n_digits, batch=256):
    """Evaluate a single-digit-trained model on n_digit-number addition.

    For digit models we compose the per-position digit distributions with the
    symbolic place-value sum (exact). The neural-only model cannot emit sums in
    this range; we score it with the most charitable surrogate available and
    record its structural inability.
    """
    from .models import compose_number_logprob
    imgs = md["imgs"]; target = md["target"]
    N = imgs.shape[0]
    correct = 0
    for i in range(0, N, batch):
        chunk = imgs[i:i + batch].to(dev)            # (b,2,nd,1,28,28)
        b = chunk.shape[0]
        if kind == "neural":
            # NeuralSum expects two 28x28 images and outputs sums 0..18 only.
            # It is structurally incapable of multi-digit sums -> count as 0.
            # (We still return 0.0 honestly rather than fabricate a mapping.)
            continue
        # digit model: predict each position's digit distribution, compose numbers
        nums = []
        for op in range(2):
            logps = [model.log_probs(chunk[:, op, p].reshape(b, 1, 28, 28))
                     for p in range(n_digits)]
            num_logp = compose_number_logprob(logps)     # (b,10**nd)
            nums.append(num_logp.argmax(-1))
        pred_sum = nums[0] + nums[1]
        correct += (pred_sum.cpu() == target[i:i + b]).sum().item()
    return correct / N


def run_grid(jobs, fn, out_name):
    t0 = time.time()
    ctx = mp.get_context("spawn")
    n_workers = min(len(GPUS) * 2, max(1, len(jobs)))   # 2 jobs per GPU
    print(f"[{out_name}] {len(jobs)} jobs on {n_workers} workers across GPUs {GPUS}")
    with ctx.Pool(n_workers, initializer=_init_worker, initargs=(GPUS,)) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(fn, jobs)):
            results.append(r)
            tag = r.get("kind", "?")
            print(f"  done {i+1}/{len(jobs)}: {tag} "
                  f"N={r.get('limit', r.get('seed'))} "
                  f"sum={r.get('sum_acc', r.get('ood_2digit_acc','-'))}")
    path = os.path.join(RAW, out_name)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{out_name}] saved {len(results)} results in {time.time()-t0:.1f}s -> {path}")
    return results


def exp_e1():
    jobs = [{"exp": "e1", "kind": k, "limit": n, "seed": s}
            for k in C.MODELS for n in C.TRAIN_SIZES for s in C.SEEDS]
    return run_grid(jobs, _run_one, "e1_e2.json")


def exp_e3():
    jobs = [{"exp": "e3", "kind": "nesy", "limit": C.E3_TRAIN_SIZE, "seed": s, **m}
            for m in C.E3_MITIGATIONS for s in C.SEEDS]
    return run_grid(jobs, _run_one, "e3.json")


def exp_e4():
    jobs = [{"exp": "e4", "kind": k, "seed": s}
            for k in ["nesy", "oracle", "neural"] for s in C.SEEDS]
    return run_grid(jobs, _run_e4, "e4.json")


def exp_e5():
    """E5: induce shortcuts via modular addition (sum mod 10), and test whether
    mitigations restore grounding when a genuine shortcut exists."""
    jobs = []
    # 5a: modular data-efficiency + grounding (mirror of E1/E2 under mod-10)
    for k in C.MODELS:
        for n in C.E5_TRAIN_SIZES:
            for s in C.SEEDS:
                jobs.append({"exp": "e5", "kind": k, "limit": n, "seed": s,
                             "modulo": 10, "phase": "curve"})
    # 5b: mitigation sweep under modular addition at a mid size
    for m in C.E3_MITIGATIONS:
        for s in C.SEEDS:
            jobs.append({"exp": "e5", "kind": "nesy", "limit": C.E5_MIT_SIZE,
                         "seed": s, "modulo": 10, "phase": "mitig", **m})
    return run_grid(jobs, _run_one, "e5.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["e1", "e3", "e4", "e5", "all"], default="all")
    args = ap.parse_args()
    if args.exp in ("e1", "all"):
        exp_e1()
    if args.exp in ("e3", "all"):
        exp_e3()
    if args.exp in ("e4", "all"):
        exp_e4()
    if args.exp in ("e5", "all"):
        exp_e5()
    print("DONE")
