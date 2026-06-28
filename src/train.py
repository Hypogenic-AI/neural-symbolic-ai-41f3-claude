"""Training loops, evaluation, and grounding metrics for the MNIST-Addition study.

Models:
  - 'neural' : NeuralSum, trained on sum cross-entropy (distant supervision).
  - 'nesy'   : DigitCNN, trained on NLL of the symbolic sum (distant supervision).
  - 'oracle' : DigitCNN, trained on direct digit cross-entropy (upper bound).

Mitigations for the NeSy model (E3) layer extra terms on top of the sum NLL:
  - aux_digit_labels (k): auxiliary digit CE on k labeled images per class subset.
  - uniform_prior (lambda): KL(batch-mean digit dist || uniform), discourages
    collapse / encourages using all symbols.
  - entropy_reg (beta): penalize per-example prediction entropy (sharpen).
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .models import DigitCNN, NeuralSum, nesy_sum_logprob
from .data import AdditionPairs


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(ds, batch_size, shuffle, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                       generator=g if shuffle else None,
                                       num_workers=0, drop_last=False)


@torch.no_grad()
def evaluate(model, kind, test_ds, device, batch_size=512, modulo=None):
    """Return dict with sum_acc and (for digit models) grounding metrics.

    grounding_acc       : raw argmax-digit accuracy vs true labels.
    grounding_acc_aligned: accuracy after the best 10x10 label permutation
                           (Hungarian) -- detects *systematic* shortcuts where the
                           network learned a consistent relabeling of digits.
    """
    model.eval()
    ld = _loader(test_ds, batch_size, shuffle=False)
    n = 0
    sum_correct = 0
    # confusion matrix pred-digit (rows true 0..9, cols pred 0..9), pooled over both images
    conf = np.zeros((10, 10), dtype=np.int64)
    digit_total = 0
    digit_correct = 0
    for x1, x2, s, d1, d2 in ld:
        x1, x2 = x1.to(device), x2.to(device)
        s = s.to(device)
        if modulo is not None:
            s = s % modulo
        if kind == "neural":
            pred = model(x1, x2).argmax(-1)
            sum_correct += (pred == s).sum().item()
        else:  # digit model (nesy / oracle)
            lp1 = model.log_probs(x1)
            lp2 = model.log_probs(x2)
            psum = nesy_sum_logprob(lp1, lp2, modulo=modulo).argmax(-1)
            sum_correct += (psum == s).sum().item()
            p1 = lp1.argmax(-1).cpu().numpy()
            p2 = lp2.argmax(-1).cpu().numpy()
            t1 = d1.numpy(); t2 = d2.numpy()
            digit_correct += int((p1 == t1).sum() + (p2 == t2).sum())
            digit_total += len(t1) * 2
            for t, p in ((t1, p1), (t2, p2)):
                np.add.at(conf, (t, p), 1)
        n += len(s)
    out = {"sum_acc": sum_correct / n, "n_test": n}
    if kind != "neural":
        out["grounding_acc"] = digit_correct / digit_total
        # Hungarian: maximize matched mass -> best permutation of predicted labels
        row, col = linear_sum_assignment(-conf)            # maps true->pred col
        # aligned accuracy: for each true class i, count conf[i, perm(i)]
        aligned = conf[row, col].sum()
        out["grounding_acc_aligned"] = float(aligned) / conf.sum()
        out["perm"] = col.tolist()                          # permutation discovered
        out["is_shortcut"] = bool(out["grounding_acc_aligned"] - out["grounding_acc"] > 0.10)
        out["confusion"] = conf.tolist()
    return out


def train_model(kind, x, y, train_pairs, test_ds, device,
                limit=None, seed=0, epochs=10, batch_size=128, lr=1e-3,
                aux_digit_k=0, uniform_prior=0.0, entropy_reg=0.0,
                modulo=None, log_every=0):
    """Train one model and return (model, history, final_eval).

    modulo: None -> standard addition (sum 0..18); int m -> task is (a+b) mod m.
    """
    set_seed(seed)
    ds = AdditionPairs(x, y, train_pairs, limit=limit, seed=seed)
    ld = _loader(ds, batch_size, shuffle=True, seed=seed)

    if kind == "neural":
        model = NeuralSum(n_out=(modulo if modulo is not None else 19)).to(device)
    else:
        model = DigitCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Optional auxiliary digit supervision (E3): a small labeled image pool.
    aux_imgs = aux_labels = None
    if aux_digit_k > 0:
        rng = np.random.default_rng(1000 + seed)
        sel = []
        for c in range(10):                    # k labeled images PER CLASS
            idx = np.where(y.numpy() == c)[0]
            sel.append(rng.choice(idx, size=aux_digit_k, replace=False))
        sel = np.concatenate(sel)
        aux_imgs = x[torch.from_numpy(sel)].to(device)
        aux_labels = y[torch.from_numpy(sel)].to(device)

    uniform = torch.full((10,), 0.1, device=device)
    history = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for x1, x2, s, d1, d2 in ld:
            x1, x2, s = x1.to(device), x2.to(device), s.to(device)
            if modulo is not None:
                s = s % modulo
            opt.zero_grad()
            if kind == "neural":
                loss = F.cross_entropy(model(x1, x2), s)
            elif kind == "oracle":
                # direct digit supervision on both images
                lp1, lp2 = model.log_probs(x1), model.log_probs(x2)
                loss = F.nll_loss(lp1, d1.to(device)) + F.nll_loss(lp2, d2.to(device))
            else:  # nesy: distant supervision via symbolic sum
                lp1, lp2 = model.log_probs(x1), model.log_probs(x2)
                loss = F.nll_loss(nesy_sum_logprob(lp1, lp2, modulo=modulo), s)
                if uniform_prior > 0:
                    batch_mean = torch.cat([lp1.exp(), lp2.exp()], 0).mean(0).clamp_min(1e-9)
                    loss = loss + uniform_prior * (batch_mean * (batch_mean / uniform).log()).sum()
                if entropy_reg > 0:
                    ent = -(lp1.exp() * lp1).sum(-1).mean() - (lp2.exp() * lp2).sum(-1).mean()
                    loss = loss + entropy_reg * ent
                if aux_imgs is not None:
                    loss = loss + F.nll_loss(model.log_probs(aux_imgs), aux_labels)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(s)
        history.append(ep_loss / len(ds))
        if log_every and (ep + 1) % log_every == 0:
            print(f"    [{kind} seed{seed} N{limit}] ep{ep+1}/{epochs} loss={history[-1]:.4f}")
    train_time = time.time() - t0
    ev = evaluate(model, "neural" if kind == "neural" else "nesy", test_ds, device,
                  modulo=modulo)
    ev["train_time_s"] = train_time
    ev["final_train_loss"] = history[-1]
    return model, history, ev
