"""Models for the MNIST-Addition study.

All three models share the SAME convolutional encoder so the only difference is
the output head / learning signal. This isolates "symbolic structure" as the
single experimental variable.

- DigitCNN: shared CNN -> 10-way digit logits per image. Used by NeSy and Oracle.
- NeuralSum: shared CNN encodes each image; features concatenated -> MLP ->
  19-way softmax over sums {0..18}. No knowledge that sum = d1 + d2.
- nesy_sum_logprob: differentiable symbolic addition. Given per-image digit
  log-probabilities, computes log P(sum=s) = log Σ_{a+b=s} P(a)P(b) by
  convolving the two digit distributions (the DeepProbLog/Scallop idea).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """LeNet-style convolutional feature extractor (shared by all models)."""

    def __init__(self, feat_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, feat_dim), nn.ReLU())
        self.feat_dim = feat_dim

    def forward(self, x):
        return self.fc(self.conv(x))


class DigitCNN(nn.Module):
    """Shared encoder + linear head producing 10-way digit logits per image.

    Used by both the NeSy model (trained via the symbolic sum) and the Oracle
    (trained with direct digit supervision). Identical capacity -> fair compare.
    """

    def __init__(self, feat_dim=128):
        super().__init__()
        self.enc = Encoder(feat_dim)
        self.head = nn.Linear(feat_dim, 10)

    def forward(self, x):
        return self.head(self.enc(x))          # logits (B,10)

    def log_probs(self, x):
        return F.log_softmax(self.forward(x), dim=-1)


class NeuralSum(nn.Module):
    """Pure-neural baseline: predict the 19-way sum directly from both images.

    Shared encoder applied to each image, features concatenated, MLP -> 19 sums.
    Knows nothing about the additive structure.
    """

    def __init__(self, feat_dim=128, n_out=19):
        super().__init__()
        self.enc = Encoder(feat_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * feat_dim, 128), nn.ReLU(), nn.Linear(128, n_out)
        )

    def forward(self, x1, x2):
        f = torch.cat([self.enc(x1), self.enc(x2)], dim=-1)
        return self.head(f)                     # logits (B,19) over sums 0..18


def nesy_sum_logprob(logp1, logp2, modulo=None):
    """Differentiable symbolic addition over digit distributions.

    Args:
        logp1, logp2: (B,10) log-probabilities of each digit for the two images.
        modulo: if None, exact sum over 0..18. If an int m (e.g. 10), computes
            the distribution of (a+b) mod m over 0..m-1. Modular addition removes
            the boundary constraints (sum=0 -> 0+0, sum=18 -> 9+9) that pin the
            grounding, so it admits *systematic* reasoning shortcuts (e.g. shifting
            every digit by m/2). Used to deliberately induce shortcuts (E5).
    Returns:
        (B, n_out) log P(result=s), n_out = 19 (modulo None) or `modulo`.
    """
    p1 = logp1.exp()
    p2 = logp2.exp()
    B = p1.shape[0]
    joint = p1.unsqueeze(2) * p2.unsqueeze(1)              # (B,10,10)
    a = torch.arange(10, device=p1.device)
    raw = a.unsqueeze(1) + a.unsqueeze(0)                  # (10,10) values 0..18
    if modulo is None:
        n_out = 19
        s_idx = raw.reshape(-1)
    else:
        n_out = modulo
        s_idx = (raw % modulo).reshape(-1)
    out = torch.zeros(B, n_out, device=p1.device, dtype=p1.dtype)
    out.index_add_(1, s_idx, joint.reshape(B, 100))
    return (out.clamp_min(1e-12)).log()                    # (B,n_out)


def compose_number_logprob(digit_logps):
    """Compose per-position digit distributions into a number distribution.

    Symbolic place-value composition for multi-digit OOD evaluation. Given a
    list of (B,10) log-prob tensors (most-significant first), returns
    (B, 10**k) log-probabilities over the integer value of the k-digit number,
    assuming positional independence. k is small (<=3) so this is cheap.
    """
    # start with the most significant digit distribution over its place values
    k = len(digit_logps)
    # values for position p contribute digit * 10**(k-1-p)
    logp = digit_logps[0] + torch.log(torch.ones(1, device=digit_logps[0].device))
    # represent as distribution over current partial number values
    place0 = 10 ** (k - 1)
    vals = (torch.arange(10, device=logp.device) * place0)
    cur_logp = logp                       # (B,10) over values {0,place0,...}
    cur_vals = vals                        # (10,)
    for p in range(1, k):
        place = 10 ** (k - 1 - p)
        d_logp = digit_logps[p]            # (B,10)
        new_vals = (cur_vals.unsqueeze(1) + torch.arange(10, device=logp.device) * place).reshape(-1)
        new_logp = (cur_logp.unsqueeze(2) + d_logp.unsqueeze(1)).reshape(cur_logp.shape[0], -1)
        cur_vals, cur_logp = new_vals, new_logp
    # reduce to a dense distribution over 0..10**k-1 (values are already unique here)
    order = torch.argsort(cur_vals)
    return cur_logp[:, order]              # (B, 10**k), index == integer value
