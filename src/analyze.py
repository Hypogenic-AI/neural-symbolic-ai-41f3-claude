"""Analysis, statistics, and figures for the MNIST-Addition NeSy study.

Loads results/raw/*.json, computes aggregate statistics and hypothesis tests,
writes results/summary.json + results/stats.json, and renders figures to figures/.
"""
import os
import json
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)


def load(name):
    return json.load(open(os.path.join(RAW, name)))


def mean_ci(vals, alpha=0.05):
    """Mean and half-width of the t-based (1-alpha) CI."""
    a = np.asarray(vals, dtype=float)
    n = len(a)
    m = a.mean()
    if n < 2:
        return m, 0.0
    se = a.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf(1 - alpha / 2, n - 1)
    return m, h


def cohen_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.inf


def holm_bonferroni(pvals, alpha=0.05):
    """Holm-Bonferroni step-down. Returns (reject, p_corrected).

    reject[i]      : whether H0_i is rejected at family-wise alpha.
    p_corrected[i] : Holm-adjusted p-value (monotone-enforced, clipped to 1).
    """
    pvals = list(pvals)
    order = np.argsort(pvals)
    m = len(pvals)
    reject = [False] * m
    p_adj = [0.0] * m
    running = 0.0
    stopped = False
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[idx])
        running = max(running, adj)          # enforce monotonicity
        p_adj[idx] = running
        if not stopped and pvals[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            stopped = True
    return reject, p_adj


def collect_curve(rows, sizes, kinds):
    """Aggregate sum/grounding by (kind,limit)."""
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r.get("phase", "curve") != "curve":
            continue
        key = (r["kind"], r["limit"])
        g[key]["sum"].append(r["sum_acc"])
        if r["kind"] != "neural":
            g[key]["gnd"].append(r["grounding_acc"])
            g[key]["aln"].append(r["grounding_acc_aligned"])
            g[key]["sc"].append(int(r["is_shortcut"]))
    return g


# ---------------------------------------------------------------- statistics
def run_stats():
    e1 = load("e1_e2.json")
    sizes = sorted({r["limit"] for r in e1})
    g = collect_curve(e1, sizes, ["neural", "nesy", "oracle"])
    out = {"standard_addition": {}, "hypothesis_tests": {}}
    for k in ["neural", "nesy", "oracle"]:
        out["standard_addition"][k] = {}
        for n in sizes:
            m, h = mean_ci(g[(k, n)]["sum"])
            entry = {"sum_acc_mean": m, "sum_acc_ci95": h}
            if g[(k, n)]["gnd"]:
                gm, gh = mean_ci(g[(k, n)]["gnd"])
                entry.update({"grounding_mean": gm, "grounding_ci95": gh,
                              "n_shortcut": int(sum(g[(k, n)]["sc"]))})
            out["standard_addition"][k][n] = entry

    # H1: Welch t-test NeSy vs neural at each size, Holm-corrected
    pvals, tests = [], []
    for n in sizes:
        a, b = g[("nesy", n)]["sum"], g[("neural", n)]["sum"]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        d = cohen_d(a, b)
        tests.append({"size": n, "nesy_mean": float(np.mean(a)),
                      "neural_mean": float(np.mean(b)), "t": float(t),
                      "p_raw": float(p), "cohens_d": float(d)})
        pvals.append(p)
    rej, p_adj = holm_bonferroni(pvals)
    for tt, r, pa in zip(tests, rej, p_adj):
        tt["reject_H0_holm"] = bool(r)
        tt["p_holm"] = float(pa)
    out["hypothesis_tests"]["H1_nesy_vs_neural_welch"] = tests

    # E5 modular: shortcut summary + standard-vs-modular grounding contrast
    e5 = load("e5.json")
    g5 = collect_curve(e5, sorted({r["limit"] for r in e5 if r.get("phase") == "curve"}),
                       ["neural", "nesy", "oracle"])
    out["modular_addition"] = {}
    for k in ["neural", "nesy", "oracle"]:
        out["modular_addition"][k] = {}
        for n in sorted({r["limit"] for r in e5 if r.get("phase") == "curve"}):
            if (k, n) not in g5:
                continue
            m, h = mean_ci(g5[(k, n)]["sum"])
            entry = {"sum_acc_mean": m, "sum_acc_ci95": h}
            if g5[(k, n)]["gnd"]:
                gm, gh = mean_ci(g5[(k, n)]["gnd"])
                am, ah = mean_ci(g5[(k, n)]["aln"])
                entry.update({"grounding_mean": gm, "grounding_ci95": gh,
                              "grounding_aligned_mean": am,
                              "n_shortcut": int(sum(g5[(k, n)]["sc"]))})
            out["modular_addition"][k][n] = entry

    # Key contrast test: NeSy grounding standard vs modular at the largest size
    big = max(sizes)
    std_g = g[("nesy", big)]["gnd"]
    mod_g = g5[("nesy", big)]["gnd"]
    t, p = stats.ttest_ind(std_g, mod_g, equal_var=False)
    out["hypothesis_tests"]["H2_grounding_standard_vs_modular"] = {
        "size": big, "standard_mean": float(np.mean(std_g)),
        "modular_mean": float(np.mean(mod_g)), "t": float(t), "p": float(p),
        "cohens_d": float(cohen_d(std_g, mod_g))}

    # E3 vs E5 mitigations
    out["mitigations_standard"] = _agg_mitig(load("e3.json"))
    out["mitigations_modular"] = _agg_mitig([r for r in e5 if r.get("phase") == "mitig"])

    # H3 test: aux_k1 vs baseline grounding under modular
    mm = [r for r in e5 if r.get("phase") == "mitig"]
    base = [r["grounding_acc"] for r in mm if r["name"] == "baseline"]
    aux = [r["grounding_acc"] for r in mm if r["name"] == "aux_k1"]
    t, p = stats.ttest_ind(aux, base, equal_var=False)
    out["hypothesis_tests"]["H3_aux_k1_vs_baseline_modular"] = {
        "baseline_grounding": float(np.mean(base)), "aux_k1_grounding": float(np.mean(aux)),
        "t": float(t), "p": float(p), "cohens_d": float(cohen_d(aux, base))}

    # E4 compositional OOD + correlation grounding<->OOD
    e4 = load("e4.json")
    out["compositional_ood"] = _agg_ood(e4)
    gnds = [r["single_grounding_acc"] for r in e4 if r["kind"] != "neural"]
    oods = [r["ood_2digit_acc"] for r in e4 if r["kind"] != "neural"]
    if len(set(np.round(gnds, 4))) > 1:
        rho, pp = stats.pearsonr(gnds, oods)
        out["hypothesis_tests"]["H4_grounding_ood_pearson"] = {"r": float(rho), "p": float(pp)}

    with open(os.path.join(RES, "stats.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/stats.json")
    return out


def _agg_mitig(rows):
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        g[r["name"]]["sum"].append(r["sum_acc"])
        g[r["name"]]["gnd"].append(r["grounding_acc"])
        g[r["name"]]["sc"].append(int(r["is_shortcut"]))
    out = {}
    for name, d in g.items():
        sm, sh = mean_ci(d["sum"]); gm, gh = mean_ci(d["gnd"])
        out[name] = {"sum_mean": sm, "sum_ci95": sh, "grounding_mean": gm,
                     "grounding_ci95": gh, "n_shortcut": int(sum(d["sc"]))}
    return out


def _agg_ood(rows):
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        g[r["kind"]]["o2"].append(r["ood_2digit_acc"])
        g[r["kind"]]["o3"].append(r["ood_3digit_acc"])
        g[r["kind"]]["g"].append(r.get("single_grounding_acc") or 0.0)
    out = {}
    for k, d in g.items():
        o2 = mean_ci(d["o2"]); o3 = mean_ci(d["o3"])
        out[k] = {"ood_2digit_mean": o2[0], "ood_2digit_ci95": o2[1],
                  "ood_3digit_mean": o3[0], "ood_3digit_ci95": o3[1],
                  "single_grounding_mean": float(np.mean(d["g"]))}
    return out


# ------------------------------------------------------------------- figures
def fig_learning_curves(stats_d):
    e1 = load("e1_e2.json")
    sizes = sorted({r["limit"] for r in e1})
    g = collect_curve(e1, sizes, ["neural", "nesy", "oracle"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"neural": "#d62728", "nesy": "#1f77b4", "oracle": "#2ca02c"}
    labels = {"neural": "Neural-only (CNN sum-classifier)",
              "nesy": "Neuro-Symbolic (sum-only supervision)",
              "oracle": "Oracle (digit-supervised)"}
    for k in ["neural", "nesy", "oracle"]:
        ms = [mean_ci(g[(k, n)]["sum"]) for n in sizes]
        m = [x[0] for x in ms]; h = [x[1] for x in ms]
        ax[0].errorbar(sizes, m, yerr=h, marker="o", capsize=3, color=colors[k], label=labels[k])
    ax[0].set_xscale("log"); ax[0].set_xlabel("# training pairs (sum-only)")
    ax[0].set_ylabel("Test sum accuracy"); ax[0].set_title("(a) Data efficiency — standard addition")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8); ax[0].set_ylim(0, 1.02)
    # grounding panel
    for k in ["nesy", "oracle"]:
        ms = [mean_ci(g[(k, n)]["gnd"]) for n in sizes]
        m = [x[0] for x in ms]; h = [x[1] for x in ms]
        ax[1].errorbar(sizes, m, yerr=h, marker="s", capsize=3, color=colors[k], label=labels[k])
    ax[1].set_xscale("log"); ax[1].set_xlabel("# training pairs")
    ax[1].set_ylabel("Latent digit-grounding accuracy")
    ax[1].set_title("(b) Grounding emerges without digit labels")
    ax[1].axhline(0.1, ls="--", c="gray", lw=1, label="chance (0.10)")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8); ax[1].set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_data_efficiency.png"), dpi=140)
    plt.close(fig); print("fig1 saved")


def fig_shortcut(stats_d):
    """Standard vs modular: task accuracy high in both, grounding collapses under modular."""
    e1 = load("e1_e2.json"); e5 = load("e5.json")
    sizes = [1000, 5000, 30000]
    gs = collect_curve(e1, sizes, ["nesy"]); gm = collect_curve(e5, sizes, ["nesy"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(sizes)); w = 0.35
    for i, (gg, lbl, c) in enumerate([(gs, "standard (a+b)", "#1f77b4"),
                                      (gm, "modular (a+b mod 10)", "#ff7f0e")]):
        sm = [mean_ci(gg[("nesy", n)]["sum"]) for n in sizes]
        ax[0].bar(x + (i - 0.5) * w, [s[0] for s in sm], w, yerr=[s[1] for s in sm],
                  capsize=3, label=lbl, color=c)
        gd = [mean_ci(gg[("nesy", n)]["gnd"]) for n in sizes]
        ax[1].bar(x + (i - 0.5) * w, [s[0] for s in gd], w, yerr=[s[1] for s in gd],
                  capsize=3, label=lbl, color=c)
    for a, t, yl in ((ax[0], "(a) Task (sum) accuracy", "Test sum accuracy"),
                     (ax[1], "(b) Latent grounding accuracy", "Digit-grounding accuracy")):
        a.set_xticks(x); a.set_xticklabels(sizes); a.set_xlabel("# training pairs")
        a.set_ylabel(yl); a.set_title(t); a.legend(fontsize=8); a.grid(alpha=0.3, axis="y")
        a.set_ylim(0, 1.02)
    ax[1].axhline(0.1, ls="--", c="gray", lw=1)
    ax[1].text(0.02, 0.12, "chance", fontsize=8, color="gray", transform=ax[1].get_yaxis_transform())
    fig.suptitle("Reasoning shortcuts: under the symmetric modular operator, digit-grounding "
                 "collapses to ~chance\nwhile the task is still (unstably) solved — high error bars "
                 "reflect 4/5 shortcut + 1/5 collapse", fontsize=10.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_shortcut.png"), dpi=140)
    plt.close(fig); print("fig2 saved")


def fig_confusion():
    """Visualize the modular-addition outcome for ALL 5 seeds, read DIRECTLY from
    the saved E5 grid (results/raw/e5.json) so the figure is faithful to the exact
    runs the statistics are computed from (no non-deterministic retraining).

    Each panel is one seed's grounding confusion matrix at N=30000 under modular
    addition, showing that correct grounding never emerges: seeds either find a
    shortcut (high task acc, raw grounding ~0.1%) or collapse (fail the task).
    """
    e5 = load("e5.json")
    rows = sorted([r for r in e5 if r.get("phase") == "curve" and r["kind"] == "nesy"
                   and r["limit"] == max(r2["limit"] for r2 in e5 if r2["kind"] == "nesy")
                   and "confusion" in r], key=lambda r: r["seed"])
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4.0))
    if n == 1:
        axes = [axes]
    im = None
    for ax, r in zip(axes, rows):
        conf = np.array(r["confusion"], dtype=float)
        conf = conf / conf.sum(1, keepdims=True)
        im = ax.imshow(conf, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(0, 10, 2)); ax.set_yticks(range(0, 10, 2))
        ax.set_xlabel("predicted symbol")
        if r["seed"] == rows[0]["seed"]:
            ax.set_ylabel("true digit")
        if not r["is_shortcut"] and r["sum_acc"] < 0.3:
            kind = "task COLLAPSE (failed to fit)"
        elif r["grounding_acc_aligned"] > 0.9:
            kind = "clean (d+c) cyclic shift"
        else:
            kind = "shortcut: systematic relabel"
        ax.set_title(f"seed {r['seed']}: sum={r['sum_acc']:.2f}\nraw gnd={r['grounding_acc']:.1%}, "
                     f"aligned={r['grounding_acc_aligned']:.2f}\n{kind}", fontsize=8.5)
    fig.suptitle("Modular addition (a+b mod 10), N=30k, all seeds (from saved E5 grid): correct "
                 "grounding NEVER emerges.\nSeeds either find a shortcut (high task accuracy, raw "
                 "grounding ~0.1%, off-diagonal relabeling) or collapse (fail the task). "
                 "Contrast: standard addition grounds at 99%.", fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.012, label="P(pred | true)")
    fig.savefig(os.path.join(FIG, "fig3_confusion.png"), dpi=130, bbox_inches="tight")
    plt.close(fig); print(f"fig3 saved ({n}-seed small multiples from e5.json)")


def fig_ood(stats_d):
    ood = stats_d["compositional_ood"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    kinds = ["neural", "nesy", "oracle"]
    labels = ["Neural-only", "Neuro-Symbolic", "Oracle"]
    x = np.arange(len(kinds)); w = 0.38
    o2 = [ood[k]["ood_2digit_mean"] for k in kinds]; o2e = [ood[k]["ood_2digit_ci95"] for k in kinds]
    o3 = [ood[k]["ood_3digit_mean"] for k in kinds]; o3e = [ood[k]["ood_3digit_ci95"] for k in kinds]
    ax.bar(x - w / 2, o2, w, yerr=o2e, capsize=3, label="2-digit addition (OOD)", color="#1f77b4")
    ax.bar(x + w / 2, o3, w, yerr=o3e, capsize=3, label="3-digit addition (OOD)", color="#9467bd")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Multi-digit addition accuracy")
    ax.set_title("Compositional generalization: trained on 1-digit sums only\n"
                 "NeSy composes digits via symbolic place-value; neural-only cannot")
    ax.legend(); ax.grid(alpha=0.3, axis="y"); ax.set_ylim(0, 1.02)
    for xi, v in zip(x - w / 2, o2):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, o3):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_compositional_ood.png"), dpi=140)
    plt.close(fig); print("fig4 saved")


def fig_mitigations(stats_d):
    order = ["baseline", "aux_k1", "aux_k5", "aux_k20", "uniform_prior", "entropy_reg"]
    std = stats_d["mitigations_standard"]; mod = stats_d["mitigations_modular"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(order)); w = 0.38
    sg = [std[n]["grounding_mean"] for n in order]; sge = [std[n]["grounding_ci95"] for n in order]
    mg = [mod[n]["grounding_mean"] for n in order]; mge = [mod[n]["grounding_ci95"] for n in order]
    ax.bar(x - w / 2, sg, w, yerr=sge, capsize=3, label="standard add (no shortcut)", color="#1f77b4")
    ax.bar(x + w / 2, mg, w, yerr=mge, capsize=3, label="modular add (shortcut present)", color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Grounding accuracy"); ax.set_ylim(0, 1.02)
    ax.set_title("Mitigations: only needed (and effective) when a shortcut exists\n"
                 "10 labeled images (aux_k1) restore grounding under modular addition")
    ax.axhline(0.1, ls="--", c="gray", lw=1)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_mitigations.png"), dpi=140)
    plt.close(fig); print("fig5 saved")


if __name__ == "__main__":
    s = run_stats()
    fig_learning_curves(s)
    fig_shortcut(s)
    fig_ood(s)
    fig_mitigations(s)
    fig_confusion()
    print("ALL ANALYSIS DONE")
