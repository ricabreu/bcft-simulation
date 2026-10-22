"""
run_experiments.py -- Reproduce all BCFT simulation experiments, figures and
the synthetic dataset.

Usage:
    python run_experiments.py            # writes to ../figures, ../data
All randomness is seeded (SEED) for exact reproducibility.

Experiments
-----------
1. Divergence and incoherence: coherence declines monotonically with controlled
   distortion, and faster for core than peripheral perturbations.
2. Geodesic vs Euclidean movement: Fisher-Rao geodesics are shorter and stay on
   the simplex's intrinsic structure compared with Euclidean interpolation.
3. Convergent flows: the natural-gradient coherence flow converges to P*.
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bcft

SEED = 20260628
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
DAT = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DAT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spridth" if False else "axes.linewidth": 0.8,
    "figure.dpi": 200,
})
C_CORE, C_PERI, C_KL, C_JS, C_GEO, C_EUC = (
    "#c0392b", "#2c3e50", "#c0392b", "#2c3e50", "#c0392b", "#7f8c8d")

results = {"seed": SEED}


# --------------------------------------------------------------------------
# Experiment 1: divergence and incoherence (monotonicity, core vs peripheral)
# --------------------------------------------------------------------------
def experiment_1():
    rng = np.random.default_rng(SEED)
    p_star = bcft.ideal_distribution()
    mags = np.linspace(0.0, 0.95, 20)
    n_rep = 200

    def curve(dims, div):
        means, ses = [], []
        for m in mags:
            vals = [div(bcft.perturb(p_star, m, dims, rng), p_star)
                    for _ in range(n_rep)]
            means.append(np.mean(vals)); ses.append(np.std(vals) / np.sqrt(n_rep))
        return np.array(means), np.array(ses)

    kl_core, kl_core_se = curve(bcft.CORE_DIMS, bcft.kl_divergence)
    kl_peri, kl_peri_se = curve(bcft.PERIPHERAL_DIMS, bcft.kl_divergence)
    js_core, js_core_se = curve(bcft.CORE_DIMS, bcft.js_divergence)

    # monotonicity check on the mean curves
    mono = bool(np.all(np.diff(kl_core) >= -1e-9) and np.all(np.diff(kl_peri) >= -1e-9))
    results["exp1"] = {
        "monotonic_kl_core": mono,
        "kl_core_at_max": float(kl_core[-1]),
        "kl_peripheral_at_max": float(kl_peri[-1]),
        "core_to_peripheral_ratio_at_max": float(kl_core[-1] / max(kl_peri[-1], 1e-9)),
    }

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.errorbar(mags, kl_core, yerr=kl_core_se, color=C_CORE, lw=2, label="core perturbation")
    a1.errorbar(mags, kl_peri, yerr=kl_peri_se, color=C_PERI, lw=2, label="peripheral perturbation")
    a1.set_xlabel("perturbation magnitude"); a1.set_ylabel(r"$D_{KL}(P_p \parallel P^\star)$")
    a1.set_title("Divergence rises monotonically with distortion"); a1.legend(frameon=False)

    a2.plot(mags, -kl_core, color=C_CORE, lw=2, label="core")
    a2.plot(mags, -kl_peri, color=C_PERI, lw=2, label="peripheral")
    a2.set_xlabel("perturbation magnitude"); a2.set_ylabel(r"coherence $C(p) = -D_{KL}$")
    a2.set_title("Coherence declines; core distortion costs more"); a2.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_exp1_monotonic.png"), bbox_inches="tight")
    plt.close(fig)
    return p_star


# --------------------------------------------------------------------------
# Experiment 2: geodesic vs Euclidean movement (K = 3 for ternary plotting)
# --------------------------------------------------------------------------
def _ternary_xy(P):
    """Map a 3-simplex point to 2D barycentric coordinates for plotting."""
    P = np.atleast_2d(P)
    x = 0.5 * (2 * P[:, 1] + P[:, 2]) / P.sum(axis=1)
    y = (np.sqrt(3) / 2) * P[:, 2] / P.sum(axis=1)
    return x, y


def _entropy(p):
    p = bcft.normalize(p)
    return float(-np.sum(p * np.log(p)))


def experiment_2():
    p = bcft.normalize(np.array([0.80, 0.15, 0.05]))   # start (core-heavy)
    q = bcft.normalize(np.array([0.05, 0.15, 0.80]))   # target (status-heavy)
    ts = np.linspace(0, 1, 80)
    geo = np.array([bcft.geodesic(p, q, t) for t in ts])
    euc = np.array([bcft.euclidean_interp(p, q, t) for t in ts])

    L_geo, L_euc = bcft.path_length(geo), bcft.path_length(euc)
    d_fr = bcft.fisher_rao_distance(p, q)
    # Fisher-Rao deviation of the Euclidean path from the intrinsic geodesic,
    # comparing states at the same interpolation parameter t.
    deviation = np.array([bcft.fisher_rao_distance(euc[i], geo[i]) for i in range(len(ts))])
    results["exp2"] = {
        "fisher_rao_distance_endpoints": d_fr,
        "geodesic_path_length": L_geo,
        "euclidean_path_length": L_euc,
        "euclidean_excess_pct": float(100 * (L_euc - L_geo) / L_geo),
        "geodesic_matches_fr_distance": bool(abs(L_geo - d_fr) < 1e-3),
        "max_fr_deviation_euclidean_from_geodesic": float(deviation.max()),
    }

    fig, (ax, ae) = plt.subplots(1, 2, figsize=(11, 5))
    # -- left: ternary paths --
    tri = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    tx, ty = _ternary_xy(tri); ax.plot(tx, ty, color="#cccccc", lw=1)
    for lab, vert in zip(["core", "mid", "status"], [[1,0,0],[0,1,0],[0,0,1]]):
        vx, vy = _ternary_xy(np.array(vert))
        ax.annotate(lab, (vx[0], vy[0]), ha="center", fontsize=9, color="#666")
    gx, gy = _ternary_xy(geo); ex, ey = _ternary_xy(euc)
    ax.plot(ex, ey, color=C_EUC, lw=2.4, ls=(0, (5, 3)),
            label=f"Euclidean (FR length {L_euc:.3f})")
    ax.plot(gx, gy, color=C_GEO, lw=2.6,
            label=f"Fisher-Rao geodesic ({L_geo:.3f})")
    for pt, col in [(p, "#16a085"), (q, "#8e44ad")]:
        px, py = _ternary_xy(np.array(pt)); ax.scatter(px, py, s=90, color=col, zorder=5, edgecolor="white")
    ax.set_title("Geodesic vs. Euclidean repositioning")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=9)
    ax.set_aspect("equal"); ax.axis("off")
    # -- right: Fisher-Rao deviation of Euclidean path from the geodesic --
    ae.fill_between(ts, 0, deviation, color=C_EUC, alpha=0.25)
    ae.plot(ts, deviation, color=C_EUC, lw=2.4)
    ae.set_xlabel("interpolation parameter $t$")
    ae.set_ylabel(r"FR distance between $P^{euc}(t)$ and $P^{geo}(t)$")
    ae.set_title(f"Euclidean path strays from the geodesic (max {deviation.max():.3f})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_exp2_geodesic.png"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Experiment 3: convergent natural-gradient coherence flows
# --------------------------------------------------------------------------
def experiment_3(p_star):
    rng = np.random.default_rng(SEED + 1)
    K = len(p_star)
    inits = [bcft.normalize(rng.dirichlet(np.ones(K))) for _ in range(6)]
    eta, steps = 0.25, 150

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    final_kls = []
    for i, p0 in enumerate(inits):
        traj, klh, frh = bcft.coherence_flow(p0, p_star, eta=eta, steps=steps)
        final_kls.append(klh[-1])
        a1.plot(klh, color=plt.cm.viridis(i / len(inits)), lw=1.6)
    a1.set_xlabel("iteration"); a1.set_ylabel(r"$D_{KL}(P_\theta \parallel P^\star)$")
    a1.set_yscale("log"); a1.set_ylim(1e-16, 3.0)
    a1.set_title("Coherence flow converges (6 random starts)")

    # convergence on a 3-d projection: rerun in K=3 for a ternary trajectory
    ps3 = bcft.normalize(np.array([0.6, 0.3, 0.1]))
    inits3 = [bcft.normalize(rng.dirichlet(np.ones(3))) for _ in range(5)]
    tri = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0]]); tx, ty = _ternary_xy(tri)
    a2.plot(tx, ty, color="#cccccc", lw=1)
    for i, p0 in enumerate(inits3):
        traj, _, _ = bcft.coherence_flow(p0, ps3, eta=0.3, steps=120)
        trx, try_ = _ternary_xy(traj)
        a2.plot(trx, try_, color=plt.cm.plasma(i / len(inits3)), lw=1.4)
        a2.scatter(trx[0], try_[0], s=30, color="#2c3e50", zorder=4)
    sx, sy = _ternary_xy(ps3); a2.scatter(sx, sy, s=160, color=C_CORE, marker="*", zorder=6, edgecolor="white")
    a2.set_title(r"Trajectories descend to the ideal $P^\star$"); a2.set_aspect("equal"); a2.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_exp3_flow.png"), bbox_inches="tight")
    plt.close(fig)

    results["exp3"] = {
        "eta": eta, "steps": steps,
        "max_final_kl": float(np.max(final_kls)),
        "converged_all": bool(np.max(final_kls) < 1e-3),
    }


# --------------------------------------------------------------------------
# Synthetic dataset export
# --------------------------------------------------------------------------
def export_dataset():
    rng = np.random.default_rng(SEED + 2)
    p_star = bcft.ideal_distribution()
    rows = []
    sources = {"institutional": (bcft.CORE_DIMS, 0.05),
               "campaign": (bcft.CORE_DIMS, 0.15),
               "reviews": (bcft.PERIPHERAL_DIMS, 0.45),
               "lifestyle": (bcft.PERIPHERAL_DIMS, 0.30)}
    for src, (dims, mag) in sources.items():
        for j in range(25):
            p = bcft.perturb(p_star, mag * rng.uniform(0.6, 1.4), dims, rng)
            rows.append({
                "observation_id": f"{src}_{j:02d}", "source": src,
                **{d: float(p[k]) for k, d in enumerate(bcft.SEMANTIC_DIMENSIONS)},
                "kl_to_ideal": bcft.kl_divergence(p, p_star),
                "js_to_ideal": bcft.js_divergence(p, p_star),
                "fisher_rao_to_ideal": bcft.fisher_rao_distance(p, p_star),
                "coherence": bcft.coherence(p, p_star),
            })
    # write CSV
    header = list(rows[0].keys())
    with open(os.path.join(DAT, "synthetic_brand_field.csv"), "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in header) + "\n")
    # ideal distribution
    with open(os.path.join(DAT, "ideal_distribution.json"), "w") as f:
        json.dump({d: float(p_star[k]) for k, d in enumerate(bcft.SEMANTIC_DIMENSIONS)}, f, indent=2)
    results["dataset"] = {"n_observations": len(rows), "sources": list(sources.keys())}


if __name__ == "__main__":
    p_star = experiment_1()
    experiment_2()
    experiment_3(p_star)
    export_dataset()
    with open(os.path.join(DAT, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
