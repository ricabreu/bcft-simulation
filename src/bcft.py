"""
bcft.py -- Core library for Brand Coherence Field Theory (BCFT) simulations.

Implements the formal objects of BCFT on the probability simplex:
  - brand states as probability distributions P over a semantic space,
  - an ideal distribution P*,
  - divergences D(P || P*) (Kullback-Leibler, Jensen-Shannon),
  - coherence C = -D(P || P*),
  - the Fisher-Rao metric, geodesic distance and geodesic interpolation,
  - natural-gradient (replicator) coherence flow toward P*.

The probability simplex Delta^{K-1} carries the Fisher-Rao Riemannian metric.
Under the diffeomorphism s = sqrt(p), the simplex maps isometrically (up to a
factor of 2) onto the positive orthant of the unit sphere S^{K-1}, where
geodesics are great-circle arcs. We use this representation throughout.

Author: Ricardo Abreu (companion to the BCFT working paper).
License: MIT.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------
# Simplex utilities
# --------------------------------------------------------------------------
def normalize(p: np.ndarray) -> np.ndarray:
    """Project a non-negative vector onto the probability simplex."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, EPS, None)
    return p / p.sum()


def is_distribution(p: np.ndarray, tol: float = 1e-8) -> bool:
    p = np.asarray(p, dtype=float)
    return bool(np.all(p >= -tol) and abs(p.sum() - 1.0) < 1e-6)


# --------------------------------------------------------------------------
# Divergences
# --------------------------------------------------------------------------
def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """D_KL(p || q) in nats."""
    p, q = normalize(p), normalize(q)
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (nats), symmetric and bounded by log 2."""
    p, q = normalize(p), normalize(q)
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))


def coherence(p: np.ndarray, p_star: np.ndarray, divergence=kl_divergence) -> float:
    """BCFT coherence C(p) = -D(P_p || P*)."""
    return -divergence(p, p_star)


# --------------------------------------------------------------------------
# Fisher-Rao geometry on the simplex
# --------------------------------------------------------------------------
def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Fisher-Rao geodesic distance on the simplex:
        d_FR(p, q) = 2 * arccos( sum_i sqrt(p_i q_i) ).
    The argument of arccos is the Bhattacharyya coefficient.
    """
    p, q = normalize(p), normalize(q)
    bc = np.sum(np.sqrt(p * q))
    bc = min(1.0, max(-1.0, bc))
    return float(2.0 * np.arccos(bc))


def geodesic(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    """
    Fisher-Rao geodesic interpolation between p and q at t in [0, 1].
    Computed as a great-circle (slerp) interpolation of s = sqrt(p) on the
    unit sphere, mapped back to the simplex by squaring.
    """
    p, q = normalize(p), normalize(q)
    sp, sq = np.sqrt(p), np.sqrt(q)
    omega = np.arccos(min(1.0, max(-1.0, float(np.dot(sp, sq)))))
    if omega < EPS:                       # nearly identical points
        s = (1 - t) * sp + t * sq
    else:
        s = (np.sin((1 - t) * omega) * sp + np.sin(t * omega) * sq) / np.sin(omega)
    return normalize(s ** 2)


def euclidean_interp(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    """Naive Euclidean (linear) interpolation in probability coordinates."""
    p, q = normalize(p), normalize(q)
    return normalize((1 - t) * p + t * q)


def path_length(path: np.ndarray) -> float:
    """Total Fisher-Rao length of a discretized path (array of distributions)."""
    return float(sum(fisher_rao_distance(path[i], path[i + 1])
                     for i in range(len(path) - 1)))


# --------------------------------------------------------------------------
# Natural-gradient coherence flow
# --------------------------------------------------------------------------
def kl_grad(p: np.ndarray, p_star: np.ndarray) -> np.ndarray:
    """Euclidean gradient wrt p of f(p) = D_KL(p || p*)."""
    p, p_star = normalize(p), normalize(p_star)
    return np.log(p / p_star) + 1.0


def natural_gradient_step(p: np.ndarray, p_star: np.ndarray, eta: float,
                          divergence_grad=kl_grad) -> np.ndarray:
    """
    One step of natural-gradient descent of D(p || p*) on the simplex.

    The Fisher metric on the simplex yields the replicator (mirror-descent)
    update, which keeps p on the simplex automatically:
        p_i <- p_i * exp(-eta * (g_i - <p, g>)) , renormalized,
    where g = grad f(p). This is the discrete-time analogue of the
    natural-gradient flow  dp/dt = -eta * p ⊙ (g - <p,g> 1).
    """
    p, p_star = normalize(p), normalize(p_star)
    g = divergence_grad(p, p_star)
    drift = g - np.dot(p, g)
    p_new = p * np.exp(-eta * drift)
    return normalize(p_new)


def coherence_flow(p0: np.ndarray, p_star: np.ndarray, eta: float = 0.2,
                   steps: int = 200, divergence=kl_divergence):
    """
    Run the natural-gradient coherence flow from p0 toward p*.
    Returns (trajectory[steps+1, K], kl_history[steps+1], fr_history[steps+1]).
    """
    p = normalize(p0)
    traj = [p.copy()]
    kl_hist = [kl_divergence(p, p_star)]
    fr_hist = [fisher_rao_distance(p, p_star)]
    for _ in range(steps):
        p = natural_gradient_step(p, p_star, eta)
        traj.append(p.copy())
        kl_hist.append(kl_divergence(p, p_star))
        fr_hist.append(fisher_rao_distance(p, p_star))
    return np.array(traj), np.array(kl_hist), np.array(fr_hist)


# --------------------------------------------------------------------------
# Synthetic semantic space (mission-led outdoor brand, BCFT running example)
# --------------------------------------------------------------------------
SEMANTIC_DIMENSIONS = [
    "durability",
    "environmental_stewardship",
    "repairability",
    "functional_competence",
    "style_aesthetics",
    "status_luxury",
]
CORE_DIMS = [0, 1, 2, 3]        # mission / technical core
PERIPHERAL_DIMS = [4, 5]       # style / status periphery


def ideal_distribution() -> np.ndarray:
    """Ideal distribution P* concentrated on the mission-led core."""
    w = np.array([0.24, 0.30, 0.20, 0.18, 0.05, 0.03])
    return normalize(w)


def perturb(p_star: np.ndarray, magnitude: float, dims, rng) -> np.ndarray:
    """
    Move probability mass away from the ideal along the given dimensions,
    reallocating it to the complementary dimensions. `magnitude` in [0, 1]
    is the fraction of mass on `dims` that is shifted out.
    """
    p = p_star.copy()
    dims = np.array(dims)
    other = np.array([i for i in range(len(p)) if i not in set(dims.tolist())])
    moved = magnitude * p[dims].sum()
    p[dims] *= (1 - magnitude)
    # distribute moved mass over the complementary dims, weighted randomly
    if len(other) > 0:
        w = rng.dirichlet(np.ones(len(other)))
        p[other] += moved * w
    return normalize(p)
