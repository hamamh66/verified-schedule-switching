"""Candidate schedules, verified banks, and design matrices."""
from __future__ import annotations
import numpy as np, stim
from .core import (parse, Noise, Schedule, apply_schedule, compile_circuit, spectator_pairs,
                   fault_distance, dem_mechanisms, model_y, duration)

CODES = {
    "surface": dict(task="surface_code:rotated_memory_x", distance=5, rounds=3),
    "color":   dict(task="color_code:memory_xyz", distance=5, rounds=3),
}


def load(code):
    spec = CODES[code]
    base = stim.Circuit.generated(spec["task"], distance=spec["distance"], rounds=spec["rounds"])
    return parse(base)


def nominal_noise(nq):
    return Noise(p1=1e-4, p2=1e-3, pm=2e-3, pr=2e-3, p_idle=np.full(nq, 2e-4))


RULES = {
    "xstripe": lambda c: int(c[0] // 2) % 2,
    "ystripe": lambda c: int(c[1] // 2) % 2,
    "diag":    lambda c: int((c[0] + c[1]) // 2) % 2,
    "anti":    lambda c: int((c[0] - c[1]) // 2) % 2,
}


def candidate_schedules(moments, coords):
    keys = []
    for m in moments:
        if m.pairs and m.key not in keys:
            keys.append(m.key)
    pairs_of = {m.key: [(a, b) for _, a, b in m.pairs] for m in moments if m.pairs}
    out = [Schedule("ref")]
    for rname, rule in RULES.items():
        def moved(k):
            # gate assigned by the coordinates of its first (control) qubit
            return {(a, b) for (a, b) in pairs_of[k] if rule(coords[a]) == 1}
        out.append(Schedule(f"{rname}-all", split={k: moved(k) for k in keys}))
        for i, k in enumerate(keys):
            out.append(Schedule(f"{rname}-L{i}", split={k: moved(k)}))
    for i, k in enumerate(keys):
        out.append(Schedule(f"sham-L{i}", sham=frozenset([k])))
    # drop schedules identical to another (e.g. empty moved sets)
    seen, uniq = set(), []
    for s in out:
        d = s.digest() if (s.split and any(s.split.values())) or s.sham else "ref"
        if d not in seen:
            seen.add(d); uniq.append(s)
    return uniq


def universe(moments_list, coords, R):
    U = set()
    for M in moments_list:
        for m in M:
            if m.pairs:
                U |= spectator_pairs(m, coords, R)
    return sorted(U)


def verify(moments, nq, coords, Eplus, theta_v=2e-3):
    """Admit a schedule if adding one ZZ crosstalk event per active pair of E+ and opportunity
    does not lower the circuit-level fault distance below that of the reference without crosstalk."""
    nz = nominal_noise(nq)
    nz.theta = {e: theta_v for e in Eplus}
    c = compile_circuit(moments, nq, nz, coords)
    d, errs = fault_distance(c)
    return d, errs


def design(moments, nq, coords, U, features, num_det, h=1e-5):
    """Exact crosstalk columns A (lambda = theta per opportunity) and first-order nuisance columns B."""
    nz = nominal_noise(nq)
    base = model_y(dem_mechanisms(compile_circuit(moments, nq, nz, coords))[0], features, num_det)
    cols_B = []
    for name in ("p1", "p2", "pm", "pr"):
        n2 = nominal_noise(nq); setattr(n2, name, getattr(n2, name) + h)
        y = model_y(dem_mechanisms(compile_circuit(moments, nq, n2, coords))[0], features, num_det)
        cols_B.append((y - base) / h)
    for q in range(nq):
        n2 = nominal_noise(nq); n2.p_idle = n2.p_idle.copy(); n2.p_idle[q] += h
        y = model_y(dem_mechanisms(compile_circuit(moments, nq, n2, coords))[0], features, num_det)
        cols_B.append((y - base) / h)
    B = np.array(cols_B).T
    th = 1e-3
    cols_A = []
    for e in U:
        nx = Noise(p1=0, p2=0, pm=0, pr=0, p_idle=np.zeros(nq), theta={e: th})
        mech, _ = dem_mechanisms(compile_circuit(moments, nq, nx, coords, crosstalk_only=True))
        cols_A.append(model_y(mech, features, num_det) / th)
    A = np.array(cols_A).T
    return base, B, A


def nuisance_adjusted(Bs, As, Ws):
    """Return A_perp for stacked units (shared nuisance)."""
    B = np.vstack([np.sqrt(w)[:, None] * b for b, w in zip(Bs, Ws)])
    A = np.vstack([np.sqrt(w)[:, None] * a for a, w in zip(As, Ws)])
    Q, R = np.linalg.qr(B)
    rank = np.sum(np.abs(np.diag(R)) > 1e-9 * np.abs(np.diag(R)).max())
    Q = Q[:, :rank] if rank < Q.shape[1] else Q
    # robust projector via SVD of B
    u, s, vt = np.linalg.svd(B, full_matrices=False)
    u = u[:, s > 1e-10 * s.max()]
    Ap = A - u @ (u.T @ A)
    return Ap


def identifiable(Ap, tol=1e-6):
    """Coordinate e is identifiable iff e_e lies in the row space of A_perp."""
    u, s, vt = np.linalg.svd(Ap, full_matrices=False)
    if s.size == 0 or s.max() == 0:
        return np.zeros(Ap.shape[1], bool), s
    V = vt[s > tol * s.max()]
    proj = np.einsum("ij,ij->j", V, V)
    return proj > 1 - 1e-6, s
