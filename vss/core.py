"""Schedule compilation, noise model, verification and exact parity model
for verified schedule switching on Stim stabilizer circuits."""
from __future__ import annotations
import itertools, math, hashlib
from dataclasses import dataclass, field
import numpy as np
import stim

TWO_Q = {"CX", "CY", "CZ", "XCX", "XCZ", "YCX", "YCZ", "XCY", "YCY", "ZCX", "ZCY", "ZCZ"}
ONE_Q = {"H", "C_XYZ", "C_ZYX", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "H_YZ", "H_XY"}
MEAS = {"M", "MX", "MY", "MR", "MRX", "MRY"}
RESET = {"R", "RX", "RY", "MR", "MRX", "MRY"}
ANNOT = {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS"}


@dataclass
class Moment:
    ops: list                      # list of (name, targets as ints)
    annotations: list = field(default_factory=list)  # stim instructions emitted after ops
    pairs: list = field(default_factory=list)        # two-qubit gate pairs (name, a, b)
    key: tuple = ()                # identifies the moment type across rounds


def parse(circuit: stim.Circuit):
    """Split a noiseless flattened circuit into moments separated by TICK."""
    flat = circuit.flattened()
    coords = {}
    moments, cur = [], Moment(ops=[])
    pre = []
    for inst in flat:
        n = inst.name
        if n == "QUBIT_COORDS":
            coords[inst.targets_copy()[0].value] = tuple(inst.gate_args_copy()[:2]); continue
        if n == "TICK":
            moments.append(cur); cur = Moment(ops=[]); continue
        if n in ANNOT:
            cur.ops.append(("@", inst)); continue
        t = [x.value for x in inst.targets_copy()]
        cur.ops.append((n, t))
        if n in TWO_Q:
            for i in range(0, len(t), 2):
                cur.pairs.append((n, t[i], t[i + 1]))
    moments.append(cur)
    for m in moments:
        m.key = tuple(sorted((a, b) for _, a, b in m.pairs))
    return moments, coords, circuit.num_qubits


@dataclass
class Noise:
    p1: float = 1e-4          # after single-qubit Clifford
    p2: float = 1e-3          # after two-qubit gate
    pm: float = 2e-3          # measurement flip
    pr: float = 2e-3          # reset flip
    p_idle: np.ndarray = None # per qubit, per unit-duration moment
    meas_dur: float = 5.0     # duration of measurement/reset moments in gate-moment units
    theta: dict = field(default_factory=dict)  # pair -> crosstalk rate per gate moment


@dataclass
class Schedule:
    """For each two-qubit moment key, the set of gate pairs moved to an extra sublayer
    placed directly after it; `sham` keys receive an extra idle moment instead."""
    name: str
    split: dict = field(default_factory=dict)
    sham: frozenset = frozenset()

    def digest(self):
        s = repr(sorted((k, sorted(v)) for k, v in self.split.items())) + repr(sorted(self.sham))
        return hashlib.sha1(s.encode()).hexdigest()[:10]


def apply_schedule(moments, sched: Schedule):
    out = []
    for m in moments:
        moved = sched.split.get(m.key, set())
        if m.pairs and moved:
            keep = [p for p in m.pairs if (p[1], p[2]) not in moved]
            mv = [p for p in m.pairs if (p[1], p[2]) in moved]
            ann = [o for o in m.ops if o[0] == "@"]
            other = [o for o in m.ops if o[0] != "@" and o[0] not in TWO_Q]
            out.append(Moment(ops=other + _pairs_to_ops(keep), pairs=keep, key=m.key + ("a",)))
            out.append(Moment(ops=_pairs_to_ops(mv) + ann, pairs=mv, key=m.key + ("b",)))
        else:
            out.append(m)
        if m.pairs and m.key in sched.sham:
            out.append(Moment(ops=[], key=m.key + ("sham",)))
    return out


def _pairs_to_ops(pairs):
    ops = {}
    for n, a, b in pairs:
        ops.setdefault(n, []).extend([a, b])
    return [(n, t) for n, t in ops.items()]


def spectator_pairs(moment, coords, rmax):
    """Pairs of qubits belonging to distinct simultaneous two-qubit gates within distance rmax."""
    res = set()
    for g, h in itertools.combinations(moment.pairs, 2):
        for q in g[1:]:
            for r in h[1:]:
                if dist(coords, q, r) <= rmax + 1e-9:
                    res.add((min(q, r), max(q, r)))
    return res


def dist(coords, a, b):
    (x1, y1), (x2, y2) = coords[a], coords[b]
    return math.hypot(x1 - x2, y1 - y2)


def compile_circuit(moments, nq, noise: Noise, coords, active_pairs=None, crosstalk_only=False):
    """Emit a noisy stim circuit. Crosstalk: one Z_qZ_q' channel per active pair per
    gate moment in which the pair is a spectator pair (one elementary opportunity),
    with flip probability (1-exp(-2 theta))/2, i.e. exactly lambda = theta."""
    c = stim.Circuit()
    for q, xy in coords.items():
        c.append("QUBIT_COORDS", [q], list(xy))
    pidle = noise.p_idle if noise.p_idle is not None else np.zeros(nq)
    for m in moments:
        touched = set()
        is_meas = False
        for n, t in m.ops:
            if n == "@":
                c.append(t); continue
            touched.update(t)
            if n in MEAS:
                is_meas = True
                c.append(n, t, 0 if crosstalk_only else noise.pm)
                if n in RESET and not crosstalk_only and noise.pr > 0:
                    c.append("X_ERROR" if n == "MR" else ("Z_ERROR" if n == "MRX" else "X_ERROR"), t, noise.pr)
            elif n in RESET:
                c.append(n, t)
                if not crosstalk_only and noise.pr > 0:
                    c.append("Z_ERROR" if n == "RX" else "X_ERROR", t, noise.pr)
            else:
                c.append(n, t)
                if not crosstalk_only:
                    if n in TWO_Q and noise.p2 > 0:
                        c.append("DEPOLARIZE2", t, noise.p2)
                    elif n in ONE_Q and noise.p1 > 0:
                        c.append("DEPOLARIZE1", t, noise.p1)
        if m.pairs and noise.theta:
            for (q, r) in sorted(spectator_pairs(m, coords, 1e9)):
                th = noise.theta.get((q, r), 0.0)
                if th > 0 and (active_pairs is None or (q, r) in active_pairs):
                    c.append("CORRELATED_ERROR", [stim.target_z(q), stim.target_z(r)], (1 - math.exp(-2 * th)) / 2)
        if not crosstalk_only:
            dur = noise.meas_dur if is_meas else 1.0
            for q in range(nq):
                if q not in touched and pidle[q] > 0:
                    c.append("DEPOLARIZE1", [q], min(0.5, pidle[q] * dur))
        c.append("TICK")
    return c


def duration(moments, meas_dur=5.0):
    return sum(meas_dur if any(n in MEAS or n in RESET for n, _ in m.ops if n != '@') else 1.0 for m in moments)


# ---------------------------------------------------------------- verification
def fault_distance(circuit: stim.Circuit):
    errs = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=6,
        dont_explore_edges_with_degree_above=6,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=True)
    return len(errs), errs


# ---------------------------------------------------------------- exact parity model
def dem_mechanisms(circuit):
    dem = circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True,
                                       ignore_decomposition_failures=True)
    mech = []
    for inst in dem.flattened():
        if inst.type == "error":
            p = inst.args_copy()[0]
            dets = frozenset(t.val for t in inst.targets_copy() if t.is_relative_detector_id())
            if p > 0 and dets:
                mech.append((p, dets))
    return mech, dem.num_detectors


def feature_set_from(mechs_list, num_det, max_pairs=4000):
    """Features: every single detector and every detector pair contained in some mechanism signature."""
    pairs = set()
    for mechs in mechs_list:
        for _, d in mechs:
            if len(d) <= 6:
                for a, b in itertools.combinations(sorted(d), 2):
                    pairs.add((a, b))
    pairs = sorted(pairs)[:max_pairs]
    return [(i,) for i in range(num_det)] + pairs


def parity_matrix(features, num_det):
    import scipy.sparse as sp
    rows, cols = [], []
    for r, f in enumerate(features):
        for d in f:
            rows.append(r); cols.append(d)
    return sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(features), num_det))


def model_y(mechs, features, num_det):
    """Exact y_u = -1/2 ln E[chi_u] = sum_j (u.x_j mod 2) lambda_j, lambda_j = -1/2 ln(1-2 p_j)."""
    U = parity_matrix(features, num_det)
    if not mechs:
        return np.zeros(len(features))
    import scipy.sparse as sp
    rows, cols = [], []
    for j, (_, d) in enumerate(mechs):
        for x in d:
            rows.append(x); cols.append(j)
    X = sp.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(num_det, len(mechs)))
    par = (U.astype(np.int32) @ X).toarray() % 2
    lam = np.array([-0.5 * math.log(1 - 2 * min(p, 0.499999)) for p, _ in mechs])
    return par @ lam


def empirical_y(det_samples, features):
    """y_hat_u = -1/2 ln mean(chi_u); returns y_hat and the delta-method variance."""
    S = det_samples.astype(np.int8)
    m = np.empty(len(features))
    for r, f in enumerate(features):
        par = S[:, f[0]].copy()
        for d in f[1:]:
            par ^= S[:, d]
        m[r] = 1 - 2 * par.mean()
    n = S.shape[0]
    mm = np.clip(m, 1e-6, None)
    y = -0.5 * np.log(mm)
    var = (1 - mm ** 2) / (4 * mm ** 2 * n)
    return y, var, m
