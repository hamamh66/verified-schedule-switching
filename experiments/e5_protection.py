"""E5: protection under drift on the distance-5 surface code (Stim + PyMatching).
A cluster of nearest-spectator couplings outside the declared E+ (r=1.0) switches on.
Policies are compared by logical failures per unit elapsed time."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pymatching
from vss.core import apply_schedule, compile_circuit, duration, empirical_y, dem_mechanisms, feature_set_from
from vss.bank import load, nominal_noise, candidate_schedules, universe, design, nuisance_adjusted, identifiable

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
SEED = 7
rng = np.random.default_rng(SEED)
M0, coords, nq = load("surface")
sc = {s.name: s for s in candidate_schedules(M0, coords)}
WORK, SAFE = "ref", "xstripe-all"           # SAFE is the only unit verified for E+(r=1.5)
PROBES = ["xstripe-L0", "xstripe-L1", "xstripe-L2", "xstripe-L3"]
UNITS = [WORK, SAFE] + PROBES
Ms = {n: apply_schedule(M0, sc[n]) for n in UNITS}
T = {n: duration(Ms[n]) for n in UNITS}
U = universe([apply_schedule(M0, s) for s in sc.values()], coords, 2.0)
onset_pairs = universe([apply_schedule(M0, s) for s in sc.values()], coords, 1.5)
THETA_ON, W, ONSET, SHOTS_W, ETA = 1e-3, 12, 4, 40_000, 0.2
t0 = time.time()

dec = {n: pymatching.Matching.from_detector_error_model(
    compile_circuit(Ms[n], nq, nominal_noise(nq), coords).detector_error_model(decompose_errors=True)) for n in UNITS}
circ_cache = {}
def run(n, on, shots, seed):
    key = (n, on)
    if key not in circ_cache:
        nz = nominal_noise(nq); nz.theta = {e: THETA_ON for e in onset_pairs} if on else {}
        circ_cache[key] = compile_circuit(Ms[n], nq, nz, coords)
    d, o = circ_cache[key].compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    fail = (dec[n].decode_batch(d) != o).any(1)
    return d, int(fail.sum())

# design for the estimator (bank used while E+ = r 1.0): working unit + probes
mechs = [dem_mechanisms(compile_circuit(Ms[n], nq, nominal_noise(nq), coords))[0] for n in [WORK] + PROBES]
nd = compile_circuit(Ms[WORK], nq, nominal_noise(nq), coords).num_detectors
mech_x = []
for n in [WORK] + PROBES:
    nz = nominal_noise(nq); nz.theta = {e: 2e-3 for e in U}
    mech_x.append(dem_mechanisms(compile_circuit(Ms[n], nq, nz, coords))[0])
feats = feature_set_from(mech_x, nd)
Dsg = {n: design(Ms[n], nq, coords, U, feats, nd) for n in [WORK] + PROBES}
print(f"designs ready {time.time()-t0:.0f}s, {len(feats)} features, |U|={len(U)}", flush=True)

def estimate(Y, N):
    names = [n for n in [WORK] + PROBES if N.get(n, 0) > 50]
    Ws = []
    for n in names:
        m = np.exp(-2 * Dsg[n][0]); Ws.append(4 * m**2 * N[n] / np.clip(1 - m**2, 1e-12, None))
    B = np.vstack([np.sqrt(w)[:, None] * Dsg[n][1] for n, w in zip(names, Ws)])
    A = np.vstack([np.sqrt(w)[:, None] * Dsg[n][2] for n, w in zip(names, Ws)])
    y = np.concatenate([np.sqrt(w) * (Y[n] - Dsg[n][0]) for n, w in zip(names, Ws)])
    Mx = np.hstack([B, A])
    F = Mx.T @ Mx
    cov = np.linalg.pinv(F, rcond=1e-10)
    phi = cov @ (Mx.T @ y)
    th = phi[B.shape[1]:]; sd = np.sqrt(np.clip(np.diag(cov)[B.shape[1]:], 1e-30, None))
    Ap = nuisance_adjusted([Dsg[n][1] for n in names], [Dsg[n][2] for n in names], Ws)
    ok, _ = identifiable(Ap)
    return th, sd, ok

ZCRIT = 4.0  # per-pair level after empirical null calibration; one-sided Gaussian p ~ 3e-5
# Null calibration on disjoint trajectories without crosstalk (same allocation as a monitoring window)
CAL = 40
k = int(ETA * SHOTS_W) // len(PROBES)
cal_alloc = {WORK: SHOTS_W - k * len(PROBES), **{p: k for p in PROBES}}
cal = []
for r in range(CAL):
    Y, N = {}, {}
    for i, (n, s_) in enumerate(cal_alloc.items()):
        d, _ = run(n, False, s_, seed=999_000 + r * 10 + i)
        Y[n] = empirical_y(d, feats)[0]; N[n] = s_
    th, sd, ok = estimate(Y, N); cal.append(th)
cal = np.array(cal); mu0 = cal.mean(0); sd0 = cal.std(0, ddof=1)
zcal = np.array([np.max(np.where(ok, (c_ - mu0) / np.maximum(sd0, sd), 0)) for c_ in cal])
ZCRIT = float(1.1 * zcal.max())   # empirical max-statistic threshold, false alarm per window about 1/(CAL+1)
print(f"threshold {ZCRIT:.2f}")
print(f"null calibration done ({time.time()-t0:.0f}s); max |bias|/sd_model {np.max(np.abs(mu0)/sd):.1f}", flush=True)
policies = {"fixed-ref": [], "fixed-serialized": [], "uniform-switching": [], "adaptive": []}
state = {"adaptive_unit": WORK, "alarm_window": None}
log = []
for w in range(W):
    on = w >= ONSET
    for pol in policies:
        fails = 0; time_used = 0.0
        if pol == "fixed-ref": alloc = {WORK: SHOTS_W}
        elif pol == "fixed-serialized": alloc = {SAFE: SHOTS_W}
        elif pol == "uniform-switching": alloc = {n: SHOTS_W // len(UNITS) for n in UNITS}
        else:
            if state["adaptive_unit"] == WORK:
                k = int(ETA * SHOTS_W) // len(PROBES)
                alloc = {WORK: SHOTS_W - k * len(PROBES), **{p: k for p in PROBES}}
            else:
                alloc = {SAFE: SHOTS_W}
        Y, N = {}, {}
        for i, (n, s) in enumerate(alloc.items()):
            d, f = run(n, on, s, seed=SEED * 10**6 + w * 1000 + i * 10 + list(policies).index(pol))
            fails += f; time_used += s * T[n]
            if pol == "adaptive": Y[n] = empirical_y(d, feats)[0]; N[n] = s
        rec = dict(window=w, policy=pol, onset=on, failures=fails, elapsed=time_used, rate=fails / time_used)
        if pol == "adaptive" and state["adaptive_unit"] == WORK:
            th, sd, ok = estimate(Y, N)
            z = np.where(ok, (th - mu0) / np.maximum(sd0, sd), 0.0)
            rec.update(max_z=float(z.max()), n_ident=int(ok.sum()),
                       flagged=[list(U[j]) for j in np.where(z > ZCRIT)[0]])
            if z.max() > ZCRIT:
                state["adaptive_unit"] = SAFE; state["alarm_window"] = w
        rec["unit_next"] = state["adaptive_unit"] if pol == "adaptive" else None
        policies[pol].append(rec); log.append(rec)
        print(f"w{w} {pol:18s} fails={fails:5d} rate={fails/time_used:.2e} "
              f"{'z='+format(rec.get('max_z',0),'.1f') if pol=='adaptive' else ''} ({time.time()-t0:.0f}s)", flush=True)
json.dump(dict(theta_on=THETA_ON, onset_window=ONSET, windows=W, shots_per_window=SHOTS_W, eta=ETA, zcrit=ZCRIT, calibration_windows=CAL,
               durations=T, units=UNITS, n_onset_pairs=len(onset_pairs), alarm_window=state["alarm_window"],
               onset_pairs=[list(e) for e in onset_pairs], log=log), open(f"{OUT}/e5_surface.json", "w"), indent=1)
