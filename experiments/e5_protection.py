"""E5: protection under crosstalk drift on the distance-5 surface code (Stim + PyMatching).
Couplings among the 72 nearest spectator pairs, outside the declared E+ (r = 1.0), switch on.
Scenarios: step onset at window 4, step onset at window 8, linear ramp from window 4.
Policies: fixed reference unit, fixed serialized unit (the only unit verified for r = 1.5),
uniform switching, adaptive (probing + nuisance-adjusted estimation + calibrated alarm),
and an oracle with privileged knowledge of the true per-window failure rates.
Logical failures are reported per unit elapsed time; each (scenario, seed) is an independent replicate."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pymatching
from vss.core import apply_schedule, compile_circuit, duration, empirical_y, dem_mechanisms, feature_set_from
from vss.bank import load, nominal_noise, candidate_schedules, universe, design, nuisance_adjusted, identifiable

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
SEEDS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else "7,11,13,17,19".split(","))]
M0, coords, nq = load("surface")
sc = {s.name: s for s in candidate_schedules(M0, coords)}
WORK, SAFE = "ref", "xstripe-all"
PROBES = ["xstripe-L0", "xstripe-L1", "xstripe-L2", "xstripe-L3"]
UNITS = [WORK, SAFE] + PROBES
Ms = {n: apply_schedule(M0, sc[n]) for n in UNITS}
T = {n: duration(Ms[n]) for n in UNITS}
allM = [apply_schedule(M0, s) for s in sc.values()]
U = universe(allM, coords, 2.0)
onset_pairs = universe(allM, coords, 1.5)
W, SHOTS_W, ETA, CAL, THETA = 12, 40_000, 0.2, 40, 1e-3
SCEN = {"step-w4": lambda w: THETA if w >= 4 else 0.0,
        "step-w8": lambda w: THETA if w >= 8 else 0.0,
        "ramp-w4": lambda w: 0.0 if w < 4 else 2 * THETA * (w - 3) / 8}
t0 = time.time()
dec = {n: pymatching.Matching.from_detector_error_model(
    compile_circuit(Ms[n], nq, nominal_noise(nq), coords).detector_error_model(decompose_errors=True)) for n in UNITS}
cache = {}
def circuit(n, th):
    k = (n, round(th, 9))
    if k not in cache:
        nz = nominal_noise(nq); nz.theta = {e: th for e in onset_pairs} if th > 0 else {}
        cache[k] = compile_circuit(Ms[n], nq, nz, coords)
    return cache[k]
def run(n, th, shots, seed):
    d, o = circuit(n, th).compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return d, int((dec[n].decode_batch(d) != o).any(1).sum())

nd = compile_circuit(Ms[WORK], nq, nominal_noise(nq), coords).num_detectors
mx = []
for n in [WORK] + PROBES:
    nz = nominal_noise(nq); nz.theta = {e: 2e-3 for e in U}
    mx.append(dem_mechanisms(compile_circuit(Ms[n], nq, nz, coords))[0])
feats = feature_set_from(mx, nd)
Dsg = {n: design(Ms[n], nq, coords, U, feats, nd) for n in [WORK] + PROBES}
print(f"designs ready ({time.time()-t0:.0f}s)", flush=True)

def estimate(Y, N):
    names = [n for n in [WORK] + PROBES if N.get(n, 0) > 50]
    Ws = []
    for n in names:
        m = np.exp(-2 * Dsg[n][0]); Ws.append(4 * m**2 * N[n] / np.clip(1 - m**2, 1e-12, None))
    B = np.vstack([np.sqrt(w)[:, None] * Dsg[n][1] for n, w in zip(names, Ws)])
    A = np.vstack([np.sqrt(w)[:, None] * Dsg[n][2] for n, w in zip(names, Ws)])
    y = np.concatenate([np.sqrt(w) * (Y[n] - Dsg[n][0]) for n, w in zip(names, Ws)])
    Mx = np.hstack([B, A]); cov = np.linalg.pinv(Mx.T @ Mx, rcond=1e-10); phi = cov @ (Mx.T @ y)
    sd = np.sqrt(np.clip(np.diag(cov)[B.shape[1]:], 1e-30, None))
    ok, _ = identifiable(nuisance_adjusted([Dsg[n][1] for n in names], [Dsg[n][2] for n in names], Ws))
    return phi[B.shape[1]:], sd, ok

k = int(ETA * SHOTS_W) // len(PROBES)
PROBE_ALLOC = {WORK: SHOTS_W - k * len(PROBES), **{p: k for p in PROBES}}
def window_stats(th, seed):
    Y, N, fails, el = {}, {}, 0, 0.0
    for i, (n, s) in enumerate(PROBE_ALLOC.items()):
        d, f = run(n, th, s, seed + i); Y[n] = empirical_y(d, feats)[0]; N[n] = s; fails += f; el += s * T[n]
    return Y, N, fails, el

out = []
for seed in SEEDS:
    cal = []
    for r in range(CAL):
        Y, N, _, _ = window_stats(0.0, 10**8 + seed * 10**5 + r * 10)
        cal.append(estimate(Y, N))
    th_c = np.array([c[0] for c in cal]); ok = cal[0][2]; sd_m = cal[0][1]
    mu0, sd0 = th_c.mean(0), th_c.std(0, ddof=1)
    zc = np.array([np.max(np.where(ok, (t - mu0) / np.maximum(sd0, sd_m), 0)) for t in th_c])
    ZCRIT = float(1.1 * zc.max())
    print(f"seed {seed}: threshold {ZCRIT:.2f} ({time.time()-t0:.0f}s)", flush=True)
    for scen, f_th in SCEN.items():
        state = {"unit": WORK, "alarm": None}
        for w in range(W):
            th = f_th(w); base = seed * 10**6 + w * 1000
            rec = {}
            for pol in ["fixed-ref", "fixed-serialized", "uniform-switching", "oracle"]:
                alloc = {"fixed-ref": {WORK: SHOTS_W}, "fixed-serialized": {SAFE: SHOTS_W},
                         "uniform-switching": {n: SHOTS_W // len(UNITS) for n in UNITS}}.get(pol)
                if pol == "oracle":
                    continue
                fails = el = 0
                for i, (n, s) in enumerate(alloc.items()):
                    _, f = run(n, th, s, base + 10 * i + hash(pol) % 7); fails += f; el += s * T[n]
                rec[pol] = (fails, el)
            # oracle: privileged choice from an independent pilot of both units at the true parameters,
            # then charged with the failures of the chosen unit in the main sample (no selection bias)
            pr = run(WORK, th, 4 * SHOTS_W, base + 900)[1] / (4 * SHOTS_W * T[WORK])
            ps = run(SAFE, th, 4 * SHOTS_W, base + 901)[1] / (4 * SHOTS_W * T[SAFE])
            rec["oracle"] = rec["fixed-ref"] if pr <= ps else rec["fixed-serialized"]
            if state["unit"] == WORK:
                Y, N, fails, el = window_stats(th, base + 500)
                thh, sd, okk = estimate(Y, N)
                z = np.where(okk, (thh - mu0) / np.maximum(sd0, sd), 0.0)
                rec["adaptive"] = (fails, el)
                flagged = [list(U[j]) for j in np.where(z > ZCRIT)[0]]
                if z.max() > ZCRIT and state["alarm"] is None:
                    state["unit"] = SAFE; state["alarm"] = w
                zmax = float(z.max())
            else:
                _, f = run(SAFE, th, SHOTS_W, base + 600); rec["adaptive"] = (f, SHOTS_W * T[SAFE])
                zmax, flagged = None, []
            for pol, (f, e) in rec.items():
                out.append(dict(seed=seed, scenario=scen, window=w, theta=th, policy=pol, failures=f, elapsed=e,
                                zmax=zmax if pol == "adaptive" else None,
                                n_flagged=len(flagged) if pol == "adaptive" else None,
                                n_flagged_true=sum(tuple(x) in set(onset_pairs) for x in flagged) if pol == "adaptive" else None))
        print(f"seed {seed} {scen}: alarm at window {state['alarm']} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(dict(theta_scale=THETA, windows=W, shots_per_window=SHOTS_W, eta=ETA, calibration_windows=CAL,
                       scenarios=list(SCEN), seeds=SEEDS, durations=T, n_onset_pairs=len(onset_pairs), log=out),
                  open(f"{OUT}/e5_surface{'_'+'-'.join(map(str,SEEDS)) if len(sys.argv)>2 else ''}.json", "w"))
