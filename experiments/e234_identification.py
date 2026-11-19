"""E2 model validation, E3 finite-sample decisions, E4 estimation and monitoring (color code)."""
import sys, os, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, stim
from vss.core import apply_schedule, compile_circuit, empirical_y, dem_mechanisms, model_y, spectator_pairs, dist
from vss.bank import load, nominal_noise, candidate_schedules, universe, design, nuisance_adjusted, identifiable

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
PARTS = sys.argv[2] if len(sys.argv) > 2 else "234"
rng = np.random.default_rng(20261119)
code = "color"
P = pickle.load(open(f"/tmp/e1_{code}.pkl", "rb"))
E1 = json.load(open(f"{OUT}/e1_{code}.json"))
feats, nd, D, U = P["feats"], P["nd"], P["D"], [tuple(e) for e in P["U"]]
M0, coords, nq = load(code)
scheds = {s.name: s for s in P["scheds"]}
Ms = {n: apply_schedule(M0, s) for n, s in scheds.items()}
t0 = time.time()

def sample(name, theta, shots, seed=None):
    nz = nominal_noise(nq); nz.theta = dict(theta)
    c = compile_circuit(Ms[name], nq, nz, coords)
    return c.compile_detector_sampler(seed=seed).sample(shots, bit_packed=False)


def part2():
    # ---------------- E2: exact parity model vs sampled features
    theta_true = {e: 2e-3 for e in U}
    res2 = {}
    for name in ["ref", "ystripe-all"]:
        nz = nominal_noise(nq); nz.theta = theta_true
        c = compile_circuit(Ms[name], nq, nz, coords)
        pred = model_y(dem_mechanisms(c)[0], feats, nd)
        S = c.compile_detector_sampler(seed=1).sample(400_000)
        y, var, m = empirical_y(S, feats)
        z = (y - pred) / np.sqrt(var)
        # first-order coincidence model for detector pairs: P(d_i d_k = 1) approx sum of rates of events containing both
        res2[name] = dict(pred=pred.tolist(), emp=y.tolist(), sd=np.sqrt(var).tolist(), z_mean=float(z.mean()), z_sd=float(z.std()))
        print(f"E2 {name}: z mean {z.mean():.3f} sd {z.std():.3f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res2, open(f"{OUT}/e2_{code}.json", "w"))


def part3():
    # ---------------- E3: finite-sample decision between two verified units
    bank = E1["frontier"]["1.5"]["bank"]
    A = {n: D[n]["A"] for n in bank}; base = {n: D[n]["base"] for n in bank}
    best = None
    for j, e in enumerate(U):
        for n in bank:
            if n == "ref": continue
            dA = A["ref"][:, j] - A[n][:, j]
            k = int(np.argmax(np.abs(dA)))
            if best is None or abs(dA[k]) > best[0]:
                best = (abs(dA[k]), j, n, k)
    _, j3, n3, k3 = best
    e3 = U[j3]; f3 = feats[k3]
    print(f"E3 pair {e3} units ref vs {n3}, feature {f3}", flush=True)
    def stat(S):
        par = S[:, f3[0]].astype(np.int8)
        for d in f3[1:]: par ^= S[:, d]
        return par.astype(float)
    theta_min = 6e-4
    pool = {}
    for lab, th in [("null", {}), ("alt", {e3: theta_min})]:
        pool[lab] = {u: stat(sample(u, th, 3_000_000, seed=hash((lab, u)) % 2**31)) for u in ["ref", n3]}
    mu = {lab: {u: pool[lab][u].mean() for u in pool[lab]} for lab in pool}
    sgn = np.sign((mu["alt"]["ref"] - mu["alt"][n3]) - (mu["null"]["ref"] - mu["null"][n3]))
    delta = sgn * (mu["null"]["ref"] - mu["null"][n3])
    g = sgn * (mu["alt"]["ref"] - mu["alt"][n3]) - delta
    v = max(sum(mu[l][u] * (1 - mu[l][u]) for u in mu[l]) for l in mu)
    Ns = np.unique(np.logspace(3, 6.3, 12).astype(int))
    reps = 200
    fa, pw = [], []
    for N in Ns:
        r_fa = r_pw = 0
        for r in range(reps):
            for lab in ("null", "alt"):
                X = rng.choice(pool[lab]["ref"], N); Y = rng.choice(pool[lab][n3], N)
                rej = sgn * (X.mean() - Y.mean()) >= delta + g / 2
                if lab == "null": r_fa += rej
                else: r_pw += rej
        fa.append(r_fa / reps); pw.append(r_pw / reps)
        print(f"E3 N={N}: false alarm {fa[-1]:.3f} power {pw[-1]:.3f}", flush=True)
    bound = np.exp(-Ns * g**2 / (8 * v + 4 * g / 3))
    Nsuff = (8 * v + 4 * g / 3) / g**2 * np.log(1 / 0.05)
    json.dump(dict(pair=list(e3), unit=n3, feature=list(f3), theta_min=theta_min, delta=float(delta), g=float(g), v=float(v),
                   N=Ns.tolist(), false_alarm=fa, power=pw, bound=bound.tolist(), N_sufficient_alpha05=float(Nsuff)),
              open(f"{OUT}/e3_{code}.json", "w"))


def part4():
    # ---------------- E4: estimation with nuisance and monitoring
    bank4 = E1["frontier"]["1.5"]["bank"]
    SH = 60_000
    Ws = [D[n]["w"] * SH / 1e5 for n in bank4]
    Bst = np.vstack([np.sqrt(w)[:, None] * D[n]["B"] for n, w in zip(bank4, Ws)])
    Ast = np.vstack([np.sqrt(w)[:, None] * D[n]["A"] for n, w in zip(bank4, Ws)])
    Mst = np.hstack([Bst, Ast])
    Ap = nuisance_adjusted([D[n]["B"] for n in bank4], [D[n]["A"] for n in bank4], Ws)
    ok, _ = identifiable(Ap)
    Mpinv = np.linalg.pinv(Mst, rcond=1e-10)
    uM, sM, _ = np.linalg.svd(Mst, full_matrices=False); uM = uM[:, sM > 1e-10 * sM.max()]
    def fit(Ydict):
        yst = np.concatenate([np.sqrt(w) * (Ydict[n] - D[n]["base"]) for n, w in zip(bank4, Ws)])
        phi = Mpinv @ yst
        res = yst - uM @ (uM.T @ yst)
        return phi, float(res @ res)
    active = [U[i] for i in rng.choice(np.where(ok)[0], 6, replace=False)]
    th_true = {e: 4e-3 for e in active}
    # omissions: pairs at distance in (R_U, 3.2] that are spectators somewhere in the bank
    omit_cand = sorted(set(universe(Ms.values(), coords, 3.2)) - set(U))
    nb = []
    for e in omit_cand:
        cols = []
        for n in bank4:
            nx = nominal_noise(nq); nx.p1 = nx.p2 = nx.pm = nx.pr = 0; nx.p_idle = np.zeros(nq); nx.theta = {e: 1e-3}
            mech, _ = dem_mechanisms(compile_circuit(Ms[n], nq, nx, coords, crosstalk_only=True))
            cols.append(model_y(mech, feats, nd) / 1e-3)
        b = np.concatenate([np.sqrt(w) * c for c, w in zip(cols, Ws)])
        bp = b - uM @ (uM.T @ b)
        nb.append((float(np.linalg.norm(bp)), float(np.linalg.norm(b)), e))
    nb.sort()
    vis = nb[-1][2]; blind = nb[0][2]
    print(f"E4 omissions: {len(nb)} candidates; min |b_perp|={nb[0][0]:.3g} (|b|={nb[0][1]:.3g}) max={nb[-1][0]:.3g}", flush=True)
    def dataset(theta, seed):
        Y = {}
        for i, n in enumerate(bank4):
            nz = nominal_noise(nq); nz.theta = dict(theta)
            c = compile_circuit(Ms[n], nq, nz, coords)
            S = c.compile_detector_sampler(seed=seed * 1000 + i).sample(SH)
            Y[n] = empirical_y(S, feats)[0]
        return Y
    null_stats, est = [], []
    for r in range(12):
        phi, rss = fit(dataset(th_true, seed=r)); null_stats.append(rss); est.append(phi[Bst.shape[1]:].tolist())
        print(f"E4 null rep {r}: rss={rss:.1f} ({time.time()-t0:.0f}s)", flush=True)
    alt = {}
    for lab, e in [("visible", vis), ("aliased", blind)]:
        th = dict(th_true); th[e] = 6e-3
        alt[lab] = []
        for r in range(6):
            alt[lab].append(fit(dataset(th, seed=100 + r + (0 if lab == "visible" else 50)))[1])
        print(f"E4 {lab} {e}: rss={np.mean(alt[lab]):.1f}", flush=True)
    json.dump(dict(bank=bank4, shots_per_unit=SH, identifiable=ok.tolist(), U=[list(e) for e in U],
                   active=[list(e) for e in active], theta_active=4e-3, estimates=est,
                   dof=int(Mst.shape[0] - uM.shape[1]), null_rss=null_stats, alt_rss=alt,
                   omission_norms=[(a, b, list(e)) for a, b, e in nb], visible=list(vis), aliased=list(blind)),
              open(f"{OUT}/e4_{code}.json", "w"))
    print("done", time.time() - t0)

for k in PARTS:
    {'2': part2, '3': part3, '4': part4}[k]()
