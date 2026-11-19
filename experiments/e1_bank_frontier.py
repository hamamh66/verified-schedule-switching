"""E1: verified banks, design matrices and the protection-information frontier."""
import sys, json, time, pickle, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from vss.core import apply_schedule, compile_circuit, dem_mechanisms, feature_set_from, model_y, duration
from vss.bank import load, nominal_noise, candidate_schedules, universe, verify, design, nuisance_adjusted, identifiable

RADII = [1.0, 1.5, 2.0, 2.5, 3.0]
R_U = 2.0
N_SHOTS = 100_000  # shots per unit used for weights (information scale)

def run(code, out):
    t0 = time.time()
    M0, coords, nq = load(code)
    scheds = candidate_schedules(M0, coords)
    Ms = {s.name: apply_schedule(M0, s) for s in scheds}
    U = universe(Ms.values(), coords, R_U)
    print(f"[{code}] {len(scheds)} candidate schedules, |U|={len(U)}", flush=True)
    # reference distance without crosstalk
    d_ref = verify(Ms["ref"], nq, coords, [])[0]
    ver = {}
    for r in RADII:
        Eplus = [e for e in universe(Ms.values(), coords, r)]
        ver[r] = {}
        for s in scheds:
            d, errs = verify(Ms[s.name], nq, coords, Eplus)
            ver[r][s.name] = d
        print(f"[{code}] r={r}: |E+|={len(Eplus)} verified {sum(v >= d_ref for v in ver[r].values())}/{len(scheds)}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
    # features from all units with all crosstalk on
    mechs = []
    for s in scheds:
        nz = nominal_noise(nq); nz.theta = {e: 2e-3 for e in U}
        c = compile_circuit(Ms[s.name], nq, nz, coords)
        m, nd = dem_mechanisms(c); mechs.append(m)
    feats = feature_set_from(mechs, nd)
    print(f"[{code}] {len(feats)} features over {nd} detectors", flush=True)
    D = {}
    for i, s in enumerate(scheds):
        base, B, A = design(Ms[s.name], nq, coords, U, feats, nd)
        m = np.exp(-2 * base)
        w = 4 * m**2 * N_SHOTS / np.clip(1 - m**2, 1e-12, None)
        D[s.name] = dict(base=base, B=B, A=A, w=w, T=duration(Ms[s.name]))
        print(f"[{code}] design {i+1}/{len(scheds)} {s.name}  ({time.time()-t0:.0f}s)", flush=True)
    front = {}
    for r in RADII:
        bank = [s.name for s in scheds if ver[r][s.name] >= d_ref]
        if not bank:
            front[r] = dict(bank=[], n_ident=0, smin=0.0); continue
        Ap = nuisance_adjusted([D[n]["B"] for n in bank], [D[n]["A"] for n in bank], [D[n]["w"] for n in bank])
        ok, s = identifiable(Ap)
        Eplus = set(universe(Ms.values(), coords, r))
        sub = Ap[:, ok]
        smin = float(np.linalg.svd(sub, compute_uv=False).min()) if ok.any() else 0.0
        front[r] = dict(bank=bank, n_ident=int(ok.sum()), smin=smin,
                        ident_in_Eplus=int(sum(ok[j] for j, e in enumerate(U) if e in Eplus)))
        print(f"[{code}] frontier r={r}: bank={len(bank)} identifiable={ok.sum()}/{len(U)} smin={smin:.3g}", flush=True)
    single = {}
    for n in ["ref"]:
        Ap = nuisance_adjusted([D[n]["B"]], [D[n]["A"]], [D[n]["w"]])
        ok, _ = identifiable(Ap); single[n] = int(ok.sum())
    res = dict(code=code, nq=nq, d_ref=d_ref, U=[list(e) for e in U], R_U=R_U, radii=RADII,
               schedules=[s.name for s in scheds], digests={s.name: s.digest() for s in scheds},
               durations={s.name: D[s.name]["T"] for s in scheds},
               verified={str(r): ver[r] for r in RADII}, frontier={str(r): front[r] for r in RADII},
               single_ref_identifiable=single["ref"], n_features=len(feats))
    json.dump(res, open(f"{out}/e1_{code}.json", "w"), indent=1)
    pickle.dump(dict(feats=feats, nd=nd, D=D, U=U, scheds=scheds), open(f"/tmp/e1_{code}.pkl", "wb"))
    print(f"[{code}] done in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
