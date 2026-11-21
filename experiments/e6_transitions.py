"""E6: verification of transitions (switching inside a trajectory).
For every ordered pair (A, B) of verified units, the trajectories A,B,B and A,A,B
(switch after round 1 or round 2) must preserve the reference fault distance with the events of E+."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vss.core import apply_schedule, apply_schedule_rounds, compile_circuit, fault_distance
from vss.bank import load, nominal_noise, candidate_schedules, universe

OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
code = sys.argv[1]
E1 = json.load(open(f"{OUT}/e1_{code}.json"))
M0, coords, nq = load(code)
sc = {s.name: s for s in candidate_schedules(M0, coords)}
allM = [apply_schedule(M0, s) for s in sc.values()]
t0 = time.time(); res = {}
for r in E1["radii"]:
    bank = E1["frontier"][str(r)]["bank"]
    Eplus = universe(allM, coords, r)
    nz = nominal_noise(nq); nz.theta = {e: 2e-3 for e in Eplus}
    edges, fails = 0, []
    for a in bank:
        for b in bank:
            if a == b: continue
            ok = True
            for seq in ([a, b, b], [a, a, b]):
                M = apply_schedule_rounds(M0, [sc[x] for x in seq])
                d, _ = fault_distance(compile_circuit(M, nq, nz, coords))
                if d < E1["d_ref"]:
                    ok = False; fails.append([seq, d]); break
            edges += ok
    n = len(bank)
    res[str(r)] = dict(units=n, ordered_pairs=n * (n - 1), verified_edges=edges, failures=fails[:20])
    print(f"[{code}] r={r}: {edges}/{n*(n-1)} transitions verified ({time.time()-t0:.0f}s)", flush=True)
json.dump(dict(code=code, d_ref=E1["d_ref"], transitions=res), open(f"{OUT}/e6_{code}.json", "w"), indent=1)
