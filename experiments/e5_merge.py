"""Merge E5 replicate files into results/e5_surface.json and write a summary."""
import json, glob, os, sys, numpy as np
OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
files = sorted(glob.glob(f"{OUT}/e5_surface_*.json"))
logs, meta = [], None
for f in files:
    d = json.load(open(f)); meta = meta or d; logs += d["log"]
seeds = sorted({x["seed"] for x in logs})
meta = dict(meta); meta["seeds"] = seeds; meta["log"] = logs
pols = ["fixed-ref", "fixed-serialized", "uniform-switching", "adaptive", "oracle"]
summ = {}
for sc in meta["scenarios"]:
    onset = {"step-w4": 4, "step-w8": 8, "ramp-w4": 4}[sc]
    summ[sc] = {}
    for p in pols:
        per = []
        for s in seeds:
            L = [x for x in logs if x["scenario"] == sc and x["policy"] == p and x["seed"] == s]
            # each window stands for an equal interval of wall-clock time: average the per-window rates
            rate = {x["window"]: x["failures"] / x["elapsed"] for x in L}
            per.append(dict(total=float(np.mean(list(rate.values()))),
                            pre=float(np.mean([v for w, v in rate.items() if w < onset])),
                            post=float(np.mean([v for w, v in rate.items() if w >= onset]))))
        summ[sc][p] = {k: dict(mean=float(np.mean([q[k] for q in per])), min=float(np.min([q[k] for q in per])),
                               max=float(np.max([q[k] for q in per]))) for k in ("total", "pre", "post")}
    al = []
    for s in seeds:
        L = sorted([x for x in logs if x["scenario"] == sc and x["policy"] == "adaptive" and x["seed"] == s and x["zmax"] is not None], key=lambda x: x["window"])
        # alarm window = last window with an estimate (the policy stops probing after the alarm)
        last = L[-1] if L else None
        alarm = last["window"] if last and len(L) < meta["windows"] else None
        al.append(dict(seed=s, alarm=alarm, flagged=last["n_flagged"] if alarm is not None else 0,
                       flagged_true=last["n_flagged_true"] if alarm is not None else 0))
    summ[sc]["alarms"] = al
meta["summary"] = summ
json.dump(meta, open(f"{OUT}/e5_surface.json", "w"))
for sc in summ:
    print(sc, [(a["alarm"], a["flagged"], a["flagged_true"]) for a in summ[sc]["alarms"]])
    for p in pols:
        t = summ[sc][p]; print(f"  {p:18s} total {t['total']['mean']:.2e} [{t['total']['min']:.2e},{t['total']['max']:.2e}]  pre {t['pre']['mean']:.2e}  post {t['post']['mean']:.2e}")
