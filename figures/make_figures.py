"""Generate all manuscript figures from results/*.json (Figs 1, 3-7) and Fig 2 schematic."""
import json, sys, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
R = sys.argv[1] if len(sys.argv) > 1 else "results"; O = sys.argv[2] if len(sys.argv) > 2 else "figures"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
C = {"surface": "#1f77b4", "color": "#d62728"}
L = {"surface": "surface code $d=5$", "color": "$[[19,1,5]]$ color code"}
J = lambda f: json.load(open(os.path.join(R, f)))

# ---------------- Fig 1: framework overview
fig = plt.figure(figsize=(7.2, 3.3)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 46)
ax.text(1, 43.5, "(a) Nested coupling sets", weight="bold")
for r, lab, col in [(10, r"$\mathcal{U}$ (estimation)", "0.85"), (7, r"$E^{+}$ (verification)", "#cfe2f3"), (4, r"$E_c$ (nominal)", "#9fc5e8")]:
    ax.add_patch(Circle((12, 26), r, fc=col, ec="0.3", lw=0.6))
ax.text(12, 26, r"$E_c$", ha="center", va="center"); ax.text(12, 32.5, r"$E^{+}$", ha="center"); ax.text(12, 35.6, r"$\mathcal{U}$", ha="center")
ax.text(1, 11, "verify against $E^{+}$;\nestimate over $\\mathcal{U}$;\nomissions outside $\\mathcal{U}$\nare seen only by the residual", fontsize=6.5, va="top")
ax.text(27, 43.5, "(b) Two layers and the closed loop", weight="bold")
boxes = [
 (36, 33, "Candidate units\nretiming, selective\nserialization, sham", "#eeeeee"),
 (56, 33, "Verification\nfault distance with\nevents of $E^{+}$ (Sec. 3.3)", "#cfe2f3"),
 (76, 33, "Verified bank\n$\\Sigma_{\\rm ver}(E^{+})$ and\ntransitions $G_T$", "#cfe2f3"),
 (76, 13, "Observation model\nparity features,\nexact $y=B\\beta+A\\theta$", "#fce5cd"),
 (56, 13, "Identification\nnuisance projected\n$A_\\perp$ (Props. 1-3)", "#fce5cd"),
 (36, 13, "Decision\ntests, residual monitor,\nunit selection (Props. 4-6)", "#d9ead3"),
]
for x, y, t, col in boxes:
    ax.add_patch(FancyBboxPatch((x - 8.5, y - 6), 17, 12, boxstyle="round,pad=0.4", fc=col, ec="0.35", lw=0.7))
    ax.text(x, y, t, ha="center", va="center", fontsize=6.6)
arr = dict(arrowstyle="->", lw=0.9, color="0.25")
for (a_, b_) in [((44.8, 33), (47.2, 33)), ((64.8, 33), (67.2, 33)), ((67.2, 13), (64.8, 13)), ((47.2, 13), (44.8, 13))]:
    ax.annotate("", xy=b_, xytext=a_, arrowprops=arr)
ax.annotate("", xy=(76, 19.6), xytext=(76, 26.4), arrowprops=arr)
ax.text(86.5, 23, "switch along $G_T$\n(randomized blocks);\nrecord detectors\nand flags", fontsize=6.2, color="0.25", va="center")
ax.annotate("", xy=(36, 26.4), xytext=(36, 19.6), arrowprops=dict(arrowstyle="->", lw=0.9, color="#cc0000"))
ax.text(34.5, 23, "alarm: enlarge $E^{+}$,\nre-verify", fontsize=6.2, color="#cc0000", ha="right", va="center")
ax.annotate("", xy=(67.2, 29), xytext=(44.8, 16), arrowprops=dict(arrowstyle="->", lw=0.9, color="#38761d", ls="--"))
ax.text(57, 21.2, "selected working unit", fontsize=6.2, color="#38761d", ha="center", rotation=29)
ax.text(27, 2, "Evaluation: Gate 1 bank and witnesses; Gate 2 contrasts and frontier; Gate 3 protection under drift; Gate 4 other codes.", fontsize=6.3, color="0.3")
fig.savefig(f"{O}/Fig1_overview.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 3: verified bank and frontier
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
for code in ["color", "surface"]:
    r = J(f"e1_{code}.json"); rad = r["radii"]
    bank = [len(r["frontier"][str(x)]["bank"]) for x in rad]
    nid = [r["frontier"][str(x)]["n_ident"] for x in rad]
    smin = [r["frontier"][str(x)]["smin"] for x in rad]
    nU = len(r["U"])
    axs[0].plot(rad, bank, "o-", color=C[code], label=L[code])
    axs[1].plot(rad, np.array(nid) / nU, "o-", color=C[code])
    axs[1].axhline(r["single_ref_identifiable"] / nU, color=C[code], ls=":", lw=1)
    axs[2].semilogy(rad, [s if s > 0 else np.nan for s in smin], "o-", color=C[code])
axs[0].set_ylabel("verified units"); axs[1].set_ylabel("identifiable fraction of $\\mathcal{U}$")
axs[2].set_ylabel("$\\sigma_{\\min}$ on identifiable set")
for a in axs: a.set_xlabel("radius of $E^{+}$")
axs[0].legend(frameon=False, fontsize=6.5, loc="upper right")
axs[1].text(2.05, 0.3, "dotted: reference\nschedule alone", fontsize=6.3)
for a, t in zip(axs, "abc"): a.set_title(f"({t})", loc="left", fontsize=8, weight="bold")
fig.tight_layout(); fig.savefig(f"{O}/Fig3_frontier.pdf"); plt.close(fig)

# ---------------- Fig 4: model validation
r = J("e2_color.json")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5))
for name, mk in [("ref", "o"), ("ystripe-all", "s")]:
    p = np.array(r[name]["pred"]); e = np.array(r[name]["emp"]); sd = np.array(r[name]["sd"])
    axs[0].errorbar(p, e, yerr=sd, fmt=mk, ms=2, lw=0.4, alpha=0.6, label=name)
    axs[1].hist((e - p) / sd, bins=40, histtype="step", density=True, label=f"{name}: mean {np.mean((e-p)/sd):.2f}, sd {np.std((e-p)/sd):.2f}")
lim = [0, max(np.max(r["ref"]["pred"]), np.max(r["ref"]["emp"])) * 1.05]
axs[0].plot(lim, lim, "k-", lw=0.6); axs[0].set_xlabel("predicted $y_u$ (exact parity model)"); axs[0].set_ylabel("sampled $\\hat y_u$")
axs[0].legend(frameon=False, fontsize=6.5)
z = np.linspace(-5, 5, 200); axs[1].plot(z, np.exp(-z**2 / 2) / np.sqrt(2 * np.pi), "k:", lw=0.8)
axs[1].set_xlabel("standardized residual"); axs[1].legend(frameon=False, fontsize=6)
for a, t in zip(axs, "ab"): a.set_title(f"({t})", loc="left", fontsize=8, weight="bold")
fig.tight_layout(); fig.savefig(f"{O}/Fig4_validation.pdf"); plt.close(fig)

# ---------------- Fig 5: finite-sample decisions
r = J("e3_color.json"); N = np.array(r["N"])
fig, ax = plt.subplots(figsize=(3.6, 2.5))
ax.semilogx(N, r["power"], "o-", color="#38761d", label="empirical power")
ax.semilogx(N, r["false_alarm"], "s-", color="#cc0000", label="empirical false alarm")
ax.semilogx(N, np.minimum(1, r["bound"]), "k--", lw=0.8, label="Bernstein bound (Prop. 4)")
ax.axvline(r["N_sufficient_alpha05"], color="0.4", ls=":", lw=0.8); ax.axhline(0.05, color="0.6", lw=0.5)
ax.text(r["N_sufficient_alpha05"] * 1.1, 0.75, "sufficient $N$\n(Prop. 4), $\\alpha=0.05$", fontsize=6.5)
ax.set_xlabel("trajectories per arm $N$"); ax.set_ylabel("probability"); ax.legend(frameon=False, fontsize=6.3, loc="center right", bbox_to_anchor=(1.0, 0.42))
fig.tight_layout(); fig.savefig(f"{O}/Fig5_power.pdf"); plt.close(fig)

# ---------------- Fig 6: estimation and monitoring
r = J("e4_color.json")
U = [tuple(e) for e in r["U"]]; ok = np.array(r["identifiable"]); act = set(map(tuple, r["active"]))
est = np.array(r["estimates"]); m = est.mean(0); s = est.std(0, ddof=1)
truth = np.array([r["theta_active"] if e in act else 0 for e in U])
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
idx = np.arange(len(U))
axs[0].errorbar(idx[ok], m[ok] * 1e3, yerr=s[ok] * 1e3, fmt="o", ms=2, lw=0.6, color="#1f77b4", label="identifiable")
axs[0].plot(idx[~ok], np.zeros((~ok).sum()), "x", color="0.5", ms=4, label="aliased (not estimated)")
axs[0].plot(idx, truth * 1e3, "_", color="#cc0000", ms=6, label="true $\\theta_e$")
axs[0].set_xlabel("pair index in $\\mathcal{U}$"); axs[0].set_ylabel("$\\hat\\theta_e$ ($10^{-3}$)"); axs[0].legend(frameon=False, fontsize=6)
nb = np.array([x[0] for x in r["omission_norms"]]); nbb = np.array([x[1] for x in r["omission_norms"]])
axs[1].semilogx(np.clip(nb / nbb, 1e-16, None), np.arange(len(nb)), "k.", ms=3)
axs[1].set_xlabel("$\\|b_\\perp\\|/\\|b\\|$"); axs[1].set_ylabel("omission (sorted)")
for lab, vals, col in [("null", r["null_rss"], "0.4"), ("visible", r["alt_rss"]["visible"], "#cc0000"), ("aliased", r["alt_rss"]["aliased"], "#1f77b4")]:
    axs[2].plot(vals, np.full(len(vals), {"null": 0, "visible": 2, "aliased": 1}[lab]), "o", color=col, ms=3)
axs[2].set_xscale("log"); axs[2].set_yticks([0, 1, 2]); axs[2].set_yticklabels(["null", "aliased\nomission", "visible\nomission"])
axs[2].set_xlabel("residual statistic")
axs[2].xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=4)); axs[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
for a, t in zip(axs, "abc"): a.set_title(f"({t})", loc="left", fontsize=8, weight="bold")
fig.tight_layout(); fig.savefig(f"{O}/Fig6_estimation_monitoring.pdf"); plt.close(fig)

# ---------------- Fig 7: protection under drift (10 replicates, 3 scenarios)
r = J("e5_surface.json"); logs = r["log"]; seeds = r["seeds"]
pols = ["fixed-ref", "uniform-switching", "adaptive", "fixed-serialized", "oracle"]
cols = {"fixed-ref": "#cc0000", "fixed-serialized": "#1f77b4", "uniform-switching": "0.5", "adaptive": "#38761d", "oracle": "k"}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.2, 1]))
W = r["windows"]
for p in pols:
    R = np.array([[next(x["failures"] / x["elapsed"] for x in logs if x["scenario"] == "step-w4" and x["policy"] == p
                        and x["seed"] == s and x["window"] == w) for w in range(W)] for s in seeds])
    axs[0].semilogy(range(W), R.mean(0), "o-" if p != "oracle" else "--", ms=2.5, lw=1, color=cols[p], label=p)
    axs[0].fill_between(range(W), R.min(0), R.max(0), color=cols[p], alpha=0.12)
axs[0].axvline(3.5, color="k", ls=":", lw=0.8); axs[0].text(3.6, 1.3e-6, "onset", fontsize=6.5)
axs[0].set_xlabel("window (scenario step-w4)"); axs[0].set_ylabel("logical failures per unit time")
h, l = axs[0].get_legend_handles_labels()
scen = r["scenarios"]; xoff = np.linspace(-0.3, 0.3, len(pols))
for i, p in enumerate(pols):
    m = [r["summary"][sc][p]["total"]["mean"] for sc in scen]
    lo = [r["summary"][sc][p]["total"]["min"] for sc in scen]; hi = [r["summary"][sc][p]["total"]["max"] for sc in scen]
    axs[1].errorbar(np.arange(len(scen)) + xoff[i], m, yerr=[np.array(m) - lo, np.array(hi) - m], fmt="o", ms=3,
                    color=cols[p], capsize=2, lw=0.8)
axs[1].set_yscale("log"); axs[1].set_xticks(range(len(scen))); axs[1].set_xticklabels(scen)
axs[1].set_ylabel("time-averaged failure rate")
for a, t in zip(axs, "ab"): a.set_title(f"({t})", loc="left", fontsize=8, weight="bold")
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=6.5)
fig.savefig(f"{O}/Fig7_protection.pdf"); plt.close(fig)
print("figures written")
