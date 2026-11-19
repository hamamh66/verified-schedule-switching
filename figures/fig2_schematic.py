import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
out=sys.argv[1]
plt.rcParams.update({"font.size":8,"font.family":"serif"})
fig=plt.figure(figsize=(7.2,2.6))
# (a) timelines
ax=fig.add_axes([0.02,0.08,0.34,0.84]); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)
ax.text(0,9.6,"(a) Two verified recovery units",weight="bold")
def tl(y0,lab,blocks,ov):
    ax.text(0,y0+1.9,lab,style="italic")
    for i,(q,segs) in enumerate(zip(["$a_1$","$a_2$","$a_3$"],blocks)):
        y=y0+1.2-i*0.8; ax.text(0.2,y-0.1,q); ax.plot([1,9.6],[y,y],color="0.75",lw=0.6)
        for s,c in segs: ax.add_patch(Rectangle((s,y-0.25),1.2,0.5,color=c))
    if ov: ax.add_patch(Rectangle((ov[0],y0+0.1),ov[1],1.4,fill=False,ls="--",ec="tab:red",lw=1))
b="tab:blue"; g="tab:green"
tl(5.6,r"$\sigma_1$: $(a_1,a_2)$ exposed",[[(1.5,b),(3,b)],[(1.6,b),(4.6,b)],[(6.2,g)]],(1.4,1.5))
ax.text(3.0,5.25,"crosstalk active",color="tab:red",fontsize=7)
tl(1.4,r"$\sigma_2$: $(a_1,a_2)$ serialized",[[(1.5,b),(3,b)],[(4.5,b),(6.0,b)],[(7.6,g)]],None)
ax.text(1.0,0.1,"longer cycle, pair not exposed",fontsize=7,color="0.3")
# (b) design matrix
ax2=fig.add_axes([0.40,0.18,0.22,0.62])
import numpy as np
D=np.array([[1,1,0,1,0,0],[1,0,1,0,1,0],[0,1,1,0,0,1],[0,0,0,1,1,1]])
ax2.imshow(D,cmap="Greys",vmin=0,vmax=1.6,aspect="auto")
ax2.set_xticks(range(6)); ax2.set_xticklabels([f"$e_{i+1}$" for i in range(6)])
for x in (1.5,3.5): ax2.axvline(x,color="tab:red",lw=1)
for x,t in ((0.5,"$E_c$"),(2.5,"$E^{+}\\setminus E_c$"),(4.5,"$\\mathcal{U}\\setminus E^{+}$")): ax2.text(x,-0.75,t,ha="center",fontsize=7,color="tab:red")
ax2.set_yticks(range(4)); ax2.set_yticklabels([f"$\\sigma_{i+1}$" for i in range(4)])
ax2.set_xlabel("candidate pairs"); ax2.set_ylabel("probe units")
fig.text(0.37,0.95,"(b) Exposure design",weight="bold")
# (c) loop
ax3=fig.add_axes([0.66,0.02,0.33,0.9]); ax3.axis("off"); ax3.set_xlim(0,10); ax3.set_ylim(0,10)
ax3.text(1.6,9.6,"(c) Closed loop",weight="bold")
boxes=[(5,8.2,"verify units and\ntransitions against $E^{+}$"),(5,6.0,"switch along verified\ntransitions (random blocks)"),
       (5,3.8,"estimate $\\hat\\theta$ with\nbaseline projected out"),(5,1.6,"residual test;\ndecision-aware choice")]
for x,y,t in boxes:
    ax3.add_patch(FancyBboxPatch((x-3.2,y-0.8),6.4,1.6,boxstyle="round,pad=0.1",fc="#eef3fb",ec="tab:blue"))
    ax3.text(x,y,t,ha="center",va="center",fontsize=7)
for y in [7.3,5.1,2.9]: ax3.annotate("",xy=(5,y-0.4),xytext=(5,y),arrowprops=dict(arrowstyle="->"))
ax3.annotate("",xy=(1.6,6.0),xytext=(1.6,1.6),arrowprops=dict(arrowstyle="->",connectionstyle="bar,fraction=-0.25"))
ax3.text(9.3,1.6,"alarm:\nleast-overlap\nunit, flag $E^{+}$",fontsize=6.5,color="tab:red",ha="center",va="center")
fig.savefig(out,bbox_inches="tight")
