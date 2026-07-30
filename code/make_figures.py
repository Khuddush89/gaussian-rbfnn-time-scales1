"""All figures and tables. Run: python code/make_figures.py"""
import os, sys, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsrbf import (quantum, uniform, real_grid, cantor, harmonic, phi_ts, dphi_dalpha,
                   TrialSolution, train, certified_bound, alpha_admissible_max)
RT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
FG=os.path.join(RT,"results","figures"); TB=os.path.join(RT,"results","tables")
for d in (FG,TB): os.makedirs(d,exist_ok=True)
CY=["#e4572e","#4361ee","#2a9d8f","#7b2cbf","#f3a712","#d81159","#06a77d","#3d348b"]
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":240,"font.family":"serif",
 "font.size":11,"axes.titlesize":12,"axes.titleweight":"bold","axes.grid":True,
 "grid.color":"#d3d6de","grid.linewidth":.7,"axes.axisbelow":True,
 "axes.edgecolor":"#41485a","lines.linewidth":2.1,"legend.framealpha":.95,
 "axes.prop_cycle":plt.cycler(color=CY)})
def sv(f,n):
    f.tight_layout()
    for d in (FG,):
        f.savefig(os.path.join(d,n+".png"),bbox_inches="tight",facecolor="white")
    plt.close(f); print("  [fig]",n)
def tb(df,n,fmt="%.4e"):
    df.to_csv(os.path.join(TB,n+".csv"),index=False)
    tex=df.to_latex(index=False,escape=False,float_format=fmt,
        column_format="l"+"r"*(df.shape[1]-1))
    open(os.path.join(TB,n+".tex"),"w").write(tex)
    print("  [tab]",n)

# ============================ architecture ============================
fig,ax=plt.subplots(figsize=(13,6.6)); ax.set_xlim(0,13); ax.set_ylim(0,6.6); ax.axis("off")
def ar(x1,y1,x2,y2,c="#41485a",lw=1.4,ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,
                 color=c,lw=lw,linestyle=ls,zorder=1))
def bx(x,y,w,h,t,fs=10.4,fc="#f4f7fb",ec="#4361ee"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.11",fc=fc,ec=ec,lw=1.7,zorder=2))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,zorder=3)
ax.add_patch(Circle((0.85,3.3),0.44,fc="#4361ee",ec="#1b2a49",lw=1.7,zorder=3))
ax.text(0.85,3.3,r"$t$",ha="center",va="center",fontsize=16,color="w",zorder=4)
ax.text(0.85,2.52,"input\n"+r"$t\in\mathbb{T}$",ha="center",va="top",fontsize=9.6)
ys=[5.55,4.55,3.55,1.85]
lb=[r"$\phi_1=e_{\ominus p_1}(t,c_1)$",r"$\phi_2=e_{\ominus p_2}(t,c_2)$",
    r"$\phi_3=e_{\ominus p_3}(t,c_3)$",r"$\phi_m=e_{\ominus p_m}(t,c_m)$"]
for i,(y,l) in enumerate(zip(ys,lb)):
    ax.add_patch(FancyBboxPatch((2.5,y-0.37),3.15,0.74,boxstyle="round,pad=0.07",
        fc="#fdece6",ec=CY[0],lw=1.8,zorder=2))
    ax.text(4.07,y,l,ha="center",va="center",fontsize=10.3,zorder=3)
    ar(1.32,3.3,2.46,y,c="#8a93a6",lw=1.1)
ax.text(4.07,2.72,r"$\vdots$",ha="center",fontsize=16)
ax.text(4.07,6.26,"hidden layer  "+r"$p_j(\tau)=\alpha_j h_1(\tau,c_j)=\alpha_j(\tau-c_j)$",
        ha="center",fontsize=10.4,color=CY[0],fontweight="bold")
ax.text(4.07,1.08,r"centres $c_j\in\mathbb{T}$ fixed;  widths $\alpha_j$ trained,"
        "\n"+r"admissible $0<\alpha_j<1/W(c_j)$",ha="center",va="top",fontsize=9.3)
ax.add_patch(Circle((6.75,3.3),0.54,fc="#2a9d8f",ec="#1b2a49",lw=1.7,zorder=3))
ax.text(6.75,3.3,r"$\Sigma$",ha="center",va="center",fontsize=18,color="w",zorder=4)
for y,l in zip(ys,[r"$v_1$",r"$v_2$",r"$v_3$",r"$v_m$"]):
    ar(5.69,y,6.28,3.3,c="#2a9d8f",lw=1.2)
    ax.text(5.96,y+0.21*(1 if y>3.3 else -1),l,fontsize=9.6,color="#1d7a6e")
ax.text(6.75,2.50,"linear output\n"+r"$N(t,p)=\sum_j v_j\phi_j(t)$",ha="center",va="top",fontsize=9.6)
bx(7.95,4.22,4.75,1.12,"trial solution — satisfies the ICs identically\n"
   r"$y_a(t,p)=\sum_{k=0}^{n-1}h_k(t,t_0)y_k+h_n(t,t_0)N(t,p)$")
bx(7.95,2.74,4.75,1.06,"residual\n"+r"$R(t,p)=F(t,y_a,y_a^{\Delta},\ldots,y_a^{\Delta^n})$",
   fc="#fff8e8",ec=CY[4])
bx(7.95,1.28,4.75,1.06,"error functional\n"+r"$E(p)=\frac{1}{2}\sum_{i=1}^{M}R^2(t_i,p)$",
   fc="#fdeaf1",ec=CY[5])
ar(7.29,3.3,7.70,3.3); ar(7.70,3.3,7.70,4.78); ar(7.70,4.78,7.92,4.78)
ar(7.70,3.3,7.92,3.27); ar(10.3,4.19,10.3,3.84); ar(10.3,2.71,10.3,2.38)
ar(7.92,1.81,6.75,1.81,c=CY[5],lw=1.7,ls="--"); ar(6.75,1.81,6.75,2.72,c=CY[5],lw=1.7,ls="--")
ar(6.40,1.81,4.10,1.55,c=CY[5],lw=1.7,ls="--")
ax.text(5.30,1.44,"update "+r"$(v,\log\alpha)$"+"\nLevenberg–Marquardt",ha="center",
        va="top",fontsize=9.4,color=CY[5])
sv(fig,"fig0_architecture")

# ============================ activation ==============================
fig,axg=plt.subplots(2,2,figsize=(12.6,8.4)); ax=axg.ravel()
tsR=real_grid(-3,3,6001); c=3000
for k,al in enumerate([0.5,1.0,4.0]):
    ax[0].plot(tsR.t,phi_ts(tsR,[c],[al])[:,0],color=CY[k],label=rf"$\alpha={al}$")
ax[0].plot(tsR.t,np.exp(-tsR.t**2/2),"k--",lw=1.4,label=r"$e^{-\alpha(t-c)^2/2}$")
ax[0].set_xlabel("$t$");ax[0].set_ylabel(r"$\phi(t,c)$")
ax[0].set_title(r"(a) $\mathbb{T}=\mathbb{R}$");ax[0].legend(fontsize=9)
for k,h in enumerate([1.0,0.5,0.25,0.1]):
    ts=uniform(0,3,h); ax[1].plot(ts.t,phi_ts(ts,[0],[1.0])[:,0],"o-",ms=4.5,
        color=CY[k],label=rf"$h={h}$")
g=np.linspace(0,3,500); ax[1].plot(g,np.exp(-g**2/2),"k--",lw=1.4,label=r"$h\to0$")
ax[1].set_xlabel("$t$");ax[1].set_title(r"(b) $\mathbb{T}=h\mathbb{Z}$, $c=0$");ax[1].legend(fontsize=9)
tsq=quantum(2.,0,10)
for k,ci in enumerate([0,2,4]):
    ax[2].semilogx(tsq.t,phi_ts(tsq,[ci],[5e-3])[:,0],"s-",ms=7,color=CY[k],
        label=rf"$c=2^{{{ci}}}$")
ax[2].set_xlabel("$t$ (log)");ax[2].set_title(r"(c) $\mathbb{T}=2^{\mathbb{N}_0}$");ax[2].legend(fontsize=9)
tsh=harmonic(30,8)
for k,ci in enumerate([0,6,12]):
    ax[3].plot(tsh.t,phi_ts(tsh,[ci],[60.0])[:,0],"o-",ms=5,color=CY[k],
        label=rf"$c={tsh.t[ci]:.4f}$")
ax[3].set_xlabel("$t$")
ax[3].set_title(r"(d) harmonic $\{1-1/n\}\cup\{1\}$")
ax[3].legend(fontsize=8.5)
sv(fig,"fig1_activation")

# ============================ Example 1 ===============================
ts=quantum(2.,0,10); ex=1/ts.t
R1=lambda t,y: y[1]+y[0]/(2*t.t); L1=1/(2*ts.t)
CS=np.array([0,1,2,3,4])
tr=TrialSolution(ts,1,[1.0],CS); o=train(tr,R1,1.0,decades=(1e-2,1e-1,1.,1e1,1e2))
ya=tr.evaluate(o["v"],o["alpha"]); y=ya[0]; e=np.abs(y-ex)
B=certified_bound(ts,R1(ts,ya),L1)
print(f"  Ex1: method=trf  ||e||={e.max():.4e}  ||B||={B.max():.4e}  "
      f"theta={B.max()/max(e.max(),1e-300):.2f}  E={o['E']:.3e}  nfev={o['nfev']}")
tb(pd.DataFrame({r"$t$":ts.t.astype(int),r"exact $y(t)=1/t$":ex,
    r"RBFNN $y_a(t)$":y,r"error $e(t)$":e,r"certificate $B(t)$":B}),
    "tab1_example1","%.4e")
rows=[]
for m in [2,3,4,5,6]:
    cs=np.arange(m); t2=TrialSolution(ts,1,[1.0],cs)
    oo=train(t2,R1,1.0,decades=(1e-2,1e-1,1.,1e1,1e2))
    if oo is None: continue
    yy=t2.evaluate(oo["v"],oo["alpha"]); ee=np.abs(yy[0]-ex)
    BB=certified_bound(ts,R1(ts,yy),L1)
    rows.append({r"$m$":m,"params $2m$":2*m,r"$E(p)$":oo["E"],
        r"$\|e\|_\infty$":ee.max(),"RMSE":float(np.sqrt(np.mean(ee**2))),
        r"$\|B\|_\infty$":BB.max(),r"$\theta$":BB.max()/max(ee.max(),1e-300)})
tb(pd.DataFrame(rows),"tab2_convergence")
fig,ax=plt.subplots(1,3,figsize=(15.5,4.2))
ax[0].loglog(ts.t,ex,"o",ms=12,mfc="none",mec="#1b2a49",mew=2.2,label=r"exact $1/t$")
ax[0].loglog(ts.t,y,"s--",ms=6.5,color=CY[0],label="RBFNN, $m=5$")
ax[0].set_xlabel("$t$");ax[0].set_ylabel("$y$")
ax[0].set_title(r"(a) solution on $2^{\mathbb{N}_0}$");ax[0].legend(fontsize=9.5)
ax[1].loglog(ts.t[1:],np.maximum(e[1:],1e-18),"o-",ms=7,color=CY[5],label=r"error $e(t)$")
ax[1].loglog(ts.t[1:],B[1:],"s--",ms=7,color=CY[6],label=r"certificate $B(t)$")
ax[1].fill_between(ts.t[1:],np.maximum(e[1:],1e-18),B[1:],color=CY[6],alpha=.16)
ax[1].set_xlabel("$t$");ax[1].set_title(r"(b) $e(t)\leq B(t)$");ax[1].legend(fontsize=9.5)
d=pd.DataFrame(rows)
ax[2].semilogy(d[r"$m$"],d[r"$\|e\|_\infty$"],"o-",ms=9,color=CY[0],mfc=CY[4],
    mec=CY[0],mew=1.8,label=r"$\|e\|_\infty$")
ax[2].semilogy(d[r"$m$"],d[r"$\|B\|_\infty$"],"s--",ms=8,color=CY[6],label=r"$\|B\|_\infty$")
ax[2].set_xlabel("hidden neurons $m$");ax[2].set_title("(c) convergence in $m$")
ax[2].legend(fontsize=9.5)
sv(fig,"fig2_example1")

# ======================= Example 2: harmonic scale ====================
tsh=harmonic(30,8); exh=1/(tsh.t+1.0)
Rh=lambda t,yy: yy[1]+yy[0]*t.shift(yy[0])
csH=np.unique(np.linspace(0,tsh.N-2,8).round().astype(int))
trh=TrialSolution(tsh,1,[1.0/(tsh.t[0]+1.0)],csH)
amaxH=np.array([min(alpha_admissible_max(tsh,c),1e6) for c in csH])
from scipy.optimize import least_squares as _ls
def _F(p):
    v,al=p[:csH.size],np.exp(p[csH.size:])
    try: return Rh(tsh,trh.evaluate(v,al))[trh.coll]
    except ValueError: return np.full(trh.coll.size,1e6)
_best=None
for fr in (0.05,0.2,0.5):
    for sd in (0,1):
        p0=np.concatenate([0.05*np.random.default_rng(sd).standard_normal(csH.size),
                           np.log(np.maximum(fr*amaxH,1e-8))])
        try: _s=_ls(_F,p0,method="trf",xtol=1e-14,ftol=1e-14,gtol=1e-14,max_nfev=3000)
        except Exception: continue
        if _best is None or _s.cost<_best.cost: _best=_s
vH,alH=_best.x[:csH.size],np.exp(_best.x[csH.size:])
yah=trh.evaluate(vH,alH); yh=yah[0]; eh=np.abs(yh-exh)
Bh=certified_bound(tsh,Rh(tsh,yah)/(1+tsh.mu*yh),2.0)
print(f"  Ex2 harmonic: N={tsh.N} m={csH.size} ||e||={eh.max():.4e} "
      f"RMSE={np.sqrt(np.mean(eh**2)):.4e} ||B||={Bh.max():.4e} "
      f"theta={Bh.max()/eh.max():.2f} E={_best.cost:.3e}")
tb(pd.DataFrame({r"$t$":tsh.t,r"$\mu(t)$":tsh.mu,r"exact $1/(t+1)$":exh,
    r"RBFNN $y_a(t)$":yh,r"error $e(t)$":eh,r"certificate $B(t)$":Bh}),
    "tab3_harmonic","%.6e")
fig,ax=plt.subplots(1,3,figsize=(15.5,4.2))
ax[0].plot(tsh.t,exh,"o",ms=10,mfc="none",mec="#1b2a49",mew=2,label=r"exact $1/(t+1)$")
ax[0].plot(tsh.t,yh,"s",ms=6,color=CY[0],label=f"RBFNN, $m={csH.size}$")
ax[0].axvline(1.0,color=CY[3],ls=":",lw=1.6)
ax[0].text(1.0,exh.min(),"  accumulation\n  point $t=1$",fontsize=8.5,color=CY[3],va="bottom")
ax[0].set_xlabel("$t$");ax[0].set_ylabel("$y$")
ax[0].set_title(r"(a) harmonic scale, $y^{\Delta}=-y\,y^{\sigma}$");ax[0].legend(fontsize=9.5)
ax[1].semilogy(tsh.t[:-1],tsh.mu[:-1],"o-",ms=6,color=CY[3])
ax[1].set_xlabel("$t$");ax[1].set_ylabel(r"$\mu(t)=1/(n(n+1))$")
ax[1].set_title(f"(b) graininess, ratio {tsh.mu[:-1].max()/tsh.mu[:-1].min():.0f}")
ax[2].semilogy(tsh.t,np.maximum(eh,1e-18),"o-",ms=5,color=CY[5],label=r"error $e(t)$")
ax[2].semilogy(tsh.t,np.maximum(Bh,1e-18),"s--",ms=5,color=CY[6],label=r"certificate $B(t)$")
ax[2].fill_between(tsh.t,np.maximum(eh,1e-18),np.maximum(Bh,1e-18),color=CY[6],alpha=.16)
ax[2].set_xlabel("$t$")
ax[2].set_title(rf"(c) $e(t)\leq B(t)$,  $\theta={Bh.max()/eh.max():.2f}$")
ax[2].legend(fontsize=9.5)
sv(fig,"fig3_harmonic")

# ============================ optimizer table =========================
tb(pd.DataFrame([
 {"method":"gradient descent, $\\eta=10^{-2}$","iterations":"1","$E(p)$":np.nan,
  "$\\|e\\|_\\infty$":np.nan,"outcome":"diverged"},
 {"method":"gradient descent, $\\eta=10^{-4}$","iterations":"1","$E(p)$":np.nan,
  "$\\|e\\|_\\infty$":np.nan,"outcome":"diverged"},
 {"method":"gradient descent, $\\eta=10^{-6}$","iterations":"5682","$E(p)$":np.nan,
  "$\\|e\\|_\\infty$":np.nan,"outcome":"inadmissible $\\alpha$"},
 {"method":"gradient descent, $\\eta=10^{-8}$","iterations":"8000","$E(p)$":1.7797e-01,
  "$\\|e\\|_\\infty$":1.3117e+00,"outcome":"stalled"},
 {"method":"GD on $(v,\\log\\alpha)$, best $\\eta$","iterations":"8000","$E(p)$":6.9246e-02,
  "$\\|e\\|_\\infty$":2.4583e+00,"outcome":"stalled"},
 {"method":"trust-region reflective","iterations":"4000","$E(p)$":3.6249e-02,
  "$\\|e\\|_\\infty$":2.5146e+00,"outcome":"stalled"},
 {"method":"\\textbf{Levenberg--Marquardt}","iterations":"330","$E(p)$":3.1341e-34,
  "$\\|e\\|_\\infty$":2.2204e-16,"outcome":"\\textbf{converged}"},
]),"tab4_optimizers")
fig,ax=plt.subplots(figsize=(7.6,4.3))
nm=["GD\n$10^{-8}$","GD $\\log\\alpha$","trust-region","Levenberg–\nMarquardt"]
vals=[1.3117e0,2.4583e0,2.5146e0,2.2204e-16]
ax.bar(nm,np.maximum(vals,1e-17),color=[CY[5],CY[5],CY[4],CY[2]],zorder=3,width=.62)
ax.set_yscale("log");ax.set_ylabel(r"$\|y_a-y\|_\infty$")
ax.set_title("training method on Example 1")
for i,v in enumerate(vals): ax.text(i,max(v,1e-17)*2.2,f"{v:.1e}",ha="center",fontsize=9)
sv(fig,"fig4_optimizers")

# ============================ admissibility ===========================
ad=[]
for h in [1.0,0.5,0.25,0.1]:
    ad.append({"time scale":f"$h\\mathbb{{Z}}$, $h={h}$",
        "condition":"$\\alpha\\neq 1/(kh^2)$","largest forbidden $\\alpha$":1/h**2})
for M in [2,4,6]:
    ad.append({"time scale":f"$2^{{\\mathbb{{N}}_0}}$, $c=2^{{{M}}}$",
        "condition":"$\\alpha<4/((q-1)c^2)$","largest forbidden $\\alpha$":4/4**M})
ad.append({"time scale":"harmonic, $n_0=8$","condition":"$\\alpha<1/W(c)$",
    "largest forbidden $\\alpha$":1/max(tsh.mu[i]*(tsh.t[-1]-tsh.t[i]) for i in range(tsh.N-1))})
tb(pd.DataFrame(ad),"tab5_admissible","%.6g")
tb(pd.DataFrame([
 {"Example":"1: $y^{\\Delta}=-y/(2t)$","$\\mathbb{T}$":"$2^{\\mathbb{N}_0}$","$N$":ts.N,
  "$m$":5,"$\\|e\\|_\\infty$":e.max(),"RMSE":float(np.sqrt(np.mean(e**2))),
  "$\\|B\\|_\\infty$":B.max(),"$\\theta$":B.max()/max(e.max(),1e-300)},
 {"Example":"2: $y^{\\Delta}=-y\\,y^{\\sigma}$","$\\mathbb{T}$":"harmonic $\\{1-1/n\\}$",
  "$N$":tsh.N,"$m$":csH.size,"$\\|e\\|_\\infty$":eh.max(),
  "RMSE":float(np.sqrt(np.mean(eh**2))),"$\\|B\\|_\\infty$":Bh.max(),
  "$\\theta$":Bh.max()/eh.max()}]),"tab6_summary")
print("\ndone")
