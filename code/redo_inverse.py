import sys
import json, numpy as np, pandas as pd, sys, warnings; sys.path.insert(0,'/home/claude/work/build')
warnings.filterwarnings('ignore')
from pathlib import Path
from scipy.optimize import least_squares
from solver import IVP, SEED0, TOL, TRF
from tsrbf import phi_and_dlogphi
OUT=Path(__file__).resolve().parents[1] / 'data'
tl=np.array([0,1,2,4,7,11,16,22,29,37,46,56],float); mul=np.diff(tl)
r_true,K,z0=0.03,1000.0,0.08; T=tl[-1]-tl[0]
def ref(r):
    z=np.empty(tl.size); z[0]=z0
    for k,h in enumerate(mul): z[k+1]=z[k]+h*r*z[k]*(1-z[k])
    return z
zl=ref(r_true); dfl=lambda t,z: r_true*(1-2*z)
m,c=6,np.round(np.linspace(0,tl.size-1,6)).astype(int)
def rbf_fit(obs,seed,wR=T):
    p=IVP(tl,z0,lambda t,z:r_true*z*(1-z),dfl,m,c,'ts')
    x0=p.init_x(seed); r0=0.01+0.04*((seed%5)/4.0)
    def joint(u):
        v,th,rr=u[:m],u[m:2*m],u[-1]
        Phi,_=phi_and_dlogphi(tl,c,np.exp(th)); y=p.trial(Phi,v)
        return np.r_[wR*((y[1:]-y[:-1])/mul - rr*y[:-1]*(1-y[:-1])), y-obs]
    lb=np.r_[np.full(m,-np.inf),np.log(p.lo),1e-4]; ub=np.r_[np.full(m,np.inf),np.log(p.hi),1.0]
    s=least_squares(joint,np.r_[x0,r0],bounds=(lb,ub),max_nfev=8000,**TOL,**TRF)
    Phi,_=phi_and_dlogphi(tl,c,np.exp(s.x[m:2*m])); y=p.trial(Phi,s.x[:m])
    return float(s.x[-1]), r0, float(K*np.max(np.abs(y-zl)))
def rec_fit(obs,r0=0.02):
    return float(least_squares(lambda u: ref(u[0])-obs,[r0],bounds=([1e-4],[1.0]),**TOL).x[0])

rng=np.random.default_rng(SEED0); obs=zl+rng.normal(0,2.0/K,zl.size)
rows=[]
for s in range(15):
    rh,r0,tm=rbf_fit(obs,SEED0+s)
    rows.append(dict(start=s+1,r0=r0,r_hat=rh,rel_error_r=abs(rh-r_true)/r_true,
                     traj_max_error_population=tm))
idf=pd.DataFrame(rows); idf.to_csv(OUT/'inverse_identification.csv',index=False)
rec_single=rec_fit(obs)

comp=[]
for n in range(15):
    rg=np.random.default_rng(1000+n); ob=zl+rg.normal(0,2.0/K,zl.size)
    rr=rec_fit(ob); bb=float(np.median([rbf_fit(ob,SEED0+s)[0] for s in range(2)]))
    comp.append(dict(realisation=n+1,r_recurrence=rr,r_rbfnn=bb,
                     rel_err_recurrence=abs(rr-r_true)/r_true,
                     rel_err_rbfnn=abs(bb-r_true)/r_true))
cdf=pd.DataFrame(comp); cdf.to_csv(OUT/'inverse_comparison.csv',index=False)
out=dict(weight_rule='residual scaled by observation horizon T=56 days',
         r_true=r_true, noise_individuals=2.0,
         rbfnn_r_median=float(idf.r_hat.median()),
         rbfnn_r_iqr=float(idf.r_hat.quantile(.75)-idf.r_hat.quantile(.25)),
         rbfnn_r_range=float(idf.r_hat.max()-idf.r_hat.min()),
         rbfnn_rel_error=float(idf.rel_error_r.median()),
         recurrence_r_single=rec_single,
         recurrence_rel_error_single=abs(rec_single-r_true)/r_true,
         median_rel_err_recurrence=float(cdf.rel_err_recurrence.median()),
         median_rel_err_rbfnn=float(cdf.rel_err_rbfnn.median()),
         rbfnn_better_fraction=float((cdf.rel_err_rbfnn<cdf.rel_err_recurrence).mean()),
         median_abs_gap=float((cdf.rel_err_rbfnn-cdf.rel_err_recurrence).abs().median()))
sc=json.loads((OUT/'stageC.json').read_text()); sc['inverse']=out
(OUT/'stageC.json').write_text(json.dumps(sc,indent=2))
print(json.dumps(out,indent=2))
