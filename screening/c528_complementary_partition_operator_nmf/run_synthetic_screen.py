"""Locked C528 complementary-partition operator NMF screen."""
from __future__ import annotations
import itertools, json, time, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

HERE=Path(__file__).resolve().parent; OUT=HERE/"results"; EPS=1e-12
P=96; K=4; REGIMES=("nominal","low_count","overlap","narrow_doublet","imbalanced","same_partition_null"); SEEDS=tuple(range(52800,52804))
METHODS=("uniform_backproject","separate_align","wrong_single_operator","random_correct_operator","c528_complementary_operator","true_w_operator_ceiling","fine_count_oracle")

def partition(width,offset):
 b=P//width; a=np.zeros((P,b))
 for j in range(P): a[j,((j-offset)%P)//width]=1.0
 return a

def simulate(regime,seed):
 rng=np.random.default_rng(seed); grid=np.arange(P)/P; centers=[.16,.38,.62,.83]
 if regime=="overlap": centers=[.28,.40,.52,.64]
 sigma=.022 if regime!="narrow_doublet" else .010
 h=[]
 for k,c in enumerate(centers):
  row=np.exp(-.5*((grid-c)/sigma)**2)+.55*np.exp(-.5*((grid-(c+.035))/(.8*sigma))**2)+.01
  h.append(row)
 h=np.array(h); h/=np.maximum(h.mean(1,keepdims=True),EPS)
 ns=[64,64,64] if regime!="imbalanced" else [112,56,24]
 offsets=[0,1,2] if regime!="same_partition_null" else [0,0,0]
 mats=[partition(3,o) for o in offsets]; ys=[]; fines=[]; ws=[]; scale=.55 if regime!="low_count" else .22
 for n,a in zip(ns,mats):
  w=rng.gamma(1.6,.75,(n,K)); w/=np.maximum(w.mean(0,keepdims=True),EPS); w*=scale
  rate=.02+w@h; fine=rng.poisson(rate).astype(float); ys.append(fine@a); fines.append(fine); ws.append(w)
 wt=rng.gamma(1.6,.75,(48,K)); wt/=np.maximum(wt.mean(0,keepdims=True),EPS); wt*=scale; rt=.02+wt@h; at=partition(4,1); yt=rng.poisson(rt).astype(float)@at
 return ys,mats,fines,ws,h,yt,at,wt,rt

def nmf(x,seed,iters=500):
 model=NMF(n_components=K,init="nndsvda",solver="mu",beta_loss="kullback-leibler",max_iter=iters,tol=1e-5,random_state=seed)
 with warnings.catch_warnings(): warnings.simplefilter("ignore",ConvergenceWarning); w=model.fit_transform(np.maximum(x,EPS))
 return np.maximum(w,EPS),np.maximum(model.components_,EPS)

def fixed_w(y,b,w,iters=250):
 den=np.ones_like(y)@b.T
 for _ in range(iters): w*=((y/np.maximum(w@b,EPS))@b.T)/np.maximum(den,EPS); w=np.maximum(w,EPS)
 return w

def fit_operator(ys,mats,seed,h0=None,w0=None,iters=420):
 rng=np.random.default_rng(seed); h=np.maximum(h0.copy() if h0 is not None else rng.gamma(1,1,(K,P)),EPS)
 ws=[np.maximum(x.copy(),EPS) for x in w0] if w0 is not None else [rng.gamma(1,1,(len(y),K)) for y in ys]
 for step in range(iters):
  for c,(y,a) in enumerate(zip(ys,mats)):
   b=h@a; ws[c]=fixed_w(y,b,ws[c],iters=1)
  num=np.zeros_like(h); den=np.zeros_like(h)
  for y,a,w in zip(ys,mats,ws):
   pred=np.maximum(w@h@a,EPS); num+=w.T@(y/pred)@a.T; den+=w.T@np.ones_like(y)@a.T
  h*=num/np.maximum(den,EPS); h=np.maximum(h,EPS)
  if step%25==24:
   scale=np.maximum(h.mean(1),EPS); h/=scale[:,None]; ws=[w*scale[None,:] for w in ws]
 return ws,h

def backproject_fit(ys,mats,seed):
 xs=[]
 for y,a in zip(ys,mats): xs.append(y@a.T/np.maximum(a.sum(0)[None,:]@a.T,EPS))
 allx=np.vstack(xs); w,h=nmf(allx,seed); cuts=np.cumsum([len(x) for x in xs])[:-1]; return list(np.split(w,cuts)),h

def separate_fit(ys,mats,seed):
 bases=[]
 for c,(y,a) in enumerate(zip(ys,mats)):
  _,hc=nmf(y,seed+c); bases.append(hc@a.T/np.maximum(a.sum(0)[None,:]@a.T,EPS))
 ref=bases[0]/np.maximum(np.linalg.norm(bases[0],axis=1,keepdims=True),EPS); aligned=[]
 for h in bases:
  hn=h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),EPS); rr,cc=linear_sum_assignment(-(ref@hn.T)); aligned.append(h[cc[np.argsort(rr)]])
 h=np.mean(aligned,axis=0); ws=[fixed_w(y,h@a,np.ones((len(y),K)),250) for y,a in zip(ys,mats)]; return ws,h

def true_w_fit(ys,mats,ws,iters=500):
 h=np.ones((K,P))
 for _ in range(iters):
  num=np.zeros_like(h); den=np.zeros_like(h)
  for y,a,w in zip(ys,mats,ws): pred=np.maximum(w@h@a,EPS); num+=w.T@(y/pred)@a.T; den+=w.T@np.ones_like(y)@a.T
  h*=num/np.maximum(den,EPS); h=np.maximum(h,EPS)
 return ws,h

def align_basis(h,trueh):
 hn=h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),EPS); tn=trueh/np.maximum(np.linalg.norm(trueh,axis=1,keepdims=True),EPS); rr,cc=linear_sum_assignment(-(hn@tn.T)); return float(np.mean((hn@tn.T)[rr,cc]))

def heldout(h,yt,at,rt):
 w=fixed_w(yt,h@at,np.ones((len(yt),K)),400); est=w@h; return float(np.sum((est-rt)**2)/max(np.sum(rt**2),EPS))

def run_case(regime,seed):
 ys,mats,fines,truews,trueh,yt,at,wt,rt=simulate(regime,seed); fits={}; runt={}
 t=time.perf_counter(); bp=backproject_fit(ys,mats,seed+1); fits["uniform_backproject"]=bp; runt["uniform_backproject"]=time.perf_counter()-t
 t=time.perf_counter(); fits["separate_align"]=separate_fit(ys,mats,seed+10); runt["separate_align"]=time.perf_counter()-t
 t=time.perf_counter(); fits["wrong_single_operator"]=fit_operator(ys,[mats[0]]*3,seed+20,h0=bp[1],w0=bp[0]); runt["wrong_single_operator"]=time.perf_counter()-t
 t=time.perf_counter(); fits["random_correct_operator"]=fit_operator(ys,mats,seed+30); runt["random_correct_operator"]=time.perf_counter()-t
 t=time.perf_counter(); fits["c528_complementary_operator"]=fit_operator(ys,mats,seed+40,h0=bp[1],w0=bp[0]); runt["c528_complementary_operator"]=time.perf_counter()-t
 t=time.perf_counter(); fits["true_w_operator_ceiling"]=true_w_fit(ys,mats,truews); runt["true_w_operator_ceiling"]=time.perf_counter()-t
 t=time.perf_counter(); allfine=np.vstack(fines); ow,oh=nmf(allfine,seed+50); cuts=np.cumsum([len(x) for x in fines])[:-1]; fits["fine_count_oracle"]=(list(np.split(ow,cuts)),oh); runt["fine_count_oracle"]=time.perf_counter()-t
 rows=[]
 for m in METHODS:
  ws,h=fits[m]; fine_true=np.vstack([.02+w@trueh for w in truews]); fine_est=np.vstack([w@h for w in ws]); row={"regime":regime,"seed":seed,"method":m,"runtime":runt[m],"basis_cosine":align_basis(h,trueh),"fine_nmse":float(np.sum((fine_est-fine_true)**2)/max(np.sum(fine_true**2),EPS)),"heldout_nmse":heldout(h,yt,at,rt)}; rows.append(row)
 return rows

def summarize(rows,total):
 p="c528_complementary_operator"; bp="uniform_backproject"; sp="separate_align"; wr="wrong_single_operator"; oracle="fine_count_oracle"; intended=REGIMES[:-1]
 def sub(m,regs=None): return [r for r in rows if r["method"]==m and(regs is None or r["regime"] in regs)]
 def avg(m,x,regs=None): return float(np.mean([r[x] for r in sub(m,regs)]))
 pc,ph=avg(p,"basis_cosine",intended),avg(p,"heldout_nmse",intended); bc,bh=avg(bp,"basis_cosine",intended),avg(bp,"heldout_nmse",intended); sc,sh=avg(sp,"basis_cosine",intended),avg(sp,"heldout_nmse",intended); wc=avg(wr,"basis_cosine",intended); oh=avg(oracle,"heldout_nmse",intended)
 wins=sum(next(r for r in sub(p,[g]) if r["seed"]==s)["heldout_nmse"]<min(next(r for r in sub(bp,[g]) if r["seed"]==s)["heldout_nmse"],next(r for r in sub(sp,[g]) if r["seed"]==s)["heldout_nmse"],next(r for r in sub(wr,[g]) if r["seed"]==s)["heldout_nmse"]) for g,s in itertools.product(intended,SEEDS))
 gains={g:float(np.mean([a["heldout_nmse"]-b["heldout_nmse"] for b,a in zip(sub(p,[g]),sub(bp,[g]))])) for g in intended}; null_ratio=avg(p,"heldout_nmse",["same_partition_null"])/max(avg(bp,"heldout_nmse",["same_partition_null"]),EPS); runtime=avg(p,"runtime")/max(avg(bp,"runtime"),EPS); finite=all(np.isfinite(r[x]) for r in rows for x in("basis_cosine","fine_nmse","heldout_nmse","runtime"))
 gates={"g1_basis":pc>=.95,"g2_heldout":ph<=.15,"g3_backproject_gain":ph<=.75*bh,"g4_basis_gain":(1-pc)<=.75*(1-sc) and (1-pc)<=.75*(1-wc),"g5_wins":wins>=16,"g6_regimes":all(v>0 for v in gains.values()),"g7_constituents":pc>max(bc,sc) and ph<min(bh,sh),"g8_oracle":ph<=2*oh,"g9_null_safety":null_ratio<=1.10,"g10_runtime_integrity":runtime<=5 and finite and len(rows)==168 and len({(r["regime"],r["seed"],r["method"]) for r in rows})==168}
 methods={m:{x:avg(m,x,intended) for x in("basis_cosine","fine_nmse","heldout_nmse","runtime")} for m in METHODS}
 return {"summary":{"basis_cosine":pc,"fine_nmse":avg(p,"fine_nmse",intended),"heldout_nmse":ph,"backproject_basis":bc,"backproject_heldout":bh,"separate_basis":sc,"separate_heldout":sh,"wrong_basis":wc,"oracle_heldout":oh,"cell_wins":wins,"regime_gain":gains,"null_ratio":null_ratio,"runtime_ratio":runtime,"total_runtime":total},"method_summary":methods,"gates":gates,"passed":int(sum(gates.values())),"total":10,"decision":"ADVANCE" if all(gates.values()) else "STOP","n_records":len(rows)}

def main():
 t=time.perf_counter(); rows=[]
 for g,s in itertools.product(REGIMES,SEEDS): rows.extend(run_case(g,s))
 rep=summarize(rows,time.perf_counter()-t); OUT.mkdir(parents=True,exist_ok=True); (OUT/"records.json").write_text(json.dumps(rows,indent=2)); (OUT/"report.json").write_text(json.dumps(rep,indent=2)); print(json.dumps(rep,indent=2))
if __name__=="__main__": main()
