"""Locked C529 partition-intersection line-spectrum NMF screen."""
from __future__ import annotations
import importlib.util,itertools,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; OUT=HERE/"results"
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
c528=load("c528_c529",ROOT/"screening"/"c528_complementary_partition_operator_nmf"/"run_synthetic_screen.py")
EPS=c528.EPS; P=c528.P; K=c528.K
REGIMES=("nominal","low_count","overlap_lines","weak_component","imbalanced","broad_spectrum_null"); SEEDS=tuple(range(52900,52904))
METHODS=("uniform_backproject","separate_arithmetic","c528_arithmetic_operator","random_correct_operator","intersection_fixed","c529_intersection_operator","true_w_operator_ceiling","fine_count_oracle")

def line_basis(regime):
 positions=[[5,14,27,43,61,78],[9,22,35,50,69,88],[3,18,39,55,72,91],[11,30,47,64,81,94]]
 if regime=="overlap_lines": positions[1][2]=positions[0][2]; positions[2][4]=positions[3][4]
 h=np.full((K,P),.003); weights=np.array([1.0,.72,.55,.90,.64,.82])
 for k,pos in enumerate(positions):
  for q,(j,v) in enumerate(zip(pos,weights)): h[k,j]+=v*(1+.08*k+.03*q)
 if regime=="broad_spectrum_null":
  grid=np.arange(P); broad=[]
  for k,c in enumerate([15,36,59,80]): broad.append(np.exp(-.5*((grid-c)/5.5)**2)+.01)
  h=np.array(broad)
 h/=np.maximum(h.mean(1,keepdims=True),EPS); return h

def simulate(regime,seed):
 rng=np.random.default_rng(seed); h=line_basis(regime); ns=[72,72,72] if regime!="imbalanced" else [128,64,24]; mats=[c528.partition(3,o) for o in (0,1,2)]; scale=.50 if regime!="low_count" else .20; ys=[]; fines=[]; ws=[]
 for n,a in zip(ns,mats):
  w=rng.gamma(1.7,.7,(n,K)); w/=np.maximum(w.mean(0,keepdims=True),EPS); w*=scale
  if regime=="weak_component": w[:,3]*=.20
  rate=.015+w@h; fine=rng.poisson(rate).astype(float); ys.append(fine@a); fines.append(fine); ws.append(w)
 wt=rng.gamma(1.7,.7,(52,K)); wt/=np.maximum(wt.mean(0,keepdims=True),EPS); wt*=scale
 if regime=="weak_component": wt[:,3]*=.20
 rt=.015+wt@h; at=c528.partition(4,1); yt=rng.poisson(rt).astype(float)@at
 return ys,mats,fines,ws,h,yt,at,wt,rt

def aligned_backprojected_bases(ys,mats,seed):
 bases=[]
 for c,(y,a) in enumerate(zip(ys,mats)):
  _,hc=c528.nmf(y,seed+c); bases.append(hc@a.T/np.maximum(a.sum(0)[None,:]@a.T,EPS))
 ref=bases[0]/np.maximum(np.linalg.norm(bases[0],axis=1,keepdims=True),EPS); out=[]
 for h in bases:
  hn=h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),EPS); rr,cc=linear_sum_assignment(-(ref@hn.T)); out.append(h[cc[np.argsort(rr)]])
 return out

def intersection_basis(bases):
 normalized=[h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),EPS) for h in bases]
 h=np.exp(np.mean([np.log(np.maximum(x,EPS)) for x in normalized],axis=0)); h/=np.maximum(h.mean(1,keepdims=True),EPS); return h

def fit_w_all(ys,mats,h): return [c528.fixed_w(y,h@a,np.ones((len(y),K)),300) for y,a in zip(ys,mats)]

def run_case(regime,seed):
 ys,mats,fines,truews,trueh,yt,at,wt,rt=simulate(regime,seed); fits={}; runt={}
 t=time.perf_counter(); bp=c528.backproject_fit(ys,mats,seed+1); fits["uniform_backproject"]=bp; runt["uniform_backproject"]=time.perf_counter()-t
 t=time.perf_counter(); bases=aligned_backprojected_bases(ys,mats,seed+10); ha=np.mean(bases,axis=0); fits["separate_arithmetic"]=(fit_w_all(ys,mats,ha),ha); runt["separate_arithmetic"]=time.perf_counter()-t
 hi=intersection_basis(bases); wi=fit_w_all(ys,mats,hi)
 t=time.perf_counter(); fits["c528_arithmetic_operator"]=c528.fit_operator(ys,mats,seed+20,h0=bp[1],w0=bp[0]); runt["c528_arithmetic_operator"]=time.perf_counter()-t
 t=time.perf_counter(); fits["random_correct_operator"]=c528.fit_operator(ys,mats,seed+30); runt["random_correct_operator"]=time.perf_counter()-t
 t=time.perf_counter(); fits["intersection_fixed"]=(wi,hi); runt["intersection_fixed"]=time.perf_counter()-t
 t=time.perf_counter(); fits["c529_intersection_operator"]=c528.fit_operator(ys,mats,seed+40,h0=hi,w0=wi); runt["c529_intersection_operator"]=time.perf_counter()-t
 t=time.perf_counter(); fits["true_w_operator_ceiling"]=c528.true_w_fit(ys,mats,truews); runt["true_w_operator_ceiling"]=time.perf_counter()-t
 t=time.perf_counter(); ow,oh=c528.nmf(np.vstack(fines),seed+50); cuts=np.cumsum([len(x) for x in fines])[:-1]; fits["fine_count_oracle"]=(list(np.split(ow,cuts)),oh); runt["fine_count_oracle"]=time.perf_counter()-t
 rows=[]
 for m in METHODS:
  ws,h=fits[m]; truth=np.vstack([.015+w@trueh for w in truews]); est=np.vstack([w@h for w in ws]); rows.append({"regime":regime,"seed":seed,"method":m,"runtime":runt[m],"basis_cosine":c528.align_basis(h,trueh),"fine_nmse":float(np.sum((est-truth)**2)/max(np.sum(truth**2),EPS)),"heldout_nmse":c528.heldout(h,yt,at,rt)})
 return rows

def summarize(rows,total):
 p="c529_intersection_operator"; direct="c528_arithmetic_operator"; bp="uniform_backproject"; fixed="intersection_fixed"; oracle="fine_count_oracle"; intended=REGIMES[:-1]
 def sub(m,regs=None): return [r for r in rows if r["method"]==m and(regs is None or r["regime"] in regs)]
 def avg(m,x,regs=None): return float(np.mean([r[x] for r in sub(m,regs)]))
 pc,ph=avg(p,"basis_cosine",intended),avg(p,"heldout_nmse",intended); dc,dh=avg(direct,"basis_cosine",intended),avg(direct,"heldout_nmse",intended); bc,bh=avg(bp,"basis_cosine",intended),avg(bp,"heldout_nmse",intended); fc,fh=avg(fixed,"basis_cosine",intended),avg(fixed,"heldout_nmse",intended); oh=avg(oracle,"heldout_nmse",intended)
 nonoracle=[m for m in METHODS if m not in(p,"true_w_operator_ceiling","fine_count_oracle")]; wins=sum(next(r for r in sub(p,[g]) if r["seed"]==s)["heldout_nmse"]<min(next(r for r in sub(m,[g]) if r["seed"]==s)["heldout_nmse"] for m in nonoracle) for g,s in itertools.product(intended,SEEDS)); gains={g:float(np.mean([a["heldout_nmse"]-b["heldout_nmse"] for b,a in zip(sub(p,[g]),sub(direct,[g]))])) for g in intended}; nullratio=avg(p,"heldout_nmse",[REGIMES[-1]])/max(avg(direct,"heldout_nmse",[REGIMES[-1]]),EPS); runtime=avg(p,"runtime")/max(avg(bp,"runtime"),EPS); finite=all(np.isfinite(r[x]) for r in rows for x in("basis_cosine","fine_nmse","heldout_nmse","runtime"))
 gates={"g1_basis":pc>=.97,"g2_heldout":ph<=.10,"g3_direct_gain":ph<=.80*dh,"g4_backproject_gain":ph<=.70*bh,"g5_wins":wins>=16,"g6_regimes":all(v>0 for v in gains.values()),"g7_interaction":pc>max(dc,fc) and ph<min(dh,fh),"g8_oracle":ph<=2*oh,"g9_broad_null":nullratio<=1.10,"g10_runtime_integrity":runtime<=5 and finite and len(rows)==192 and len({(r["regime"],r["seed"],r["method"]) for r in rows})==192}
 methods={m:{x:avg(m,x,intended) for x in("basis_cosine","fine_nmse","heldout_nmse","runtime")} for m in METHODS}
 return {"summary":{"basis_cosine":pc,"fine_nmse":avg(p,"fine_nmse",intended),"heldout_nmse":ph,"direct_basis":dc,"direct_heldout":dh,"backproject_basis":bc,"backproject_heldout":bh,"fixed_basis":fc,"fixed_heldout":fh,"oracle_heldout":oh,"cell_wins":wins,"regime_gain":gains,"broad_null_ratio":nullratio,"runtime_ratio":runtime,"total_runtime":total},"method_summary":methods,"gates":gates,"passed":int(sum(gates.values())),"total":10,"decision":"ADVANCE" if all(gates.values()) else "STOP","n_records":len(rows)}

def main():
 t=time.perf_counter(); rows=[]
 for g,s in itertools.product(REGIMES,SEEDS): rows.extend(run_case(g,s))
 rep=summarize(rows,time.perf_counter()-t); OUT.mkdir(parents=True,exist_ok=True); (OUT/"records.json").write_text(json.dumps(rows,indent=2)); (OUT/"report.json").write_text(json.dumps(rep,indent=2)); print(json.dumps(rep,indent=2))
if __name__=="__main__": main()
