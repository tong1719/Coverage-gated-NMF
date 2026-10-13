"""Locked C530 confirmatory complementary-bin line-spectrum NMF."""
from __future__ import annotations
import importlib.util,itertools,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; OUT=HERE/"results"
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
c528=load("c528_c530",ROOT/"screening"/"c528_complementary_partition_operator_nmf"/"run_synthetic_screen.py")
c529=load("c529_c530",ROOT/"screening"/"c529_partition_intersection_line_nmf"/"run_synthetic_screen.py")
EPS=c528.EPS; REGIMES=c529.REGIMES; SEEDS=tuple(range(53000,53008))
METHODS=("uniform_backproject","separate_arithmetic","intersection_fixed","wrong_operator_multistart","single_correct_operator","c530_multistart_operator","true_w_operator_ceiling","fine_count_oracle")

def deviance(ys,mats,fit):
 ws,h=fit; total=0.0
 for y,a,w in zip(ys,mats,ws):
  mu=np.maximum(w@h@a,EPS); total+=float(np.sum(mu-y*np.log(mu)))
 return total

def align_similarity(h,ref):
 hn=h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),EPS); rn=ref/np.maximum(np.linalg.norm(ref,axis=1,keepdims=True),EPS); rr,cc=linear_sum_assignment(-(hn@rn.T)); return float(np.mean((hn@rn.T)[rr,cc]))

def multistart(ys,mats,seed,bp,hi,wi,n_random=4):
 fits=[c528.fit_operator(ys,mats,seed,h0=bp[1],w0=bp[0]),c528.fit_operator(ys,mats,seed+1,h0=hi,w0=wi)]
 fits.extend(c528.fit_operator(ys,mats,seed+2+i) for i in range(n_random))
 losses=[deviance(ys,mats,x) for x in fits]; best=fits[int(np.argmin(losses))]; stability=float(np.mean([align_similarity(x[1],best[1]) for x in fits])); return best,stability

def run_case(regime,seed):
 ys,mats,fines,truews,trueh,yt,at,wt,rt=c529.simulate(regime,seed); fits={}; runt={}
 t=time.perf_counter(); bp=c528.backproject_fit(ys,mats,seed+1); fits["uniform_backproject"]=bp; runt["uniform_backproject"]=time.perf_counter()-t
 t=time.perf_counter(); bases=c529.aligned_backprojected_bases(ys,mats,seed+10); ha=np.mean(bases,axis=0); fits["separate_arithmetic"]=(c529.fit_w_all(ys,mats,ha),ha); runt["separate_arithmetic"]=time.perf_counter()-t
 hi=c529.intersection_basis(bases); wi=c529.fit_w_all(ys,mats,hi); fits["intersection_fixed"]=(wi,hi); runt["intersection_fixed"]=0.0
 t=time.perf_counter(); wrong,wrong_stab=multistart(ys,[mats[0]]*3,seed+20,bp,hi,wi,n_random=2); fits["wrong_operator_multistart"]=wrong; runt["wrong_operator_multistart"]=time.perf_counter()-t
 t=time.perf_counter(); single=c528.fit_operator(ys,mats,seed+30); fits["single_correct_operator"]=single; runt["single_correct_operator"]=time.perf_counter()-t
 t=time.perf_counter(); candidate,stability=multistart(ys,mats,seed+40,bp,hi,wi,n_random=4); fits["c530_multistart_operator"]=candidate; runt["c530_multistart_operator"]=time.perf_counter()-t
 t=time.perf_counter(); fits["true_w_operator_ceiling"]=c528.true_w_fit(ys,mats,truews); runt["true_w_operator_ceiling"]=time.perf_counter()-t
 t=time.perf_counter(); ow,oh=c528.nmf(np.vstack(fines),seed+60); cuts=np.cumsum([len(x) for x in fines])[:-1]; fits["fine_count_oracle"]=(list(np.split(ow,cuts)),oh); runt["fine_count_oracle"]=time.perf_counter()-t
 rows=[]
 for m in METHODS:
  ws,h=fits[m]; truth=np.vstack([.015+w@trueh for w in truews]); est=np.vstack([w@h for w in ws]); rows.append({"regime":regime,"seed":seed,"method":m,"runtime":runt[m],"basis_cosine":c528.align_basis(h,trueh),"fine_nmse":float(np.sum((est-truth)**2)/max(np.sum(truth**2),EPS)),"heldout_nmse":c528.heldout(h,yt,at,rt),"start_stability":stability})
 return rows

def summarize(rows,total):
 p="c530_multistart_operator"; sep="separate_arithmetic"; wrong="wrong_operator_multistart"; single="single_correct_operator"; oracle="fine_count_oracle"; intended=REGIMES[:-1]; nonops=("uniform_backproject","separate_arithmetic","intersection_fixed","wrong_operator_multistart")
 def sub(m,regs=None): return [r for r in rows if r["method"]==m and(regs is None or r["regime"] in regs)]
 def avg(m,x,regs=None): return float(np.mean([r[x] for r in sub(m,regs)]))
 pc,ph=avg(p,"basis_cosine",intended),avg(p,"heldout_nmse",intended); sc,sh=avg(sep,"basis_cosine",intended),avg(sep,"heldout_nmse",intended); wc,wh=avg(wrong,"basis_cosine",intended),avg(wrong,"heldout_nmse",intended); oneh=avg(single,"heldout_nmse",intended); oh=avg(oracle,"heldout_nmse",intended); strongest=min(avg(m,"heldout_nmse",intended) for m in nonops)
 wins=sum(next(r for r in sub(p,[g]) if r["seed"]==s)["heldout_nmse"]<min(next(r for r in sub(m,[g]) if r["seed"]==s)["heldout_nmse"] for m in nonops) for g,s in itertools.product(intended,SEEDS)); gains={g:float(min(avg(m,"heldout_nmse",[g]) for m in nonops)-avg(p,"heldout_nmse",[g])) for g in intended}; stab=avg(p,"start_stability",intended); nullratio=avg(p,"heldout_nmse",[REGIMES[-1]])/max(avg(single,"heldout_nmse",[REGIMES[-1]]),EPS); runtime=avg(p,"runtime")/max(avg("uniform_backproject","runtime"),EPS); finite=all(np.isfinite(r[x]) for r in rows for x in("basis_cosine","fine_nmse","heldout_nmse","runtime","start_stability"))
 gates={"g1_basis":pc>=.99,"g2_heldout":ph<=.03,"g3_nonoperator_gain":ph<=.30*strongest,"g4_wins":wins>=36,"g5_regimes":all(v>0 for v in gains.values()),"g6_operator_specificity":pc>max(sc,wc) and ph<min(sh,wh),"g7_oracle":ph<=1.25*oh,"g8_stability_single":stab>=.98 and ph<=1.02*oneh,"g9_broad_null":nullratio<=1.10,"g10_runtime_integrity":runtime<=20 and finite and len(rows)==384 and len({(r["regime"],r["seed"],r["method"]) for r in rows})==384}
 methods={m:{x:avg(m,x,intended) for x in("basis_cosine","fine_nmse","heldout_nmse","runtime")} for m in METHODS}
 return {"summary":{"basis_cosine":pc,"fine_nmse":avg(p,"fine_nmse",intended),"heldout_nmse":ph,"separate_basis":sc,"separate_heldout":sh,"wrong_basis":wc,"wrong_heldout":wh,"single_heldout":oneh,"strongest_nonoperator":strongest,"oracle_heldout":oh,"cell_wins":wins,"regime_gain":gains,"start_stability":stab,"broad_null_ratio":nullratio,"runtime_ratio":runtime,"total_runtime":total},"method_summary":methods,"gates":gates,"passed":int(sum(gates.values())),"total":10,"decision":"ADVANCE" if all(gates.values()) else "STOP","n_records":len(rows)}

def main():
 t=time.perf_counter(); rows=[]
 for g,s in itertools.product(REGIMES,SEEDS): rows.extend(run_case(g,s))
 rep=summarize(rows,time.perf_counter()-t); OUT.mkdir(parents=True,exist_ok=True); (OUT/"records.json").write_text(json.dumps(rows,indent=2)); (OUT/"report.json").write_text(json.dumps(rep,indent=2)); print(json.dumps(rep,indent=2))
if __name__=="__main__": main()
