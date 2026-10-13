"""Locked C530 Phase-B coverage certificate and certified-design audit."""
from __future__ import annotations
import importlib.util,itertools,json,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; OUT=HERE/"results_coverage"
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
c528=load("c528_cov",ROOT/"screening"/"c528_complementary_partition_operator_nmf"/"run_synthetic_screen.py")
c529=load("c529_cov",ROOT/"screening"/"c529_partition_intersection_line_nmf"/"run_synthetic_screen.py")
c530=load("c530_cov",ROOT/"screening"/"c530_confirmatory_line_operator_nmf"/"run_confirmatory_screen.py")
EPS=c528.EPS; K=c528.K; P=c528.P; SEEDS=tuple(range(53200,53204)); SCENARIOS=("width3_full","width6_full","ultra_low_full","boundary_full")
METHODS=("uniform_backproject","separate_arithmetic","wrong_operator","single_correct","c530_certified_operator","fine_count_oracle")

def coverage(width,offsets):
 actual=np.concatenate([c528.partition(width,o) for o in offsets],axis=1); complete=np.concatenate([c528.partition(width,o) for o in range(width)],axis=1); return float(np.linalg.matrix_rank(actual)/np.linalg.matrix_rank(complete))

def spec(name):
 if name=="width6_full": return 6,list(range(6)),[36]*6,.50,0
 if name=="ultra_low_full": return 3,[0,1,2],[72]*3,.10,0
 if name=="boundary_full": return 3,[0,1,2],[72]*3,.50,.08
 return 3,[0,1,2],[72]*3,.50,0

def simulate(name,seed):
 width,offsets,ns,scale,leak=spec(name); rng=np.random.default_rng(seed); h=c529.line_basis("nominal"); ys=[]; fines=[]; ws=[]; mats=[]
 for n,o in zip(ns,offsets):
  a=c528.partition(width,o); w=rng.gamma(1.7,.7,(n,K)); w/=np.maximum(w.mean(0,keepdims=True),EPS); w*=scale; rate=.015+w@h; fine=rng.poisson(rate).astype(float); ys.append(fine@a); fines.append(fine); ws.append(w); mats.append((1-leak)*a+leak*np.roll(a,1,axis=0))
 wt=rng.gamma(1.7,.7,(60,K)); wt/=np.maximum(wt.mean(0,keepdims=True),EPS); wt*=scale; rt=.015+wt@h; at=c528.partition(8,3); yt=rng.poisson(rt).astype(float)@at
 return ys,mats,fines,ws,h,yt,at,rt,width,offsets

def run_case(name,seed):
 ys,mats,fines,truews,trueh,yt,at,rt,width,offsets=simulate(name,seed); assert coverage(width,offsets)>=1-1e-12; fits={}; runt={}
 t=time.perf_counter(); bp=c528.backproject_fit(ys,mats,seed+1); fits["uniform_backproject"]=bp; runt["uniform_backproject"]=time.perf_counter()-t
 t=time.perf_counter(); bases=c529.aligned_backprojected_bases(ys,mats,seed+10); ha=np.mean(bases,axis=0); fits["separate_arithmetic"]=(c529.fit_w_all(ys,mats,ha),ha); runt["separate_arithmetic"]=time.perf_counter()-t
 hi=c529.intersection_basis(bases); wi=c529.fit_w_all(ys,mats,hi)
 t=time.perf_counter(); fits["wrong_operator"],_=c530.multistart(ys,[mats[0]]*len(ys),seed+20,bp,hi,wi,n_random=2); runt["wrong_operator"]=time.perf_counter()-t
 t=time.perf_counter(); fits["single_correct"]=c528.fit_operator(ys,mats,seed+30); runt["single_correct"]=time.perf_counter()-t
 t=time.perf_counter(); fits["c530_certified_operator"],stability=c530.multistart(ys,mats,seed+40,bp,hi,wi,n_random=4); runt["c530_certified_operator"]=time.perf_counter()-t
 t=time.perf_counter(); ow,oh=c528.nmf(np.vstack(fines),seed+60); cuts=np.cumsum([len(x) for x in fines])[:-1]; fits["fine_count_oracle"]=(list(np.split(ow,cuts)),oh); runt["fine_count_oracle"]=time.perf_counter()-t
 rows=[]
 for m in METHODS:
  wlist,h=fits[m]; truth=np.vstack([.015+w@trueh for w in truews]); est=np.vstack([w@h for w in wlist]); rows.append({"scenario":name,"seed":seed,"method":m,"runtime":runt[m],"basis_cosine":c528.align_basis(h,trueh),"fine_nmse":float(np.sum((est-truth)**2)/max(np.sum(truth**2),EPS)),"heldout_nmse":c528.heldout(h,yt,at,rt),"coverage_ratio":coverage(width,offsets),"stability":stability})
 return rows

def summarize(rows,total):
 p="c530_certified_operator"; sep="separate_arithmetic"; wrong="wrong_operator"; oracle="fine_count_oracle"; nonops=("uniform_backproject","separate_arithmetic","wrong_operator")
 def sub(m,sc=None): return [r for r in rows if r["method"]==m and(sc is None or r["scenario"]==sc)]
 def avg(m,x,sc=None): return float(np.mean([r[x] for r in sub(m,sc)]))
 pc,ph=avg(p,"basis_cosine"),avg(p,"heldout_nmse"); sc,sh=avg(sep,"basis_cosine"),avg(sep,"heldout_nmse"); wc,wh=avg(wrong,"basis_cosine"),avg(wrong,"heldout_nmse"); oh=avg(oracle,"heldout_nmse"); strongest=min(avg(m,"heldout_nmse") for m in nonops)
 wins=sum(next(r for r in sub(p,g) if r["seed"]==s)["heldout_nmse"]<min(next(r for r in sub(m,g) if r["seed"]==s)["heldout_nmse"] for m in nonops) for g,s in itertools.product(SCENARIOS,SEEDS)); gains={g:float(min(avg(m,"heldout_nmse",g) for m in nonops)-avg(p,"heldout_nmse",g)) for g in SCENARIOS}; ratios={g:avg(p,"heldout_nmse",g)/max(avg(oracle,"heldout_nmse",g),EPS) for g in SCENARIOS}
 diagnostics=[{"width":2,"offsets":[0,1],"ratio":coverage(2,[0,1]),"accept":True},{"width":3,"offsets":[0,1,2],"ratio":coverage(3,[0,1,2]),"accept":True},{"width":6,"offsets":list(range(6)),"ratio":coverage(6,list(range(6))),"accept":True},{"width":3,"offsets":[0,1],"ratio":coverage(3,[0,1]),"accept":False},{"width":6,"offsets":[0,2,4],"ratio":coverage(6,[0,2,4]),"accept":False}]
 diag_ok=all((d["ratio"]>=1-1e-12)==d["accept"] for d in diagnostics); runtime=avg(p,"runtime")/max(avg("uniform_backproject","runtime"),EPS); finite=all(np.isfinite(r[x]) for r in rows for x in("basis_cosine","fine_nmse","heldout_nmse","runtime","coverage_ratio","stability"))
 gates={"g1_basis":pc>=.985,"g2_oracle_mean":ph<=1.25*oh,"g3_nonoperator_gain":ph<=.40*strongest,"g4_wins":wins==16,"g5_scenarios":all(v>0 for v in gains.values()),"g6_specificity":pc>max(sc,wc) and ph<min(sh,wh),"g7_width6":avg(p,"basis_cosine","width6_full")>=.98,"g8_stress_oracle":ratios["boundary_full"]<=1.2 and ratios["ultra_low_full"]<=1.1,"g9_diagnostic":diag_ok,"g10_integrity_runtime":runtime<=25 and finite and len(rows)==96 and len({(r["scenario"],r["seed"],r["method"]) for r in rows})==96}
 methods={m:{x:avg(m,x) for x in("basis_cosine","fine_nmse","heldout_nmse","runtime")} for m in METHODS}
 return {"summary":{"basis_cosine":pc,"fine_nmse":avg(p,"fine_nmse"),"heldout_nmse":ph,"oracle_heldout":oh,"oracle_ratios":ratios,"separate_basis":sc,"separate_heldout":sh,"wrong_basis":wc,"wrong_heldout":wh,"strongest_nonoperator":strongest,"cell_wins":wins,"scenario_gain":gains,"runtime_ratio":runtime,"total_runtime":total},"coverage_diagnostics":diagnostics,"method_summary":methods,"gates":gates,"passed":int(sum(gates.values())),"total":10,"decision":"PASS" if all(gates.values()) else "FAIL","n_records":len(rows)}

def main():
 t=time.perf_counter(); rows=[]
 for g,s in itertools.product(SCENARIOS,SEEDS): rows.extend(run_case(g,s))
 rep=summarize(rows,time.perf_counter()-t); OUT.mkdir(parents=True,exist_ok=True); (OUT/"records.json").write_text(json.dumps(rows,indent=2)); (OUT/"report.json").write_text(json.dumps(rep,indent=2)); print(json.dumps(rep,indent=2))
if __name__=="__main__": main()
