"""Locked MassBank 2026.03 public-line-shape falsification for C530."""
from __future__ import annotations
import base64,glob,itertools,json,re,time,warnings
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

HERE=Path(__file__).resolve().parent; DATA=HERE/"massbank_2026_03_fixed"; OUT=HERE/"results_massbank"; EPS=1e-12
P=1280; K=4; LOW=80.0; STEP=.25; SEEDS=tuple(range(53300,53304)); DEPTHS={"standard":.35,"low":.12}
METHODS=("uniform_backproject","separate_arithmetic","wrong_operator","single_correct","c530_certified_operator","fine_count_oracle")

def parse_record(path):
 d=json.loads(path.read_text(encoding="utf-8")); text=base64.b64decode(d["content"]).decode("utf-8");
 def field(pattern,default=""):
  m=re.search(pattern,text,re.MULTILINE); return m.group(1).strip() if m else default
 peaks=[]; active=False
 for line in text.splitlines():
  if line.startswith("PK$PEAK:"): active=True; continue
  if active:
   if line.startswith("//"): break
   parts=line.split()
   if len(parts)>=3:
    try: peaks.append((float(parts[0]),float(parts[2])))
    except ValueError: pass
 vector=np.zeros(P)
 for mz,intensity in peaks:
  j=int(np.floor((mz-LOW)/STEP))
  if 0<=j<P: vector[j]+=intensity
 occupied=int(np.count_nonzero(vector)); mode=field(r"AC\$MASS_SPECTROMETRY: ION_MODE (.+)"); ms=field(r"AC\$MASS_SPECTROMETRY: MS_TYPE (.+)")
 eligible="DEPRECATED:" not in text and mode=="POSITIVE" and ms=="MS2" and occupied>=5
 return {"accession":field(r"ACCESSION: (.+)"),"title":field(r"RECORD_TITLE: (.+)"),"license":field(r"LICENSE: (.+)"),"sha":d.get("sha",""),"occupied":occupied,"eligible":eligible,"vector":vector}

def panels_and_provenance():
 recs=[parse_record(Path(p)) for p in sorted(glob.glob(str(DATA/"*.json")))]; eligible=[r for r in recs if r["eligible"]]; panels=[]
 for start in range(0,len(eligible)-3,4):
  group=eligible[start:start+4]; h=np.vstack([r["vector"] for r in group]); hn=h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),EPS); pair=hn@hn.T
  if np.max(pair-np.eye(4))<.98: panels.append(group)
 provenance=[{k:r[k] for k in ("accession","title","license","sha","occupied","eligible")} for r in recs]
 return panels,provenance

def bin_map(width,offset): return ((np.arange(P)-offset)%P)//width
def aggregate(x,mapping):
 b=int(mapping.max()+1); out=np.zeros((len(x),b))
 for j in range(P): out[:,mapping[j]]+=x[:,j]
 return out
def aggregate_basis(h,mapping):
 b=int(mapping.max()+1); out=np.zeros((K,b))
 for j in range(P): out[:,mapping[j]]+=h[:,j]
 return out
def backproject(y,mapping):
 counts=np.bincount(mapping,minlength=int(mapping.max()+1)); return y[:,mapping]/counts[mapping][None,:]

def nmf(x,seed,iters=500):
 model=NMF(n_components=K,init="nndsvda",solver="mu",beta_loss="kullback-leibler",max_iter=iters,tol=1e-5,random_state=seed)
 with warnings.catch_warnings(): warnings.simplefilter("ignore",ConvergenceWarning); w=model.fit_transform(np.maximum(x,EPS))
 return np.maximum(w,EPS),np.maximum(model.components_,EPS)
def fixed_w(y,b,w,iters=250):
 den=np.ones_like(y)@b.T
 for _ in range(iters): w*=((y/np.maximum(w@b,EPS))@b.T)/np.maximum(den,EPS); w=np.maximum(w,EPS)
 return w

def fit_operator(ys,maps,seed,h0=None,w0=None,iters=420):
 rng=np.random.default_rng(seed); h=np.maximum(h0.copy() if h0 is not None else rng.gamma(1,1,(K,P)),EPS); ws=[np.maximum(w.copy(),EPS) for w in w0] if w0 is not None else [rng.gamma(1,1,(len(y),K)) for y in ys]
 for step in range(iters):
  for c,(y,mapping) in enumerate(zip(ys,maps)): ws[c]=fixed_w(y,aggregate_basis(h,mapping),ws[c],1)
  num=np.zeros_like(h); den=np.zeros_like(h)
  for y,mapping,w in zip(ys,maps,ws):
   b=aggregate_basis(h,mapping); pred=np.maximum(w@b,EPS); q=w.T@(y/pred); d=w.T@np.ones_like(y); num+=q[:,mapping]; den+=d[:,mapping]
  h*=num/np.maximum(den,EPS); h=np.maximum(h,EPS)
  if step%25==24:
   scale=np.maximum(h.mean(1),EPS); h/=scale[:,None]; ws=[w*scale[None,:] for w in ws]
 return ws,h

def deviance(ys,maps,fit):
 ws,h=fit; return float(sum(np.sum(np.maximum(w@aggregate_basis(h,m),EPS)-y*np.log(np.maximum(w@aggregate_basis(h,m),EPS))) for y,m,w in zip(ys,maps,ws)))
def align_similarity(h,ref):
 hn=h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),EPS); rn=ref/np.maximum(np.linalg.norm(ref,axis=1,keepdims=True),EPS); rr,cc=linear_sum_assignment(-(hn@rn.T)); return float(np.mean((hn@rn.T)[rr,cc]))
def multistart(ys,maps,seed,bp,sep,n_random):
 fits=[fit_operator(ys,maps,seed,h0=bp[1],w0=bp[0]),fit_operator(ys,maps,seed+1,h0=sep[1],w0=sep[0])]; fits.extend(fit_operator(ys,maps,seed+2+i) for i in range(n_random)); best=fits[int(np.argmin([deviance(ys,maps,x) for x in fits]))]; return best,float(np.mean([align_similarity(x[1],best[1]) for x in fits]))
def backproject_fit(ys,maps,seed):
 xs=[backproject(y,m) for y,m in zip(ys,maps)]; w,h=nmf(np.vstack(xs),seed); cuts=np.cumsum([len(x) for x in xs])[:-1]; return list(np.split(w,cuts)),h
def separate_fit(ys,maps,seed):
 bases=[]
 for c,(y,m) in enumerate(zip(ys,maps)):
  _,hc=nmf(y,seed+c); bases.append(backproject(hc,m))
 ref=bases[0]/np.maximum(np.linalg.norm(bases[0],axis=1,keepdims=True),EPS); aligned=[]
 for h in bases:
  hn=h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),EPS); rr,cc=linear_sum_assignment(-(ref@hn.T)); aligned.append(h[cc[np.argsort(rr)]])
 h=np.mean(aligned,axis=0); return [fixed_w(y,aggregate_basis(h,m),np.ones((len(y),K)),300) for y,m in zip(ys,maps)],h
def basis_cosine(h,trueh):
 hn=h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),EPS); tn=trueh/np.maximum(np.linalg.norm(trueh,axis=1,keepdims=True),EPS); rr,cc=linear_sum_assignment(-(hn@tn.T)); return float(np.mean((hn@tn.T)[rr,cc]))
def heldout(h,yt,mt,rt):
 w=fixed_w(yt,aggregate_basis(h,mt),np.ones((len(yt),K)),400); return float(np.sum((w@h-rt)**2)/max(np.sum(rt**2),EPS))

def run_cell(panel_index,panel,depth_name,seed):
 rng=np.random.default_rng(seed+100*panel_index); trueh=np.vstack([r["vector"] for r in panel]); trueh/=np.maximum(trueh.mean(1,keepdims=True),EPS); maps=[bin_map(4,o) for o in range(4)]; ys=[]; fines=[]; truews=[]; scale=DEPTHS[depth_name]
 for mapping in maps:
  w=rng.gamma(1.7,.7,(40,K)); w/=np.maximum(w.mean(0,keepdims=True),EPS); w*=scale; rate=.002+w@trueh; fine=rng.poisson(rate).astype(float); ys.append(aggregate(fine,mapping)); fines.append(fine); truews.append(w)
 wt=rng.gamma(1.7,.7,(48,K)); wt/=np.maximum(wt.mean(0,keepdims=True),EPS); wt*=scale; rt=.002+wt@trueh; mt=bin_map(8,3); yt=aggregate(rng.poisson(rt).astype(float),mt); fits={}; runt={}
 t=time.perf_counter(); bp=backproject_fit(ys,maps,seed+1); fits["uniform_backproject"]=bp; runt["uniform_backproject"]=time.perf_counter()-t
 t=time.perf_counter(); sep=separate_fit(ys,maps,seed+10); fits["separate_arithmetic"]=sep; runt["separate_arithmetic"]=time.perf_counter()-t
 t=time.perf_counter(); fits["wrong_operator"],_=multistart(ys,[maps[0]]*4,seed+20,bp,sep,1); runt["wrong_operator"]=time.perf_counter()-t
 t=time.perf_counter(); fits["single_correct"]=fit_operator(ys,maps,seed+30); runt["single_correct"]=time.perf_counter()-t
 t=time.perf_counter(); fits["c530_certified_operator"],stability=multistart(ys,maps,seed+40,bp,sep,4); runt["c530_certified_operator"]=time.perf_counter()-t
 t=time.perf_counter(); ow,oh=nmf(np.vstack(fines),seed+60); cuts=np.cumsum([len(x) for x in fines])[:-1]; fits["fine_count_oracle"]=(list(np.split(ow,cuts)),oh); runt["fine_count_oracle"]=time.perf_counter()-t
 rows=[]
 for method in METHODS:
  ws,h=fits[method]; truth=np.vstack([.002+w@trueh for w in truews]); est=np.vstack([w@h for w in ws]); rows.append({"panel":panel_index,"accessions":[r["accession"] for r in panel],"depth":depth_name,"seed":seed,"method":method,"runtime":runt[method],"basis_cosine":basis_cosine(h,trueh),"fine_nmse":float(np.sum((est-truth)**2)/max(np.sum(truth**2),EPS)),"heldout_nmse":heldout(h,yt,mt,rt),"stability":stability})
 return rows

def summarize(rows,provenance,n_panels,total):
 p="c530_certified_operator"; sep="separate_arithmetic"; wrong="wrong_operator"; oracle="fine_count_oracle"; nonops=("uniform_backproject","separate_arithmetic","wrong_operator")
 def sub(m,panel=None,depth=None): return [r for r in rows if r["method"]==m and(panel is None or r["panel"]==panel) and(depth is None or r["depth"]==depth)]
 def avg(m,x,panel=None,depth=None): return float(np.mean([r[x] for r in sub(m,panel,depth)]))
 pc,ph=avg(p,"basis_cosine"),avg(p,"heldout_nmse"); oh=avg(oracle,"heldout_nmse"); strongest=min(avg(m,"heldout_nmse") for m in nonops); cells=sorted({(r["panel"],r["depth"],r["seed"]) for r in rows}); wins=sum(next(r for r in rows if (r["panel"],r["depth"],r["seed"])==cell and r["method"]==p)["heldout_nmse"]<min(next(r for r in rows if (r["panel"],r["depth"],r["seed"])==cell and r["method"]==m)["heldout_nmse"] for m in nonops) for cell in cells); strata={(q,d):float(min(avg(m,"heldout_nmse",q,d) for m in nonops)-avg(p,"heldout_nmse",q,d)) for q in range(n_panels) for d in DEPTHS}; stability=avg(p,"stability"); runtime=avg(p,"runtime")/max(avg("uniform_backproject","runtime"),EPS); finite=all(np.isfinite(r[x]) for r in rows for x in("basis_cosine","fine_nmse","heldout_nmse","runtime","stability")); expected=len(cells)*len(METHODS)
 gates={"g1_panels":n_panels>=2,"g2_basis":pc>=.95,"g3_oracle":ph<=1.5*oh,"g4_nonoperator_gain":ph<=.25*strongest,"g5_wins":wins>=.75*len(cells),"g6_strata":all(v>0 for v in strata.values()),"g7_specificity":pc>max(avg(sep,"basis_cosine"),avg(wrong,"basis_cosine")) and ph<min(avg(sep,"heldout_nmse"),avg(wrong,"heldout_nmse")),"g8_stability":stability>=.95,"g9_runtime":runtime<=30,"g10_provenance_integrity":all(r["sha"] and r["license"] for r in provenance if r["eligible"]) and finite and len(rows)==expected and len({(r["panel"],r["depth"],r["seed"],r["method"]) for r in rows})==expected}
 methods={m:{x:avg(m,x) for x in("basis_cosine","fine_nmse","heldout_nmse","runtime")} for m in METHODS}
 return {"summary":{"n_panels":n_panels,"n_cells":len(cells),"basis_cosine":pc,"fine_nmse":avg(p,"fine_nmse"),"heldout_nmse":ph,"oracle_heldout":oh,"strongest_nonoperator":strongest,"cell_wins":wins,"stratum_gain":{f"panel{q}_{d}":v for (q,d),v in strata.items()},"stability":stability,"runtime_ratio":runtime,"total_runtime":total},"provenance":provenance,"method_summary":methods,"gates":gates,"passed":int(sum(gates.values())),"total":10,"decision":"PASS" if all(gates.values()) else "FAIL","n_records":len(rows)}

def main():
 panels,provenance=panels_and_provenance(); started=time.perf_counter(); rows=[]
 for q,panel in enumerate(panels):
  for depth,seed in itertools.product(DEPTHS,SEEDS): rows.extend(run_cell(q,panel,depth,seed))
 rep=summarize(rows,provenance,len(panels),time.perf_counter()-started); OUT.mkdir(parents=True,exist_ok=True); (OUT/"records.json").write_text(json.dumps(rows,indent=2)); (OUT/"report.json").write_text(json.dumps(rep,indent=2)); print(json.dumps(rep,indent=2))
if __name__=="__main__": main()
