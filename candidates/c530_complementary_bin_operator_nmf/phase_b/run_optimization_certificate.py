"""Locked top-two observed-likelihood basin certificate for C530 MassBank."""
from __future__ import annotations
import importlib.util,itertools,json,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; OUT=HERE/"results_optimization_certificate"
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
m=load("massbank_cert",HERE/"run_massbank_public.py")

def run_cell(panel_index,panel,depth_name,seed):
 rng=np.random.default_rng(seed+100*panel_index); trueh=np.vstack([r["vector"] for r in panel]); trueh/=np.maximum(trueh.mean(1,keepdims=True),m.EPS); maps=[m.bin_map(4,o) for o in range(4)]; ys=[]; fines=[]; truews=[]; scale=m.DEPTHS[depth_name]
 for mapping in maps:
  w=rng.gamma(1.7,.7,(40,m.K)); w/=np.maximum(w.mean(0,keepdims=True),m.EPS); w*=scale; rate=.002+w@trueh; fine=rng.poisson(rate).astype(float); ys.append(m.aggregate(fine,mapping)); fines.append(fine); truews.append(w)
 wt=rng.gamma(1.7,.7,(48,m.K)); wt/=np.maximum(wt.mean(0,keepdims=True),m.EPS); wt*=scale; rt=.002+wt@trueh; mt=m.bin_map(8,3); yt=m.aggregate(rng.poisson(rt).astype(float),mt)
 t=time.perf_counter(); bp=m.backproject_fit(ys,maps,seed+1); bp_time=time.perf_counter()-t; sep=m.separate_fit(ys,maps,seed+10)
 t=time.perf_counter(); fits=[m.fit_operator(ys,maps,seed+40,h0=bp[1],w0=bp[0]),m.fit_operator(ys,maps,seed+41,h0=sep[1],w0=sep[0])]; fits.extend(m.fit_operator(ys,maps,seed+42+i) for i in range(4)); runtime=time.perf_counter()-t
 losses=np.array([m.deviance(ys,maps,x) for x in fits]); order=np.argsort(losses); first,second=fits[int(order[0])],fits[int(order[1])]; similarity=m.align_similarity(first[1],second[1]); gap=float((losses[order[1]]-losses[order[0]])/max(abs(losses[order[0]]),m.EPS)); basis=m.basis_cosine(first[1],trueh); held=m.heldout(first[1],yt,mt,rt)
 return {"panel":panel_index,"depth":depth_name,"seed":seed,"top_two_cosine":similarity,"relative_deviance_gap":gap,"certified":similarity>=.95,"selected_basis_cosine":basis,"selected_heldout_nmse":held,"selected_start":int(order[0]),"second_start":int(order[1]),"runtime":runtime,"backproject_runtime":bp_time}

def main():
 panels,provenance=m.panels_and_provenance(); started=time.perf_counter(); rows=[]
 for q,panel in enumerate(panels):
  for depth,seed in itertools.product(m.DEPTHS,m.SEEDS): rows.append(run_cell(q,panel,depth,seed))
 ref=json.loads((HERE/"results_massbank"/"report.json").read_text())["summary"]; mean=lambda key:float(np.mean([r[key] for r in rows])); strata={f"panel{q}_{d}":float(min(r["top_two_cosine"] for r in rows if r["panel"]==q and r["depth"]==d)) for q in range(len(panels)) for d in m.DEPTHS}; runtime_ratio=mean("runtime")/max(mean("backproject_runtime"),m.EPS); finite=all(np.isfinite(r[k]) for r in rows for k in ("top_two_cosine","relative_deviance_gap","selected_basis_cosine","selected_heldout_nmse","runtime","backproject_runtime"))
 gates={"g1_all_certified":all(r["certified"] for r in rows) and len(rows)==16,"g2_mean_agreement":mean("top_two_cosine")>=.99,"g3_basis":mean("selected_basis_cosine")>=.98,"g4_reproduce":abs(mean("selected_heldout_nmse")-ref["heldout_nmse"])<=1e-8,"g5_deviance_gap":mean("relative_deviance_gap")<=.01,"g6_strata":all(v>=.95 for v in strata.values()),"g7_oracle":mean("selected_heldout_nmse")<=1.5*ref["oracle_heldout"],"g8_selection_integrity":all(r["selected_start"] in range(6) and r["second_start"] in range(6) for r in rows),"g9_runtime":runtime_ratio<=30,"g10_integrity":finite and len({(r["panel"],r["depth"],r["seed"]) for r in rows})==len(rows)}
 report={"summary":{"n_cells":len(rows),"certified_cells":sum(r["certified"] for r in rows),"mean_top_two_cosine":mean("top_two_cosine"),"mean_relative_deviance_gap":mean("relative_deviance_gap"),"selected_basis_cosine":mean("selected_basis_cosine"),"selected_heldout_nmse":mean("selected_heldout_nmse"),"locked_oracle_heldout":ref["oracle_heldout"],"stratum_min_agreement":strata,"runtime_ratio":runtime_ratio,"total_runtime":time.perf_counter()-started},"gates":gates,"passed":int(sum(gates.values())),"total":10,"decision":"PASS" if all(gates.values()) else "FAIL","rows":rows}
 OUT.mkdir(parents=True,exist_ok=True); (OUT/"report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
