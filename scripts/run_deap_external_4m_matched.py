"""Run matched DEAP-4M core LOSO matrix on exactly the 5M windows."""
import argparse, json
from pathlib import Path
from cmf.loso import run_loso
def matrix(): return [("late",None),("full",None)]+[(m,k) for m in ("directed_topk","contrastive_topk") for k in (3,6,9)]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/deap_external_4m_matched.yaml"); p.add_argument("--seeds",type=int,nargs="+",default=[42]); p.add_argument("--subject",default=None); p.add_argument("--skip-existing",action="store_true"); a=p.parse_args()
 for seed in a.seeds:
  for mode,k in matrix():
   name=f"{mode}{'' if k is None else f'_K{k}'}_seed{seed}"; summary=Path("runs")/"deap_external_4m_matched"/"loso"/f"{name}_summary.json"
   if a.subject is None and a.skip_existing and summary.exists(): print("SKIP",summary); continue
   run_loso(a.config,subject=a.subject,run_all=(a.subject is None),mode=mode,topk=k,seed=seed,experiment_name=name)
if __name__=="__main__": main()
