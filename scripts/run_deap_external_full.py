"""Run the full DEAP-4M LOSO benchmark on validated frozen external features.

Default matrix:
  Late, Full,
  Random global K={3,6,9}, Similarity global K={3,6,9},
  Directed global K={3,6,9}, Contrastive Directed global K={3,6,9}.

Use --core-only for the primary matrix (Late, Full, Directed, Contrastive).
Runs are named by method/K/seed so checkpoints and summaries never collide.
"""
import argparse, json
from pathlib import Path
from cmf.loso import run_loso

SPARSE=("random_topk","similarity_topk","directed_topk","contrastive_topk")

def matrix(core_only=False):
    out=[("late",None),("full",None)]
    modes=("directed_topk","contrastive_topk") if core_only else SPARSE
    out += [(m,k) for m in modes for k in (3,6,9)]
    return out

def tag(mode,k,seed):
    base=mode if k is None else f"{mode}_K{k}"
    return f"{base}_seed{seed}"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/deap_external.yaml")
    p.add_argument("--seeds",type=int,nargs="+",default=[42])
    p.add_argument("--core-only",action="store_true")
    p.add_argument("--subject",default=None,help="Run one held-out subject instead of all 32 folds.")
    p.add_argument("--skip-existing",action="store_true",help="Skip an experiment if its LOSO summary already exists (all-fold runs only).")
    a=p.parse_args()
    cfg_name="deap_external"
    completed=[]
    for seed in a.seeds:
        for mode,k in matrix(a.core_only):
            name=tag(mode,k,seed)
            summary=Path("runs")/cfg_name/"loso"/f"{name}_summary.json"
            if a.subject is None and a.skip_existing and summary.exists():
                print(f"SKIP existing: {summary}"); continue
            print("\n"+"="*80)
            print(f"EXPERIMENT {name} | "+("32-fold LOSO" if a.subject is None else f"subject={a.subject}"))
            print("="*80)
            res=run_loso(a.config,subject=a.subject,run_all=(a.subject is None),
                         mode=mode,topk=k,seed=seed,experiment_name=name)
            completed.append({"name":name,"mode":mode,"topk":k,"seed":seed,"folds":len(res)})
    print("\nCOMPLETED")
    print(json.dumps(completed,indent=2))

if __name__=="__main__":
    main()
