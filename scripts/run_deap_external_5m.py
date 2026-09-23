"""Run the primary DEAP-5M LOSO matrix on EEG+PPG+EDA+TEMP+FACE.

Five modalities give 20 possible directed interactions.
K={5,10,15} therefore corresponds to 25%, 50%, 75% selected interaction budget.
"""
import argparse, json
from pathlib import Path
from cmf.config import load_config
from cmf.loso import run_loso

def matrix():
    return [("late",None),("full",None)] + [(m,k) for m in ("directed_topk","contrastive_topk") for k in (5,10,15)]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/deap_external_5m.yaml")
    p.add_argument("--seeds",type=int,nargs="+",default=[42])
    p.add_argument("--subject",default=None)
    p.add_argument("--skip-existing",action="store_true")
    a=p.parse_args()
    cfg=load_config(a.config); cfg_name=cfg["dataset"]["name"]; done=[]
    for seed in a.seeds:
        for mode,k in matrix():
            name=f"{mode}{'' if k is None else f'_K{k}'}_seed{seed}"
            summary=Path(cfg.get("output_dir","runs"))/cfg_name/"loso"/f"{name}_summary.json"
            if a.subject is None and a.skip_existing and summary.exists():
                print("SKIP",summary); continue
            print("\n"+"="*80+f"\nDEAP-5M {name}\n"+"="*80)
            res=run_loso(a.config,subject=a.subject,run_all=(a.subject is None),
                         mode=mode,topk=k,seed=seed,experiment_name=name)
            done.append({"name":name,"folds":len(res)})
    print(json.dumps(done,indent=2))

if __name__=="__main__":
    main()
