"""Run DEAP-5M LOSO experiments on EEG+PPG+EDA+TEMP+FACE.

Five modalities give 20 possible directed interactions.
K={5,10,15} corresponds to 25%, 50%, 75% selected interaction budget.
Use --mode/--topk for a single diagnostic experiment; omit them for the full matrix.
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
    p.add_argument("--seed",type=int,default=None,help="Convenience alias for one seed")
    p.add_argument("--subject",default=None)
    p.add_argument("--mode",choices=["late","full","directed_topk","contrastive_topk"],default=None)
    p.add_argument("--topk",type=int,default=None)
    p.add_argument("--skip-existing",action="store_true")
    a=p.parse_args()
    if a.topk is not None and a.mode not in {"directed_topk","contrastive_topk"}:
        p.error("--topk is only valid with --mode directed_topk or contrastive_topk")
    if a.mode in {"directed_topk","contrastive_topk"} and a.topk is None:
        p.error("--topk is required for sparse modes")
    seeds=[a.seed] if a.seed is not None else a.seeds
    experiments=[(a.mode,a.topk)] if a.mode is not None else matrix()
    cfg=load_config(a.config); cfg_name=cfg["dataset"]["name"]; done=[]
    for seed in seeds:
        for mode,k in experiments:
            name=f"{mode}{'' if k is None else f'_K{k}'}_seed{seed}"
            summary=Path(cfg.get("output_dir","runs"))/cfg_name/"loso"/f"{name}_summary.json"
            if a.subject is None and a.skip_existing and summary.exists():
                print("SKIP",summary); continue
            print("\n"+"="*80+f"\nDEAP-5M {name}\n"+"="*80)
            res=run_loso(a.config,subject=a.subject,run_all=(a.subject is None),mode=mode,topk=k,seed=seed,experiment_name=name)
            done.append({"name":name,"folds":len(res)})
    print(json.dumps(done,indent=2))

if __name__=="__main__":
    main()
