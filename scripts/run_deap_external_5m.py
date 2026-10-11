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
    return [(f"unimodal_{n}",None) for n in ("eeg","ppg","eda","temp","face")] + [("late",None),("early",None),("full",None)] + [(m,k) for m in ("directed_topk","contrastive_topk") for k in (5,10,15)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/deap_external_5m.yaml")
    p.add_argument("--seeds",type=int,nargs="+",default=[42])
    p.add_argument("--seed",type=int,default=None,help="Convenience alias for one seed")
    p.add_argument("--subject",default=None)
    p.add_argument("--mode",choices=["late","early","full","unimodal_eeg","unimodal_ppg","unimodal_eda","unimodal_temp","unimodal_face","directed_topk","contrastive_topk","hsic_topk","clam_topk","utility_topk","hybrid_topk"],default=None)
    p.add_argument("--topk",type=int,default=None)
    p.add_argument("--skip-existing",action="store_true")
    p.add_argument("--dry-run",action="store_true",help="Print train/validation/test subject splits without training")
    a=p.parse_args()
    if a.topk is not None and a.mode not in {"directed_topk","contrastive_topk","hsic_topk","clam_topk","utility_topk","hybrid_topk"}:
        p.error("--topk is only valid with --mode directed_topk or contrastive_topk")
    if a.mode in {"directed_topk","contrastive_topk","hsic_topk","clam_topk","utility_topk","hybrid_topk"} and a.topk is None:
        p.error("--topk is required for sparse modes")
    seeds=[a.seed] if a.seed is not None else a.seeds
    experiments=[(a.mode,a.topk)] if a.mode is not None else matrix()
    cfg=load_config(a.config); cfg_name=cfg["dataset"]["name"]; done=[]
    if a.dry_run:
        import pandas as pd
        from cmf.loso import _subjects, _validation_subjects, _fold_manifests
        cache=Path(cfg["dataset"]["cache_dir"])
        manifest=pd.read_csv(cache/"manifest.csv")
        col=cfg.get("loso",{}).get("subject_column","subject")
        subjects=_subjects(manifest,col)
        tests=[a.subject] if a.subject is not None else subjects
        for test_subject in tests:
            if test_subject not in subjects:
                raise ValueError(f"Unknown test subject {test_subject}. Available: {subjects}")
            val_subjects=_validation_subjects(subjects,test_subject,cfg.get("loso",{}))
            splits=_fold_manifests(manifest,test_subject,val_subjects,col)
            train_subjects=sorted(set(splits["train"][col].astype(str)))
            print(f"test={test_subject} val={val_subjects} train={train_subjects} "
                  f"subject_counts={len(train_subjects)}/{len(val_subjects)}/1 "
                  f"windows(train/val/test)={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
        return
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
