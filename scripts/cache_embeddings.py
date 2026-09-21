import argparse
from pathlib import Path
import pandas as pd

from cmf.config import load_config
from cmf.loso import _subjects, _fold_manifests
from cmf.pretrain import pretrain_fold
from cmf.embedding_cache import cache_fold_embeddings

p=argparse.ArgumentParser()
p.add_argument("--config",required=True)
p.add_argument("--subject",default=None)
p.add_argument("--force",action="store_true")
a=p.parse_args()

cfg=load_config(a.config); dcfg=cfg["dataset"]; lcfg=cfg.get("loso",{})
manifest=pd.read_csv(Path(dcfg["cache_dir"])/"manifest.csv")
subject_column=lcfg.get("subject_column","subject"); subjects=_subjects(manifest,subject_column)
test_subject=str(a.subject or lcfg.get("test_subject",subjects[-1]))
if test_subject not in subjects: raise ValueError(f"Unknown subject {test_subject}")
fixed=lcfg.get("validation_subject")
if fixed is not None: val_subject=str(fixed)
elif lcfg.get("validation_strategy","previous_subject")=="previous_subject":
    i=subjects.index(test_subject); val_subject=subjects[(i-1)%len(subjects)]
else:
    i=subjects.index(test_subject); val_subject=subjects[(i+1)%len(subjects)]
splits=_fold_manifests(manifest,test_subject,val_subject,subject_column)
pre=pretrain_fold(a.config,splits,test_subject,val_subject)
checkpoints={m:v["checkpoint"] for m,v in pre.items()}
cache_fold_embeddings(a.config,splits,test_subject,val_subject,checkpoints,force=a.force)
