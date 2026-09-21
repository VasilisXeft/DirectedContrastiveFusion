"""Validate the migrated DEAP external-feature cache against the original CMF cache."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

EXPECTED={"eeg":(10,200),"ppg":(1,512),"eda":(10,1024),"temp":(10,1024)}

def same(a,b):
    return bool(np.array_equal(np.asarray(a),np.asarray(b)))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="data/processed/deap")
    p.add_argument("--features",default="data/features/deap_external")
    p.add_argument("--max-errors",type=int,default=20)
    a=p.parse_args(); src=Path(a.source); feat=Path(a.features)
    sm=pd.read_csv(src/"manifest.csv"); fm=pd.read_csv(feat/"manifest.csv")
    errors=[]
    def err(x):
        if len(errors)<a.max_errors: errors.append(x)

    print(f"Source manifest:  {len(sm)}")
    print(f"Feature manifest: {len(fm)}")
    if len(sm)!=len(fm): err(f"manifest length mismatch: {len(sm)} vs {len(fm)}")
    if "file" not in sm.columns or "file" not in fm.columns: raise SystemExit("Both manifests must contain 'file'.")
    if sm["file"].tolist()!=fm["file"].tolist(): err("manifest file order/content mismatch")
    src_names=set(sm["file"]); feat_names={x.name for x in feat.glob("sample_*.npz")}
    missing=sorted(src_names-feat_names); extra=sorted(feat_names-src_names)
    print(f"Feature NPZ files: {len(feat_names)}")
    print(f"Missing: {len(missing)}; extra: {len(extra)}")
    if missing: err(f"missing examples: {missing[:5]}")
    if extra: err(f"extra examples: {extra[:5]}")

    counts={k:0 for k in EXPECTED}; label_mismatch=meta_mismatch=0
    subjects=set(); trials=set()
    for i,row in sm.iterrows():
        fn=row["file"]; fp=feat/fn; sp=src/fn
        if not fp.exists(): continue
        try:
            with np.load(fp,allow_pickle=False) as f, np.load(sp,allow_pickle=False) as s:
                for mod,shape in EXPECTED.items():
                    key=f"x__{mod}"
                    if key not in f.files:
                        err(f"{fn}: missing {key}"); continue
                    x=f[key]
                    if x.shape!=shape: err(f"{fn}: {key} shape {x.shape} != {shape}")
                    if x.dtype!=np.float32: err(f"{fn}: {key} dtype {x.dtype} != float32")
                    if not np.isfinite(x).all(): err(f"{fn}: {key} has NaN/Inf")
                    counts[mod]+=1
                if "target" not in f.files or not same(s["target"],f["target"]):
                    label_mismatch+=1; err(f"{fn}: target mismatch")
                if "meta_json" in s.files:
                    if "meta_json" not in f.files or not same(s["meta_json"],f["meta_json"]):
                        meta_mismatch+=1; err(f"{fn}: meta_json mismatch")
                    meta=json.loads(str(s["meta_json"]))
                    if "subject" in meta: subjects.add(str(meta["subject"]))
                    if "trial" in meta: trials.add(str(meta["trial"]))
        except Exception as e:
            err(f"{fn}: {type(e).__name__}: {e}")
        if (i+1)%1000==0: print(f"Validated {i+1}/{len(sm)}")

    # Manifest is authoritative for LOSO; report it too.
    manifest_subjects=sm["subject"].astype(str).nunique() if "subject" in sm.columns else None
    print("\n--- Validation summary ---")
    print("Expected shapes:",EXPECTED)
    print("Valid modality entries:",counts)
    print("Label mismatches:",label_mismatch)
    print("Metadata mismatches:",meta_mismatch)
    print("Subjects observed in metadata:",len(subjects))
    print("Subjects in manifest:",manifest_subjects)
    print("Trial IDs observed in metadata:",len(trials))
    spec=feat/"feature_spec.json"
    if spec.exists():
        d=json.loads(spec.read_text(encoding="utf-8")); print("Feature spec samples:",d.get("samples"))

    if errors:
        print(f"\nVALIDATION FAILED ({len(errors)} shown, capped at {a.max_errors})")
        for e in errors: print(" -",e)
        raise SystemExit(1)
    print("\nDEAP external feature cache validation: PASS")

if __name__=="__main__":
    main()
