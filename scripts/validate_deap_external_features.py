"""Validate the global frozen DEAP external-feature cache.

Checks file coverage, manifest alignment, exact feature shapes/dtypes/finiteness,
labels and metadata against the original preprocessed DEAP cache. Read-only.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd


EXPECTED={"eeg":(10,200),"ppg":(1,512),"eda":(10,1024),"temp":(10,1024)}


def scalar_equal(a,b):
    try:
        return bool(np.array_equal(np.asarray(a),np.asarray(b)))
    except Exception:
        return False


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="data/processed/deap")
    p.add_argument("--features",default="data/features/deap_external")
    p.add_argument("--max-errors",type=int,default=20)
    a=p.parse_args()
    src=Path(a.source); feat=Path(a.features)
    sm=pd.read_csv(src/"manifest.csv"); fm=pd.read_csv(feat/"manifest.csv")
    errors=[]

    def err(msg):
        if len(errors)<a.max_errors: errors.append(msg)

    print(f"Source manifest:  {len(sm)}")
    print(f"Feature manifest: {len(fm)}")
    if len(sm)!=len(fm): err(f"manifest length mismatch: {len(sm)} vs {len(fm)}")
    if "file" not in sm or "file" not in fm: raise SystemExit("Both manifests must contain 'file'.")
    if sm["file"].tolist()!=fm["file"].tolist(): err("manifest file order/content mismatch")

    src_files=set(sm["file"]); feat_files={p.name for p in feat.glob("sample_*.npz")}
    missing=sorted(src_files-feat_files); extra=sorted(feat_files-src_files)
    print(f"Feature NPZ files: {len(feat_files)}")
    print(f"Missing: {len(missing)}; extra: {len(extra)}")
    if missing: err(f"missing examples: {missing[:5]}")
    if extra: err(f"extra examples: {extra[:5]}")

    counts={k:0 for k in EXPECTED}; label_mismatch=meta_mismatch=0
    subjects=set(); trials=set()
    for i,row in sm.iterrows():
        fn=row["file"]; fp=feat/fn; sp=src/fn
        if not fp.exists(): continue
        try:
            f=np.load(fp,allow_pickle=True); s=np.load(sp,allow_pickle=True)
            for k,shape in EXPECTED.items():
                if k not in f.files:
                    err(f"{fn}: missing {k}"); continue
                x=f[k]
                if x.shape!=shape: err(f"{fn}: {k} shape {x.shape} != {shape}")
                if x.dtype!=np.float32: err(f"{fn}: {k} dtype {x.dtype} != float32")
                if not np.isfinite(x).all(): err(f"{fn}: {k} has NaN/Inf")
                counts[k]+=1
            # Cache script preserves whichever label key exists.
            for key in ("y","label"):
                if key in s.files:
                    if key not in f.files or not scalar_equal(s[key],f[key]):
                        label_mismatch+=1; err(f"{fn}: {key} mismatch")
            if "meta" in s.files:
                if "meta" not in f.files or not scalar_equal(s["meta"],f["meta"]):
                    meta_mismatch+=1; err(f"{fn}: meta mismatch")
                try:
                    m=s["meta"].item()
                    if isinstance(m,dict):
                        subjects.add(m.get("subject")); trials.add(m.get("trial"))
                except Exception: pass
        except Exception as e:
            err(f"{fn}: {type(e).__name__}: {e}")
        if (i+1)%1000==0: print(f"Validated {i+1}/{len(sm)}")

    print("\n--- Validation summary ---")
    print("Expected shapes:",EXPECTED)
    print("Valid modality entries:",counts)
    print("Label mismatches:",label_mismatch)
    print("Metadata mismatches:",meta_mismatch)
    print("Subjects observed:",len(subjects))
    print("Trial IDs observed:",len(trials))

    spec=feat/"feature_spec.json"
    if spec.exists():
        d=json.loads(spec.read_text(encoding="utf-8"))
        print("Feature spec samples:",d.get("samples"))

    if errors:
        print(f"\nVALIDATION FAILED ({len(errors)} shown, capped at {a.max_errors})")
        for e in errors: print(" -",e)
        raise SystemExit(1)
    print("\nDEAP external feature cache validation: PASS")


if __name__=="__main__":
    main()
