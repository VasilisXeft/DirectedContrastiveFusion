"""Migrate an existing DEAP external-feature cache to the standard CMF NPZ schema.

No foundation-model inference is performed. Existing feature arrays are copied
and target/meta_json are restored from the validated original DEAP cache.
Migration is atomic per file and resumable.
"""
from pathlib import Path
import argparse, os, tempfile
import numpy as np
import pandas as pd

MODS=("eeg","ppg","eda","temp")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="data/processed/deap")
    p.add_argument("--features",default="data/features/deap_external")
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    src=Path(a.source); feat=Path(a.features)
    man=pd.read_csv(src/"manifest.csv")
    migrated=already=failed=0

    for i,row in man.iterrows():
        fn=row["file"]; fp=feat/fn; sp=src/fn
        try:
            if not fp.exists(): raise FileNotFoundError(f"feature file missing: {fp}")
            with np.load(fp,allow_pickle=False) as f:
                # Already migrated and structurally complete: leave untouched.
                required={*(f"x__{m}" for m in MODS),"target","meta_json"}
                if required.issubset(f.files):
                    already+=1; continue
                arrays={}
                for m in MODS:
                    key=f"x__{m}" if f"x__{m}" in f.files else m
                    if key not in f.files: raise KeyError(f"missing feature {m}")
                    arrays[f"x__{m}"]=np.asarray(f[key]).copy()

            with np.load(sp,allow_pickle=False) as s:
                arrays["target"]=np.asarray(s["target"]).copy()
                arrays["meta_json"]=np.asarray(s["meta_json"]).copy() if "meta_json" in s.files else np.asarray("{}")

            if not a.dry_run:
                # Same directory + os.replace gives an atomic per-file replacement.
                fd,tmp=tempfile.mkstemp(prefix=fp.stem+"_",suffix=".npz",dir=feat)
                os.close(fd)
                try:
                    np.savez_compressed(tmp,**arrays)
                    # Verify the temporary archive before replacing the original.
                    with np.load(tmp,allow_pickle=False) as z:
                        for m in MODS:
                            if f"x__{m}" not in z.files: raise RuntimeError(f"temp verify failed: {m}")
                        if "target" not in z.files or "meta_json" not in z.files:
                            raise RuntimeError("temp verify failed: target/meta_json")
                    os.replace(tmp,fp)
                finally:
                    if os.path.exists(tmp): os.remove(tmp)
            migrated+=1
            if migrated%500==0 or i==0:
                print(f"Processed {i+1}/{len(man)} migrated={migrated} already={already}")
        except Exception as e:
            failed+=1
            print(f"ERROR {fn}: {type(e).__name__}: {e}")
            if failed>=20: raise SystemExit("Stopping after 20 errors.")

    print(f"Migration complete: total={len(man)} migrated={migrated} already={already} failed={failed} dry_run={a.dry_run}")
    if failed: raise SystemExit(1)

if __name__=="__main__":
    main()
