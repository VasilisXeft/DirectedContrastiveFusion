"""Create a strict DEAP-4M control matched to the DEAP-5M manifest.

Every retained 5M window is copied from the validated original 4M external
cache, preserving exactly the same sample filenames, labels and metadata while
omitting FACE. This makes 4M-vs-5M comparisons differ only by the FACE modality.
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-4m",default="data/features/deap_external")
    p.add_argument("--source-5m",default="data/features/deap_external_5m")
    p.add_argument("--output",default="data/features/deap_external_4m_matched")
    p.add_argument("--overwrite",action="store_true")
    a=p.parse_args()
    s4=Path(a.source_4m); s5=Path(a.source_5m); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    m5=pd.read_csv(s5/"manifest.csv")
    rows=[]
    for i,row in m5.iterrows():
        src=s4/row["file"]; dst=out/row["file"]
        if not src.exists(): raise FileNotFoundError(f"5M sample has no validated 4M source: {src}")
        if a.overwrite or not dst.exists(): shutil.copy2(src,dst)
        rows.append(row.to_dict())
    pd.DataFrame(rows).to_csv(out/"manifest.csv",index=False)
    # Strong equality checks: same ordered files and same labels/meta.
    for row in rows:
        z4=np.load(out/row["file"],allow_pickle=False); z5=np.load(s5/row["file"],allow_pickle=False)
        if "x__face" in z4.files: raise RuntimeError("FACE leaked into matched 4M cache")
        if int(z4["target"]) != int(z5["target"]): raise RuntimeError(f"Target mismatch: {row['file']}")
        if str(z4["meta_json"]) != str(z5["meta_json"]): raise RuntimeError(f"Metadata mismatch: {row['file']}")
    spec={"source_4m":str(s4),"matched_to_5m":str(s5),"samples":len(rows),
          "subjects":sorted(pd.DataFrame(rows).subject.astype(str).unique().tolist()),
          "modalities":["eeg","ppg","eda","temp"],
          "note":"Exact-window matched 4M control; FACE intentionally omitted."}
    (out/"feature_spec.json").write_text(json.dumps(spec,indent=2),encoding="utf-8")
    print(f"DONE: matched 4M cache has {len(rows)} windows, {len(spec['subjects'])} subjects")

if __name__=="__main__":
    main()
