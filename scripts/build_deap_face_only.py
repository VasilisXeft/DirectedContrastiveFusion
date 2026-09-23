"""Create FACE-only cache from the completed DEAP-5M cache."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from cmf.utils.io import save_npz

def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",default="data/features/deap_external_5m"); p.add_argument("--output",default="data/features/deap_face_only"); a=p.parse_args()
    src=Path(a.source); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); man=pd.read_csv(src/"manifest.csv")
    for _,row in man.iterrows():
        z=np.load(src/row["file"],allow_pickle=False); meta=json.loads(str(z["meta_json"]))
        save_npz(out/row["file"],{"face":z["x__face"]},z["target"],meta)
    man.to_csv(out/"manifest.csv",index=False)
    (out/"feature_spec.json").write_text(json.dumps({"source":str(src),"samples":len(man),"modalities":["face"],"shape":[10,256]},indent=2),encoding="utf-8")
    print(f"DONE: FACE-only cache has {len(man)} windows, {man.subject.nunique()} subjects")

if __name__=="__main__":
    main()
