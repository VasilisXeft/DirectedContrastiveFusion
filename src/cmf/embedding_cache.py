"""Fold-specific caching of frozen unimodal token embeddings."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cmf.config import load_config
from cmf.datasets.common import CachedMultimodalDataset, collate_multimodal
from cmf.models.encoders import load_pretrained_encoder
from cmf.utils.io import ensure_dir, save_npz


def cache_fold_embeddings(config_path, splits, test_subject, val_subject, encoder_checkpoints, force=False):
    cfg=load_config(config_path); dcfg=cfg["dataset"]; mcfg=cfg["model"]; ecfg=cfg.get("embedding_cache",{})
    raw_cache=Path(dcfg["cache_dir"])
    root=Path(ecfg.get("cache_dir","data/features"))/dcfg["name"]/f"test_{test_subject}_val_{val_subject}"
    manifest_path=root/"manifest.csv"
    if manifest_path.exists() and not force:
        print(f"Embedding cache: reusing {root}")
        return root

    ensure_dir(root)
    device=torch.device("cuda" if torch.cuda.is_available() and not ecfg.get("cpu",False) else "cpu")
    batch_size=int(ecfg.get("batch_size",128)); workers=int(ecfg.get("workers",0))
    # Infer raw modality shapes once.
    probe=CachedMultimodalDataset(raw_cache,"train",0,splits["train"])[0]
    shapes={n:tuple(x.shape) for n,x in probe["modalities"].items()}
    encoders={}
    for name,ckpt in encoder_checkpoints.items():
        enc=load_pretrained_encoder(name,shapes[name],mcfg.get("d_model",128),mcfg["encoders"][name],ckpt,freeze=True,device="cpu")
        enc.eval().to(device); encoders[name]=enc

    rows=[]; token_shapes={}
    with torch.inference_mode():
        for split_name,frame in splits.items():
            ds=CachedMultimodalDataset(raw_cache,split_name,0,frame)
            loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=workers,collate_fn=collate_multimodal)
            offset=0
            for batch in loader:
                encoded={n:encoders[n](batch["modalities"][n].to(device)).cpu().numpy().astype(np.float32) for n in encoders}
                bs=batch["target"].shape[0]
                for j in range(bs):
                    src_row=ds.manifest.iloc[offset+j]
                    mods={n:encoded[n][j] for n in encoded}
                    target=batch["target"][j].cpu().numpy()
                    meta=dict(batch["meta"][j])
                    meta["source_file"]=str(src_row["file"])
                    rel=Path(split_name)/f"{offset+j:06d}.npz"
                    ensure_dir((root/rel).parent)
                    save_npz(root/rel,mods,target,meta)
                    row=src_row.to_dict(); row["file"]=rel.as_posix(); row["split"]=split_name; rows.append(row)
                    for n,x in mods.items(): token_shapes[n]=list(x.shape)
                offset += bs

    pd.DataFrame(rows).to_csv(manifest_path,index=False)
    info={"test_subject":str(test_subject),"val_subject":str(val_subject),"d_model":mcfg.get("d_model",128),
          "modalities":list(encoders),"token_shapes":token_shapes,"samples":len(rows)}
    (root/"cache_info.json").write_text(json.dumps(info,indent=2))
    print(f"Embedding cache written: {root}")
    print(f"Cached samples: {len(rows)}; token shapes: {token_shapes}")
    return root
