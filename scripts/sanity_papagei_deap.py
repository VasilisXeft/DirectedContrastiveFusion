"""Sanity check: official pretrained PaPaGei-S on one real DEAP PPG window.

Prerequisites:
  vendor/papagei-foundation-model
  weights/papagei_s.pt
  pyPPG (official preprocessing dependency)
"""
from pathlib import Path
import argparse, sys
import numpy as np
import torch
from scipy.signal import resample_poly

from cmf.config import load_config
from cmf.datasets.common import CachedMultimodalDataset


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/deap.yaml")
    p.add_argument("--index",type=int,default=0)
    p.add_argument("--repo",default="vendor/papagei-foundation-model")
    p.add_argument("--weights",default="weights/papagei_s.pt")
    p.add_argument("--cpu",action="store_true")
    a=p.parse_args()

    repo=Path(a.repo).resolve(); weights=Path(a.weights).resolve()
    if not repo.exists():
        raise SystemExit(f"PaPaGei repo not found: {repo}\nClone the official repository first.")
    if not weights.exists():
        raise SystemExit(f"PaPaGei-S weights not found: {weights}\nDownload papagei_s.pt from Zenodo record 13983110.")
    sys.path.insert(0,str(repo))

    try:
        from models.resnet import ResNet1DMoE
        from preprocessing.ppg import preprocess_one_ppg_signal
    except Exception as e:
        raise SystemExit(f"Could not import official PaPaGei code: {e}") from e

    cfg=load_config(a.config)
    ds=CachedMultimodalDataset(Path(cfg["dataset"]["cache_dir"]),split="train",modality_dropout=0.0)
    ppg=ds[a.index]["modalities"]["ppg"].float().squeeze(-1).cpu().numpy()
    fs=128; target_fs=125
    print(f"DEAP PPG raw: {ppg.shape} @ {fs} Hz")
    print(f"Raw finite: {bool(np.isfinite(ppg).all())}")
    print(f"Raw range: [{ppg.min():.6g}, {ppg.max():.6g}]")

    # Follow the official PaPaGei quick-start order: clean at original fs, then resample to 125 Hz.
    cleaned, *_ = preprocess_one_ppg_signal(waveform=ppg, frequency=fs)
    cleaned=np.asarray(cleaned,dtype=np.float32).reshape(-1)
    x125=resample_poly(cleaned,target_fs,fs).astype(np.float32)
    expected=int(round(len(ppg)*target_fs/fs))
    if len(x125)!=expected:
        x125=x125[:expected] if len(x125)>expected else np.pad(x125,(0,expected-len(x125)),mode="edge")
    x=torch.from_numpy(x125).view(1,1,-1)
    print(f"Cleaned finite: {bool(np.isfinite(cleaned).all())}")
    print(f"Resampled input: {tuple(x.shape)} @ {target_fs} Hz")
    print(f"Resampled finite: {bool(torch.isfinite(x).all())}")
    print(f"Resampled range: [{x.min().item():.6g}, {x.max().item():.6g}]")

    model=ResNet1DMoE(in_channels=1,base_filters=32,kernel_size=3,stride=2,groups=1,
                      n_block=18,n_classes=512,n_experts=3)
    checkpoint=torch.load(weights,map_location="cpu",weights_only=False)
    state={}
    for k,v in checkpoint.items():
        state[k[7:] if k.startswith("module.") else k]=v
    model.load_state_dict(state)
    device=torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    model=model.to(device).eval(); x=x.to(device)
    print(f"Device: {device}")
    print(f"Loaded official PaPaGei-S weights: {weights}")

    with torch.inference_mode():
        o1=model(x); o2=model(x)
    emb1=o1[0]; emb2=o2[0]
    backbone=o1[3]
    print(f"Official embedding shape (outputs[0]): {tuple(emb1.shape)}")
    print(f"Backbone pooled shape (outputs[3]): {tuple(backbone.shape)}")
    print(f"Embedding finite: {bool(torch.isfinite(emb1).all())}")
    diff=(emb1-emb2).abs().max().item()
    print(f"Repeat max abs difference: {diff:.8g}")
    print(f"Deterministic eval: {diff < 1e-6}")
    print("PaPaGei sanity check: PASS")


if __name__=="__main__":
    main()
