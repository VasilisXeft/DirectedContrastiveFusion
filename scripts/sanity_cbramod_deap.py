"""Sanity check: pretrained CBraMod on one real DEAP EEG window.

This script is intentionally isolated from the main fusion pipeline.
It loads the curated pretrained CBraMod checkpoint, resamples a cached
10-second DEAP EEG window from 128 Hz to 200 Hz, and checks feature extraction.
"""
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from cmf.config import load_config
from cmf.datasets.common import CachedMultimodalDataset


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/deap.yaml")
    p.add_argument("--index",type=int,default=0)
    p.add_argument("--cpu",action="store_true")
    a=p.parse_args()

    try:
        from braindecode.models import CBraMod
    except ImportError as e:
        raise SystemExit(
            "Braindecode is not installed. Install only the required EEG dependency with:\n"
            "  python -m pip install \"braindecode[hub]\"\n"
            "Then rerun this script."
        ) from e

    cfg=load_config(a.config)
    ds=CachedMultimodalDataset(Path(cfg["dataset"]["cache_dir"]),split="train",modality_dropout=0.0)
    sample=ds[a.index]
    eeg=sample["modalities"]["eeg"].float()  # [T,C]
    if eeg.ndim != 2:
        raise RuntimeError(f"Expected DEAP EEG [T,C], got {tuple(eeg.shape)}")

    src_hz=128
    dst_hz=200
    seconds=eeg.shape[0]/src_hz
    dst_n=int(round(seconds*dst_hz))
    # [T,C] -> [1,C,T] -> interpolate along time.
    x=eeg.T.unsqueeze(0)
    x200=F.interpolate(x,size=dst_n,mode="linear",align_corners=False)

    if dst_n % 200 != 0:
        raise RuntimeError(f"Resampled length {dst_n} is not divisible by CBraMod patch_size=200")

    device=torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    print(f"DEAP EEG raw: {tuple(eeg.shape)} @ {src_hz} Hz")
    print(f"Resampled model input: {tuple(x200.shape)} @ {dst_hz} Hz")
    print(f"Duration: {seconds:.3f} s; expected patches/channel: {dst_n//200}")
    print(f"Input finite: {bool(torch.isfinite(x200).all())}")
    print(f"Input range: [{x200.min().item():.6g}, {x200.max().item():.6g}]")

    print("Loading pretrained CBraMod: braindecode/cbramod-pretrained")
    model=CBraMod.from_pretrained(
        "braindecode/cbramod-pretrained",
        n_chans=x200.shape[1],
        n_times=dst_n,
        sfreq=dst_hz,
        return_encoder_output=True,
    ).to(device)
    model.eval()
    x200=x200.to(device)

    with torch.inference_mode():
        y1=model(x200)
        y2=model(x200)

    def describe(obj,prefix="output"):
        if torch.is_tensor(obj):
            print(f"{prefix} shape: {tuple(obj.shape)}")
            print(f"{prefix} finite: {bool(torch.isfinite(obj).all())}")
            return obj
        if isinstance(obj,dict):
            print(f"{prefix} keys: {list(obj.keys())}")
            tensors=[]
            for k,v in obj.items():
                if torch.is_tensor(v):
                    print(f"{prefix}[{k}] shape: {tuple(v.shape)} finite={bool(torch.isfinite(v).all())}")
                    tensors.append(v)
            return tensors[0] if tensors else None
        print(f"{prefix} type: {type(obj)}")
        return None

    t1=describe(y1)
    t2=None
    if torch.is_tensor(y2): t2=y2
    elif isinstance(y2,dict):
        for v in y2.values():
            if torch.is_tensor(v): t2=v; break
    if t1 is not None and t2 is not None:
        diff=(t1-t2).abs().max().item()
        print(f"Repeat max abs difference: {diff:.8g}")
        print(f"Deterministic eval: {diff < 1e-6}")

    print("CBraMod sanity check: PASS")


if __name__=="__main__":
    main()
