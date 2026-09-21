"""Sanity check: pretrained MOMENT on one real DEAP EDA window.

Loads the official AutonLab/MOMENT-1-large checkpoint in embedding mode.
MOMENT performs its own normalization internally. Because MOMENT uses a
fixed context length, the DEAP 10-second EDA window is right-padded and
an input_mask marks only the real samples as valid.
"""
from pathlib import Path
import argparse
import torch

from cmf.config import load_config
from cmf.datasets.common import CachedMultimodalDataset


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/deap.yaml")
    p.add_argument("--index",type=int,default=0)
    p.add_argument("--model",default="AutonLab/MOMENT-1-large")
    p.add_argument("--cpu",action="store_true")
    a=p.parse_args()

    try:
        from momentfm import MOMENTPipeline
    except ImportError as e:
        raise SystemExit(
            "MOMENT is not installed. The PyPI 0.1.4 release pins an old NumPy.\n"
            "Install the current official GitHub package instead:\n"
            "  python -m pip install --no-deps git+https://github.com/moment-timeseries-foundation-model/moment.git\n"
            "Then rerun this script."
        ) from e

    cfg=load_config(a.config)
    ds=CachedMultimodalDataset(Path(cfg["dataset"]["cache_dir"]),split="train",modality_dropout=0.0)
    eda=ds[a.index]["modalities"]["eda"].float().squeeze(-1)
    if eda.ndim != 1:
        raise RuntimeError(f"Expected 1-D DEAP EDA window, got {tuple(eda.shape)}")

    print(f"DEAP EDA raw: {tuple(eda.shape)} @ 128 Hz")
    print(f"Raw finite: {bool(torch.isfinite(eda).all())}")
    print(f"Raw range: [{eda.min().item():.6g}, {eda.max().item():.6g}]")

    device=torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    print(f"Loading pretrained MOMENT: {a.model}")

    model=MOMENTPipeline.from_pretrained(
        a.model,
        model_kwargs={"task_name":"embedding"},
    )
    model.init()
    model=model.to(device).eval()

    # Official MOMENT checkpoints use a fixed sequence context. Read it from
    # the loaded model instead of hard-coding it.
    seq_len=int(model.config.seq_len)
    n=int(eda.numel())
    if n > seq_len:
        raise RuntimeError(f"EDA window ({n}) is longer than MOMENT context ({seq_len}).")
    x=torch.zeros((1,1,seq_len),dtype=torch.float32)
    mask=torch.zeros((1,seq_len),dtype=torch.long)
    x[0,0,:n]=eda
    mask[0,:n]=1
    x=x.to(device); mask=mask.to(device)

    print(f"MOMENT context: {seq_len}")
    print(f"Model input: {tuple(x.shape)}")
    print(f"Valid samples in input_mask: {int(mask.sum())}/{seq_len}")

    with torch.inference_mode():
        o1=model(x_enc=x,input_mask=mask)
        o2=model(x_enc=x,input_mask=mask)

    e1=o1.embeddings
    e2=o2.embeddings
    print(f"Embedding shape: {tuple(e1.shape)}")
    print(f"Embedding finite: {bool(torch.isfinite(e1).all())}")
    diff=(e1-e2).abs().max().item()
    print(f"Repeat max abs difference: {diff:.8g}")
    print(f"Deterministic eval: {diff < 1e-6}")

    # Also expose unreduced patch tokens if supported by the installed MOMENT.
    try:
        with torch.inference_mode():
            tok=model.embed(x_enc=x,input_mask=mask,reduction="none").embeddings
        print(f"Unreduced token shape: {tuple(tok.shape)}")
        print(f"Unreduced tokens finite: {bool(torch.isfinite(tok).all())}")
    except Exception as exc:
        print(f"Unreduced token check skipped: {type(exc).__name__}: {exc}")

    print("MOMENT EDA sanity check: PASS")


if __name__=="__main__":
    main()
