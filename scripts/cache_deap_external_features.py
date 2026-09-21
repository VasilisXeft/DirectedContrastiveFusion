"""Cache fixed external pretrained representations for every DEAP window.

Outputs one NPZ per input sample, preserving labels/metadata. Foundation models
are loaded once. Features are cached BEFORE trainable projection/fusion.

EEG: CBraMod -> channel-mean -> 10 temporal tokens x 200
PPG: PaPaGei-S -> official 512-D embedding -> 1 token x 512
EDA/TEMP: MOMENT-1-large unreduced patches -> valid patch tokens -> adaptive
           temporal pooling to --moment-tokens tokens (default 10) x 1024

The cache is resumable: existing valid files are skipped unless --overwrite.
"""
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly, cheby2, filtfilt

from cmf.config import load_config
from cmf.datasets.common import CachedMultimodalDataset


def ppg_preprocess(waveform, fs=128, target_fs=125):
    b,a=cheby2(4,20,[0.5,12.0],btype="bandpass",fs=fs)
    y=filtfilt(b,a,np.asarray(waveform,dtype=np.float64))
    win=round(fs*50/1000)
    y=filtfilt(np.ones(win)/win,[1.0],y)
    return resample_poly(y,target_fs,fs).astype(np.float32)


def moment_chunks(signal, seq_len):
    n=int(signal.numel()); k=(n+seq_len-1)//seq_len
    x=torch.zeros((k,1,seq_len),dtype=torch.float32)
    mask=torch.zeros((k,seq_len),dtype=torch.long)
    for i in range(k):
        s=i*seq_len; e=min(s+seq_len,n)
        x[i,0,:e-s]=signal[s:e]; mask[i,:e-s]=1
    return x,mask


def pool_moment_tokens(tok, mask, out_tokens):
    # tok [chunks,1,patches,D]. Drop patches that contain no real samples,
    # concatenate chronologically, then adaptive-average-pool to fixed length.
    patch_samples=mask.shape[1]//tok.shape[2]
    valid_patch_counts=torch.div(mask.sum(1)+patch_samples-1,patch_samples,rounding_mode="floor")
    parts=[tok[i,0,:int(valid_patch_counts[i])] for i in range(tok.shape[0])]
    z=torch.cat(parts,dim=0)                       # [T,D]
    z=F.adaptive_avg_pool1d(z.T.unsqueeze(0),out_tokens).squeeze(0).T
    return z


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/deap.yaml")
    p.add_argument("--output",default="data/features/deap_external")
    p.add_argument("--papagei-repo",default="vendor/papagei-foundation-model")
    p.add_argument("--papagei-weights",default="weights/papagei_s.pt")
    p.add_argument("--moment-model",default="AutonLab/MOMENT-1-large")
    p.add_argument("--moment-tokens",type=int,default=10)
    p.add_argument("--limit",type=int,default=None,help="Process only N samples for sanity testing")
    p.add_argument("--overwrite",action="store_true")
    p.add_argument("--cpu",action="store_true")
    a=p.parse_args()

    from braindecode.models import CBraMod
    from momentfm import MOMENTPipeline
    repo=Path(a.papagei_repo).resolve(); weights=Path(a.papagei_weights).resolve()
    if not repo.exists() or not weights.exists():
        raise SystemExit("PaPaGei repo/weights missing; use the paths from the successful sanity check.")
    sys.path.insert(0,str(repo))
    from models.resnet import ResNet1DMoE

    cfg=load_config(a.config); src=Path(cfg["dataset"]["cache_dir"])
    ds=CachedMultimodalDataset(src,split="train",modality_dropout=0.0)
    manifest=pd.read_csv(src/"manifest.csv")
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}; source samples: {len(ds)}")

    # Models loaded exactly once.
    cbra=CBraMod.from_pretrained("braindecode/cbramod-pretrained",n_chans=32,n_times=2000,
                                 sfreq=200,return_encoder_output=True).to(device).eval()
    papa=ResNet1DMoE(in_channels=1,base_filters=32,kernel_size=3,stride=2,groups=1,
                     n_block=18,n_classes=512,n_experts=3)
    ck=torch.load(weights,map_location="cpu",weights_only=False)
    papa.load_state_dict({(k[7:] if k.startswith("module.") else k):v for k,v in ck.items()})
    papa=papa.to(device).eval()
    moment=MOMENTPipeline.from_pretrained(a.moment_model,model_kwargs={"task_name":"embedding"})
    moment.init(); moment=moment.to(device).eval(); seq_len=int(moment.config.seq_len)

    n_total=min(len(ds),a.limit) if a.limit else len(ds)
    records=[]
    for i in range(n_total):
        fname=manifest.iloc[i]["file"]
        dst=out/fname
        if dst.exists() and not a.overwrite:
            records.append(manifest.iloc[i].to_dict()); continue
        s=ds[i]; m=s["modalities"]

        eeg=m["eeg"].float().T.unsqueeze(0)
        eeg=F.interpolate(eeg,size=2000,mode="linear",align_corners=False).to(device)
        ppg=ppg_preprocess(m["ppg"].float().squeeze(-1).cpu().numpy())
        ppg=torch.from_numpy(ppg).view(1,1,-1).to(device)

        with torch.inference_mode():
            # [1,32,10,200] -> [10,200], preserving time while aggregating channels.
            eeg_f=cbra(eeg).mean(dim=1).squeeze(0)
            ppg_f=papa(ppg)[0].squeeze(0).unsqueeze(0)
            mts={}
            for name in ("eda","temp"):
                x,mask=moment_chunks(m[name].float().squeeze(-1),seq_len)
                x=x.to(device); mask=mask.to(device)
                tok=moment.embed(x_enc=x,input_mask=mask,reduction="none").embeddings
                mts[name]=pool_moment_tokens(tok,mask,a.moment_tokens)

        feats={"eeg":eeg_f.cpu().numpy().astype(np.float32),
               "ppg":ppg_f.cpu().numpy().astype(np.float32),
               "eda":mts["eda"].cpu().numpy().astype(np.float32),
               "temp":mts["temp"].cpu().numpy().astype(np.float32)}
        # Keep the original label/meta verbatim where possible.
        raw=np.load(src/fname,allow_pickle=True)
        payload={**feats}
        for key in ("y","label","meta"):
            if key in raw.files: payload[key]=raw[key]
        np.savez_compressed(dst,**payload)
        records.append(manifest.iloc[i].to_dict())
        if (i+1)%25==0 or i==0:
            print(f"[{i+1}/{n_total}] {fname} shapes: "+", ".join(f"{k}={v.shape}" for k,v in feats.items()))

    pd.DataFrame(records).to_csv(out/"manifest.csv",index=False)
    spec={"source":str(src),"samples":n_total,"models":{"eeg":"braindecode/cbramod-pretrained",
          "ppg":"PaPaGei-S official papagei_s.pt","eda":a.moment_model,"temp":a.moment_model},
          "feature_shapes":{"eeg":[10,200],"ppg":[1,512],"eda":[a.moment_tokens,1024],
                            "temp":[a.moment_tokens,1024]},
          "note":"Frozen external features before trainable projection/fusion."}
    (out/"feature_spec.json").write_text(json.dumps(spec,indent=2),encoding="utf-8")
    print("Feature cache complete:",out)


if __name__=="__main__":
    main()
