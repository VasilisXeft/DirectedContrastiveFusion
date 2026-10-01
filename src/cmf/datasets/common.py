from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d

from cmf.utils.io import load_npz


def resample_numeric(x, length):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim==1: x=x[:,None]
    if x.shape[0]==length: return x
    if x.shape[0]<2: return np.repeat(x[:1],length,axis=0)
    old=np.linspace(0,1,x.shape[0]); new=np.linspace(0,1,length); f=interp1d(old,x,axis=0,kind="linear",bounds_error=False,fill_value="extrapolate"); return f(new).astype(np.float32)


def sample_video_frames(path,n_frames=16,image_size=112):
    import cv2
    cap=cv2.VideoCapture(str(path)); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total<=0: cap.release(); raise RuntimeError(f"Could not read frames from {path}")
    idxs=np.linspace(0,total-1,n_frames).astype(int); frames,wanted=[],set(idxs.tolist()); i=0
    while cap.isOpened() and len(frames)<len(idxs):
        ok,frame=cap.read()
        if not ok: break
        if i in wanted:
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); frame=cv2.resize(frame,(image_size,image_size),interpolation=cv2.INTER_AREA); frames.append(frame)
        i+=1
    cap.release()
    if not frames: raise RuntimeError(f"No frames decoded from {path}")
    while len(frames)<n_frames: frames.append(frames[-1].copy())
    return np.stack(frames[:n_frames],axis=0).astype(np.uint8)


def sample_image_volume(x,n_frames=16,image_size=112):
    import cv2
    x=np.asarray(x)
    if x.ndim==3:
        t_axis=2 if x.shape[2]<=max(x.shape[0],x.shape[1]) else 0; x=np.moveaxis(x,t_axis,0); x=x[...,None]
    elif x.ndim==4 and x.shape[-1] not in (1,3): x=np.moveaxis(x,-1,0)
    idxs=np.linspace(0,x.shape[0]-1,n_frames).astype(int); out=[]
    for i in idxs:
        fr=x[i]
        if fr.ndim==2: fr=fr[...,None]
        if fr.shape[-1]==1: fr=np.repeat(fr,3,axis=-1)
        fr=fr.astype(np.float32); lo,hi=np.nanpercentile(fr,[1,99]); fr=np.clip((fr-lo)/(hi-lo+1e-6),0,1); fr=(fr*255).astype(np.uint8); fr=cv2.resize(fr,(image_size,image_size),interpolation=cv2.INTER_AREA); out.append(fr)
    return np.stack(out,axis=0)


def _quality_penalty(x):
    """Generic cached-feature quality penalty in [0,1].

    Penalizes non-finite values, near-zero temporal variability and extreme
    temporal jumps. This is deliberately representation-level so it works for
    pretrained EEG/PPG/EDA/TEMP/FACE tokens without pretending to be a raw
    sensor-specific SQI.
    """
    a=np.asarray(x,dtype=np.float32)
    finite=np.isfinite(a); nonfinite=1.0-float(finite.mean()) if a.size else 1.0
    a=np.nan_to_num(a,nan=0.0,posinf=0.0,neginf=0.0)
    if a.ndim<2 or a.shape[0]<2: return float(np.clip(nonfinite,0,1))
    scale=float(np.mean(np.abs(a)))+1e-6; temporal_std=float(np.mean(np.std(a,axis=0))); flat=np.exp(-temporal_std/scale)
    dif=np.diff(a,axis=0); jump=float(np.mean(np.abs(dif)))/(scale+1e-6); jump_pen=max(0.0,min(1.0,(jump-2.0)/4.0))
    return float(np.clip(max(nonfinite,0.5*flat+0.5*jump_pen),0,1))


def _synthetic_corrupt(t,severity):
    """Feature-space corruption with known severity for reliability supervision."""
    if severity<=0: return t
    scale=t.detach().std().clamp_min(1e-3); noisy=t+torch.randn_like(t)*(float(severity)*scale)
    if t.ndim>=2 and t.shape[0]>1 and severity>0.5:
        n=max(1,int(round(t.shape[0]*0.25*float(severity)))); start=np.random.randint(0,max(1,t.shape[0]-n+1)); noisy[start:start+n]=0
    return noisy


class CachedMultimodalDataset(Dataset):
    def __init__(self,cache_dir,split="train",modality_dropout=0.0,manifest=None,reliability_corruption=0.0):
        self.cache_dir=Path(cache_dir); self.manifest=pd.read_csv(self.cache_dir/"manifest.csv") if manifest is None else manifest.copy()
        if split is not None: self.manifest=self.manifest[self.manifest["split"]==split].reset_index(drop=True)
        else: self.manifest=self.manifest.reset_index(drop=True)
        self.modality_dropout=float(modality_dropout); self.reliability_corruption=float(reliability_corruption)
        if len(self.manifest)==0: raise ValueError(f"No samples for split={split} in {self.cache_dir/'manifest.csv'}")

    def __len__(self): return len(self.manifest)

    def __getitem__(self,idx):
        row=self.manifest.iloc[idx]; mods,target,meta=load_npz(self.cache_dir/row["file"]); out={}; present={}; quality_penalty={}; reliability_target={}; missing=set(meta.get("missing_modalities",[]))
        for name,x in mods.items():
            naturally_missing=name in missing; drop=(not naturally_missing) and self.modality_dropout>0 and np.random.rand()<self.modality_dropout; present[name]=0.0 if (drop or naturally_missing) else 1.0
            base_pen=_quality_penalty(x)
            if x.ndim==4: t=torch.from_numpy(x).float().permute(0,3,1,2)/255.0
            else: t=torch.from_numpy(x).float()
            severity=0.0
            if (not naturally_missing) and (not drop) and self.reliability_corruption>0 and np.random.rand()<self.reliability_corruption:
                severity=float(np.random.uniform(0.25,1.0)); t=_synthetic_corrupt(t,severity)
            if drop: t=torch.zeros_like(t)
            q=max(base_pen,severity); quality_penalty[name]=float(q); reliability_target[name]=float(1.0-q) if present[name]>0 else 0.0; out[name]=t
        y=torch.as_tensor(target)
        return {"modalities":out,"present":present,"quality_penalty":quality_penalty,"reliability_target":reliability_target,"target":y,"meta":meta}


def collate_multimodal(batch):
    names=sorted(batch[0]["modalities"].keys()); modalities={n:torch.stack([b["modalities"][n] for b in batch]) for n in names}; present={n:torch.tensor([b["present"][n] for b in batch],dtype=torch.float32) for n in names}; targets=torch.stack([b["target"] for b in batch])
    quality_penalty={n:torch.tensor([b.get("quality_penalty",{}).get(n,0.0) for b in batch],dtype=torch.float32) for n in names}; reliability_target={n:torch.tensor([b.get("reliability_target",{}).get(n,1.0) for b in batch],dtype=torch.float32) for n in names}
    return {"modalities":modalities,"present":present,"quality_penalty":quality_penalty,"reliability_target":reliability_target,"target":targets,"meta":[b["meta"] for b in batch]}
