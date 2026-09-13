from pathlib import Path
import re, json
import numpy as np, pandas as pd
from cmf.datasets.common import resample_numeric, sample_video_frames
from cmf.utils.io import ensure_dir, save_npz


def _key(p):
    s=p.stem.lower(); s=re.sub(r'(_rgb|_video|_pose|_skeleton|_acc_phone_clip|_acc_watch_clip|_gyro_clip|_orientation_clip|_acc_phone|_acc_watch|_gyro|_orientation)$','',s); return s

def _read_csv(p,length):
    df=pd.read_csv(p,header=None); num=df.apply(pd.to_numeric,errors='coerce').dropna(axis=1,how='all').dropna(axis=0,how='all'); return resample_numeric(num.to_numpy(np.float32),length)
def _read_pose(p,length):
    obj=json.loads(p.read_text(errors='ignore')); frames=obj if isinstance(obj,list) else obj.get('frames',obj.get('data',[])); arr=[]
    for fr in frames:
        if isinstance(fr,dict): vals=fr.get('keypoints',fr.get('pose_keypoints_2d',fr.get('joints',[])))
        else: vals=fr
        a=np.asarray(vals,dtype=np.float32).reshape(-1); arr.append(a)
    return resample_numeric(np.stack(arr),length) if arr else None

def preprocess_mmact(cfg):
    dcfg=cfg['dataset']; root=Path(dcfg['raw_dir']); out=ensure_dir(dcfg['cache_dir']); samples=ensure_dir(out/'samples'); length=dcfg.get('length',128); inc=set(dcfg.get('include_modalities',[])); maps={m:{} for m in inc}
    for p in root.rglob('*'):
        if not p.is_file(): continue
        low=str(p).lower(); k=_key(p)
        if 'rgb' in inc and p.suffix.lower() in {'.mp4','.avi'}: maps['rgb'][k]=p
        if 'pose' in inc and p.suffix.lower()=='.json' and ('pose' in low or 'skeleton' in low): maps['pose'][k]=p
        if p.suffix.lower()=='.csv':
            for m in inc-{'rgb','pose'}:
                if m.replace('_','') in low.replace('_',''): maps[m][k]=p
    keys=sorted(set().union(*[set(v) for v in maps.values()])); labels={}; rows=[]
    for k in keys:
        mods={}
        for m,mp in maps.items():
            if k not in mp: continue
            try: mods[m]=sample_video_frames(mp[k],dcfg.get('video_frames',16),dcfg.get('image_size',112)) if m=='rgb' else _read_pose(mp[k],length) if m=='pose' else _read_csv(mp[k],length)
            except Exception: pass
        if len(mods)<2: continue
        # MMAct names normally encode subject/action; keep regex permissive.
        sm=re.search(r'(?:subject|s)(\d+)',k); am=re.search(r'(?:action|a)(\d+)',k); subject=int(sm.group(1)) if sm else 0; action=int(am.group(1)) if am else k.split('_')[0]; labels.setdefault(action,len(labels)); y=labels[action]; split='test' if subject in {17,18,19,20} else 'val' if subject in {15,16} else 'train'; rel=f'samples/{len(rows):06d}.npz'; missing=sorted(inc-set(mods)); # absent modalities are not saved; use only complete common subset for batching
        if missing: continue
        save_npz(out/rel,mods,np.array(y,dtype=np.int64),{'key':k,'subject':subject}); rows.append({'sample_id':len(rows),'subject':subject,'label':y,'split':split,'file':rel})
    if not rows: raise RuntimeError('No MMAct clips aligned. Inspect archive layout and adapt _key() to the release filenames.')
    pd.DataFrame(rows).to_csv(out/'manifest.csv',index=False); return out
