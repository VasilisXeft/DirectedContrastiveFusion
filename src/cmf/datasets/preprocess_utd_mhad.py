from pathlib import Path
import re
import numpy as np, pandas as pd
from scipy.io import loadmat
from cmf.datasets.common import resample_numeric, sample_video_frames, sample_image_volume
from cmf.utils.io import ensure_dir, save_npz

PAT=re.compile(r"a(\d+)_s(\d+)_t(\d+)",re.I)
def key(p):
    m=PAT.search(p.name); return tuple(map(int,m.groups())) if m else None

def mat_array(p):
    d=loadmat(p)
    arr=[v for k,v in d.items() if not k.startswith('__') and isinstance(v,np.ndarray)]
    return max(arr,key=lambda x:x.size) if arr else None

def preprocess_utd_mhad(cfg):
    dcfg=cfg['dataset']; root=Path(dcfg['raw_dir']); out=ensure_dir(dcfg['cache_dir']); samples=ensure_dir(out/'samples')
    files={};
    for p in root.rglob('*'):
        if not p.is_file() or key(p) is None: continue
        low=str(p).lower(); mod='rgb' if p.suffix.lower() in {'.avi','.mp4'} or 'color' in low or 'rgb' in low else 'depth' if 'depth' in low else 'skeleton' if 'skel' in low else 'inertial' if 'inertial' in low or 'imu' in low else None
        if mod: files.setdefault(key(p),{})[mod]=p
    rows=[]; train=set(dcfg.get('train_subjects',[1,3,5,7])); val=set(dcfg.get('val_subjects',[2])); include=set(dcfg.get('include_modalities',['rgb','depth','skeleton','inertial']))
    if dcfg.get('split_inertial',False):
        include.discard('inertial'); include.update(['accel','gyro'])
    for k,fs in sorted(files.items()):
        a,s,t=k; mods={}
        try:
            if 'rgb' in include and 'rgb' in fs: mods['rgb']=sample_video_frames(fs['rgb'],dcfg.get('video_frames',16),dcfg.get('image_size',112))
            if 'depth' in include and 'depth' in fs: mods['depth']=sample_image_volume(mat_array(fs['depth']),dcfg.get('video_frames',16),dcfg.get('image_size',112))
            if 'skeleton' in include and 'skeleton' in fs:
                x=mat_array(fs['skeleton']); x=np.moveaxis(x,-1,0).reshape(x.shape[-1],-1) if x.ndim==3 and x.shape[-1]>10 else x.reshape(x.shape[0],-1); mods['skeleton']=resample_numeric(x,dcfg.get('length',64))
            if ('inertial' in include or 'accel' in include or 'gyro' in include) and 'inertial' in fs:
                x=mat_array(fs['inertial']); x=x if x.shape[0]>=x.shape[-1] else x.T; x=resample_numeric(x,dcfg.get('length',64))
                if 'inertial' in include: mods['inertial']=x
                if 'accel' in include: mods['accel']=x[:,:3]
                if 'gyro' in include: mods['gyro']=x[:,3:6]
        except Exception as e: continue
        if not include.issubset(mods): continue
        split='train' if s in train else 'val' if s in val else 'test'; rel=f'samples/{len(rows):05d}.npz'; save_npz(out/rel,mods,np.array(a-1,dtype=np.int64),{'subject':s,'trial':t,'action':a}); rows.append({'sample_id':len(rows),'subject':s,'label':a-1,'split':split,'file':rel})
    if not rows: raise RuntimeError('No UTD-MHAD trials were matched. Check raw_dir and filenames.')
    pd.DataFrame(rows).to_csv(out/'manifest.csv',index=False); return out
