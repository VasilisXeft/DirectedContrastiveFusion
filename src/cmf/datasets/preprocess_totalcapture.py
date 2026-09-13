from pathlib import Path
import re
import numpy as np, pandas as pd
from cmf.datasets.common import resample_numeric, sample_video_frames
from cmf.utils.io import ensure_dir, save_npz


def _read_numeric_text(p):
    rows=[]
    for line in p.read_text(errors='ignore').splitlines():
        vals=[]
        for tok in re.split(r'[\s,;]+',line.strip()):
            try: vals.append(float(tok))
            except: pass
        if vals: rows.append(vals)
    if not rows: return None
    w=min(map(len,rows)); return np.asarray([r[:w] for r in rows],dtype=np.float32)

def preprocess_totalcapture(cfg):
    dcfg=cfg['dataset']; root=Path(dcfg['raw_dir']); out=ensure_dir(dcfg['cache_dir']); samples=ensure_dir(out/'samples'); length=dcfg.get('length',64); rows=[]
    for posefile in root.rglob('gt_skel_gbl_pos.txt'):
        subject=next((int(x[1:]) for x in posefile.parts if re.fullmatch(r'S\d+',x,re.I)),0); seq=posefile.parent.name; pose=_read_numeric_text(posefile)
        if pose is None: continue
        pose=resample_numeric(pose,length); parent=posefile.parents[1] if len(posefile.parents)>1 else posefile.parent; candidates=list(parent.rglob(f'*{seq}*.sensors'))+list(parent.rglob(f'*{seq}*.txt')); imu=None
        for c in candidates:
            if c==posefile: continue
            a=_read_numeric_text(c)
            if a is not None and a.shape[1]>=6: imu=resample_numeric(a,length); break
        vids=list(parent.rglob(f'*{seq}*cam{dcfg.get("camera",1)}*.mp4'))+list(parent.rglob(f'*{seq}*cam{dcfg.get("camera",1)}*.avi'))
        if imu is None or not vids: continue
        try: rgb=sample_video_frames(vids[0],dcfg.get('video_frames',16),dcfg.get('image_size',112))
        except Exception: continue
        mods={'imu':imu,'rgb':rgb}; target=pose.astype(np.float32); split='test' if subject==5 else 'val' if subject==4 else 'train'; rel=f'samples/{len(rows):06d}.npz'; save_npz(out/rel,mods,target,{'subject':subject,'sequence':seq}); rows.append({'sample_id':len(rows),'subject':subject,'label':-1,'split':split,'file':rel})
    if not rows: raise RuntimeError('No TotalCapture sequences parsed. Check raw layout; final experiments require exact official synchronization/calibration.')
    pd.DataFrame(rows).to_csv(out/'manifest.csv',index=False); return out
