from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np, pandas as pd
from cmf.datasets.common import resample_numeric, sample_video_frames
from cmf.utils.io import ensure_dir, save_npz


def _attr(root, keys):
    for k in keys:
        if k in root.attrib:
            try: return float(root.attrib[k])
            except: pass
    for el in root.iter():
        for k in keys:
            if k in el.attrib:
                try: return float(el.attrib[k])
                except: pass
    return None


def preprocess_mahnob(cfg):
    import mne
    dcfg=cfg['dataset']; root=Path(dcfg['raw_dir']); out=ensure_dir(dcfg['cache_dir']); samples=ensure_dir(out/'samples')
    sessions=[p.parent for p in root.rglob('session.xml')]; rows=[]; label_map={}
    for sess in sessions:
        xml=sess/'session.xml'; xr=ET.parse(xml).getroot(); target_name=dcfg.get('target','valence')
        val=_attr(xr,['feltVlnc','feltValence']) if target_name=='valence' else _attr(xr,['feltArsl','feltArousal'])
        if val is None: continue
        bins=dcfg.get(f'{target_name}_bins',[3,6]); y=int(np.digitize([val],bins)[0]); label_map.setdefault(y,y)
        bdfs=list(sess.glob('*.bdf'))
        if not bdfs: continue
        raw=mne.io.read_raw_bdf(bdfs[0],preload=True,verbose='ERROR'); names=raw.ch_names; data=raw.get_data().T
        def chans(keys): return [i for i,n in enumerate(names) if any(k.lower() in n.lower() for k in keys)]
        mods={}
        eeg_idx=list(range(min(32,data.shape[1]))); mods['eeg']=resample_numeric(data[:,eeg_idx],dcfg.get('length',256))
        for mod,keys in [('ecg',['EXG1','EXG2','EXG3','ECG']),('gsr',['GSR']),('temp',['Temp','TEMP'])]:
            idx=chans(keys)
            if idx: mods[mod]=resample_numeric(data[:,idx],dcfg.get('length',256))
        vids=list(sess.glob(dcfg.get('face_video_glob','*.avi')))
        if vids:
            try: mods['video']=sample_video_frames(vids[0],dcfg.get('video_frames',16),dcfg.get('image_size',112))
            except Exception: pass
        if len(mods)<2: continue
        sid=int(sess.name) if sess.name.isdigit() else len(rows); split='test' if sid%5==0 else 'val' if sid%5==1 else 'train'; rel=f'samples/{sid:05d}.npz'; save_npz(out/rel,mods,np.array(y,dtype=np.int64),{'session':sid,'raw_value':val}); rows.append({'sample_id':sid,'subject':sid,'label':y,'split':split,'file':rel})
    if not rows: raise RuntimeError('No MAHNOB-HCI sessions parsed. Check raw_dir/session layout and BDF installation.')
    pd.DataFrame(rows).to_csv(out/'manifest.csv',index=False); return out
