"""Diagnose cached DEAP FACE features and the AffectNet head.

Run in .venv-face. It checks cached embedding variance/uniqueness and samples
raw DEAP frames through the complete mobilenet_7.h5 emotion classifier.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, cv2, tensorflow as tf

EMOTIONS=["Anger","Disgust","Fear","Happiness","Sadness","Surprise","Neutral"]

def cache_stats(cache,max_windows=1000):
    man=pd.read_csv(Path(cache)/"manifest.csv")
    if len(man)>max_windows: man=man.sample(max_windows,random_state=42)
    xs=[]; means=[]
    for f in man.file:
        z=np.load(Path(cache)/f,allow_pickle=False); x=z["x__face"].astype(np.float32)
        xs.append(x); means.append(x.mean(0))
    x=np.concatenate(xs,0); w=np.stack(means)
    print("\n=== CACHED FEATURE DIAGNOSTICS ===")
    print("windows:",len(w),"tokens:",len(x),"dim:",x.shape[-1])
    print("global std:",float(x.std()),"mean feature std across tokens:",float(x.std(0).mean()))
    print("mean feature std across windows:",float(w.std(0).mean()))
    print("near-constant dims (std<1e-6):",int((x.std(0)<1e-6).sum()),"/",x.shape[1])
    # cosine similarity among window means
    q=w[:min(500,len(w))]; q=q/(np.linalg.norm(q,axis=1,keepdims=True)+1e-8)
    sim=q@q.T; tri=sim[np.triu_indices(len(q),1)]
    print("window cosine similarity mean/std/min/max:",*[float(v) for v in (tri.mean(),tri.std(),tri.min(),tri.max())])

def preprocess(img,mode):
    img=cv2.resize(img,(224,224)); rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)
    if mode=="minus1_1": return rgb/127.5-1.0
    if mode=="zero1": return rgb/255.0
    return rgb

def find_video(root,s,t):
    p=Path(root)/s/f"{s}_trial{t:02d}.avi"
    return p if p.exists() else None

def head_test(model,video_root,subjects=("s01","s04","s07"),trials=(1,10,20),frames_per=6):
    print("\n=== FULL AFFECTNET HEAD DIAGNOSTICS ===")
    for mode in ("minus1_1","zero1","raw255"):
        probs=[]; print("\npreprocess:",mode)
        for s in subjects:
            for t in trials:
                p=find_video(video_root,s,t)
                if not p: continue
                cap=cv2.VideoCapture(str(p)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for idx in np.linspace(0,max(n-1,0),frames_per,dtype=int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx)); ok,frame=cap.read()
                    if ok: probs.append(model.predict(preprocess(frame,mode)[None],verbose=0)[0])
                cap.release()
        p=np.asarray(probs)
        # Model output is expected probabilities; softmax only if needed.
        if p.size and (p.min()<0 or not np.allclose(p.sum(1),1,atol=.05)):
            p=tf.nn.softmax(p,axis=1).numpy()
        pred=p.argmax(1); counts=np.bincount(pred,minlength=p.shape[1])
        print("n=",len(p),"class counts=",counts.tolist())
        print("mean probs=",np.round(p.mean(0),3).tolist())
        print("mean max confidence=",round(float(p.max(1).mean()),4))
        if p.shape[1]==7: print("labels=",EMOTIONS)

def main():
    a=argparse.ArgumentParser(); a.add_argument("--cache",default="data/features/deap_external_5m"); a.add_argument("--video-root",required=True); a.add_argument("--model",default="weights/mobilenet_7_affectnet.h5"); a.add_argument("--max-windows",type=int,default=1000); args=a.parse_args()
    cache_stats(args.cache,args.max_windows)
    model=tf.keras.models.load_model(args.model,compile=False)
    print("\nmodel input/output:",model.input_shape,model.output_shape)
    head_test(model,args.video_root)

if __name__=="__main__": main()
