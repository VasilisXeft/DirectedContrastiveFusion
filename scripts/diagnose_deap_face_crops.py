"""Small apples-to-apples AffectNet preprocessing check on MediaPipe face crops.

Uses the same detector/crop geometry as cache_deap_face_features.py and compares
three input scalings on identical crops. No cache is modified.
"""
import argparse
from pathlib import Path
import cv2, numpy as np, tensorflow as tf, mediapipe as mp

EMOTIONS=["Anger","Disgust","Fear","Happiness","Sadness","Surprise","Neutral"]

def crop_face(frame,detector,margin=.18):
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res=detector.process(rgb)
    if not res.detections: return None
    d=max(res.detections,key=lambda x: float(x.score[0]))
    box=d.location_data.relative_bounding_box
    h,w=frame.shape[:2]
    x1=box.xmin*w; y1=box.ymin*h; x2=(box.xmin+box.width)*w; y2=(box.ymin+box.height)*h
    bw=x2-x1; bh=y2-y1
    x1=max(0,int(x1-margin*bw)); x2=min(w,int(x2+margin*bw))
    y1=max(0,int(y1-margin*bh)); y2=min(h,int(y2+margin*bh))
    if x2<=x1 or y2<=y1: return None
    return rgb[y1:y2,x1:x2]

def prep(crop,mode):
    x=cv2.resize(crop,(224,224)).astype(np.float32)
    if mode=="minus1_1": x=x/127.5-1.
    elif mode=="zero1": x=x/255.
    return x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video-root",required=True)
    ap.add_argument("--model",default="weights/mobilenet_7_affectnet.h5")
    ap.add_argument("--subjects",nargs="+",default=["s01","s04","s07","s10","s15","s20"])
    ap.add_argument("--trials",type=int,nargs="+",default=[1,10,20,30])
    ap.add_argument("--frames-per-video",type=int,default=5)
    a=ap.parse_args()
    model=tf.keras.models.load_model(a.model,compile=False)
    feat=tf.keras.Model(model.input,model.get_layer("feats").output)
    crops=[]
    with mp.solutions.face_detection.FaceDetection(model_selection=0,min_detection_confidence=.5) as det:
        for s in a.subjects:
            for t in a.trials:
                p=Path(a.video_root)/s/f"{s}_trial{t:02d}.avi"
                if not p.exists(): continue
                cap=cv2.VideoCapture(str(p)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for idx in np.linspace(0,max(0,n-1),a.frames_per_video,dtype=int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx)); ok,frame=cap.read()
                    if ok:
                        c=crop_face(frame,det)
                        if c is not None: crops.append(c)
                cap.release()
    print(f"Identical MediaPipe crops collected: {len(crops)}")
    for mode in ("minus1_1","zero1","raw255"):
        X=np.stack([prep(c,mode) for c in crops])
        P=model.predict(X,batch_size=64,verbose=0)
        if P.min()<0 or not np.allclose(P.sum(1),1,atol=.05): P=tf.nn.softmax(P,axis=1).numpy()
        F=feat.predict(X,batch_size=64,verbose=0)
        pred=P.argmax(1); counts=np.bincount(pred,minlength=P.shape[1])
        fn=F/(np.linalg.norm(F,axis=1,keepdims=True)+1e-8); sim=fn@fn.T; tri=sim[np.triu_indices(len(fn),1)]
        print("\n==",mode,"==")
        print("class counts:",counts.tolist(),EMOTIONS)
        print("mean probs:",np.round(P.mean(0),3).tolist())
        print("mean max confidence:",round(float(P.max(1).mean()),4))
        print("feature mean-dim std:",float(F.std(0).mean()))
        print("near-constant dims std<1e-6:",int((F.std(0)<1e-6).sum()),"/",F.shape[1])
        print("feature cosine mean/std/min/max:",*[round(float(v),6) for v in (tri.mean(),tri.std(),tri.min(),tri.max())])

if __name__=="__main__": main()
