"""Build DEAP FACE temporal features and merge them with the validated 4M cache.

Pipeline:
  DEAP face video -> MediaPipe face crop -> AffectNet MobileNetV2 -> frame embedding
  -> mean pooling in 1-second bins -> 60 trial tokens -> 10 tokens per 10-s CMF window.

The default AffectNet checkpoint is EmotiEffLib's public mobilenet_7.h5.  The
classifier is never trained on DEAP; the deepest vector layer before its
7-class head is used as a frozen visual representation.

IMPORTANT: run this in a separate face environment (see requirements-face.txt)
so TensorFlow/Keras dependencies cannot disturb the working PyTorch CMF env.
"""
from __future__ import annotations
import argparse, json, re, shutil, urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MODEL_URL = "https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/main/models/affectnet_emotions/mobilenet_7.h5"


def download_model(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    print("Downloading AffectNet MobileNetV2:", MODEL_URL)
    urllib.request.urlretrieve(MODEL_URL, path)


def build_embedder(model_path: Path):
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    # Pick the deepest non-classifier vector representation. This avoids
    # depending on a fragile hard-coded layer name in the external checkpoint.
    chosen = None
    for layer in reversed(model.layers[:-1]):
        try:
            shape = tuple(layer.output.shape)
        except Exception:
            continue
        if len(shape) == 2 and shape[-1] is not None and int(shape[-1]) >= 64:
            chosen = layer
            break
    if chosen is None:
        raise RuntimeError("Could not identify a vector feature layer before the AffectNet classifier.")
    embedder = tf.keras.Model(model.input, chosen.output)
    input_shape = tuple(int(x) if x is not None else -1 for x in model.input_shape)
    print(f"AffectNet model: input={model.input_shape}; feature_layer={chosen.name}; feature_dim={chosen.output.shape[-1]}")
    return embedder, input_shape, chosen.name


def make_detector(conf=0.5):
    import mediapipe as mp
    return mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=conf)


def crop_face(frame_bgr, detector, margin=0.18):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)
    if not result.detections:
        return None
    # DEAP has one participant; use the most confident detection.
    det = max(result.detections, key=lambda d: float(d.score[0]))
    box = det.location_data.relative_bounding_box
    h, w = rgb.shape[:2]
    x0, y0 = box.xmin*w, box.ymin*h
    x1, y1 = (box.xmin+box.width)*w, (box.ymin+box.height)*h
    bw, bh = x1-x0, y1-y0
    x0=max(0,int(x0-margin*bw)); x1=min(w,int(x1+margin*bw))
    y0=max(0,int(y0-margin*bh)); y1=min(h,int(y1+margin*bh))
    if x1 <= x0 or y1 <= y0:
        return None
    return rgb[y0:y1, x0:x1]


def prep_faces(crops, input_shape):
    # EmotiEffLib AffectNet checkpoint: keep RGB float32 on original 0..255 scale.\n    # Verified on identical MediaPipe crops: normalized inputs collapse the 256-D features.
    h = input_shape[1] if len(input_shape) == 4 and input_shape[1] > 0 else 224
    w = input_shape[2] if len(input_shape) == 4 and input_shape[2] > 0 else 224
    arr=np.stack([cv2.resize(x,(w,h),interpolation=cv2.INTER_AREA) for x in crops]).astype(np.float32)
    return arr


def fill_missing(tokens, valid):
    if not np.any(valid):
        raise RuntimeError("No faces detected in this trial.")
    idx=np.arange(len(tokens)); good=idx[valid]
    for i in idx[~valid]:
        tokens[i]=tokens[good[np.argmin(np.abs(good-i))]]
    return tokens


def extract_trial(video_path, embedder, input_shape, detector, sample_fps=10.0, batch_size=64):
    cap=cv2.VideoCapture(str(video_path))
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    nframes=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or nframes <= 0:
        cap.release(); raise RuntimeError(f"Unreadable video: {video_path}")
    duration=nframes/fps
    step=max(1,int(round(fps/sample_fps)))
    crops=[]; secs=[]; frame_idx=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if frame_idx % step == 0:
            c=crop_face(frame,detector)
            if c is not None:
                crops.append(c); secs.append(min(59,int(frame_idx/fps)))
        frame_idx += 1
    cap.release()
    if not crops:
        raise RuntimeError(f"No faces detected: {video_path}")
    all_f=[]
    for s in range(0,len(crops),batch_size):
        x=prep_faces(crops[s:s+batch_size],input_shape)
        z=np.asarray(embedder.predict(x,verbose=0),dtype=np.float32)
        all_f.append(z.reshape(z.shape[0],-1))
    feats=np.concatenate(all_f,axis=0)
    d=feats.shape[1]; tokens=np.zeros((60,d),np.float32); counts=np.zeros(60,np.int32)
    for z,sec in zip(feats,secs):
        tokens[sec]+=z; counts[sec]+=1
    valid=counts>0
    tokens[valid]/=counts[valid,None]
    tokens=fill_missing(tokens,valid)
    return tokens, {"video_fps":fps,"duration_s":duration,"sample_fps":sample_fps,
                    "sampled_frames":int((nframes+step-1)//step),"detected_frames":len(crops),
                    "detection_rate":float(len(crops)/max(1,(nframes+step-1)//step))}


def video_for(video_root: Path, subject: str, trial: int):
    # Official DEAP convention is sXX/sXX_trialYY.avi. Accept optional underscore
    # variants and recursive layouts to make the script robust to extraction layout.
    candidates=[
        video_root/subject/f"{subject}_trial{trial:02d}.avi",
        video_root/subject/f"{subject}_trial_{trial:02d}.avi",
        video_root/subject/f"{subject}_trial{trial}.avi",
    ]
    for p in candidates:
        if p.exists(): return p
    pat=re.compile(rf"^{re.escape(subject)}_trial_?0*{trial}\.avi$",re.I)
    for p in (video_root/subject).glob("*.avi") if (video_root/subject).exists() else video_root.rglob("*.avi"):
        if pat.match(p.name): return p
    return None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-cache",default="data/features/deap_external")
    ap.add_argument("--video-root",required=True)
    ap.add_argument("--output",default="data/features/deap_external_5m")
    ap.add_argument("--trial-cache",default="data/features/deap_face_trials")
    ap.add_argument("--model",default="weights/mobilenet_7_affectnet.h5")
    ap.add_argument("--sample-fps",type=float,default=10.0)
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--min-detection-rate",type=float,default=0.50)
    ap.add_argument("--limit-trials",type=int,default=None)
    ap.add_argument("--overwrite",action="store_true")
    a=ap.parse_args()

    src=Path(a.source_cache); out=Path(a.output); tc=Path(a.trial_cache); vr=Path(a.video_root)
    model_path=Path(a.model); download_model(model_path)
    out.mkdir(parents=True,exist_ok=True); tc.mkdir(parents=True,exist_ok=True)
    embedder,input_shape,feature_layer=build_embedder(model_path)
    detector=make_detector()

    manifest=pd.read_csv(src/"manifest.csv")
    # Only samples with official face-video subjects are retained.
    subjects=sorted([s for s in manifest.subject.unique() if int(str(s)[1:]) <= 22])
    trial_stats={}; processed_trials=0
    for subject in subjects:
        sub=manifest[manifest.subject==subject]
        trials=sorted({int(json.loads(str(np.load(src/f,allow_pickle=False)["meta_json"]))["trial"]) for f in sub.file})
        for trial in trials:
            vp=video_for(vr,subject,trial)
            if vp is None:
                print(f"MISSING video {subject} trial {trial:02d}; samples will be excluded.")
                continue
            dst=tc/f"{subject}_trial{trial:02d}.npz"
            if dst.exists() and not a.overwrite:
                z=np.load(dst,allow_pickle=False); tokens=z["tokens"]; stats=json.loads(str(z["stats_json"]))
            else:
                print(f"FACE {subject} trial {trial:02d}: {vp}")
                try:
                    tokens,stats=extract_trial(vp,embedder,input_shape,detector,a.sample_fps,a.batch_size)
                except RuntimeError as e:
                    print("  SKIP:",e); continue
                np.savez_compressed(dst,tokens=tokens,stats_json=np.asarray(json.dumps(stats)))
            trial_stats[f"{subject}_trial{trial:02d}"]=stats
            print(f"  tokens={tokens.shape}; detection={100*stats['detection_rate']:.1f}%")
            processed_trials += 1
            if a.limit_trials and processed_trials >= a.limit_trials: break
        if a.limit_trials and processed_trials >= a.limit_trials: break

    detector.close()
    # Merge FACE into the already validated external 4M cache. Each physiological
    # 10-s window maps exactly to ten 1-s visual tokens.
    rows=[]; missing=0
    for _,row in manifest.iterrows():
        subject=str(row["subject"])
        if subject not in subjects: continue
        raw=np.load(src/row["file"],allow_pickle=False)
        meta=json.loads(str(raw["meta_json"])); trial=int(meta["trial"])
        tf=tc/f"{subject}_trial{trial:02d}.npz"
        if not tf.exists():
            missing += 1; continue
        face=np.load(tf,allow_pickle=False)["tokens"]
        start=int(round(float(meta["window_start_s"])))
        vis=face[start:start+10]
        if vis.shape[0] != 10:
            missing += 1; continue
        payload={k:raw[k] for k in raw.files}
        payload["x__face"]=vis.astype(np.float32)
        np.savez_compressed(out/row["file"],**payload)
        rows.append(row.to_dict())

    if not rows:
        raise RuntimeError("No 5M samples created. First run --limit-trials 1 and inspect video/model compatibility.")
    pd.DataFrame(rows).to_csv(out/"manifest.csv",index=False)
    first=np.load(out/rows[0]["file"],allow_pickle=False)
    face_shape=list(first["x__face"].shape)
    spec={"source_cache":str(src),"video_root":str(vr),"samples":len(rows),"subjects":len(set(r["subject"] for r in rows)),
          "face_model":"sb-ai-lab/EmotiEffLib mobilenet_7.h5 (AffectNet 7-class)",
          "face_model_url":MODEL_URL,"feature_layer":feature_layer,"face_shape":face_shape,
          "temporal_pooling":"frame embeddings -> mean per 1-second bin -> 10 consecutive tokens per 10-s window",
          "sample_fps":a.sample_fps,"input_preprocessing":"RGB float32 raw 0..255",
          "missing_windows_excluded":missing,"trial_stats":trial_stats}
    (out/"feature_spec.json").write_text(json.dumps(spec,indent=2),encoding="utf-8")
    print(f"DONE: {len(rows)} 5M windows, {len(set(r['subject'] for r in rows))} subjects, face={face_shape}; excluded={missing}")


if __name__=="__main__":
    main()
