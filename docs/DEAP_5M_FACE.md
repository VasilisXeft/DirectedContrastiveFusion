# DEAP 5M FACE pipeline

The 5-modality benchmark adds facial affect to the validated frozen DEAP physiology cache.

Pipeline: **DEAP face video -> MediaPipe face detection/crop -> AffectNet MobileNetV2 -> frame embeddings -> 1-second mean pooling -> 10 FACE tokens per 10-second window**.

The default visual checkpoint is the public `mobilenet_7.h5` from `sb-ai-lab/EmotiEffLib`, trained for 7-class AffectNet facial-expression recognition. The extractor uses the deepest vector layer before the classifier as the frozen representation. No DEAP labels are used to train the visual encoder.

Use a separate environment so TensorFlow does not modify the validated PyTorch environment:

```powershell
py -3.10 -m venv .venv-face
.\.venv-face\Scripts\Activate.ps1
python -m pip install -r requirements-face.txt
python scripts/cache_deap_face_features.py --video-root "C:/Users/vxefteris/Desktop/D/MindSpaces/DEAP Dataset/face_video" --limit-trials 1
```

The first command is a sanity run. Confirm that it prints a feature dimension, `tokens=(60,D)`, a reasonable detection rate, and creates a 5M cache sample. Then run the complete extraction:

```powershell
python scripts/cache_deap_face_features.py --video-root "C:/Users/vxefteris/Desktop/D/MindSpaces/DEAP Dataset/face_video"
```

The script processes each trial only once, samples frames at 10 FPS by default, pools frame embeddings into 60 one-second tokens, and then reuses those tokens for the 11 overlapping 10-second windows. Trials/videos that are genuinely missing are excluded rather than imputed across trials.

After extraction, inspect `data/features/deap_external_5m/feature_spec.json`. If the reported FACE feature dimension differs from 1280, update `model.encoders.face.feature_dim` in `configs/deap_external_5m.yaml` to the reported value.

Then reactivate the normal CMF PyTorch environment and run:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/run_deap_external_5m.py --subject s22
python scripts/run_deap_external_5m.py --skip-existing
```

With five modalities there are 20 directed candidate edges. The primary sparse budgets are K=5/10/15 (25/50/75%). Keep the existing DEAP-4M results unchanged; for a strict modality-addition comparison, run a matched 4M control restricted to the exact subjects/trials retained by the 5M manifest.
