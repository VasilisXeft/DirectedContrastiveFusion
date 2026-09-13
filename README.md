# DirectedContrastiveFusion

PyTorch research framework for **trainable, sparse, directed and sample-dependent cross-modal interaction selection**. The framework is **pretrained-first and encoder-agnostic**: each modality can use its strongest appropriate unimodal pretrained encoder, while the proposed method learns which directional cross-modal interactions are worth computing.

## Architecture

```text
modality x_m
    -> modality-specific pretrained encoder E_m
    -> projection adapter to shared d_model
    -> directed sample-dependent selector
    -> sparse cross-modal attention
    -> multimodal task head
```

For modalities `A` and `B`, `A -> B` and `B -> A` are distinct interactions. Separate source and target projections make directional scores asymmetric. For `src -> tgt`, target tokens query source tokens and source information enriches the target representation.

`directed_topk` and `contrastive_topk` use hard top-k routing in the forward pass with a straight-through Gumbel/softmax estimator during training. `contrastive_topk` additionally uses symmetric InfoNCE between modality representations:

```text
L = L_task + lambda_c * L_contrastive
```

No diversity/entropy penalty is enabled by default: deterministic routing is allowed when it is genuinely optimal.

## Pretrained unimodal encoders

Publication experiments are intended to exploit existing unimodal knowledge rather than relearn every modality from scratch. Encoder choice is configured independently for each modality in YAML. Built-in registry paths include `torchvision`, `timm`, `huggingface`, `precomputed`, and lightweight generic fallbacks. Every encoder is adapted to the common token contract `[B, T, d_model]`.

Specialist encoders (e.g. EEG, IMU, skeleton, audio, video) can either be wrapped in the registry or used offline with `type: precomputed`. See `docs/ENCODERS.md`. Generic encoders are retained primarily for smoke tests and backbone-controlled ablations.

## Fusion modes

- `late` — pooled late fusion, no cross-modal attention
- `full` — exhaustive directed pairwise cross-attention
- `random_topk` — random `k` outgoing edges per source
- `similarity_topk` — cosine-similarity sparse baseline
- `directed_topk` — trainable asymmetric selector, task loss only
- `contrastive_topk` — same selector + contrastive objective

With `M` modalities, full directed fusion has `M(M-1)` interactions; top-k routing keeps at most `M*k`.

## Dataset-agnostic design

Dataset peculiarities live in `src/cmf/datasets/`; the encoder/selector/fusion/training code is shared. Included adapters/configurations cover generic preprocessed data, synthetic smoke tests, UTD-MHAD, MAHNOB-HCI, MMAct and TotalCapture. For a new dataset, align/window the streams and convert them to the common cache format described in `docs/CUSTOM_DATASETS.md`.

## Installation

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

For CUDA, install the appropriate PyTorch build first. Pretrained registry support uses torchvision, timm and Hugging Face Transformers.

## Smoke test

```bash
python scripts/preprocess.py --config configs/synthetic.yaml
python scripts/train.py --config configs/synthetic.yaml
python scripts/evaluate.py --checkpoint runs/synthetic/contrastive_topk/best.pt --cache data/processed/synthetic
python scripts/run_baselines.py --config configs/synthetic.yaml
```

## New datasets / encoder configuration

Copy `configs/generic.yaml`, set the cache/task, and define an encoder per modality. Example:

```yaml
model:
  d_model: 128
  encoders:
    rgb:
      type: timm
      name: vit_base_patch16_224
      pretrained: true
    depth:
      type: timm
      name: resnet18
      pretrained: true
      in_chans: 1
    skeleton:
      type: precomputed
      feature_dim: 256
    inertial:
      type: precomputed
      feature_dim: 256
```

All fusion baselines should use the **same unimodal encoders** so differences measure the proposed routing/fusion method rather than backbone quality.

## Missing/noisy modalities

Samples carry modality-presence masks; training can additionally simulate missing streams through `training.modality_dropout`. Learned reliability gates influence fused features and directional scores. Controlled corruption is a diagnostic of quality-adaptive routing, not by itself proof of selector failure.

## Evaluation

Classification and fixed-shape regression are supported. Subject-independent evaluation is recommended. `scripts/run_loso_utd.py` is the current LOSO example. The model exposes per-sample `hard_masks`, directional scores, reliability values and selected-pair counts for routing analysis.

## Dataset notes

For UTD-MHAD, the intended publication experiment is RGB + Depth + Skeleton + Inertial. The accelerometer/gyroscope split used previously is architecture validation only. MAHNOB-HCI provides heterogeneous affective modalities; MMAct is the intended scalability benchmark; TotalCapture tests generalization to structured pose regression.

## Repository structure

```text
configs/                    experiment + encoder configurations
scripts/                    CLI entry points
src/cmf/models/encoders.py  pretrained encoder registry + adapters
src/cmf/models/fusion.py    directed sparse fusion
src/cmf/datasets/           dataset adapters/common cache
src/cmf/train.py            dataset-agnostic training
docs/ENCODERS.md            pretrained encoder contract
docs/CUSTOM_DATASETS.md     new-dataset specification
examples/                   adapter templates
tests/                      smoke tests
```

## Research status

Active research code. Existing UTD-MHAD 3-stream results are architecture-validation experiments, not final benchmark claims. Publication experiments should use genuinely heterogeneous modalities, strong unimodal encoders, identical backbones across fusion baselines, multiple seeds/folds and controlled reliability diagnostics.
