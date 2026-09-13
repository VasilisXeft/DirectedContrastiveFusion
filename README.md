# DirectedContrastiveFusion

PyTorch research framework for **trainable, sparse, directed and sample-dependent cross-modal interaction selection**. It is designed to test whether a multimodal model can learn which directional cross-modal interactions are worth computing instead of executing the complete directed interaction graph.

## Core method

For modalities `A` and `B`, `A -> B` and `B -> A` are distinct interactions. The learned selector uses separate source and target projections, so the scores are asymmetric. For an edge `src -> tgt`, target tokens query source tokens and source information enriches the target representation.

`directed_topk` and `contrastive_topk` use hard top-k routing in the forward pass with a straight-through Gumbel/softmax estimator during training. Therefore the task loss can optimize the selector end-to-end. `contrastive_topk` additionally uses symmetric InfoNCE between modality representations.

```text
L = L_task + lambda_c * L_contrastive
```

No diversity/entropy penalty is enabled by default: a deterministic graph is allowed when it is genuinely optimal for the data.

## Implemented fusion modes

- `late` — pooled late fusion, no cross-modal attention
- `full` — exhaustive directed pairwise cross-attention
- `random_topk` — random `k` outgoing edges per source modality
- `similarity_topk` — cosine-similarity sparse baseline
- `directed_topk` — trainable asymmetric directed selector, task loss only
- `contrastive_topk` — same trainable selector + contrastive objective

With `M` modalities, full directed fusion has `M(M-1)` interactions; top-k routing keeps at most `M*k`.

## Dataset-agnostic design

The selector, fusion, training, evaluation and benchmarking code do not depend on a specific dataset. Raw dataset peculiarities live only in `src/cmf/datasets/`.

Included adapters/configurations:

- generic preprocessed multimodal datasets
- synthetic smoke-test dataset
- UTD-MHAD
- MAHNOB-HCI
- MMAct
- TotalCapture

For a new dataset, convert each aligned sample once to the common `.npz` cache format. See [`docs/CUSTOM_DATASETS.md`](docs/CUSTOM_DATASETS.md) and [`examples/custom_dataset_adapter.py`](examples/custom_dataset_adapter.py).

## Installation

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

For CUDA, install the PyTorch build appropriate for the machine first, then install the package.

## Smoke test

```bash
python scripts/preprocess.py --config configs/synthetic.yaml
python scripts/train.py --config configs/synthetic.yaml
python scripts/evaluate.py --checkpoint runs/synthetic/contrastive_topk/best.pt --cache data/processed/synthetic
python scripts/benchmark.py --checkpoint runs/synthetic/contrastive_topk/best.pt --cache data/processed/synthetic
```

Run all baselines/ablations:

```bash
python scripts/run_baselines.py --config configs/synthetic.yaml
```

## Any new dataset

1. Align/window/resample the raw modalities into fixed per-modality shapes.
2. Save samples using `cmf.utils.io.save_npz` and create `manifest.csv`.
3. Copy `configs/generic.yaml` and set the task/cache path.
4. Validate the cache:

```bash
python scripts/preprocess.py --config configs/generic.yaml
```

5. Train any fusion mode:

```bash
python scripts/train.py --config configs/generic.yaml --mode full
python scripts/train.py --config configs/generic.yaml --mode directed_topk
python scripts/train.py --config configs/generic.yaml --mode contrastive_topk
```

## Missing/noisy modalities

Samples carry a modality-presence mask. Naturally missing streams are listed in `meta["missing_modalities"]`; training can additionally simulate missing inputs through `training.modality_dropout`. Reliability gates are learned per modality and influence both fused features and directional selection scores.

Controlled corruption should be treated as a diagnostic of quality-adaptive routing, not as evidence of selector failure by itself.

## Evaluation protocol

The framework supports classification and fixed-shape regression. Subject-independent splits are recommended. Dataset-specific LOSO utilities can be built on the same common cache format; `scripts/run_loso_utd.py` is the current UTD-MHAD example.

For routing analysis, the model exposes per-sample `hard_masks`, directional scores, reliability values and selected-pair counts through its auxiliary output.

## Existing dataset notes

### UTD-MHAD
Use RGB + Depth + Skeleton + Inertial for the publication-quality heterogeneous-modality experiment. The `utd_mhad_subset.yaml` configuration that splits the 6-D inertial stream into accelerometer and gyroscope exists only for architecture validation; accelerometer and gyroscope from the same unit should not be presented as fully independent sensing modalities.

### MAHNOB-HCI
Useful for heterogeneous affective HCI and reliability/missing-modality experiments. Adjust protected-dataset paths to your local release.

### MMAct
The main intended scalability benchmark because it exposes many sensing streams. Different distributed layouts may require adapting only the raw filename/key matching in its preprocessor.

### TotalCapture
Useful for testing whether the selector generalizes beyond classification to structured pose regression. Publication-quality pose metrics require the exact synchronization/calibration protocol chosen for the experiment.

## Repository structure

```text
configs/                    experiment configurations
scripts/                    CLI entry points
src/cmf/models/             encoders + directed sparse fusion
src/cmf/datasets/           common cache interface + raw adapters
src/cmf/train.py            dataset-agnostic training
src/cmf/evaluate.py         evaluation
src/cmf/benchmark.py        latency/memory/interaction benchmarking
docs/CUSTOM_DATASETS.md     new-dataset specification
examples/                   adapter template
tests/                      smoke tests
```

## Research status

This is active research code. Existing UTD-MHAD 3-stream experiments are architecture-validation results, not final benchmark claims. The next stronger validation should use genuinely independent modalities and multiple datasets before drawing conclusions about routing behavior.
