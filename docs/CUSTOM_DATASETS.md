# Using a new dataset

The model is dataset-agnostic after preprocessing. New datasets should be converted once into the common cache format.

## Cache layout

```text
data/processed/my_dataset/
  manifest.csv
  samples/
    000000.npz
    000001.npz
    ...
```

`manifest.csv` must contain at least:

```text
file,split
samples/000000.npz,train
samples/000001.npz,val
samples/000002.npz,test
```

Recommended additional columns are `sample_id`, `subject`, and `label`.

Create each sample with `cmf.utils.io.save_npz(path, modalities, target, meta)`.

`modalities` is a dictionary with arbitrary modality names. The framework currently supports:

- numeric/time-series sequences: `[T, F]`
- image/video sequences: `[T, H, W, C]`

After preprocessing, a modality must have a fixed shape across samples so that batches can be stacked. Different modalities may have different sequence lengths and feature dimensions.

## Missing modalities

Keep the modality key present, supply a correctly shaped zero array, and add its name to:

```python
meta = {"missing_modalities": ["eeg"]}
```

The model receives a presence mask and will gate invalid directed edges.

## Classification

Use a scalar integer target starting at zero. Set `dataset.task: classification`. `num_classes` can be provided explicitly in YAML or inferred from `manifest.csv` when a `label` column is present.

## Regression

Use any fixed-shape numeric target, set `dataset.task: regression`, and remove `num_classes`. The framework flattens the target internally for the prediction head.

## Adding a raw-data adapter

For a reusable public dataset, copy `examples/custom_dataset_adapter.py` into `src/cmf/datasets/preprocess_<name>.py`, implement the raw parsing/alignment, and register it in `src/cmf/datasets/registry.py`. Nothing in the selector/fusion/training code should require modification.
