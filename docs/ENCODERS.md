# Pretrained unimodal encoders

DirectedContrastiveFusion is **pretrained-first and encoder-agnostic**. The intended publication setup uses the strongest appropriate unimodal representation learner for each modality, then projects every encoder output to the shared `d_model` space before directed selection and cross-modal attention.

## Common contract

Every encoder must return `[batch, tokens, d_model]`. This is the only representation contract seen by the fusion module.

```text
raw modality -> pretrained unimodal encoder -> projection to d_model
             -> directed selector -> sparse cross-attention -> task head
```

## Registry types

- `generic` / `auto`: lightweight fallback used for ablations and smoke tests.
- `timeseries`: lightweight temporal feature projection.
- `image_cnn`: lightweight frame-wise CNN.
- `precomputed`: consumes token/feature embeddings extracted offline by any specialist pretrained model.
- `torchvision`: pretrained torchvision image backbone, applied frame-wise.
- `timm`: pretrained timm image/ViT backbone, applied frame-wise.
- `huggingface`: generic Hugging Face `AutoModel` wrapper for compatible tensor inputs.

The `precomputed` path is deliberately first-class. It lets the framework use specialist models such as pretrained EEG, IMU, skeleton, audio or video encoders without coupling their preprocessing/dependencies to the fusion implementation.

## Freezing

Pretrained backbones can be frozen with `freeze: true`. Recommended experimental stages are: (1) frozen-backbone fusion training for clean selector ablations; (2) partial/full fine-tuning where compute/data permit. Report the setting explicitly.

## Adding a specialist encoder

Add an `nn.Module` wrapper to `src/cmf/models/encoders.py` whose `forward` returns `[B,T,d_model]`, then register a new `type` in `build_encoder`. No changes to `fusion.py` are required.

## Scientific comparison

For the final evaluation, keep the same unimodal encoders across `late`, `full`, `random_topk`, `similarity_topk`, `directed_topk` and `contrastive_topk`. This isolates the effect of interaction selection/fusion rather than backbone quality.
