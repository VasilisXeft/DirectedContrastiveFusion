# DirectedContrastiveFusion

A research framework for **trainable, sparse, directed and sample-dependent cross-modal interaction selection** in multimodal learning.

## Core idea

For modalities `MOD_i` and `MOD_j`, the interactions `MOD_i -> MOD_j` and `MOD_j -> MOD_i` are treated as distinct directed edges. A trainable selector learns directional scores using asymmetric source/target projections. During training, sparse top-k routing is optimized end-to-end with the downstream task; at inference, hard top-k selection allows non-selected cross-modal interactions to be skipped.

The main objective is:

```text
L = L_task + lambda_c * L_contrastive
```

Additional reliability or regularization terms can be evaluated separately without forcing routing diversity.

## Fusion modes

- `late`: no cross-modal attention
- `full`: all directed cross-modal interactions
- `random_topk`: random sparse directed interactions
- `similarity_topk`: similarity-based sparse baseline
- `directed_topk`: trainable directed selector, task loss only
- `contrastive_topk`: trainable directed selector + contrastive objective

## Evaluation

The framework targets subject-independent evaluation and records per-sample selected directed interaction graphs. Diagnostics include edge-selection frequencies, graph diversity/entropy and controlled modality corruption.

## Datasets

Planned/implemented dataset adapters include:

- UTD-MHAD
- MAHNOB-HCI
- MMAct
- TotalCapture

The initial UTD-MHAD proof-of-concept uses Skeleton plus Accelerometer/Gyroscope streams. Splitting the inertial stream is only an architecture-validation setup; the intended UTD-MHAD experiment uses genuinely heterogeneous RGB, Depth, Skeleton and Inertial modalities.

## Status

Research code under active development. Current results should be treated as preliminary architecture-validation experiments rather than final benchmark claims.
