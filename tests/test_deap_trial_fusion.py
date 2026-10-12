"""Regression tests for DEAP trial grouping and development-only neural benchmark."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cmf.datasets.common import collate_multimodal
from cmf.datasets.deap_trials import DEAPTrialDataset, MODALITY_ORDER, group_trials
from cmf.feature_normalization import FeatureStandardizer
from cmf.utils.io import save_npz
from scripts.run_deap_trial_fusion import (
    build_neural_model, metric_summary, ridge_predict, split_development,
)


class TestDEAPTrialBenchmark(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rows = []
        self.rng = np.random.default_rng(42)
        for subj, split in [("s01", "train"), ("s02", "val")]:
            for trial in range(2):
                for index in range(11):
                    start = index * 640
                    filename = f"{subj}_trial{trial:02d}_start{start:06d}.npz"
                    label = trial
                    data = {}
                    for m in MODALITY_ORDER:
                        # Exercise independent modalities with contrasting scales.
                        scale = 1e-4 if m == "eeg" else 100 if m == "face" else 2
                        x = self.rng.normal(loc=label, scale=scale, size=(4, 8))
                        data[m] = x.astype(np.float32)
                    save_npz(
                        self.root / filename, data, np.asarray(label, dtype=np.int64),
                        {"subject": subj, "trial": trial, "start": start},
                    )
                    self.rows.append({
                        "subject": subj, "trial": trial, "file": filename,
                        "label": label, "split": split,
                    })
        self.manifest = pd.DataFrame(self.rows)

    def tearDown(self):
        self.tmp.cleanup()

    def test_trial_windows_chronological_and_one_label_per_trial(self):
        train_man = self.manifest[self.manifest.split == "train"]
        ordered = group_trials(train_man)
        self.assertEqual(len(ordered), 2)
        self.assertEqual(len(ordered[0]["files"]), 11)
        self.assertTrue(ordered[0]["files"][0].endswith("start000000.npz"))
        self.assertTrue(ordered[0]["files"][-1].endswith("start006400.npz"))

        norm = FeatureStandardizer(min_std=1e-5, clip=8).fit(
            self.root, train_man, modalities=MODALITY_ORDER,
        )
        before = {
            modality: np.array(norm.stats[modality]["mean"], copy=True)
            for modality in MODALITY_ORDER
        }
        train = DEAPTrialDataset(self.root, train_man, norm)
        val = DEAPTrialDataset(
            self.root, self.manifest[self.manifest.split == "val"], norm
        )
        self.assertEqual(len(train), 2)
        self.assertEqual(len(val), 2)
        self.assertEqual(train.shapes(), {m: (11, 8) for m in MODALITY_ORDER})
        self.assertEqual(train.labels().tolist(), [0, 1])
        self.assertEqual(val.subjects(), ["s02", "s02"])
        for mod in MODALITY_ORDER:
            np.testing.assert_array_equal(before[mod], norm.stats[mod]["mean"])
            self.assertTrue(torch.isfinite(val[0]["modalities"][mod]).all())
        # Cached embeddings must remain original.
        with np.load(self.root / train_man.iloc[0]["file"]) as cached:
            self.assertGreater(float(np.std(cached["x__face"])), 1.)

    def test_incomplete_or_inconsistent_trial_fails(self):
        training = self.manifest[self.manifest.split == "train"].copy()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            group_trials(training.iloc[1:])
        corrupted = training.copy()
        corrupted.loc[corrupted.index[0], "label"] = 1
        with self.assertRaisesRegex(ValueError, "conflicting"):
            group_trials(corrupted)

    def test_same_trial_tensors_for_ridge_and_neural(self):
        norm = FeatureStandardizer().fit(
            self.root, self.manifest[self.manifest.split == "train"],
            modalities=MODALITY_ORDER,
        )
        train = DEAPTrialDataset(
            self.root, self.manifest[self.manifest.split == "train"], norm
        )
        val = DEAPTrialDataset(
            self.root, self.manifest[self.manifest.split == "val"], norm
        )
        metrics, predictions = ridge_predict("ridge_concat", train, val)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(set(metrics["per_subject"]), {"s02"})
        batch = collate_multimodal([train[0], train[1]])
        self.assertEqual(batch["modalities"]["eeg"].shape, (2, 11, 8))

        for mode in ("unimodal_eeg", "early", "late", "directed_topk"):
            model = build_neural_model(
                mode, train.shapes(), topk=5, d_model=16, heads=4
            ).eval()
            with torch.no_grad():
                logits, aux = model(
                    batch["modalities"], batch["present"],
                    batch["quality_penalty"],
                )
            self.assertEqual(tuple(logits.shape), (2, 2))
            self.assertTrue(torch.isfinite(logits).all())
            count = float(aux["pair_count"].item())
            self.assertEqual(count, 5 if mode == "directed_topk" else 0)

    def test_subject_mean_detects_one_class_prediction(self):
        y = [0, 1, 0, 1, 0, 1]
        predictions = [1] * 6
        subjects = ["s01"] * 2 + ["s02"] * 2 + ["s03"] * 2
        score = metric_summary(y, predictions, subjects)
        self.assertAlmostEqual(score["subject_mean"]["balanced_accuracy"], 0.5)
        self.assertLess(score["subject_mean"]["macro_f1"], 0.35)
        self.assertEqual(score["pooled"]["predicted_class_counts"], [0, 6])

    def test_18_3_reserved_subject_split_without_loading_npz(self):
        m = pd.DataFrame([
            {"file": f"s{i:02d}.npz", "subject": f"s{i:02d}",
             "trial": 0, "label": i % 2}
            for i in range(1, 23)
        ])
        train, val, val_ids = split_development(m)
        self.assertEqual(train.subject.nunique(), 18)
        self.assertEqual(val.subject.nunique(), 3)
        self.assertEqual(val_ids, ["s21", "s14", "s07"])
        self.assertNotIn("s22", set(train.subject) | set(val.subject))


if __name__ == "__main__":
    unittest.main()
