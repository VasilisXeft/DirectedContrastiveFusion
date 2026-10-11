"""Checks normalization is fold-fitted, shape-preserving and reusable at test time."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cmf.feature_normalization import FeatureStandardizer
from cmf.datasets.common import CachedMultimodalDataset
from cmf.utils.io import save_npz


class TestFeatureNormalization(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rs = np.random.default_rng(123)
        # Different modality scales and shifted test subject. Never fit on test.
        self.train_eeg = rs.normal(0.0005, 0.0002, (4, 3)).astype(np.float32)
        self.train_face = rs.normal(30, 10, (4, 2)).astype(np.float32)
        self.val_eeg = np.full((4, 3), 1.0, dtype=np.float32)
        self.val_face = np.full((4, 2), -300, dtype=np.float32)
        for sub, eeg, face, split in [
            ("s01", self.train_eeg, self.train_face, "train"),
            ("s02", self.val_eeg, self.val_face, "val"),
        ]:
            save_npz(self.root / f"{sub}.npz", {"eeg": eeg, "face": face},
                     np.array(1, dtype=np.int64), {"subject": sub, "trial": 0})
        self.manifest = pd.DataFrame([
            {"file": "s01.npz", "subject": "s01", "split": "train"},
            {"file": "s02.npz", "subject": "s02", "split": "val"},
        ])

    def tearDown(self):
        self.temp.cleanup()

    def test_training_only_and_replay(self):
        scaler = FeatureStandardizer(min_std=1e-6, clip=8).fit(
            self.root, self.manifest[self.manifest.split == "train"],
            modalities=["eeg", "face"])
        np.testing.assert_allclose(scaler.stats["eeg"]["mean"],
                                   self.train_eeg.mean(axis=0), atol=1e-7)
        np.testing.assert_allclose(scaler.stats["face"]["mean"],
                                   self.train_face.mean(axis=0), atol=1e-5)
        transformed_eeg = scaler.transform("eeg", self.train_eeg)
        transformed_face = scaler.transform("face", self.train_face)
        self.assertEqual(transformed_eeg.shape, self.train_eeg.shape)
        self.assertEqual(transformed_face.shape, self.train_face.shape)
        np.testing.assert_allclose(transformed_eeg.mean(axis=0), 0, atol=1e-4)
        np.testing.assert_allclose(transformed_face.mean(axis=0), 0, atol=1e-4)
        heldout = scaler.transform("face", self.val_face)
        self.assertTrue((heldout == -8).all())  # Uses train stats, not val stats.
        restored = FeatureStandardizer.from_state_dict(scaler.state_dict())
        np.testing.assert_allclose(restored.transform("eeg", self.train_eeg), transformed_eeg)
        np.testing.assert_allclose(restored.transform("face", self.val_face), heldout)
        train_ds = CachedMultimodalDataset(self.root, "train",
                                           manifest=self.manifest,
                                           feature_normalizer=restored)
        val_ds = CachedMultimodalDataset(self.root, "val",
                                         manifest=self.manifest,
                                         feature_normalizer=restored)
        self.assertEqual(train_ds[0]["modalities"]["eeg"].shape, (4, 3))
        self.assertTrue((val_ds[0]["modalities"]["face"].numpy() == -8).all())
        # Original cache must remain untouched.
        with np.load(self.root / "s02.npz") as source:
            np.testing.assert_array_equal(source["x__face"], self.val_face)

    def test_fit_rejects_validation_data(self):
        with self.assertRaisesRegex(ValueError, "train-only"):
            FeatureStandardizer().fit(self.root, self.manifest, modalities=["eeg"])

    def test_constant_embedding_and_dimension_check(self):
        save_npz(self.root / "flat.npz", {"eeg": np.ones((4, 3), np.float32)},
                 np.array(0))
        manifest = pd.DataFrame([{"file": "flat.npz", "split": "train"}])
        scaler = FeatureStandardizer(min_std=1e-3).fit(self.root, manifest)
        transformed = scaler.transform("eeg", np.ones((4, 3), np.float32))
        self.assertTrue(np.all(np.isfinite(transformed)))
        np.testing.assert_allclose(transformed, 0)
        with self.assertRaisesRegex(ValueError, "dimension"):
            scaler.transform("eeg", np.ones((4, 2), np.float32))


if __name__ == "__main__":
    unittest.main()
