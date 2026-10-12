"""Unit checks for paired EEGPT preprocessing evaluation utilities (no model download)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compare_deap_eegpt_inputs import (
    VARIANTS,
    contract,
    evaluate_loso,
    load_cached_subject,
    subject_trials,
    window_starts,
)


class TestEEGPTInputComparison(unittest.TestCase):
    def test_window_starts_match_existing_cache_grid(self):
        np.testing.assert_array_equal(window_starts(3), [0, 3200, 6400])
        np.testing.assert_array_equal(window_starts(6), [0, 1280, 2560, 3840, 5120, 6400])
        np.testing.assert_array_equal(window_starts(11), np.arange(11) * 640)
        with self.assertRaisesRegex(ValueError, "3, 6 or 11"):
            window_starts(4)

    def test_subject_trials_and_contract(self):
        names = [
            {"subject": "s01", "trial": 0, "label": 1,
             "file": f"s01_trial00_start{int(start):06d}.npz"}
            for start in window_starts(3)
        ]
        frame = pd.DataFrame(names)
        entries = subject_trials(frame, "s01")
        self.assertEqual(len(entries), 1)
        spec = contract("s01", entries, window_starts(3), "braindecode/eegpt-pretrained")
        self.assertEqual(spec["subject"], "s01")
        self.assertEqual(spec["window_starts"], [0, 3200, 6400])
        self.assertEqual(spec["variants"], VARIANTS)
        with self.assertRaisesRegex(ValueError, "Conflicting labels"):
            subject_trials(pd.concat([frame, frame.assign(label=0)]), "s01")

    def test_subject_cache_contract_and_replay(self):
        frame = pd.DataFrame([{
            "subject": "s01", "trial": 0, "label": 1,
            "file": "s01_trial00_start000000.npz",
        }])
        spec = contract("s01", subject_trials(frame, "s01"),
                        window_starts(3), "braindecode/eegpt-pretrained")
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "s01.npz"
            data = {name: np.ones((1, 512), dtype=np.float32) for name in VARIANTS}
            np.savez_compressed(
                dest, **data, trials=np.array([0]), labels=np.array([1]),
                contract_json=np.asarray(json.dumps(spec)),
            )
            self.assertEqual(len(load_cached_subject(dest, spec, 1)["labels"]), 1)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                load_cached_subject(dest, dict(spec, target_rate=256), 1)

    def test_loso_splits_by_subject_and_returns_all_arms(self):
        rng = np.random.default_rng(4)
        subjects = ["s01", "s02", "s03", "s04"]
        data = {}
        for subject in subjects:
            labels = np.array([0, 1] * 3)
            values = rng.normal(size=(6, 512)).astype(np.float32)
            data[subject] = {
                "labels": labels, "trials": np.arange(6),
                **{name: values.copy() for name in VARIANTS},
            }
        scores = evaluate_loso(subjects, data, alpha=100.)
        self.assertEqual(len(scores), 4 * len(VARIANTS))
        self.assertEqual(set(scores["heldout_subject"]), set(subjects))
        self.assertTrue((scores["train_trials"] == 18).all())
        self.assertTrue((scores["validation_trials"] == 6).all())
        self.assertTrue(np.isfinite(scores["balanced_accuracy"]).all())


if __name__ == "__main__":
    unittest.main()
