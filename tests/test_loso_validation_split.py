"""Tests for leakage-safe multi-subject LOSO participant splits."""
import unittest

import pandas as pd

from cmf.loso import _fold_manifests, _validation_subjects


class TestLOSOValidationSplits(unittest.TestCase):
    def setUp(self):
        self.subjects = [f"s{i:02d}" for i in range(1, 23)]
        self.manifest = pd.DataFrame([
            {"file": f"{subj}_trial{trial}.npz", "subject": subj, "trial": trial}
            for subj in self.subjects for trial in range(40)
        ])
        self.config = {"validation_strategy": "spaced_previous", "validation_subject_count": 3}

    def test_18_3_1_s22(self):
        val = _validation_subjects(self.subjects, "s22", self.config)
        self.assertEqual(val, ["s21", "s14", "s07"])
        split = _fold_manifests(self.manifest, "s22", val, "subject")
        expected = {"train": (18, 720), "val": (3, 120), "test": (1, 40)}
        for name, (subjects, samples) in expected.items():
            self.assertEqual(split[name]["subject"].nunique(), subjects)
            self.assertEqual(len(split[name]), samples)
            self.assertTrue((split[name]["split"] == name).all())
        self.assertEqual(set(split["test"]["subject"]), {"s22"})
        self.assertTrue(set(split["train"]["subject"]).isdisjoint(set(val + ["s22"])))
        self.assertEqual(sum(len(s) for s in split.values()), len(self.manifest))

    def test_all_outer_folds_and_balanced_validation_rotation(self):
        counts = dict.fromkeys(self.subjects, 0)
        for test in self.subjects:
            val = _validation_subjects(self.subjects, test, self.config)
            self.assertEqual(len(set(val)), 3)
            self.assertNotIn(test, val)
            split = _fold_manifests(self.manifest, test, val, "subject")
            self.assertEqual(split["train"]["subject"].nunique(), 18)
            for subject in val:
                counts[subject] += 1
        self.assertEqual(set(counts.values()), {3})

    def test_default_single_validation_compatibility(self):
        val = _validation_subjects(self.subjects, "s22", {})
        self.assertEqual(val, ["s21"])
        split = _fold_manifests(self.manifest, "s22", val, "subject")
        self.assertEqual(split["train"]["subject"].nunique(), 20)

    def test_reject_leakage_and_invalid_counts(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            _validation_subjects(self.subjects, "s22", {
                "validation_subject_count": 3,
                "validation_subjects": ["s01", "s01", "s22"],
            })
        with self.assertRaisesRegex(ValueError, "between"):
            _validation_subjects(self.subjects, "s22", {
                "validation_subject_count": 22,
            })


if __name__ == "__main__":
    unittest.main()
