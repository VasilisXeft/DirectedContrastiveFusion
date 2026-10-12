"""Regression tests for aggregated seven-group development CV metrics."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_deap_trial_dev_cv import gather


class TestTrialDevCVRunner(unittest.TestCase):
    def test_aggregate_reports_per_subject_and_one_class_collapse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for group in (0, 1):
                for mode in ("ridge_eeg", "early"):
                    path = root / f"dev_group_{group:02d}" / "seed42" / mode
                    path.mkdir(parents=True)
                    people = [f"s{i:02d}" for i in range(group * 3 + 1, group * 3 + 4)]
                    per = {}
                    for index, subject in enumerate(people):
                        collapsed = index == 0 and mode == "early"
                        per[subject] = {
                            "trials": 20,
                            "balanced_accuracy": 0.5 if collapsed else 0.6,
                            "macro_f1": 0.33 if collapsed else 0.57,
                            "accuracy": 0.5 if collapsed else 0.6,
                            "predicted_class_counts": [0, 20] if collapsed else [10, 10],
                        }
                    metrics = {
                        "subject_mean": {
                            "balanced_accuracy": sum(v["balanced_accuracy"] for v in per.values()) / 3,
                            "macro_f1": sum(v["macro_f1"] for v in per.values()) / 3,
                        },
                        "pooled": {
                            "balanced_accuracy": .6,
                            "macro_f1": .55,
                        },
                        "per_subject": per,
                    }
                    (path / "validation_metrics.json").write_text(json.dumps(metrics))
            groups, subjects, summary = gather(root, [0, 1], [42], ["ridge_eeg", "early"])
            self.assertEqual(len(groups), 4)
            self.assertEqual(len(subjects), 12)
            self.assertEqual(summary.loc["early", "subject_folds"], 6)
            self.assertEqual(summary.loc["early", "one_class_folds"], 2)
            self.assertEqual(summary.loc["ridge_eeg", "one_class_folds"], 0)

    def test_missing_run_fails_loudly(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                gather(Path(folder), [0], [42], ["directed_topk"])


if __name__ == "__main__":
    unittest.main()
