import unittest

import torch
import torch.nn.functional as F

from cmf.losses import classification_loss


class TestClassificationLoss(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor(
            [[1.2, -0.3], [-0.4, 1.7], [0.3, -0.1]],
            dtype=torch.float32, requires_grad=True,
        )
        self.labels = torch.tensor([0, 1, 1])

    def test_unweighted_ce_matches_torch(self):
        actual = classification_loss(self.logits, self.labels, kind="ce")
        expected = F.cross_entropy(self.logits, self.labels)
        torch.testing.assert_close(actual, expected)

    def test_smoothed_ce_matches_torch(self):
        actual = classification_loss(
            self.logits, self.labels, kind="label_smoothing", smoothing=0.1
        )
        expected = F.cross_entropy(self.logits, self.labels, label_smoothing=0.1)
        torch.testing.assert_close(actual, expected)

    def test_focal_gamma_zero_equals_ce(self):
        focal = classification_loss(self.logits, self.labels, kind="focal", gamma=0)
        ce = classification_loss(self.logits, self.labels, kind="ce")
        torch.testing.assert_close(focal, ce)

    def test_weighted_ce_matches_torch(self):
        weights = torch.tensor([1.5, 0.7])
        actual = classification_loss(
            self.logits, self.labels, kind="ce", class_weights=weights
        )
        expected = F.cross_entropy(self.logits, self.labels, weight=weights)
        torch.testing.assert_close(actual, expected)

    def test_all_losses_backward(self):
        for kind in ("ce", "label_smoothing", "focal"):
            logits = self.logits.detach().clone().requires_grad_(True)
            loss = classification_loss(
                logits, self.labels, kind=kind, class_weights=torch.tensor([1., 1.2])
            )
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_invalid_options(self):
        with self.assertRaises(ValueError):
            classification_loss(self.logits, self.labels, kind="invalid")
        with self.assertRaises(ValueError):
            classification_loss(self.logits, self.labels, smoothing=1.0)
        with self.assertRaises(ValueError):
            classification_loss(self.logits, self.labels, gamma=-1.0)


if __name__ == "__main__":
    unittest.main()
