"""Unit-test EEGPT resampling and amplitude conversion without model downloads."""
import unittest

import numpy as np
import torch

from cmf.models.deap_eegpt import EEGPTFeatures


class SpyEEGPT:
    def __init__(self):
        self.inputs = []

    def __call__(self, x):
        self.inputs.append(x.detach().cpu().numpy().copy())
        # Match official 4D encoder output shape; encode one central sample.
        value = x[:, :, x.shape[-1] // 2].mean(dim=1).view(-1, 1, 1, 1)
        return value.expand(1, 1, 1, 512)


class TestEEGPTConversion(unittest.TestCase):
    def make_adapter(self, rate, amplitude):
        adapter = EEGPTFeatures.__new__(EEGPTFeatures)
        adapter.device = "cpu"
        adapter.tokens = 4
        adapter.target_sfreq = rate
        adapter.window_samples = rate * 4
        adapter.input_scale = amplitude
        adapter.indices = list(range(32))
        adapter.model = SpyEEGPT()
        return adapter

    def test_hosted_checkpoint_250hz_1000_samples(self):
        adapter = self.make_adapter(250, 1e-3)
        x = np.full((32, 1280), 10.0, dtype=np.float32)
        y = adapter(x)
        self.assertEqual(y.shape, (4, 512))
        self.assertEqual(len(adapter.model.inputs), 4)
        self.assertEqual(adapter.model.inputs[0].shape, (1, 32, 1000))
        np.testing.assert_allclose(y, 0.01, rtol=1e-3, atol=1e-5)

    def test_legacy_256hz_1024_samples_and_scale(self):
        adapter = self.make_adapter(256, 1e-6)
        x = np.full((32, 1280), 10.0, dtype=np.float32)
        y = adapter(x)
        self.assertEqual(y.shape, (4, 512))
        self.assertEqual(adapter.model.inputs[0].shape, (1, 32, 1024))
        np.testing.assert_allclose(y, 1e-5, rtol=1e-3, atol=1e-8)

    def test_bad_channel_count(self):
        adapter = self.make_adapter(250, 1e-3)
        with self.assertRaisesRegex(ValueError, "32 EEG channels"):
            adapter(np.zeros((30, 1280), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
