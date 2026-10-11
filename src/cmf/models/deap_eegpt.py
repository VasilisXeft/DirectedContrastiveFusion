"""Frozen EEGPT features from DEAP 32-channel EEG, using Braindecode pretrained weights.

No classifier outputs or randomly initialized replacement encoders are used.
"""
import numpy as np
import torch
import mne
from scipy.signal import resample_poly

# Exact order of DEAP's first 32 preprocessed EEG channels.
DEAP_CHANNELS = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "F4", "F8", "FC6", "FC2", "C4", "T8",
    "CP6", "CP2", "P4", "P8", "PO4", "O2", "Fz", "Cz",
]


class EEGPTFeatures:
    def __init__(self, device, model_id="braindecode/eegpt-pretrained", tokens=4):
        from braindecode.models import EEGPT
        from braindecode.models.eegpt import EEGPT_CHANNELS

        allowed = {c.upper() for c in EEGPT_CHANNELS}
        self.indices = [i for i, name in enumerate(DEAP_CHANNELS) if name.upper() in allowed]
        self.names = [DEAP_CHANNELS[i] for i in self.indices]
        if len(self.indices) < 16:
            raise RuntimeError(f"Too few DEAP channels recognized by EEGPT: {self.names}")
        self.device = device
        self.tokens = tokens
        self.window_samples = 1024  # 4 seconds at 256 Hz
        info = mne.create_info(self.names, sfreq=256.0, ch_types=["eeg"] * len(self.names))
        info.set_montage("standard_1020", on_missing="raise")
        # The pretrained checkpoint stores a montage-specific chans_id tensor.
        # Instantiate for DEAP, then load the pretrained tensors excluding only
        # that channel-ID buffer, which must reflect the current electrode names.
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        self.model = EEGPT(
            n_chans=len(self.names), n_times=self.window_samples,
            sfreq=256.0, chs_info=info["chs"], return_encoder_output=True,
        )
        checkpoint = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        weights = load_file(checkpoint)
        skipped = [k for k in weights if k == "chans_id" or k.endswith(".chans_id")]
        for key in skipped:
            weights.pop(key)
        # Check for architectural mismatch rather than silently using random weights.
        own = self.model.state_dict()
        mismatches = [(k, tuple(v.shape), tuple(own[k].shape))
                      for k, v in weights.items()
                      if k in own and v.shape != own[k].shape]
        if mismatches:
            raise RuntimeError(f"EEGPT checkpoint shape mismatches: {mismatches[:8]}")
        result = self.model.load_state_dict(weights, strict=False)
        missing = [k for k in result.missing_keys if k not in skipped]
        unexpected = list(result.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                f"EEGPT pretrained weight coverage incomplete: missing={missing[:12]}, "
                f"unexpected={unexpected[:12]}"
            )
        print(f"EEGPT loaded {len(weights)} pretrained tensors; "
              f"rebuilt DEAP channel IDs {tuple(self.model.chans_id.shape)}")
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, signal):
        x = np.asarray(signal, dtype=np.float32)
        if x.shape[0] != 32:
            raise ValueError(f"Expected 32 EEG channels, got {x.shape}")
        # DEAP preprocessed EEG is 128 Hz and typically expressed in microvolts.
        # EEGPT expects physical EEG signal units; convert µV to V.
        x = x[self.indices] * 1e-6
        x = resample_poly(x, 2, 1, axis=-1)
        if x.shape[-1] < self.window_samples:
            raise ValueError("EEGPT requires at least 4 seconds of EEG")
        # Cover the full 10-second window with three overlapping 4-second crops.
        starts = np.linspace(0, x.shape[-1] - self.window_samples, self.tokens).round().astype(int)
        outputs = []
        for start in starts:
            chunk = torch.from_numpy(x[:, start:start + self.window_samples].copy()).unsqueeze(0).to(self.device)
            z = self.model(chunk)
            if isinstance(z, (tuple, list)):
                z = z[0]
            if isinstance(z, dict):
                z = z.get("last_hidden_state", z.get("features"))
            if z is None or not torch.is_tensor(z):
                raise TypeError("EEGPT returned no tensor features")
            # Braindecode encoder returns [B, patches, summary_tokens, 512].
            # Pool patch and summary dimensions while retaining one vector per crop.
            if z.ndim == 4:
                z = z.mean(dim=(1, 2))
            elif z.ndim == 3:
                z = z.mean(dim=1)
            elif z.ndim != 2:
                raise ValueError(f"Unexpected EEGPT output: {tuple(z.shape)}")
            if z.shape[-1] != 512:
                raise ValueError(f"Expected 512-dimensional EEGPT features, got {tuple(z.shape)}")
            outputs.append(z[0].float().cpu().numpy())
        return np.stack(outputs).astype(np.float32)
