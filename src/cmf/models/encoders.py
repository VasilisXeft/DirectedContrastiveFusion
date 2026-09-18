"""Unimodal encoder registry.

Fusion is encoder-agnostic: every encoder returns token sequences [B,T,D] in a
shared latent dimension. Strong pretrained encoders are the intended default for
publication experiments; generic encoders remain useful for ablations/smoke tests.
"""
from __future__ import annotations

import torch
from torch import nn


class TimeSeriesEncoder(nn.Module):
    def __init__(self, in_dim, d_model=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.GELU())
    def forward(self, x):
        return self.net(x)



class EEGNetEncoder(nn.Module):
    """EEGNet-style encoder for inputs [B,T,C], returning temporal tokens [B,T',D]."""
    def __init__(self, chans, d_model=128, f1=8, depth_multiplier=2, kernel=64, dropout=0.25):
        super().__init__()
        f2=f1*depth_multiplier
        self.temporal=nn.Sequential(
            nn.Conv2d(1,f1,(1,kernel),padding=(0,kernel//2),bias=False),
            nn.BatchNorm2d(f1))
        self.spatial=nn.Sequential(
            nn.Conv2d(f1,f2,(chans,1),groups=f1,bias=False),
            nn.BatchNorm2d(f2),nn.ELU(),nn.AvgPool2d((1,4)),nn.Dropout(dropout))
        self.separable=nn.Sequential(
            nn.Conv2d(f2,f2,(1,16),padding=(0,8),groups=f2,bias=False),
            nn.Conv2d(f2,f2,(1,1),bias=False),nn.BatchNorm2d(f2),nn.ELU(),
            nn.AvgPool2d((1,8)),nn.Dropout(dropout))
        self.proj=nn.Linear(f2,d_model)
    def forward(self,x):
        # cached DEAP EEG is [B,T,C]
        x=x.transpose(1,2).unsqueeze(1)
        y=self.separable(self.spatial(self.temporal(x))).squeeze(2).transpose(1,2)
        return self.proj(y)


class Physio1DCNNEncoder(nn.Module):
    """Compact modality-specific 1D CNN for PPG/EDA/TEMP-like signals."""
    def __init__(self, in_dim=1, d_model=128, channels=(32,64,96), kernel=7, dropout=0.2):
        super().__init__()
        layers=[]; cin=in_dim
        for cout in channels:
            layers += [nn.Conv1d(cin,cout,kernel,padding=kernel//2,bias=False),
                       nn.BatchNorm1d(cout),nn.GELU(),nn.MaxPool1d(2),nn.Dropout(dropout)]
            cin=cout
        self.net=nn.Sequential(*layers); self.proj=nn.Linear(cin,d_model)
    def forward(self,x):
        y=self.net(x.transpose(1,2)).transpose(1,2)
        return self.proj(y)


class ImageSequenceEncoder(nn.Module):
    def __init__(self, in_ch=3, d_model=128):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(in_ch,32,5,2,2),nn.BatchNorm2d(32),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.BatchNorm2d(64),nn.GELU(),nn.Conv2d(64,96,3,2,1),nn.BatchNorm2d(96),nn.GELU(),nn.AdaptiveAvgPool2d(1))
        self.proj=nn.Linear(96,d_model)
    def forward(self,x):
        b,t,c,h,w=x.shape; y=self.cnn(x.reshape(b*t,c,h,w)).flatten(1); return self.proj(y).reshape(b,t,-1)


class PrecomputedEncoder(nn.Module):
    """Use features extracted offline by any pretrained model."""
    def __init__(self, in_dim, d_model=128):
        super().__init__(); self.proj=nn.Linear(in_dim,d_model) if in_dim != d_model else nn.Identity()
    def forward(self,x):
        if x.ndim==2: x=x.unsqueeze(1)
        return self.proj(x)


class TorchvisionImageEncoder(nn.Module):
    """Pretrained torchvision image backbone applied frame-wise."""
    def __init__(self, name, d_model=128, weights="DEFAULT", freeze=False):
        super().__init__()
        try: import torchvision.models as tvm
        except ImportError as e: raise ImportError("torchvision is required for torchvision encoders") from e
        factory=getattr(tvm,name); w=weights if weights in (None,"DEFAULT") else weights
        model=factory(weights=w)
        if hasattr(model,"fc"): dim=model.fc.in_features; model.fc=nn.Identity()
        elif hasattr(model,"classifier"):
            clf=model.classifier; dim=clf[-1].in_features if isinstance(clf,nn.Sequential) else clf.in_features; model.classifier=nn.Identity()
        else: raise ValueError(f"Unsupported torchvision backbone: {name}")
        self.backbone=model; self.proj=nn.Linear(dim,d_model)
        if freeze:
            for p in self.backbone.parameters(): p.requires_grad=False
    def forward(self,x):
        if x.ndim==4: x=x.unsqueeze(1)
        b,t,c,h,w=x.shape; y=self.backbone(x.reshape(b*t,c,h,w)); return self.proj(y).reshape(b,t,-1)


class TimmEncoder(nn.Module):
    """Image/ViT backbones from timm, useful for pretrained RGB/depth models."""
    def __init__(self,name,d_model=128,pretrained=True,freeze=False,in_chans=3):
        super().__init__()
        try: import timm
        except ImportError as e: raise ImportError("timm is required for encoder type 'timm'") from e
        self.backbone=timm.create_model(name,pretrained=pretrained,num_classes=0,in_chans=in_chans)
        dim=self.backbone.num_features; self.proj=nn.Linear(dim,d_model)
        if freeze:
            for p in self.backbone.parameters(): p.requires_grad=False
    def forward(self,x):
        if x.ndim==4: x=x.unsqueeze(1)
        b,t,c,h,w=x.shape; y=self.backbone(x.reshape(b*t,c,h,w)); return self.proj(y).reshape(b,t,-1)


class HuggingFaceEncoder(nn.Module):
    """Generic Hugging Face backbone wrapper. Inputs should already match the model's tensor input."""
    def __init__(self,name,d_model=128,freeze=False):
        super().__init__()
        try: from transformers import AutoModel
        except ImportError as e: raise ImportError("transformers is required for encoder type 'huggingface'") from e
        self.backbone=AutoModel.from_pretrained(name); dim=self.backbone.config.hidden_size; self.proj=nn.Linear(dim,d_model)
        if freeze:
            for p in self.backbone.parameters(): p.requires_grad=False
    def forward(self,x):
        out=self.backbone(x); h=out.last_hidden_state if hasattr(out,"last_hidden_state") else out[0]
        return self.proj(h)


def build_encoder(name, shape, d_model, cfg=None):
    """Build a modality encoder from YAML configuration."""
    cfg=cfg or {}; typ=cfg.get("type","auto")
    if typ in {"auto","generic","generic_sequence"}:
        return ImageSequenceEncoder(shape[1],d_model) if len(shape)==4 else TimeSeriesEncoder(shape[-1],d_model)
    if typ=="timeseries": return TimeSeriesEncoder(shape[-1],d_model)
    if typ=="eegnet": return EEGNetEncoder(shape[-1],d_model,cfg.get("f1",8),cfg.get("depth_multiplier",2),cfg.get("kernel",64),cfg.get("dropout",.25))
    if typ=="physio_cnn1d": return Physio1DCNNEncoder(shape[-1],d_model,tuple(cfg.get("channels",[32,64,96])),cfg.get("kernel",7),cfg.get("dropout",.2))
    if typ=="image_cnn": return ImageSequenceEncoder(cfg.get("in_ch",shape[1]),d_model)
    if typ=="precomputed": return PrecomputedEncoder(cfg.get("feature_dim",shape[-1]),d_model)
    if typ=="torchvision": return TorchvisionImageEncoder(cfg["name"],d_model,cfg.get("weights","DEFAULT"),cfg.get("freeze",False))
    if typ=="timm": return TimmEncoder(cfg["name"],d_model,cfg.get("pretrained",True),cfg.get("freeze",False),cfg.get("in_chans",shape[1] if len(shape)==4 else 3))
    if typ=="huggingface": return HuggingFaceEncoder(cfg["name"],d_model,cfg.get("freeze",False))
    raise ValueError(f"Unknown encoder type '{typ}' for modality '{name}'")
