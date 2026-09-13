from pathlib import Path
import torch
from torch.utils.data import DataLoader
from cmf.datasets.common import CachedMultimodalDataset, collate_multimodal
from cmf.models.fusion import ContrastiveSparseFusion
from cmf.train import evaluate


def evaluate_checkpoint(checkpoint, cache_dir, split="test"):
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False); cfg=ck["config"]; mcfg=cfg["model"]
    model=ContrastiveSparseFusion(ck["shapes"],ck["num_outputs"],ck["task"],mcfg.get("d_model",128),mcfg.get("heads",4),mcfg.get("topk",1),mcfg.get("mode","contrastive_topk"),mcfg.get("temperature",.1),mcfg.get("reliability",True),mcfg.get("selector_temperature",.7),mcfg.get("gumbel",True)); model.load_state_dict(ck["model"])
    ds=CachedMultimodalDataset(cache_dir,split); loader=DataLoader(ds,batch_size=16,shuffle=False,collate_fn=collate_multimodal)
    return evaluate(model,loader,torch.device("cpu"),ck["task"])
