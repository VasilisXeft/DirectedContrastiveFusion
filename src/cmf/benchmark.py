import time, os
from pathlib import Path
import numpy as np
import psutil, torch
from torch.utils.data import DataLoader
from cmf.datasets.common import CachedMultimodalDataset, collate_multimodal
from cmf.models.fusion import ContrastiveSparseFusion


def benchmark_checkpoint(checkpoint,cache_dir,split="test",warmup=3,runs=20):
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False); cfg=ck["config"]; mcfg=cfg["model"]
    model=ContrastiveSparseFusion(ck["shapes"],ck["num_outputs"],ck["task"],mcfg.get("d_model",128),mcfg.get("heads",4),mcfg.get("topk",1),mcfg.get("mode","contrastive_topk"),mcfg.get("temperature",.1),mcfg.get("reliability",True),mcfg.get("selector_temperature",.7),mcfg.get("gumbel",True)); model.load_state_dict(ck["model"]); model.eval()
    ds=CachedMultimodalDataset(cache_dir,split); loader=DataLoader(ds,batch_size=1,shuffle=False,collate_fn=collate_multimodal); batch=next(iter(loader)); mods=batch["modalities"]; present=batch["present"]
    with torch.no_grad():
        for _ in range(warmup): model(mods,present)
        times=[]; pairs=[]
        for _ in range(runs):
            t=time.perf_counter(); _,aux=model(mods,present); times.append((time.perf_counter()-t)*1000); pairs.append(aux["pair_count"].item())
    return {"latency_ms_mean":float(np.mean(times)),"latency_ms_std":float(np.std(times)),"selected_pairs_mean":float(np.mean(pairs)),"rss_mb":psutil.Process(os.getpid()).memory_info().rss/1024**2,"parameters":sum(p.numel() for p in model.parameters())}
