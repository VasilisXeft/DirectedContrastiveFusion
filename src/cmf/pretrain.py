from pathlib import Path
import json, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from cmf.config import load_config
from cmf.datasets.common import CachedMultimodalDataset, collate_multimodal
from cmf.models.encoders import build_encoder
from cmf.utils.io import ensure_dir
from cmf.utils.seed import seed_everything


class UnimodalClassifier(nn.Module):
    def __init__(self, name, shape, encoder_cfg, d_model, num_classes):
        super().__init__(); self.encoder=build_encoder(name,shape,d_model,encoder_cfg)
        self.head=nn.Sequential(nn.LayerNorm(d_model),nn.Dropout(.2),nn.Linear(d_model,num_classes))
    def forward(self,x): return self.head(self.encoder(x).mean(1))


def _eval(model, loader, device, modality):
    model.eval(); ys=[]; ps=[]; losses=[]
    with torch.no_grad():
        for b in loader:
            x=b["modalities"][modality].to(device); y=b["target"].long().view(-1).to(device)
            p=model(x); losses.append(nn.functional.cross_entropy(p,y).item())
            ys += y.cpu().tolist(); ps += p.argmax(-1).cpu().tolist()
    return {"loss":float(np.mean(losses)),"accuracy":accuracy_score(ys,ps),"macro_f1":f1_score(ys,ps,average="macro")}


def pretrain_fold(config_path, splits, test_subject, val_subject):
    cfg=load_config(config_path); seed_everything(cfg.get("seed",42)); dcfg=cfg["dataset"]; pcfg=cfg["pretraining"]; mcfg=cfg["model"]
    cache=Path(dcfg["cache_dir"]); device=torch.device("cuda" if torch.cuda.is_available() and not pcfg.get("cpu",False) else "cpu")
    train_ds=CachedMultimodalDataset(cache,"train",0,splits["train"]); val_ds=CachedMultimodalDataset(cache,"val",0,splits["val"])
    item=train_ds[0]; shapes={n:tuple(x.shape) for n,x in item["modalities"].items()}
    modalities=pcfg.get("modalities",list(shapes)); outroot=ensure_dir(Path(cfg.get("output_dir","runs"))/dcfg["name"]/"unimodal_pretrain"/f"test_{test_subject}_val_{val_subject}")
    results={}
    reuse=bool(pcfg.get("reuse_checkpoints",True))
    for modality in modalities:
        path=outroot/f"{modality}.pt"
        if reuse and path.exists():
            print(f"{modality}: reusing {path}")
            results[modality]={"checkpoint":str(path),"reused":True}
            continue
        model=UnimodalClassifier(modality,shapes[modality],mcfg["encoders"][modality],mcfg.get("d_model",128),dcfg["num_classes"]).to(device)
        tr=DataLoader(train_ds,batch_size=pcfg.get("batch_size",64),shuffle=True,num_workers=pcfg.get("workers",0),collate_fn=collate_multimodal)
        va=DataLoader(val_ds,batch_size=pcfg.get("batch_size",64),shuffle=False,num_workers=pcfg.get("workers",0),collate_fn=collate_multimodal)
        opt=torch.optim.AdamW(model.parameters(),lr=float(pcfg.get("lr",1e-3)),weight_decay=float(pcfg.get("weight_decay",1e-4)))
        patience=int(pcfg.get("patience",5)); best=float("inf"); bad=0; hist=[]
        for epoch in range(1,int(pcfg.get("epochs",30))+1):
            model.train(); ls=[]; start=time.time()
            for b in tr:
                x=b["modalities"][modality].to(device); y=b["target"].long().view(-1).to(device)
                p=model(x); loss=nn.functional.cross_entropy(p,y); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); ls.append(loss.item())
            met=_eval(model,va,device,modality); rec={"epoch":epoch,"train_loss":float(np.mean(ls)),"seconds":time.time()-start,**met}; hist.append(rec); print(modality,rec)
            if met["loss"] < best-1e-5:
                best=met["loss"]; bad=0; torch.save({"encoder":model.encoder.state_dict(),"shape":shapes[modality],"encoder_cfg":mcfg["encoders"][modality],"d_model":mcfg.get("d_model",128)},path)
            else:
                bad+=1
                if bad>=patience: print(f"{modality}: early stopping at epoch {epoch}"); break
        results[modality]={"checkpoint":str(path),"best_val_loss":best,"history":hist}
    (outroot/"summary.json").write_text(json.dumps(results,indent=2)); return results
