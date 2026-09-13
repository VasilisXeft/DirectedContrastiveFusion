from pathlib import Path
import json, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from cmf.config import load_config
from cmf.datasets.common import CachedMultimodalDataset, collate_multimodal
from cmf.models.fusion import ContrastiveSparseFusion
from cmf.utils.seed import seed_everything
from cmf.utils.io import ensure_dir


def infer_shapes(dataset):
    item=dataset[0]
    return {n:tuple(x.shape) for n,x in item["modalities"].items()}, tuple(item["target"].shape)


def move(batch,device):
    return ({k:v.to(device) for k,v in batch["modalities"].items()},
            {k:v.to(device) for k,v in batch["present"].items()},batch["target"].to(device))


def evaluate(model,loader,device,task):
    model.eval(); ys=[]; ps=[]; losses=[]
    with torch.no_grad():
        for batch in loader:
            mods,present,y=move(batch,device); pred,aux=model(mods,present)
            if task=="classification":
                loss=torch.nn.functional.cross_entropy(pred,y.long().view(-1)); ps += pred.argmax(-1).cpu().tolist(); ys += y.cpu().view(-1).tolist()
            else:
                target=y.float().reshape(y.shape[0],-1); loss=torch.nn.functional.mse_loss(pred,target); ps.append(pred.cpu().numpy()); ys.append(target.cpu().numpy())
            losses.append(loss.item())
    if task=="classification":
        return {"loss":float(np.mean(losses)),"accuracy":accuracy_score(ys,ps),"macro_f1":f1_score(ys,ps,average="macro")}
    ys=np.concatenate(ys); ps=np.concatenate(ps); return {"loss":float(np.mean(losses)),"rmse":float(mean_squared_error(ys,ps)**0.5)}


def train_from_config(config_path, mode=None):
    cfg=load_config(config_path); seed_everything(cfg.get("seed",42)); dcfg=cfg["dataset"]; tcfg=cfg["training"]; mcfg=cfg["model"]
    cache=Path(dcfg["cache_dir"]); task=dcfg.get("task","classification")
    train_ds=CachedMultimodalDataset(cache,"train",tcfg.get("modality_dropout",0.0)); val_ds=CachedMultimodalDataset(cache,"val",0)
    shapes,target_shape=infer_shapes(train_ds)
    if task=="classification":
        import pandas as pd
        man=pd.read_csv(cache/"manifest.csv"); num_outputs=int(dcfg.get("num_classes",man.label.max()+1))
    else:
        num_outputs=int(np.prod(target_shape))
    model=ContrastiveSparseFusion(shapes,num_outputs,task,d_model=mcfg.get("d_model",128),heads=mcfg.get("heads",4),topk=mcfg.get("topk",1),mode=mode or mcfg.get("mode","contrastive_topk"),temperature=mcfg.get("temperature",.1),reliability=mcfg.get("reliability",True),selector_temperature=mcfg.get("selector_temperature",.7),gumbel=mcfg.get("gumbel",True))
    device=torch.device("cuda" if torch.cuda.is_available() and not tcfg.get("cpu",False) else "cpu"); model.to(device)
    train_loader=DataLoader(train_ds,batch_size=tcfg.get("batch_size",16),shuffle=True,num_workers=tcfg.get("workers",0),collate_fn=collate_multimodal)
    val_loader=DataLoader(val_ds,batch_size=tcfg.get("batch_size",16),shuffle=False,num_workers=tcfg.get("workers",0),collate_fn=collate_multimodal)
    opt=torch.optim.AdamW(model.parameters(),lr=float(tcfg.get("lr",3e-4)),weight_decay=float(tcfg.get("weight_decay",1e-4)))
    outdir=ensure_dir(Path(cfg.get("output_dir","runs"))/dcfg["name"]/(mode or mcfg.get("mode","contrastive_topk")))
    best=float("inf"); history=[]
    for epoch in range(1,int(tcfg.get("epochs",30))+1):
        model.train(); total=[]; pairs=[]; start=time.time()
        for batch in train_loader:
            mods,present,y=move(batch,device); pred,aux=model(mods,present)
            if task=="classification": base=torch.nn.functional.cross_entropy(pred,y.long().view(-1))
            else: base=torch.nn.functional.mse_loss(pred,y.float().reshape(y.shape[0],-1))
            cweight = float(mcfg.get("contrastive_weight",0.1)) if (mode or mcfg.get("mode","contrastive_topk")) == "contrastive_topk" else 0.0
            loss=base+cweight*aux["contrastive_loss"]
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            total.append(loss.item()); pairs.append(aux["pair_count"].item())
        metrics=evaluate(model,val_loader,device,task); rec={"epoch":epoch,"train_loss":float(np.mean(total)),"mean_selected_pairs":float(np.mean(pairs)),"seconds":time.time()-start,**metrics}; history.append(rec); print(rec)
        if metrics["loss"]<best:
            best=metrics["loss"]; torch.save({"model":model.state_dict(),"shapes":shapes,"num_outputs":num_outputs,"task":task,"config":cfg},outdir/"best.pt")
    (outdir/"history.json").write_text(json.dumps(history,indent=2)); return outdir
