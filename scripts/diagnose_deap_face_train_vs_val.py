"""Diagnostic FACE-only LOSO fold with train/validation metrics each epoch.

Purpose: distinguish failure to fit the training subjects from cross-subject
generalization failure. This is diagnostic-only and does not alter train.py.
"""
from pathlib import Path
import argparse, json, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from cmf.config import load_config
from cmf.datasets.common import CachedMultimodalDataset, collate_multimodal
from cmf.models.fusion import ContrastiveSparseFusion
from cmf.utils.seed import seed_everything

def move(batch,dev):
    return ({k:v.to(dev) for k,v in batch["modalities"].items()},
            {k:v.to(dev) for k,v in batch["present"].items()},
            batch["target"].to(dev))

@torch.no_grad()
def metrics(model,loader,dev):
    model.eval(); ys=[]; ps=[]; losses=[]
    for b in loader:
        x,p,y=move(b,dev); out,_=model(x,p)
        losses.append(torch.nn.functional.cross_entropy(out,y.long().view(-1)).item())
        ys.extend(y.view(-1).cpu().tolist()); ps.extend(out.argmax(1).cpu().tolist())
    return {"loss":float(np.mean(losses)),"accuracy":accuracy_score(ys,ps),
            "macro_f1":f1_score(ys,ps,average="macro"),
            "pred_counts":np.bincount(np.asarray(ps),minlength=2).tolist()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/deap_face_only_raw255_long.yaml")
    ap.add_argument("--test-subject",default="s22")
    ap.add_argument("--epochs",type=int,default=30)
    ap.add_argument("--lr",type=float,default=None)
    a=ap.parse_args(); cfg=load_config(a.config); seed_everything(int(cfg.get("seed",42)))
    cache=Path(cfg["dataset"]["cache_dir"]); man=pd.read_csv(cache/"manifest.csv")
    subjects=sorted(man.subject.unique().tolist()); ti=subjects.index(a.test_subject)
    val_subject=subjects[ti-1] if ti>0 else subjects[-1]
    train_subjects=[s for s in subjects if s not in {a.test_subject,val_subject}]
    train_man=man[man.subject.isin(train_subjects)].reset_index(drop=True)
    val_man=man[man.subject==val_subject].reset_index(drop=True)
    tr=CachedMultimodalDataset(cache,"train",0.,train_man)
    va=CachedMultimodalDataset(cache,"val",0.,val_man)
    shape={n:tuple(x.shape) for n,x in tr[0]["modalities"].items()}
    mcfg=cfg["model"]; tcfg=cfg["training"]
    model=ContrastiveSparseFusion(shape,2,task="classification",d_model=mcfg.get("d_model",128),
        heads=mcfg.get("heads",4),topk=mcfg.get("topk",1),mode="late",
        temperature=mcfg.get("temperature",.1),reliability=mcfg.get("reliability",True),
        selector_temperature=mcfg.get("selector_temperature",.7),gumbel=mcfg.get("gumbel",True),
        encoder_configs=mcfg.get("encoders",{}))
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(dev)
    labels=np.asarray([int(tr[i]["target"].view(-1)[0]) for i in range(len(tr))])
    cc=np.bincount(labels,minlength=2); weights=torch.tensor(len(labels)/(2*cc),dtype=torch.float32,device=dev)
    bs=int(tcfg.get("batch_size",32))
    trl=DataLoader(tr,batch_size=bs,shuffle=True,collate_fn=collate_multimodal)
    tre=DataLoader(tr,batch_size=bs,shuffle=False,collate_fn=collate_multimodal)
    val=DataLoader(va,batch_size=bs,shuffle=False,collate_fn=collate_multimodal)
    lr=float(a.lr if a.lr is not None else tcfg.get("lr",3e-4))
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=float(tcfg.get("weight_decay",1e-4)))
    print(f"test={a.test_subject} val={val_subject} train_subjects={len(train_subjects)} samples={len(tr)}/{len(va)}")
    print("train class counts:",cc.tolist(),"weights:",weights.detach().cpu().tolist(),"lr:",lr)
    for ep in range(1,a.epochs+1):
        model.train()
        for b in trl:
            x,p,y=move(b,dev); out,aux=model(x,p)
            loss=torch.nn.functional.cross_entropy(out,y.long().view(-1),weight=weights)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
        tm=metrics(model,tre,dev); vm=metrics(model,val,dev)
        print({"epoch":ep,"train_acc":round(tm["accuracy"],4),"train_f1":round(tm["macro_f1"],4),
               "train_loss_unweighted":round(tm["loss"],5),"train_pred_counts":tm["pred_counts"],
               "val_acc":round(vm["accuracy"],4),"val_f1":round(vm["macro_f1"],4),
               "val_pred_counts":vm["pred_counts"]})

if __name__=="__main__": main()
