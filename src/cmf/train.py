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
    item=dataset[0]; return {n:tuple(x.shape) for n,x in item["modalities"].items()},tuple(item["target"].shape)

def move(batch,device):
    return ({k:v.to(device) for k,v in batch["modalities"].items()},{k:v.to(device) for k,v in batch["present"].items()},batch["target"].to(device))

def evaluate(model,loader,device,task):
    model.eval(); ys=[]; ps=[]; losses=[]
    with torch.no_grad():
        for batch in loader:
            mods,present,y=move(batch,device); pred,_=model(mods,present)
            if task=="classification": loss=torch.nn.functional.cross_entropy(pred,y.long().view(-1)); ps+=pred.argmax(-1).cpu().tolist(); ys+=y.cpu().view(-1).tolist()
            else: target=y.float().reshape(y.shape[0],-1); loss=torch.nn.functional.mse_loss(pred,target); ps.append(pred.cpu().numpy()); ys.append(target.cpu().numpy())
            losses.append(loss.item())
    if task=="classification": return {"loss":float(np.mean(losses)),"accuracy":accuracy_score(ys,ps),"macro_f1":f1_score(ys,ps,average="macro")}
    ys=np.concatenate(ys); ps=np.concatenate(ps); return {"loss":float(np.mean(losses)),"rmse":float(mean_squared_error(ys,ps)**.5)}

def train_from_config(config_path,mode=None,split_manifests=None,run_name=None,return_metrics=False,encoder_checkpoints=None):
    cfg=load_config(config_path); seed_everything(cfg.get("seed",42)); dcfg=cfg["dataset"]; tcfg=cfg["training"]; mcfg=cfg["model"]; cache=Path(dcfg["cache_dir"]); task=dcfg.get("task","classification")
    train_manifest = split_manifests.get("train") if split_manifests else None
    val_manifest = split_manifests.get("val") if split_manifests else None
    test_manifest = split_manifests.get("test") if split_manifests else None
    train_ds=CachedMultimodalDataset(cache,"train",tcfg.get("modality_dropout",0.),train_manifest); val_ds=CachedMultimodalDataset(cache,"val",0,val_manifest); shapes,target_shape=infer_shapes(train_ds)
    if task=="classification":
        import pandas as pd
        man=pd.read_csv(cache/"manifest.csv")
        if "num_classes" in dcfg:
            num_outputs=int(dcfg["num_classes"])
        elif "label" in man.columns:
            num_outputs=int(man["label"].max()+1)
        else:
            # Targets are stored inside the cached NPZ files; infer classes from the dataset when the manifest has no label column.
            labels=[int(train_ds[i]["target"].view(-1)[0].item()) for i in range(len(train_ds))]
            num_outputs=int(max(labels)+1)
    else: num_outputs=int(np.prod(target_shape))
    model=ContrastiveSparseFusion(shapes,num_outputs,task,d_model=mcfg.get("d_model",128),heads=mcfg.get("heads",4),topk=mcfg.get("topk",1),mode=mode or mcfg.get("mode","contrastive_topk"),temperature=mcfg.get("temperature",.1),reliability=mcfg.get("reliability",True),selector_temperature=mcfg.get("selector_temperature",.7),gumbel=mcfg.get("gumbel",True),encoder_configs=mcfg.get("encoders",{}),encoder_checkpoints=encoder_checkpoints,freeze_pretrained=cfg.get("pretraining",{}).get("freeze_for_fusion",True))
    device=torch.device("cuda" if torch.cuda.is_available() and not tcfg.get("cpu",False) else "cpu"); model.to(device); train_loader=DataLoader(train_ds,batch_size=tcfg.get("batch_size",16),shuffle=True,num_workers=tcfg.get("workers",0),collate_fn=collate_multimodal); val_loader=DataLoader(val_ds,batch_size=tcfg.get("batch_size",16),shuffle=False,num_workers=tcfg.get("workers",0),collate_fn=collate_multimodal)
    opt=torch.optim.AdamW(model.parameters(),lr=float(tcfg.get("lr",3e-4)),weight_decay=float(tcfg.get("weight_decay",1e-4))); outdir=Path(cfg.get("output_dir","runs"))/dcfg["name"]/(mode or mcfg.get("mode","contrastive_topk"))
    if run_name: outdir=outdir/run_name
    outdir=ensure_dir(outdir); best=float("inf"); history=[]; best_metrics=None; bad_epochs=0; patience=int(tcfg.get("patience",0))
    for epoch in range(1,int(tcfg.get("epochs",30))+1):
        model.train(); total=[]; pairs=[]; start=time.time()
        for batch in train_loader:
            mods,present,y=move(batch,device); pred,aux=model(mods,present); base=torch.nn.functional.cross_entropy(pred,y.long().view(-1)) if task=="classification" else torch.nn.functional.mse_loss(pred,y.float().reshape(y.shape[0],-1)); cweight=float(mcfg.get("contrastive_weight",.1)) if (mode or mcfg.get("mode","contrastive_topk"))=="contrastive_topk" else 0.; loss=base+cweight*aux["contrastive_loss"]; opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); total.append(loss.item()); pairs.append(aux["pair_count"].item())
        metrics=evaluate(model,val_loader,device,task); rec={"epoch":epoch,"train_loss":float(np.mean(total)),"mean_selected_pairs":float(np.mean(pairs)),"seconds":time.time()-start,**metrics}; history.append(rec); print(rec)
        if metrics["loss"]<best:
            best=metrics["loss"]; best_metrics=metrics.copy(); bad_epochs=0; torch.save({"model":model.state_dict(),"shapes":shapes,"num_outputs":num_outputs,"task":task,"config":cfg},outdir/"best.pt")
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}; best validation loss={best:.6f}")
                break
    (outdir/"history.json").write_text(json.dumps(history,indent=2))
    result={"outdir":str(outdir),"best_val":best_metrics}
    if test_manifest is not None:
        test_ds=CachedMultimodalDataset(cache,"test",0,test_manifest); test_loader=DataLoader(test_ds,batch_size=tcfg.get("batch_size",16),shuffle=False,num_workers=tcfg.get("workers",0),collate_fn=collate_multimodal)
        ckpt=torch.load(outdir/"best.pt",map_location=device); model.load_state_dict(ckpt["model"]); result["test"]=evaluate(model,test_loader,device,task)
        (outdir/"metrics.json").write_text(json.dumps(result,indent=2))
    return result if return_metrics else outdir
