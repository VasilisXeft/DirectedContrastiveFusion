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
def move(batch,device): return ({k:v.to(device) for k,v in batch["modalities"].items()},{k:v.to(device) for k,v in batch["present"].items()},batch["target"].to(device))
def extras(batch,device): return ({k:v.to(device) for k,v in batch.get("quality_penalty",{}).items()},{k:v.to(device) for k,v in batch.get("reliability_target",{}).items()})
def reliability_loss(rel,target,present):
    vals=[]
    for n,r in rel.items():
        if n not in target: continue
        mask=present[n]>0
        if mask.any(): vals.append(torch.nn.functional.mse_loss(r[mask],target[n][mask]))
    return torch.stack(vals).mean() if vals else next(iter(rel.values())).new_tensor(0.)
def evaluate(model,loader,device,task):
    model.eval(); ys=[]; ps=[]; losses=[]
    with torch.no_grad():
        for batch in loader:
            mods,present,y=move(batch,device); q,_=extras(batch,device); pred,_=model(mods,present,q)
            if not torch.isfinite(pred).all(): raise FloatingPointError("Non-finite prediction during evaluation")
            if task=="classification": loss=torch.nn.functional.cross_entropy(pred,y.long().view(-1)); ps+=pred.argmax(-1).cpu().tolist(); ys+=y.cpu().view(-1).tolist()
            else: target=y.float().reshape(y.shape[0],-1); loss=torch.nn.functional.mse_loss(pred,target); ps.append(pred.cpu().numpy()); ys.append(target.cpu().numpy())
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite evaluation loss")
            losses.append(loss.item())
    if task=="classification": return {"loss":float(np.mean(losses)),"accuracy":accuracy_score(ys,ps),"macro_f1":f1_score(ys,ps,average="macro")}
    ys=np.concatenate(ys); ps=np.concatenate(ps); return {"loss":float(np.mean(losses)),"rmse":float(mean_squared_error(ys,ps)**.5)}
def routing_diagnostics(model,loader,device):
    model.eval(); edge_counts={}; rel_sums={}; useful_sums={}; q_sums={}; n=0; pair_sum=0.0
    with torch.no_grad():
        for batch in loader:
            mods,present,_=move(batch,device); q,_=extras(batch,device); _,aux=model(mods,present,q); b=next(iter(mods.values())).shape[0]; n+=b; pair_sum+=float(aux["pair_count"].item())*b
            for (src,tgt),mask in aux["hard_masks"].items(): edge_counts[f"{src}->{tgt}"]=edge_counts.get(f"{src}->{tgt}",0)+int(mask.sum().item())
            for name,val in aux["reliability"].items(): rel_sums[name]=rel_sums.get(name,0.0)+float(val.sum().item())
            for name,val in aux["usefulness"].items(): useful_sums[name]=useful_sums.get(name,0.0)+float(val.sum().item())
            for name,val in q.items(): q_sums[name]=q_sums.get(name,0.0)+float(val.sum().item())
    return {"samples":n,"mean_selected_pairs":pair_sum/max(n,1),"edge_selection_rate":{k:v/max(n,1) for k,v in sorted(edge_counts.items())},"mean_usefulness":{k:v/max(n,1) for k,v in sorted(useful_sums.items())},"mean_quality_penalty":{k:v/max(n,1) for k,v in sorted(q_sums.items())},"mean_reliability":{k:v/max(n,1) for k,v in sorted(rel_sums.items())}}
def train_from_config(config_path,mode=None,split_manifests=None,run_name=None,return_metrics=False,encoder_checkpoints=None,topk=None,seed=None):
    cfg=load_config(config_path); effective_seed=int(cfg.get("seed",42) if seed is None else seed); seed_everything(effective_seed); dcfg=cfg["dataset"]; tcfg=cfg["training"]; mcfg=cfg["model"]; cache=Path(dcfg["cache_dir"]); task=dcfg.get("task","classification")
    train_manifest=split_manifests.get("train") if split_manifests else None; val_manifest=split_manifests.get("val") if split_manifests else None; test_manifest=split_manifests.get("test") if split_manifests else None
    rc=float(tcfg.get("reliability_corruption",0.0)); train_ds=CachedMultimodalDataset(cache,"train",tcfg.get("modality_dropout",0.),train_manifest,reliability_corruption=rc); val_ds=CachedMultimodalDataset(cache,"val",0,val_manifest,reliability_corruption=0.); shapes,target_shape=infer_shapes(train_ds)
    if task=="classification":
        import pandas as pd
        man=pd.read_csv(cache/"manifest.csv"); num_outputs=int(dcfg["num_classes"]) if "num_classes" in dcfg else int(man["label"].max()+1); labels=np.asarray([int(train_ds[i]["target"].view(-1)[0].item()) for i in range(len(train_ds))],dtype=np.int64); class_counts=np.bincount(labels,minlength=num_outputs)
        if np.any(class_counts==0): raise ValueError(f"Training split has an empty class: counts={class_counts.tolist()}")
        class_weights=torch.tensor(len(labels)/(num_outputs*class_counts.astype(np.float64)),dtype=torch.float32); print(f"Fusion training class distribution: {class_counts.tolist()}"); print(f"Fusion class weights: {class_weights.tolist()}")
    else: num_outputs=int(np.prod(target_shape)); class_weights=None
    model=ContrastiveSparseFusion(shapes,num_outputs,task,d_model=mcfg.get("d_model",128),heads=mcfg.get("heads",4),topk=(mcfg.get("topk",1) if topk is None else int(topk)),mode=mode or mcfg.get("mode","contrastive_topk"),temperature=mcfg.get("temperature",.1),reliability=mcfg.get("reliability",True),selector_temperature=mcfg.get("selector_temperature",.7),gumbel=mcfg.get("gumbel",True),encoder_configs=mcfg.get("encoders",{}),encoder_checkpoints=encoder_checkpoints,freeze_pretrained=cfg.get("pretraining",{}).get("freeze_for_fusion",True),quality_lambda=mcfg.get("quality_lambda",1.0),reliability_src_weight=mcfg.get("reliability_src_weight",.5),reliability_tgt_weight=mcfg.get("reliability_tgt_weight",.25))
    device=torch.device("cuda" if torch.cuda.is_available() and not tcfg.get("cpu",False) else "cpu"); model.to(device); class_weights=class_weights.to(device) if class_weights is not None else None; train_loader=DataLoader(train_ds,batch_size=tcfg.get("batch_size",16),shuffle=True,num_workers=tcfg.get("workers",0),collate_fn=collate_multimodal); val_loader=DataLoader(val_ds,batch_size=tcfg.get("batch_size",16),shuffle=False,num_workers=tcfg.get("workers",0),collate_fn=collate_multimodal)
    opt=torch.optim.AdamW(model.parameters(),lr=float(tcfg.get("lr",1e-4)),weight_decay=float(tcfg.get("weight_decay",1e-4)),eps=float(tcfg.get("adam_eps",1e-7))); outdir=Path(cfg.get("output_dir","runs"))/dcfg["name"]/(mode or mcfg.get("mode","contrastive_topk")); outdir=outdir/run_name if run_name else outdir; outdir=ensure_dir(outdir); best=(-float("inf") if task=="classification" else float("inf")); history=[]; best_metrics=None; bad_epochs=0; patience=int(tcfg.get("patience",0)); grad_clip=float(tcfg.get("grad_clip",1.0))
    for epoch in range(1,int(tcfg.get("epochs",30))+1):
        model.train(); total=[]; bases=[]; rel_losses=[]; pairs=[]; grad_norms=[]; start=time.time()
        for bi,batch in enumerate(train_loader):
            mods,present,y=move(batch,device); q,rt=extras(batch,device); pred,aux=model(mods,present,q); base=torch.nn.functional.cross_entropy(pred,y.long().view(-1),weight=class_weights) if task=="classification" else torch.nn.functional.mse_loss(pred,y.float().reshape(y.shape[0],-1)); cweight=float(mcfg.get("contrastive_weight",.1)) if (mode or mcfg.get("mode","contrastive_topk"))=="contrastive_topk" else 0.; rloss=reliability_loss(aux["reliability"],rt,present); rweight=float(mcfg.get("reliability_loss_weight",.1)); loss=base+cweight*aux["contrastive_loss"]+rweight*rloss
            if not torch.isfinite(loss): raise FloatingPointError(f"Non-finite training loss at epoch={epoch} batch={bi}: base={base.item()} reliability={rloss.item()} contrastive={aux['contrastive_loss'].item()}")
            opt.zero_grad(set_to_none=True); loss.backward(); gn=torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip,error_if_nonfinite=True); opt.step()
            for name,p in model.named_parameters():
                if not torch.isfinite(p).all(): raise FloatingPointError(f"Non-finite parameter after optimizer step: {name}, epoch={epoch}, batch={bi}")
            total.append(loss.item()); bases.append(base.item()); rel_losses.append(rloss.item()); pairs.append(aux["pair_count"].item()); grad_norms.append(float(gn))
        metrics=evaluate(model,val_loader,device,task); rec={"epoch":epoch,"train_loss":float(np.mean(total)),"base_loss":float(np.mean(bases)),"reliability_loss":float(np.mean(rel_losses)),"mean_selected_pairs":float(np.mean(pairs)),"mean_grad_norm":float(np.mean(grad_norms)),"max_grad_norm":float(np.max(grad_norms)),"lr":float(opt.param_groups[0]["lr"]),"seconds":time.time()-start,**metrics}; history.append(rec); print(rec); improved=(metrics["macro_f1"]>best+1e-6) if task=="classification" else (metrics["loss"]<best-1e-6)
        if improved: best=metrics["macro_f1"] if task=="classification" else metrics["loss"]; best_metrics=metrics.copy(); bad_epochs=0; torch.save({"model":model.state_dict(),"shapes":shapes,"num_outputs":num_outputs,"task":task,"config":cfg},outdir/"best.pt")
        else:
            bad_epochs+=1
            if patience>0 and bad_epochs>=patience: print(f"Early stopping at epoch {epoch}; best validation {'macro-F1' if task=='classification' else 'loss'}={best:.6f}"); break
    (outdir/"history.json").write_text(json.dumps(history,indent=2)); result={"outdir":str(outdir),"best_val":best_metrics,"seed":effective_seed,"topk":(mcfg.get("topk",1) if topk is None else int(topk))}
    if test_manifest is not None:
        test_ds=CachedMultimodalDataset(cache,"test",0,test_manifest,reliability_corruption=0.); test_loader=DataLoader(test_ds,batch_size=tcfg.get("batch_size",16),shuffle=False,num_workers=tcfg.get("workers",0),collate_fn=collate_multimodal); ckpt=torch.load(outdir/"best.pt",map_location=device); model.load_state_dict(ckpt["model"]); result["test"]=evaluate(model,test_loader,device,task); result["routing"]=routing_diagnostics(model,test_loader,device); (outdir/"metrics.json").write_text(json.dumps(result,indent=2))
    return result if return_metrics else outdir
