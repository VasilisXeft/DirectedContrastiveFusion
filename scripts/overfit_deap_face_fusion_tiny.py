"""Overfit the actual ContrastiveSparseFusion FACE-only late-mode path on 64 samples."""
from pathlib import Path
import argparse, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from cmf.models.fusion import ContrastiveSparseFusion

class TinyFace(Dataset):
    def __init__(self, cache, n_per_class=32, seed=42):
        self.cache=Path(cache); man=pd.read_csv(self.cache/"manifest.csv"); rows=[]
        for _,r in man.iterrows():
            z=np.load(self.cache/r["file"],allow_pickle=False)
            rows.append((r["file"],int(np.asarray(z["target"]).reshape(-1)[0])))
        rng=np.random.default_rng(seed); self.rows=[]
        for y in sorted(set(y for _,y in rows)):
            pool=[r for r in rows if r[1]==y]
            idx=rng.choice(len(pool),min(n_per_class,len(pool)),replace=False)
            self.rows += [pool[i] for i in idx]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        f,y=self.rows[i]; x=np.load(self.cache/f,allow_pickle=False)["x__face"].astype("float32")
        return torch.from_numpy(x),torch.tensor(y),torch.tensor(1.,dtype=torch.float32)

def collate(batch):
    x=torch.stack([b[0] for b in batch]); y=torch.stack([b[1] for b in batch]); p=torch.stack([b[2] for b in batch])
    return x,y,p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",default="data/features/deap_face_only_raw255")
    ap.add_argument("--n-per-class",type=int,default=32)
    ap.add_argument("--epochs",type=int,default=150)
    ap.add_argument("--lr",type=float,default=1e-3)
    a=ap.parse_args(); torch.manual_seed(42); np.random.seed(42)
    ds=TinyFace(a.cache,a.n_per_class); dl=DataLoader(ds,batch_size=len(ds),shuffle=True,collate_fn=collate)
    x0,_,_=ds[0]; shape=tuple(x0.shape)
    model=ContrastiveSparseFusion(
        {"face":shape},2,task="classification",d_model=128,heads=4,topk=1,
        mode="late",temperature=.1,reliability=True,selector_temperature=.7,gumbel=True,
        encoder_configs={"face":{"type":"precomputed","feature_dim":shape[-1]}}
    )
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    print("samples:",len(ds),"shape:",shape,"device:",dev)
    for ep in range(1,a.epochs+1):
        model.train()
        for x,y,p in dl:
            x,y,p=x.to(dev),y.to(dev),p.to(dev)
            logits,aux=model({"face":x},{"face":p})
            loss=torch.nn.functional.cross_entropy(logits,y)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep==1 or ep%10==0:
            model.eval()
            with torch.no_grad():
                x,y,p=next(iter(DataLoader(ds,batch_size=len(ds),shuffle=False,collate_fn=collate)))
                x,y,p=x.to(dev),y.to(dev),p.to(dev)
                logits,aux=model({"face":x},{"face":p}); pred=logits.argmax(1)
                l=torch.nn.functional.cross_entropy(logits,y).item()
                acc=(pred==y).float().mean().item()
                counts=torch.bincount(pred,minlength=2).cpu().tolist()
                rel=float(aux["reliability"]["face"].mean().item())
            print({"epoch":ep,"loss":round(l,6),"train_acc":round(acc,4),"pred_counts":counts,"mean_reliability":round(rel,4)})
    print(("PASS" if acc>=.95 else "FAIL"),"- actual fusion tiny-set memorization",round(acc,4))

if __name__=="__main__": main()
