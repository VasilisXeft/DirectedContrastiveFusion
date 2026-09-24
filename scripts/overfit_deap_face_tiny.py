"""Diagnostic: can the FACE-only model memorize a tiny balanced subset?

If this fails, investigate the training/model path. If it succeeds, the pipeline is
trainable and weak LOSO performance is a data/label/generalization issue rather
than a broken optimizer path.
"""
from pathlib import Path
import argparse, numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class TinyFace(Dataset):
    def __init__(self, cache, n_per_class=32, seed=42):
        cache=Path(cache); man=pd.read_csv(cache/"manifest.csv")
        rows=[]
        for _,r in man.iterrows():
            z=np.load(cache/r["file"],allow_pickle=False)
            y=int(np.asarray(z["target"]).reshape(-1)[0])
            rows.append((r["file"],y))
        rng=np.random.default_rng(seed); chosen=[]
        for y in sorted(set(y for _,y in rows)):
            pool=[x for x in rows if x[1]==y]; idx=rng.choice(len(pool),min(n_per_class,len(pool)),replace=False)
            chosen += [pool[i] for i in idx]
        self.cache=cache; self.rows=chosen
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        f,y=self.rows[i]; x=np.load(self.cache/f,allow_pickle=False)["x__face"].astype("float32")
        return torch.from_numpy(x),torch.tensor(y,dtype=torch.long)

class Model(nn.Module):
    def __init__(self,dim=256,d=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,d),nn.GELU(),nn.LayerNorm(d),nn.Linear(d,d),nn.GELU(),nn.Linear(d,2))
    def forward(self,x): return self.net(x.mean(1))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cache",default="data/features/deap_face_only_raw255")
    ap.add_argument("--n-per-class",type=int,default=32); ap.add_argument("--epochs",type=int,default=100)
    a=ap.parse_args(); torch.manual_seed(42)
    ds=TinyFace(a.cache,a.n_per_class); dl=DataLoader(ds,batch_size=len(ds),shuffle=True)
    x0,_=ds[0]; model=Model(x0.shape[-1]).to("cuda" if torch.cuda.is_available() else "cpu")
    dev=next(model.parameters()).device; opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    print("samples:",len(ds),"device:",dev)
    for ep in range(1,a.epochs+1):
        model.train()
        for x,y in dl:
            x,y=x.to(dev),y.to(dev); p=model(x); loss=nn.functional.cross_entropy(p,y)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep==1 or ep%10==0:
            model.eval()
            with torch.no_grad():
                x,y=next(iter(DataLoader(ds,batch_size=len(ds),shuffle=False))); x,y=x.to(dev),y.to(dev)
                p=model(x); acc=(p.argmax(1)==y).float().mean().item(); l=nn.functional.cross_entropy(p,y).item()
            print({"epoch":ep,"loss":round(l,6),"train_acc":round(acc,4)})
    print("PASS" if acc>=0.95 else "FAIL", "- tiny-set memorization",round(acc,4))

if __name__=="__main__": main()
