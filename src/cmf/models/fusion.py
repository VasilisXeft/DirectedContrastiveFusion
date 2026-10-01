import itertools
import torch
from torch import nn
import torch.nn.functional as F
from cmf.models.encoders import build_encoder, load_pretrained_encoder


class ContrastiveSparseFusion(nn.Module):
    """Encoder-agnostic directed sparse cross-modal fusion."""
    def __init__(self, modality_shapes, num_outputs, task="classification", d_model=128, heads=4,
                 topk=3, mode="contrastive_topk", temperature=0.1, reliability=True,
                 selector_temperature=0.7, gumbel=True, encoder_configs=None,
                 encoder_checkpoints=None, freeze_pretrained=True, quality_lambda=1.0,
                 reliability_src_weight=0.5, reliability_tgt_weight=0.25):
        super().__init__(); self.names=list(modality_shapes); self.mode=mode; self.topk=int(topk); self.temperature=float(temperature); self.selector_temperature=float(selector_temperature); self.use_gumbel=bool(gumbel); self.task=task
        self.quality_lambda=float(quality_lambda); self.reliability_src_weight=float(reliability_src_weight); self.reliability_tgt_weight=float(reliability_tgt_weight)
        encoder_configs=encoder_configs or {}; encoder_checkpoints=encoder_checkpoints or {}
        self.encoders=nn.ModuleDict({n:(load_pretrained_encoder(n,shape,d_model,encoder_configs.get(n,{}),encoder_checkpoints[n],freeze_pretrained) if n in encoder_checkpoints else build_encoder(n,shape,d_model,encoder_configs.get(n,{}))) for n,shape in modality_shapes.items()})
        self.projectors=nn.ModuleDict({n:nn.Sequential(nn.Linear(d_model,d_model),nn.GELU(),nn.Linear(d_model,d_model)) for n in self.names})
        self.src_selector=nn.ModuleDict({n:nn.Linear(d_model,d_model,bias=False) for n in self.names}); self.tgt_selector=nn.ModuleDict({n:nn.Linear(d_model,d_model,bias=False) for n in self.names})
        self.reliability=nn.ModuleDict({n:nn.Linear(d_model,1) for n in self.names}) if reliability else None
        self.attn=nn.ModuleDict({f"{s}__{t}":nn.MultiheadAttention(d_model,heads,batch_first=True) for s,t in itertools.permutations(self.names,2)})
        self.norm=nn.LayerNorm(d_model); self.head=nn.Sequential(nn.Linear(d_model,d_model),nn.GELU(),nn.Dropout(.2),nn.Linear(d_model,num_outputs))

    @staticmethod
    def _finite(x, lo=-1e4, hi=1e4):
        return torch.nan_to_num(x,nan=0.0,posinf=hi,neginf=lo).clamp(lo,hi)

    def _encode(self,modalities):
        toks={n:self._finite(self.encoders[n](modalities[n])) for n in self.names}
        pooled={n:self._finite(t.mean(1)) for n,t in toks.items()}
        z={n:F.normalize(self._finite(self.projectors[n](pooled[n])),dim=-1,eps=1e-6) for n in self.names}
        return toks,pooled,z

    def contrastive_loss(self,z,present):
        losses=[]
        for i,a in enumerate(self.names):
            for b in self.names[i+1:]:
                valid=(present[a]>0)&(present[b]>0)
                if valid.sum()>1:
                    idx=torch.where(valid)[0]; za,zb=z[a][idx],z[b][idx]
                    logits=self._finite(za@zb.T,-50.,50.)/max(self.temperature,1e-4)
                    logits=logits.clamp(-80.,80.); lab=torch.arange(len(idx),device=logits.device)
                    losses += [F.cross_entropy(logits,lab),F.cross_entropy(logits.T,lab)]
        return torch.stack(losses).mean() if losses else next(iter(z.values())).new_tensor(0.)

    def _reliability(self,pooled,quality_penalty=None):
        b=len(next(iter(pooled.values()))); d=next(iter(pooled.values())).device; usefulness={}; rel={}
        for n in self.names:
            if self.reliability is None:
                usefulness[n]=torch.ones(b,device=d); rel[n]=torch.ones(b,device=d); continue
            logits=self._finite(self.reliability[n](pooled[n]).squeeze(-1),-20.,20.)
            usefulness[n]=torch.sigmoid(logits)
            q=torch.zeros_like(logits) if quality_penalty is None or n not in quality_penalty else quality_penalty[n].to(d).float().view(-1)
            q=torch.nan_to_num(q,nan=1.0,posinf=1.0,neginf=1.0).clamp(0.,1.)
            rel[n]=torch.sigmoid((logits-self.quality_lambda*q).clamp(-20.,20.))
        return usefulness,rel

    def _directional_scores(self,z,present,rel):
        scores={}
        for src,tgt in itertools.permutations(self.names,2):
            qs=F.normalize(self._finite(self.src_selector[src](z[src])),dim=-1,eps=1e-6); kt=F.normalize(self._finite(self.tgt_selector[tgt](z[tgt])),dim=-1,eps=1e-6)
            s=self._finite((qs*kt).sum(-1),-1.,1.)+self.reliability_src_weight*rel[src]+self.reliability_tgt_weight*rel[tgt]
            valid=present[src]*present[tgt]; scores[(src,tgt)]=torch.where(valid>0,self._finite(s,-10.,10.),torch.full_like(s,-1e4))
        return scores

    def _static_similarity_scores(self,z,present):
        return {(s,t):torch.where(present[s]*present[t]>0,self._finite((z[s]*z[t]).sum(-1),-1.,1.),torch.full_like(present[s],-1e4)) for s,t in itertools.permutations(self.names,2)}

    def _st_topk(self,mat, valid=None):
        logits=self._finite(mat/max(self.selector_temperature,1e-5),-50.,50.)
        if valid is not None: logits=torch.where(valid,logits,torch.full_like(logits,-1e4))
        if self.training and self.use_gumbel:
            u=torch.rand_like(logits).clamp_(1e-6,1-1e-6); g=-torch.log(-torch.log(u)); logits=self._finite(logits+g,-50.,50.)
            if valid is not None: logits=torch.where(valid,logits,torch.full_like(logits,-1e4))
        probs=F.softmax(logits,dim=1)
        if valid is not None: probs=probs*valid.float(); probs=probs/probs.sum(1,keepdim=True).clamp_min(1e-8)
        probs=torch.nan_to_num(probs,nan=0.0,posinf=0.0,neginf=0.0)
        k=min(self.topk,mat.shape[1]); idx=logits.topk(k,dim=1).indices; hard=torch.zeros_like(probs).scatter_(1,idx,1.)
        if valid is not None: hard=hard*valid.float()
        return (hard+probs-probs.detach(),hard) if self.training else (hard,hard)

    def _selection(self,scores,batch_size,device):
        pairs=list(scores.keys()); weights={p:torch.zeros(batch_size,device=device) for p in pairs}; hard_masks={p:torch.zeros(batch_size,dtype=torch.bool,device=device) for p in pairs}
        if self.mode=="full":
            for p in weights: weights[p].fill_(1.); hard_masks[p].fill_(True)
        elif self.mode=="random_topk":
            mat=torch.stack([scores[p] for p in pairs],1); valid=mat>-1e3; rand=torch.rand_like(mat).masked_fill(~valid,-1.); k=min(self.topk,len(pairs)); idx=rand.topk(k,dim=1).indices; hard=torch.zeros_like(mat).scatter_(1,idx,1.)*valid.float()
            for j,p in enumerate(pairs): weights[p]=hard[:,j]; hard_masks[p]=hard[:,j].bool()
        elif self.mode in {"contrastive_topk","directed_topk","similarity_topk"}:
            mat=torch.stack([scores[p] for p in pairs],1); valid=mat>-1e3
            if self.mode in {"contrastive_topk","directed_topk"}: st,hard=self._st_topk(mat,valid)
            else:
                k=min(self.topk,len(pairs)); idx=mat.topk(k,dim=1).indices; hard=torch.zeros_like(mat).scatter_(1,idx,1.)*valid.float(); st=hard
            for j,p in enumerate(pairs): weights[p]=st[:,j]; hard_masks[p]=hard[:,j].bool()
        elif self.mode!="late": raise ValueError(f"Unknown fusion mode {self.mode}")
        return weights,hard_masks

    def forward(self,modalities,present,quality_penalty=None):
        toks,pooled,z=self._encode(modalities); usefulness,rel=self._reliability(pooled,quality_penalty); scores=self._static_similarity_scores(z,present) if self.mode=="similarity_topk" else self._directional_scores(z,present,rel); b=next(iter(pooled.values())).shape[0]; dev=next(iter(pooled.values())).device; weights,hard_masks=self._selection(scores,b,dev)
        enriched={n:self._finite(pooled[n]*rel[n].unsqueeze(-1)*present[n].unsqueeze(-1)) for n in self.names}; counts={n:present[n].clone() for n in self.names}; pair_count=torch.zeros(b,device=dev)
        if self.mode!="late":
            for (src,tgt),w in weights.items():
                if not hard_masks[(src,tgt)].any() and not (self.training and w.requires_grad): continue
                y,_=self.attn[f"{src}__{tgt}"](toks[tgt],toks[src],toks[src],need_weights=False); y=self._finite(y); enriched[tgt]=self._finite(enriched[tgt]+y.mean(1)*w.unsqueeze(-1)); counts[tgt]+=w; pair_count+=hard_masks[(src,tgt)].float()
        per=[self._finite(enriched[n]/counts[n].clamp_min(1).unsqueeze(-1)) for n in self.names]; stack=torch.stack(per,1); pmask=torch.stack([present[n] for n in self.names],1).unsqueeze(-1); fused=self._finite((stack*pmask).sum(1)/pmask.sum(1).clamp_min(1.)); out=self.head(self.norm(fused)); out=self._finite(out)
        return out,{"contrastive_loss":self.contrastive_loss(z,present),"pair_count":pair_count.mean(),"reliability":rel,"usefulness":usefulness,"scores":scores,"hard_masks":hard_masks}
