import torch
from cmf.models.fusion import ContrastiveSparseFusion


def test_forward():
    shapes={'a':(32,5),'b':(32,6),'c':(32,4)}
    m=ContrastiveSparseFusion(shapes,3,d_model=32,heads=4,topk=1)
    x={k:torch.randn(2,*s) for k,s in shapes.items()}; p={k:torch.ones(2) for k in shapes}
    y,aux=m(x,p); assert y.shape==(2,3); assert 'contrastive_loss' in aux
