import argparse
from cmf.train import train_from_config
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args()
for mode in ['late','full','random_topk','similarity_topk','directed_topk','contrastive_topk']:
    print(f'\n===== {mode} =====')
    train_from_config(a.config,mode)
