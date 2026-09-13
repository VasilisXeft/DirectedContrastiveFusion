import argparse
from cmf.config import load_config
from cmf.datasets.registry import preprocess_from_config

p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args()
cfg=load_config(a.config); out=preprocess_from_config(cfg); print(f'Prepared cache: {out}')
