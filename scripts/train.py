import argparse
from cmf.train import train_from_config
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--mode',default=None); a=p.parse_args(); print(train_from_config(a.config,a.mode))
