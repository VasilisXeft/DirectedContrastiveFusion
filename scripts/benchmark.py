import argparse, json
from cmf.benchmark import benchmark_checkpoint
p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--cache',required=True); p.add_argument('--split',default='test'); a=p.parse_args(); print(json.dumps(benchmark_checkpoint(a.checkpoint,a.cache,a.split),indent=2))
