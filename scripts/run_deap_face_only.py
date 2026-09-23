"""Run FACE-only LOSO diagnostic on the exact DEAP-5M sample set."""
import argparse
from cmf.loso import run_loso
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/deap_face_only.yaml"); p.add_argument("--subject",default=None); p.add_argument("--seed",type=int,default=42); a=p.parse_args()
    run_loso(a.config,subject=a.subject,run_all=(a.subject is None),mode="late",seed=a.seed,experiment_name=f"face_only_seed{a.seed}")
if __name__=="__main__": main()
