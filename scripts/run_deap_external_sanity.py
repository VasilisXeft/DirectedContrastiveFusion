"""Run the controlled DEAP external-feature s32 smoke experiment.

Runs Late, Full and Global Directed K=6 on the same frozen foundation features.
"""
import argparse, json
from cmf.loso import run_loso

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/deap_external.yaml")
    p.add_argument("--subject",default="s32")
    a=p.parse_args()
    experiments=[("late",None),("full",None),("directed_topk",6)]
    results={}
    for mode,k in experiments:
        print("\n"+"="*72)
        print(f"DEAP external-feature sanity: subject={a.subject} mode={mode}"+(f" K={k}" if k else ""))
        print("="*72)
        # Config carries K=6; late/full ignore the sparse budget.
        r=run_loso(a.config,subject=a.subject,run_all=False,mode=mode)[0]
        results[mode]={"best_val":r.get("best_val"),"test":r.get("test"),"outdir":r.get("outdir")}
    print("\nSANITY SUMMARY")
    print(json.dumps(results,indent=2))

if __name__=="__main__":
    main()
