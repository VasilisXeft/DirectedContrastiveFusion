import argparse
from cmf.loso import run_loso

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--subject", default=None, help="Held-out subject for a single LOSO fold.")
p.add_argument("--all", action="store_true", help="Run all LOSO folds.")
p.add_argument("--mode", default=None, help="Optional fusion-mode override.")
a = p.parse_args()
if not a.all and a.subject is None:
    print("No --subject supplied; using loso.test_subject from the config.")
run_loso(a.config, subject=a.subject, run_all=a.all, mode=a.mode)
