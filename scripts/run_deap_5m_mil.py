"""Run DEAP trial-level MIL, one subject or all LOSO folds."""
import argparse
import json
from cmf.train_mil import run_loso_mil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deap_5m_mil.yaml")
    parser.add_argument("--subject", default=None, help="Held-out subject, e.g. s22; omit for all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_loso_mil(args.config, subject=args.subject, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
