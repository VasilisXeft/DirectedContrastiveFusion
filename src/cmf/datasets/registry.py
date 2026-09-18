from cmf.datasets.preprocess_synthetic import preprocess_synthetic
from cmf.datasets.preprocess_utd_mhad import preprocess_utd_mhad
from cmf.datasets.preprocess_mahnob import preprocess_mahnob
from cmf.datasets.preprocess_mmact import preprocess_mmact
from cmf.datasets.preprocess_totalcapture import preprocess_totalcapture
from cmf.datasets.preprocess_generic import preprocess_generic
from cmf.datasets.preprocess_deap import preprocess_deap

REGISTRY = {
    "generic": preprocess_generic,
    "deap": preprocess_deap,
    "synthetic": preprocess_synthetic,
    "utd_mhad": preprocess_utd_mhad,
    "mahnob_hci": preprocess_mahnob,
    "mmact": preprocess_mmact,
    "totalcapture": preprocess_totalcapture,
}


def preprocess_from_config(cfg):
    name = cfg["dataset"]["name"]
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset {name}. Available: {list(REGISTRY)}")
    return REGISTRY[name](cfg)
