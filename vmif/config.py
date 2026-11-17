import os
from pathlib import Path

import yaml

_CFG = Path(os.environ.get('VMIF_CONFIG', Path(__file__).resolve().parents[1] / 'config.yaml'))
CFG = yaml.safe_load(_CFG.read_text())

DATA = Path(CFG['data_root'])
WORK = Path(CFG['work_root'])
MODELS = {k: Path(v) for k, v in CFG['models'].items()}
COHORTS = ['PTRC-HGSOC', 'TCGA-OV']
SEED = 2026
N_JOBS = int(os.environ.get('VMIF_JOBS', CFG.get('n_jobs', 8)))
EXCLUDE = set(CFG.get('exclude_slides') or [])


def work(*parts):
    p = WORK.joinpath(*parts)
    (p.parent if p.suffix else p).mkdir(parents=True, exist_ok=True)
    return p


def data(*parts):
    return DATA.joinpath(*parts)
