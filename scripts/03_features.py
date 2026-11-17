import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vmif.cohorts import slides
from vmif.config import COHORTS
from vmif.features import build, extract_path

for cohort in COHORTS:
    sl = slides(cohort, tumour_only=False)
    ids = [s for s in sl.index if extract_path(cohort, s).exists()]
    for config in ['all', 'gigatime512', 'gigatime256', 'flash', 'hex']:
        print(cohort, config, build(cohort, ids, config), flush=True)
