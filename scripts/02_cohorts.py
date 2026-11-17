import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vmif.cohorts import build_ptrc, build_tcga

for name, fn in [('PTRC-HGSOC', build_ptrc), ('TCGA-OV', build_tcga)]:
    sl, truth = fn()
    print(name, 'slides', len(sl), 'patients', len(truth), 'responders', int(truth.label.sum()))
