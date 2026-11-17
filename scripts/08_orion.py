import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vmif import decoders as D
from vmif import orion as O
from vmif.channels import GIGA, HEX, ORION, ORION_MAP
from vmif.config import data, work
from vmif.stats import boot_median

ROOT = data('orion')
OUT = work('results', 'orion')
MIN_CELLS, N_PERM = 20, 200
DECODERS = {'giga512': ('giga512', GIGA, 'giga'), 'giga256': ('giga256', GIGA, 'giga'), 'flash': ('flash', GIGA, 'giga'),
            'hex': ('hex', HEX, 'hex')}
nets = {'giga': D.load_gigatime(), 'flash': D.load_flash(), 'hex': D.load_hex()}
rng = np.random.default_rng(2026)
COMMON = list(ORION_MAP['giga'])  # specificity S uses the ten channels shared by all decoders


def infer(sec, tiles, step, batch=32):
    out = {k: [] for k in DECODERS}
    n = int(np.ceil(step))
    for i in range(0, len(tiles), batch):
        imgs = [np.asarray(Image.fromarray(sec.read(0, int(round(y * step)), int(round(x * step)), n, n)).resize((512, 512), Image.BILINEAR))
                for y, x in tiles[i:i + batch]]
        x = D.to_tensor(imgs)
        w, b = D.windows(x)
        out['giga512'].append(D.gigatime_mean(nets['giga'], x).cpu().numpy())
        out['giga256'].append(D.gigatime_mean(nets['giga'], w).view(b, 4, -1).mean(1).cpu().numpy())
        out['flash'].append(D.gigatime_mean(nets['flash'], w).view(b, 4, -1).mean(1).cpu().numpy())
        out['hex'].append(D.hex_pred(nets['hex'], w).view(b, 4, -1).mean(1).cpu().numpy())
    torch.cuda.empty_cache()
    return {k: np.concatenate(v) for k, v in out.items()}


rows, srows = [], []
oi = {c: i for i, c in enumerate(ORION)}
for folder in sorted(p for p in ROOT.glob('CRC*') if p.is_dir()):
    he = next(folder.glob('*registered.ome.tif'))
    cells = pd.read_csv(next(p for p in folder.glob('*.csv') if not p.name.endswith('markers.csv')), usecols=ORION + ['X_centroid', 'Y_centroid'])
    sec = O.Section(he)
    tiles, step = sec.grid()
    truth = O.cell_truth(cells, tiles, step)
    V = infer(sec, tiles, step)
    for dname, (key, chs, mapkey) in DECODERS.items():
        mp = ORION_MAP[mapkey]
        vi, ti = [chs.index(c) for c in mp], [oi[mp[c]] for c in mp]
        for kind in ['mean', 'pos']:
            ok = (truth['n'] >= MIN_CELLS) & np.isfinite(truth[kind][:, ti]).all(1)
            X, Y, logn = V[key][ok][:, vi], truth[kind][ok][:, ti], np.log(truth['n'][ok])
            R, Rp = O.corr_cols(X, Y), O.partial_cols(X, Y, logn)
            Rnull = np.array([np.diag(O.corr_cols(X, Y[rng.permutation(len(Y))])) for _ in range(N_PERM)])
            vc, tc = [chs.index(c) for c in COMMON], [oi[ORION_MAP['giga'][c]] for c in COMMON]
            okc = (truth['n'] >= MIN_CELLS) & np.isfinite(truth[kind][:, tc]).all(1)
            Xc, Yc, lognc = V[key][okc][:, vc], truth[kind][okc][:, tc], np.log(truth['n'][okc])
            S = O.specificity(O.corr_cols(Xc, Yc))
            Snull = [O.specificity(O.corr_cols(Xc, Yc[rng.permutation(len(Yc))])) for _ in range(N_PERM)]
            srows.append(dict(section=folder.name, decoder=dname, truth=kind, n_tiles=int(okc.sum()), S=S,
                              S_partial=O.specificity(O.partial_cols(Xc, Yc, lognc)),
                              p=(1 + np.sum(np.array(Snull) >= S)) / (1 + N_PERM)))
            rows += [dict(section=folder.name, decoder=dname, truth=kind, channel=c, rho=R[j, j], rho_partial=Rp[j, j],
                          p=(1 + np.sum(Rnull[:, j] >= R[j, j])) / (1 + N_PERM)) for j, c in enumerate(mp)]
    print(folder.name, len(tiles), flush=True)

D_, S_ = pd.DataFrame(rows), pd.DataFrame(srows)
D_.to_csv(OUT / 'per_section_channel.csv', index=False)
S_.to_csv(OUT / 'per_section_S.csv', index=False)
summary = [dict(decoder=d, truth=t, channel=c, **dict(zip(['rho', 'lo', 'hi'], boot_median(g.rho))),
                partial=boot_median(g.rho_partial)[0]) for (d, t, c), g in D_.groupby(['decoder', 'truth', 'channel'])]
summary += [dict(decoder=d, truth=t, channel='S', **dict(zip(['rho', 'lo', 'hi'], boot_median(g.S))), partial=boot_median(g.S_partial)[0])
            for (d, t), g in S_.groupby(['decoder', 'truth'])]
pd.DataFrame(summary).to_csv(OUT / 'summary.csv', index=False)
