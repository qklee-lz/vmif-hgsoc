import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vmif import decoders as D
from vmif.config import data
from vmif.features import extract_path
from vmif.slides import Slide, tile_layout

ap = argparse.ArgumentParser()
ap.add_argument('cohort', choices=['PTRC-HGSOC', 'TCGA-OV'])
ap.add_argument('--batch', type=int, default=64)
args = ap.parse_args()

nets = {'uni': D.load_uni(), 'giga': D.load_gigatime(), 'flash': D.load_flash(), 'hex': D.load_hex(), 'musk': D.load_musk()}
folder = data('ptrc' if args.cohort == 'PTRC-HGSOC' else 'tcga', 'slides')


def run(slide, coords, size, fns):
    out = {k: [] for k in fns}
    for i in range(0, len(coords), args.batch):
        x = D.to_tensor([slide.read(r, c, size) for r, c in coords[i:i + args.batch]])
        for k, f in fns.items():
            out[k].append(f(x).cpu().numpy())
    return {k: np.concatenate(v) if v else np.zeros((0,)) for k, v in out.items()}


for path in sorted(folder.glob('*.svs')):
    dst = extract_path(args.cohort, path.stem)
    if dst.exists():
        continue
    s = Slide(path)
    tiles = s.grid()
    if len(tiles) == 0:
        continue
    patches, index = tile_layout(path.stem, tiles)
    uni = run(s, tiles, 256, {'uni': lambda x: D.uni_feat(nets['uni'], x)})['uni']
    g512 = run(s, patches, 512, {'g': lambda x: D.gigatime_mean(nets['giga'], x)})['g']
    rest = run(s, tiles[index], 256, {
        'giga256': lambda x: D.gigatime_mean(nets['giga'], x),
        'flash': lambda x: D.gigatime_mean(nets['flash'], x),
        'hex': lambda x: D.hex_pred(nets['hex'], x),
        'musk': lambda x: D.musk_feat(nets['musk'], x),
    })
    np.savez_compressed(dst, tiles=tiles, patches=patches, index=index, mpp=s.mpp, level=s.level,
                        uni=uni.astype(np.float16), giga512=g512.astype(np.float32), giga256=rest['giga256'],
                        flash=rest['flash'], hex=rest['hex'], musk=rest['musk'].astype(np.float16))
    print(path.stem, len(tiles), len(patches), len(index), flush=True)
    torch.cuda.empty_cache()
