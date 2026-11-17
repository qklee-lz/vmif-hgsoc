import zlib

import numpy as np
import openslide
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

TILE = 256
TARGET_MPP = 0.5
MAX_INDEX = 1024


class Slide:
    def __init__(self, path):
        self.osr = openslide.OpenSlide(str(path))
        try:
            self.mpp = float(self.osr.properties.get('openslide.mpp-x') or 0.25)
        except ValueError:
            self.mpp = 0.25
        self.level = self.osr.get_best_level_for_downsample(max(1.0, TARGET_MPP / self.mpp))
        self.ds = self.osr.level_downsamples[self.level]

    def read(self, r, c, size=TILE):
        x, y = int(c * size * self.ds), int(r * size * self.ds)
        return np.asarray(self.osr.read_region((x, y), self.level, (size, size)).convert('RGB'))

    def tissue_mask(self, ds_target=32):
        lv = self.osr.get_best_level_for_downsample(ds_target)
        w, h = self.osr.level_dimensions[lv]
        img = np.asarray(self.osr.read_region((0, 0), lv, (w, h)).convert('RGB'))
        sat = rgb2hsv(img)[..., 1]
        return sat > threshold_otsu(sat), self.osr.level_downsamples[lv]

    def grid(self, min_tissue=0.5):
        mask, mds = self.tissue_mask()
        step = TILE * self.ds / mds
        W, H = self.osr.level_dimensions[self.level]
        tiles = []
        for r in range(H // TILE):
            for c in range(W // TILE):
                a = mask[int(r * step):int((r + 1) * step), int(c * step):int((c + 1) * step)]
                if a.size and a.mean() >= min_tissue:
                    tiles.append((r, c))
        return np.array(tiles, dtype=np.int32).reshape(-1, 2)


def tile_layout(slide_id, tiles):
    patches = np.unique(tiles // 2, axis=0)
    rng = np.random.default_rng(zlib.crc32(slide_id.encode()))
    index = np.sort(rng.choice(len(tiles), min(MAX_INDEX, len(tiles)), replace=False))
    return patches, index
