import re

import numpy as np
import pandas as pd
import tifffile
import zarr
from scipy.ndimage import binary_closing, binary_fill_holes
from scipy.stats import rankdata
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture

from .channels import ORION

TARGET_MPP, TILE = 0.5, 512


class Section:
    def __init__(self, path):
        self.tf = tifffile.TiffFile(str(path))
        ser = self.tf.series[0]
        m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', self.tf.ome_metadata or '')
        self.mpp = float(m.group(1)) if m else 0.325
        g = zarr.open(ser.aszarr(), mode='r')
        self.levels = [g[str(i)] for i in range(len(ser.levels))] if isinstance(g, zarr.Group) else [g]
        self.axes = ser.axes
        self.yxs = self.axes.endswith('S')

    def shape(self, level=0):
        L = self.levels[level]
        return L.shape[:2] if self.yxs else L.shape[1:3]

    def read(self, level, y0, x0, h, w):
        L = self.levels[level]
        a = np.asarray(L[y0:y0 + h, x0:x0 + w, :3]) if self.yxs else np.moveaxis(np.asarray(L[:3, y0:y0 + h, x0:x0 + w]), 0, -1)
        if a.shape[0] < h or a.shape[1] < w:
            b = np.full((h, w, 3), 255, np.uint8)
            b[:a.shape[0], :a.shape[1]] = a
            a = b
        return a

    def grid(self, min_tissue=0.5):
        H0, W0 = self.shape(0)
        dss = [H0 / self.shape(i)[0] for i in range(len(self.levels))]
        li = int(np.argmin([abs(d - 16) for d in dss]))
        img = self.read(li, 0, 0, *self.shape(li))
        fg = binary_fill_holes(binary_closing((rgb2hsv(img)[..., 1] > 0.05) & (img.mean(-1) < 235), iterations=2))
        step, ds = TILE * TARGET_MPP / self.mpp, dss[li]
        tiles = [(iy, ix) for iy in range(int(H0 // step)) for ix in range(int(W0 // step))
                 if fg[int(iy * step / ds):int((iy + 1) * step / ds), int(ix * step / ds):int((ix + 1) * step / ds)].mean() >= min_tissue]
        return np.array(tiles), step


def _threshold(v, seed=0):
    x = np.log1p(v[np.isfinite(v)])
    x = x[np.random.default_rng(seed).permutation(len(x))[:200000]]
    try:
        g = GaussianMixture(2, random_state=seed).fit(x[:, None])
        lo, hi = np.argsort(g.means_.ravel())
        grid = np.linspace(np.percentile(x, 1), np.percentile(x, 99.9), 2000)
        between = (grid > g.means_.ravel()[lo]) & (grid < g.means_.ravel()[hi])
        if not between.any() or g.weights_[hi] < 0.005:
            raise ValueError
        pp = g.predict_proba(grid[:, None])
        return float(np.expm1(grid[between][np.argmin(np.abs(pp[between, hi] - 0.5))]))
    except ValueError:
        return float(np.expm1(threshold_otsu(x)))


def cell_truth(cells, tiles, step):
    thr = np.array([_threshold(cells[c].values) for c in ORION])
    iy = np.floor(cells.Y_centroid.values / step).astype(int)
    ix = np.floor(cells.X_centroid.values / step).astype(int)
    key = pd.Series(np.arange(len(tiles)), index=pd.MultiIndex.from_arrays([tiles[:, 0], tiles[:, 1]]))
    tid = key.reindex(pd.MultiIndex.from_arrays([iy, ix])).values
    ok = ~np.isnan(tid)
    tid = tid[ok].astype(int)
    X = cells[ORION].values[ok].astype(np.float64)
    P = (X > thr).astype(np.float64)
    T = len(tiles)
    n = np.bincount(tid, minlength=T).astype(float)
    mean = np.stack([np.bincount(tid, X[:, j], T) for j in range(len(ORION))], 1) / np.maximum(n, 1)[:, None]
    pos = np.stack([np.bincount(tid, P[:, j], T) for j in range(len(ORION))], 1) / np.maximum(n, 1)[:, None]
    mean[n == 0], pos[n == 0] = np.nan, np.nan
    return dict(n=n, mean=mean, pos=pos, thresholds=thr)


def _rank(a):
    return rankdata(a, axis=0)


def _z(a):
    return (a - a.mean(0)) / (a.std(0) + 1e-12)


def corr_cols(A, B):
    return _z(_rank(A)).T @ _z(_rank(B)) / len(A)


def partial_cols(A, B, z):
    Z = np.c_[np.ones(len(z)), rankdata(z)]
    res = lambda M: M - Z @ np.linalg.lstsq(Z, M, rcond=None)[0]
    return _z(res(_rank(A))).T @ _z(res(_rank(B))) / len(A)


def specificity(R):
    return float(np.nanmean([R[i, i] - np.nanmedian(np.delete(R[i], i)) for i in range(R.shape[0])]))
