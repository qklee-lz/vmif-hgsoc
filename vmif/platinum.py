import numpy as np
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PARTITIONS = [0, 1, 2, 3, 42]
SPLIT_SEED = 2026
GRIDS = {'G1': [0.01, 0.1, 1.0, 10.0], 'G2': list(10.0 ** np.arange(-4, 3))}
HP = dict(width=128, attn=64, dropout=0.25, lr=2e-4, wd=1e-2, batch=8, epochs=100, patience=15)


def folds(y, sites):
    out = [('random', p, f, te) for p in PARTITIONS
           for f, (_, te) in enumerate(StratifiedKFold(3, shuffle=True, random_state=p).split(np.zeros(len(y)), y))]
    return out + [('centre', -1, s, np.where(sites == s)[0]) for s in sorted(set(sites))]


def inner_split(tr, y):
    a, b = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=SPLIT_SEED).split(tr, y[tr]))
    return tr[a], tr[b]


class MIL(nn.Module):
    def __init__(self, d, pool, k=0):
        super().__init__()
        h, ha, p = HP['width'], HP['attn'], HP['dropout']
        self.pool = pool
        self.proj = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p))
        if pool == 'attn':
            self.V, self.U, self.w = nn.Linear(h, ha), nn.Linear(h, ha), nn.Linear(ha, 1)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(h + k, 1))

    def forward(self, x, m, c=None):
        h = self.proj(x)
        if self.pool == 'attn':
            s = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1).masked_fill(~m, float('-inf'))
            p = (torch.softmax(s, 1).unsqueeze(-1) * h).sum(1)
        else:
            mf = m.unsqueeze(-1).float()
            p = (h * mf).sum(1) / mf.sum(1)
        if c is not None:
            p = torch.cat([p, c], 1)
        return self.head(p).squeeze(-1)


class Bags:
    def __init__(self, branches, tile_patient, n_patients, y, cov=None, device='cuda'):
        self.dev = device
        order = np.argsort(tile_patient, kind='stable')
        self.counts = np.bincount(tile_patient, minlength=n_patients)
        starts = np.r_[0, np.cumsum(self.counts)[:-1]]
        tmax = self.counts.max()
        self.X = {}
        for name, X in branches.items():
            X = X[order]
            T = torch.zeros(n_patients, tmax, X.shape[1], dtype=torch.float16 if X.dtype == np.float16 else torch.float32, device=device)
            for i in range(n_patients):
                T[i, :self.counts[i]] = torch.from_numpy(np.asarray(X[starts[i]:starts[i] + self.counts[i]])).to(device)
            self.X[name] = T
        self.mask = torch.zeros(n_patients, tmax, dtype=torch.bool, device=device)
        for i in range(n_patients):
            self.mask[i, :self.counts[i]] = True
        self.y = y
        self.yt = torch.tensor(y, dtype=torch.float32, device=device)
        self.cov = None if cov is None else torch.tensor(cov, dtype=torch.float32, device=device)

    def stats(self, b, idx):
        sel = torch.as_tensor(idx, device=self.dev)
        X, M = self.X[b][sel].float(), self.mask[sel].unsqueeze(-1)
        mu = (X * M).sum((0, 1)) / M.sum()
        return mu, ((((X - mu) ** 2) * M).sum((0, 1)) / M.sum()).sqrt() + 1e-6

    def batches(self, idx, rng=None):
        idx = np.array(idx)
        if rng is not None:
            idx = idx[rng.permutation(len(idx))]
        for k in range(0, len(idx), HP['batch']):
            bi = idx[k:k + HP['batch']]
            yield bi, torch.as_tensor(bi, device=self.dev), int(self.counts[bi].max())


def fit_predict(bags, arm, pool, init, tr, va, te, fixed_epochs=0):
    use_cov = arm.endswith('+cov')
    br = arm.replace('+cov', '').split('+')
    st = {b: bags.stats(b, tr) for b in br}
    if use_cov:
        cm = bags.cov[tr].mean(0)
        cs = bags.cov[tr].std(0, unbiased=False)
        cs = torch.where(cs > 0, cs, torch.ones_like(cs))
    y = bags.y
    w_res = (y[tr] == 1).sum() / (y[tr] == 0).sum()

    def loss(lg, yy):
        w = torch.where(yy == 0, torch.as_tensor(w_res, device=bags.dev, dtype=torch.float32), torch.ones_like(yy))
        return (nn.functional.binary_cross_entropy_with_logits(lg, yy, reduction='none') * w).sum(), w.sum()

    def forward(model, sel, t):
        x = torch.cat([(bags.X[b][sel, :t].float() - st[b][0]) / st[b][1] for b in br], -1)
        return model(x, bags.mask[sel, :t], (bags.cov[sel] - cm) / cs if use_cov else None)

    torch.manual_seed(init)
    model = MIL(sum(bags.X[b].shape[-1] for b in br), pool, bags.cov.shape[1] if use_cov else 0).to(bags.dev)
    opt = torch.optim.AdamW(model.parameters(), lr=HP['lr'], weight_decay=HP['wd'])
    rng = np.random.default_rng(init)
    best, best_ep, best_state, wait = np.inf, -1, None, 0
    max_ep = fixed_epochs or HP['epochs']
    for ep in range(1, max_ep + 1):
        model.train()
        for _, sel, t in bags.batches(tr, rng):
            l, ws = loss(forward(model, sel, t), bags.yt[sel])
            l = l / ws
            if not torch.isfinite(l):
                continue
            opt.zero_grad()
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            L = W = 0.0
            for _, sel, t in bags.batches(va):
                l, ws = loss(forward(model, sel, t), bags.yt[sel])
                L, W = L + l.item(), W + ws.item()
        vl = L / W
        if fixed_epochs:
            if ep == max_ep:
                best, best_ep, best_state = vl, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
            continue
        if vl < best - 1e-6:
            best, best_ep, wait = vl, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= HP['patience']:
                break
    model.load_state_dict(best_state)
    model.eval()
    out = np.zeros(len(te))
    with torch.no_grad():
        for k, (bi, sel, t) in enumerate(bags.batches(te)):
            out[k * HP['batch']:k * HP['batch'] + len(bi)] = forward(model, sel, t).cpu().numpy()
    return out, dict(epochs=ep, best_epoch=best_ep, val_loss=best)


def probe(X, y, tr, te, grid, seed=0):
    pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=5000))
    gs = GridSearchCV(pipe, {'logisticregression__C': GRIDS[grid]}, scoring='roc_auc', cv=StratifiedKFold(3, shuffle=True, random_state=seed))
    gs.fit(X[tr], y[tr])
    return gs.decision_function(X[te])


def probe_random(X, y, grid):
    out = {}
    for p in PARTITIONS:
        oof = np.full(len(y), np.nan)
        for tr, te in StratifiedKFold(3, shuffle=True, random_state=p).split(X, y):
            oof[te] = probe(X, y, tr, te, grid, seed=p)
        out[p] = oof
    return out


def probe_centre(X, y, sites, grid):
    oof = np.full(len(y), np.nan)
    for s in sorted(set(sites)):
        te, tr = np.where(sites == s)[0], np.where(sites != s)[0]
        if len(np.unique(y[tr])) == 2:
            oof[te] = probe(X, y, tr, te, grid)
    return oof
