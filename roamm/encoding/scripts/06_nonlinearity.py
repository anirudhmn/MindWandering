#!/usr/bin/env python3
"""Is the mapping from word properties to the fixation response nonlinear?

Both arms share a learned rank-R spatiotemporal basis and differ only in how the component
amplitudes are produced from the word properties:

    linear     A = x W          a rank-R linear kernel, the fair baseline
    nonlinear  A = MLP(x)

so a difference in held-out D isolates nonlinearity rather than capacity in the kernel. Overlap
is handled as in the linear model: a run's prediction is the sum of every fixation's kernel, and
the loss is taken on the residual recording from 02.

Restricted to the occipitotemporal and centroparietal channels, where the response lives; the
full 64-channel target does not fit in accelerator memory for a pooled fit. Folds are five
articles crossed with four reader groups, so every evaluated cell is held out on both.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, RES, OCC_I, CP_I, TEXT_BASE, boot_ci

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
NT, NL = len(TEXT_BASE), 129
CHS = sorted(set(OCC_I) | set(CP_I))
NCH = len(CHS)
RESID = ART / "resid"


class KernelNet(nn.Module):
    def __init__(self, R=8, nonlinear=True, h=32):
        super().__init__()
        self.amp = (nn.Sequential(nn.Linear(NT, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                                  nn.Linear(h, R)) if nonlinear else nn.Linear(NT, R, bias=False))
        self.B = nn.Parameter(torch.randn(R, NL, NCH) * 0.01)

    def kernels(self, x):
        return torch.einsum("er,rlc->elc", self.amp(x), self.B)


def run_pred(K, onsets, runlen, lagrange):
    """Overlap-sum the event kernels into a continuous run prediction, in one scatter."""
    pred = torch.zeros(runlen + NL, NCH, device=K.device, dtype=K.dtype)
    idx = (onsets[:, None] + lagrange[None, :]).reshape(-1)
    pred.index_add_(0, idx, K.reshape(-1, NCH))
    return pred[:runlen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-3)
    a = ap.parse_args()

    ev = pd.read_parquet(ART / "events.parquet")
    mu, sd = ev[TEXT_BASE].mean().to_numpy(), ev[TEXT_BASE].std().to_numpy()
    cells = []
    for f in sorted(RESID.glob("s*_r*.npz")):
        d = np.load(f, allow_pickle=True)
        cells.append(dict(subj=int(f.stem.split("_")[0][1:]), story=str(d["story"]),
                          X=((d["text"].astype(np.float64) - mu) / sd).astype(np.float32),
                          mw=d["mw"].astype(np.float32), rel=d["onset_rel"].astype(np.int64),
                          runlen=int(d["runlen"]),
                          resid=d["resid"][:, :, CHS].astype(np.float32)))
    subs = sorted({c["subj"] for c in cells})
    stories = sorted({c["story"] for c in cells})
    grp = {s: i % 4 for i, s in enumerate(subs)}
    print(f"{len(cells)} cells, {NCH} channels, device {DEV}", flush=True)

    def to_dev(c):
        return (torch.tensor(c["X"], device=DEV), torch.tensor(c["rel"], device=DEV),
                torch.tensor(c["resid"], device=DEV), torch.tensor(c["mw"], device=DEV))

    out = {}
    for mode in ("linear", "nonlinear"):
        Ds, sjs, mws = [], [], []
        t0 = time.time()
        LR = torch.arange(NL, device=DEV)
        for st in stories:
            for g in range(4):
                te = [c for c in cells if c["story"] == st and grp[c["subj"]] == g]
                trn = [c for c in cells if c["story"] != st and grp[c["subj"]] != g]
                if not te:
                    continue
                torch.manual_seed(0)
                m = KernelNet(a.rank, mode == "nonlinear").to(DEV)
                opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-5)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(trn))
                for _ in range(a.epochs):
                    np.random.shuffle(trn)
                    for c in trn:
                        X, rel, R, mwv = to_dev(c)
                        on = mwv == 0
                        if on.sum() < 20:
                            continue
                        pred = run_pred(m.kernels(X[on]), rel[on], c["runlen"], LR)
                        rows = rel[:, None] + LR[None, :]
                        loss = ((R - pred[rows]) ** 2)[on].mean()
                        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
                        del X, rel, R, mwv, pred
                with torch.no_grad():
                    for c in te:
                        X, rel, R, _ = to_dev(c)
                        pred = run_pred(m.kernels(X), rel, c["runlen"], LR)
                        rows = rel[:, None] + LR[None, :]
                        P = pred[rows]
                        Ds.append((R ** 2 - (R - P) ** 2).mean(dim=(1, 2)).cpu().numpy())
                        sjs.append(np.full(len(c["rel"]), c["subj"]))
                        mws.append(c["mw"])
                print(f"  [{mode}] {st} group {g} [{time.time()-t0:.0f}s]", flush=True)
        D, sj, mw = np.concatenate(Ds), np.concatenate(sjs), np.concatenate(mws)
        don = np.array([D[(sj == s) & (mw == 0)].mean() for s in subs])
        gate = boot_ci(don)
        out[mode] = dict(D_on=float(don.mean()), t=gate["t"], p=gate["p"],
                         readers_positive=int((don > 0).sum()))
        np.savez_compressed(ART / f"nonlinearity_{mode}.npz", D=D, subject=sj, mw=mw)
        print(f"{mode}: D_on={don.mean():+.6f} (t={gate['t']:.1f})", flush=True)

    dl = np.load(ART / "nonlinearity_linear.npz")
    dn = np.load(ART / "nonlinearity_nonlinear.npz")
    sA = np.unique(dl["subject"])
    gl = np.array([dl["D"][(dl["subject"] == s) & (dl["mw"] == 0)].mean() for s in sA])
    gn = np.array([dn["D"][(dn["subject"] == s) & (dn["mw"] == 0)].mean() for s in sA])
    t, p = stats.ttest_rel(gn, gl)
    out["nonlinear_minus_linear"] = dict(delta=float((gn - gl).mean()), t=float(t), p=float(p),
                                         readers_better=int((gn > gl).sum()), n_readers=int(len(sA)))
    print(f"nonlinear - linear = {(gn-gl).mean():+.6f} uV^2  t={t:.2f} p={p:.3g} "
          f"({int((gn>gl).sum())}/{len(sA)} readers)", flush=True)
    (RES / "nonlinearity.json").write_text(json.dumps(out, indent=2))
    print("wrote", RES / "nonlinearity.json")


if __name__ == "__main__":
    main()
