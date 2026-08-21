#!/usr/bin/env python3
"""Fit the reading policy and read out what the text buys, per transition.

Two heads, each given a geometry block and, in the compared model, a text block:

  target   scores the 41 candidate words, trained with a softmax over the set
  duration predicts log fixation duration

Both are trained on ON-TASK transitions only and evaluated doubly held out: five article folds
crossed with four reader groups, so each of the 20 fitted models is tested on transitions from
an article and from readers it never saw.

The target head is compared as two separately trained models. The duration head is two-stage --
a geometry network is fitted first and the text block is fitted to its held-out residual --
because the lexical contribution to duration is about 0.2% of the variance while the
seed-to-seed spread of two independently trained networks is larger than that.

Read-outs, per transition:
  target   log2 p_text(true) - log2 p_notext(true)                          [bits]
  duration r^2 - (r - text prediction)^2, r the geometry residual           [log-ms^2]

--shuffle SEED permutes which word occupies which position within each page, preserving layout,
timing and the reader's own movements: the negative control.

Needs the tables from 00_build_transitions.py.
"""
from __future__ import annotations
import argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, TEXT, W, NC

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def build(shuffle=0):
    tr = pd.read_parquet(ART / "policy_trans.parquet")
    wd = pd.read_parquet(ART / "policy_words.parquet")
    cand = np.load(ART / "policy_cand_widx.npy")
    prior = np.load(ART / "policy_cand_prior.npy").astype(np.float32)

    for c in TEXT:
        wd[c] = wd[c].astype(float)
        wd[c] = wd[c].fillna(wd[c].median())
    if shuffle:
        rng = np.random.default_rng(shuffle)
        for _, g in wd.groupby(["story", "page"]):
            wd.loc[g.index, TEXT] = g[TEXT].to_numpy()[rng.permutation(len(g))]
        print(f"negative control: word features permuted within page (seed {shuffle})", flush=True)

    WT = wd[TEXT].to_numpy(np.float32)
    WT = (WT - WT.mean(0)) / (WT.std(0) + 1e-9)
    WG = wd[["line_pos", "line_len", "center_x", "center_y", "page_max_pos", "pos"]].to_numpy(np.float32)

    n = len(tr)
    widx = wd.set_index("word_key")["widx"].reindex(tr["word_key"]).to_numpy()
    valid = cand >= 0
    ci = np.where(valid, cand, 0)
    x = tr["x"].to_numpy(np.float32)
    y = tr["y"].to_numpy(np.float32)

    G = np.column_stack([
        tr["page_prog"], tr["line"], tr["line_pos"], tr["line_len"],
        (tr["line_len"] - tr["line_pos"]), x / 1000., y / 1000.,
        (x - tr["center_x"].to_numpy(np.float32)) / 100.,
        (y - tr["center_y"].to_numpy(np.float32)) / 100.,
        tr["log_in_amp"], tr["log_prev_dur"], tr["first_pass"].astype(float),
        tr["fix_order_all"] / 100., prior[np.arange(n), W], tr["page_max_pos"],
    ]).astype(np.float32)
    G = (G - G.mean(0)) / (G.std(0) + 1e-9)

    pos = tr["pos"].to_numpy()
    mx = tr["page_max_pos"].to_numpy()
    base = widx - pos
    TD = np.concatenate([WT[base + np.clip(pos + o, 0, mx)] for o in (0, -1, 1, 2)], 1).astype(np.float32)

    cx, cy = WG[ci, 2], WG[ci, 3]
    offs = np.broadcast_to(np.arange(-W, W + 1, dtype=np.float32)[None, :], (n, NC))
    dx = (cx - x[:, None]) / 1000.
    dy = (cy - y[:, None]) / 1000.
    CG = np.stack([
        offs / 10., np.abs(offs) / 10., (offs > 0).astype(np.float32), dx, dy,
        np.sqrt(dx ** 2 + dy ** 2),
        (np.abs(cy - tr["center_y"].to_numpy(np.float32)[:, None]) < 20).astype(np.float32),
        WG[ci, 0] / 10., (WG[ci, 1] - WG[ci, 0]) / 10., WG[ci, 5] / np.clip(WG[ci, 4], 1, None),
        np.log1p(prior), valid.astype(np.float32),
    ], -1).astype(np.float32)
    Gb = np.broadcast_to(G[:, None, :], (n, NC, G.shape[1]))
    Ca = np.ascontiguousarray(np.concatenate([CG, Gb], -1))
    Cb = np.ascontiguousarray(np.concatenate(
        [CG, Gb, WT[ci], np.broadcast_to(WT[widx][:, None, :], (n, NC, len(TEXT)))], -1))
    del CG, Gb
    return tr, G, TD, Ca, Cb, valid


class MLP(nn.Module):
    def __init__(self, d, h=96):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.f(x)


def fit_duration(Xtr, ytr, Xte_list, epochs, seed, bs=4096, lr=3e-3):
    """Geometry network for log fixation duration. The target is centred on the training mean:
    without that the network spends its budget on a constant offset of about 5.5 log-ms."""
    torch.manual_seed(seed)
    mu = float(ytr.mean())
    m = MLP(Xtr.shape[1]).to(DEV)
    Xt = torch.tensor(Xtr, device=DEV)
    yt = torch.tensor(ytr - mu, device=DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    ns = int(np.ceil(len(Xt) / bs))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * ns)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=DEV)
        for s in range(ns):
            i = perm[s * bs:(s + 1) * bs]
            loss = ((m(Xt[i]).squeeze(-1) - yt[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval()
    out = []
    with torch.no_grad():
        for Z in Xte_list:
            o = [m(torch.tensor(Z[s:s + 65536], device=DEV)).squeeze(-1).cpu().numpy()
                 for s in range(0, len(Z), 65536)]
            out.append(np.concatenate(o) + mu)
    return out


def text_stage(TDtr, rtr, TDte, seed, epochs=60, lam=None):
    """Stage two: predict the geometry residual from the text block, ridge and network."""
    Xtr = np.column_stack([np.ones(len(TDtr)), TDtr])
    Xte = np.column_stack([np.ones(len(TDte)), TDte])
    XtX = Xtr.T @ Xtr
    lam = lam if lam is not None else 1e-3 * np.trace(XtX) / XtX.shape[0]
    w = np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), Xtr.T @ rtr)
    torch.manual_seed(seed + 1000)
    m = MLP(TDtr.shape[1], h=64).to(DEV)
    Xt = torch.tensor(TDtr.astype(np.float32), device=DEV)
    yt = torch.tensor(rtr.astype(np.float32), device=DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-3)
    bs = 8192
    ns = int(np.ceil(len(Xt) / bs))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * ns)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=DEV)
        for s in range(ns):
            i = perm[s * bs:(s + 1) * bs]
            loss = ((m(Xt[i]).squeeze(-1) - yt[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval()
    with torch.no_grad():
        mlp = np.concatenate([m(torch.tensor(TDte[s:s + 65536].astype(np.float32),
                                             device=DEV)).squeeze(-1).cpu().numpy()
                              for s in range(0, len(TDte), 65536)])
    return Xte @ w, mlp


def fit_target(Xtr, vtr, ytr, Xte, vte, yte, epochs, seed, bs=1024, lr=2e-3):
    """Candidate scorer; returns held-out log p of the word actually fixated next."""
    torch.manual_seed(seed)
    m = MLP(Xtr.shape[-1]).to(DEV)
    Xt = torch.tensor(Xtr, device=DEV)
    vt = torch.tensor(vtr, device=DEV)
    yt = torch.tensor(ytr, device=DEV, dtype=torch.long)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    ns = int(np.ceil(len(Xt) / bs))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * ns)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=DEV)
        for s in range(ns):
            i = perm[s * bs:(s + 1) * bs]
            lg = m(Xt[i]).squeeze(-1).masked_fill(~vt[i], -1e9)
            loss = lossf(lg, yt[i])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(Xte), 8192):
            lg = m(torch.tensor(Xte[s:s + 8192], device=DEV)).squeeze(-1)
            lg = lg.masked_fill(~torch.tensor(vte[s:s + 8192], device=DEV), -1e9)
            lp = torch.log_softmax(lg, -1)
            j = torch.tensor(yte[s:s + 8192], device=DEV, dtype=torch.long)[:, None]
            out.append(lp.gather(1, j).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8, help="target head")
    ap.add_argument("--dur-epochs", type=int, default=40, help="duration geometry network")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", type=int, default=0, help="seed > 0 runs the negative control")
    ap.add_argument("--tag", default="real")
    a = ap.parse_args()

    tr, G, TD, Ca, Cb, valid = build(shuffle=a.shuffle)
    n = len(tr)
    print(f"n={n}  candidate tensors {Ca.nbytes/1e9:.2f}+{Cb.nbytes/1e9:.2f} GB  device {DEV}", flush=True)

    subs = np.sort(tr.subject.unique())
    tr["grp"] = tr["subject"].map({s: i % 4 for i, s in enumerate(subs)})
    ontask = tr["mw"].to_numpy() == 0
    y = tr["log_fix_dur"].to_numpy(np.float32)
    tgt = tr["target_idx"].to_numpy(np.int64)
    story_a, grp_a = tr["story"].to_numpy(), tr["grp"].to_numpy()

    Dtgt = np.full(n, np.nan)
    Ddur = np.full(n, np.nan)
    Ddur_mlp = np.full(n, np.nan)
    lp_t = np.full(n, np.nan)
    lp_n = np.full(n, np.nan)
    pred_geom = np.full(n, np.nan)
    pred_text = np.full(n, np.nan)

    t0 = time.time()
    for k, st in enumerate(np.sort(tr.story.unique())):
        for g in range(4):
            te = (story_a == st) & (grp_a == g)
            trn = (story_a != st) & (grp_a != g) & ontask
            if te.sum() == 0:
                continue
            gtr, gte = fit_duration(G[trn], y[trn], [G[trn], G[te]], a.dur_epochs, a.seed)
            rte = y[te] - gte
            lin, mlp = text_stage(TD[trn], y[trn] - gtr, TD[te], a.seed)
            Ddur[te] = rte ** 2 - (rte - lin) ** 2
            Ddur_mlp[te] = rte ** 2 - (rte - mlp) ** 2
            pred_geom[te] = gte
            pred_text[te] = gte + lin

            la = fit_target(Ca[trn], valid[trn], tgt[trn], Ca[te], valid[te], tgt[te], a.epochs, a.seed)
            lb = fit_target(Cb[trn], valid[trn], tgt[trn], Cb[te], valid[te], tgt[te], a.epochs, a.seed)
            lp_n[te], lp_t[te] = la / np.log(2), lb / np.log(2)
            Dtgt[te] = (lb - la) / np.log(2)
            print(f"  {st} g{g}: n_te={te.sum():6d} Dtgt={np.nanmean(Dtgt[te]):+.4f} bits  "
                  f"Ddur={np.nanmean(Ddur[te]):+.6f}  [{time.time()-t0:.0f}s]", flush=True)

    np.savez_compressed(
        ART / f"policy_D_{a.tag}.npz",
        Dtgt=Dtgt, Ddur=Ddur, Ddur_mlp=Ddur_mlp, lp_text=lp_t, lp_notext=lp_n,
        pred_geom=pred_geom, pred_text=pred_text, TD=TD.astype(np.float32),
        subject=tr.subject.to_numpy(), story=story_a.astype(str), mw=tr.mw.to_numpy(),
        block_id=tr.block_id.to_numpy(), word_key=tr.word_key.to_numpy().astype(str),
        pos=tr.pos.to_numpy(), page=tr.page.to_numpy(), line=tr.line.to_numpy(),
        line_pos=tr.line_pos.to_numpy(), log_fix_dur=y, fix_dur=tr.fix_dur.to_numpy(),
        target_idx=tgt, kind=tr["kind"].to_numpy().astype(str))
    print("saved %s  Dtgt=%.4f bits  Ddur=%.6f  Ddur_mlp=%.6f"
          % (f"policy_D_{a.tag}.npz", np.nanmean(Dtgt), np.nanmean(Ddur), np.nanmean(Ddur_mlp)))


if __name__ == "__main__":
    main()
