#!/usr/bin/env python
"""
Linguistic features per (task, sent_idx, word_idx): wordfreq Zipf + GPT-2 surprisal
(BOS-conditioned, boundary-correct via tokenizer offset_mapping), length, content/function.
Sentences are identical & index-aligned across subjects -> compute once per task from a canonical
subject's word lists. Runs in .venv (transformers + torch). Output: artifacts/linguistic_<task>.parquet
"""
import os, re, numpy as np, pandas as pd, scipy.io as sio, torch
from wordfreq import zipf_frequency
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
OUT = f'{ROOT}/analysis/artifacts'
CANON = {'NR': 'task2_NR_matlab/resultsZDM_NR.mat', 'TSR': 'task3_TSR_matlab/resultsZDM_TSR.mat',
         'SR': 'task1_SR_matlab/resultsZDM_SR.mat'}

FUNCTION = set("""a an the this that these those of in on at to from by with for as into onto upon over under
between among through during before after about against without within is are was were be been being am
do does did have has had having will would shall should can could may might must and or but nor so yet
if then than because while although though whether he she it they we you i him her them us me his her its
their our your my mine yours theirs ours who whom whose which what not no nor there here up down out off
also very just only even more most some any all each both few many much such per""".split())

def load_words(path):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    out = []
    for s in np.atleast_1d(m['sentenceData']):
        words = [str(getattr(w, 'content', '')) for w in np.atleast_1d(s.word)]
        out.append((str(s.content), words))
    return out

def word_char_spans(text, words):
    """Locate each ZuCo word's char span in the sentence text by forward scan."""
    spans, cur = [], 0
    for w in words:
        if w == '':
            spans.append((cur, cur)); continue
        i = text.find(w, cur)
        if i < 0:                       # fallback: strip and retry
            i = text.find(w.strip(), cur)
        if i < 0:
            spans.append((None, None))
        else:
            spans.append((i, i + len(w))); cur = i + len(w)
    return spans

def main():
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').eval()
    bos = tok.bos_token_id
    for task, rel in CANON.items():
        sents = load_words(f'{ROOT}/{rel}')
        rows = []
        cover_hit = cover_tot = 0
        for si, (text, words) in enumerate(sents):
            spans = word_char_spans(text, words)
            enc = tok(text, return_offsets_mapping=True, return_tensors='pt')
            ids = enc['input_ids'][0]
            offs = enc['offset_mapping'][0].tolist()
            with torch.no_grad():
                inp = torch.cat([torch.tensor([[bos]]), enc['input_ids']], dim=1)
                logits = model(inp).logits[0]                       # [1+L, V]
            logp = torch.log_softmax(logits, dim=-1)
            # surprisal (bits) of token t (0-indexed in enc) predicted from prefix (BOS..t-1)
            surp = np.array([-(logp[t, ids[t]].item()) / np.log(2) for t in range(len(ids))])
            # map subword tokens -> word by MAX CHARACTER OVERLAP (GPT-2 leading-space safe:
            # a " company" token's offset starts at the space, so require overlap not containment)
            wsurp = np.zeros(len(words)); wntok = np.zeros(len(words), int)
            for t, (a, b) in enumerate(offs):
                if a == b: continue
                best, best_ov = -1, 0
                for wi, (ws, we) in enumerate(spans):
                    if ws is None: continue
                    ov = min(b, we) - max(a, ws)          # char overlap
                    if ov > best_ov:
                        best_ov, best = ov, wi
                if best >= 0:
                    wsurp[best] += surp[t]; wntok[best] += 1
            for wi, w in enumerate(words):
                cover_tot += 1; cover_hit += int(wntok[wi] > 0)
                clean = re.sub(r"[^A-Za-z'-]", '', w).lower()
                rows.append(dict(task=task, sent_idx=si, word_idx=wi, word=w,
                                 surprisal=wsurp[wi] if wntok[wi] > 0 else np.nan,
                                 n_subtok=int(wntok[wi]),
                                 zipf=zipf_frequency(clean, 'en') if clean else np.nan,
                                 wlen=len(clean),
                                 is_content=int(clean not in FUNCTION and len(clean) > 0)))
        df = pd.DataFrame(rows)
        df.to_parquet(f'{OUT}/linguistic_{task}.parquet')
        print(f'{task}: {len(df)} words, {df.sent_idx.nunique()} sentences, '
              f'token-coverage {cover_hit}/{cover_tot}={cover_hit/cover_tot:.3f}, '
              f'zipf n={df.zipf.notna().sum()}, surprisal n={df.surprisal.notna().sum()}', flush=True)

if __name__ == '__main__':
    main()
