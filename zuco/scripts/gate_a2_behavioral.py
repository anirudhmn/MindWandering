#!/usr/bin/env python
"""GATE A2 (behavioral) — additive vs multiplicative disengagement under task-induced shallow reading.
Per subject x task, multiple regression of log(reading-time) on z(zipf)+z(surprisal)+z(wlen) -> unique
lexical (zipf) & semantic (surprisal) coupling slopes. Then paired NR-vs-TSR contrast per slope, and
the DOUBLE DISSOCIATION: is the surprisal-slope attenuation (semantic) larger than the zipf-slope
attenuation (lexical)? 12 subjects in both tasks. Also global (intercept) shift = additive component.
"""
import glob, os, numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
A=str(Path(__file__).resolve().parents[1]/'artifacts'); FS=500

def zscore(x): x=np.asarray(x,float); return (x-np.nanmean(x))/(np.nanstd(x)+1e-12)

def load(task):
    ling=pd.read_parquet(f'{A}/linguistic_{task}.parquet'); recs=[]
    for mp in sorted(glob.glob(f'{A}/frp/meta_*_{task}.parquet')):
        subj=os.path.basename(mp).split('_')[1]; m=pd.read_parquet(mp)
        m=m.merge(ling,on=['task','sent_idx','word_idx'],how='left'); m['subject']=subj
        recs.append(m)
    return pd.concat(recs,ignore_index=True)

def subj_slopes(df, ycol):
    """per subject: multiple reg y ~ z(zipf)+z(surprisal)+z(wlen); return DataFrame of slopes."""
    out=[]
    for s,g in df.groupby('subject'):
        g=g.dropna(subset=[ycol,'zipf','surprisal','wlen'])
        if len(g)<100: continue
        X=np.column_stack([zscore(g.zipf),zscore(g.surprisal),zscore(g.wlen),np.ones(len(g))])
        b,*_=np.linalg.lstsq(X,g[ycol].astype(float).values,rcond=None)
        out.append(dict(subject=s,zipf=b[0],surprisal=b[1],wlen=b[2],intercept=b[3]))
    return pd.DataFrame(out).set_index('subject')

def main():
    NR=load('NR'); TSR=load('TSR')
    for c in ['FFD','GD','TRT']:
        NR['log'+c]=np.log(NR[c].clip(lower=1)/FS*1000); TSR['log'+c]=np.log(TSR[c].clip(lower=1)/FS*1000)
    for ycol in ['logTRT','logGD','logFFD']:
        sn=subj_slopes(NR,ycol); st=subj_slopes(TSR,ycol)
        subs=sorted(set(sn.index)&set(st.index)); sn=sn.loc[subs]; st=st.loc[subs]
        print(f'\n===== {ycol}  (n={len(subs)} subjects) =====')
        print(f"{'term':10s} {'NR':>9s} {'TSR':>9s} {'Δ(TSR-NR)':>11s} {'t':>6s} {'p':>9s} {'%↓':>5s}")
        d={}
        for term in ['zipf','surprisal','intercept']:
            a=sn[term].values; b=st[term].values; dd=b-a; t,p=stats.ttest_rel(b,a)
            pctdown=100*(1-abs(b.mean())/ (abs(a.mean())+1e-9)) if term!='intercept' else np.nan
            print(f"{term:10s} {a.mean():+9.4f} {b.mean():+9.4f} {dd.mean():+11.4f} {t:6.2f} {p:9.2e} {pctdown:5.0f}")
            d[term]=dd
        # DOUBLE DISSOCIATION: |surprisal| attenuation vs |zipf| attenuation
        # attenuation = reduction in magnitude toward 0 (NR->TSR). Use signed toward-zero change.
        att_surp = np.abs(sn['surprisal'].values)-np.abs(st['surprisal'].values)  # +ve = attenuated
        att_zipf = np.abs(sn['zipf'].values)-np.abs(st['zipf'].values)
        t,p=stats.ttest_rel(att_surp,att_zipf)
        print(f"  DISSOCIATION (semantic attenuation > lexical attenuation): "
              f"surp_att={att_surp.mean():+.4f} vs zipf_att={att_zipf.mean():+.4f}  Δ={att_surp.mean()-att_zipf.mean():+.4f} t={t:.2f} p={p:.3f}")

if __name__=='__main__': main()
