#!/usr/bin/env python
"""
2x2 transfer test — WORD-LEVEL, matched gaze-duration measures across datasets.
ROAMM per-fixation is aggregated to word level: GD=sum first-pass fix_dur, TRT=sum all, FFD=first fix,
first-fixation FRP ROI (occ_N1 freq, cp_N400 surprisal), word MW state = first-pass first fixation is_mw.
ZuCo already word-level (FFD/GD/TRT + first-fix FRP occ/cp window means). Coupling = per-subject bivariate
slope on z(predictor) in engaged vs disengaged; retention=|beta_dis|/|beta_eng|; interaction paired t.
"""
import os, glob, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); from scipy import stats
from pathlib import Path
_R=Path(__file__).resolve().parents[2]
RO=str(_R/'roamm/artifacts/coupling')
ZA=str(_R/'zuco/artifacts')
import gate_a0 as G; occ,cp=G.rois(); FS_Z=500; PRE=50
def ms2i(ms): return PRE+int(round(ms/1000*FS_Z))
OCCW=slice(ms2i(150),ms2i(290)); CPW=slice(ms2i(300),ms2i(450)); BASE=slice(0,PRE)
def zc(x): x=np.asarray(x,float); return (x-np.nanmean(x))/(np.nanstd(x)+1e-12)

def roamm_word():
    fx=pd.read_parquet(f'{RO}/fixations_frp.parquet'); wf=pd.read_parquet(f'{RO}/word_features.parquet')
    fx=fx.sort_values(['subject','word_key','fix_order'])
    fp=fx[fx.is_firstpass==1]
    # gaze duration (first-pass) & first fixation info
    gd=fp.groupby(['subject','word_key']).agg(GD=('fix_dur','sum'),FFD=('fix_dur','first'),
        occ_N1=('frp_occ_N1','first'),cp_N400=('frp_cp_N400','first'),is_mw=('is_mw','first'),
        mw_frac=('mw_frac','first')).reset_index()
    trt=fx.groupby(['subject','word_key']).agg(TRT=('fix_dur','sum')).reset_index()
    d=gd.merge(trt,on=['subject','word_key']).merge(wf[['word_key','zipf','surprisal']],on='word_key',how='left')
    for c in ['GD','FFD','TRT']: d['log'+c]=np.log(d[c].clip(lower=1))
    d['state']=d.is_mw.astype(int)
    return d

def zuco_word():
    recs=[]
    for task in ['NR','TSR']:
        ling=pd.read_parquet(f'{ZA}/linguistic_{task}.parquet')
        for mp in sorted(glob.glob(f'{ZA}/frp/meta_*_{task}.parquet')):
            subj=os.path.basename(mp).split('_')[1]; m=pd.read_parquet(mp)
            frp=np.load(mp.replace('meta_','frp_').replace('.parquet','.npy'))
            frp=frp-np.nanmean(frp[:,:,BASE],axis=2,keepdims=True)
            m=m.copy(); m['occ_N1']=np.nanmean(frp[:,occ,:][:,:,OCCW],axis=(1,2))
            m['cp_N400']=np.nanmean(frp[:,cp,:][:,:,CPW],axis=(1,2)); m['subject']=subj; m['state']=0 if task=='NR' else 1
            for c in ['FFD','GD','TRT']: m['log'+c]=np.log((m[c]/FS_Z*1000).clip(lower=1))
            recs.append(m.merge(ling,on=['task','sent_idx','word_idx'],how='left'))
    return pd.concat(recs,ignore_index=True)

def slopes(df,ycol,xcol,minn=80):
    r=[]
    for s,g in df.groupby('subject'):
        ge=g[g.state==0].dropna(subset=[ycol,xcol]); gd=g[g.state==1].dropna(subset=[ycol,xcol])
        if len(ge)<minn or len(gd)<minn: continue
        r.append((np.polyfit(zc(ge[xcol]),ge[ycol].astype(float),1)[0],
                  np.polyfit(zc(gd[xcol]),gd[ycol].astype(float),1)[0]))
    return np.array(r)

def cell(df,ycol,xcol,label):
    s=slopes(df,ycol,xcol); be,bd=s[:,0],s[:,1]
    ret=np.abs(bd).mean()/(np.abs(be).mean()+1e-12); t,p=stats.ttest_rel(np.abs(bd),np.abs(be))
    print(f'  {label:32s} eng={be.mean():+.4f} dis={bd.mean():+.4f} RET={ret*100:4.0f}% Δ|β| t={t:+.2f} p={p:.3f} n={len(s)}')
    return dict(label=label,eng=float(be.mean()),dis=float(bd.mean()),retention=float(ret),t=float(t),p=float(p),n=len(s))

if __name__=='__main__':
    RD=roamm_word(); ZD=zuco_word()
    print(f'ROAMM words: {len(RD)} ({RD.subject.nunique()} subj, MW word-rate {RD.state.mean():.3f})')
    print(f'ZuCo words: {len(ZD)} ({ZD.subject.nunique()} subj)')
    out={}
    for meas in ['logGD','logTRT','logFFD']:
        print(f'\n===== BEHAVIORAL coupling, measure={meas} — zipf & surprisal, retention (dis/eng) =====')
        print(' ROAMM (on-task->MW):')
        out[f'beh_{meas}_zipf_ROAMM']=cell(RD,meas,'zipf','zipf->'+meas)
        out[f'beh_{meas}_surp_ROAMM']=cell(RD,meas,'surprisal','surp->'+meas)
        print(' ZuCo (NR->TSR):')
        out[f'beh_{meas}_zipf_ZuCo']=cell(ZD,meas,'zipf','zipf->'+meas)
        out[f'beh_{meas}_surp_ZuCo']=cell(ZD,meas,'surprisal','surp->'+meas)
    print('\n===== NEURAL coupling (first-fixation FRP) =====')
    print(' ROAMM:')
    out['neu_zipf_ROAMM']=cell(RD,'occ_N1','zipf','zipf->occ_N1')
    out['neu_surp_ROAMM']=cell(RD,'cp_N400','surprisal','surp->cp_N400')
    print(' ZuCo:')
    out['neu_zipf_ZuCo']=cell(ZD,'occ_N1','zipf','zipf->occFRP')
    out['neu_surp_ZuCo']=cell(ZD,'cp_N400','surprisal','surp->cpFRP')
    json.dump(out,open(f'{ZA}/transfer_2x2_v2.json','w'),indent=1)
    print('\nsaved transfer_2x2_v2.json')
