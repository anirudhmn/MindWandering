"""Sample construction for the attribution analysis, and a replication of the
self-report effect on the same rows."""
import numpy as np, pandas as pd
import statsmodels.api as sm
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
SP = str(ROOT / 'roamm/localisation/results') + '/'

RS = 20260819
d = pd.read_parquet(str(ROOT) + '/roamm/localisation/artifacts/evidence_trials_llm.parquet')

# redundancy check flagged in inspection
print("corr n_fix/n_refix/n_frp:")
print(d[['evidence_n_fix','evidence_n_refix','evidence_n_frp']].corr().round(4).to_string(), "\n")

EYE  = ['evidence_cov','evidence_dwell_per_word','evidence_n_fix','evidence_mean_fix_ms']
NEUR = ['evidence_n400','evidence_occ_n1','evidence_occ_p2','evidence_front_late']
NEED = EYE + NEUR + ['mw_frac_evidence','mw_frac_elsewhere','coverage','correct']

d = d[~d.mis_keyed.astype(bool)].copy()
s = d.dropna(subset=NEED).copy()
print(f"analysis sample: {len(s)} rows, {s.sub_id.nunique()} readers, {s.item.nunique()} items")
print(f"any-MW-on-evidence rate: {(s.mw_frac_evidence>0).mean():.3f} | accuracy {s.correct.mean():.3f}\n")

# within-reader z-scoring of neural amplitudes (no outcome info used)
for c in NEUR + EYE:
    s[c+'_z'] = s.groupby('sub_id')[c].transform(lambda v: (v-v.mean())/(v.std(ddof=0)+1e-9))

def lpm(df, terms, y='correct', absorb=('sub_id','item')):
    """LPM absorbing subject+item FE, SE clustered by reader."""
    X = [df[t].astype(float) for t in terms]
    X = pd.concat(X, axis=1); X.columns = terms
    for a in absorb:
        X = pd.concat([X, pd.get_dummies(df[a], prefix=a, drop_first=True).astype(float)], axis=1)
    X = sm.add_constant(X)
    m = sm.OLS(df[y].astype(float), X).fit(cov_type='cluster',
                                           cov_kwds={'groups': df['sub_id']})
    return m

# --- replicate iteration-60: MW on the answer span predicts failure -------
s['mw_ev_z']  = (s.mw_frac_evidence - s.mw_frac_evidence.mean())/s.mw_frac_evidence.std(ddof=0)
s['mw_el_z']  = (s.mw_frac_elsewhere - s.mw_frac_elsewhere.mean())/s.mw_frac_elsewhere.std(ddof=0)
s['cov_z']    = (s.coverage - s.coverage.mean())/s.coverage.std(ddof=0)

m = lpm(s, ['mw_ev_z','mw_el_z','cov_z'])
print("--- iteration-60 replication (subject+item FE, cluster-by-reader SE) ---")
for t in ['mw_ev_z','mw_el_z','cov_z']:
    print(f"  {t:9s} b={m.params[t]:+.4f}  SE={m.bse[t]:.4f}  p={m.pvalues[t]:.3g}  "
          f"CI=[{m.conf_int().loc[t,0]:+.4f},{m.conf_int().loc[t,1]:+.4f}]")

g = s.assign(bin=pd.cut(s.mw_frac_evidence,[-.001,0,.5,1.001],labels=['none','partial','full']))
print("\naccuracy by MW-on-answer-span:")
print(g.groupby('bin', observed=True).agg(n=('correct','size'), acc=('correct','mean')).round(3).to_string())

s.to_parquet(SP + 'sample.parquet')
