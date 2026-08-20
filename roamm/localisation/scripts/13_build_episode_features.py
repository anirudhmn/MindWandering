"""Build: MW episode structure + repair features per (reader, item)."""
import numpy as np, pandas as pd, json
import pathlib
ROOT=pathlib.Path(__file__).resolve().parents[3]
SP=str(ROOT/'roamm/localisation/results')+'/'

fx = pd.read_parquet(str(ROOT)+'/roamm/artifacts/coupling/all_fixations.parquet')
it = pd.read_parquet(str(ROOT)+'/roamm/localisation/artifacts/item_evidence_llm.parquet')
smap = json.load(open(str(ROOT)+'/roamm/artifacts/comprehension/subject_map.json'))['mapping']
inv = {v:k for k,v in smap.items()}

print("story keys  fx:", sorted(fx.story.unique()))
print("story keys  it:", sorted(it.story_phys.unique()))
fx = fx.dropna(subset=['word_key','page']).copy()
fx['page'] = fx.page.astype(int)
fx['sub_id'] = fx.subject.map(inv)

# episodes per page: dataset permits at most one MW report per page
g = fx.sort_values(['subject','story','page','fix_order_all'])
ep = g.groupby(['subject','story','page']).is_mw.agg(
        n_mw='sum', n='size',
        n_runs=lambda v: int(((v.astype(int).diff()==1).sum()) + (1 if v.iloc[0] else 0)))
print("\npages with MW:", (ep.n_mw>0).sum(), "| >1 contiguous run:", (ep.n_runs>1).sum())

rows=[]
for _, r in it.iterrows():
    ev = set(r.evidence_word_keys); ct = set(r.control_word_keys)
    pg = fx[(fx.story==r.story_phys) & (fx.page==int(r.page))]
    if not len(pg): continue
    for sub, d in pg.groupby('subject'):
        d = d.sort_values('fix_order_all')
        mw = d.is_mw.values
        in_ev = d.word_key.isin(ev).values
        in_ct = d.word_key.isin(ct).values
        dur = d.fix_dur.values; t = d.tStart.values
        rec = dict(sub_id=inv[sub], item=r['item'],
                   ev_n_fix=int(in_ev.sum()), ct_n_fix=int(in_ct.sum()),
                   ev_dwell=float(dur[in_ev].sum()), ct_dwell=float(dur[in_ct].sum()),
                   mw_n_ev=int((mw&in_ev).sum()), mw_n_ct=int((mw&in_ct).sum()),
                   mw_n_page=int(mw.sum()), page_n_fix=int(len(d)))
        if mw.any():
            i = np.where(mw)[0]
            first, last = i[0], i[-1]
            rec['ep_n_fix']  = int(mw.sum())
            rec['ep_dur_ms'] = float(t[last]+dur[last]/1000.0 - t[first])*1000.0
            # repair: fixations AFTER the episode ends (reader resumes same page)
            post = np.zeros(len(d), bool); post[last+1:] = True
            rec['post_n_fix']    = int(post.sum())
            rec['repair_ev_fix'] = int((post&in_ev).sum())
            rec['repair_ct_fix'] = int((post&in_ct).sum())
            rec['repair_ev_dwell']= float(dur[post&in_ev].sum())
            rec['repair_ct_dwell']= float(dur[post&in_ct].sum())
            # read BEFORE the episode began
            pre = np.zeros(len(d), bool); pre[:first] = True
            rec['pre_ev_fix'] = int((pre&in_ev).sum())
            # depth proxy: distance of evidence-MW fixations from the awareness moment
            sel = np.where(mw&in_ev)[0]
            if len(sel):
                rec['pos_in_ep']  = float(np.mean((sel-first)/max(last-first,1)))
                rec['t_to_aware'] = float(np.mean(t[last]-t[sel]))*1000.0
        rows.append(rec)

E = pd.DataFrame(rows)
for c in ['ep_n_fix','ep_dur_ms','post_n_fix','repair_ev_fix','repair_ct_fix',
          'repair_ev_dwell','repair_ct_dwell','pre_ev_fix']:
    E[c] = E[c].fillna(0)
print(f"\nbuilt {len(E)} (reader,item) rows | any page MW: {(E.mw_n_page>0).mean():.3f}"
      f" | MW on evidence: {(E.mw_n_ev>0).mean():.3f}")
print(E[['ev_n_fix','mw_n_ev','ep_n_fix','post_n_fix','repair_ev_fix','pre_ev_fix']].describe().T.to_string())
E.to_parquet(SP+'episodes.parquet')

s = pd.read_parquet(SP+'sample2.parquet')
m = s.merge(E, on=['sub_id','item'], how='inner', suffixes=('','_ep'))
print(f"\nmerged with analysis sample: {len(m)} rows")
print("check mw_n_ev>0 vs mw_frac_evidence>0 agreement:",
      ((m.mw_n_ev>0)==(m.mw_frac_evidence>0)).mean().round(4))
m.to_parquet(SP+'merged.parquet')
