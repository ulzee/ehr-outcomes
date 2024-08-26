#%%
import os, sys
import math
sys.path.append('.')
sys.path.append('..')
import pandas as pd
from paths import mimic_root, project_root
from datetime import datetime, timedelta
import numpy as np
import pickle as pk
#%%
adf = pd.read_csv(f'{mimic_root}/hosp/admissions.csv.gz')
adf = adf.sort_values('admittime')
adf['deathtime_d'] = adf['deathtime'].str.split(n=1, expand=True)[0].values
adf
#%%
pdf = pd.read_csv(f'{mimic_root}/hosp/patients.csv')
pdf
# %%
icudf = pd.read_csv(f'{mimic_root}/icu/icustays.csv.gz')
icudf['intime_d'] = icudf['intime'].str.split(n=1, expand=True)[0].values
icudf['outtime_d'] = icudf['outtime'].str.split(n=1, expand=True)[0].values
icudf
# %%
ddf = icudf.set_index('subject_id') \
    .join(pdf.set_index('subject_id')[['dod', 'gender', 'anchor_age', 'anchor_year']], on='subject_id', how='left') \
        .reset_index().set_index('hadm_id').join(adf.set_index('hadm_id')['deathtime_d'], on='hadm_id', how='left')
ddf = ddf.sort_values('intime')
ddf
# %%
# determine targets for downstream benchmark
death_future_inclusion_range = 1  # year
los_range = 72 #hours
ncases = 0
ncontrols = 0
mortality_cases = []
los_cases = []
# parse both death times available just in case
for intimef, outtimef, intime, outtime, dtime, dtime2 in zip(*[ddf[c] for c in ['intime', 'outtime', 'intime_d', 'outtime_d', 'dod', 'deathtime_d']]):
    dy = int(outtime.split('-')[0])
    outtime = outtime.replace(str(dy), str(dy+death_future_inclusion_range))
    iscase = False

    dtime = dtime2 if type(dtime2) == str else dtime

    if type(dtime) == str:
        if dtime >= intime and dtime <= outtime:
            iscase = True
    ncontrols += not iscase
    ncases += iscase

    outf = datetime.strptime(outtimef, '%Y-%m-%d %H:%M:%S')
    inf = datetime.strptime(intimef, '%Y-%m-%d %H:%M:%S')
    stay_seconds = (outf - inf).total_seconds()
    stay_case = stay_seconds > 60*60*los_range

    mortality_cases += [iscase]
    los_cases += [stay_case]
ddf['mortality'] = mortality_cases
ddf[f'los{los_range}'] = los_cases
ddf['mortality'].sum(), ddf[f'los{los_range}'].sum()
#%%
hgrouped = adf[['subject_id', 'hadm_id', 'admittime']].groupby('subject_id').agg(list).reset_index()
hgrouped = { sid: dict(hids=hids, ds=ds) for sid, hids, ds in zip(*[hgrouped[c] for c in hgrouped.columns])}
#%%
# attach previous visits to fetch them easily later
# NOTE: age can be inferred from anchor times, but not absolute time of visits

def get_age_since_anchor(when, ancyear, ancage):
    when_t = datetime.strptime(when, '%Y-%m-%d %H:%M:%S')
    dy = (when_t - datetime.strptime(str(ancyear), '%Y')).total_seconds()/60/60/24/365
    since_born = dy+ancage
    return float('%.2f' % since_born)
#%%

past_visits = []
past_ages = []
hist_len = []
for sid, intime, ancage, ancyear in zip(ddf['subject_id'], ddf['intime_d'], ddf['anchor_age'], ddf['anchor_year']):

    hhist = hgrouped[sid]
    ls_visits = []
    ls_dates = []
    for hid, d in zip(hhist['hids'], hhist['ds']):
        if d >= intime:
            break
        ls_visits += [hid]
        ls_dates += [get_age_since_anchor(d, ancyear, ancage)]

    past_visits += [ls_visits]
    past_ages += [ls_dates]
    hist_len += [len(ls_visits)]

ddf['past_visits'] = past_visits
ddf['past_visit_ages'] = past_ages
ddf['num_past_visits'] = hist_len
#%%
race_ddf = ddf.reset_index().set_index('subject_id').join(adf.set_index('subject_id')[['race']], on='subject_id', how='left').reset_index()
race_ddf
#%%
min_hist_requirement = 3
has_hist = race_ddf[race_ddf['num_past_visits'] >= min_hist_requirement]
has_hist['mortality'].sum(), len(has_hist), has_hist[f'los{los_range}'].sum()
# %%
has_hist.to_csv(f'{project_root}/saved/targets_by_icustay.csv')
#%%
dxdf = pd.read_csv(f'{project_root}/saved/hosp-diagnoses_icd10.csv')
dxdf
#%%
covdf = adf.set_index('subject_id').join(pdf.set_index('subject_id'), on='subject_id', how='left').reset_index()

age_at_visit = []
for sid, admittime, ancage, ancyear in zip(covdf['subject_id'], covdf['admittime'], covdf['anchor_age'], covdf['anchor_year']):
    # intime = datetime.strptime(admittime, '%Y-%m-%d %H:%M:%S')

    age_at_visit += [get_age_since_anchor(admittime, ancyear, ancage)]

covdf['age'] = age_at_visit
#%%
covdf_main = covdf[['subject_id', 'hadm_id', 'gender', 'age', 'race', 'marital_status', 'discharge_location']]
covdf_main.to_csv('../saved/cov.csv', index=False)
# %%
dxlookup = dict()
for hid, code in zip(dxdf['hadm_id'], dxdf['icd10']):
    if hid not in dxlookup: dxlookup[hid] = {}
    dxlookup[hid][code] = True
with open(f'{project_root}/saved/dx.pk', 'wb') as fl:
    pk.dump(dxlookup, fl)
len(dxlookup)
# %%
# sids = pdf['subject_id'].values.tolist()
# # %%
# np.random.shuffle(sids)
# # %%
# c = len(sids)//10
# np.savetxt(f'{project_root}/files/train_ids.txt', sids[:c*7], fmt='%d')
# np.savetxt(f'{project_root}/files/val_ids.txt', sids[c*7:int(c*8.5)], fmt='%d')
# np.savetxt(f'{project_root}/files/test_ids.txt', sids[int(c*8.5):], fmt='%d')
# # %%
