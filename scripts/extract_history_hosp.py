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
from tqdm import tqdm
#%%
with open(f'{project_root}/saved/dx.pk', 'rb') as fl:
    dxlookup= pk.load(fl)
len(dxlookup)
#%%
adf = pd.read_csv(f'{mimic_root}/hosp/admissions.csv.gz')
adf = adf.sort_values('admittime')
adf['deathtime_d'] = adf['deathtime'].str.split(n=1, expand=True)[0].values
adf
#%%
pdf = pd.read_csv(f'{mimic_root}/hosp/patients.csv')
pdf
#%%
anchor_times = dict()
for sid, aage, ayear in zip(pdf['subject_id'], pdf['anchor_age'], pdf['anchor_year']):
    anchor_times[sid] = [aage, datetime.strptime(str(ayear), '%Y')]
#%%
icddf = pd.read_csv(f'{project_root}/saved/hosp-diagnoses_icd10.csv')
icddf
#%%

# %%
ddf = adf.set_index('subject_id') \
    .join(pdf.set_index('subject_id')[['dod', 'gender', 'anchor_age', 'anchor_year']], on='subject_id', how='left') \
        .reset_index()
ddf = ddf.sort_values('admittime')
ddf
# %%
# organize visits and their times by patient

empty_visit = 0
visits_by_patient = dict()
for sid, admittime, dischtime,  hadm_id in zip(ddf['subject_id'], ddf['admittime'], ddf['dischtime'], ddf['hadm_id']):
    empty_visit += hadm_id not in dxlookup
    if sid not in visits_by_patient:
        visits_by_patient[sid] = [list(), list(), list()]
    visits_by_patient[sid][0] += [hadm_id]
    visits_by_patient[sid][1] += [datetime.strptime(admittime, '%Y-%m-%d %H:%M:%S')]
    visits_by_patient[sid][2] += [datetime.strptime(dischtime, '%Y-%m-%d %H:%M:%S')]
empty_visit
# %%
min_visits = 2
# min_visits = 0
pad_last = 1 # we don't know what happens after the last record of each patient

fwd_ranges = [30, 60, 120, 180] # days

class stats:
    s_too_few = 0

fwd_visits = dict()
for sid, (hids, intime, outtime) in visits_by_patient.items():
    fwd_visits[sid] = dict()

    if len(intime) < min_visits + pad_last:
        stats.s_too_few += 1
        continue

    hids = hids[:-pad_last]
    intime = intime[:-pad_last]
    outtime = outtime[:-pad_last]

    for hi, (h, st, ed) in enumerate(zip(hids, intime, outtime)):
        fwd_visits[sid][h] = { f'days{r}': dict() for r in fwd_ranges }
        fwd_visits[sid][h]['history'] = hids[:hi+1]

        for _h, _st, _ed in zip(hids, intime, outtime):
            if _st < ed: continue
            dt_fwd = (_st - ed).total_seconds() / (60*60*24)
            for r in fwd_ranges:
                if dt_fwd <= r:
                    fwd_visits[sid][h][f'days{r}'][_h] = dt_fwd

stats.s_too_few
# %%
# target = 'F329'
bench_range = 180
match_control_min_ratio = 10

hadf = adf.set_index('hadm_id')

def match_any(targ, codes):
    return any([targ == c[:len(targ)] for c in codes if type(c) == str])

# example parse of a phenotype
# for target in ['F329', 'I25']:
for target in ['G20']:
    unique_s = hadf.loc[[h for h in dxlookup if match_any(target, dxlookup[h])]]['subject_id'].nunique()
    print(target, unique_s)

    samples = []

    iadf = adf.set_index('subject_id')

    idate = dict()
    iadf = dict()
    for sid, hid, t in zip(adf['subject_id'], adf['hadm_id'], adf['admittime']):
        if sid not in iadf: iadf[sid] = []
        iadf[sid] += [hid]
        idate[hid] = t

    case_dts = []
    for sid, visits in tqdm(fwd_visits.items()):

        converted = False

        all_visits = iadf[sid]

        for hid, fwd_info in visits.items():

            # 1. if current visit has the targe phenotype
            #  stop tracing this patient
            if hid not in dxlookup: continue
            if match_any(target, dxlookup[hid]):
                converted = True
                break

            # any_fwds = fwd_info[f'days{bench_range}']
            any_fwds = all_visits[all_visits.index(hid)+1:-1]

            # 2. check if any nearby future visits have the target
            iscase = False
            for hfwd in any_fwds:
                if hfwd not in dxlookup: continue
                if match_any(target, dxlookup[hfwd]):

                    dt = (
                            datetime.strptime(idate[hfwd], '%Y-%m-%d %H:%M:%S') - \
                            datetime.strptime(idate[hid], '%Y-%m-%d %H:%M:%S')
                        ).total_seconds()/60/60/24

                    if dt <= bench_range:
                        iscase = True
                        case_dts += [dt]

                        break

            # TODO: match against raw hist
            visit_past = fwd_info['history'] # current inclusive
            aage, ayear = anchor_times[sid]
            t_at_visit = (datetime.strptime(idate[hid], '%Y-%m-%d %H:%M:%S') - ayear).total_seconds() / 60/60/24/365+aage
            samples += [(sid, hid, t_at_visit, visit_past, iscase)]

    print('# of cases', len([s for s in samples if s[-1]]))
    print('# samples (visit)', len(samples))

    if match_control_min_ratio is not None:
        case_ids = [s[1] for s in samples if s[-1]]
        ncases = len(case_ids)
        control_ids = [s[1] for s in samples if not s[-1]]
        ncontrols = len(control_ids)

        if ncases < ncontrols / match_control_min_ratio:
            controls_targ_amount = ncases * match_control_min_ratio
            reduce_controls = ncontrols / controls_targ_amount

            print('Reducing # controls by factor of', reduce_controls)

            np.random.shuffle(control_ids)

            keep_control_ids = { h: True for h in control_ids[:10*ncases] }
            print('Controls kept', len(keep_control_ids))

            keep_ids = { h: True for h in list(keep_control_ids.keys()) + case_ids }

            samples = [s for s in samples if s[1] in keep_ids]
            print('Reduced to', len(samples))


    sid_ls, hid_ls, past_visit_ages, visit_past_ls, iscase_ls = zip(*samples)
    dset = dict(
        subject_id=sid_ls,
        past_visits=visit_past_ls,
        past_visit_ages=[float('%.2f' % a) for a in past_visit_ages],
    )
    dset[target] = iscase_ls
    pd.DataFrame(dset).to_csv(
        f'{project_root}/saved/targets_diagnosis_{target}.csv', index=False)

    # match agaisnt test set and save bootstraps
    test_ids = { i: True for i in np.genfromtxt('../files/test_ids.txt', dtype=int) }
    test_ids_in_bench = [i for i in sid_ls if i in test_ids ]
    ntest = len(test_ids_in_bench)
    boot_ixs = [np.random.choice(ntest, size=ntest, replace=True).tolist() for _ in range(10)]
    with open(f'{project_root}/files/boot_ixs_{target}.pk', 'wb') as fl:
        pk.dump(boot_ixs, fl)
#%%
# import matplotlib.pyplot as plt
# plt.figure()
# plt.hist(case_dts)
# plt.show()
# len(case_dts)
# #%%

