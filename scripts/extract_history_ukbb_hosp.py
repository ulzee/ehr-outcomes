#%%
import os, sys
import math
sys.path.append('.')
sys.path.append('..')
import pandas as pd
from paths import mimic_root, ukbb_root, project_root
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pickle as pk
from tqdm import tqdm
from icdmappings import Mapper
mapper = Mapper()
#%%
code_resolution = 5
#%%
adf = pd.read_csv(f'{ukbb_root}/omop/omop_visit_occurrence.txt', usecols=['eid', 'visit_occurrence_id', 'visit_start_date', 'visit_end_date'], sep='\t')
adf.columns = ['subject_id', 'hadm_id', 'admittime', 'dischtime']
adf
#%%
pdf = pd.read_csv(f'{ukbb_root}/omop/omop_person.txt', usecols=['eid', 'gender_concept_id', 'year_of_birth', 'month_of_birth'], sep='\t')
pdf.columns = ['subject_id', 'gender_concept_id', 'byear', 'bmonth']
pdf['gender'] = [['M', 'F'][v == 8532] for v in pdf['gender_concept_id']]
pdf['birthdate'] = [datetime.strptime(f'{y}-{m}', '%Y-%m') for y, m in zip(pdf['byear'], pdf['bmonth'])]
pdf = pdf[['subject_id', 'gender', 'birthdate']]
pdf
#%%
# sids = pdf['subject_id'].values.tolist()
# np.random.shuffle(sids)
# chunk = len(sids)//10
# train_ids = sids[:7*chunk]
# val_ids = sids[7*chunk:7*chunk+int(1.5*chunk)]
# test_ids = sids[7*chunk+int(1.5*chunk):]
# np.savetxt(f'{project_root}/files/ukbb/train_ids', train_ids, fmt='%d')
# np.savetxt(f'{project_root}/files/ukbb/val_ids', val_ids, fmt='%d')
# np.savetxt(f'{project_root}/files/ukbb/test_ids', test_ids, fmt='%d')
#%%
icddf = pd.read_csv(f'{ukbb_root}/omop/omop_condition_occurrence.txt', usecols=['eid', 'visit_occurrence_id', 'condition_type_concept_id', 'condition_source_concept_id', 'condition_source_value'], sep='\t')
icddf
# %%
icd10df = icddf[(icddf['condition_type_concept_id'] == 32817) & ~icddf['visit_occurrence_id'].isna()]
icd10df
#%%
dxlookup = dict()
for hid, code in zip(icd10df['visit_occurrence_id'].astype(int), icd10df['condition_source_value']):
    if hid not in dxlookup: dxlookup[hid] = dict()
    code = code[:code_resolution]
    dxlookup[hid][code] = True
#%%
with open(f'{project_root}/saved/ukbb/dx.pk', 'wb') as fl:
    pk.dump(dxlookup, fl)
# %%
ddf = adf.set_index('subject_id') \
    .join(pdf.set_index('subject_id')[['gender', 'birthdate']], on='subject_id', how='left') \
        .reset_index()
ddf = ddf.sort_values('admittime')
ddf
# %%
# organize visits and their times by patient
visits_by_patient = dict()
for sid, admittime, dischtime,  hadm_id in tqdm(zip(ddf['subject_id'], ddf['admittime'], ddf['dischtime'], ddf['hadm_id'])):
    if sid not in visits_by_patient:
        visits_by_patient[sid] = [list(), list(), list()]
    visits_by_patient[sid][0] += [hadm_id]
    visits_by_patient[sid][1] += [datetime.strptime(admittime, '%d/%m/%Y')]
    visits_by_patient[sid][2] += [datetime.strptime(dischtime, '%d/%m/%Y')]
# %%
min_visits = 2
pad_last = 1 # we don't know what happens after the last record of each patient

fwd_ranges = [180] # days

class stats:
    s_too_few = 0

fwd_visits = dict()
for sid, (hids, intime, outtime) in tqdm(list(visits_by_patient.items())):
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
bench_range = 180
match_control_min_ratio = 10

hadf = adf.set_index('hadm_id')

def match_any(targ, codes):
    return any([targ == c[:len(targ)] for c in codes if type(c) == str])

bdates = { sid: bdate for sid, bdate in zip(pdf['subject_id'], pdf['birthdate']) }
# example parse of a phenotype
for target in ['F329', 'I25', 'G2']:
    unique_s = hadf.loc[[h for h in dxlookup if match_any(target, dxlookup[h])]]['subject_id'].nunique()
    print(target, unique_s)

    samples = []

    iadf = adf.set_index('subject_id')

    idate = dict()
    iadf = dict()
    for sid, hid, t in zip(adf['subject_id'], adf['hadm_id'], adf['admittime']):
        if sid not in iadf: iadf[sid] = []
        iadf[sid] += [hid]
        idate[hid] = datetime.strptime(t, '%d/%m/%Y')

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
                            idate[hfwd] - idate[hid]
                        ).total_seconds()/60/60/24

                    if dt <= bench_range:
                        iscase = True
                        case_dts += [dt]

                        break

            # TODO: match against raw hist
            visit_past = fwd_info['history'] # current inclusive
            # aage, ayear = anchor_times[sid]
            t_at_visit = (idate[hid] - bdates[sid]).total_seconds() / 60/60/24/365
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
        f'{project_root}/saved/ukbb/targets_diagnosis_{target}.csv', index=False)

    # match agaisnt test set and save bootstraps
    test_ids = { i: True for i in np.genfromtxt('../files/ukbb/test_ids.txt', dtype=int) }
    test_ids_in_bench = [i for i in sid_ls if i in test_ids ]
    ntest = len(test_ids_in_bench)
    boot_ixs = [np.random.choice(ntest, size=ntest, replace=True).tolist() for _ in range(10)]
    with open(f'{project_root}/files/ukbb/boot_ixs_{target}.pk', 'wb') as fl:
        pk.dump(boot_ixs, fl)
# %%
