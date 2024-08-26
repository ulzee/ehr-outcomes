#%%
import pandas as pd
import numpy as np
import math
from paths import mimic_root, project_root
from tqdm import tqdm
import pickle as pk
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, f1_score
from time import time
from hyperopt import hp, Trials, fmin, tpe, STATUS_OK
from xgboost import XGBClassifier
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, default='linear')
parser.add_argument('--icd_resolution', type=int, default=4)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--subsample', type=int, default=1)
parser.add_argument('--no_agesex', action='store_true', default=False)
args = parser.parse_args()

#%%
covdf = pd.read_csv(f'{project_root}/saved/{args.dataset}/cov.csv')
covdf
#%%
with open(f'{project_root}/saved/{args.dataset}/dx.pk', 'rb') as fl:
    dx = pk.load(fl)
#%%
def format_icd(c):
    return c[:args.icd_resolution]

unique_codes = dict()

for codes in dx.values():
    for code in codes:
        if type(code) != str: continue
        c = format_icd(code)
        if c in unique_codes: continue
        unique_codes[c] = len(unique_codes)
#%%

if args.task in ['mortality', 'los72']:
    tdf = pd.read_csv(f'{project_root}/saved/{args.dataset}/targets_by_icustay.csv')
else:
    tdf = pd.read_csv(f'{project_root}/saved/{args.dataset}/targets_diagnosis_{args.task}.csv')
tdf
#%%
agesex_by_visit = dict()
for hid, age, sex in zip(covdf['hadm_id'], covdf['age'], covdf['gender']):
    agesex_by_visit[hid] = (age, [0, 1][sex == 'F'])
# %%
phases = ['train', 'val', 'test']
allids = { ph: np.genfromtxt(f'{project_root}/files/{args.dataset}/{ph}_ids.txt', dtype=int) for ph in phases }
# %%
datamats = dict()
for ph, ids in allids.items():
    nocov = 0
    print(ph, len(ids))

    select_ids = tdf['subject_id'][tdf['subject_id'].isin(ids)].unique()

    sub_tdf = tdf.set_index('subject_id').loc[select_ids]
    print(' stays', len(sub_tdf))

    obsrows = []
    outcomes = []
    for pvs, ages, iscase in tqdm(zip(sub_tdf['past_visits'], sub_tdf['past_visit_ages'], sub_tdf[args.task])):
        ivec = np.zeros(len(unique_codes) + (0 if args.no_agesex else 2))
        for hid in eval(pvs):
            if hid not in dx: continue
            for code in dx[hid]:
                if type(code) != str: continue
                ivec[unique_codes[format_icd(code)]] += 1
        if not args.no_agesex:
            last_age, sex = agesex_by_visit[hid]
            ivec[-2] = last_age / 100
            ivec[-1] = sex
        obsrows += [ivec]
        outcomes += [iscase]
    X = np.array(obsrows)
    y = np.array(outcomes)

    datamats[ph] = [X, y]


print('# Total:', sum([len(y) for _, y in datamats.values()]))
print('# Cases:', sum([sum(y) for _, y in datamats.values()]))
#%%
# ntest = len(datamats['test'][1])
# boot_ixs = [np.random.choice(ntest, size=ntest).tolist() for _ in range(10)]
if args.task in ['mortality', 'los72']:
    with open(f'{project_root}/files/{args.dataset}/boot_ixs.pk', 'rb') as fl:
        boot_ixs = pk.load(fl)
else:
    with open(f'{project_root}/files/{args.dataset}/boot_ixs_{args.task}.pk', 'rb') as fl:
        boot_ixs = pk.load(fl)
#%%
mdl_list = [args.model]
if ',' in args.model:
    mdl_list = args.model.split(',')
#%%
def get_boot_metrics(save_name, ypred):
    ls = []
    for bxs in [range(len(ypred))] + boot_ixs[:10]:
        ytrue = datamats['test'][1]
        ls += [[
            average_precision_score(ytrue[bxs], ypred[bxs]),
            roc_auc_score(ytrue[bxs], ypred[bxs]),
            f1_score(ytrue[bxs], ypred[bxs] > 0.5, average='micro'),
        ]]

    out = ''
    for tag, replicates in zip(['ap', 'roc', 'f1'], zip(*ls)):
        est, replicates = replicates[0], replicates[1:]
        ci = 1.95*np.std(replicates)
        out += f'{tag}:{est*100:.2f} ({ci*100:.2f}) '

    print(out)
    with open(f'saved/{args.dataset}_{args.task}_{save_name}.txt', 'w') as fl:
        fl.write(out)
#%%
if 'linear' in mdl_list:
    t0 = time()
    model = LogisticRegression(random_state=0, penalty=None).fit(*[m[::args.subsample] for m in datamats['train']])
    print('fit', time() - t0)

    ypred = model.predict_proba(datamats['test'][0])[:, 1]

    get_boot_metrics('linear', ypred)
if 'xgb' in mdl_list:
    t0 = time()
    print('Tuning hyperparams...')

    space = {
        'max_depth': hp.quniform("max_depth", 2, 10, 1),
        'n_estimators': hp.quniform("n_estimators", 10, 100, 1),
    }

    def objective(space):
        clf = XGBClassifier(
            n_estimators=int(space['n_estimators']),
            max_depth=int(space['max_depth']),
        )
        clf.fit(
            *[m[::args.subsample] for m in datamats['train']],
            eval_set=[datamats['val']],
            verbose=False)

        ypred = clf.predict_proba(datamats['val'][0])[:, 1]
        ytarg = datamats['val'][1]
        ap = average_precision_score(ytarg, ypred)
        roc = roc_auc_score(ytarg, ypred)
        final_loss = clf.evals_result()['validation_0']['logloss'][-1]

        print(f'{final_loss:.4f} {ap*100:.2f} {roc*100:.2f}', space)

        return { 'loss': final_loss, 'status': STATUS_OK }

    trials = Trials()

    best_hyperparams = fmin(
        fn = objective,
        space = space,
        algo = tpe.suggest,
        max_evals = 20,
        trials = trials,
        rstate=np.random.default_rng(0))

    # best_hyperparams = dict(max_depth=2, n_estimators=24)
    # best_hyperparams = dict(max_depth=4, n_estimators=64)

    print('Best:', best_hyperparams)

    # with open(f'saved/scores/{args.code}/{baseline_tag}/best_hp.pk', 'wb') as fl:
    #     pk.dump(best_hyperparams, fl)

    model = XGBClassifier(
        max_depth=int(best_hyperparams['max_depth']),
        n_estimators=int(best_hyperparams['n_estimators']),
        random_state=0
    ).fit(*[m[::args.subsample] for m in datamats['train']])
    print('fit', time() - t0)

    ypred = model.predict_proba(datamats['test'][0])[:, 1]

    get_boot_metrics('xgb', ypred)
# %%
