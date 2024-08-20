# EHR Outcomes

A few scripts to generate benchmark datasets of outcomes from EHR datasets.

## Getting started

### Requirements

Python 3.10 is required to parse the dataset due to the `icdmappings` package.
Previous versions of pythons will work just for the purpose of benchmarking if you can obtain the parsed dataset files separately.

Requirements for parsing only:
```bash
numpy
pandas
icdmappings
pickle
tqdm
```

All requirements to parse and run baseline methods:
```bash
numpy
pandas
icdmappings
pickle
tqdm
scikit-learn
hyperopt
xgboost
```

### Project setup

The scripts need to be told the project root and the dataset root directories in the `paths.py` folder such as:

```python
mimic_root = '/home/ulzee/gpt/data/mimic4/mimiciv/2.2'
project_root = '/home/ulzee/gpt/ehr-outcomes'
```

## 0. Train test splits

The patient ids of the train, val, and test splits are provided in `files/`. The scripts below will refer to them automatically.

When exploring any pretraining step, user must make sure that the pretrained model only observes patients in `train_ids.txt`. The id files can be copied over directly if running the icdbert project.

## 1. Parsing MIMIC

### A. Convert ICD9 codes to ICD10 codes

The following script will save a copy of the diagnosis records files where all codes are attempted to be converted to ICD10 in `saved/`.
```bash
python scripts/convert_icd10.py
```


### B. Extract visits for benchmarking ICU outcomes

Currently "mortality" and "length of stay" tasks are supported.
Both are treated as binary classification tasks (LOS ~ does the stay exceed 72 hours).

Some options that may be configured (leave as-is for default settings):
* `death_future_inclusion_range` (default: 1 year) Time range beyond admission time to detect mortality
* `los_range` (default: 72 hours) Time range to treat a stay length as short or long
* `min_hist_requirement` (default: 3 visits) Min number of visits a patient must have to be considered in this benchmark

```bash
python scripts/extract_history_icu.py
```

### C. Extract visits for benchmarking ICU outcomes

Currently the script will automatically save benchmark cohorts for ICD codes:
* F329: Major depressive disorder
single episode, unspecified
* I25: Chronic ischaemic heart disease
* G20: Parkinson’s disease

Some options that may be configured (leave as-is for default settings):
* `min_visits` (default: 2 visits) Mininum number of visits required for a patient to be considered
* `pad_last` (default: 1 visits) How many last known vists to ignore
* `bench_range` (default: 180 days) How far ahead to check to see if a patient converted for a given disease
* `match_control_min_ratio` (default: 10) The amount of controls to keep for a meaningful benchmark as a ratio of the number of cases found

```bash
python scripts/extract_history_hosp.py
```

## 2. (optional) Run baseline methods

Logistic regression and XGB can be run in one go with a command such as:

```bash
python baselines.py --model linear,xgb --task los72
```

The tasks currently supported are: `mortality`, `los72`, `F329`, `I25`, `G20`.

After fitting each baseline method, the script will report the estimated PR, ROC, and F1 scores with 95%-CI (obtained with bootstrapping).

