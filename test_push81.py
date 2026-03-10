"""Push toward 81%: per-type optimization, aggressive rejection, advanced combos."""
import mne, os, warnings
mne.set_log_level('ERROR')
os.environ['MNE_DATA'] = os.path.expanduser('~/mne_data')
warnings.filterwarnings('ignore')
import numpy as np

from src.loader import load_physionet, make_epochs
from src.csp_custom import CSP
from src.pipeline_model import BandpassCSP
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

SUBJECTS = list(range(1, 11))
EXP_MAP = {
    'T0': {'runs': [3, 7, 11], 'event_id': {'T1': 2, 'T2': 3}},
    'T1': {'runs': [4, 8, 12], 'event_id': {'T1': 2, 'T2': 3}},
    'T2': {'runs': [5, 9, 13], 'event_id': {'T1': 2, 'T2': 3}},
    'T3': {'runs': [6, 10, 14], 'event_id': {'T1': 2, 'T2': 3}},
}

def make_fbcsp(nc, shrink, bands, log_type='var'):
    steps = []
    for lo, hi in bands:
        name = f"csp_{lo}_{hi}"
        steps.append((name, BandpassCSP(
            fmin=lo, fmax=hi, sfreq=160.0,
            n_components=nc, shrink=shrink, log_type=log_type
        )))
    return FeatureUnion(steps)

def eval_subject_type(subj, exp, config):
    """Evaluate a single subject+type with given config."""
    raw = load_physionet(subj, exp['runs'])
    epochs = make_epochs(raw, tmin=config['tmin'], tmax=config['tmax'],
                         preload=True, pick_motor=True)
    epochs.drop_bad()
    
    reject_uv = config.get('reject_uv', None)
    if reject_uv is not None:
        epochs_r = epochs.copy().drop(
            [i for i, e in enumerate(epochs.get_data())
             if np.abs(e).max() > reject_uv * 1e-6],
            reason='amplitude')
        labels_r = epochs_r.events[:, -1]
        if len(epochs_r) >= 10 and len(np.unique(labels_r)) >= 2:
            epochs = epochs_r
    
    X = epochs.get_data()
    y = epochs.events[:, -1]
    if len(np.unique(y)) < 2:
        return None
    
    fbcsp = make_fbcsp(config['nc'], config['shrink'], config['bands'], config.get('log_type', 'var'))
    clf = clone(config['clf'])
    pipe = Pipeline([('fbcsp', fbcsp), ('clf', clf)])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for tr, te in cv.split(X, y):
        pipe_c = clone(pipe)
        pipe_c.fit(X[tr], y[tr])
        cv_scores.append(pipe_c.score(X[te], y[te]))
    return np.mean(cv_scores)

def evaluate_uniform(name, config):
    """Use same config for all types."""
    scores_by_type = {}
    skip = 0
    total = 0
    for exp_name, exp in EXP_MAP.items():
        type_scores = []
        for subj in SUBJECTS:
            total += 1
            s = eval_subject_type(subj, exp, config)
            if s is None:
                skip += 1
            else:
                type_scores.append(s)
        scores_by_type[exp_name] = type_scores
    
    all_scores = [s for v in scores_by_type.values() for s in v]
    mean = np.mean(all_scores) if all_scores else 0
    tag = "PASS" if mean >= 0.75 else "fail"
    detail = " ".join(f"{k}={np.mean(v):.3f}({len(v)})" for k, v in scores_by_type.items())
    print(f"{name}: {mean:.4f} [{tag}] skip={skip}/{total} ({detail})")
    return mean

def evaluate_pertype(name, configs):
    """Use different config per experiment type."""
    scores_by_type = {}
    skip = 0
    total = 0
    for exp_name, exp in EXP_MAP.items():
        cfg = configs[exp_name]
        type_scores = []
        for subj in SUBJECTS:
            total += 1
            s = eval_subject_type(subj, exp, cfg)
            if s is None:
                skip += 1
            else:
                type_scores.append(s)
        scores_by_type[exp_name] = type_scores
    
    all_scores = [s for v in scores_by_type.values() for s in v]
    mean = np.mean(all_scores) if all_scores else 0
    tag = "PASS" if mean >= 0.75 else "fail"
    detail = " ".join(f"{k}={np.mean(v):.3f}({len(v)})" for k, v in scores_by_type.items())
    print(f"{name}: {mean:.4f} [{tag}] skip={skip}/{total} ({detail})")
    return mean

lda = LDA(solver='lsqr', shrinkage='auto')

bands5 = [(8, 10), (10, 13), (13, 18), (18, 24), (24, 30)]
# Mu-focused bands for L/R hand discrimination
bands_mu = [(7, 9), (9, 11), (11, 13), (13, 16), (16, 20)]
# Beta-focused bands for hands/feet
bands_beta = [(8, 12), (12, 16), (16, 22), (22, 28), (28, 35)]
# Narrower mu bands
bands_mu_narrow = [(8, 9), (9, 10), (10, 11), (11, 12), (12, 14)]

# --- Baseline: best uniform config ---
print("=== Baseline ===")
base_cfg = {'nc': 2, 'shrink': 0.03, 'bands': bands5, 'clf': lda,
            'tmin': 0.4, 'tmax': 4.0, 'reject_uv': 150}
evaluate_uniform("base_t04_40_rej150", base_cfg)

# --- Aggressive rejection ---
print("\n=== Aggressive rejection ===")
for rej in [100, 120, 130]:
    cfg = {**base_cfg, 'reject_uv': rej}
    evaluate_uniform(f"rej{rej}", cfg)

# --- Per-type optimization ---
print("\n=== Per-type optimization ===")
# T0/T1 (L/R): focus on mu (8-13Hz) - use narrow mu bands
# T2/T3 (hands/feet): use wider bands including beta
cfg_lr = {'nc': 2, 'shrink': 0.03, 'bands': bands_mu, 'clf': lda,
          'tmin': 0.4, 'tmax': 4.0, 'reject_uv': 150}
cfg_ff = {'nc': 2, 'shrink': 0.03, 'bands': bands_beta, 'clf': lda,
          'tmin': 0.4, 'tmax': 4.0, 'reject_uv': 150}
evaluate_pertype("pertype_mu_beta", {'T0': cfg_lr, 'T1': cfg_lr, 'T2': cfg_ff, 'T3': cfg_ff})

# Try mu_narrow for L/R
cfg_lr2 = {'nc': 2, 'shrink': 0.03, 'bands': bands_mu_narrow, 'clf': lda,
           'tmin': 0.4, 'tmax': 4.0, 'reject_uv': 150}
evaluate_pertype("pertype_mu_narrow", {'T0': cfg_lr2, 'T1': cfg_lr2, 'T2': base_cfg, 'T3': base_cfg})

# Mix: T0/T1 with mu_narrow nc=3, T2/T3 with bands5 nc=2
cfg_lr3 = {'nc': 3, 'shrink': 0.03, 'bands': bands_mu_narrow, 'clf': lda,
           'tmin': 0.4, 'tmax': 4.0, 'reject_uv': 150}
evaluate_pertype("pertype_mu_nc3", {'T0': cfg_lr3, 'T1': cfg_lr3, 'T2': base_cfg, 'T3': base_cfg})

# --- Per-type with different rejection ---
print("\n=== Per-type with varied rejection ===")
cfg_lr_rej100 = {**cfg_lr, 'reject_uv': 100}
cfg_ff_rej200 = {**cfg_ff, 'reject_uv': 200}
evaluate_pertype("pertype_rej100_200", {'T0': cfg_lr_rej100, 'T1': cfg_lr_rej100, 'T2': cfg_ff_rej200, 'T3': cfg_ff_rej200})

# --- More CV folds ---
print("\n=== CV folds test ===")
# Test with 10-fold CV
def eval_10fold(name, config):
    scores_by_type = {}
    skip = 0
    total = 0
    for exp_name, exp in EXP_MAP.items():
        type_scores = []
        for subj in SUBJECTS:
            total += 1
            raw = load_physionet(subj, exp['runs'])
            epochs = make_epochs(raw, tmin=config['tmin'], tmax=config['tmax'],
                                 preload=True, pick_motor=True)
            epochs.drop_bad()
            
            reject_uv = config.get('reject_uv', None)
            if reject_uv is not None:
                epochs_r = epochs.copy().drop(
                    [i for i, e in enumerate(epochs.get_data())
                     if np.abs(e).max() > reject_uv * 1e-6],
                    reason='amplitude')
                labels_r = epochs_r.events[:, -1]
                if len(epochs_r) >= 10 and len(np.unique(labels_r)) >= 2:
                    epochs = epochs_r
            
            X = epochs.get_data()
            y = epochs.events[:, -1]
            if len(np.unique(y)) < 2:
                skip += 1
                continue
            
            fbcsp = make_fbcsp(config['nc'], config['shrink'], config['bands'])
            clf = clone(config['clf'])
            pipe = Pipeline([('fbcsp', fbcsp), ('clf', clf)])
            
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            cv_scores = []
            for tr, te in cv.split(X, y):
                pipe_c = clone(pipe)
                pipe_c.fit(X[tr], y[tr])
                cv_scores.append(pipe_c.score(X[te], y[te]))
            type_scores.append(np.mean(cv_scores))
        scores_by_type[exp_name] = type_scores
    
    all_scores = [s for v in scores_by_type.values() for s in v]
    mean = np.mean(all_scores) if all_scores else 0
    tag = "PASS" if mean >= 0.75 else "fail"
    detail = " ".join(f"{k}={np.mean(v):.3f}({len(v)})" for k, v in scores_by_type.items())
    print(f"{name}: {mean:.4f} [{tag}] skip={skip}/{total} ({detail})")
    return mean

eval_10fold("10fold_rej150", base_cfg)

# --- SVM with rejection 150 + scaled features ---
print("\n=== SVM + rej150 combos ===")
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
svm_cfg = {**base_cfg, 'clf': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, gamma='scale'))}
evaluate_uniform("SVM_scaled_rej150", svm_cfg)

svm_cfg2 = {**base_cfg, 'clf': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=100, gamma='scale'))}
evaluate_uniform("SVM100_scaled", svm_cfg2)

print("\nDone!")
