"""Test combinations of best fine-tuning settings."""
import mne, os, warnings
mne.set_log_level('ERROR')
os.environ['MNE_DATA'] = os.path.expanduser('~/mne_data')
warnings.filterwarnings('ignore')
import numpy as np

from src.loader import load_physionet, make_epochs
from src.csp_custom import CSP
from src.pipeline_model import BandpassCSP
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
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
CHANNELS = ['FC5','FC3','FC1','FCz','FC2','FC4','FC6',
            'C5','C3','C1','Cz','C2','C4','C6',
            'CP5','CP3','CP1','CPz','CP2','CP4','CP6']

bands5 = [(8, 10), (10, 13), (13, 18), (18, 24), (24, 30)]

def make_fbcsp(nc, shrink, bands, log_type='var'):
    steps = []
    for lo, hi in bands:
        name = f"csp_{lo}_{hi}"
        steps.append((name, BandpassCSP(
            fmin=lo, fmax=hi, sfreq=160.0,
            n_components=nc, shrink=shrink, log_type=log_type
        )))
    return FeatureUnion(steps)

def evaluate_config(name, nc, shrink, bands, clf, tmin, tmax, reject_uv, log_type='var'):
    scores_by_type = {}
    skip = 0
    total = 0
    for exp_name, exp in EXP_MAP.items():
        type_scores = []
        for subj in SUBJECTS:
            total += 1
            raw = load_physionet(subj, exp['runs'])
            epochs = make_epochs(raw, tmin=tmin, tmax=tmax, preload=True,
                                pick_motor=True)
            epochs.drop_bad()
            
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
            
            fbcsp = make_fbcsp(nc, shrink, bands, log_type)
            pipe = Pipeline([('fbcsp', fbcsp), ('clf', clone(clf))])
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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

lda = LDA(solver='lsqr', shrinkage='auto')
svm_rbf = SVC(kernel='rbf', C=100, gamma='scale')

# Combo 1: t=0.4-3.6 + rej150
print("--- Best combos ---")
evaluate_config("t04_rej150_LDA", 2, 0.03, bands5, lda, 0.4, 3.6, 150)
evaluate_config("t04_rej200_LDA", 2, 0.03, bands5, lda, 0.4, 3.6, 200)
evaluate_config("t04_rej300_LDA", 2, 0.03, bands5, lda, 0.4, 3.6, 300)
evaluate_config("t05_rej150_LDA", 2, 0.03, bands5, lda, 0.5, 3.5, 150)

# Combo 2: t=0.4-3.6 + SVM
evaluate_config("t04_SVM100", 2, 0.03, bands5, svm_rbf, 0.4, 3.6, None)
evaluate_config("t04_rej150_SVM100", 2, 0.03, bands5, svm_rbf, 0.4, 3.6, 150)

# Combo 3: logvar vs ratio
evaluate_config("t04_rej150_logvar", 2, 0.03, bands5, lda, 0.4, 3.6, 150, 'var')
evaluate_config("t04_rej150_logratio", 2, 0.03, bands5, lda, 0.4, 3.6, 150, 'ratio')

# Combo 4: shrinkage sweep with rej150
evaluate_config("t04_rej150_s005", 2, 0.05, bands5, lda, 0.4, 3.6, 150)
evaluate_config("t04_rej150_s001", 2, 0.01, bands5, lda, 0.4, 3.6, 150)
evaluate_config("t04_rej150_s000", 2, 0.0, bands5, lda, 0.4, 3.6, 150)

# Combo 5: try with 5b bands + rej150
bands5b = [(7, 10), (10, 13), (13, 18), (18, 24), (24, 32)]
evaluate_config("5bb_t04_rej150", 2, 0.03, bands5b, lda, 0.4, 3.6, 150)

# Combo 6: nc=3 + rej150
evaluate_config("nc3_t04_rej150", 3, 0.03, bands5, lda, 0.4, 3.6, 150)

# Combo 7: wider time window + rejection
evaluate_config("t05_40_rej150", 2, 0.03, bands5, lda, 0.5, 4.0, 150)
evaluate_config("t04_40_rej150", 2, 0.03, bands5, lda, 0.4, 4.0, 150)

print("Done!")
