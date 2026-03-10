"""Push toward 81%: overlapping windows, CAR, surface Laplacian, random seeds."""
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
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

SUBJECTS = list(range(1, 11))
EXP_MAP = {
    'T0': {'runs': [3, 7, 11], 'event_id': {'T1': 2, 'T2': 3}},
    'T1': {'runs': [4, 8, 12], 'event_id': {'T1': 2, 'T2': 3}},
    'T2': {'runs': [5, 9, 13], 'event_id': {'T1': 2, 'T2': 3}},
    'T3': {'runs': [6, 10, 14], 'event_id': {'T1': 2, 'T2': 3}},
}

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

lda = LDA(solver='lsqr', shrinkage='auto')

def evaluate(name, subjects=SUBJECTS, tmin=0.4, tmax=4.0, reject_uv=150,
             nc=2, shrink=0.03, bands=None, clf=None, preprocess=None,
             use_all_channels=False, window_sec=None, overlap=0.5,
             n_splits=5, random_state=42):
    if bands is None:
        bands = bands5
    if clf is None:
        clf = lda
    
    scores_by_type = {}
    skip = 0
    total = 0
    for exp_name, exp in EXP_MAP.items():
        type_scores = []
        for subj in subjects:
            total += 1
            raw = load_physionet(subj, exp['runs'])
            
            # Optional preprocessing
            if preprocess == 'car':
                raw.set_eeg_reference('average', projection=False)
            elif preprocess == 'laplacian':
                try:
                    raw_sl = mne.preprocessing.compute_current_source_density(raw)
                    raw = raw_sl
                except Exception:
                    pass  # fall back to raw
            
            if window_sec is not None:
                epochs = make_epochs(raw, tmin=tmin, tmax=tmax, preload=True,
                                     pick_motor=(not use_all_channels),
                                     window_sec=window_sec, overlap=overlap)
            else:
                epochs = make_epochs(raw, tmin=tmin, tmax=tmax, preload=True,
                                     pick_motor=(not use_all_channels))
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
            
            fbcsp = make_fbcsp(nc, shrink, bands)
            pipe = Pipeline([('fbcsp', fbcsp), ('clf', clone(clf))])
            
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
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

# === Baseline ===
print("=== Baseline ===")
evaluate("base")

# === Common Average Reference ===
print("\n=== Common Average Reference ===")
evaluate("CAR", preprocess='car')
evaluate("CAR_norej", preprocess='car', reject_uv=None)

# === Surface Laplacian (CSD) ===
print("\n=== Surface Laplacian ===")
evaluate("SL", preprocess='laplacian')
evaluate("SL_norej", preprocess='laplacian', reject_uv=None)

# === All channels (64 EEG) ===
print("\n=== All channels ===")
evaluate("all_ch", use_all_channels=True)
evaluate("all_ch_CAR", use_all_channels=True, preprocess='car')

# === Overlapping windows ===
print("\n=== Overlapping windows ===")
evaluate("win1.5_o75", window_sec=1.5, overlap=0.75)
evaluate("win2.0_o50", window_sec=2.0, overlap=0.5)
evaluate("win2.0_o75", window_sec=2.0, overlap=0.75)

# === Random seed stability ===
print("\n=== Random seeds ===")
for seed in [0, 1, 2, 3, 42, 100, 999]:
    evaluate(f"seed{seed}", random_state=seed)

print("\nDone!")
