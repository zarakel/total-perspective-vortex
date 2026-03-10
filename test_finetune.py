#!/usr/bin/env python3
"""Fine-tune the 5-band nc=2 config + explore nc=3 and variations."""
import mne, os, warnings, sys
mne.set_log_level('ERROR')
os.environ['MNE_DATA'] = os.path.expanduser('~/mne_data')
warnings.filterwarnings('ignore')
import numpy as np
from src.loader import load_physionet, make_epochs
from src.preprocessing import bandpass_filter
from src.csp_custom import CSP
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin

EXPERIMENT_TYPES = [
    {'runs': [3, 7, 11]}, {'runs': [4, 8, 12]},
    {'runs': [5, 9, 13]}, {'runs': [6, 10, 14]},
]
subjects = list(range(1, 11))

class BandpassCSP(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=160, fmin=8, fmax=13, n_components=4, shrink=0.05):
        self.sfreq = sfreq; self.fmin = fmin; self.fmax = fmax
        self.n_components = n_components; self.shrink = shrink
        self.csp = CSP(n_components=n_components, shrink=shrink, log_type='var')
    def _f(self, X):
        from mne.filter import filter_data
        return filter_data(X.astype(np.float64), self.sfreq, self.fmin, self.fmax, verbose=False)
    def fit(self, X, y): self.csp.fit(self._f(X), y); return self
    def transform(self, X): return self.csp.transform(self._f(X))

def build_fbcsp(sfreq, nc, shrink, clf, bands):
    csp_list = [(f'b{f1}_{f2}', BandpassCSP(sfreq=sfreq, fmin=f1, fmax=f2,
                 n_components=nc, shrink=shrink)) for f1, f2 in bands]
    return Pipeline([('fbcsp', FeatureUnion(csp_list)), ('scaler', StandardScaler()), ('clf', clf)])

def evaluate(name, pipe_fn, reject_uv=200e-6, adaptive=True, tmin=0.5, tmax=3.5):
    type_accs = {i: [] for i in range(4)}
    sk = 0
    for subj in subjects:
        for ti, exp in enumerate(EXPERIMENT_TYPES):
            try:
                raw = load_physionet(subj, exp['runs'])
                sfreq = int(raw.info['sfreq'])
                raw = bandpass_filter(raw, 4.0, 40.0)
                epochs = make_epochs(raw, tmin=tmin, tmax=tmax, pick_motor=True)
                if adaptive and reject_uv:
                    ep_r = epochs.copy().drop_bad(reject=dict(eeg=reject_uv))
                    Xr, yr = ep_r.get_data(), ep_r.events[:, -1]
                    if len(np.unique(yr)) >= 2 and len(yr) >= 10 and min(np.bincount(yr - yr.min())) >= 2:
                        X, y = Xr, yr
                    else:
                        X, y = epochs.get_data(), epochs.events[:, -1]
                else:
                    X, y = epochs.get_data(), epochs.events[:, -1]
                if len(np.unique(y)) < 2 or len(y) < 10:
                    sk += 1; continue
                pipe = pipe_fn(sfreq)
                n_cv = min(5, min(np.bincount(y - y.min())))
                if n_cv < 2: sk += 1; continue
                skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
                scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
                type_accs[ti].append(scores.mean())
            except: sk += 1
    tm = [np.mean(type_accs[i]) for i in range(4) if type_accs[i]]
    gm = np.mean(tm) if tm else 0
    det = ' '.join([f'T{i}={np.mean(type_accs[i]):.3f}({len(type_accs[i])})' for i in range(4) if type_accs[i]])
    s = "PASS" if gm >= 0.75 else "fail"
    print(f'{name}: {gm:.4f} [{s}] skip={sk}/{len(subjects)*4} ({det})')
    sys.stdout.flush()

lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
lr = LogisticRegression(max_iter=1000)

# Sub-band configurations
bands5 = [(8, 10), (10, 13), (13, 18), (18, 24), (24, 30)]
bands5b = [(7, 10), (10, 13), (13, 18), (18, 24), (24, 32)]
bands5c = [(8, 11), (11, 14), (14, 19), (19, 25), (25, 30)]
bands4a = [(8, 11), (11, 15), (15, 22), (22, 30)]
bands4b = [(8, 10), (10, 14), (14, 22), (22, 30)]
bands7 = [(7, 9), (9, 11), (11, 14), (14, 18), (18, 22), (22, 26), (26, 32)]

print(f"=== Fine-tuning on {len(subjects)} subjects ===")

# Baseline: FBCSP5_nc2 (best from previous)
print("\n--- nc=2 variations ---")
evaluate('5b_nc2_s003_LDA', lambda sf: build_fbcsp(sf, 2, 0.03, lda, bands5))
evaluate('5b_nc2_s005_LDA', lambda sf: build_fbcsp(sf, 2, 0.05, lda, bands5))
evaluate('5b_nc2_s001_LDA', lambda sf: build_fbcsp(sf, 2, 0.01, lda, bands5))
evaluate('5b_nc2_s000_LDA', lambda sf: build_fbcsp(sf, 2, 0.0, lda, bands5))
evaluate('5bb_nc2_s003_LDA', lambda sf: build_fbcsp(sf, 2, 0.03, lda, bands5b))
evaluate('5bc_nc2_s003_LDA', lambda sf: build_fbcsp(sf, 2, 0.03, lda, bands5c))
evaluate('4a_nc2_s003_LDA', lambda sf: build_fbcsp(sf, 2, 0.03, lda, bands4a))
evaluate('4b_nc2_s003_LDA', lambda sf: build_fbcsp(sf, 2, 0.03, lda, bands4b))
evaluate('7b_nc2_s003_LDA', lambda sf: build_fbcsp(sf, 2, 0.03, lda, bands7))

# nc=3
print("\n--- nc=3 variations ---")
evaluate('5b_nc3_s003_LDA', lambda sf: build_fbcsp(sf, 3, 0.03, lda, bands5))
evaluate('4a_nc3_s003_LDA', lambda sf: build_fbcsp(sf, 3, 0.03, lda, bands4a))

# Time windows
print("\n--- Time window variations ---")
evaluate('5b_nc2_t04-36', lambda sf: build_fbcsp(sf, 2, 0.03, lda, bands5), tmin=0.4, tmax=3.6)
evaluate('5b_nc2_t03-37', lambda sf: build_fbcsp(sf, 2, 0.03, lda, bands5), tmin=0.3, tmax=3.7)
evaluate('5b_nc2_t05-40', lambda sf: build_fbcsp(sf, 2, 0.03, lda, bands5), tmin=0.5, tmax=4.0)

# Rejection thresholds
print("\n--- Rejection variations ---")
evaluate('5b_nc2_rej150', lambda sf: build_fbcsp(sf, 2, 0.03, lda, bands5), reject_uv=150e-6)
evaluate('5b_nc2_rej300', lambda sf: build_fbcsp(sf, 2, 0.03, lda, bands5), reject_uv=300e-6)
evaluate('5b_nc2_norej', lambda sf: build_fbcsp(sf, 2, 0.03, lda, bands5), reject_uv=None, adaptive=False)

# SVM fine-tuning
print("\n--- SVM fine-tuning ---")
evaluate('5b_nc2_SVM1', lambda sf: build_fbcsp(sf, 2, 0.03, SVC(kernel='rbf', C=1.0), bands5))
evaluate('5b_nc2_SVM10', lambda sf: build_fbcsp(sf, 2, 0.03, SVC(kernel='rbf', C=10.0), bands5))
evaluate('5b_nc2_SVM100', lambda sf: build_fbcsp(sf, 2, 0.03, SVC(kernel='rbf', C=100.0), bands5))
evaluate('5b_nc2_SVM_lin', lambda sf: build_fbcsp(sf, 2, 0.03, SVC(kernel='linear', C=1.0), bands5))

print("\nDone!")
