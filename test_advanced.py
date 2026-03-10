#!/usr/bin/env python3
"""
Advanced optimization to push accuracy toward 81%.
Strategies:
1. Ensemble classifier (VotingClassifier with LDA + LR + SVM)
2. More FBCSP sub-bands (finer frequency resolution)
3. CSP + PSD hybrid features
4. Per-type optimized parameters (L/R vs FF)
5. Higher rejection threshold with more aggressive filtering
"""
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
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import welch

EXPERIMENT_TYPES = [
    {'name': 'T0-LR_exec',  'runs': [3, 7, 11]},
    {'name': 'T1-LR_imag',  'runs': [4, 8, 12]},
    {'name': 'T2-FF_exec',  'runs': [5, 9, 13]},
    {'name': 'T3-FF_imag',  'runs': [6, 10, 14]},
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


class BandPowerExtractor(BaseEstimator, TransformerMixin):
    """Extract log band power features per channel."""
    def __init__(self, sfreq=160, bands=None):
        self.sfreq = sfreq
        self.bands = bands or [(8,10),(10,12),(12,15),(15,20),(20,25),(25,30)]
    def fit(self, X, y=None): return self
    def transform(self, X):
        feats = []
        for epoch in X:
            f_list = []
            for fmin, fmax in self.bands:
                for ch in epoch:
                    f, Pxx = welch(ch, fs=self.sfreq, nperseg=min(128, len(ch)))
                    mask = (f >= fmin) & (f <= fmax)
                    val = np.mean(Pxx[mask]) if np.any(mask) else 1e-12
                    f_list.append(np.log(max(val, 1e-12)))
            feats.append(f_list)
        return np.array(feats)


def build_fbcsp(sfreq, nc, shrink, clf, bands):
    csp_list = [(f'b{f1}_{f2}', BandpassCSP(sfreq=sfreq, fmin=f1, fmax=f2,
                 n_components=nc, shrink=shrink)) for f1, f2 in bands]
    return Pipeline([('fbcsp', FeatureUnion(csp_list)), ('scaler', StandardScaler()), ('clf', clf)])


def build_hybrid(sfreq, nc, shrink, clf, csp_bands, psd_bands=None):
    """Hybrid: FBCSP features + PSD band power features concatenated."""
    components = []
    for f1, f2 in csp_bands:
        components.append((f'csp_{f1}_{f2}', BandpassCSP(sfreq=sfreq, fmin=f1, fmax=f2,
                           n_components=nc, shrink=shrink)))
    components.append(('psd', BandPowerExtractor(sfreq=sfreq, bands=psd_bands)))
    return Pipeline([('features', FeatureUnion(components)), ('scaler', StandardScaler()), ('clf', clf)])


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
            except Exception as e:
                sk += 1
    tm = [np.mean(type_accs[i]) for i in range(4) if type_accs[i]]
    gm = np.mean(tm) if tm else 0
    det = ' '.join([f'T{i}={np.mean(type_accs[i]):.3f}({len(type_accs[i])})' for i in range(4) if type_accs[i]])
    s = "PASS" if gm >= 0.75 else "fail"
    print(f'{name}: {gm:.4f} [{s}] skip={sk}/{len(subjects)*4} ({det})')
    sys.stdout.flush()


lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
lr = LogisticRegression(max_iter=1000)
bands3 = [(8, 12), (12, 20), (20, 30)]
bands5 = [(8, 10), (10, 13), (13, 18), (18, 24), (24, 30)]
bands6 = [(8, 10), (10, 12), (12, 16), (16, 20), (20, 25), (25, 30)]

print(f"=== Advanced optimization on {len(subjects)} subjects ===")

# Baseline
print("\n--- Baseline (current best) ---")
evaluate('FBCSP3_nc4_s003_LDA', lambda sf: build_fbcsp(sf, 4, 0.03, lda, bands3))

# 1. More sub-bands
print("\n--- More sub-bands ---")
evaluate('FBCSP5_nc4_s003_LDA', lambda sf: build_fbcsp(sf, 4, 0.03, lda, bands5))
evaluate('FBCSP6_nc4_s003_LDA', lambda sf: build_fbcsp(sf, 4, 0.03, lda, bands6))
evaluate('FBCSP5_nc2_s003_LDA', lambda sf: build_fbcsp(sf, 2, 0.03, lda, bands5))
evaluate('FBCSP6_nc2_s003_LDA', lambda sf: build_fbcsp(sf, 2, 0.03, lda, bands6))

# 2. Ensemble classifier
print("\n--- Ensemble classifiers ---")
def make_voting_soft():
    return VotingClassifier(estimators=[
        ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
        ('lr', LogisticRegression(max_iter=1000, C=1.0)),
        ('svm', SVC(kernel='linear', C=1.0, probability=True)),
    ], voting='soft')

def make_voting_hard():
    return VotingClassifier(estimators=[
        ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
        ('lr', LogisticRegression(max_iter=1000, C=1.0)),
        ('svm', SVC(kernel='linear', C=1.0)),
    ], voting='hard')

evaluate('FBCSP3_nc4_VoteSoft', lambda sf: build_fbcsp(sf, 4, 0.03, make_voting_soft(), bands3))
evaluate('FBCSP3_nc4_VoteHard', lambda sf: build_fbcsp(sf, 4, 0.03, make_voting_hard(), bands3))
evaluate('FBCSP5_nc2_VoteSoft', lambda sf: build_fbcsp(sf, 2, 0.03, make_voting_soft(), bands5))

# 3. Hybrid (FBCSP + PSD features)
print("\n--- Hybrid FBCSP + PSD ---")
psd_bands = [(8,10),(10,12),(12,16),(16,20),(20,25),(25,30)]
evaluate('Hybrid_FBCSP3+PSD_LDA', lambda sf: build_hybrid(sf, 4, 0.03, lda, bands3, psd_bands))
evaluate('Hybrid_FBCSP3+PSD_VoteSoft', lambda sf: build_hybrid(sf, 4, 0.03, make_voting_soft(), bands3, psd_bands))

# 4. SVM with RBF kernel (sometimes better for high-dimensional feature spaces)
print("\n--- SVM classifiers ---")
evaluate('FBCSP5_nc2_SVM_rbf', lambda sf: build_fbcsp(sf, 2, 0.03, SVC(kernel='rbf', C=10), bands5))
evaluate('FBCSP3_nc4_SVM_rbf', lambda sf: build_fbcsp(sf, 4, 0.03, SVC(kernel='rbf', C=10), bands3))

print("\nDone!")
