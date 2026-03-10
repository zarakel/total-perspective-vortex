# src/pipeline_model.py
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.features import FeatureExtractor
from src.csp_custom import CSP
from src.lda_custom import LDAClassifier
import joblib as _joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class BandpassCSP(BaseEstimator, TransformerMixin):
    """Apply bandpass filter to a specific frequency sub-band, then CSP.
    Used in Filter Bank CSP (FBCSP): run CSP on multiple frequency
    sub-bands independently and concatenate the resulting features.
    """

    def __init__(self, sfreq=160, fmin=8, fmax=13, n_components=4, shrink=0.05,
                 log_type='var', use_custom_eigen=False):
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
        self.n_components = n_components
        self.shrink = shrink
        self.log_type = log_type
        self.use_custom_eigen = use_custom_eigen
        self.csp = CSP(n_components=n_components, shrink=shrink,
                       log_type=log_type, use_custom_eigen=use_custom_eigen)

    def _filter(self, X):
        from mne.filter import filter_data
        return filter_data(X.astype(np.float64), self.sfreq,
                           self.fmin, self.fmax, verbose=False)

    def fit(self, X, y):
        self.csp.fit(self._filter(X), y)
        return self

    def transform(self, X):
        return self.csp.transform(self._filter(X))

def build_pipeline(sfreq, reducer='csp', reducer_params=None, classifier=None,
                   memory=None, use_features=False, use_custom_clf=False,
                   use_custom_eigen=False, use_fbcsp=False,
                   fbcsp_bands=None):
    """
    Build the processing pipeline:
    - reducer == 'csp' : raw -> CSP -> scaler -> clf
    - use_features True : raw -> FeatureExtractor -> scaler -> clf
    - use_fbcsp True    : raw -> FilterBankCSP (multi-band) -> scaler -> clf

    Bonus flags:
    - use_custom_clf=True  → use custom LDA instead of LogisticRegression
    - use_custom_eigen=True → pass to CSP to use custom eigenvalue decomposition
    """
    if reducer_params is None:
        reducer_params = {}
    if classifier is None:
        if use_custom_clf:
            classifier = LDAClassifier(shrinkage=0.01)
        else:
            classifier = LogisticRegression(max_iter=1000, class_weight='balanced')

    steps = []
    if use_features:
        # Feature-based approach
        steps.append(('fe', FeatureExtractor(sfreq=sfreq, **reducer_params.get('fe', {}))))
        steps.append(('scaler', StandardScaler()))
        steps.append(('clf', classifier))
    elif use_fbcsp:
        # Filter Bank CSP: apply CSP on multiple frequency sub-bands
        if fbcsp_bands is None:
            fbcsp_bands = [(8, 12), (12, 20), (20, 30)]
        csp_params = reducer_params.get('csp', {})
        nc = csp_params.get('n_components', 4)
        shrink = csp_params.get('shrink', 0.03)
        log_type = csp_params.get('log_type', 'var')
        csp_list = []
        for fmin, fmax in fbcsp_bands:
            csp_list.append((
                f'band_{fmin}_{fmax}',
                BandpassCSP(sfreq=sfreq, fmin=fmin, fmax=fmax,
                            n_components=nc, shrink=shrink, log_type=log_type,
                            use_custom_eigen=use_custom_eigen)
            ))
        steps.append(('fbcsp', FeatureUnion(csp_list)))
        steps.append(('scaler', StandardScaler()))
        steps.append(('clf', classifier))
    else:
        # CSP-based approach (expects raw epochs input)
        if reducer == 'pca':
            steps.append(('reducer', PCA(**reducer_params.get('pca', {}))))
        elif reducer == 'csp':
            csp_params = reducer_params.get('csp', {})
            csp_params['use_custom_eigen'] = use_custom_eigen
            steps.append(('reducer', CSP(**csp_params)))
        else:
            raise ValueError("reducer must be 'pca' or 'csp' when use_features is False")
        steps.append(('scaler', StandardScaler()))
        steps.append(('clf', classifier))

    pipe = Pipeline(steps, memory=_joblib.Memory(location=memory) if memory else None)
    return pipe

def train_and_evaluate(pipe, X, y, cv=5, tune=False, param_grid=None, n_jobs=-1, scoring='accuracy'):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    if tune:
        if param_grid is None:
            param_grid = [
                {'reducer__n_components': [4, 6, 8],
                 'clf': [LogisticRegression(max_iter=1000, class_weight='balanced')],
                 'clf__C': [0.01, 0.1, 1, 10]},
                {'reducer__n_components': [4, 6, 8],
                 'clf': [SVC()],
                 'clf__C': [0.1, 1],
                 'clf__kernel': ['rbf', 'linear']},
                {'reducer__n_components': [4, 6, 8],
                 'clf': [RandomForestClassifier()],
                 'clf__n_estimators': [50, 100]}
            ]
        gs = GridSearchCV(pipe, param_grid, cv=skf, n_jobs=n_jobs, scoring=scoring, refit=True, verbose=2)
        gs.fit(X, y)
        print("Best params:", gs.best_params_, "best score:", gs.best_score_)
        best_est = gs.best_estimator_
        scores = cross_val_score(best_est, X, y, cv=skf, n_jobs=n_jobs, scoring=scoring)
        return gs.best_score_, scores, gs
    else:
        scores = cross_val_score(pipe, X, y, cv=skf, n_jobs=n_jobs, scoring=scoring)
        return scores.mean(), scores