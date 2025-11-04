# src/pipeline_model.py
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.features import FeatureExtractor
from src.csp_custom import CSP
import joblib as _joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def build_pipeline(sfreq, reducer='csp', reducer_params=None, classifier=None, memory=None, use_features=False):
    """
    Construis la pipeline correctement :
    - si reducer == 'csp' et use_features False : raw -> CSP -> scaler -> clf
    - si use_features True : raw -> FeatureExtractor -> scaler -> clf
    (on n'essaie pas de mettre CSP après FeatureExtractor)
    """
    if reducer_params is None:
        reducer_params = {}
    if classifier is None:
        classifier = LogisticRegression(max_iter=1000, class_weight='balanced')

    steps = []
    if use_features:
        # Feature-based approach
        steps.append(('fe', FeatureExtractor(sfreq=sfreq, **reducer_params.get('fe', {}))))
        steps.append(('scaler', StandardScaler()))
        steps.append(('clf', classifier))
    else:
        # CSP-based approach (expects raw epochs input)
        if reducer == 'pca':
            steps.append(('reducer', PCA(**reducer_params.get('pca', {}))))
        elif reducer == 'csp':
            steps.append(('reducer', CSP(**reducer_params.get('csp', {}))))
        else:
            raise ValueError("reducer must be 'pca' or 'csp' when use_features is False")
        steps.append(('scaler', StandardScaler()))
        steps.append(('clf', classifier))

    pipe = Pipeline(steps, memory=_joblib.Memory(location=memory) if memory else None)
    return pipe

def train_and_evaluate(pipe, X, y, cv=5, tune=False, param_grid=None, n_jobs=-1, scoring='balanced_accuracy'):
    print("Unique labels in y:", np.unique(y))
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