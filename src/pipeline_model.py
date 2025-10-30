# src/pipeline_model.py
import joblib
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .features import FeatureExtractor
from .csp_custom import CSP

def build_pipeline(sfreq, reducer='csp', reducer_params=None, classifier=None):
    if reducer_params is None: reducer_params = {}
    if classifier is None:
        classifier = LogisticRegression(max_iter=500)

    steps = []
    steps.append(('fe', FeatureExtractor(sfreq=sfreq, use_wavelet=True)))
    if reducer == 'pca':
        steps.append(('reducer', PCA(**reducer_params)))
    elif reducer == 'csp':
        steps.append(('reducer', CSP(**reducer_params)))
    else:
        raise ValueError("reducer must be 'pca' or 'csp'")

    steps.append(('clf', classifier))
    pipe = Pipeline(steps)
    return pipe

def train_and_evaluate(pipe, X, y, cv=5):
    scores = cross_val_score(pipe, X, y, cv=cv, n_jobs=1)
    return scores.mean(), scores
