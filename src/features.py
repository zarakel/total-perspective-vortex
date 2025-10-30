import numpy as np
from scipy.signal import welch
import pywt

def compute_psd_epoch(epoch, sfreq, bands=None, nperseg=256):
    """
    epoch: array (n_channels, n_times)
    returns: band power per channel (n_channels * n_bands)
    """
    if bands is None:
        bands = {'theta':(4,8), 'alpha':(8,13), 'beta':(13,30), 'mu':(8,12)}
    n_ch = epoch.shape[0]
    features = []
    for ch in range(n_ch):
        f, Pxx = welch(epoch[ch], fs=sfreq, nperseg=nperseg)
        bandpowers = []
        for (bmin,bmax) in bands.values():
            idx = np.logical_and(f>=bmin, f<=bmax)
            bandpowers.append(Pxx[idx].mean())
        features.append(bandpowers)
    return np.asarray(features).ravel()

def compute_wavelet_epoch(epoch, wavelet='db4', level=4):
    """
    Use discrete wavelet transform coefficients energy as features per channel.
    """
    n_ch = epoch.shape[0]
    feats = []
    for ch in range(n_ch):
        coeffs = pywt.wavedec(epoch[ch], wavelet, level=level)
        # energy per level
        energies = [np.sum(c**2)/len(c) for c in coeffs]
        feats.append(energies)
    return np.asarray(feats).ravel()

class FeatureExtractor:
    """
    sklearn-like transformer that computes concatenated features (PSD + wavelet)
    """
    def __init__(self, sfreq, use_wavelet=True, bands=None):
        self.sfreq = sfreq
        self.use_wavelet = use_wavelet
        self.bands = bands

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X expected shape: (n_epochs, n_channels, n_times)
        feats = []
        for epoch in X:
            p = compute_psd_epoch(epoch, self.sfreq, bands=self.bands)
            if self.use_wavelet:
                w = compute_wavelet_epoch(epoch)
                feats.append(np.concatenate([p, w]))
            else:
                feats.append(p)
        return np.asarray(feats)