from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pywt
from scipy.signal import welch, decimate

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformateur scikit-learn compatible qui extrait PSD (par bande) et features wavelet
    - accepts: sfreq, use_wavelet, wavelet_level, use_car, downsample, bands
    - fit() est une no-op (retourne self) pour compatibilité pipeline
    """
    def __init__(self, sfreq, use_wavelet=True, wavelet='db4', wavelet_level=3,
                 use_car=True, downsample=None, bands=None):
        self.sfreq = sfreq
        self.use_wavelet = use_wavelet
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.use_car = use_car
        self.downsample = downsample
        self.bands = bands or [(8, 12), (12, 16), (16, 25), (25, 30)]
        self.wavelet_feature_len = (self.wavelet_level + 1) * 2

    def fit(self, X, y=None):
        # no training required, kept for sklearn compatibility
        return self

    def transform(self, X):
        """
        X: array-like, shape (n_epochs, n_channels, n_times)
        returns: array shape (n_epochs, n_features)
        """
        feats = []
        for epoch in X:
            ep = epoch.copy()
            # Common average reference
            if self.use_car:
                ep = ep - np.mean(ep, axis=0, keepdims=True)

            sfreq = self.sfreq
            # optional downsample
            if self.downsample and int(self.downsample) > 0 and int(self.downsample) != int(self.sfreq):
                factor = max(1, int(round(self.sfreq / self.downsample)))
                ep = np.array([decimate(ch, factor, zero_phase=True) for ch in ep])
                sfreq = int(round(self.sfreq / factor))

            # PSD per band (log1p of mean band power per channel)
            band_powers = []
            for (fmin, fmax) in self.bands:
                for ch in ep:
                    f, Pxx = welch(ch, fs=sfreq, nperseg=min(256, len(ch)))
                    mask = (f >= fmin) & (f <= fmax)
                    val = np.mean(Pxx[mask]) if np.any(mask) else 0.0
                    band_powers.append(np.log1p(val))
            p = np.asarray(band_powers, dtype=np.float32)

            if self.use_wavelet:
                w_epoch = []
                for ch in ep:
                    try:
                        coeffs = pywt.wavedec(ch, wavelet=self.wavelet, level=self.wavelet_level)
                        energies = np.concatenate([[np.mean(np.abs(c)), np.std(c)] for c in coeffs]).astype(np.float32)
                    except Exception:
                        energies = np.zeros(self.wavelet_feature_len, dtype=np.float32)
                    # pad/truncate to fixed length per channel
                    if energies.size < self.wavelet_feature_len:
                        energies = np.pad(energies, (0, self.wavelet_feature_len - energies.size), 'constant')
                    elif energies.size > self.wavelet_feature_len:
                        energies = energies[:self.wavelet_feature_len]
                    w_epoch.append(energies)
                w_epoch = np.concatenate(w_epoch)
                feats.append(np.concatenate([p, w_epoch]))
            else:
                feats.append(p)

        feats = np.asarray(feats, dtype=np.float32)
        return feats