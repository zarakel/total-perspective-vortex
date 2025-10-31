import numpy as np
import pywt

def compute_psd_epoch(epoch, sfreq, bands):
    """Calcule la puissance moyenne dans les bandes spécifiées."""
    from scipy.signal import welch
    psd_feats = []
    for ch in epoch:
        freqs, psd = welch(ch, sfreq, nperseg=min(256, len(ch)))
        band_powers = []
        for (fmin, fmax) in bands:
            mask = (freqs >= fmin) & (freqs <= fmax)
            band_powers.append(np.mean(psd[mask]))
        psd_feats.append(band_powers)
    return np.asarray(psd_feats).flatten()

def compute_wavelet_epoch(signal, wavelet='db4', level=3):
    """
    Calcule la décomposition en ondelettes discrètes pour un signal 1D
    et retourne un petit vecteur de caractéristiques résumées (énergies par niveau).
    """
    signal = np.asarray(signal, dtype=np.float32)

    # Si signal constant ou nul
    if np.allclose(signal, 0):
        return np.zeros(level + 1, dtype=np.float32)

    try:
        coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
        # ⚡ On résume chaque niveau d’ondelette par son énergie moyenne absolue
        energies = np.array([np.mean(np.abs(c)) for c in coeffs], dtype=np.float32)
        return energies
    except Exception as e:
        print(f"[⚠️ Wavelet failed for signal of shape {signal.shape}: {e}]")
        return np.zeros(level + 1, dtype=np.float32)


class FeatureExtractor:
    def __init__(self, sfreq, use_wavelet=True):
        self.sfreq = sfreq
        self.use_wavelet = use_wavelet
        self.bands = [(8, 12), (12, 16), (16, 25), (25, 30)]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for epoch in X:
            p = compute_psd_epoch(epoch, self.sfreq, self.bands)

            if self.use_wavelet:
                w_epoch = []
                for ch in epoch:
                    w = compute_wavelet_epoch(ch)
                    if w is None or not np.any(np.isfinite(w)):
                        w = np.zeros(64)
                    w_epoch.append(w)

                # Concatène tous les canaux + PSD
                w_epoch = np.concatenate(w_epoch)
                feats.append(np.concatenate([p, w_epoch]))
            else:
                feats.append(p)

        feats = np.asarray(feats)
        print(f"✅ FeatureExtractor output shape: {feats.shape}")
        return feats
