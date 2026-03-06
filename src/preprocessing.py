# src/preprocessing.py
import mne
import matplotlib.pyplot as plt
import numpy as np

def visualize_raw(raw, show=True, duration=10, title=None):
    """Display raw signal in interactive window (requires X11)."""
    fig = raw.plot(n_channels=10, duration=duration, show=False)
    if title:
        fig.suptitle(title)
    plt.show(block=True)

def visualize_spectrum(raw, title=None, fmin=0.5, fmax=50.0):
    """
    Plot PSD (Power Spectral Density) of the raw signal.
    Useful for verifying filtering and exploring frequency content.
    """
    fig = raw.compute_psd(fmin=fmin, fmax=fmax).plot(show=False)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show(block=True)

def bandpass_filter(raw, l_freq=8.0, h_freq=30.0):
    """
    Apply bandpass filter to raw data in-place and return filtered raw.
    Keep filter params simple and robust for low memory.
    """
    raw.filter(l_freq, h_freq, fir_design='firwin', verbose=False)
    return raw

def notch_filter(raw, freqs=[50.0, 100.0]):
    raw.notch_filter(freqs, verbose=False)
    return raw
