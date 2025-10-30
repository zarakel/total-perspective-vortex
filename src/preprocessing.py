# src/preprocessing.py
import mne

def visualize_raw(raw, show=True, duration=10):
    """Display raw signal in interactive window (requires X11)."""
    raw.plot(n_channels=10, duration=duration, show=show)

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
