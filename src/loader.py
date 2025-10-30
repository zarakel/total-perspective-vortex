# src/loader.py
import mne
import numpy as np

def load_physionet(subject:int, runs:list, preload=True):
    """
    Example loader that uses MNE's Physionet retrieval for motor imagery dataset (EEGBCI)
    subject: subject id (int)
    runs: list of run numbers to load
    returns: raw (mne.io.Raw)
    """
    # This uses mne.datasets.eegbci. Adjust depending on dataset you choose.
    from mne.datasets import eegbci
    files = eegbci.load_data(subject, runs)
    raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=preload) for f in files])
    # set montage if available
    try:
        raw.set_montage('standard_1005', on_missing='ignore')
    except Exception:
        pass
    return raw

def make_epochs(raw, tmin=0.0, tmax=4.0, event_id=None, preload=True):
    events, event_id_map = mne.events_from_annotations(raw)
    if event_id is None:
        event_id = event_id_map
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False)
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, picks=picks, preload=preload)
    # returns data as numpy: shape (n_epochs, n_channels, n_times)
    return epochs
