# src/loader.py
import mne
import numpy as np
import os

# Set MNE data path to avoid interactive prompts
_mne_data_dir = os.path.expanduser('~/mne_data')
os.makedirs(_mne_data_dir, exist_ok=True)
os.environ['MNE_DATA'] = _mne_data_dir
mne.set_config('MNE_DATA', _mne_data_dir, set_env=False)

def load_bci4_2a(subject, preload=True):
    """Load BCI Competition IV dataset 2a (9 subjects, 22 channels, 2-class motor imagery).
    Uses MOABB for standardized access. Returns mne.io.Raw.
    """
    from moabb.datasets import BNCI2014001
    dataset = BNCI2014001()
    sessions = dataset.get_data(subjects=[subject])
    # Concatenate all session runs into one Raw
    raws = []
    for sess_name, sess_runs in sessions[subject].items():
        for run_name, raw in sess_runs.items():
            raws.append(raw)
    raw = mne.io.concatenate_raws(raws)
    return raw


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
    # Standardize channel names (PhysioNet uses 'C3..' instead of 'C3')
    eegbci.standardize(raw)
    # set montage if available
    try:
        raw.set_montage('standard_1005', on_missing='ignore')
    except Exception:
        pass
    return raw

def make_epochs(raw, tmin=0.5, tmax=2.5, preload=True, window_sec=None, overlap=0.5, pick_motor=True, event_labels=None):
    """Create epochs from raw EEG.
    event_labels: list of annotation labels to keep (default: ['T1', 'T2'] for PhysioNet).
    """
    import mne
    import numpy as np

    if event_labels is None:
        event_labels = ["T1", "T2"]
    events, event_id = mne.events_from_annotations(raw)
    event_id = {k: v for k, v in event_id.items() if k in event_labels}
    if len(event_id) < 2:
        raise ValueError(f"Pas assez de classes valides (trouvées : {list(event_id.keys())})")
    if pick_motor:
        # Select sensorimotor channels only (better for CSP with limited data)
        motor_channels = [
            'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
        ]
        available = [ch for ch in motor_channels if ch in raw.info['ch_names']]
        if len(available) >= 10:
            picks = mne.pick_channels(raw.info['ch_names'], available, ordered=True)
        else:
            picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    else:
        picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    base_epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, picks=picks, preload=preload, baseline=None)

    # Si pas d'augmentation demandée, retourne les epochs MNE classiques
    if window_sec is None:
        return base_epochs

    sfreq = int(raw.info['sfreq'])
    win_samps = int(window_sec * sfreq)
    if win_samps <= 0:
        return base_epochs

    step = max(1, int(win_samps * (1.0 - overlap)))
    data = base_epochs.get_data()  # shape (n_epochs, n_chan, n_times)
    labels = base_epochs.events[:, -1]

    new_data = []
    new_events = []
    for ie, ep in enumerate(data):
        n_times = ep.shape[1]
        for start in range(0, n_times - win_samps + 1, step):
            seg = ep[:, start:start + win_samps]
            new_data.append(seg)
            # events: [sample_index, 0, label] sample_index unused here but required shape
            new_events.append([len(new_data)-1, 0, int(labels[ie])])

    new_data = np.asarray(new_data)
    info = base_epochs.info.copy()
    epochs_array = mne.EpochsArray(new_data, info=info, events=np.array(new_events), tmin=0.0, event_id=event_id)
    return epochs_array
