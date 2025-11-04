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

def make_epochs(raw, tmin=0.5, tmax=2.5, preload=True, window_sec=None, overlap=0.5):
    import mne
    import numpy as np

    events, event_id = mne.events_from_annotations(raw)
    event_id = {k: v for k, v in event_id.items() if k in ["T1", "T2"]}
    if len(event_id) < 2:
        raise ValueError(f"Pas assez de classes valides (trouvées : {list(event_id.keys())})")
    target_channels = ['C3', 'C4', 'Cz', 'FC3', 'FC4', 'CP3', 'CP4']
    picks = mne.pick_channels(raw.info['ch_names'], target_channels, ordered=True)
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
