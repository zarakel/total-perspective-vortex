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

def make_epochs(raw, tmin=0.0, tmax=4.0, preload=True):
    import mne
    import numpy as np

    # Extraire les événements
    events, event_id = mne.events_from_annotations(raw)

    # Filtrer uniquement T1 et T2 (main gauche / main droite)
    # → tu peux ajuster selon ton besoin
    event_id = {k: v for k, v in event_id.items() if k in ["T1", "T2"]}

    if len(event_id) < 2:
        raise ValueError(f"Pas assez de classes valides (trouvées : {list(event_id.keys())})")

    print(f"Événements conservés : {list(event_id.keys())}")

    # Créer les epochs
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, picks=picks, preload=preload, baseline=None)

    return epochs

