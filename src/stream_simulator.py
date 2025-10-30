# src/stream_simulator.py
import time
import numpy as np

def stream_epochs(epochs, labels=None, chunk_seconds=None, sfreq=250, delay_sim=0.0):
    """
    Yields (epoch_array, label, timestamp_end).
    If chunk_seconds specified, will split epoch into chunks (simulate streaming).
    delay_sim simulates network delay.
    """
    for i, e in enumerate(epochs):
        # e shape: (n_channels, n_times)
        if chunk_seconds is None:
            t_end = time.time()
            if delay_sim:
                time.sleep(delay_sim)
            yield e, (labels[i] if labels is not None else None), t_end
        else:
            n_samples = e.shape[1]
            chunk_samples = int(chunk_seconds * sfreq)
            for start in range(0, n_samples, chunk_samples):
                chunk = e[:, start:start+chunk_samples]
                t_end = time.time()
                if delay_sim:
                    time.sleep(delay_sim)
                yield chunk, (labels[i] if labels is not None else None), t_end
