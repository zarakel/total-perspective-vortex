# Dockerfile
FROM python:3.11-slim

# Installer juste ce qu'il faut pour tkinter et X11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    && rm -rf /var/lib/apt/lists/*


# Copier uniquement le requirements.txt pour le cache Docker
COPY src/requirements.txt /tmp/requirements.txt

# Installer les dépendances Python
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-configure MNE data path to avoid interactive prompts
RUN mkdir -p /root/mne_data /root/.mne && \
    echo '{"MNE_DATA": "/root/mne_data", "MNE_DATASETS_EEGBCI_PATH": "/root/mne_data"}' > /root/.mne/mne-python.json

WORKDIR /app
CMD ["python3"]