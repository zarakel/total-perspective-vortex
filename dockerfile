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

WORKDIR /app
CMD ["python3"]