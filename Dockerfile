FROM tensorflow/tensorflow:2.10.0
ENV DEBIAN_FRONTEND=noninteractive

# Dépendances système pour OpenCV + autres
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libsm6 libxext6 libxrender1 && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirement.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirement.txt && \
    pip install gunicorn flask gevent

# Copie des fichiers 
COPY . /app
WORKDIR /app

# Configuration SageMaker
ENV MODEL_DIR=/opt/ml/model
ENV PYTHONUNBUFFERED=TRUE

# Copie du modèle 
RUN tar -xzf model.tar.gz -C ${MODEL_DIR}/  # Décompresse le modèle dans /opt/ml/model

CMD ["python", "inference.py"]