import tensorflow as tf
import numpy as np
import cv2
import json
import io
from PIL import Image

# Liste des classes correspondantes aux indices du modèle
classes = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']

def model_fn(model_dir):
    """Charge le modèle TensorFlow depuis le dossier model_dir"""
    model = tf.keras.models.load_model(f"{model_dir}/model.h5")
    return model

def input_fn(request_body, request_content_type):
    """Transforme les données d'entrée en format utilisable par le modèle"""
    if request_content_type == 'application/json':
        # Si les données sont envoyées en JSON (contient le chemin ou les bytes)
        request = json.loads(request_body)
        if 'image' in request:
            # Décoder l'image si elle est encodée en base64
            image_bytes = io.BytesIO(base64.b64decode(request['image']))
            image = Image.open(image_bytes)
        else:
            raise ValueError("Clé 'image' manquante dans la requête JSON")
    elif request_content_type.startswith('image/'):
        # Si l'image est envoyée directement (binary data)
        image = Image.open(io.BytesIO(request_body))
    else:
        raise ValueError(f"Type de contenu non supporté: {request_content_type}")
    
    # Convertir en array numpy et prétraiter
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Si nécessaire
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def predict_fn(input_data, model):
    """Fait la prédiction avec le modèle et les données d'entrée"""
    prediction = model.predict(input_data)
    return prediction

def output_fn(prediction, response_content_type):
    """Transforme la prédiction en format de réponse"""
    predicted_class = np.argmax(prediction)
    class_name = classes[predicted_class]
    probabilities = prediction.tolist()[0]
    
    if response_content_type == 'application/json':
        return json.dumps({
            'predicted_class': class_name,
            'probabilities': probabilities,
            'class_index': int(predicted_class)
        })
    else:
        raise ValueError(f"Type de réponse non supporté: {response_content_type}")