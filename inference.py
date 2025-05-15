import tensorflow as tf
import numpy as np
import cv2 # OpenCV pour la manipulation d'images
# Charger le modèle entraîné
model = tf.keras.models.load_model("model.h5")
# Charger une image test
image_path = "test.jpg" # Remplace par une vraie image de test
image = cv2.imread(image_path) # Charger l'image
image = cv2.resize(image, (224, 224)) # Adapter à la taille d’entrée du
modèle
image = image / 255.0 # Normalisation des pixels entre 0 et 1
image = np.expand_dims(image, axis=0) # Ajouter une dimension batch
# Faire une prédiction
prediction = model.predict(image)
predicted_class = np.argmax(prediction) # Trouver la classe avec la plus
haute probabilité
print(f"Classe prédite : {predicted_class}, Probabilités : {prediction}")