# Projet-Synthese
Projet-Synthèse_Sihem
# 🌸 Classification de Fleurs avec CNN – Projet Cosmétique & Parfumerie Yves Rocher

## 🧠 Problématique Métier

Dans l’industrie de la parfumerie et des cosmétiques naturels Yves Rocher, l’identification précise des fleurs est cruciale pour :

- ✅ Assurer la qualité des extraits aromatiques (huiles essentielles, absolues)
- ✅ Éviter les erreurs de formulation (certaines fleurs ont des propriétés similaires mais des coûts très différents)
- ✅ Automatiser le contrôle qualité des matières premières florales reçues chez les fabricants

### 🎯 Exemple concret :
Un parfumeur reçoit un lot de "Lavande", mais certaines fleurs sont en réalité du "Lavandin" (moins noble). Une erreur d’identification pourrait entraîner :

- Perte de qualité olfactive
- Surcoûts ou perte financière
- Risques d’allergies en cas de mauvaise variété

## 🤖 Objectif du Projet

Ce projet vise à développer un **modèle de deep learning basé sur un CNN (Convolutional Neural Network)** capable de :

- 🏷️ Classer automatiquement 5 types de fleurs : **Lilly, Lotus, Tulip, Orchidée, et Sunflower**
- 🔍 Vérifier la conformité d’un lot floral à réception
- ⚙️ Être intégré à un processus de contrôle qualité automatisé

## 🗃️ Données

- 📸 Base de données d’images de fleurs collectées sur Kaggle
- 📂 Ces données sont divisées en 5 classes : `Lilly`, `Lotus`, `Tulip`, `Orchid`, `Sunflower`

## 🧪 Technologies utilisées

- Python 3.x
- TensorFlow / Keras
- OpenCV
- Docker
- YAML (pour la gestion de pipeline)
- Jupyter Notebook

## 🏗️ Structure du projet


## Étapes à suivre pour exécuter
    Docker installé
    Python 3.8+
    Git

# 🚀Installation:

    git clone https://github.com/MEP2025/Projet-Synthese/tree/projet_ml_sihem
    cd Projet-Synthèse

# Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirement.txt

# 🏃‍♀️ Exécution: Préparation des données, Entraînement du modèle, l'évaluation, et Déploiement avec Docker
python src/data_preparation.py
python src/train.py --epochs 20 --batch_size 32
python src/evaluate.py --model_path models/best_model.h5
docker build -t flower_classifier .
docker run -p 5000:5000 flower_classifier
