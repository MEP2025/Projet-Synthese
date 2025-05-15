import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

def load_data(data_dir):
    # Chargement des données
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='training')
    
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation')
    
    return train_generator, validation_generator

def build_model(num_classes):
    # Construction du modèle
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def train_model(model, train_generator, validation_generator, epochs=10):
    # L'entraîner du modèle
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size)
    
    return history, model

def save_model(model, model_path='model.h5'):
    model.save(model_path)

if __name__ == '__main__':
    # Chemin vers les données
    data_dir = '/C:Users/zhlif/Documents/Intelligence Artificielle/Mise en place/Projet Synthèse/Projet_ML/flower_images'
    
    # Charger les données
    train_gen, val_gen = load_data(data_dir)
    
    # Construire le modèle
    num_classes = len(train_gen.class_indices)
    model = build_model(num_classes)
    
    # Entraîner le modèle
    history, trained_model = train_model(model, train_gen, val_gen, epochs=10)
    
    # Sauvegarder le modèle
    save_model(trained_model)
    
    print("Modèle entraîné et sauvegardé avec succès!")