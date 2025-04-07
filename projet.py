import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# Pour le traitement des images
import cv2
from PIL import Image

# Pour le deep learning
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configuration de base
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 15

def prepare_data(data_dir):
    """
    Charge et prépare les données depuis le répertoire des images
    avec une structure train/test déjà existante
    """
    print(f"Chargement des données depuis: {data_dir}")

    # Classes: chiens (0) et chats (1)
    class_folders = ['dogs', 'cats']
    
    # DataFrames pour stocker les données
    train_data = {'image_path': [], 'label': []}
    test_data = {'image_path': [], 'label': []}
    
    # Accéder aux dossiers train et test
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    
    # Parcourir le dossier d'entraînement
    print("Chargement des données d'entraînement...")
    for class_index, class_name in enumerate(class_folders):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            # Parcourir toutes les images de la classe
            img_count = 0
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.JPG', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    train_data['image_path'].append(img_path)
                    train_data['label'].append(class_index)
                    img_count += 1
            print(f"  - Classe '{class_name}' (train): {img_count} images")
    
    # Parcourir le dossier de test
    print("Chargement des données de test...")
    for class_index, class_name in enumerate(class_folders):
        class_dir = os.path.join(test_dir, class_name)
        if os.path.isdir(class_dir):
            # Parcourir toutes les images de la classe
            img_count = 0
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.JPG', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    test_data['image_path'].append(img_path)
                    test_data['label'].append(class_index)
                    img_count += 1
            print(f"  - Classe '{class_name}' (test): {img_count} images")
    
    # Créer les DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    # Séparer une partie des données d'entraînement pour la validation
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=RANDOM_SEED, stratify=train_df['label']
    )
    
    print(f"Taille de l'ensemble d'entraînement: {len(train_df)}")
    print(f"Taille de l'ensemble de validation: {len(val_df)}")
    print(f"Taille de l'ensemble de test: {len(test_df)}")
    
    return train_df, val_df, test_df, class_folders

# Étape 2 : Visualisation des données
def visualize_data(train_df, val_df, test_df, class_names):
    """
    Visualise la distribution des classes et quelques exemples d'images
    """
    # Distribution des classes dans chaque ensemble
    plt.figure(figsize=(12, 5))
    
    # Ensemble d'entraînement
    plt.subplot(1, 3, 1)
    train_counts = train_df['label'].value_counts().sort_index()
    plt.bar(range(len(train_counts)), train_counts, tick_label=class_names)
    plt.title('Distribution - Entraînement')
    plt.xlabel('Classe')
    plt.ylabel('Nombre d\'images')
    
    # Ensemble de validation
    plt.subplot(1, 3, 2)
    val_counts = val_df['label'].value_counts().sort_index()
    plt.bar(range(len(val_counts)), val_counts, tick_label=class_names)
    plt.title('Distribution - Validation')
    plt.xlabel('Classe')
    
    # Ensemble de test
    plt.subplot(1, 3, 3)
    test_counts = test_df['label'].value_counts().sort_index()
    plt.bar(range(len(test_counts)), test_counts, tick_label=class_names)
    plt.title('Distribution - Test')
    plt.xlabel('Classe')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()
    
    # Afficher quelques exemples d'images (depuis l'ensemble d'entraînement)
    plt.figure(figsize=(10, 5))
    for i, class_idx in enumerate([0, 1]):  # Chiens (0) et Chats (1)
        samples = train_df[train_df['label'] == class_idx].sample(3, random_state=RANDOM_SEED)
        
        for j, (_, row) in enumerate(samples.iterrows()):
            plt.subplot(2, 3, i*3 + j + 1)
            img = Image.open(row['image_path'])
            plt.imshow(img)
            plt.title(class_names[class_idx])
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('example_images.png')
    plt.close()

# Étape 3 : Sélection de features
def extract_features(df, limit=500):
    """
    Extraction de caractéristiques basiques
    """
    print("Extraction de caractéristiques ...")
    
    # Échantillonner un sous-ensemble pour l'extraction de caractéristiques
    if limit:
        sample_df = df.sample(limit, random_state=RANDOM_SEED)
    else:
        sample_df = df
    
    features = []
    labels = []
    
    for _, row in sample_df.iterrows():
        # Charger l'image
        img = cv2.imread(row['image_path'])
        if img is None:
            continue
            
        # Redimensionner
        img = cv2.resize(img, (64, 64))
        
        # Extraire des caractéristiques simples (histogrammes de couleur)
        hist_features = []
        for i in range(3):  # Pour chaque canal (B, G, R)
            hist = cv2.calcHist([img], [i], None, [8], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)
        
        features.append(hist_features)
        labels.append(row['label'])
    
    # Convertir en arrays numpy
    X = np.array(features)
    y = np.array(labels)
    
    print(f"Caractéristiques extraites: {X.shape}")
    
    return X, y

# Étape 4 : Sélection d'hyperparamètres pour le CNN
def hyperparameter_selection(train_df, val_df=None, feature_data=None):
    """
    Fonction pour définir les hyperparamètres du modèle CNN
    """
    print("Configuration des hyperparamètres du modèle...")
    
    # Hyperparamètres pour le CNN
    hyperparams = {
        'learning_rate': 0.001,
        'batch_size': BATCH_SIZE,
        'dropout_rate': 0.5,
        'l2_regularization': 0.001,
        'early_stopping_patience': 3
    }
    
    # Si on veut utiliser des données extraites pour un test préliminaire
    if feature_data is not None:
        X, y = feature_data
        
        print("Exécution d'une recherche préliminaire avec modèle SVM sur caractéristiques extraites...")
        
        # Diviser les données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_SEED
        )
        
        # Définir une grille simple
        param_grid = {
            'C': [1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf']
        }
        
        # Recherche par grille
        grid_search = GridSearchCV(
            SVC(), param_grid, cv=3, scoring='accuracy', verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Afficher les meilleurs paramètres
        print(f"Meilleurs paramètres SVM (test préliminaire): {grid_search.best_params_}")
        test_score = grid_search.score(X_test, y_test)
        print(f"Score du modèle SVM sur les données de test: {test_score:.4f}")
        
        # Ce test préliminaire sert surtout à vérifier si les données sont séparables
        # mais n'influence pas directement les hyperparamètres du CNN
    
    # Afficher les hyperparamètres retenus pour le CNN
    print("Hyperparamètres retenus pour le modèle CNN:")
    for param, value in hyperparams.items():
        print(f"  - {param}: {value}")
    
    return hyperparams

# Étape 5 : Générateurs de données
def create_data_generators(train_df, val_df, test_df):
    """
    Crée des générateurs de données pour l'entraînement, la validation et le test
    """
    # Générateur avec augmentation pour l'entraînement
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Générateur sans augmentation pour validation et test
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Convertir les labels en chaînes pour compatibilité avec flow_from_dataframe
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df['label'] = train_df['label'].astype(str)
    val_df['label'] = val_df['label'].astype(str)
    test_df['label'] = test_df['label'].astype(str)
    
    # Créer les générateurs
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='label',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='image_path',
        y_col='label',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='image_path',
        y_col='label',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

# Étape 6 : Création du modèle CNN
def create_model(hyperparams):
    """
    Crée un modèle CNN pour la classification binaire chien vs chat
    """
    dropout_rate = hyperparams.get('dropout_rate', 0.5)
    l2_reg = hyperparams.get('l2_regularization', 0.001)
    
    model = Sequential([
        # Premier bloc convolutionnel
        Conv2D(32, (3, 3), activation='relu', padding='same', 
               kernel_regularizer=l2(l2_reg),
               input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Deuxième bloc convolutionnel
        Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(l2_reg)),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Troisième bloc convolutionnel
        Conv2D(128, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(l2_reg)),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Aplatissement et couches fully connected
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),  # Pour éviter le surapprentissage
        Dense(2, activation='softmax')  # 2 classes: chien et chat
    ])
    
    # Compilation du modèle
    learning_rate = hyperparams.get('learning_rate', 0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Résumé du modèle
    model.summary()
    
    return model

# Étape 7 : Entraînement du modèle
def train_model(model, train_generator, val_generator, hyperparams):
    """
    Entraîne le modèle avec les données d'entraînement et de validation
    """
    # Callbacks pour améliorer l'entraînement
    patience = hyperparams.get('early_stopping_patience', 3)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath='best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Entraînement du modèle
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks
    )
    
    return history

# Étape 8 : Évaluation du modèle
def evaluate_model(model, test_generator, class_names, history):
    """
    Évalue le modèle sur l'ensemble de test et affiche les métriques
    """
    # Évaluation sur l'ensemble de test
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Précision sur l\'ensemble de test: {test_acc:.4f}')
    
    # Prédictions sur l'ensemble de test
    test_generator.reset()
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Vraies étiquettes
    y_true = test_generator.classes
    
    # Rapport de classification
    print("\nRapport de classification:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    # Matrice de confusion
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prédiction')
    plt.ylabel('Vérité terrain')
    plt.title('Matrice de confusion')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Visualisation des courbes d'apprentissage
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Précision du modèle')
    plt.ylabel('Précision')
    plt.xlabel('Époque')
    plt.legend(['Entraînement', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perte du modèle')
    plt.ylabel('Perte')
    plt.xlabel('Époque')
    plt.legend(['Entraînement', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()
    
    return test_acc

# Fonction principale qui exécute toutes les étapes
def main(data_dir):
    """
    Fonction principale qui exécute toutes les étapes du pipeline
    """
    print("Étape 1: Chargement et préparation des données...")
    train_df, val_df, test_df, class_names = prepare_data(data_dir)
    
    print("\nÉtape 2: Visualisation des données...")
    visualize_data(train_df, val_df, test_df, class_names)
    
    print("\nÉtape 3: Extraction de caractéristiques simples...")
    X, y = extract_features(train_df, limit=500)
    
    print("\nÉtape 4: Sélection d'hyperparamètres pour le CNN...")
    best_params = hyperparameter_selection(train_df, val_df, feature_data=(X, y))
    
    print("\nÉtape 5: Création des générateurs de données...")
    train_generator, val_generator, test_generator = create_data_generators(train_df, val_df, test_df)
    
    print("\nÉtape 6: Création du modèle CNN...")
    model = create_model(best_params)
    
    print("\nÉtape 7: Entraînement du modèle...")
    history = train_model(model, train_generator, val_generator, best_params)
    
    print("\nÉtape 8: Évaluation du modèle...")
    test_acc = evaluate_model(model, test_generator, class_names, history)
    
    print(f"\nModèle entraîné avec succès et sauvegardé sous 'best_model.h5'")
    print(f"Précision sur l'ensemble de test: {test_acc:.4f}")
    
    return model, class_names, test_acc


if __name__ == "__main__":
    # Spécifier le chemin vers le dataset de chiens et chats
    data_dir = "dogs_vs_cats" 
    
    # Exécuter le pipeline complet
    model, class_names, accuracy = main(data_dir)