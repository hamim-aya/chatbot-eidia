"""
Chatbot EIDIA-UEMF avec TensorFlow/Keras
=========================================
Système de chatbot avec réseau de neurones pour comprendre les intentions utilisateur

CORRECTIONS APPLIQUÉES POUR L'OVERFITTING:
- Réduction de la complexité du modèle
- Augmentation du dropout (0.7 et 0.5)
- Ajout de régularisation L2 (0.001)
- Réduction du learning rate (0.0005)
- Implémentation d'Early Stopping
"""

import os
import json
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dropout, Dense
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

from config import (
    INTENTS_FILE, RESPONSES_FILE, MODEL_PATH, TOKENIZER_PATH,
    LABEL_ENCODER_PATH, HISTORY_PATH, CURVES_PATH,
    MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, LSTM_UNITS, HIDDEN_UNITS,
    DROPOUT_RATE_1, DROPOUT_RATE_2, L2_REGULARIZATION,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, VALIDATION_SPLIT,
    EARLY_STOPPING_PATIENCE, CONFIDENCE_THRESHOLD, RANDOM_SEED,
    ENABLE_VERBOSE
)

warnings.filterwarnings('ignore')

# Fixer la graine pour la reproductibilité
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class ChatbotModel:
    """Classe principal pour gérer le chatbot"""
    
    def __init__(self):
        self.tokenizer = None
        self.label_encoder = None
        self.model = None
        self.intents = {}
        self.responses = {}
        self.history = None
        
    def load_intents(self):
        """Charger les intentions depuis intents.json"""
        with open(INTENTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Gérer le cas où les intents sont sous une clé 'intents'
            if isinstance(data, dict) and 'intents' in data:
                self.intents = data['intents']
            else:
                self.intents = data
        print(f"✓ Loaded {len(self.intents)} intents from {INTENTS_FILE}")
        return self.intents
    
    def load_responses(self):
        """Charger les réponses depuis responses.json"""
        with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Gérer le cas où les réponses sont sous une clé 'responses'
            if isinstance(data, dict) and 'responses' in data:
                self.responses = data['responses']
            else:
                self.responses = data
        print(f"✓ Loaded responses for {len(self.responses)} tags")
        return self.responses
    
    def preprocess_data(self):
        """Prétraiter les données: tokenization et padding avec stratified split"""
        
        # Récupérer tous les patterns et leurs tags
        patterns = []
        labels = []
        
        for intent in self.intents:
            for pattern in intent.get('patterns', []):
                patterns.append(pattern.lower())
                labels.append(intent['tag'])
        
        print(f"✓ Total patterns: {len(patterns)}")
        print(f"✓ Total intents: {len(set(labels))}")
        
        # Tokenizer
        self.tokenizer = Tokenizer(num_words=2000)
        self.tokenizer.fit_on_texts(patterns)
        sequences = self.tokenizer.texts_to_sequences(patterns)
        X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        
        # Label [tag]Encoder
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(labels)
        y = tf.keras.utils.to_categorical(y_encoded, num_classes=len(self.label_encoder.classes_))
        
        # IMPORTANT: Utiliser stratified split pour garantir une distribution équilibrée
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=VALIDATION_SPLIT,
            random_state=RANDOM_SEED,
            stratify=y_encoded  # KEY: Assurer que chaque classe est représentée
        )
        
        print(f"✓ X_train shape: {X_train.shape} | X_val shape: {X_val.shape}")
        print(f"✓ y_train shape: {y_train.shape} | y_val shape: {y_val.shape}")
        
        # Analyser la distribution
        train_dist = np.sum(y_train, axis=0)
        val_dist = np.sum(y_val, axis=0)
        print(f"\n✓ Training distribution (first 5 classes): {train_dist[:5]}")
        print(f"✓ Validation distribution (first 5 classes): {val_dist[:5]}")
        
        return X_train, X_val, y_train, y_val, patterns, labels
    
    def build_model(self, num_intents):
        """Construire le modèle de réseau de neurones avec régularisation L2"""
        
        self.model = Sequential([
            # Couche d'embedding
            Embedding(
                input_dim=2000,
                output_dim=EMBEDDING_DIM,
                input_length=MAX_SEQUENCE_LENGTH,
                name='embedding'
            ),
            
            # Bidirectional LSTM (réduit à 128 pour éviter l'overfitting)
            Bidirectional(
                LSTM(LSTM_UNITS, return_sequences=False, dropout=0.2),
                name='bilstm'
            ),
            
            # Dropout 1 (augmenté à 0.7)
            Dropout(DROPOUT_RATE_1, name='dropout_1'),
            
            # Couche Dense cachée avec régularisation L2
            Dense(
                HIDDEN_UNITS,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION),
                name='dense_hidden'
            ),
            
            # Dropout 2 (augmenté à 0.5)
            Dropout(DROPOUT_RATE_2, name='dropout_2'),
            
            # Couche de sortie (softmax pour classification multi-classe)
            Dense(num_intents, activation='softmax', name='output')
        ])
        
        # Compiler le modèle avec learning rate réduit
        optimizer = Adam(learning_rate=LEARNING_RATE)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✓ Model compiled successfully")
        self.model.summary()
        
        return self.model
    
    def train_model(self, X_train, X_val, y_train, y_val):
        """Entraîner le modèle avec Early Stopping et validation set séparé"""
        
        # Early Stopping pour arrêter l'entraînement si pas d'amélioration
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        
        print("\n" + "="*70)
        print("ENTRAÎNEMENT DU MODÈLE AVEC CORRECTIONS D'OVERFITTING")
        print("="*70)
        print(f"Dropout 1: {DROPOUT_RATE_1} | Dropout 2: {DROPOUT_RATE_2}")
        print(f"L2 Regularization: {L2_REGULARIZATION}")
        print(f"Learning Rate: {LEARNING_RATE}")
        print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE} epochs")
        print(f"Training set: {X_train.shape[0]} samples | Validation set: {X_val.shape[0]} samples")
        print("="*70 + "\n")
        
        # Utiliser validation_data au lieu de validation_split
        # Cela utilise l'ensemble de validation que nous avons créé avec stratification
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),  # KEY: Utiliser le validation set stratifié
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[early_stopping],
            verbose=1 if ENABLE_VERBOSE else 0
        )
        
        # Afficher les résultats finaux
        print("\n" + "="*70)
        print("RÉSULTATS D'ENTRAÎNEMENT FINAUX")
        print("="*70)
        final_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        final_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        
        print(f"Training Accuracy: {final_acc:.4f}")
        print(f"Validation Accuracy: {final_val_acc:.4f}")
        print(f"Training Loss: {final_loss:.4f}")
        print(f"Validation Loss: {final_val_loss:.4f}")
        print(f"Écart train/val: {abs(final_acc - final_val_acc):.4f}")
        
        if abs(final_acc - final_val_acc) < 0.15:
            print("✓ BON SIGNE: L'écart train/validation est acceptable (<15%)")
        else:
            print("⚠ ATTENTION: L'écart est important, overfitting détecté")
        print("="*70 + "\n")
        
        return self.history
    
    def save_model(self):
        """Sauvegarder le modèle et les composants"""
        
        # Sauvegarder le modèle
        self.model.save(MODEL_PATH)
        print(f"✓ Model saved to {MODEL_PATH}")
        
        # Sauvegarder le tokenizer
        with open(TOKENIZER_PATH, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"✓ Tokenizer saved to {TOKENIZER_PATH}")
        
        # Sauvegarder le label encoder
        with open(LABEL_ENCODER_PATH, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✓ Label encoder saved to {LABEL_ENCODER_PATH}")
        
        # Sauvegarder l'historique
        with open(HISTORY_PATH, 'wb') as f:
            pickle.dump(self.history.history, f)
        print(f"✓ History saved to {HISTORY_PATH}")
    
    def plot_training_curves(self):
        """Tracer les courbes d'entraînement"""
        
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(CURVES_PATH)
        print(f"✓ Training curves saved to {CURVES_PATH}")
        plt.close()
    
    def load_model(self):
        """Charger un modèle existant"""
        
        if not os.path.exists(MODEL_PATH):
            print(f"Model not found at {MODEL_PATH}")
            return False
        
        self.model = tf.keras.models.load_model(MODEL_PATH)
        
        with open(TOKENIZER_PATH, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print("✓ Model loaded successfully")
        return True
    
    def predict(self, user_input):
        """Prédire l'intention de l'utilisateur"""
        
        # Prétraiter l'entrée
        sequence = self.tokenizer.texts_to_sequences([user_input.lower()])
        padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        
        # Prédire
        prediction = self.model.predict(padded, verbose=0)[0]
        
        # Obtenir le tag
        intent_index = np.argmax(prediction)
        intent_tag = self.label_encoder.inverse_transform([intent_index])[0]
        confidence = float(prediction[intent_index])
        
        return intent_tag, confidence, prediction
    
    def chat_response(self, user_input):
        """Générer une réponse basée sur l'intention prédite"""
        
        intent_tag, confidence, _ = self.predict(user_input)
        
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                'response': "Je n'ai pas bien compris. Pouvez-vous reformuler?",
                'intent': 'unknown',
                'confidence': 0
            }
        
        # Récupérer une réponse aléatoire pour cet intent
        if intent_tag in self.responses:
            resp_data = self.responses[intent_tag]
            # Gérer deux cas: liste directe ou {'templates': [...]}
            if isinstance(resp_data, dict) and 'templates' in resp_data:
                response = random.choice(resp_data['templates'])
            elif isinstance(resp_data, list):
                response = random.choice(resp_data)
            else:
                response = str(resp_data)
        else:
            response = "Je n'ai pas de réponse pour cette intention."
        
        return {
            'response': response,
            'intent': str(intent_tag),
            'confidence': confidence
        }


def main():
    """Fonction principale"""
    
    # Créer et configurer le chatbot
    chatbot = ChatbotModel()
    
    # Charger les données
    chatbot.load_intents()
    chatbot.load_responses()
    
    # Prétraiter les données avec STRATIFIED SPLIT
    X_train, X_val, y_train, y_val, patterns, labels = chatbot.preprocess_data()
    
    # Construire et entraîner le modèle
    num_intents = len(chatbot.label_encoder.classes_)
    chatbot.build_model(num_intents)
    chatbot.train_model(X_train, X_val, y_train, y_val)
    
    # Sauvegarder et visualiser
    chatbot.save_model()
    chatbot.plot_training_curves()
    
    print("\n✓ Chatbot trained and saved successfully!")
    print("✓ To use the chatbot, import ChatbotModel and call load_model() then chat_response()")


if __name__ == "__main__":
    main()
