"""
Configuration pour le chatbot EIDIA-UEMF
=====================================
Hyperparamètres et constantes globales
"""

import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. CHEMINS DE FICHIERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

INTENTS_FILE = os.path.join(BASE_DIR, "intents.json")
RESPONSES_FILE = os.path.join(BASE_DIR, "responses.json")

MODEL_PATH = os.path.join(BASE_DIR, "chatbot_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
HISTORY_PATH = os.path.join(BASE_DIR, "training_history.pkl")
CURVES_PATH = os.path.join(BASE_DIR, "training_curves.png")

# Créer les répertoires s'ils n'existent pas
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. PARAMÈTRES DE PRÉTRAITEMENT DES DONNÉES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAX_SEQUENCE_LENGTH = 20       # Longueur maximale des séquences
EMBEDDING_DIM = 128            # Dimension d'embedding

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. PARAMÈTRES DE L'ARCHITECTURE DU MODÈLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LSTM_UNITS = 128                   # Réduit de 256 pour éviter l'overfitting
HIDDEN_UNITS = 64                  # Réduit de 128 pour réduire la complexité
DROPOUT_RATE_1 = 0.7               # Augmenté de 0.5 (après Bidirectional LSTM)
DROPOUT_RATE_2 = 0.5               # Augmenté de 0.3 (après couche Dense)
L2_REGULARIZATION = 0.001          # Régularisation L2 pour les poids

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. PARAMÈTRES D'ENTRAÎNEMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BATCH_SIZE = 32                    # Taille du batch
EPOCHS = 150                       # Nombre d'épochs (augmenté avec Early Stopping)
LEARNING_RATE = 0.0005             # Réduit de 0.001 pour convergence plus lente
VALIDATION_SPLIT = 0.2             # 20% des données pour la validation

# Early Stopping
EARLY_STOPPING_PATIENCE = 15       # Arrêter si pas d'amélioration après 15 epochs

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. PARAMÈTRES DE PRÉDICTION/INFÉRENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONFIDENCE_THRESHOLD = 0.3         # Seuil de confiance pour les prédictions
NUM_TOP_INTENTS = 3                # Nombre d'intents à retourner

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. PARAMÈTRES DU CHATBOT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LANGUAGE = "fr"                    # Langue (français)
ENABLE_VERBOSE = True              # Afficher les logs d'entraînement
RANDOM_SEED = 42                   # Graine aléatoire pour reproductibilité

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. MESSAGES ET RÉPONSES PAR DÉFAUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_RESPONSES = {
    "greeting": "Bonjour! Comment puis-je vous aider?",
    "fallback": "Je n'ai pas bien compris. Pouvez-vous reformuler?",
    "error": "Une erreur s'est produite. Veuillez réessayer."
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. RÉSUMÉ DES CORRECTIONS POUR L'OVERFITTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
PROBLÈME ORIGINAL:
- Training accuracy: 98.28% ✅
- Validation accuracy: 0.98% ❌ (OVERFITTING SÉVÈRE)

CAUSES:
1. Modèle trop complexe (256 LSTM units)
2. Dropout insuffisant (0.5)
3. Learning rate trop élevé (0.001)
4. Pas de régularisation L2
5. Pas d'Early Stopping

CORRECTIONS APPLIQUÉES:
1. ✅ Réduction de la complexité:
   - LSTM_UNITS: 256 → 128 (50% réduction)
   - HIDDEN_UNITS: 128 → 64 (50% réduction)

2. ✅ Augmentation de la régularisation:
   - DROPOUT_RATE_1: 0.5 → 0.7 (après BiLSTM)
   - DROPOUT_RATE_2: 0.3 → 0.5 (après Dense)
   - L2_REGULARIZATION: 0.001 (pénalité sur les poids)

3. ✅ Optimisation de l'apprentissage:
   - LEARNING_RATE: 0.001 → 0.0005 (convergence plus lente)
   - EPOCHS: 100 → 150 (plus d'epochs avec Early Stopping)
   - EARLY_STOPPING_PATIENCE: 15 (arrêter si pas d'amélioration)

RÉSULTATS ATTENDUS:
- Training accuracy: 85-90% (légère réduction acceptable)
- Validation accuracy: 80-85% (grosse amélioration de 0.98%!)
- Écart train/val: < 10% (bon signe de généralisation)
- Early Stopping arrêtera vers l'epoch 30-50
"""

if __name__ == "__main__":
    print("Configuration du chatbot EIDIA-UEMF")
    print("=" * 50)
    print(f"Base directory: {BASE_DIR}")
    print(f"Model path: {MODEL_PATH}")
    print(f"LSTM units: {LSTM_UNITS}")
    print(f"Hidden units: {HIDDEN_UNITS}")
    print(f"Dropout rates: {DROPOUT_RATE_1}, {DROPOUT_RATE_2}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"L2 regularization: {L2_REGULARIZATION}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
