import yaml
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score

from Informer.model.informer import Informer
from Informer.data.saver import load_splits

# === 1) Charger la config ===
config = "../training/training.yaml"
with open(config, "r") as f:
    cfg = yaml.safe_load(f)

# === 2) Charger les données ===
npz_path = cfg["data"]["file"]
((X_train, y_train), (X_val, y_val), (X_test, y_test)) = load_splits(npz_path)

# === 3) Charger le modèle complet sauvegardé ===
# (fichier généré par ModelCheckpoint avec save_weights_only=False)
model = tf.keras.models.load_model("../training/informer_best.model.keras", custom_objects={"Informer": Informer})

# === 4) Évaluation sur le validation set ===
val_results = model.evaluate(X_val, y_val, verbose=1)
print("\nValidation results:", dict(zip(model.metrics_names, val_results)))

# === 5) Calcul de l’AUC avec sklearn (double check) ===
y_val_proba = model.predict(X_val).ravel()
val_auc = roc_auc_score(y_val, y_val_proba)
print("Validation AUC (sklearn):", val_auc)

# === 6) Évaluation sur le test set ===
test_results = model.evaluate(X_test, y_test, verbose=1)
print("\nTest results:", dict(zip(model.metrics_names, test_results)))
# === 7) Calcul de l’AUC avec sklearn (double check) ===
y_test_proba = model.predict(X_test).ravel()
test_auc = roc_auc_score(y_test, y_test_proba)
print("Test AUC (sklearn):", test_auc)

