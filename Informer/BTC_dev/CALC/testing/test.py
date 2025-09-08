import yaml
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

from Informer.data.saver import load_splits
from Informer.model.informer import Informer

# === 1) Charger la config ===
config = "../training/training_set1.yaml"
with open(config, "r") as f:
    cfg = yaml.safe_load(f)

# === 2) Charger les données ===
npz_path = cfg["data"]["file"]
((X_train, y_train), (X_val, y_val), (X_test, y_test)) = load_splits(npz_path)

# === 3) Reconstruire le modèle ===
model_data = cfg["model"]
model = Informer(
    d_model=model_data["d_model"],
    num_heads=model_data["num_heads"],
    d_ff=model_data["d_ff"],
    num_layers=model_data["num_layers"],
    horizons=model_data["horizons"],
    dropout=model_data["dropout"],
    u=model_data["u"]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

# 3) Modele compilation
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    # loss=tf.keras.losses.Huber(delta=1.0),
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

# === 4) Charger les meilleurs poids sauvegardés ===
# Build le modèle avec la bonne shape avant de charger les poids
model.build(input_shape=(None, X_train.shape[1], X_train.shape[2]))

model.load_weights("../training/informer_best.weights_set1.h5")

# === 5) Évaluer sur le test set ===
results = model.evaluate(X_test, y_test, verbose=1)
print("\nTest results:", dict(zip(model.metrics_names, results)))

# === 6) Prédictions détaillées ===
y_pred_proba = model.predict(X_test).ravel()
y_pred = (y_pred_proba > 0.5).astype(int)

print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Test AUC:", roc_auc_score(y_test, y_pred_proba))
