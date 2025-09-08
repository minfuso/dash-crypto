import yaml
import tensorflow as tf

from Informer.data.saver import load_splits
from Informer.utils.logging import setup_logger
from Informer.model.informer import Informer
from Informer.stats.metrics import rmse  
from Informer.loss.gmadl import GMADLBinary, GMADL

import numpy as np

tf.keras.backend.clear_session()

config = "training.yaml"

with open(config, "r") as f:
    cfg = yaml.safe_load(f)
    
logger = setup_logger(cfg["logging"]["log_dir"], cfg["logging"]["log_name"])

#  1) Load sets from npz files
npz_path = cfg["data"]["file"]
logger.info(f"Loading data from {npz_path}")
((X_train, y_train), (X_val, y_val), (X_test, y_test)) = load_splits(npz_path)
logger.info(f"Train shapes: X_train={X_train.shape}, y_train={y_train.shape}")
logger.info(f"Val shapes:   X_val={X_val.shape}, y_val={y_val.shape}")
logger.info(f"Test shapes:  X_test={X_test.shape}, y_test={y_test.shape}")

# 2) Create the model

model_data = cfg["model"]

logger.info(f"Model config: {model_data}")

model = Informer(
    d_model=model_data["d_model"],
    num_heads=model_data["num_heads"],
    d_ff=model_data["d_ff"],
    num_layers=model_data["num_layers"],
    horizons=model_data["horizons"],
    dropout=model_data["dropout"],
    u=model_data["u"],
    activation_last_layer="sigmoid"
)

# 2.1) Learning rate evolution
def scheduler(epoch):
    initial_lr = 1e-2
    final_lr = 1e-4
    epochs = 300
    decay = (final_lr / initial_lr) ** (1 / epochs)
    return initial_lr * (decay ** epoch)

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Callback pour logger le LR dans le CSV
class LrLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = self.model.optimizer.learning_rate
        # Si le LR est un schedule, il faut l'évaluer
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(epoch)
        else:
            lr = tf.keras.backend.get_value(lr)
        logs["lr"] = float(lr)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",       # on surveille val_loss
    mode="min",              # parce qu'on veut minimiser loss
    patience=20,             # arrête après 20 epochs sans amélioration
    restore_best_weights=False, # recharge automatiquement les meilleurs poids si true
    verbose=1
)

# 3) Modele compilation
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    # loss=tf.keras.losses.Huber(delta=1.0),
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
    # loss = GMADLBinary(alpha=1.0, beta=0.5),
    # loss = GMADL(a=1.0, b=1),
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

# 4) Model training
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath="informer_best.model.keras",
    monitor="val_auc",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Logger clair
csv_logger = tf.keras.callbacks.CSVLogger("training_log.csv", append=False)

# Entraînement
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=32,
    callbacks=[csv_logger, checkpoint_cb, early_stopping],
    verbose=1
)