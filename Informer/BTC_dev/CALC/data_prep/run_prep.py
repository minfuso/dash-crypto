import yaml
import numpy as np
from Informer.data.splits import load_split_dfs
from Informer.data.normalization import fit_scalers_on_train, transform_with_scalers, save_scalers
from Informer.data.windowing import df_list_to_sequences
from Informer.data.saver import save_splits  # (ta fonction np.savez_compressed)
from Informer.utils.logging import setup_logger

config = "data_config.yaml"

with open(config, "r") as f:
    cfg = yaml.safe_load(f)

logger = setup_logger(cfg["logging"]["log_dir"], cfg["logging"]["log_name"])

feature_config = cfg["data"]["features_map"]
features = list(feature_config.keys())
target = cfg["data"]["target"]
seq_len = cfg["data"]["sequence_length"]

# 1) Charger les DF non normalisés, par split, **par fichier**
train_dfs_raw, val_dfs_raw, test_dfs_raw = load_split_dfs(cfg["data"]["files_by_split"], features, target)
logger.info(f"Loaded raw: train={len(train_dfs_raw)} files, val={len(val_dfs_raw)}, test={len(test_dfs_raw)}")

# 2) Fit des scalers sur le TRAIN uniquement (concat de ses fichiers)
scalers = fit_scalers_on_train(train_dfs_raw, feature_config)
save_scalers(scalers, cfg["outputs"]["scalers_pkl"])
logger.info(f"Fitted & saved scalers to {cfg['outputs']['scalers_pkl']}")

# 3) Transformer train/val/test AVEC ces scalers (toujours par fichier)
train_dfs = transform_with_scalers(train_dfs_raw, feature_config, scalers)
val_dfs   = transform_with_scalers(val_dfs_raw,   feature_config, scalers)
test_dfs  = transform_with_scalers(test_dfs_raw,  feature_config, scalers)

# 4) Fenêtrage en séquences par fichier (sans croiser), puis concat
X_train, y_train = df_list_to_sequences(train_dfs, features, target, seq_len)
X_val,   y_val   = df_list_to_sequences(val_dfs,   features, target, seq_len)
X_test,  y_test  = df_list_to_sequences(test_dfs,  features, target, seq_len)

logger.info(f"Shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
logger.info("Percentage of training set among all data: {0:.2f}%".format(100 * X_train.shape[0] / (X_train.shape[0] + X_val.shape[0] + X_test.shape[0])))
logger.info("Percentage of val set among all data: {0:.2f}%".format(100 * X_val.shape[0] / (X_train.shape[0] + X_val.shape[0] + X_test.shape[0])))
logger.info("Percentage of test set among all data: {0:.2f}%".format(100 * X_test.shape[0] / (X_train.shape[0] + X_val.shape[0] + X_test.shape[0])))

# 5) Sauvegarde binaire unique (réutilisable pour tous tes runs)
save_splits(X_train, y_train, X_val, y_val, X_test, y_test,
            out_dir="outputs/splits",
            name="btc_seq24")  # ou cfg["outputs"]["splits_npz"] si tu préfères passer un chemin complet
logger.info("Saved splits.")
