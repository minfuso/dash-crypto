import sys
import os
import yaml

from Informer.data.windowing import build_sequences_from_files
from Informer.data.splitter import split_sequences
from Informer.data.saver import save_splits
from Informer.utils.logging import setup_logger

def data_preparation(config_path="configs/data_config.yaml",):
    # Charger config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)["data"]

    # Logger
    logger = setup_logger(config["logging"]["log_dir"], config["logging"]["log_name"])

    # Construire les séquences
    X, Y = build_sequences_from_files(
        files=config["files"],
        feature_config=config["features"],
        target=config["target"],
        sequence_length=config["sequence_length"]
    )
    logger.info(f"Built sequences: X={X.shape}, Y={Y.shape}")

    # Split train/val/test
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_sequences(
        X, Y,
        train_size=config["split"]["train"],
        val_size=config["split"]["val"],
        test_size=config["split"]["test"]
    )
    logger.info(f"Splits -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


    # Sauvegarde
    save_splits(X_train, y_train, X_val, y_val, X_test, y_test,
                out_dir="outputs/splits", name=config["logging"]["log_name"])

