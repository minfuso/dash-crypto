import tensorflow as tf

from Informer.layers.positional_encoding import SinusoidalPE
from Informer.layers.encoder import InformerEncoder
from Informer.layers.prediction import PredictionHead

class Informer(tf.keras.Model):
    def __init__(self, d_model, num_heads, d_ff, num_layers, horizons, dropout=0.1, u=20):
        super().__init__()
        self.input_proj = tf.keras.layers.Dense(d_model)  # projette tes features -> d_model
        self.pe = SinusoidalPE(d_model)  # positional encoding
        self.encoder = InformerEncoder(d_model, num_heads, d_ff, num_layers, dropout, u)
        self.head = PredictionHead(d_model, horizons, dropout)

    def call(self, x, training=None):
        # x: (B, T, d_model) déjà normalisé
        x = self.input_proj(x)  # projette les features -> d_model (B, T, d_model)
        x = self.pe(x)                       # ajoute l'encodage positionnel
        x = self.encoder(x, training=training)  # encodeur Informer
        x = self.head(x, training=training)     # prédicteur multi-horizon
        return x  # (B, horizons)
