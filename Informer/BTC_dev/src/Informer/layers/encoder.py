import tensorflow as tf
from tensorflow.keras import layers

from Informer.layers.attention.probsparse import ProbSparseAttention
from Informer.layers.distilling import DistillingLayer

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff=512, dropout=0.1, u=20):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, num_heads, dropout=dropout, u=u)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            layers.Dropout(dropout),
            layers.Dense(d_model, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        ])
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        # 1. Attention + résidu
        attn_out = self.attn(x, training=training)
        x = self.norm1(x + self.drop(attn_out, training=training))

        # 2. FeedForward + résidu
        ffn_out = self.ffn(x, training=training)
        x = self.norm2(x + self.drop(ffn_out, training=training))

        return x
    
    
class InformerEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff=512, num_layers=2, dropout=0.1, u=20):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            self.layers.append(EncoderLayer(d_model, num_heads, d_ff, dropout, u))
            # Ajouter une distillation sauf après la dernière couche
            if i < num_layers - 1:
                self.layers.append(DistillingLayer(d_model))

    def call(self, x, training=False):
        # x: (B, T, d_model)
        for layer in self.layers:
            x = layer(x, training=training)
        return x  # (B, T', d_model) avec T' réduit par distillation
