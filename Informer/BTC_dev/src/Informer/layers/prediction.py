import tensorflow as tf

class PredictionHead(tf.keras.layers.Layer):
    def __init__(self, d_model, horizons, dropout=0.1):
        super().__init__()
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(horizons, activation="sigmoid")
        ])

    def call(self, x, training=False):
        # x: (B, T', d_model)
        x = self.pool(x)       # (B, d_model)
        x = self.ffn(x, training=training)  # (B, horizons)
        return x