import tensorflow as tf

class DistillingLayer(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.conv = tf.keras.layers.Conv1D(
            filters=d_model, kernel_size=3, padding="same", activation="relu"
        )
        self.pool = tf.keras.layers.MaxPooling1D(pool_size=2)

    def call(self, x):
        # x: (B, T, d_model)
        x = self.conv(x)   # (B, T, d_model)
        x = self.pool(x)   # (B, T/2, d_model)
        return x
