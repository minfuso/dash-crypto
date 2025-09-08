import tensorflow as tf
from tensorflow.keras import layers

class ProbSparseAttention(layers.Layer):
    """
    Version pédagogique de ProbSparse Self-Attention.
    Sélectionne seulement les top-u queries les plus informatives.
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1, u=15, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model doit être divisible par num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        self.u = u  # nombre de queries gardées

        # Projections linéaires
        self.W_q = layers.Dense(d_model, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.W_k = layers.Dense(d_model, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.W_v = layers.Dense(d_model, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.W_o = layers.Dense(d_model, kernel_regularizer=tf.keras.regularizers.l2(1e-4))

        self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        """
        x: (batch, T, d_model)
        """
        # 0. Récupération de la taille du batch B et de la séquence T
        B, T, _ = tf.unstack(tf.shape(x))

        # 1. Projeter en Q, K, V
        Q = self.W_q(x)  # (B, T, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Découper en têtes
        def split_heads(tensor):
            return tf.reshape(tensor, (B, T, self.num_heads, self.d_k))  # (B, T, h, d_k)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # 3. Aplatir batch*têtes
        Q_ = tf.reshape(Q, (B*self.num_heads, T, self.d_k)) # (B*h, T, d_k)
        K_ = tf.reshape(K, (B*self.num_heads, T, self.d_k))
        V_ = tf.reshape(V, (B*self.num_heads, T, self.d_k))

        # 4. Calcul score de sparsité (max similitude)
        norm_q = tf.nn.l2_normalize(Q_, axis=-1)
        norm_k = tf.nn.l2_normalize(K_, axis=-1)
        scores = tf.matmul(norm_q, norm_k, transpose_b=True)  # (B*h, T, T)
        max_scores = tf.reduce_max(scores, axis=-1)  # (B*h, T)

        # 5. Top-u queries
        u = tf.minimum(self.u, T)
        top_idx = tf.argsort(max_scores, axis=-1, direction="DESCENDING")[:, :u]  # (B*h, u)

        # 6. Extraire les queries sélectionnées
        Q_sel = tf.gather(Q_, top_idx, batch_dims=1)  # (B*h, u, d_k)

        # 7. Attention seulement sur Q_sel vs tous K,V
        attn_weights = tf.matmul(Q_sel, K_, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_weights = self.drop(attn_weights, training=training)

        out = tf.matmul(attn_weights, V_)  # (B*h, u, d_k)

        # 8. Reconstruire la séquence complète (remplir les queries non sélectionnées par 0)
        out_full = tf.zeros_like(Q_)  # (B*h, T, d_k)

        # 9. indices pour scatter update
        batch_idx = tf.repeat(tf.range(B*self.num_heads)[:, None], u, axis=1)  # (B*h, u)
        idx = tf.stack([batch_idx, top_idx], axis=-1)  # (B*h, u, 2)

        out_full = tf.tensor_scatter_nd_update(out_full, tf.reshape(idx, (-1, 2)), tf.reshape(out, (-1, self.d_k)))

        # 10. Reconstruire (B, T, d_model)
        out_full = tf.reshape(out_full, (B, T, self.d_model))
        return self.W_o(out_full)
