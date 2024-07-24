"""Multi-Head Self-Attention layer for LSTM fusion architecture."""
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


class MultiHeadSelfAttention(Layer):
    """Multi-head self-attention mechanism for temporal feature fusion.

    Implements scaled dot-product attention with multiple heads,
    allowing the model to jointly attend to information from different
    representation subspaces at different positions.

    Args:
        embed_size: Total embedding dimension.
        num_heads: Number of attention heads.
    """

    def __init__(self, embed_size: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        if embed_size % num_heads != 0:
            raise ValueError(
                f"embed_size ({embed_size}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.projection_dim = embed_size // num_heads

        self.query_dense = Dense(embed_size)
        self.key_dense = Dense(embed_size)
        self.value_dense = Dense(embed_size)
        self.combine_heads = Dense(embed_size)

    def _split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Split the last dimension into (num_heads, projection_dim)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]

        query = self._split_heads(self.query_dense(inputs), batch_size)
        key = self._split_heads(self.key_dense(inputs), batch_size)
        value = self._split_heads(self.value_dense(inputs), batch_size)

        # Scaled dot-product attention
        scale = tf.cast(self.projection_dim, tf.float32)
        scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(scale)
        weights = tf.nn.softmax(scores, axis=-1)
        attended = tf.matmul(weights, value)

        # Recombine heads
        attended = tf.transpose(attended, perm=[0, 2, 1, 3])
        concatenated = tf.reshape(attended, (batch_size, -1, self.embed_size))
        return self.combine_heads(concatenated)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "embed_size": self.embed_size,
            "num_heads": self.num_heads,
        })
        return config
