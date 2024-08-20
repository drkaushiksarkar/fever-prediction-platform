"""Tests for Multi-Head Self-Attention layer."""
import numpy as np
import pytest
import tensorflow as tf

from fever_platform.models.attention import MultiHeadSelfAttention


class TestMultiHeadSelfAttention:
    def test_output_shape(self):
        layer = MultiHeadSelfAttention(embed_size=64, num_heads=4)
        x = tf.random.normal((2, 10, 64))
        out = layer(x)
        assert out.shape == (2, 10, 64)

    def test_single_head(self):
        layer = MultiHeadSelfAttention(embed_size=32, num_heads=1)
        x = tf.random.normal((1, 5, 32))
        out = layer(x)
        assert out.shape == (1, 5, 32)

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadSelfAttention(embed_size=63, num_heads=4)

    def test_serialization(self):
        layer = MultiHeadSelfAttention(embed_size=64, num_heads=4)
        config = layer.get_config()
        assert config["embed_size"] == 64
        assert config["num_heads"] == 4
