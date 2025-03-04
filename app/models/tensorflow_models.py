import tensorflow as tf
from tensorflow import keras

def l2_normalize_fn(x):
    """L2 normalization for unit-length embeddings"""
    return tf.math.l2_normalize(x, axis=1)


@tf.keras.utils.register_keras_serializable()
class SimpleAttentionFusion(tf.keras.layers.Layer):
    """Simple attention mechanism to fuse important features"""

    def __init__(self, units=256, dropout_rate=0.1, **kwargs):
        super(SimpleAttentionFusion, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.attention = None
        self.dense = None
        self.dropout = None

    def build(self, input_shape):
        self.attention = tf.keras.layers.Dense(
            1, activation="gelu")  # Compute attention scores
        self.dense = tf.keras.layers.Dense(
            self.units, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dropout = tf.keras.layers.Dropout(
            self.dropout_rate)  # Dropout layer
        super().build(input_shape)

    def call(self, inputs, training=None):
        expanded_inputs = tf.expand_dims(
            inputs, axis=1)  # (batch, 1, features)
        scores = self.attention(expanded_inputs)  # (batch, 1, 1)
        # Optional: Apply dropout to attention scores
        scores = self.dropout(scores, training=training)
        attention_weights = tf.nn.softmax(scores, axis=1)  # Normalize
        context_vector = attention_weights * expanded_inputs  # Weighted features
        output = self.dense(tf.squeeze(context_vector, axis=1))  # Project back
        # Apply dropout to the output of the dense layer
        output = self.dropout(output, training=training)
        return output

    def get_config(self):
        config = super(SimpleAttentionFusion, self).get_config()
        config.update({"units": self.units, "dropout_rate": self.dropout_rate})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def load_siamese_branch(model_path):
    """
    Load the saved siamese branch model with custom objects
    Args:
        model_path: Path to the saved branch model
    Returns:
        The loaded siamese branch model
    """
    loaded_branch = keras.models.load_model(
        model_path,
        custom_objects={
            'l2_normalize_fn': l2_normalize_fn,
            'SimpleAttentionFusion': SimpleAttentionFusion
        },
        compile=False
    )
    return loaded_branch

