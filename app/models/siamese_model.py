import tensorflow as tf
from tensorflow import keras

# L2 normalization function


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

    def build(self, input_shape):
        self.attention = tf.keras.layers.Dense(
            input_shape[-1], activation="gelu")  # Feature-wise attention scores
        self.dense = tf.keras.layers.Dense(
            self.units, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dropout = tf.keras.layers.Dropout(
            self.dropout_rate)  # Dropout layer
        super().build(input_shape)

    def call(self, inputs, training=None):
        scores = self.attention(inputs)  # (batch, features)
        scores = self.dropout(scores, training=training)  # Apply dropout
        attention_weights = tf.nn.softmax(
            scores, axis=-1)  # Normalize across features
        context_vector = attention_weights * inputs  # Weighted feature representation
        output = self.dense(context_vector)  # Project back
        output = self.dropout(output, training=training)  # Apply dropout
        return output

    def get_config(self):
        config = super(SimpleAttentionFusion, self).get_config()
        config.update({"units": self.units, "dropout_rate": self.dropout_rate})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def SiameseWithCategories(d_feature=768, dropout_rate=0.4, num_cat1=3, num_cat2=20, cat_embedding_dim_1=3, cat_embedding_dim_2=6):
    # Create embedding layers for categories
    cat1_embedding = tf.keras.layers.Embedding(
        num_cat1, cat_embedding_dim_1, name="cat1_embedding", embeddings_regularizer=tf.keras.regularizers.l2(1e-4))
    cat2_embedding = tf.keras.layers.Embedding(
        num_cat2, cat_embedding_dim_2, name="cat2_embedding", embeddings_regularizer=tf.keras.regularizers.l2(1e-4))

    # Define inputs
    input1_text = tf.keras.layers.Input(
        shape=(d_feature,), dtype=tf.float32, name="input_1_text")
    input2_text = tf.keras.layers.Input(
        shape=(d_feature,), dtype=tf.float32, name="input_2_text")

    input1_cat1 = tf.keras.layers.Input(
        shape=(1,), dtype=tf.int32, name="input_1_cat1")
    input1_cat2 = tf.keras.layers.Input(
        shape=(1,), dtype=tf.int32, name="input_1_cat2")

    input2_cat1 = tf.keras.layers.Input(
        shape=(1,), dtype=tf.int32, name="input_2_cat1")
    input2_cat2 = tf.keras.layers.Input(
        shape=(1,), dtype=tf.int32, name="input_2_cat2")

    # Process categories
    cat1_vec1 = tf.keras.layers.Flatten()(cat1_embedding(input1_cat1))
    cat2_vec1 = tf.keras.layers.Flatten()(cat2_embedding(input1_cat2))

    cat1_vec2 = tf.keras.layers.Flatten()(cat1_embedding(input2_cat1))
    cat2_vec2 = tf.keras.layers.Flatten()(cat2_embedding(input2_cat2))

    # Combine text with categories before processing
    combined_input1 = tf.keras.layers.Concatenate()(
        [input1_text, cat1_vec1, cat2_vec1])
    combined_input2 = tf.keras.layers.Concatenate()(
        [input2_text, cat1_vec2, cat2_vec2])

    # Define shared feature extraction branch
    def create_branch(input_layer):
        # Dense Block 1
        x = tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(
            1e-4))(input_layer)  # No activation here
        x = tf.keras.layers.BatchNormalization()(x)  # BatchNorm before activation
        x = tf.keras.layers.Activation("gelu")(x)  # Apply GELU after BatchNorm
        x = tf.keras.layers.Dropout(dropout_rate)(x)

        # Dense Block 2
        x = tf.keras.layers.Dense(384, kernel_regularizer=tf.keras.regularizers.l2(
            1e-4))(x)  # No activation here
        x = tf.keras.layers.BatchNormalization()(x)  # BatchNorm before activation
        x = tf.keras.layers.Activation("gelu")(x)  # Apply GELU after BatchNorm
        x = tf.keras.layers.Dropout(dropout_rate * 0.3)(x)

        # Attention Fusion Layer
        x = SimpleAttentionFusion(256)(x)
        # x = SimpleAttentionFusion(300)(x)

        # L2 Normalization
        output = tf.keras.layers.Lambda(l2_normalize_fn)(x)
        return output

    # Create siamese branch as a shared model - FIXED VERSION
    shared_input = tf.keras.layers.Input(
        shape=(d_feature + cat_embedding_dim_1 + cat_embedding_dim_2,))
    shared_output = create_branch(shared_input)
    siamese_branch = tf.keras.Model(
        inputs=shared_input,
        outputs=shared_output,
        name="siamese_branch"
    )

    # Apply shared model to both inputs
    branch1 = siamese_branch(combined_input1)
    branch2 = siamese_branch(combined_input2)

    # Concatenate embeddings (maintaining same output structure)
    concat = tf.keras.layers.Concatenate(
        name="concatenated")([branch1, branch2])

    return tf.keras.Model(
        inputs=[input1_text, input2_text, input1_cat1,
                input1_cat2, input2_cat1, input2_cat2],
        outputs=concat,
        name="siamese_model_with_categories"
    )
# For loading the model


def load_siamese_model(model_path):
    """Load the saved Siamese model with custom objects"""
    loaded_model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'l2_normalize_fn': l2_normalize_fn,
            'SimpleAttentionFusion': SimpleAttentionFusion
        },
        compile=False
    )
    return loaded_model




