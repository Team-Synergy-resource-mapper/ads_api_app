import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model


# class NNElectronicsClassifier:
#     def __init__(self, input_dim, num_classes):
#         self.model = self.create_mlp_model(input_dim, num_classes)

#     def fit(self, X, y):
#         self.model.fit(X, y, epochs=10, batch_size=32, verbose=1)
#         return self

#     def predict(self, X):
#         probs = self.model.predict(X)
#         return np.argmax(probs, axis=1)

#     def save(self, file_path):
#         self.model.save(file_path)
#         print(f"Model saved to {file_path}")

#     def load(self, file_path):
#         self.model = load_model(file_path)
#         print(f"Model loaded from {file_path}")

#     # Create the neural network (MLP) model
#     def create_mlp_model(self, input_dim, num_classes):
#         model = models.Sequential([
#             layers.InputLayer(shape=(input_dim,)),

#             # layers.Dense(512, activation='relu'),
#             # layers.BatchNormalization(),
#             # layers.Dropout(0.3),

#             # layers.Dense(256, activation='relu'),
#             # layers.BatchNormalization(),
#             # layers.Dropout(0.3),

#             layers.Dense(128, activation='relu'),
#             layers.BatchNormalization(),
#             layers.Dropout(0.3),

#             layers.Dense(64, activation='relu'),
#             layers.BatchNormalization(),
#             layers.Dropout(0.3),

#             layers.Dense(num_classes, activation='softmax')
#         ])

#         model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#         return model

class NNElectronicsClassifier:
    def __init__(self, input_dim, num_classes):
        self.model = self.create_mlp_model(input_dim, num_classes)

    def fit(self, X, y):
        self.model.fit(X, y, epochs=100, batch_size=32, verbose=1)
        return self

    def predict(self, X):
        probs = self.model.predict(X)
        return np.argmax(probs, axis=1)

    def save(self, file_path):
        self.model.save(file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path)
        print(f"Model loaded from {file_path}")

    # Create the neural network (MLP) model
    def create_mlp_model(self, input_dim, num_classes):
        model = models.Sequential([
            layers.InputLayer(shape=(input_dim,)),

            # layers.Dense(512, activation='relu'),
            # layers.BatchNormalization(),
            # layers.Dropout(0.3),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
