import numpy as np
import tensorflow as tf
from tensorflow import keras

class NNElectronicsClassifier:
    """
    Neural Network classifier for electronics categories
    """
    
    def __init__(self, input_dim=768, num_classes=10, hidden_layers=None):
        """
        Initialize the neural network classifier
        
        Args:
            input_dim (int): Dimension of the input features (default: 768 for embeddings)
            num_classes (int): Number of output classes
            hidden_layers (list, optional): List of hidden layer sizes. Default is [256, 128]
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers or [256, 128]
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Build the neural network architecture"""
        model = keras.Sequential()
        
        # Input layer
        model.add(keras.layers.Dense(self.hidden_layers[0], 
                                    input_dim=self.input_dim, 
                                    activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(keras.layers.Dense(units, activation='relu'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(0.3))
        
        # Output layer
        model.add(keras.layers.Dense(self.num_classes, activation='softmax'))
        
        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        self.model = model
    
    def fit(self, X, y, validation_data=None, epochs=10, batch_size=32):
        """
        Train the model
        
        Args:
            X: Input features
            y: Target labels (one-hot encoded)
            validation_data: Tuple of (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            History object
        """
        return self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predicted class probabilities
        """
        return self.model.predict(X)
    
    def predict_classes(self, X):
        """
        Predict class labels
        
        Args:
            X: Input features
            
        Returns:
            Predicted class indices
        """
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)
    
    def evaluate(self, X, y):
        """
        Evaluate the model
        
        Args:
            X: Input features
            y: Target labels (one-hot encoded)
            
        Returns:
            Tuple of (loss, accuracy)
        """
        return self.model.evaluate(X, y)
    
    def save(self, filepath):
        """
        Save the model
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
    
    def load(self, filepath):
        """
        Load the model
        
        Args:
            filepath: Path to the model file
        """
        self.model = keras.models.load_model(filepath)