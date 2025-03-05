from abc import ABC, abstractmethod
import numpy as np

class Predictor(ABC):
  @abstractmethod
  def predict(self, data):
    pass


class CategoryPredictor(Predictor):
  def __init__(self, model, label_to_category):
    self.model = model
    self.label_to_category = label_to_category

  def predict(self, embeddings):
    print(embeddings.shape)
    labels = self.model.predict(embeddings)
    
    # If output is a probability distribution (NN output), convert to class index
    if isinstance(labels, np.ndarray) and labels.ndim > 1:
        labels = np.argmax(labels, axis=1)  # Convert probabilities to class indices

    # Convert to list if necessary
    if hasattr(labels, 'tolist'):
        labels = labels.tolist()

    # Ensure we have a list
    if not isinstance(labels, list):
        labels = [labels]

    # Map labels to categories
    result = []
    for label in labels:
        if isinstance(label, (int, float, np.integer, np.float32, np.float64)):
            try:
                index = int(label)
                result.append(self.label_to_category[index])
            except (KeyError, ValueError):
                result.append(str(label))  # Fallback if mapping fails
        else:
            result.append(label)  # Handle string labels

    return result