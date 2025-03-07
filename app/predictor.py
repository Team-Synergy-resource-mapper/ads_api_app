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
    
    if isinstance(labels, np.ndarray):
        if labels.ndim == 2 and labels.shape[1] == 1:  
            # Case 1: Binary classification (single probability per sample)
            labels = labels.flatten()  # Convert (n_samples, 1) to (n_samples,)
            labels = (labels >= 0.5).astype(int)  # Convert probability to class (0 or 1)
        elif labels.ndim == 2 and labels.shape[1] > 1:  
            # Case 2: Multi-class classification (multiple probabilities per sample)
            labels = np.argmax(labels, axis=1)  # Convert probability distribution to class index
        elif labels.ndim == 1:  
            # Case 3: Already a 1D array (potential direct outputs)
            labels = (labels >= 0.5).astype(int) if labels.dtype == np.float32 else labels

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