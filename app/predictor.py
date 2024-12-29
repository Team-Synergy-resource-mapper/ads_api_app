from abc import ABC, abstractmethod

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
    return [self.label_to_category[label] for label in labels]
