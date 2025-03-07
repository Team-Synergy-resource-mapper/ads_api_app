
class ModelManager:
  def __init__(self, categories, label_to_category):
    self.categories = categories
    self.models = {}
    self.label_to_category = label_to_category

  def set_model(self, category, model):
    if category not in self.categories:
      raise ValueError(f"Invalid category: {category}. Allowed: {self.categories}.")
    self.models[category] = model

  def get_model(self, category):
    if category not in self.categories:
      raise ValueError(f"Invalid category: {category}. Allowed: {self.categories}.")
    return self.models[category]

  def get_label_to_category_dict(self, category):
    if category not in self.label_to_category:
      raise ValueError(f"Invalid category: {category}. Allowed: {self.label_to_category.keys()}")  
    return self.label_to_category[category]  
  





   