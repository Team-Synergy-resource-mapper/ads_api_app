import logging
from .xlmr_embeddings import generate_sentence_embeddings
from collections import defaultdict
import numpy as np

class AdClassifier:
  def __init__(self, model_manager):
    self.model_manager = model_manager
    self.main_predictor = model_manager.get_model("Main")

  def classify(self, ads):
      logging.info("Generating embeddings...")
      embeddings = generate_sentence_embeddings(ads)
      print(embeddings.shape)

      logging.info("Predicting main categories...")
      main_category_predictions = self.main_predictor.predict(embeddings)

      logging.info("Grouping ads by main category...")
      category_idx_list_dict = self._group_by_main_category(main_category_predictions)

      logging.info("Processing subcategories sequentially...")
      sub_category_predictions = []
      for main_category, idx_list in category_idx_list_dict.items():
          sub_category_predictions.extend(self._process_subcategory(main_category, idx_list, embeddings))

      logging.info("Combining results...")
      sub_category_predictions_sorted = [prediction for (_, prediction) in sorted(sub_category_predictions, key= lambda x : x[0] )]

      results = [
          (main_category, sub_category)
          for main_category, sub_category in zip(main_category_predictions, sub_category_predictions_sorted)
      ]
      return results

  def _group_by_main_category(self, main_category_predictions):
      grouped_data = defaultdict(list)
      for i, prediction in enumerate(main_category_predictions):
        grouped_data[prediction].append(i)
      return grouped_data


  def _process_subcategory(self, main_category, idx_list, embeddings):
      predictor = self.model_manager.get_model(main_category)
      selected_embeddings = np.array([embeddings[i] for i in idx_list])
      predictions = predictor.predict(selected_embeddings)
      return [(idx, prediction) for idx, prediction in zip(idx_list, predictions)]
