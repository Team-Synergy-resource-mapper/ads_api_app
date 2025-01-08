import logging
from .core.embeddings import generate_sentence_embeddings
from .core.parallel_processor import ParallelProcessor
from collections import defaultdict
import numpy as np

class AdClassifier:
  def __init__(self, model_manager):
    self.model_manager = model_manager
    self.main_predictor = model_manager.get_model("Main")
    self.processor = ParallelProcessor(self._process_subcategory)

  def classify(self, ads):
      logging.info("Generating embeddings...")
      embeddings = generate_sentence_embeddings(ads)
      print(embeddings.shape)

      logging.info("Predicting main categories...")
      main_category_predictions = self.main_predictor.predict(embeddings)

      logging.info("Grouping ads by main category...")
      category_idx_list_dict = self._group_by_main_category(main_category_predictions)

      logging.info("Processing subcategories in parallel...")
      tasks = [
          (main_category, idx_list, embeddings)
          for main_category, idx_list in category_idx_list_dict.items()
      ]

      # make subcategory predictions
      sub_category_predictions = self.processor.execute(tasks)  # [(idx,subcategory)]


      logging.info("Combining results...")
      # [subcategory]
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
      # model = self.model_manager.get_model(main_category)
      # label_to_category = self.model_manager.get_label_to_category_dict(
      #     main_category)
      predictor = self.model_manager.get_model(main_category)
      # get the embeddings corresponding to the main category type
      # selected_embeddings = [embeddings[i] for i in idx_list]
      selected_embeddings = np.array([embeddings[i] for i in idx_list])
      predictions = predictor.predict(selected_embeddings)
      return [(idx, prediction) for idx, prediction in zip(idx_list, predictions)]
