# from joblib import load
import os 
from core.embeddings import generate_sentence_embeddings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from app.config.setup_models import manager
# from .nn_models.nn_classifier import NN_Classifier




# vehicle_category_classifier = load("./app/ml_models/vehicles/vehicle_cat_svm_classifier.pkl")
# main_category_clsssifier = load("./app/ml_models/main/main_cat_svm_classifier.pkl")
# property_category_clsssifier = load("./app/ml_models/property/property_cat_lr_classifier.pkl")
# # electronic_category_clsssifier = load("./app/ml_models/electronics/electronics_cat_lr_classifier.pkl")
# electronic_category_clsssifier = NN_Classifier(768, 10)
# electronic_category_clsssifier.load("./app/ml_models/electronics/electronics_cat_nn_classifier.keras")

# label_to_category_main_cat = {0: 'Electronics', 2: 'Vehicle', 1: 'Property'}

# label_to_category_electronics_cat = {0: 'Air Conditions & Electrical fittings',
#                               1: 'Audio & MP3',
#                               2: 'Cameras & Camcorders',
#                               5: 'Electronic Home Appliances',
#                               7: 'Mobile Phones & Tablets',
#                               4: 'Computers',
#                               3: 'Computer Accessories',
#                               6: 'Mobile Phone Accessories',
#                               8: 'Other Electronics',
#                               9: 'TVs'}

# label_to_category_property_cat = {3: 'Land',
#                                   0: 'Apartment',
#                                   2: 'House',
#                                   1: 'Commercial property',
#                                   4: 'Room & Annex'}

# label_to_category_vehicle_cat = {2: 'Car',
#                                  5: 'Van',
#                                  4: 'Three-wheeler',
#                                  1: 'Bike',
#                                  3: 'Lorry_truck',
#                                  0: 'Bicycle'}


# label_to_category = {
#    "Vehicle" : label_to_category_vehicle_cat,
#    "Electronics" : label_to_category_electronics_cat,
#    "Property" : label_to_category_property_cat,
# }

# sub_category_models = {
#    "Vehicle": vehicle_category_classifier,
#    "Electronics": electronic_category_clsssifier,
#    "Property": property_category_clsssifier,
# }

def classify_ads(ads: list[str]):
  
  # genarate embeddings for each sentence
  embeddings = generate_sentence_embeddings(ads)
  # predict main categories
  main_category_predictions = predict_main_category(embeddings)
  print(main_category_predictions)
  # group sentence idx by main categories 
  category_idx_list_dict = group_by_main_category(main_category_predictions)
  print(category_idx_list_dict)
  results = []
  with ProcessPoolExecutor() as executor:
        futures = []
        for main_category, idx_list in category_idx_list_dict.items():
            # Submit each main category for processing
            futures.append(executor.submit(
              process_sub_category, main_category, idx_list, embeddings))

        # Step 3: Gather results
        for future in as_completed(futures):
            results.extend(future.result())
  print(results)
  sorted_results = sorted(results, key=lambda x: x[0])
  print(sorted_results)
  sub_category_predictions = [ prediction for (idx, prediction) in sorted(results, key= lambda x: x[0])]
  results = list(zip(main_category_predictions, sub_category_predictions))
  print(results)
  return results




def predict_main_category(embeddings, model):
   
    main_category_predictions_labels = model.predict(embeddings)
    main_category_predictions_cat_names = [label_to_category_main_cat[i] for i in main_category_predictions_labels]
    # [ "Vehicle", "Vehicle, "Property", "Vehicle" , "Elctronics"]
    return main_category_predictions_cat_names


def group_by_main_category(main_category_predictions):
    grouped_data = defaultdict(list)
    for i, prediction in enumerate(main_category_predictions):
      grouped_data[prediction].append(i)
    return grouped_data  # { "Vehicle" :[1 , 4]}


def predict_sub_category(model, label_to_cat_dic, idx_list, embeddings):
    selected_embeddings = [embeddings[i] for i in idx_list]
    predictions_labels = model.predict(selected_embeddings)
    predictions_cat_names = [label_to_cat_dic[i] for i in predictions_labels]
    return [(idx, predictions_cat_names[i]) for i, idx in enumerate(idx_list)]


def process_sub_category(main_category, idx_list, embeddings):
       
            return predict_sub_category(
                sub_category_models[main_category],
                label_to_cat_dic = label_to_category[main_category],
                idx_list= idx_list,
                embeddings= embeddings
            )
       

# class ClassifiationModelManager():
#   def __init__(self, main_cat_classifier, 
#               vehicle_cat_classifier,
#               electronis_cat_classifier,
#               property_cat_classifier):
#     self.main_cat_classifier = main_cat_classifier
#     self.vehicle_cat_classifier = vehicle_cat_classifier
#     self.electronis_cat_classifier = electronis_cat_classifier
#     self.property_cat_classifier = property_cat_classifier

#   label_to_category_main_cat = {0: 'Electronics', 2: 'Vehicle', 1: 'Property'}

#   def predict_main_category(self, embeddings):
#     main_category_predictions_labels = self.main_cat_classifier.predict(embeddings)
#     main_category_predictions_cat_names = [ self.label_to_category_main_cat[i] for i in main_category_predictions_labels]
#     return main_category_predictions_cat_names # [ "Vehicle", "Vehicle, "Property", "Vehicle" , "Elctronics"]

#   def group_by_main_category(self, main_category_predictions):
#     grouped_data = defaultdict(list)
#     for i, prediction in enumerate(main_category_predictions):
#       grouped_data[prediction].append(i)
#     return grouped_data #  { "Vehicle" :[1 , 4]}


#   def predict_sub_category(self, model,label_to_cat_dic, idx_list, embeddings):
#     selected_embeddings = [embeddings[i] for i in idx_list]
#     predictions_labels = model.predict(selected_embeddings)
#     predictions_cat_names = [ label_to_cat_dic[i] for i in predictions_labels]
#     return [(idx, predictions_cat_names(i)) for i , idx in enumerate(idx_list)]
  
#   def classify_ads(self, embeddings):
#     main_category_predictions = self.predict_main_category(embeddings)
#     grouped_data = self.group_by_main_category(main_category_predictions)

#     def process_sub_category(main_category, idx_list):
#         if main_category == 'Vehicle':
#             return main_category, self.predict_sub_category(
#                 self.vehicle_cat_classifier,
#                 self.label_to_vehicle_cat,
#                 idx_list,
#                 embeddings
#             )
#         elif main_category == 'Electronics':
#             return main_category, self.predict_sub_category(
#                 self.electronis_cat_classifier,
#                 self.label_to_electronis_cat,
#                 idx_list,
#                 embeddings
#             )
#         elif main_category == 'Property':
#             return main_category, self.predict_sub_category(
#                 self.property_cat_classifier,
#                 self.label_to_property_cat,
#                 idx_list,
#                 embeddings
#             )
    
#     results = []
#     with ProcessPoolExecutor() as executor:
#         futures = []
#         for main_category, idx_list in grouped_data.items():
#             # Submit each main category for processing
#             futures.append(executor.submit(process_sub_category, main_category, idx_list))

#         # Step 3: Gather results
#         for future in as_completed(futures):
#             results.extend(future.result())

#     return results
    
    




  