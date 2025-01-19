import os
from joblib import load
from ..ml.model_manager import ModelManager
from ..core.ad_classifier import AdClassifier
from ..predictor import CategoryPredictor
from ..nn_models_classes.nn_electronics_classifier import NNElectronicsClassifier


# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
model_dir = os.path.join(project_root, 'ml_models')



categories = {"Vehicle", "Electronics", "Property", "Main"}

label_to_category = {
    "Main": {
        0: 'Electronics',
        1: 'Property',
        2: 'Vehicle'
    },
    "Vehicle": {
        0: 'Bicycle',
        1: 'Bike',
        2: 'Car',
        3: 'Lorry_truck',
        4: 'Three-wheeler',
        5: 'Van'
    },
    "Electronics": {
        0: 'Air Conditions & Electrical fittings',
        1: 'Audio & MP3',
        2: 'Cameras & Camcorders',
        3: 'Computer Accessories',
        4: 'Computers',
        5: 'Electronic Home Appliances',
        6: 'Mobile Phone Accessories',
        7: 'Mobile Phones & Tablets',
        8: 'TVs'
    },
    "Property": {
        0: 'Apartment',
        1: 'Commercial property',
        2: 'House',
        3: 'Land',
        4: 'Room & Annex'
    }
}
vehicle_category_classifier_svm = load(
    os.path.join(model_dir, 'vehicles/vehicle_cat_svm_classifier.pkl'))
main_category_classifier_svm = load(
    os.path.join(model_dir, 'main/main_cat_svm_classifier.pkl'))
property_category_classifier_lr = load(
    os.path.join(model_dir, 'property/property_cat_lr_classifier.pkl'))
electronic_category_classifier_lr = load(
    os.path.join(model_dir, 'electronics/electronics_cat_lr_classifier.pkl'))

electronic_category_classifier_nn = NNElectronicsClassifier(input_dim=768, num_classes=len(label_to_category['Electronics']))
electronic_category_classifier_nn.load(
    os.path.join(model_dir, 'electronics/electronics_cat_nn_classifier_new_2.keras'))

manager = ModelManager(categories= categories, label_to_category=label_to_category)

main_category_predictor = CategoryPredictor(
  model= main_category_classifier_svm, 
  label_to_category= label_to_category['Main'])

property_category_predictor = CategoryPredictor(
  model=property_category_classifier_lr, 
  label_to_category=label_to_category['Property']
)

electronic_category_predictor = CategoryPredictor(
  model=electronic_category_classifier_nn, 
  label_to_category=label_to_category['Electronics']
)
vehicle_category_predictor = CategoryPredictor(
  model=vehicle_category_classifier_svm, 
  label_to_category=label_to_category['Vehicle']
) 


manager.set_model("Main", main_category_predictor)
manager.set_model("Vehicle", vehicle_category_predictor)
manager.set_model("Electronics",electronic_category_predictor)
manager.set_model("Property", property_category_predictor)

ad_classifier = AdClassifier(manager)
print("setup completed..")



