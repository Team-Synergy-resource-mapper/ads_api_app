import os
from joblib import load
from app.models.model_manager import ModelManager
from app.core.ad_classifier import AdClassifier
from app.predictor import CategoryPredictor

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
main_category_classifier = load(
    os.path.join(model_dir, 'main/main_cat_lr_classifier.pkl'))
electronic_category_classifier = load(
    os.path.join(model_dir, 'electronics/electronic_cat2_svmrbf_classifier.pkl'))
property_category_classifier = load(
    os.path.join(model_dir, 'property/property_cat2_nn_classifier.pkl'))
vehicle_category_classifier = load(
    os.path.join(model_dir, 'vehicles/vehicle_cat2_nn_classifier.pkl'))

manager = ModelManager(categories= categories, label_to_category=label_to_category)

main_category_predictor = CategoryPredictor(
  model= main_category_classifier, 
  label_to_category= label_to_category['Main'])

electronic_category_predictor = CategoryPredictor(
  model=electronic_category_classifier, 
  label_to_category=label_to_category['Electronics']
)
property_category_predictor = CategoryPredictor(
  model=property_category_classifier, 
  label_to_category=label_to_category['Property']
)
vehicle_category_predictor = CategoryPredictor(
  model=vehicle_category_classifier, 
  label_to_category=label_to_category['Vehicle']
) 

manager.set_model("Main", main_category_predictor)
manager.set_model("Vehicle", vehicle_category_predictor)
manager.set_model("Electronics",electronic_category_predictor)
manager.set_model("Property", property_category_predictor)

ad_classifier = AdClassifier(manager)

print("setup completed..")



