import os
from joblib import load
from app.models.model_manager import ModelManager
from app.core.ad_classifier import AdClassifier
from app.core.predictor import CategoryPredictor

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
model_dir = os.path.join(project_root, 'ml_models')

categories = {"Vehicle", "Ad_type", "Electronics", "Property", "Main"}

label_to_category = {
    "Main": {
        0: 'electronics',
        1: 'property',
        2: 'vehicle'
    },
    "Vehicle": {
        0: 'bicycle',
        1: 'bike',
        2: 'car',
        3: 'lorry_truck',
        4: 'three-wheeler',
        5: 'van'
    },
    # "Electronics": {
    #     0: 'air conditions & electrical fittings',
    #     1: 'audio & mp3',
    #     2: 'cameras & camcorders',
    #     3: 'computer accessories',
    #     4: 'computers',
    #     5: 'electronic home appliances',
    #     6: 'mobile phone accessories',
    #     7: 'mobile phones & tablets',
    #     8: 'tvs',
    #     9 : "other electronics"
    # },
    "Electronics": {
        0: 'air conditions & electrical fittings',
        1: 'audio & mp3',
        2: 'cameras & camcorders',
        3: 'computer accessories',
        4: 'computers',
        5: 'electronic home appliances',
        6: 'mobile phone accessories',
        7: 'mobile phones & tablets',
        8: 'tvs',
       
    },
    "Property": {
        0: 'apartment',
        1: 'commercial property',
        2: 'house',
        3: 'land',
        4: 'room & annex'
    },
    "Ad_type" : {
        0: "wanted",
        1: "offering"
    }
}
main_category_classifier = load(
    os.path.join(model_dir, 'main/main_cat_lr_classifier.pkl'))
ad_type_classifier = load(
    os.path.join(model_dir, 'ad_type/adtype_nn_classifier.pkl'))
# electronic_category_classifier = load(
#     os.path.join(model_dir, 'electronics/electronic_cat2_svmrbf_classifier.pkl'))
electronic_category_classifier = load(
    os.path.join(model_dir, 'electronics/electronic_svm_rbf_pipeline.pkl'))
property_category_classifier = load(
    os.path.join(model_dir, 'property/property_cat2_nn_classifier.pkl'))
vehicle_category_classifier = load(
    os.path.join(model_dir, 'vehicles/vehicle_cat2_nn_classifier.pkl'))

manager = ModelManager(categories= categories, label_to_category=label_to_category)

main_category_predictor = CategoryPredictor(
  model= main_category_classifier, 
  label_to_category= label_to_category['Main'])
adtype_category_predictor = CategoryPredictor(
  model= ad_type_classifier, 
  label_to_category= label_to_category['Ad_type'])
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
manager.set_model("Ad_type", adtype_category_predictor)
manager.set_model("Vehicle", vehicle_category_predictor)
manager.set_model("Electronics",electronic_category_predictor)
manager.set_model("Property", property_category_predictor)

ad_classifier = AdClassifier(manager)

print("setup completed..")



