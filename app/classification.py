from .embeddings import generate_sentence_embeddings
from joblib import load
import os 

vehicle_category_scaler = load("./app/ml_models/vehicles/vehicle_category_scaler.joblib")

vehicle_category_classifier = load(
    "./app/ml_models/vehicles/vehicle_category_svm_classifier_liner_kernel.joblib")
def classify_ads(ads: list[str]):
  embeddings = generate_sentence_embeddings(ads)
  scaled_embeddings = vehicle_category_scaler.transform(embeddings)
  predictions = vehicle_category_classifier.predict(scaled_embeddings)
  
  return predictions.tolist()
  