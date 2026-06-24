import pandas as pd
import joblib

model = joblib.load("iris_model.pkl")
scaler = joblib.load("iris_scaler.pkl")
iris_target_names = joblib.load("iris_target_names.pkl")

def predict_iris(X):

    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    prediction_id = int(prediction[0])

    species = str(iris_target_names[prediction_id])

    return {
        "prediction": species,
        "prediction_id": prediction_id,
        "model": "RandomForestClassifier"
    }

    


