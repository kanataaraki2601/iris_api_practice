from fastapi import FastAPI
from schemas import InputIris, IrisPredictionResponse
from model_service import predict_iris

import pandas as pd

app = FastAPI()

@app.get("/")
def hello():
    return {
        "message": "Iris is running",
        "status": "ok"
    }

@app.post("/predict", response_model=IrisPredictionResponse)
def predict(data: InputIris):

    X = pd.DataFrame(
        {"sepal length (cm)": [data.sepal_length],
        "sepal width (cm)": [data.sepal_width],
        "petal length (cm)": [data.petal_length],
        "petal width (cm)": [data.petal_width]}
        )

    result = predict_iris(X)

    return result