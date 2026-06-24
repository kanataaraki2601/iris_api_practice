# Iris Classifier API

This is a small machine learning API built with FastAPI.

The API recieves iris flower measurements and predicts the iris species using a trained scikit-learn model.

## Features

- Train an iris classifier with scikit-learn
- Save the trained model with joblib
- Load the model in a FastAPI backend
- Validata request data with Pydantic
- Return predictions as JSON
- Test the API using FastAPI's automatic `/docs` page

## Tech Stack

- Python
- FastAPI
- Pydantic
- pandas
- scikit-learn
- joblib
- uvicorn

## Project Structure

```text
iris_api/
├── README.md
├── __pycache__
│   ├── app.cpython-313.pyc
│   ├── model_service.cpython-313.pyc
│   └── schemas.cpython-313.pyc
├── app.py
├── iris_model.pkl
├── iris_scaler.pkl
├── iris_target_names.pkl
├── model_service.py
├── requirement.txt
├── schemas.py
└── train.py
```
---


## Installation

### 1. Clone the repository

```bash

git clone <your-repository-url>
cd iris_api
```


### 2.Create a virtual environment

```bash
python3 -m venv .venv
```

### 3.Activate the virtual environment

macOS / Linux:

```bash
.venv/bin/activate
```

Windows:

```bash
.venv/Scripts/activate
```


---

## Train the Model

Run the training script:

```bash
python train.py
```


After training, the following files will be created:

```text
iris_model.pkl
iris_scaler.pkl
iris_target_names.pkl
```


These files store:

- Trained machine learning model
- Feature scaler
- Species names


---



## Run the API

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

You should see output similar to:

```text
INFO: Uvicorn running on http://127.0.0.1:8000
```

Open your browserand visit:

```text
http://127.0.0.1:8000/
```

Example response:

```json
{
  "message": "Iris API is running"
}
```

---

## API Documentation

FastAPI automatically generates interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

From this page you can:

- View all endpoints
- Test API requests
- Inspect request and response schemas

---

## Prediction Endpoint

### POST `/predict`

Predict the iris species from flower measurements.

### Request Body

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Response Body

```json
{
  "prediction": "setosa",
  "prediction_id": 0
}
```

---

## Example Prediction Flow

```text
Client Request
       │
       ▼
JSON Input
       │
       ▼
Pydantic Validation
       │
       ▼
pandas DataFrame
       │
       ▼
StandardScaler
       │
       ▼
RandomForest Model
       │
       ▼
Predicted Species
       │
       ▼
JSON Response
```

---

## Requirements

Dependencies are listed in:

```text
requirements.txt
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## Example Workflow

### Train the model

```bash
python train.py
```

### Start the API

```bash
uvicorn app:app --reload
```

### Open Swagger UI

```text
http://127.0.0.1:8000/docs
```

### Send prediction request

```json
{
  "sepal_length": 6.3,
  "sepal_width": 3.3,
  "petal_length": 6.0,
  "petal_width": 2.5
}
```

### Example response

```json
{
  "prediction": "virginica",
  "prediction_id": 2
}
```

---

## Learning Objectives

This project demonstrates:

- Machine learning model training
- Model serialization with joblib
- Feature scaling
- Data validation with Pydantic
- Backend API development using FastAPI
- JSON request and response handling
- Deploying machine learning models as web services

---

## Author

Araki Kanata

Built as part of a machine learning and backend engineering learning project.

