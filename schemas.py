from pydantic import BaseModel, Field


class InputIris(BaseModel):
    sepal_length : float = Field(gt=0)
    sepal_width : float = Field(gt=0)
    petal_length : float = Field(gt=0)
    petal_width : float = Field(gt=0)

class IrisPredictionResponse(BaseModel):
    prediction : str
    prediction_id : int
    model : str

