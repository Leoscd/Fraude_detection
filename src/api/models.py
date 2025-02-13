from pydantic import BaseModel
from typing import List



class PredictionInput(BaseModel):
    V14: float
    V10: float
    V4: float
    V12: float
    V1: float

class PredictionOutput(BaseModel):
    prediction: int
    probability: float