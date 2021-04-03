#load lybrary
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd 

app = FastAPI()

class IrisSpecies(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width :float

#Expose the prediction functionality, make a prediction from the passed
#JSON data and return the predicted flower species with the confidence
@app.post("/predict")
def predict_species(iris: IrisSpecies):
    data = iris.dict()
    load_model = pickle.load(open("rf_model.pkl", "rb"))
    datain = [[data["sepal_length"], data["sepal_width"], data["petal_length"], data["petal_width"]]]
    prediction = load_model.predict(datain)
    probability = load_model.predict_proba(datain).max()
    return {
        "prediction ": prediction[0],
        "probability ": probability
    }