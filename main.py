from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

def classify_message(model, message):
    label = model.predict([message])[0]
    spam_prob = model.predict_proba([message])
    return {'label': label, 'spam_probability': spam_prob[0][1]}

@app.get("/")
async def root():
    return {"message": "Predict Income of Customers!"}

#### docker imajını tekrardan yap!!!
import pandas as pd

from fastapi import Request
"""
@app.get('/predict')
async def predict(request: Request):
    # Getting the JSON from the body of the request
    input_data = await request.json()

    # Converting JSON to Pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # Getting the prediction from the Logistic Regression model
    pred = model.predict(input_df)[0]

    return pred
"""
from pydantic import BaseModel

class Data(BaseModel):
    Income: int
    Wines: int
    Meat: int
    Gold: int
    Spent: int

from fastapi import Depends
@app.post("/predict")
def predict(data: Data = Depends()):
    income = data.Income
    wines = data.Wines
    meat = data.Meat
    gold = data.Gold
    spent = data.Spent
    data = data.dict()
    data_in = [[data['Income'], data['Wines'], data['Meat'], data['Gold'], data['Spent']]]
    prediction = model.predict(data_in)
    probability = model.predict_proba(data_in).max()
    return {
        'prediction': prediction[0],
        'probability': probability
    }


