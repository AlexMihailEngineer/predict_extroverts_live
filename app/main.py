from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

path = 'models/'

imputer = joblib.load(path + 'imputer_v1.pkl')  # Adjust path as needed
imputer_cat = joblib.load(path + 'imputer_cat_v1.pkl')  # Adjust path as needed
cat_encoder = joblib.load(path + 'cat_encoder_v1.pkl')  # Adjust path as needed
scaler = joblib.load(path + 'scaler_v1.pkl')  # Adjust path as needed
stackingC = joblib.load(path + 'extroverts_v1.pkl')  # Adjust path as needed


app = FastAPI()

# Define input data schema
class ModelInput(BaseModel):
    Time_spent_Alone: float
    Stage_fear: str
    Social_event_attendance: float
    Going_outside: float
    Drained_after_socializing: str
    Friends_circle_size: float
    Post_frequency: float

    class Config:
        schema_extra = {
            "example": {
                "Time_spent_Alone": 0.0,
                "Stage_fear": "No",
                "Social_event_attendance": 6.0,
                "Going_outside": 4.0,
                "Drained_after_socializing": "No",
                "Friends_circle_size": 15.0,
                "Post_frequency": 5.0
            }
        }



@app.post('/predict')
def predict_personality(input_data: ModelInput):
    # Convert input to DataFrame (single row)
    data = pd.DataFrame([input_data.dict()])

    # Define numerical and categorical columns (based on your training data)
    num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    cat_cols = ['Stage_fear', 'Drained_after_socializing']

    # Split into num and cat
    X_num = data[num_cols]
    X_cat = data[cat_cols]

    # Impute numerical features
    X_num_imputed = pd.DataFrame(imputer.transform(X_num), columns=num_cols)

    # Impute categorical features
    X_cat_imputed = pd.DataFrame(imputer_cat.transform(X_cat), columns=cat_cols)

    # One-hot encode categorical features
    X_cat_1hot = pd.DataFrame(
        cat_encoder.transform(X_cat_imputed).toarray(),
        columns=cat_encoder.get_feature_names_out()
    )

    # Scale numerical features
    X_num_scaled = pd.DataFrame(scaler.transform(X_num_imputed), columns=num_cols)

    # Concatenate processed numerical and categorical features
    X_final = pd.concat([X_num_scaled, X_cat_1hot], axis=1)

    # Make prediction
    pred = stackingC.predict(X_final)[0]

    # Map back to label
    personality = 'Extrovert' if pred == 1 else 'Introvert'

    return {'Personality': personality}

app.mount("/", StaticFiles(directory="static", html=True), name="static")