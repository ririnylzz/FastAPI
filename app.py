from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Inisialisasi FastAPI
app = FastAPI(title="Product Category Prediction API (GnB + Scaler)")

# Load model dan scaler
with open("WebMinnersbaru.pkl", "rb") as f:
    saved_objects = pickle.load(f)
    model = saved_objects['model']
    scaler = saved_objects['scaler']

# Skema input sesuai fitur data
class InputData(BaseModel):
    Product_Name: int
    Product_Price: float
    Quantity: float
    Total: int
    Month: int
    Quantity_Monthly: float
    Day: int
    Year: int

# Preprocessing input
def preprocess_input(data: InputData):
    df = pd.DataFrame([data.dict()])
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled

# Endpoint root
@app.get("/")
def read_root():
    return {"message": "✅ Product Category Prediction API is running"}

# Endpoint prediksi
@app.post("/predict")
def predict_category(data: InputData):
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]
    label_map = {0: "Sedikit", 1: "Sedang", 2: "Banyak"}
    return {
        "predicted_category": label_map.get(prediction, "Unknown")
    }