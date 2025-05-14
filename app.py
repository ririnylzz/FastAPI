from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import secrets
import pickle
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Inisialisasi FastAPI
app = FastAPI(
    title="Product Category Prediction API (GnB + Scaler)",
    description="API untuk prediksi kategori produk dengan autentikasi token",
    version="1.0.0"
    )

# Setup security
security = HTTPBearer()

# Valid tokens from environment
VALID_TOKENS = set(os.getenv("API_TOKENS", "").split(","))

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

# Fungsi verifikasi token
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token not in VALID_TOKENS:
        raise HTTPException(
            status_code=403,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return token
    
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

# Endpoint prediksi dengan autentikasi
@app.post("/predict")
def predict_category(
    data: InputData,
    token: str = Depends(verify_token)  # Token verification
):
    try:
        processed = preprocess_input(data)
        prediction = model.predict(processed)[0]
        label_map = {0: "Sedikit", 1: "Sedang", 2: "Banyak"}
        
        return {
            "status": "success",
            "predicted_category": label_map.get(prediction, "Unknown"),
            "model_used": "Gaussian Naive Bayes with Scaler",
            "token_used": token[-4:]  # Show last 4 chars for verification
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
