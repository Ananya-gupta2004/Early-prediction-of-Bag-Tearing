import joblib
from tensorflow.keras.models import load_model

MODEL_DIR = r"C:\Users\anany\Downloads\baghouse_outputs"

cnn_bilstm = load_model(f"{MODEL_DIR}/cnn_bilstm_model.keras")

embedding_model = load_model(f"{MODEL_DIR}/embedding_model.keras")

lgb_model = joblib.load(f"{MODEL_DIR}/lightgbm_model.pkl")

scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")

le = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")

print("All models loaded successfully")