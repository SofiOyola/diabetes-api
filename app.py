# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import uvicorn

# Crear instancia de FastAPI
app = FastAPI(title="API Predicción de Diabetes", version="1.0")

# Cargar el modelo y el scaler
modelo = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")

# Definir el esquema de entrada con Pydantic
class DatosEntrada(BaseModel):
    Age: float
    Pregnancies: float
    BMI: float
    Glucose: float
    BloodPressure: float
    HbA1c: float
    LDL: float
    HDL: float
    Triglycerides: float
    WaistCircumference: float
    HipCircumference: float
    WHR: float
    FamilyHistory: float
    DietType: float
    Hypertension: float
    MedicationUse: float

@app.post("/diabetes")
def predecir_diabetes(datos: DatosEntrada):
    # Convertir datos a numpy array
    entrada = np.array([[datos.Age, datos.Pregnancies, datos.BMI, datos.Glucose,
                         datos.BloodPressure, datos.HbA1c, datos.LDL, datos.HDL,
                         datos.Triglycerides, datos.WaistCircumference, datos.HipCircumference,
                         datos.WHR, datos.FamilyHistory, datos.DietType,
                         datos.Hypertension, datos.MedicationUse]])

    # Escalar los datos
    entrada_escalada = scaler.transform(entrada)

    # Obtener probabilidad de predicción
    probabilidad = modelo.predict_proba(entrada_escalada)[0][1]

    # Aplicar el umbral de 0.5
    prediccion = 1 if probabilidad > 0.5 else 0

    # Generar respuesta
    resultado = {
        "riesgo_diabetes": "Sí tiene riesgo de diabetes" if prediccion == 1 else "No tiene riesgo de diabetes",
        "probabilidad": float(probabilidad)
    }

    return resultado


# Ejecutar el servidor en puerto 5001
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
