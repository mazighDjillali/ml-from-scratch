import os
import numpy as np
import joblib
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

# --- charge l'artefact UNE SEULE FOIS au demarrage (comme hello-api) ---
# ART = le dict que tu as sauve : {w, b, mean, std, features, cols_zero}
ART = joblib.load(os.path.join(os.path.dirname(__file__), "data", "model.joblib"))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


FORM_HTML = """
<h2>Pima - risque de diabete</h2>
<form method="post" action="/predict">
  <label>Pregnancies: <input name="Pregnancies" type="number" step="any" required></label><br>
  <label>Glucose: <input name="Glucose" type="number" step="any" required></label><br>
  <label>BloodPressure: <input name="BloodPressure" type="number" step="any" required></label><br>
  <label>SkinThickness: <input name="SkinThickness" type="number" step="any" required></label><br>
  <label>Insulin: <input name="Insulin" type="number" step="any" required></label><br>
  <label>BMI: <input name="BMI" type="number" step="any" required></label><br>
  <label>DiabetesPedigreeFunction: <input name="DiabetesPedigreeFunction" type="number" step="any" required></label><br>
  <label>Age: <input name="Age" type="number" step="any" required></label><br>
  <button type="submit">Predire</button>
</form>
"""

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Pima predict API - va sur /predict"}


@app.get("/predict", response_class=HTMLResponse)
def predict_form():
    return FORM_HTML


@app.post("/predict", response_class=HTMLResponse)
def predict(Pregnancies: float = Form(...), Glucose: float = Form(...),
            BloodPressure: float = Form(...), SkinThickness: float = Form(...),
            Insulin: float = Form(...), BMI: float = Form(...),
            DiabetesPedigreeFunction: float = Form(...), Age: float = Form(...)):
    # 1. les 8 valeurs du formulaire dans un dict {nom: valeur}
    vals = {
        "Pregnancies": Pregnancies, "Glucose": Glucose,
        "BloodPressure": BloodPressure, "SkinThickness": SkinThickness,
        "Insulin": Insulin, "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction, "Age": Age,
    }

    # 2. construire X dans l'ordre EXACT des features d'entrainement -> shape (1, 8)
    X = np.array([[vals[f] for f in ART["features"]]], dtype=float)

    # 3. imputation des 0 avec les valeurs GELEES du train
    for j in ART["cols_zero"]:
        X[X[:, j] == 0, j] = ART["mean"][j]

    # 4. standardisation avec mean/std GELES du train
    X = (X - ART["mean"]) / ART["std"]

    # 5. prediction
    proba = sigmoid(X @ ART["w"] + ART["b"])[0]
    classe = int(proba > 0.5)

    # 6. reponse HTML
    return f"<h3>Proba de diabete : {proba:.3f}</h3><p>Classe predite : {classe} (1 = diabete)</p>"
