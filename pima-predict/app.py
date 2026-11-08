import os
import numpy as np
import joblib
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse

# --- charge l'artefact UNE SEULE FOIS au demarrage ---
# ART = {w, b, mean, std, features, cols_zero}
ART = joblib.load(os.path.join(os.path.dirname(__file__), "data", "model.joblib"))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


PAGE = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pima - Risque de diabete</title>
<style>
  :root{
    --bg:#0f172a; --card:#1e293b; --muted:#94a3b8; --text:#e2e8f0;
    --accent:#38bdf8; --line:#334155; --field:#0f172a;
  }
  *{box-sizing:border-box}
  body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
       background:radial-gradient(1200px 600px at 20% -10%, #1e3a5f 0%, var(--bg) 55%);
       color:var(--text);font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:24px}
  .card{width:100%;max-width:560px;background:var(--card);border:1px solid var(--line);
        border-radius:18px;padding:28px 28px 24px;box-shadow:0 20px 60px rgba(0,0,0,.45)}
  h1{margin:0 0 2px;font-size:22px;letter-spacing:.2px}
  .sub{color:var(--muted);font-size:13px;margin-bottom:20px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px 16px}
  label{display:flex;flex-direction:column;gap:5px;font-size:12px;color:var(--muted)}
  input{background:var(--field);border:1px solid var(--line);color:var(--text);
        border-radius:10px;padding:9px 11px;font-size:14px;outline:none}
  input:focus{border-color:var(--accent)}
  .row{display:flex;gap:10px;margin-top:18px}
  button{flex:1;border:0;border-radius:11px;padding:12px;font-size:14px;font-weight:600;cursor:pointer}
  .ghost{background:transparent;border:1px solid var(--line);color:var(--text)}
  .ghost:hover{border-color:var(--accent);color:var(--accent)}
  .primary{background:var(--accent);color:#06283d}
  .primary:hover{filter:brightness(1.08)}
  .result{margin-top:22px;opacity:0;transform:translateY(6px);transition:.35s;}
  .result.show{opacity:1;transform:none}
  .pct{font-size:40px;font-weight:700;line-height:1}
  .verdict{font-size:14px;font-weight:600;margin-top:4px}
  .bar{height:12px;border-radius:999px;background:#0b1220;overflow:hidden;margin-top:14px;border:1px solid var(--line)}
  .fill{height:100%;width:0;border-radius:999px;transition:width .6s ease}
  .foot{margin-top:16px;font-size:11px;color:var(--muted);text-align:center}
</style>
</head>
<body>
  <div class="card">
    <h1>Risque de diabete</h1>
    <div class="sub">Regression logistique codee de zero - 8 mesures cliniques (Pima)</div>
    <form id="f" class="grid">
      <label>Grossesses <input name="Pregnancies" type="number" step="any" required></label>
      <label>Glucose <input name="Glucose" type="number" step="any" required></label>
      <label>Tension (BloodPressure) <input name="BloodPressure" type="number" step="any" required></label>
      <label>Pli cutane (SkinThickness) <input name="SkinThickness" type="number" step="any" required></label>
      <label>Insuline <input name="Insulin" type="number" step="any" required></label>
      <label>IMC (BMI) <input name="BMI" type="number" step="any" required></label>
      <label>Pedigree diabete <input name="DiabetesPedigreeFunction" type="number" step="any" required></label>
      <label>Age <input name="Age" type="number" step="any" required></label>
    </form>
    <div class="row">
      <button class="ghost" onclick="randomize()">&#127922; Donnees aleatoires</button>
      <button class="primary" onclick="predict()">Predire</button>
    </div>
    <div id="result" class="result">
      <div class="pct" id="pct">--</div>
      <div class="verdict" id="verdict"></div>
      <div class="bar"><div class="fill" id="fill"></div></div>
    </div>
    <div class="foot">Modele servi depuis model.joblib - prediction en direct</div>
  </div>

<script>
const F = {
  Pregnancies:[0,15,0], Glucose:[70,200,0], BloodPressure:[50,110,0],
  SkinThickness:[10,50,0], Insulin:[15,300,0], BMI:[18,50,1],
  DiabetesPedigreeFunction:[0.08,2.4,3], Age:[21,70,0]
};
function rnd(min,max,dec){ const v=Math.random()*(max-min)+min; return v.toFixed(dec); }
function randomize(){
  for(const k in F){ document.querySelector(`[name="${k}"]`).value = rnd(...F[k]); }
}
async function predict(){
  const form = document.getElementById('f');
  if(!form.reportValidity()) return;
  const body = new URLSearchParams(new FormData(form));
  const r = await fetch('/predict', {method:'POST', body});
  if(!r.ok){ alert('Erreur serveur'); return; }
  const j = await r.json();
  const p = j.proba, pct = Math.round(p*1000)/10;
  const res = document.getElementById('result');
  document.getElementById('pct').textContent = pct + '%';
  const high = p >= 0.5;
  document.getElementById('verdict').textContent =
     high ? 'Diabete probable (classe 1)' : 'Faible risque (classe 0)';
  const color = p < 0.3 ? '#22c55e' : p < 0.6 ? '#f59e0b' : '#ef4444';
  document.getElementById('verdict').style.color = color;
  const fill = document.getElementById('fill');
  fill.style.width = pct + '%';
  fill.style.background = color;
  res.classList.add('show');
}
randomize();
</script>
</body>
</html>
"""


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


@app.get("/predict", response_class=HTMLResponse)
def predict_form():
    return PAGE


@app.post("/predict")
def predict(Pregnancies: float = Form(...), Glucose: float = Form(...),
            BloodPressure: float = Form(...), SkinThickness: float = Form(...),
            Insulin: float = Form(...), BMI: float = Form(...),
            DiabetesPedigreeFunction: float = Form(...), Age: float = Form(...)):
    # --- ta logique (contrat d'inference), inchangee ---
    vals = {
        "Pregnancies": Pregnancies, "Glucose": Glucose,
        "BloodPressure": BloodPressure, "SkinThickness": SkinThickness,
        "Insulin": Insulin, "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction, "Age": Age,
    }
    X = np.array([[vals[f] for f in ART["features"]]], dtype=float)
    for j in ART["cols_zero"]:
        X[X[:, j] == 0, j] = ART["mean"][j]
    X = (X - ART["mean"]) / ART["std"]
    proba = float(sigmoid(X @ ART["w"] + ART["b"])[0])
    classe = int(proba > 0.5)
    # --- sortie JSON pour le front ---
    return JSONResponse({"proba": proba, "classe": classe})
