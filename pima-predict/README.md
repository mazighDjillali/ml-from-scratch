# Pima Diabetes — régression logistique *from scratch* + contrat d'inférence

Prédiction du diabète à partir de 8 mesures cliniques (dataset Pima Indians).
Régression logistique codée **de zéro en numpy** (pas de scikit-learn pour le modèle),
avec une attention particulière sur **le passage en production** : le *contrat d'inférence*.

## Ce que ce projet démontre

Déployer un modèle, ce n'est pas « sauver les poids ». C'est **geler et expédier tout le
preprocessing** avec le modèle, pour que l'inférence refasse *exactement* ce que
l'entraînement a fait — sinon le modèle se trompe **sans aucune erreur visible**
(*train/serve skew*).

Concrètement :

- **Standardisation calculée sur le TRAIN uniquement** (`mean`/`std` gelés), puis
  ré-appliquée à l'identique au test et à l'inférence.
- **Artefact autonome** `model.joblib` = `{ w, b, mean, std, features, cols_zero }` :
  tout ce qu'il faut pour prédire, et rien des hyperparamètres (inutiles après l'entraînement).
- **`predict_from_artifact(...)`** ne lit **que** le joblib — zéro variable du notebook.
  Vérifié dans un processus Python neuf : il recharge l'artefact et prédit, comme le ferait un serveur.

## Résultat

- Exactitude test : **0.748** (bat la baseline « classe majoritaire » ≈ 0.65).
- La standardisation supprime l'overflow numérique et empêche le modèle de s'effondrer
  sur la classe majoritaire.

## Fichiers

| Fichier | Rôle |
|---|---|
| `train.ipynb` | Pipeline complet : charger → split → standardiser (train-only) → entraîner → sauver l'artefact → `predict_from_artifact` |
| `data/pima.csv` | Dataset (768 lignes, 8 features + `Outcome`) |
| `data/model.joblib` | L'artefact déployable |
| `plan-pipeline-v1.pdf` | Schéma du flux entraînement → artefact → inférence |
| `plan-code-v1.pdf` | Charpente du code |

## Limites connues (v2)

- Les `0` physiologiquement impossibles (Glucose, Insulin…) sont des valeurs manquantes
  déguisées : imputation à durcir et à figer proprement dans l'artefact.
- Régularisation L2 + courbes d'apprentissage à ajouter.
- Exposition derrière une route `/predict` (le code est déjà prêt pour).
