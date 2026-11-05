import numpy as np

# =====================================================================
# Overfitting (sur-ajustement) + regularisation, illustres de zero.
#
# Idee : ajuster des polynomes de flexibilite CROISSANTE sur 8 points
# BRUITES issus d'une simple droite y = 2x + 1, puis comparer l'erreur
# sur l'entrainement a l'erreur sur des points NEUFS.
# =====================================================================

rng = np.random.default_rng(1)

def vraie_fonction(x):
    return 2 * x + 1                      # la verite : une droite

# 8 points d'entrainement, bruites (de vraies mesures ne tombent jamais
# pile sur la droite)
x_train = np.linspace(0, 1, 8)
y_train = vraie_fonction(x_train) + rng.normal(0, 0.12, size=8)

# 50 points NEUFS, meme plage, jamais vus a l'entrainement
x_test = np.linspace(0.03, 0.97, 50)
y_test = vraie_fonction(x_test) + rng.normal(0, 0.12, size=50)

def features_polynomiales(x, degre):
    # transforme x en [1, x, x^2, ..., x^degre]
    return np.vstack([x ** k for k in range(degre + 1)]).T

def ajuste(degre, lam=0.0):
    X = features_polynomiales(x_train, degre)
    penalite = np.eye(degre + 1)
    penalite[0, 0] = 0                    # on ne penalise PAS le biais
    # cout = MSE + lambda * somme(w^2) ; l'ajustement n'est pas l'objet
    # ici, c'est la generalisation qui compte
    w = np.linalg.solve(X.T @ X + lam * penalite, X.T @ y_train)
    err_train = np.mean((X @ w - y_train) ** 2)
    err_test = np.mean((features_polynomiales(x_test, degre) @ w - y_test) ** 2)
    return w, err_train, err_test

# --- 1) sous-ajuster vs sur-ajuster : degre 1 (droite) vs degre 7 ------
print("=== Flexibilite du modele (sans regularisation) ===")
print(f"{'degre':>6} | {'erreur TRAIN':>13} | {'erreur TEST':>12}")
for degre in [1, 7]:
    _, etr, ete = ajuste(degre)
    print(f"{degre:>6} | {etr:13.6f} | {ete:12.4f}")
# Degre 7 = 8 poids pour 8 points -> la courbe passe PILE par tous les
# points -> erreur d'entrainement = 0. Mais pour y arriver elle a epouse
# le BRUIT propre a ces 8 points -> catastrophe sur les points neufs.

# --- 2) soigner le degre 7 avec la regularisation L2 (Ridge) ----------
print("\n=== Regularisation du degre 7 : balayage de lambda ===")
print(f"{'lambda':>10} | {'err TRAIN':>10} | {'err TEST':>10} | {'somme w^2':>12}")
for lam in [0, 1e-4, 1e-2, 1.0, 100.0]:
    w, etr, ete = ajuste(7, lam)
    print(f"{lam:>10g} | {etr:10.5f} | {ete:10.5f} | {np.sum(w[1:] ** 2):12.2f}")
# lambda = 0     -> rien ne punit les poids -> poids immenses, zigzag  -> OVERFIT
# lambda enorme  -> poids ecrases vers 0 -> courbe plate               -> UNDERFIT
# Le bon lambda est AU MILIEU : l'erreur de test descend puis remonte (forme en U).
