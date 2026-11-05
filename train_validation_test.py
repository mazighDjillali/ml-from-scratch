import numpy as np

# =====================================================================
# Pourquoi on decoupe les donnees en TROIS : train / validation / test.
#
# On choisit l'hyperparametre lambda SUR LA VALIDATION, jamais sur le
# test. Si on choisissait lambda en regardant le test, le test aurait
# servi a DECIDER (pas seulement a mesurer) -> son chiffre deviendrait
# trop optimiste et ne mesurerait plus rien d'honnete.
# =====================================================================

rng = np.random.default_rng(7)

# --- jeu de donnees realiste : 120 exemples, 40 mesures, cible binaire.
# Seules 5 mesures portent vraiment le signal ; les 35 autres sont du
# bruit distrayant (c'est la que la regularisation devient utile).
n, p, p_info = 120, 40, 5
X = rng.normal(size=(n, p))
w_vrai = np.zeros(p)
w_vrai[:p_info] = rng.normal(0, 2.0, size=p_info)
y = ((X @ w_vrai + rng.normal(0, 0.5, size=n)) > 0).astype(float)

# --- decoupage 60 train / 30 validation / 30 test
idx = rng.permutation(n)
tr, va, te = idx[:60], idx[60:90], idx[90:]
X_train, y_train = X[tr], y[tr]
X_val,   y_val   = X[va], y[va]
X_test,  y_test  = X[te], y[te]

# standardisation : moyenne et ecart-type calcules SUR LE TRAIN seulement
# (le modele n'a pas le droit de "voir" la validation ni le test)
mu, sd = X_train.mean(0), X_train.std(0) + 1e-9
X_train = (X_train - mu) / sd
X_val   = (X_val   - mu) / sd
X_test  = (X_test  - mu) / sd

def sigmoid(z):
    z = np.clip(z, -500, 500)   # evite l'overflow de exp quand les poids explosent (lambda=0)
    return 1 / (1 + np.exp(-z))

def entraine(lam, lr=0.3, nb_iteration=300):
    w = np.zeros(p)
    b = 0.0
    for _ in range(nb_iteration):
        y_pred = sigmoid(X_train @ w + b)
        erreur = y_pred - y_train
        grad_w = X_train.T @ erreur / len(y_train) + lam * w   # L2 sur w, pas sur b
        grad_b = np.mean(erreur)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

def exactitude(w, b, Xs, ys):
    return np.mean((sigmoid(Xs @ w + b) > 0.5) == ys)

# --- 1) choisir lambda en ne regardant QUE la validation --------------
print(f"{'lambda':>8} | {'acc TRAIN':>10} | {'acc VALID':>10}")
meilleur_lambda, meilleure_val = None, -1.0
for lam in [0.0, 0.01, 0.1, 1.0, 10.0]:
    w, b = entraine(lam)
    a_tr = exactitude(w, b, X_train, y_train)
    a_va = exactitude(w, b, X_val, y_val)
    print(f"{lam:>8g} | {a_tr:10.3f} | {a_va:10.3f}")
    if a_va > meilleure_val:              # critere : le MAX de l'accuracy de validation
        meilleure_val, meilleur_lambda = a_va, lam

print(f"\nlambda choisi (meilleure validation) : {meilleur_lambda}")

# --- 2) UNE SEULE mesure finale et honnete, sur le test jamais vu ------
w, b = entraine(meilleur_lambda)
print(f"accuracy VALIDATION : {exactitude(w, b, X_val, y_val):.3f}  (optimiste : a servi a choisir lambda)")
print(f"accuracy TEST       : {exactitude(w, b, X_test, y_test):.3f}  <- le chiffre HONNETE a annoncer")
# La validation surestime (elle a servi a choisir). Le test, jamais touche
# jusqu'ici, donne la vraie estimation. Et on NE revient PAS en arriere pour
# choisir un autre lambda apres avoir vu le test : ce serait tricher a nouveau.
