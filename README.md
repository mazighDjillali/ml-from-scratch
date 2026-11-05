# ML from scratch

Implémentations d'algorithmes de machine learning **codées à la main en NumPy**, sans scikit-learn — l'objectif est de comprendre les mécanismes (gradient, fonction de coût, mise à jour des poids), pas d'utiliser une boîte noire.

## Contenu

### `logistic_regression_from_scratch.py`
Régression logistique binaire par **descente de gradient**, codée de zéro :

- prédiction `ŷ = sigmoid(w·x + b)`
- gradient de la log-loss dérivé à la main : `grad_w = moyenne((ŷ − y)·x)`, `grad_b = moyenne(ŷ − y)` (même forme que le gradient de la MSE, seul `ŷ` passe par la sigmoïde)
- mise à jour `w ← w − lr·grad_w`
- décision par **seuillage à 0.5** puis mesure de l'**exactitude**

**Jeu de données jouet** : heures de révision → réussite/échec à un examen.

Ce que l'exemple met en évidence : avec une seule variable, deux élèves se contredisent (2.5 h reçu vs 3 h recalé) → les données ne sont **pas séparables** → l'exactitude plafonne à **0.8** quel que soit le nombre d'itérations. C'est une **erreur irréductible** : elle est dans les données, pas dans l'entraînement. Le remède n'est pas plus d'itérations mais **plus d'information** (une variable supplémentaire).

### `logistic_regression_multivariable.py`
Même modèle, mais **plusieurs variables**, écrit en **forme matricielle** (le passage à l'échelle) :

- `z = X @ w + b` où `X` est `(n × d)`, `w` un vecteur `(d,)` → tous les exemples d'un coup
- gradient vectorisé : `grad_w = (1/n) · X.T @ erreur`, de forme `(d,)` — un seul produit matriciel pour tous les poids
- la boucle est identique pour 2 variables comme pour 500 : on ne touche qu'à `X` et à la taille de `w`

**Démonstration** : on ajoute une 2ᵉ variable (heures de sommeil) qui départage les deux élèves contradictoires du cas à une variable → les données redeviennent **séparables** → l'exactitude passe de `0.8` à **`1.0`**. La leçon : pour franchir le plafond, il fallait **plus d'information**, pas plus d'entraînement.
