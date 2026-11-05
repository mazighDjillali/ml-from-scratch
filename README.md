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

### `overfitting_et_regularisation.py`
Voir le **sur-ajustement (overfitting)** arriver, le **mesurer**, puis le **soigner** — sur des polynômes ajustés à 8 points bruités issus d'une droite `y = 2x + 1` :

- un polynôme de **degré 7** a 8 poids pour 8 points → il passe **pile** par tous → **erreur d'entraînement = 0**… mais il a épousé le **bruit** propre à ces 8 points, pas le signal (la droite)
- résultat : erreur de **test** du degré 7 (`0.032`) **pire** que celle de la simple droite (`0.012`) — le modèle « parfait » à l'entraînement est le plus mauvais sur des données neuves
- remède, la **régularisation L2 (Ridge)** : `coût = MSE + λ · Σ wⱼ²` → on **punit les gros poids**. `λ` règle le compromis.

**Démonstration** (balayage de `λ` sur le degré 7) : la somme des poids au carré fond de `85 000 000` (λ=0, zigzag) à `~0` (λ énorme, courbe plate), et l'erreur de test décrit un **U** — minimum **au milieu**.

| diagnostic | erreur train | erreur test |
|---|---|---|
| **overfit** (λ=0) | basse | **haute** |
| **bon modèle** (λ bien choisi) | basse | basse |
| **underfit** (λ énorme) | **haute** | **haute** |

La leçon : la flexibilité d'un modèle doit **correspondre** à la complexité du vrai signal ; trop de flexibilité apprend le bruit, pas assez rate le signal. On lit l'overfitting dans l'**écart train↔test**.

### `train_validation_test.py`
Pourquoi on découpe les données en **trois** (train / validation / test) — illustré sur un dataset réaliste (120 exemples, 40 mesures dont 5 seulement portent le signal) avec une régression logistique L2 :

- **train** → ajuster les poids `w`, `b`
- **validation** → choisir l'hyperparamètre `λ` (on prend le `λ` qui **maximise l'exactitude de validation**)
- **test** → la mesure finale **honnête**, touchée **une seule fois, à la toute fin**

**Démonstration** : on choisit `λ = 0.1` (meilleure validation, `0.933`). Mais le **test** donne `0.767` — plus bas. Pourquoi ? La validation a servi à **choisir** → son chiffre est **optimiste**. Le test, jamais regardé pendant le choix, reste la seule estimation honnête.

La leçon : **choisir un modèle sur un jeu de données rend l'erreur de ce jeu trop optimiste.** Le test n'est fiable que parce qu'on ne l'a *jamais* consulté pour décider — et on ne revient **pas** en arrière changer `λ` après l'avoir vu (ce serait tricher à nouveau).
