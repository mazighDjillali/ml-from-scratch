# Entraînement SQL — base Pima

Base de pratique SQL construite à partir des données de `pima-predict/` (768 patients,
8 mesures cliniques + `Outcome`). But : apprendre SQL sur des données que je connais déjà
par cœur, et faire le pont avec mon modèle de régression logistique.

## Reprendre depuis n'importe quel PC

```bash
git pull
cd sql-practice
python build_pima_db.py        # (re)construit pima.db depuis le CSV du repo
sqlite3 pima.db                # ouvre le terminal SQL
```

Puis, à l'invite `sqlite>`, pour un affichage lisible :

```
.headers on
.mode column
```

> `pima.db` n'est **pas** versionné (fichier binaire, dérivé) — on le reconstruit avec le script.

## Déjà couvert (séance 2026-11-21)

| Brique | Ce qu'elle fait |
|---|---|
| `SELECT col FROM pima` | choisir les colonnes |
| `WHERE cond` | filtrer les lignes (`=` **simple**, jamais `==` → portable Postgres/MySQL) |
| `COUNT(*)` | compter les lignes (= mesurer une taille) |
| `AVG / SUM / MIN / MAX` | agréger des valeurs en un seul nombre |
| `GROUP BY col` | découper en paquets, agréger **dans chaque paquet** |

Ponts avec le ML : équilibre des classes (268 diabétiques / 500 sains = ma baseline) ;
`AVG(Glucose)` par groupe (141 vs 110) confirme le gros poids de `Glucose` dans mon modèle ;
pour comparer équitablement les écarts entre features d'échelles différentes → **standardiser**
(÷ écart-type) : Glucose 0.98 ≫ BMI 0.61 > Age 0.50.

## Prochaine brique (en cours)

- `ORDER BY col [DESC|ASC]` + `LIMIT n` → le motif *top-N* (ex. les 5 patients les plus âgés).
- Exercice en attente : requête des **5 patients les plus âgés** (afficher `Age, Glucose, Outcome`),
  prédire d'abord s'ils seront plutôt diabétiques ou sains.
- Ensuite : `DISTINCT`, puis une 2ᵉ table → les **jointures**.

## À faire pour l'instructeur à la prochaine séance

Ouvrir par le recall que j'ai esquivé 3× (non négociable) :
1. le seuil fait-il partie du modèle ? et « déployé » = quoi ?
2. `GROUP BY` en mes propres mots + déclaratif (SQL) vs boucle (Python).
