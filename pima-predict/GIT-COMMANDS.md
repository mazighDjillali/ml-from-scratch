# Commandes git utilisées — mémo

Toutes les commandes git qui ont servi à versionner ce projet et à le pousser dans
`ml-from-scratch`, dans l'ordre, avec une explication pour chacune. Garde ce fichier
comme antisèche git.

---

## 1. Versionner le projet en local (jour 1)

```bash
git init
```
Crée un dépôt git **local** dans le dossier (un sous-dossier caché `.git/`). À faire une seule fois par projet.

```bash
git add -A
```
« Stage » (prépare) **tous** les fichiers modifiés/ajoutés/supprimés pour le prochain commit.

```bash
git commit -m "message"
```
Enregistre un **instantané** des fichiers stagés, avec un message. C'est le point de sauvegarde.

```bash
git log --oneline
```
Affiche l'historique des commits, un par ligne (hash court + message).

### `.gitignore`
Fichier texte qui liste ce que git doit **ignorer** (ne jamais versionner). Ici :
```
__pycache__/
.ipynb_checkpoints/
```

---

## 2. Pousser dans un dépôt GitHub existant (`ml-from-scratch`)

```bash
git clone --depth 1 https://github.com/mazighDjillali/ml-from-scratch.git
```
Copie le dépôt distant en local. `--depth 1` = seulement le dernier commit (clone rapide, suffisant pour ajouter des fichiers).

```bash
# (copie des fichiers du projet dans un nouveau dossier pima-predict/)
```
On ajoute le projet **comme un dossier** du repo, sans toucher au reste.

```bash
git add -A
git commit -m "Ajout du projet pima-predict (logistique from scratch + contrat d'inference)"
```
On stage et on enregistre l'ajout.

```bash
git pull --rebase --autostash
```
Récupère d'éventuels changements distants **avant** de pousser, pour éviter les conflits.
`--rebase` rejoue tes commits par-dessus ; `--autostash` met de côté le travail non commité le temps du pull.

```bash
git push
```
Envoie tes commits locaux vers GitHub. C'est l'étape qui rend le code public/visible en ligne.

---

## 3. Commandes de contrôle utiles (à connaître)

```bash
git status          # ce qui est modifié / stagé / non suivi
git diff            # voir les changements ligne par ligne (avant de stager)
git remote -v       # vers quel dépôt distant on pousse
git branch          # lister les branches ; branche par défaut souvent 'main'
git restore <fichier>   # annuler les modifs non commitées d'un fichier
```

---

## Le modèle mental

- **working dir** (tes fichiers) → `git add` → **staging** → `git commit` → **historique local** → `git push` → **GitHub**.
- Un commit = une photo. Un push = tu envoies tes photos sur le serveur.
- On commit **souvent** (petits pas), on push quand on veut partager / sauvegarder en ligne.
