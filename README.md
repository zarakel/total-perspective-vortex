# Total Perspective Vortex — BCI Motor Imagery Classification

## 1. Présentation

Projet de classification EEG par imagerie motrice (Motor Imagery) utilisant le dataset **PhysioNet EEGBCI** (109 sujets, 64 canaux EEG).  
Le but : entraîner un pipeline sklearn capable de prédire si un sujet imagine bouger sa **main gauche** ou **main droite** (ou poings/pieds), à partir de ses signaux cérébraux, avec une accuracy ≥ 60%.

---

## 2. Architecture

```
EEG brut (64 canaux × N temps)
    │dans un premier temps
    ▼
Filtrage passe-bande FIR (8-30 Hz)      ← src/preprocessing.py
    │
    ▼
Epochage                                 ← src/loader.py
    │
    ▼
┌─── Pipeline sklearn ──────────────────────────────────┐
│  CSP custom (BaseEstimator + TransformerMixin)         │  ← src/csp_custom.py
│    → Résout le problème d'eigenvalues généralisé       │
│    → Projette sur les filtres spatiaux                 │
│    → Retourne les log-variances                        │
│  StandardScaler                                        │
│  Classifieur (LogisticRegression ou LDA custom)        │  ← src/lda_custom.py (bonus)
└────────────────────────────────────────────────────────┘
    │
    ▼
Prédiction : "main gauche" ou "main droite"
```

### Pipeline alternatif (FeatureExtractor)

Activable via `--use-features` : remplace le CSP par une extraction PSD + wavelets.

```
Epochs → FeatureExtractor (PSD par bande + ondelettes) → StandardScaler → Classifieur
```

---

## 3. Commandes

### Build

```bash
docker compose build
```

### Train

```bash
# Entraînement classique (CSP + LogisticRegression)
docker compose run --rm matplotlib python tpv.py train \
  --subject 1 --runs 4 8 12 --model-path model.joblib

# Avec visualisation (raw + filtré + spectre PSD)
docker compose run --rm matplotlib python tpv.py train \
  --subject 1 --runs 4 8 12 --model-path model.joblib --show-raw

# Avec les bonus (LDA custom + eigenvalue custom)
docker compose run --rm matplotlib python tpv.py train \
  --subject 1 --runs 4 8 12 --model-path model.joblib \
  --use-custom-clf --use-custom-eigen

# Avec GridSearch pour optimiser les hyperparamètres
docker compose run --rm matplotlib python tpv.py train \
  --subject 1 --runs 4 8 12 --model-path model.joblib --tune
```

### Predict (sur des données jamais vues)

```bash
# Les runs de prediction doivent être DIFFERENTS des runs d'entraînement
docker compose run --rm matplotlib python tpv.py predict \
  --subject 1 --runs 3 7 11 --model-path model.joblib
```

### Evaluate-all (109 sujets × 4 types d'expériences)

```bash
# Évaluation complète sur PhysioNet (StratifiedKFold cross-validation)
docker compose run --rm matplotlib python tpv.py evaluate-all

# Test rapide sur N sujets
docker compose run --rm matplotlib python tpv.py evaluate-all --max-subjects 5

# Évaluation sur un autre dataset : BCI Competition IV-2a (9 sujets, 22 canaux)
docker compose run --rm matplotlib python tpv.py evaluate-all --dataset bci4-2a
docker compose run --rm matplotlib python tpv.py evaluate-all --dataset bci4-2a --max-subjects 3
```

### Arguments disponibles

| Argument | Défaut | Description |
|---|---|---|
| `--subject N` | requis | Numéro du sujet (1-109) |
| `--runs N [N ...]` | requis | Numéros des runs à charger |
| `--model-path` | `model.joblib` | Chemin de sauvegarde/chargement du modèle |
| `--show-raw` | `false` | Affiche les visualisations EEG (raw, filtré, PSD) |
| `--tune` | `false` | Active le GridSearchCV |
| `--window-sec` | `None` | Fenêtre d'augmentation (en secondes) |
| `--overlap` | `0.5` | Chevauchement pour l'augmentation |
| `--use-features` | `false` | Pipeline FeatureExtractor au lieu de CSP |
| `--use-custom-clf` | `false` | **Bonus** : classifieur LDA custom |
| `--use-custom-eigen` | `false` | **Bonus** : eigenvalue decomposition custom |
| `--max-subjects` | `109` | Nombre de sujets pour evaluate-all |
| `--dataset` | `physionet` | Dataset pour evaluate-all : `physionet` ou `bci4-2a` |

---

## 4. Tableau des runs PhysioNet

| Run(s) | Tâche | Type |
|---|---|---|
| 1 | Repos yeux ouverts | Baseline |
| 2 | Repos yeux fermés | Baseline |
| 3, 7, 11 | Mouvement réel main G/D | Exécution motrice |
| **4, 8, 12** | **Imagery main G/D** | **Imagerie motrice** |
| 5, 9, 13 | Mouvement réel poings/pieds | Exécution motrice |
| **6, 10, 14** | **Imagery poings/pieds** | **Imagerie motrice** |

Chaque triplet (ex: runs 4, 8, 12) correspond aux **3 répétitions** de la même tâche.

---

## 5. Fichiers sources

| Fichier | Rôle |
|---|---|
| `tpv.py` | Script principal — CLI (train / predict / evaluate-all) |
| `src/loader.py` | Chargement PhysioNet via MNE + epochage |
| `src/preprocessing.py` | Filtrage passe-bande FIR, filtre notch, visualisation raw/PSD |
| `src/csp_custom.py` | CSP from scratch (BaseEstimator + TransformerMixin) |
| `src/pipeline_model.py` | Construction du Pipeline sklearn + cross_val_score + GridSearch |
| `src/features.py` | FeatureExtractor (PSD + wavelets) — pipeline alternatif |
| `src/stream_simulator.py` | Simulation de flux temps réel pour le predict |
| `src/lda_custom.py` | **Bonus** : classifieur LDA from scratch (BaseEstimator + ClassifierMixin) |
| `src/eigen_custom.py` | **Bonus** : eigenvalue decomposition from scratch (Householder + QR) |
| `dockerfile` | Image Docker Python 3.11 avec dépendances |
| `docker-compose.yml` | Service Docker avec support X11 (affichage graphique) |

---

## 6. Détail technique par module

### `src/preprocessing.py` — Filtrage & Visualisation

- **Filtrage passe-bande (8-30 Hz)** : conserve uniquement les bandes μ (Mu : 8-12 Hz) et β (Beta : 18-30 Hz) pertinentes pour le Motor Imagery. Implémentation MNE avec méthode FIR (`firwin`).
- **Filtre notch (50/100 Hz)** : supprime les interférences de la ligne électrique et ses harmoniques.
- **Visualisation** : `visualize_raw()` affiche le signal brut et la version filtrée ; `visualize_spectrum()` affiche la PSD pour vérifier le filtrage.

### `src/loader.py` — Chargement & Epochage

- **`load_physionet()`** : téléchargement automatique via `mne.datasets.eegbci`, standardisation des noms de canaux (`eegbci.standardize()`), montage `standard_1005`.
- **`make_epochs()`** : 
  - Sélection des événements T1 (classe 1) et T2 (classe 2)
  - Sélection des 21 canaux sensori-moteurs par défaut (`pick_motor=True`) : FC5/3/1/z/2/4/6, C5/3/1/z/2/4/6, CP5/3/1/z/2/4/6
  - Fenêtre temporelle : `tmin=0.5` à `tmax=2.5` par défaut (la commande `train` surcharge à `tmin=0.0, tmax=4.0`)
  - Option d'augmentation par fenêtrage chevauchant (`window_sec` + `overlap`)
  - ⚠️ Le fenêtrage chevauchant crée des epochs non-indépendants — risque de data leakage si utilisé avec cross_val_score. Désactivé par défaut (`window_sec=None`).

### `src/csp_custom.py` — CSP (Common Spatial Patterns)

Implémentation from scratch respectant `BaseEstimator` + `TransformerMixin` sklearn :

1. **`fit(X, y)`** :
   - Calcul de la matrice de covariance par classe (normalisée par la trace pour égaliser l'énergie entre epochs)
   - Shrinkage (régularisation) de la covariance : `cov_reg = (1-λ)·cov + λ·(tr/d)·I`
   - Résolution du problème d'eigenvalues généralisé : `Σ₁·w = λ·(Σ₁+Σ₂)·w`
   - Sélection des `n_left` eigenvectors supérieurs + `n_right` inférieurs → filtres spatiaux

2. **`transform(X)`** :
   - Projection : `X_csp = W^T · X`
   - Extraction (paramètre `log_type`) :
     - `'ratio'` (défaut) : `features = log(var(X_csp) / Σvar(X_csp))`
     - `'var'` : `features = log(var(X_csp))` — utilisé par evaluate-all (FBCSP)
   - Résultat : vecteur `(n_epochs, n_components)`

Le CSP agit donc à la fois comme **réducteur de dimensionnalité spatiale** ET **extracteur de features**.

### `src/pipeline_model.py` — Pipeline & Évaluation

- **`build_pipeline()`** : construit un `Pipeline` sklearn. 3 modes :
  - CSP classique : CSP → StandardScaler → Classifieur
  - **FBCSP** (`use_fbcsp=True`) : `FeatureUnion` de `BandpassCSP` sur plusieurs sous-bandes (mu, low-beta, high-beta) → StandardScaler → Classifieur. Chaque `BandpassCSP` applique un filtre passe-bande + CSP indépendant.
  - FeatureExtractor : PSD + wavelets → StandardScaler → Classifieur
- **`train_and_evaluate()`** : `StratifiedKFold` + `cross_val_score` avec `accuracy`
- **GridSearchCV** : recherche automatique sur `n_components`, `C`, type de classifieur, kernel SVM

### `src/features.py` — FeatureExtractor (pipeline alternatif)

- Common Average Reference (CAR) : soustrait la moyenne spatiale pour réduire le bruit commun
- PSD par bande de fréquence : extraction `welch()` sur 4 sous-bandes `[(8,12), (12,16), (16,25), (25,30)]`
- Wavelets (bonus) : décomposition en ondelettes `db4` → énergie par niveau de décomposition
- Downsampling optionnel

⚠️ Ce pipeline est une **alternative** au CSP. Il ne doit **pas** être placé avant le CSP dans le pipeline — le CSP attend des données temporelles brutes `(n_channels, n_times)`.

### `src/stream_simulator.py` — Simulation temps réel

Simule un flux BCI : les epochs arrivent une par une avec un délai configurable. Le modèle doit prédire en < 2 secondes par epoch.

---

## 7. Bonus implémentés

### Bonus 1 : Classifieur LDA custom (`src/lda_custom.py`)

Fisher's Linear Discriminant Analysis from scratch :

1. Calcul des moyennes de classe μ₁, μ₂
2. Matrice de dispersion intra-classe : `Sw = Σ_c (X_c - μ_c)^T (X_c - μ_c)`
3. Régularisation Ledoit-Wolf : `Sw_reg = (1-λ)·Sw + λ·(tr/d)·I`
4. Vecteur de projection : `w = Sw⁻¹ · (μ₁ - μ₂)` via `np.linalg.solve()`
5. Seuil = point milieu des projections de moyennes
6. `predict_proba()` via sigmoïde pour compatibilité sklearn

Activable avec `--use-custom-clf`.

### Bonus 2 : Eigenvalue decomposition custom (`src/eigen_custom.py`)

Remplacement de `scipy.linalg.eigh` dans le CSP par un algorithme from scratch :

1. **Tridiagonalisation de Householder** : réduit la matrice symétrique en forme tridiagonale
2. **QR iteration avec shifts de Wilkinson** : convergence cubique vers les eigenvalues
3. **Rotations de Givens** : zero-out efficace pour matrices tridiagonales
4. **Réduction Cholesky** pour le problème généralisé `Av = λBv`

Activable avec `--use-custom-eigen`.

### Bonus 3 : Wavelet transform (`src/features.py`)

Décomposition en ondelettes (`db4`, niveau 3) pour capturer l'information temps-fréquence. Intégré dans le `FeatureExtractor`, activable via `--use-features`.

---

## 8. Mode `evaluate-all` — FBCSP + StratifiedKFold

Pour chaque sujet, 4 types d'expériences sont évalués indépendamment.
Chaque type charge ses 3 runs (3 répétitions) ensemble, puis effectue une cross-validation `StratifiedKFold` (k=5, shuffle, seed=42).

| # | Type d'expérience | Runs |
|---|---|---|
| 0 | L/R Fist (execution) | 3, 7, 11 |
| 1 | L/R Fist (imagery) | 4, 8, 12 |
| 2 | Fists/Feet (execution) | 5, 9, 13 |
| 3 | Fists/Feet (imagery) | 6, 10, 14 |

### Configuration optimisée

- Filtre passe-bande large : **4-40 Hz** (le FBCSP fait le découpage en sous-bandes)
- Fenêtre temporelle : **tmin=0.5s, tmax=3.5s** (évite le temps de réaction)
- **21 canaux sensori-moteurs** (`pick_motor=True`)
- **Rejet adaptatif** : seuil 200 µV, fallback si trop d'epochs rejetées
- **FBCSP** (Filter Bank CSP) : 3 sous-bandes (8-12 Hz mu, 12-20 Hz low-beta, 20-30 Hz high-beta) × 4 composantes CSP = **12 features**
- Classifieur : **LDA** (`LinearDiscriminantAnalysis`, shrinkage='auto')

### Per-subject breakdown

À la fin de l'évaluation, un tableau trié affiche chaque sujet du pire au meilleur, avec l'impact des sujets faibles (< 60%, phénomène de "BCI illiteracy") sur la moyenne globale.

Mean accuracy attendue ≥ 60% sur l'ensemble des sujets/expériences.

### Dataset alternatif : BCI Competition IV-2a

Le pipeline supporte également le dataset **BCI Competition IV dataset 2a** (9 sujets, 22 canaux EEG, 250 Hz) via le package **MOABB**. Ce dataset est un benchmark de référence en BCI.

```bash
python tpv.py evaluate-all --dataset bci4-2a
```

Même pipeline FBCSP, mêmes paramètres. Le score obtenu (~81%) est supérieur à PhysioNet (~74%) grâce à une meilleure qualité de signal (250 Hz vs 160 Hz, protocole mieux contrôlé).

---

## 9. Dépendances

```
numpy, scipy, mne, scikit-learn, matplotlib, joblib, PyWavelets, moabb
```

Installées automatiquement via Docker (`src/requirements.txt`).

    