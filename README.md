# Total Perspective Vortex — BCI Motor Imagery Classification

## 1. Présentation et Résumé du Projet

Le projet **Total Perspective Vortex** est une application d'interface cerveau-machine (Brain-Computer Interface - BCI) visant à classifier des signaux électroencéphalographiques (EEG) issus d'imageries motrices (par exemple, imaginer le mouvement de la main gauche par rapport à la main droite). 

En exploitant le jeu de données de référence **PhysioNet EEGBCI** (109 sujets, 64 canaux EEG) ainsi que le dataset **BCI Competition IV-2a**, le système implémente un pipeline complet de traitement : du filtrage temporel et spatial des signaux bruts à la classification par apprentissage automatique. L'objectif principal est de concevoir un pipeline scikit-learn robuste atteignant une précision de classification $\ge 60\%$ (le pipeline optimisé atteint en moyenne ~74% sur PhysioNet et ~81% sur BCI Competition IV-2a).

Le projet se distingue par l'intégration d'algorithmes clés réécrits intégralement *from scratch* (classification LDA et décomposition en valeurs propres généralisées), démontrant la maîtrise des fondements mathématiques sous-jacents aux BCIs.

---

## 2. Techniques et Compétences Acquises

Ce projet couvre un large spectre de compétences scientifiques, mathématiques et d'ingénierie logicielle :

### A. Traitement Numérique du Signal EEG
*   **Filtrage Temporel Physiologique** : Conception et application de filtres passe-bande FIR (Finite Impulse Response) via [src/preprocessing.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/preprocessing.py) dans les bandes spectrales clés de l'imagerie motrice : la bande **$\mu$ (8-12 Hz)** et la bande **$\beta$ (18-30 Hz)**, sièges des phénomènes de désynchronisation/synchronisation liée aux événements (ERD/ERS).
*   **Suppression du Bruit Électrique** : Implémentation d'un filtre notch à 50 Hz (et ses harmoniques à 100 Hz) pour éliminer les interférences électriques industrielles.
*   **Réduction Spatiale de Référence (CAR)** : Application du *Common Average Reference* pour soustraire le potentiel moyen instantané de tous les capteurs afin d'atténuer les bruits globaux.
*   **Analyse Temps-Fréquence** : Estimation de la densité spectrale de puissance (PSD) via la méthode de Welch et extraction de caractéristiques d'énergie à l'aide de la décomposition en ondelettes discrètes (*Discrete Wavelet Transform* avec ondelettes de Daubechies `db4`).

### B. Machine Learning et Modélisation (BCI)
*   **Filtrage Spatial Supervisé (CSP)** : Conception d'un transformateur [CSP (Common Spatial Patterns)](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/csp_custom.py) customisé respectant les interfaces de scikit-learn (`BaseEstimator`, `TransformerMixin`). Le CSP projette les signaux multicanaux de manière à maximiser la variance d'une classe tout en minimisant celle de l'autre.
*   **Filter Bank Common Spatial Patterns (FBCSP)** : Implémentation d'un pipeline FBCSP sous forme de `FeatureUnion` exécutant plusieurs CSP indépendants en parallèle sur différentes sous-bandes de fréquences (mu, low-beta, high-beta).
*   **Régularisation des Matrices de Covariance** : Application de techniques de *shrinkage* (régularisation linéaire) pour fiabiliser l'estimation des matrices de covariance sur des jeux de données comportant peu d'epochs.
*   **Recherche d'Hyperparamètres et Validation Croisée** : Utilisation de `StratifiedKFold` et `GridSearchCV` pour optimiser le nombre de composantes CSP, le paramètre de régularisation et le choix du classifieur.
*   **Analyse de l'analphabétisme BCI (BCI Illiteracy)** : Analyse statistique approfondie de l'impact des sujets non-répondeurs (accuracy < 60%) sur la moyenne globale et per-subject.

### C. Algorithmique et Mathématiques Appliquées (Bonus "From Scratch")
*   **Décomposition Spectrale Customisée** : Implémentation dans [src/eigen_custom.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/eigen_custom.py) d'un algorithme de décomposition en valeurs/vecteurs propres pour matrices symétriques :
    *   **Tridiagonalisation de Householder** pour transformer la matrice symétrique dense en matrice tridiagonale.
    *   **Itérations QR avec décalages de Wilkinson (Wilkinson shifts)** et rotations de Givens pour converger cubiquement vers les valeurs propres.
    *   **Factorisation de Cholesky** pour transformer le problème généralisé $Av = \lambda Bv$ en problème symétrique standard.
*   **Classifieur LDA Customisé** : Développement dans [src/lda_custom.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/lda_custom.py) d'un classifieur d'analyse discriminante linéaire (Fisher's LDA) résolu analytiquement (calcul des moyennes de classes, matrice de dispersion intra-classe, régularisation de covariance par shrinkage et prédictions probabilistes par sigmoïde).

### D. Génie Logiciel et Temps Réel
*   **Simulation de Flux Temps Réel** : Implémentation d'un générateur de streaming d'epochs EEG dans [src/stream_simulator.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/stream_simulator.py), mesurant et garantissant une latence de prédiction inférieure au seuil critique de 2 secondes.
*   **Conteneurisation et Support Graphique** : Configuration d'un environnement Docker complet ([dockerfile](file:///home/jbuan/sgoinfre/total-perspective-vortex/dockerfile) et [docker-compose.yml](file:///home/jbuan/sgoinfre/total-perspective-vortex/docker-compose.yml)) avec transmission du serveur X11 pour permettre la visualisation interactive des signaux EEG et matrices de confusion depuis le conteneur.

---

## 3. Architecture du Projet

```
EEG brut (64 canaux × N temps)
    │
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
Activable via `--use-features` : remplace le CSP par une extraction PSD + ondelettes.
```
Epochs → FeatureExtractor (PSD par bande + ondelettes) → StandardScaler → Classifieur
```

---

## 4. Structure des Fichiers Sources

| Fichier | Rôle |
|---|---|
| [tpv.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/tpv.py) | Point d'entrée principal - CLI (`train`, `predict`, `evaluate-all`). |
| [src/loader.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/loader.py) | Téléchargement automatique de PhysioNet via MNE, normalisation et epochage des signaux. |
| [src/preprocessing.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/preprocessing.py) | Filtrages passe-bande, répulseur (Notch) et visualisations graphiques (raw/PSD). |
| [src/csp_custom.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/csp_custom.py) | Implémentation du transformateur CSP from scratch (compatible scikit-learn). |
| [src/pipeline_model.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/pipeline_model.py) | Assemblage du pipeline sklearn, support FBCSP et GridSearch. |
| [src/features.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/features.py) | Extracteur de caractéristiques alternatif (PSD par Welch, ondelettes discrètes et CAR). |
| [src/stream_simulator.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/stream_simulator.py) | Générateur de streaming d'epochs pour valider les contraintes de latence temps réel. |
| [src/lda_custom.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/lda_custom.py) | **Bonus** : Classifieur Linear Discriminant Analysis codé from scratch. |
| [src/eigen_custom.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/eigen_custom.py) | **Bonus** : Algorithmes d'eigen-décomposition (Householder, QR, Givens, Cholesky) codés from scratch. |
| [dockerfile](file:///home/jbuan/sgoinfre/total-perspective-vortex/dockerfile) | Image Docker Python 3.11 avec l'ensemble des bibliothèques de calcul scientifique requises. |
| [docker-compose.yml](file:///home/jbuan/sgoinfre/total-perspective-vortex/docker-compose.yml) | Service Docker avec redirection X11 pour affichage graphique. |

---

## 5. Commandes Principales

### Construction de l'image Docker
```bash
docker compose build
```

### Entraînement d'un modèle (Train)
```bash
# Entraînement classique (CSP + LogisticRegression) sur un sujet
docker compose run --rm matplotlib python tpv.py train \
  --subject 1 --runs 4 8 12 --model-path model.joblib

# Entraînement avec visualisations graphiques (signaux bruts, filtrés et PSD)
docker compose run --rm matplotlib python tpv.py train \
  --subject 1 --runs 4 8 12 --model-path model.joblib --show-raw

# Entraînement avec les composants réécrits (LDA custom + eigen custom)
docker compose run --rm matplotlib python tpv.py train \
  --subject 1 --runs 4 8 12 --model-path model.joblib \
  --use-custom-clf --use-custom-eigen

# Entraînement avec optimisation des hyperparamètres par GridSearch
docker compose run --rm matplotlib python tpv.py train \
  --subject 1 --runs 4 8 12 --model-path model.joblib --tune
```

### Prédiction en flux simulé (Predict)
*Les runs de test doivent être distincts des runs d'entraînement.*
```bash
docker compose run --rm matplotlib python tpv.py predict \
  --subject 1 --runs 3 7 11 --model-path model.joblib
```

### Évaluation globale (Evaluate-all)
Permet de tester le pipeline FBCSP + LDA sur l'ensemble des sujets et expériences.
```bash
# Évaluation complète sur les 109 sujets de PhysioNet
docker compose run --rm matplotlib python tpv.py evaluate-all

# Test rapide restreint aux 5 premiers sujets
docker compose run --rm matplotlib python tpv.py evaluate-all --max-subjects 5

# Évaluation sur le dataset BCI Competition IV-2a (9 sujets, 22 canaux)
docker compose run --rm matplotlib python tpv.py evaluate-all --dataset bci4-2a
```

---

## 6. Détails Techniques par Module

### [src/preprocessing.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/preprocessing.py) — Filtrage & Visualisation
*   **Filtrage passe-bande (8-30 Hz)** : Conserve uniquement les bandes physiologiques d'intérêt pour le Motor Imagery ($\mu$ et $\beta$). Implémentation avec filtre FIR MNE (`firwin`).
*   **Filtre notch (50/100 Hz)** : Élimine la fréquence du réseau électrique alternatif et son premier harmonique.
*   **Visualisation** : Outils pour afficher le signal temporel brut/filtré et le spectre de densité de puissance (PSD).

### [src/loader.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/loader.py) — Chargement & Epochage
*   **Standardisation** : Téléchargement automatique et application d'un montage standard des électrodes (`standard_1005`).
*   **Sélection de canaux** : Filtrage automatique pour restreindre le traitement aux **21 canaux sensori-moteurs** par défaut (FC5/3/1/z/2/4/6, C5/3/1/z/2/4/6, CP5/3/1/z/2/4/6), optimisant le rapport signal/bruit pour l'imagerie motrice.
*   **Augmentation de données** : Option d'augmentation par fenêtrage temporel chevauchant (attention aux risques de fuite de données si appliqué avant découpe train/test).

### [src/csp_custom.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/csp_custom.py) — Common Spatial Patterns
*   **Calcul des Covariances** : Estimation normalisée par la trace pour équilibrer l'importance de chaque epoch.
*   **Régularisation (Shrinkage)** : Mélange linéaire de la matrice de covariance estimée avec une matrice diagonale uniforme : $\Sigma_{\text{reg}} = (1-\lambda)\Sigma + \lambda \frac{\text{Tr}(\Sigma)}{d} I$.
*   **Extraction de caractéristiques** : Projection spatiale et extraction des logarithmes des variances des signaux filtrés spatialement.

### [src/features.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/features.py) — Extraction alternative
*   **PSD par bande de fréquence** : Calcul de Welch sur les bandes d'ondes cérébrales standard.
*   **Ondelettes discrètes** : Décomposition en cascade (ondelette Daubechies 4 à 3 niveaux de détails) permettant d'extraire l'énergie et la dispersion à différentes résolutions temporelles et fréquentielles.

---

## 7. Fonctionnement des Algorithmes "From Scratch" (Bonus)

### Décomposition en valeurs propres généralisées ([src/eigen_custom.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/eigen_custom.py))
Pour résoudre le problème spatial $A v = \lambda B v$ où $A$ et $B$ sont des matrices de covariance symétriques réelles :
1.  **Factorisation de Cholesky** : On décompose $B = L L^T$.
2.  **Transformation standard** : On définit $C = L^{-1} A L^{-T}$, ramenant le problème à un problème classique de recherche de valeurs propres symétriques $C u = \lambda u$ (où $v = L^{-T} u$).
3.  **Tridiagonalisation de Householder** : Par réflexions successives de Householder, on annule les éléments hors de la sous-diagonale pour transformer $C$ en matrice tridiagonale symétrique $T = Q_0^T C Q_0$.
4.  **Algorithme QR avec Décalages de Wilkinson** : On effectue des factorisations QR successives sur $T$. L'utilisation de décalages de Wilkinson accélère grandement la convergence vers la forme diagonale (valeurs propres sur la diagonale).
5.  **Rotations de Givens** : Implémentées pour effectuer les étapes de factorisation QR sur la matrice tridiagonale de manière efficace (complexité linéaire par itération).

### Analyse Discriminante Linéaire ([src/lda_custom.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/lda_custom.py))
1.  **Centres de classes** : Calcul des centroïdes de caractéristiques $\mu_1$ et $\mu_2$ pour les deux classes.
2.  **Matrice de dispersion intra-classe ($S_w$)** : Somme pondérée des matrices de covariance de chaque classe.
3.  **Régularisation Ledoit-Wolf** : Ajustement de $S_w$ pour garantir sa non-singularité.
4.  **Vecteur de projection ($w$)** : Résolution du système linéaire $S_{w} w = \mu_1 - \mu_2$.
5.  **Classification** : Projection des nouveaux points et comparaison au seuil médian : $x \mapsto w^T x - \text{seuil}$.

---

## 8. Mode Évaluation Globale (`evaluate-all`)

Le script évalue chaque sujet sur **4 tâches expérimentales distinctes** à l'aide d'une validation croisée à 5 folds (`StratifiedKFold`) :

| Expérience | Tâche associée | Runs PhysioNet |
|---|---|---|
| **0** | Exécution motrice (Mouvement réel poing gauche/droit) | 3, 7, 11 |
| **1** | Imagerie motrice (Imaginer le mouvement poing gauche/droit) | 4, 8, 12 |
| **2** | Exécution motrice (Mouvement réel poings/pieds) | 5, 9, 13 |
| **3** | Imagerie motrice (Imaginer le mouvement poings/pieds) | 6, 10, 14 |

### Pipeline de Classification Optimisé
Pour obtenir des performances maximales et stables, le pipeline configuré par défaut dans `evaluate-all` utilise :
*   Un filtrage large **4-40 Hz** en entrée.
*   L'extraction multi-bandes **FBCSP** (bandes 8-12 Hz, 12-20 Hz et 20-30 Hz).
*   Un rejet d'artefacts adaptatif éliminant les signaux présentant des amplitudes aberrantes ($> 200\,\mu\text{V}$).
*   Une classification par **LDA avec shrinkage automatique**.

---

## 9. Dépendances requises

Les dépendances du projet sont listées dans [src/requirements.txt](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/requirements.txt) et installées automatiquement dans le conteneur Docker :
*   `numpy` & `scipy` (Calculs matriciels et scientifiques)
*   `mne` & `moabb` (Chargement et manipulation des données EEG)
*   `scikit-learn` (Pipelines ML, scalers, métriques)
*   `matplotlib` (Affichage graphique)
*   `joblib` (Sauvegarde de modèles)
*   `PyWavelets` (Décomposition en ondelettes)