ENTRÉE : EEG brut du sujet (64 capteurs × 5 min de signal)
    │
    ▼
FILTRAGE 8-30 Hz (identique dans les 3 variantes)
    │
    ▼
EPOCHAGE (identique dans les 3 variantes)
    ~45 epochs étiquetés T1 ou T2
    │
    ▼
════════════════════════════════════════════════════════════════
             VARIANTE 1 : STANDARD (par défaut)
════════════════════════════════════════════════════════════════
    │
CSP.fit() — uses scipy.linalg.eigh (optimisé en C/Fortran)
    │
    ├── Covariance par classe → normalisation trace → shrinkage
    │
    └── scipy.eigh(Σ₁, Σ₁+Σ₂)
        → Boîte noire : on donne deux matrices, ça retourne
          les eigenvalues et eigenvectors directement
        → Très rapide (~0.001s pour 64×64)
    │
CSP.transform() → 6 log-variances par epoch
    │
StandardScaler
    │
LogisticRegression (classifieur par défaut)
    │
    ├── ENTRAÎNEMENT (gradient descent) :
    │   │
    │   │  Initialise : w = [0, 0, 0, 0, 0, 0], b = 0
    │   │
    │   │  Pour chaque itération (jusqu'à 1000 max) :
    │   │    │
    │   │    ├── Pour chaque epoch, calcule :
    │   │    │     score = w₁·f₁ + w₂·f₂ + ... + w₆·f₆ + b
    │   │    │     probabilité = 1 / (1 + e^(-score))     ← sigmoïde
    │   │    │
    │   │    ├── Calcule l'erreur (log-loss) :
    │   │    │     loss = -Σ [y·log(p) + (1-y)·log(1-p)]
    │   │    │     "Combien les probabilités prédites diffèrent des vrais labels"
    │   │    │
    │   │    ├── Calcule le gradient :
    │   │    │     ∂loss/∂w₁ = Σ (p - y) · f₁
    │   │    │     "Dans quelle direction faut-il modifier w₁ pour réduire l'erreur"
    │   │    │
    │   │    └── Met à jour les poids :
    │   │          w₁ ← w₁ - α · ∂loss/∂w₁
    │   │          (α = learning rate, petit pas dans la bonne direction)
    │   │
    │   │  Résultat : poids w et biais b optimisés
    │   │
    │   │  Le paramètre C (régularisation) contrôle :
    │   │    C grand → le modèle suit les données à fond (risque overfitting)
    │   │    C petit → le modèle reste "prudent" (poids plus petits)
    │   │    Formule : loss_total = loss + (1/C) · Σ w²
    │   │
    │
    ├── PRÉDICTION :
    │     score = w·features + b
    │     p = sigmoïde(score)
    │     si p > 0.5 → classe 1, sinon → classe 2
    │
    ▼
SORTIE : prédiction + probabilité de confiance

════════════════════════════════════════════════════════════════
       VARIANTE 2 : BONUS --use-custom-eigen --use-custom-clf
════════════════════════════════════════════════════════════════
    │
CSP.fit() — uses eigen_custom.py (from scratch en Python)
    │
    ├── Covariance par classe → normalisation trace → shrinkage
    │
    └── eigh_generalized_custom(Σ₁, Σ₁+Σ₂)
        │
        ├── Cholesky : B = L·Lᵀ  (décompose B en "racine carrée")
        │   Transforme Av=λBv en problème standard Cv=λv
        │
        ├── Householder : C (64×64 pleine) → T (64×64 tridiagonale)
        │   COÛT : O(n³) = 262k opérations, UNE FOIS
        │
        ├── QR iteration + Wilkinson shift :
        │   T (tridiagonale) → D (diagonale)
        │   ~3 itérations par eigenvalue × 64 eigenvalues
        │   Chaque itération : rotation de Givens (O(n))
        │
        └── Back-transform : v = L⁻ᵀ · u
            (reconvertit les eigenvectors au problème original)
        │
        → Même résultat que scipy, mais ~50× plus lent (Python vs C)
        → Intérêt : montrer qu'on comprend l'algorithme sous le capot
    │
CSP.transform() → 6 log-variances par epoch (identique)
    │
StandardScaler (identique)
    │
LDAClassifier custom (au lieu de LogisticRegression)
    │
    ├── ENTRAÎNEMENT (solution analytique, PAS de gradient) :
    │   │
    │   │  1. Calcule les centres de chaque classe :
    │   │       μ₁ = moyenne des features de tous les epochs classe 1
    │   │       μ₂ = moyenne des features de tous les epochs classe 2
    │   │
    │   │  2. Calcule la dispersion intra-classe Sw :
    │   │       "Combien les epochs s'écartent de leur centre de classe"
    │   │       Sw = (X₁-μ₁)ᵀ(X₁-μ₁) + (X₂-μ₂)ᵀ(X₂-μ₂)
    │   │
    │   │  3. Régularise (shrinkage) :
    │   │       Sw_reg = 99% × Sw + 1% × Identité
    │   │
    │   │  4. Calcule le vecteur de projection :
    │   │       w = Sw⁻¹ · (μ₁ - μ₂)
    │   │       "La direction qui maximise la distance entre classes
    │   │        tout en minimisant la dispersion à l'intérieur"
    │   │
    │   │  5. Seuil = milieu entre les projections des deux centres :
    │   │       threshold = (w·μ₁ + w·μ₂) / 2
    │   │
    │   │  → TERMINÉ en 1 passe. Pas de boucle, pas de gradient.
    │   │
    │
    ├── PRÉDICTION :
    │     score = w · features - threshold
    │     si score > 0 → classe de haute projection
    │     si score ≤ 0 → classe de basse projection
    │
    │     Probabilité (optionnel) :
    │       p = sigmoïde(score) → compatibilité sklearn
    │
    ▼
SORTIE : prédiction + probabilité

════════════════════════════════════════════════════════════════
          VARIANTE 3 : --use-features (pipeline alternatif)
════════════════════════════════════════════════════════════════
    │
PAS de CSP ! Remplacé par FeatureExtractor :
    │
    ├── Common Average Reference (CAR) :
    │     Soustrait la moyenne de tous les capteurs à chaque instant
    │     "Enlève le bruit commun à toute la tête"
    │
    ├── PSD par bande (Welch) :
    │     Pour chaque capteur × chaque sous-bande :
    │       (8-12 Hz), (12-16 Hz), (16-25 Hz), (25-30 Hz)
    │     Calcule la puissance spectrale moyenne → log1p
    │     Résultat : 64 capteurs × 4 bandes = 256 features PSD
    │
    └── Wavelets (bonus, activé par défaut) :
          Pour chaque capteur :
            Décompose le signal en ondelettes db4 (3 niveaux)
            Extrait l'énergie (mean + std) par niveau
            Résultat : 64 capteurs × 8 features wavelet = 512 features
          │
          Total : 256 + 512 = 768 features par epoch
    │
StandardScaler (normalise les 768 features)
    │
Classifieur (LogReg ou LDA)
    │
    ▼
SORTIE : prédiction

    ⚠️ Cette variante est souvent MOINS performante que le CSP car :
    - Beaucoup plus de features (768 vs 6) → risque d'overfitting
    - Les features PSD/wavelet ne sont pas optimisées pour la discrimination
      entre classes (contrairement au CSP qui est supervisé)
    - Utile comme comparaison ou comme bonus