build cmd :
    docker compose build

train cmd : # subject -> Differents sujets de 1 a X #runs -> différents extraits de sujets numéroté  
    docker compose run --rm matplotlib python tpv.py train --subject 5 --runs 3 5 2 6 --model-path qd_model.joblib --show-raw

predict cmd : # subject -> Differents sujets de 1 a X #runs -> différents extraits de sujets numérotés
    docker compose run --rm matplotlib python tpv.py predict --subject 5 --runs 3 5 2 6  --model-path qd_model.joblib

* **`Preprocessing`**

1. Filtrage Passe-Bande

    Bandes de Fréquences (8-30 Hz) : Ce choix est très pertinent pour les BCI de type Motor Imagery (MI), car il couvre les bandes μ (Mu : 8-12 Hz) et β (Beta : ≈ 18-26 Hz), qui présentent l'activité de désynchronisation liée au mouvement.

    Implémentation : L'utilisation de mne.filter avec firwin est standard et robuste.

2. Filtre Coupe-Bande

    C'est essentiel pour supprimer les interférences de la ligne électrique (50 Hz en Europe, 60 Hz aux USA) et ses harmoniques (100 Hz, etc.).

*`Potentiel Problème dans le Pré-traitement`* :

    Le seul point manquant ici est la Sélection des Canaux.

    Pour le Motor Imagery (MI), les canaux sur le cortex sensorimoteur sont les plus importants (e.g., C3, Cz, C4 du système 10-20). Si vous traitez tous les canaux (y compris les canaux oculaires, frontaux, et occipitaux moins pertinents), vous introduisez beaucoup de bruit non informatif qui peut diluer le signal utile et nuire à l'efficacité du CSP.

    Recommandation : Assurez-vous que votre fonction de chargement (loader.py) ou une étape supplémentaire dans le pipeline sélectionne un sous-ensemble de canaux pertinents avant l'extraction des caractéristiques.

* **`Features`** :

    1. Common Average Reference (CAR)

    Python

    if self.use_car:
        ep = ep - np.mean(ep, axis=0, keepdims=True)

        Pertinence : Le CAR est un excellent choix de pré-traitement spatial qui améliore souvent le rapport signal/bruit et la performance CSP en réduisant le bruit commun à tous les canaux. C'est un bon ajout.

    2. Downsampling (Sous-échantillonnage)

    Python

    # ... decimate() code

        Pertinence : Le sous-échantillonnage peut accélérer le traitement et potentiellement réduire la dimensionnalité si la fréquence d'échantillonnage originale est très élevée, mais il peut aussi dégrader la qualité des estimations de puissance si le facteur est trop agressif. Ce n'est probablement pas la cause de l'accuracy de 0.5, mais il faut s'assurer qu'il est désactivé ou bien réglé.

    3. Extraction de Puissance Spectrale (PSD)

    Python

    self.bands = bands or [(8, 12), (12, 16), (16, 25), (25, 30)]
    # ...
    f, Pxx = welch(ch, fs=sfreq, nperseg=min(256, len(ch)))
    # ...
    val = np.mean(Pxx[mask]) if np.any(mask) else 0.0
    band_powers.append(np.log1p(val))

        Bandes : Les bandes sont bien choisies (Alpha/Mu, SMR, Beta), mais le CSP est souvent plus efficace si on lui donne une bande de fréquence unique et large (e.g., 8-30 Hz) ou s'il est utilisé en cascade (CSP sur la bande 8-30 Hz, puis extraction de log-variance sur les sous-bandes). Ici, vous extrayez des caractéristiques PSD pour 4 sous-bandes avant la réduction de dimensionnalité.

        Conflit potentiel avec CSP : Si vous utilisez FeatureExtractor avant votre étape CSP (dans le Pipeline), vous donnez des caractéristiques déjà extraites au lieu des données brutes filtrées, ce qui pourrait empêcher le CSP de fonctionner correctement, car le CSP s'attend à des données de la forme (canaux×temps).

            Si votre pipeline est : (FeatureExtractor) -> CSP : C'est presque certainement la cause de la basse performance. Le CSP doit être appliqué directement sur les données temporelles filtrées.

            Si votre pipeline est : (Preprocessing/Filtering) -> CSP -> (FeatureExtractor) : C'est la bonne approche. Dans ce cas, ce FeatureExtractor ne devrait pas être dans votre pipeline principal, à moins que vous ne l'utilisiez pour des tests sans CSP, car le CSP produit déjà les meilleures caractéristiques à partir de la log-variance de ses composantes projetées.

    4. Extraction d'Ondelettes

    Python

    if self.use_wavelet:
        # ... extraction d'énergies par ondelettes

        Dimensionalité : Ajouter des caractéristiques d'ondelettes (en plus de la PSD) augmente considérablement la dimensionnalité de votre vecteur de caractéristiques avant la classification.

        Règle BCI : Pour un premier essai, garder le pipeline le plus simple possible est souvent le meilleur : Données Filtrées -> CSP (Log-Variance) -> Classifieur (LDA). Ajouter les ondelettes est un Bonus (comme mentionné dans le sujet) et pourrait nuire aux performances si la régularisation du classifieur n'est pas suffisante ou si ces caractéristiques ne sont pas aussi discriminantes que la puissance spectrale optimisée par CSP.

**`loader.py`**

    1. Chargement et Pré-traitement de Base (load_physionet)

        Source de Données : L'utilisation de mne.datasets.eegbci est la méthode standard. Le sujet demande l'utilisation de PhysioNet, donc c'est parfait.

        Montage : La tentative de raw.set_montage('standard_1005') est bonne, mais pas toujours strictement nécessaire si vous travaillez uniquement avec des indices de canaux.

    2. Épochage et Fenêtrage (make_epochs)

    Python

    event_id = {k: v for k, v in event_id.items() if k in ["T1", "T2"]}
    # ...
    tmin=0.0, tmax=4.0

        Classes : La sélection des événements T1 (Mouvement Imaginé de la main gauche/pied) et T2 (Mouvement Imaginé de la main droite/pied) est correcte pour une tâche de classification binaire MI.

        Fenêtre Temporelle : L'utilisation de tmax=4.0 (soit 4 secondes) est une fenêtre d'analyse large. Le signal MI le plus discriminant (désynchronisation du rythme μ) se produit généralement dans les ≈ 0.5s à 2.5s suivant le signal de début. Inclure le temps de repos ou la fin de l'événement (jusqu'à 4s) peut diluer le signal utile et nuire à la performance CSP.

    🚨 Problème Potentiel Majeur : Augmentation/Fenêtrage par Chevauchement

    Python

    # Si pas d'augmentation demandée, retourne les epochs MNE classiques
    if window_sec is None:
        return base_epochs
    # ...
    # Code pour créer de nouvelles époques chevauchantes (augmentation de données)

    Votre fonction make_epochs implémente une technique d'augmentation de données où une seule époque est découpée en segments plus courts et chevauchants.

        Avantage : Augmente le nombre d'échantillons d'entraînement (N) pour le classifieur.

        Inconvénient : Ces segments ne sont pas indépendants. Ils partagent une grande partie des données temporelles. Si vous utilisez ces échantillons dépendants dans votre cross_val_score, vous risquez un "Data Leakage" (Fuite de Données). Le modèle est testé sur des données trop similaires à celles qu'il a apprises, ce qui gonfle artificiellement le score de validation croisée sans refléter la performance sur des données réellement nouvelles.

        Recommandation : Pour un premier objectif d'accuracy (0.60), désactivez le fenêtrage chevauchant (window_sec=None) et utilisez les époques classiques. Si vous l'utilisez, assurez-vous que votre split de validation croisée (V.1.4) sépare les époques originales avant le fenêtrage, ou que vous testez sur un sujet/une run complètement indépendant(e).
    
**`pipeline_model.py`** 

    1. Architecture du Pipeline (build_pipeline)

    Python

    if use_features:
        # Feature-based approach
        steps.append(('fe', FeatureExtractor(sfreq=sfreq, **reducer_params.get('fe', {}))))
        steps.append(('scaler', StandardScaler()))
        steps.append(('clf', classifier))
    else:
        # CSP-based approach (expects raw epochs input)
        # ...
        steps.append(('reducer', CSP(**reducer_params.get('csp', {}))))
        steps.append(('scaler', StandardScaler()))
        steps.append(('clf', classifier))

        Clarté : L'architecture qui distingue l'approche CSP-based (avec réducteur sur les données temporelles) et l'approche Feature-based (avec FeatureExtractor sur les données temporelles) est excellente et répond à l'exigence Pipeline de sklearn.

    🚨 Problème Probable : Ordre des étapes en Mode CSP

    Comme discuté précédemment, l'étape CSP (Common Spatial Pattern) ne produit pas directement un vecteur de caractéristiques pour le classifieur. Le CSP est un transformateur spatial, qui, dans votre implémentation custom (CSP dans csp_custom.py), doit effectuer deux choses :

        Calculer et appliquer la matrice de projection spatiale (W).

        Extraire les caractéristiques à partir des signaux projetés (Log-Variance).

    Si votre classe CSP ne fait que la projection spatiale, l'étape suivante (StandardScaler) et le classifieur recevront des données de la forme (n_eˊpoques,n_composantes_CSP,n_temps), ce qui n'est pas le format attendu par la plupart des classifieurs ((n_samples, n_features)).

    La classe CSP doit retourner le vecteur de caractéristiques (Log-Variance) :
    Xfeatures​=log(∑Var(XCSP​)Var(XCSP​)​)∈RNepochs​×Mcomponents​

    2. Évaluation (train_and_evaluate)

        Validation Croisée : L'utilisation de StratifiedKFold et cross_val_score est correcte et répond aux exigences de la section V.1.4.

        Métriques : L'utilisation de scoring='balanced_accuracy' est judicieuse car elle est robuste aux déséquilibres de classes, fréquents en BCI.


**`csp_custom.py`** :

    La méthode fit calcule les filtres spatiaux W (stockés dans self.filters_).

        La méthode transform applique ces filtres et retourne la Log-Variance (Log-Power), qui est la caractéristique discriminante à donner au classifieur.
        Caracteˊristiques=log(∑Var(XCSP​)Var(XCSP​)​)

    Conclusion sur le Pipeline : Votre pipeline CSP-based (raw -> CSP -> scaler -> clf) est correctement structuré grâce au fait que votre classe CSP agit à la fois comme réducteur de dimensionnalité spatiale et extracteur de caractéristiques (Log-Variance).

    