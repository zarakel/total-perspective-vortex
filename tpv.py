# tpv.py
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.loader import load_physionet, make_epochs
from src.preprocessing import bandpass_filter, visualize_raw, visualize_spectrum
from src.pipeline_model import build_pipeline, train_and_evaluate
from src.stream_simulator import stream_epochs

def plot_cv_scores(scores):
    """Affiche les scores de validation croisée."""
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(scores) + 1), scores, color='skyblue')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title(f'Cross-validation scores\nMean = {np.mean(scores):.3f}')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show(block=True)
    input("Appuie sur Entrée pour fermer les graphiques...")

def plot_confusion(y_true, y_pred, title="Confusion matrix"):
    """Affiche une matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.show(block=True)
    input("Appuie sur Entrée pour fermer les graphiques...")

def cmd_train(args):
    raw = load_physionet(args.subject, args.runs)
    if args.show_raw:
        visualize_raw(raw, title="Raw EEG (before filtering)")
    raw = bandpass_filter(raw, l_freq=8.0, h_freq=30.0)
    if args.show_raw:
        visualize_raw(raw, title="Filtered EEG (8-30 Hz bandpass)")
        visualize_spectrum(raw, title="Power Spectral Density (after filtering)")

    print(f"Using window_sec={args.window_sec} overlap={args.overlap}")
    epochs = make_epochs(raw, tmin=0.0, tmax=4.0, window_sec=args.window_sec, overlap=args.overlap)
    X = epochs.get_data()
    y = epochs.events[:, -1]

    print(f"Shape de X : {X.shape}, Shape de y : {y.shape}")

    # build pipeline
    reducer_params = {'csp': {'n_components': 8}}
    pipe = build_pipeline(
        sfreq=int(raw.info['sfreq']),
        reducer='csp',
        reducer_params=reducer_params,
        memory=args.memory,
        use_features=getattr(args, 'use_features', False),
        use_custom_clf=getattr(args, 'use_custom_clf', False),
        use_custom_eigen=getattr(args, 'use_custom_eigen', False)
    )

    # ✅ Validation croisée + visualisation
    mean_score, scores = train_and_evaluate(pipe, X, y, cv=5)
    print("cross_val_score:", mean_score, scores)
    plot_cv_scores(scores)

    if getattr(args, "tune", False):
        best_score, scores, gs = train_and_evaluate(pipe, X, y, cv=5, tune=True)
        print("GridSearch best score:", best_score)
        pipe = gs.best_estimator_
    else:
        mean_score, scores = train_and_evaluate(pipe, X, y, cv=5)
        print("cross_val_score:", mean_score, scores)    

    # ✅ Fit complet sur le jeu d'entraînement
    pipe.fit(X, y)

    # ✅ Évaluation sur le même set (simple aperçu visuel)
    y_pred = pipe.predict(X)
    plot_confusion(y, y_pred, title="Confusion matrix (train set)")

    # ✅ Sauvegarde du modèle
    joblib.dump(pipe, args.model_path)
    print("Model saved to", args.model_path)

def cmd_predict(args):
    pipe = joblib.load(args.model_path)
    raw = load_physionet(args.subject, args.runs)
    raw = bandpass_filter(raw, l_freq=8.0, h_freq=30.0)
    epochs = make_epochs(raw, tmin=0.0, tmax=4.0, window_sec=args.window_sec, overlap=args.overlap)
    X = epochs.get_data()
    labels = epochs.events[:, -1]

    correct = 0
    total = 0
    preds = []
    latencies = []

    for epoch, y_true, t_end in stream_epochs(X, labels, delay_sim=0.1):
        t_start_pred = time.time()
        pred = pipe.predict(epoch[np.newaxis, ...])
        t_pred = time.time()
        latency = t_pred - t_end

        preds.append(pred[0])
        latencies.append(latency)
        total += 1
        if pred[0] == y_true:
            correct += 1

        print(f"epoch {total:02d}: pred={pred[0]} truth={y_true} latency={latency:.3f}s")
        if latency > 2.0:
            print("⚠️ WARNING: prediction latency exceeded 2s!")

    acc = correct / total
    print(f"✅ Accuracy: {acc:.3f}")

    # ✅ Visualisation post-prédiction
    plt.figure(figsize=(6, 3))
    plt.plot(latencies, marker='o')
    plt.xlabel("Epoch index")
    plt.ylabel("Latency (s)")
    plt.title("Prediction latency over time")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show(block=True)
    input("Appuie sur Entrée pour fermer les graphiques...")

    plot_confusion(labels[:len(preds)], preds, title="Confusion matrix (prediction stream)")


# ── 4 experiment types (matching checklist: 4 types of experiment runs) ──
# For each type, load all 3 runs together and cross-validate (StratifiedKFold k=5).
# The 4 types are:
#   Type 0: L/R fist execution (runs 3, 7, 11)
#   Type 1: L/R fist imagery   (runs 4, 8, 12)
#   Type 2: Fists/Feet execution (runs 5, 9, 13)
#   Type 3: Fists/Feet imagery   (runs 6, 10, 14)
EXPERIMENT_TYPES = [
    {'name': 'L/R Fist (execution)',   'runs': [3, 7, 11]},
    {'name': 'L/R Fist (imagery)',     'runs': [4, 8, 12]},
    {'name': 'Fists/Feet (execution)', 'runs': [5, 9, 13]},
    {'name': 'Fists/Feet (imagery)',   'runs': [6, 10, 14]},
]


def _evaluate_subject_experiment_type(subject, exp_type):
    """For one experiment type (3 runs = 3 repetitions):
    Load all 3 runs, apply filter bank CSP (FBCSP), run cross_val_score.
    Returns the mean cross-validated accuracy.
    
    Uses optimized configuration:
    - Broadband 4-40 Hz preprocessing (FBCSP does sub-band filtering)
    - 21 sensorimotor channels
    - tmin=0.5s, tmax=3.5s (skip reaction time)
    - FBCSP with 3 sub-bands: mu (8-12), low-beta (12-20), high-beta (20-30)
    - 4 CSP components per band (12 features total)
    - Adaptive 200µV epoch rejection (fallback to no rejection)
    - LDA classifier with auto shrinkage
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    runs = exp_type['runs']

    try:
        raw = load_physionet(subject, runs)
        # Broadband filter for FBCSP (sub-band filtering done inside pipeline)
        raw = bandpass_filter(raw, 4.0, 40.0)
        epochs = make_epochs(raw, tmin=0.5, tmax=3.5, pick_motor=True)

        # Adaptive artifact rejection: try 200µV, fall back if too few epochs
        epochs_rej = epochs.copy().drop_bad(reject=dict(eeg=200e-6))
        X_rej = epochs_rej.get_data()
        y_rej = epochs_rej.events[:, -1]
        if (len(np.unique(y_rej)) >= 2 and len(y_rej) >= 10
                and min(np.bincount(y_rej - y_rej.min())) >= 2):
            X, y = X_rej, y_rej
        else:
            X = epochs.get_data()
            y = epochs.events[:, -1]

        if len(np.unique(y)) < 2 or len(y) < 6:
            return None

        pipe = build_pipeline(
            sfreq=int(raw.info['sfreq']),
            reducer='csp',
            reducer_params={'csp': {'n_components': 4, 'shrink': 0.03, 'log_type': 'var'}},
            classifier=LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
            use_fbcsp=True,
            fbcsp_bands=[(8, 12), (12, 20), (20, 30)],
        )
        n_cv = min(5, min(np.bincount(y - y.min())))
        if n_cv < 2:
            return None
        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
        return scores.mean()
    except Exception as e:
        print(f"  [WARN] subject {subject:03d} exp '{exp_type['name']}': {e}")
        return None


def cmd_evaluate_all(args):
    """Evaluate all subjects on 4 experiment types, report mean accuracy."""
    import warnings
    warnings.filterwarnings('ignore')

    n_subjects = getattr(args, 'max_subjects', 109)
    type_accs = {i: [] for i in range(len(EXPERIMENT_TYPES))}

    for subj in range(1, n_subjects + 1):
        for ti, exp_type in enumerate(EXPERIMENT_TYPES):
            acc = _evaluate_subject_experiment_type(subj, exp_type)
            if acc is not None:
                type_accs[ti].append(acc)
                print(f"experiment {ti}: subject {subj:03d}: accuracy = {acc:.4f}")

    print("\nMean accuracy of the four different experiments for all subjects:")
    type_means = []
    for ti in range(len(EXPERIMENT_TYPES)):
        if type_accs[ti]:
            m = np.mean(type_accs[ti])
            type_means.append(m)
            print(f"experiment {ti} ({EXPERIMENT_TYPES[ti]['name']}): accuracy = {m:.4f}")
        else:
            print(f"experiment {ti}: no valid results")
    if type_means:
        global_mean = np.mean(type_means)
        print(f"\nMean accuracy of {len(type_means)} experiments: {global_mean:.4f}")
        if global_mean >= 0.75:
            print(f"Score >= 75% ({global_mean*100:.1f}%)")
            bonus = int((global_mean - 0.75) / 0.03)
            if bonus > 0:
                print(f"Bonus points: +{bonus} (for {(global_mean - 0.75)*100:.1f}% over 75%)")
        else:
            print(f"Score < 75% ({global_mean*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    ptrain = sub.add_parser('train')
    ptrain.add_argument('--subject', type=int, required=True)
    ptrain.add_argument('--runs', type=int, nargs='+', required=True,
                        help='Runs used for training (use distinct runs for testing)')
    ptrain.add_argument('--model-path', default='model.joblib')
    ptrain.add_argument('--show-raw', action='store_true')
    ptrain.add_argument('--tune', action='store_true')
    ptrain.add_argument('--window-sec', type=float, default=None)
    ptrain.add_argument('--overlap', type=float, default=0.5)
    ptrain.add_argument('--memory', type=str, default=None)
    ptrain.add_argument('--downsample', type=int, default=None)
    ptrain.add_argument('--use-features', action='store_true',
                        help='If set, use FeatureExtractor-based pipeline instead of CSP.')
    ptrain.add_argument('--use-custom-clf', action='store_true',
                        help='Bonus: use custom LDA classifier instead of LogisticRegression.')
    ptrain.add_argument('--use-custom-eigen', action='store_true',
                        help='Bonus: use custom eigenvalue decomposition in CSP.')

    ppredict = sub.add_parser('predict')
    ppredict.add_argument('--subject', type=int, required=True)
    ppredict.add_argument('--runs', type=int, nargs='+', required=True,
                        help='Runs used for prediction/evaluation (should be different from train runs)')
    ppredict.add_argument('--model-path', default='model.joblib')
    ppredict.add_argument('--window-sec', type=float, default=None)
    ppredict.add_argument('--overlap', type=float, default=0.5)

    pevalall = sub.add_parser('evaluate-all',
                              help='Evaluate all 109 subjects on 4 experiment types')
    pevalall.add_argument('--max-subjects', type=int, default=109,
                          help='Number of subjects to evaluate (default: all 109)')

    args = parser.parse_args()
    if args.cmd == 'train':
        cmd_train(args)
    elif args.cmd == 'predict':
        cmd_predict(args)
    elif args.cmd == 'evaluate-all':
        cmd_evaluate_all(args)
    else:
        parser.print_help()
