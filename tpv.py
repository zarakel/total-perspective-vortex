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
# For each type, load all 3 runs together and cross-validate (StratifiedKFold k=3).
# The 4 types are:
#   Type 1: L/R fist execution (runs 3, 7, 11)
#   Type 2: L/R fist imagery   (runs 4, 8, 12)
#   Type 3: Fists/Feet execution (runs 5, 9, 13)
#   Type 4: Fists/Feet imagery   (runs 6, 10, 14)
EXPERIMENT_TYPES = [
    {'name': 'L/R Fist (execution)',   'runs': [3, 7, 11]},
    {'name': 'L/R Fist (imagery)',     'runs': [4, 8, 12]},
    {'name': 'Fists/Feet (execution)', 'runs': [5, 9, 13]},
    {'name': 'Fists/Feet (imagery)',   'runs': [6, 10, 14]},
]


def _evaluate_subject_experiment_type(subject, exp_type):
    """For one experiment type (3 runs = 3 repetitions):
    Load all 3 runs, run cross_val_score (StratifiedKFold, k=3).
    Returns the mean cross-validated accuracy."""
    from src.loader import load_physionet, make_epochs
    from src.preprocessing import bandpass_filter
    from src.pipeline_model import build_pipeline, train_and_evaluate

    runs = exp_type['runs']  # e.g. [4, 8, 12]

    try:
        raw = load_physionet(subject, runs)
        raw = bandpass_filter(raw, 8.0, 30.0)
        epochs = make_epochs(raw, tmin=0.5, tmax=2.5)
        X = epochs.get_data()
        y = epochs.events[:, -1]

        if len(np.unique(y)) < 2:
            return None

        pipe = build_pipeline(
            sfreq=int(raw.info['sfreq']),
            reducer='csp',
            reducer_params={'csp': {'n_components': 8}}
        )
        mean_score, scores = train_and_evaluate(pipe, X, y, cv=3)
        return mean_score
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
                print(f"type {ti} ({exp_type['name']}): subject {subj:03d}: accuracy = {acc:.4f}")

    print("\n" + "=" * 60)
    print("Mean accuracy per experiment type (all subjects):")
    type_means = []
    for ti in range(len(EXPERIMENT_TYPES)):
        if type_accs[ti]:
            m = np.mean(type_accs[ti])
            type_means.append(m)
            print(f"  type {ti} ({EXPERIMENT_TYPES[ti]['name']}): mean accuracy = {m:.4f}")
        else:
            print(f"  type {ti}: no valid results")
    if type_means:
        global_mean = np.mean(type_means)
        print(f"\nMean of {len(type_means)} type means: {global_mean:.4f}")
        if global_mean >= 0.75:
            print(f"✅ Score >= 75% ({global_mean*100:.1f}%)")
        else:
            print(f"⚠️  Score < 75% ({global_mean*100:.1f}%)")
        # Bonus points: +1 for each 3% over 75%
        if global_mean > 0.75:
            bonus = int((global_mean - 0.75) / 0.03)
            print(f"   Bonus points: +{bonus} (for {(global_mean - 0.75)*100:.1f}% over 75%)")


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
                              help='Run all 109 subjects on 6 experiments (train/test split by runs)')
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
