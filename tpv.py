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


# ── 6 experiments definition ──────────────────────────────────────────
# Leave-one-repetition-out: train on 4 runs, test on 2 runs (never-learned data)
EXPERIMENTS = [
    # Left/Right fist (exec + imagery): runs 3,4 / 7,8 / 11,12
    {'name': 'L/R Fist (test rep 1)', 'train': [7, 8, 11, 12], 'test': [3, 4]},
    {'name': 'L/R Fist (test rep 2)', 'train': [3, 4, 11, 12], 'test': [7, 8]},
    {'name': 'L/R Fist (test rep 3)', 'train': [3, 4, 7, 8],   'test': [11, 12]},
    # Fists/Feet (exec + imagery): runs 5,6 / 9,10 / 13,14
    {'name': 'Fists/Feet (test rep 1)', 'train': [9, 10, 13, 14], 'test': [5, 6]},
    {'name': 'Fists/Feet (test rep 2)', 'train': [5, 6, 13, 14],  'test': [9, 10]},
    {'name': 'Fists/Feet (test rep 3)', 'train': [5, 6, 9, 10],   'test': [13, 14]},
]


def _evaluate_subject_experiment(subject, exp, sfreq_cache={}):
    """Train on exp['train'] runs, test on exp['test'] runs for one subject.
    Returns accuracy on the test set (never-learned data)."""
    from src.loader import load_physionet, make_epochs
    from src.preprocessing import bandpass_filter
    from src.pipeline_model import build_pipeline

    try:
        # ── Train ──
        raw_train = load_physionet(subject, exp['train'])
        raw_train = bandpass_filter(raw_train, 8.0, 30.0)
        epochs_train = make_epochs(raw_train, tmin=0.0, tmax=4.0)
        X_train = epochs_train.get_data()
        y_train = epochs_train.events[:, -1]

        if len(np.unique(y_train)) < 2:
            return None  # skip if single-class

        # ── Test ──
        raw_test = load_physionet(subject, exp['test'])
        raw_test = bandpass_filter(raw_test, 8.0, 30.0)
        epochs_test = make_epochs(raw_test, tmin=0.0, tmax=4.0)
        X_test = epochs_test.get_data()
        y_test = epochs_test.events[:, -1]

        if len(np.unique(y_test)) < 2:
            return None

        pipe = build_pipeline(
            sfreq=int(raw_train.info['sfreq']),
            reducer='csp',
            reducer_params={'csp': {'n_components': 6}}
        )
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        return acc
    except Exception as e:
        print(f"  [WARN] subject {subject:03d} exp '{exp['name']}': {e}")
        return None


def cmd_evaluate_all(args):
    """Evaluate all 109 subjects on 6 experiments, report mean accuracy."""
    import warnings
    warnings.filterwarnings('ignore')

    n_subjects = getattr(args, 'max_subjects', 109)
    exp_accs = {i: [] for i in range(len(EXPERIMENTS))}

    for subj in range(1, n_subjects + 1):
        for ei, exp in enumerate(EXPERIMENTS):
            acc = _evaluate_subject_experiment(subj, exp)
            if acc is not None:
                exp_accs[ei].append(acc)
                print(f"experiment {ei}: subject {subj:03d}: accuracy = {acc:.4f}")

    print("\n" + "=" * 60)
    print("Mean accuracy of the six different experiments for all subjects:")
    all_means = []
    for ei in range(len(EXPERIMENTS)):
        if exp_accs[ei]:
            m = np.mean(exp_accs[ei])
            all_means.append(m)
            print(f"  experiment {ei} ({EXPERIMENTS[ei]['name']}): accuracy = {m:.4f}")
        else:
            print(f"  experiment {ei}: no valid results")
    if all_means:
        print(f"Mean accuracy of {len(EXPERIMENTS)} experiments: {np.mean(all_means):.4f}")


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
