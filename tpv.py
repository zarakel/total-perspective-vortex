# tpv.py
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.loader import load_physionet, make_epochs
from src.preprocessing import bandpass_filter, visualize_raw
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
        visualize_raw(raw)
    raw = bandpass_filter(raw, l_freq=8.0, h_freq=30.0)

    print(f"Using window_sec={args.window_sec} overlap={args.overlap}")
    epochs = make_epochs(raw, tmin=0.0, tmax=4.0, window_sec=args.window_sec, overlap=args.overlap)
    X = epochs.get_data()
    y = epochs.events[:, -1]

    print(f"Shape de X : {X.shape}, Shape de y : {y.shape}")

    # build pipeline, try to pass memory if supported by build_pipeline
    try:
        pipe = build_pipeline(
            sfreq=int(raw.info['sfreq']),
            reducer='csp',
            reducer_params={'n_components': 8},
            memory=args.memory
        )
    except TypeError:
        pipe = build_pipeline(
            sfreq=int(raw.info['sfreq']),
            reducer='csp',
            reducer_params={'n_components': 8}
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

    ppredict = sub.add_parser('predict')
    ppredict.add_argument('--subject', type=int, required=True)
    ppredict.add_argument('--runs', type=int, nargs='+', required=True,
                        help='Runs used for prediction/evaluation (should be different from train runs)')
    ppredict.add_argument('--model-path', default='model.joblib')
    ppredict.add_argument('--window-sec', type=float, default=None)
    ppredict.add_argument('--overlap', type=float, default=0.5)

    args = parser.parse_args()
    if args.cmd == 'train':
        cmd_train(args)
    elif args.cmd == 'predict':
        cmd_predict(args)
    else:
        parser.print_help()
