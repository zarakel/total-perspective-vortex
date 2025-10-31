# tpv.py
import argparse
import time
import numpy as np
from src.loader import load_physionet, make_epochs
from src.preprocessing import bandpass_filter, visualize_raw
from src.pipeline_model import build_pipeline, train_and_evaluate
from src.stream_simulator import stream_epochs
import joblib

def cmd_train(args):
    raw = load_physionet(args.subject, args.runs)
    # visualize raw optionally
    if args.show_raw:
        visualize_raw(raw)
    raw = bandpass_filter(raw, l_freq=8.0, h_freq=30.0)
    epochs = make_epochs(raw, tmin=0.0, tmax=4.0)
    X = epochs.get_data()  # shape (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]  # adjust to your event_id mapping
    print(f"Shape de X : {X.shape}, Shape de y : {y.shape}")
    pipe = build_pipeline(sfreq=int(raw.info['sfreq']), reducer='csp', reducer_params={'n_components':4})
    mean_score, scores = train_and_evaluate(pipe, X, y, cv=5)
    print("cross_val_score:", mean_score, scores)
    # fit on whole train set and save
    pipe.fit(X, y)
    joblib.dump(pipe, args.model_path)
    print("Model saved to", args.model_path)

def cmd_predict(args):
    pipe = joblib.load(args.model_path)
    raw = load_physionet(args.subject, args.runs)  # or load a different file for test
    raw = bandpass_filter(raw, l_freq=8.0, h_freq=30.0)
    epochs = make_epochs(raw, tmin=0.0, tmax=4.0)
    X = epochs.get_data()
    labels = epochs.events[:, -1]
    # simulate stream: we require prediction to be available within 2s after chunk end
    correct = 0
    total = 0
    for epoch, y_true, t_end in stream_epochs(X, labels, delay_sim=0.1):
        t_start_pred = time.time()
        # pipeline expects full epoch shape (n_epochs, n_ch, n_times)
        pred = pipe.predict(epoch[np.newaxis, ...])
        t_pred = time.time()
        latency = t_pred - t_end
        print(f"epoch {total:02d}: pred={pred[0]} truth={y_true} latency={latency:.3f}s")
        total += 1
        if pred[0] == y_true:
            correct += 1
        # check latency constraint:
        if latency > 2.0:
            print("WARNING: prediction latency exceeded 2s!")
    print("Accuracy:", correct/total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    ptrain = sub.add_parser('train')
    ptrain.add_argument('--subject', type=int, required=True)
    ptrain.add_argument('--runs', type=int, nargs='+', required=True)
    ptrain.add_argument('--model-path', default='model.joblib')
    ptrain.add_argument('--show-raw', action='store_true')

    ppredict = sub.add_parser('predict')
    ppredict.add_argument('--subject', type=int, required=True)
    ppredict.add_argument('--runs', type=int, nargs='+', required=True)
    ppredict.add_argument('--model-path', default='model.joblib')

    args = parser.parse_args()
    if args.cmd == 'train':
        cmd_train(args)
    elif args.cmd == 'predict':
        cmd_predict(args)
    else:
        parser.print_help()
