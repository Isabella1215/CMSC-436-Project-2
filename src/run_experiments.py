# src/run_experiments.py
import os, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import normalize_minmax, train_perceptron, evaluate

def ensure_outputs():
    os.makedirs("outputs", exist_ok=True)

def plot_decision_boundary(w, X, y, title, outpath):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
    if abs(w[1]) > 1e-12:
        x_min, x_max = X[:,0].min(), X[:,0].max()
        xs = np.linspace(x_min, x_max, 200)
        ys = -(w[0]*xs + w[2]) / w[1]
        plt.plot(xs, ys, "k--")
    plt.title(title)
    plt.xlabel("feature1 (normalized)")
    plt.ylabel("feature2 (normalized)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def split(df, frac, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(frac * len(df))
    tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    return tr.reset_index(drop=True), te.reset_index(drop=True)

def run_once(df, tag, activation, alpha, gain, max_iters, epsilon, seed):
    # features & labels
    X = df[["feature1","feature2"]].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    # train
    w, TE, iters = train_perceptron(
        X, y,
        activation=activation,
        alpha=alpha,
        gain=gain,
        max_iters=max_iters,
        epsilon=epsilon,
        seed=seed
    )

    # evaluate (on the same split’s test or train you pass)
    metrics = evaluate(w, X, y, activation=activation, gain=gain)

    # save artifacts
    ensure_outputs()
    plot_decision_boundary(
        w, X, y,
        title=f"Dataset B – {activation} – {tag}",
        outpath=f"outputs/B_{activation}_{tag}.png"
    )
    return dict(weights=w.tolist(), TE=float(TE), iters=int(iters), **metrics)

def main(args):
    ensure_outputs()

    # --- LOAD DATASET B (3 columns: f1,f2,label) ---
    df = pd.read_csv(
        "data/groupB.txt",
        header=None,
        names=["feature1","feature2","label"]
    )  # labels appear in the 3rd column. :contentReference[oaicite:2]{index=2}

    # normalize features only
    df = normalize_minmax(df, ["feature1","feature2"])
    df["label"] = df["label"].astype(int)

    results = {}

    # ----- 75/25 split -----
    tr, te = split(df, frac=0.75, seed=args.seed)
    tr.to_csv("data/splits/B_train_75.csv", index=False)
    te.to_csv("data/splits/B_test_25.csv", index=False)

    # train on train, eval/plot separately for train and test
    train_res_hard = run_once(tr, "train75", "hard", args.alpha, args.gain, args.max_iters, args.epsilon_B, args.seed)
    test_res_hard  = run_once(te, "test25",  "hard", args.alpha, args.gain, args.max_iters, args.epsilon_B, args.seed)

    train_res_soft = run_once(tr, "train75", "soft", args.alpha, args.gain, args.max_iters, args.epsilon_B, args.seed)
    test_res_soft  = run_once(te, "test25",  "soft", args.alpha, args.gain, args.max_iters, args.epsilon_B, args.seed)

    results["B_75_25"] = {
        "hard": {"train": train_res_hard, "test": test_res_hard},
        "soft": {"train": train_res_soft, "test": test_res_soft},
    }

    # ----- 25/75 split -----
    tr, te = split(df, frac=0.25, seed=args.seed)
    tr.to_csv("data/splits/B_train_25.csv", index=False)
    te.to_csv("data/splits/B_test_75.csv", index=False)

    trai
