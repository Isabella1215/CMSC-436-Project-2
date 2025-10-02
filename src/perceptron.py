# src/run_experiments.py
import os, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import normalize_minmax, train_perceptron, evaluate  # your file

def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data/splits", exist_ok=True)

def split_df(df, frac, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(frac * len(df))
    tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    return tr.reset_index(drop=True), te.reset_index(drop=True)

def plot_decision_boundary(w, X, y, title, path):
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=y, cmap="bwr", edgecolors="k")
    # line: w0*x + w1*y + w2 = 0 -> y = -(w0*x + w2)/w1
    if abs(w[1]) > 1e-12:
        xs = np.linspace(X[:,0].min(), X[:,0].max(), 200)
        ys = -(w[0]*xs + w[2]) / w[1]
        plt.plot(xs, ys, "k--")
    plt.title(title)
    plt.xlabel("feature1 (normalized)")
    plt.ylabel("feature2 (normalized)")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def run_phase(df, tag, activation, alpha, gain, max_iters, epsilon, seed):
    # features + labels
    X = df[["feature1","feature2"]].to_numpy(float)
    y = df["label"].to_numpy(int)
    # train
    w, TE, iters = train_perceptron(
        X, y, activation=activation, alpha=alpha, gain=gain,
        max_iters=max_iters, epsilon=epsilon, seed=seed
    )
    # eval
    metrics = evaluate(w, X, y, activation=activation, gain=gain)
    # save plot
    plot_decision_boundary(
        w, X, y,
        f"B – {activation} – {tag}",
        f"outputs/B_{activation}_{tag}.png"
    )
    # pack
    out = {"weights": list(map(float, w)), "TE": float(TE), "iters": int(iters)}
    out.update({k: int(v) if k in ["TP","TN","FP","FN"] else float(v) for k,v in metrics.items()})
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    # per spec: B stops when TE < 40
    parser.add_argument("--epsilonB", type=float, default=40.0)
    args = parser.parse_args()

    ensure_dirs()

    # Load Dataset B: 3 cols (f1,f2,label)
    dfB = pd.read_csv("data/groupB.txt", header=None, names=["feature1","feature2","label"])
    dfB = normalize_minmax(dfB, ["feature1","feature2"])
    dfB["label"] = dfB["label"].astype(int)

    results = {}

    # 75/25
    tr, te = split_df(dfB, frac=0.75, seed=args.seed)
    tr.to_csv("data/splits/B_train_75.csv", index=False)
    te.to_csv("data/splits/B_test_25.csv", index=False)

    results["B_75_25"] = {
        "hard": {
            "train": run_phase(tr, "train75", "hard", args.alpha, args.gain, args.max_iters, args.epsilonB, args.seed),
            "test":  run_phase(te, "test25",  "hard", args.alpha, args.gain, args.max_iters, args.epsilonB, args.seed),
        },
        "soft": {
            "train": run_phase(tr, "train75", "soft", args.alpha, args.gain, args.max_iters, args.epsilonB, args.seed),
            "test":  run_phase(te, "test25",  "soft", args.alpha, args.gain, args.max_iters, args.epsilonB, args.seed),
        }
    }

    # 25/75
    tr, te = split_df(dfB, frac=0.25, seed=args.seed)
    tr.to_csv("data/splits/B_train_25.csv", index=False)
    te.to_csv("data/splits/B_test_75.csv", index=False)

    results["B_25_75"] = {
        "hard": {
            "train": run_phase(tr, "train25", "hard", args.alpha, args.gain, args.max_iters, args.epsilonB, args.seed),
            "test":  run_phase(te, "test75",  "hard", args.alpha, args.gain, args.max_iters, args.epsilonB, args.seed),
        },
        "soft": {
            "train": run_phase(tr, "train25", "soft", args.alpha, args.gain, args.max_iters, args.epsilonB, args.seed),
            "test":  run_phase(te, "test75",  "soft", args.alpha, args.gain, args.max_iters, args.epsilonB, args.seed),
        }
    }

    # dump one metrics file for the report
    with open("outputs/B_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
