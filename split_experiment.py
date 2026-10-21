"""
Train/test split experiment: KNN vs LDA accuracy (mean ± std over repeated splits).

Usage:
  1. Save your features and labels as X.npy (n_samples, n_features) and y.npy (n_samples,)
     -> or call run_experiment(X, y) directly from your notebook.
  2. Optional: groups.npy (recording ID per sample) to avoid leakage between
     windows of the same recording.
  3. python split_experiment.py
  -> prints the LaTeX table rows and saves split_experiment.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

TEST_SIZES = [0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9]
N_REPEATS = 50
MODELS = {
    "KNN": KNeighborsClassifier(n_neighbors=5),                        # put your k here
    "LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
}


def run_experiment(X, y, groups=None, test_sizes=TEST_SIZES, n_repeats=N_REPEATS, seed=0):
    results = {name: {"mean": [], "std": []} for name in MODELS}
    for ts in test_sizes:
        if groups is None:
            splitter = StratifiedShuffleSplit(n_splits=n_repeats, test_size=ts, random_state=seed)
            splits = splitter.split(X, y)
        else:
            splitter = GroupShuffleSplit(n_splits=n_repeats, test_size=ts, random_state=seed)
            splits = splitter.split(X, y, groups)

        scores = {name: [] for name in MODELS}
        for tr, te in splits:
            for name, model in MODELS.items():
                m = clone(model).fit(X[tr], y[tr])
                scores[name].append(100 * m.score(X[te], y[te]))

        for name in MODELS:
            results[name]["mean"].append(np.mean(scores[name]))
            results[name]["std"].append(np.std(scores[name]))
        print(f"test size {ts:.1f} done")
    return results


def print_latex(results, test_sizes=TEST_SIZES):
    print("\n% ---- paste into your tabular ----")
    print(r"\textbf{Test size ($N_{test}/N_{data}$)} & \textbf{KNN accuracy (\%)} & \textbf{LDA accuracy (\%)} \\")
    print(r"\hline")
    for i, ts in enumerate(test_sizes):
        cells = [f"${results[n]['mean'][i]:.1f} \\pm {results[n]['std'][i]:.1f}$" for n in MODELS]
        print(f"{ts:.1f} & " + " & ".join(cells) + r" \\")
        print(r"\hline")


def plot(results, test_sizes=TEST_SIZES, path="split_experiment.png"):
    plt.figure(figsize=(6, 4))
    for name in MODELS:
        plt.errorbar(test_sizes, results[name]["mean"], yerr=results[name]["std"],
                     marker="o", capsize=4, label=name)
    plt.xlabel(r"Test size $N_{test}/N_{data}$")
    plt.ylabel("Accuracy (%)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    print(f"plot saved to {path}")


if __name__ == "__main__":
    if os.path.exists("X.npy") and os.path.exists("y.npy"):
        X, y = np.load("X.npy"), np.load("y.npy")
        groups = np.load("groups.npy") if os.path.exists("groups.npy") else None
    else:
        # Demo only: lets you test the script. Do NOT report these numbers.
        print("X.npy / y.npy not found -> running on sklearn's digits demo dataset")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=400, n_features=20, n_informative=10,
                           n_redundant=0, n_classes=5, n_clusters_per_class=1,
                           class_sep=1.0, random_state=0)
        groups = None

    res = run_experiment(X, y, groups)
    print_latex(res)
    plot(res)
