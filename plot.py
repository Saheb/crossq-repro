"""Plot learning curves from runs/<env>/<algo>/seed*/log.csv (mean +- std over seeds).

    uv run plot.py                # all envs found under runs/
    uv run plot.py --runs runs --out results.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--runs", default="runs")
parser.add_argument("--out", default="results.png")
args = parser.parse_args()

# {env: {algo: DataFrame(step, seed0, seed1, ...)}}
curves: dict[str, dict[str, pd.DataFrame]] = {}
for log in sorted(Path(args.runs).glob("*/*/seed*/log.csv")):
    env, algo, seed = log.parts[-4], log.parts[-3], log.parts[-2]
    df = pd.read_csv(log).set_index("step")["eval_return"].rename(seed)
    curves.setdefault(env, {}).setdefault(algo, []).append(df)

fig, axes = plt.subplots(1, len(curves), figsize=(5 * len(curves), 4), squeeze=False)
for ax, (env, algos) in zip(axes[0], curves.items()):
    for algo, seeds in algos.items():
        df = pd.concat(seeds, axis=1)
        mean, std = df.mean(axis=1), df.std(axis=1).fillna(0)
        ax.plot(df.index, mean, label=f"{algo} (n={df.shape[1]})")
        ax.fill_between(df.index, mean - std, mean + std, alpha=0.2)
    ax.set(title=env, xlabel="environment steps", ylabel="eval return")
    ax.legend()
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(args.out, dpi=120)
print(f"saved {args.out}")
