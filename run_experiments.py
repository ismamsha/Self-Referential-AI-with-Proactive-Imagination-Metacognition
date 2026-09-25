"""
Train every learning variant on every difficulty over several seeds, evaluate
fixed baseline policies on the same environment, and write:

  results/<difficulty>_<variant>_seed<k>.json   per-run learning curve + final eval
  results/summary.md                            final evaluation table
  results/learning_curves.png                   evaluation curves during training

Usage: python run_experiments.py [--episodes 400] [--seeds 0 1 2] [--workers 4]
"""

import argparse
import json
import os
import random
from multiprocessing import Pool

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DIFFICULTIES = ["normal", "hard", "expert"]
LABELS = {
    "dqn": "DQN (value learning only)",
    "msra_no_doubt": "MSRA-L without doubt",
    "msra": "MSRA-L with doubt",
    "msra_anxiety": "MSRA-L with doubt + fear of death",
    "anxiety_real": "Doubt + fear of death, no imagination",
}
COLORS = {"dqn": "#2a78d6", "msra_no_doubt": "#eb6834", "msra": "#1baf7a",
          "msra_anxiety": "#eda100", "anxiety_real": "#e87ba4"}
MARKERS = {"dqn": "o", "msra_no_doubt": "s", "msra": "^", "msra_anxiety": "D", "anxiety_real": "v"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def always_rest(r, d):
    return 0


def threshold_rule(r, d):
    if d > 0.7 and r > 0.4:
        return 1
    if r < 0.6:
        return 0
    return 2 if r > 0.8 else 4


BASELINES = {"Always rest": always_rest, "Four-line threshold rule": threshold_rule}


def run_baseline(policy, difficulty, seed, episodes):
    from main import HomeostasisEnv
    from learning_agent import survived
    random.seed(seed)
    env = HomeostasisEnv(obs_dim=10, difficulty=difficulty)
    surv, rew, work = [], [], []
    for _ in range(episodes):
        env.reset()
        total, w = 0.0, 0
        while True:
            a = policy(env.r, env.d)
            _, r, _, done, _, _ = env.step(a)
            total += r
            w += a >= 2
            if done:
                break
        surv.append(survived(env))
        rew.append(total)
        work.append(w / env.step_count)
    return {"survival": np.mean(surv) * 100, "reward": np.mean(rew), "work_pct": np.mean(work) * 100}


def run_job(job):
    import torch
    torch.set_num_threads(1)
    from learning_agent import Config, train
    variant, difficulty, seed, episodes = job
    path = os.path.join(RESULTS, f"{difficulty}_{variant}_seed{seed}.json")
    if os.path.exists(path):
        return path
    result = train(Config(variant=variant, difficulty=difficulty, episodes=episodes, seed=seed),
                   verbose=False)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    final = result["final"]
    print(f"done {difficulty:6s} {variant:13s} seed {seed}: survival {final['survival']:5.1f}%  "
          f"reward {final['reward']:6.1f}", flush=True)
    return path


def load(difficulty, variant, seeds):
    runs = []
    for s in seeds:
        path = os.path.join(RESULTS, f"{difficulty}_{variant}_seed{s}.json")
        if os.path.exists(path):
            with open(path) as f:
                runs.append(json.load(f))
    return runs


# Two-sided 95% Student-t critical values by degrees of freedom.
T_975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}


def mean_ci(values):
    values = np.asarray(values, float)
    if len(values) < 2:
        return values.mean(), 0.0
    t = T_975.get(len(values) - 1, 1.96)
    return values.mean(), t * values.std(ddof=1) / np.sqrt(len(values))


def write_summary(seeds, baselines):
    lines = ["# Results", "",
             f"Final evaluation: 200 greedy episodes (no exploration) per run, "
             f"mean ± 95% CI over {len(seeds)} seeds. Survival means reaching step 100 "
             f"with resources > 0 and degradation < 1.", ""]
    for diff in DIFFICULTIES:
        lines += [f"## {diff.capitalize()}", "",
                  "| Policy | Survival | Reward / episode | Time working | Caution changed the action |",
                  "|---|---|---|---|---|"]
        for name, b in baselines[diff].items():
            lines.append(f"| {name} | {b['survival']:.1f}% | {b['reward']:.1f} | {b['work_pct']:.0f}% | – |")
        for v in LABELS:
            runs = load(diff, v, seeds)
            if not runs:
                continue
            cells = []
            for key, fmt in (("survival", "{:.1f} ± {:.1f}%"), ("reward", "{:.1f} ± {:.1f}"),
                             ("work_pct", "{:.0f} ± {:.0f}%")):
                cells.append(fmt.format(*mean_ci([r["final"][key] for r in runs])))
            caution = "–" if v in ("dqn", "msra_no_doubt") else \
                "{:.0f}%".format(np.mean([r["final"]["caution_pct"] for r in runs]))
            lines.append(f"| {LABELS[v]} | " + " | ".join(cells) + f" | {caution} |")
        lines.append("")
    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write("\n".join(lines))


def plot(seeds, baselines):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    for col, diff in enumerate(DIFFICULTIES):
        for row, key in enumerate(("survival", "reward")):
            ax = axes[row, col]
            for v in LABELS:
                runs = load(diff, v, seeds)
                if not runs:
                    continue
                x = [p["episode"] for p in runs[0]["curve"]]
                y = np.array([[p[key] for p in r["curve"]] for r in runs])
                m = y.mean(0)
                ax.plot(x, m, color=COLORS[v], lw=2, label=LABELS[v],
                        marker=MARKERS[v], markersize=5, markevery=4)
                if len(runs) > 1:
                    sd = y.std(0)
                    ax.fill_between(x, m - sd, m + sd, color=COLORS[v], alpha=0.08, lw=0)
            vals = {name: baselines[diff][name][key] for name in BASELINES}
            lowest = min(vals, key=lambda n: (vals[n], list(BASELINES).index(n)))
            for name, style in zip(BASELINES, (":", "--")):
                below = name == lowest
                ax.axhline(vals[name], color=MUTED, lw=1.2, ls=style)
                ax.annotate(name, xy=(1.0, vals[name]), xycoords=("axes fraction", "data"),
                            xytext=(-4, -3 if below else 3), textcoords="offset points",
                            ha="right", va="top" if below else "bottom", fontsize=8, color=MUTED)
            ax.grid(True, color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=9)
            if row == 0:
                ax.set_title(f"{diff.capitalize()} difficulty", color=INK, fontsize=12)
                ax.set_ylim(-3, 103)
                if col == 0:
                    ax.set_ylabel("Survival (%)", color=INK)
            else:
                ax.set_xlabel("Training episode", color=INK)
                if col == 0:
                    ax.set_ylabel("Reward per episode", color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10)
    fig.text(0.5, 0.005, "Greedy evaluation every 20 training episodes; line = mean over seeds, "
             "band = ±1 std. Grey lines are fixed baseline policies.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(os.path.join(RESULTS, "learning_curves.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    jobs = [(v, d, s, args.episodes) for d in DIFFICULTIES for s in args.seeds for v in LABELS]
    with Pool(args.workers) as pool:
        pool.map(run_job, jobs, chunksize=1)

    baselines = {}
    for d in DIFFICULTIES:
        baselines[d] = {}
        for name, policy in BASELINES.items():
            runs = [run_baseline(policy, d, s, 200) for s in args.seeds]
            baselines[d][name] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    write_summary(args.seeds, baselines)
    plot(args.seeds, baselines)
    print("wrote results/summary.md and results/learning_curves.png")


if __name__ == "__main__":
    main()
