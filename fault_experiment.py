"""
Hidden-fault experiment: can the agent notice that its own body has changed?

In half of all episodes the agent's body develops a fault at a random step.
Two settings:

  mild    actuator: every work action drains 0.25 x intensity extra resources
          battery:  resting recovers only a quarter as much
  severe  actuator: every work action drains 0.5 x intensity extra resources
          fragile:  random shocks become three times as frequent

The fault never appears in the observation. The agent sees only its resources,
degradation and the time, so it can only infer the fault from how its body
responds.

Agents (all trained from scratch in the faulty environment). The first four
share the same machinery and differ only in one extra input to their state:

  metacognitive       + surprise-doubt: a running average of how far reality
                      lands from the self-model's predictions (signed, in units
                      of the predicted spread)
  anxiety_history     + raw memory of the last 4 transitions
  oracle              + the true fault flag (upper bound)
  anxiety_real        nothing extra: the fault-blind agent from learning_agent.py
  metacognitive_beta  first design (mild setting only): surprise-doubt in the
                      state and also driving beta
  dqn                 plain value learning

Usage: python fault_experiment.py [--faults mild|severe] [--episodes 500] [--seeds 0 1 2]
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

from main import HomeostasisEnv
from learning_agent import (Config, LearningMSRA, PlainObserver, N_ACTIONS, STATE_DIM,
                            epsilon, internal_state, run_episode, survived)

ROOT = os.path.dirname(os.path.abspath(__file__))
BEFORE, AFTER = 10, 30  # steps around fault onset kept for aligned traces
ONSET = (20, 70)
WINDOW = 15  # steps after onset in which a fault counts as detected

FAULT_SETTINGS = {
    "mild": {"types": ("actuator", "battery"), "actuator_drain": 0.25,
             "battery_factor": 0.25, "shock_factor": 1.0},
    "severe": {"types": ("actuator", "fragile"), "actuator_drain": 0.5,
               "battery_factor": 1.0, "shock_factor": 3.0},
}
FAULT_NAMES = {"actuator": "Motor", "battery": "Battery", "fragile": "Fragile"}


def results_dir(setting):
    return os.path.join(ROOT, "results", f"faults_{setting}")


class FaultyHomeostasisEnv(HomeostasisEnv):
    INTENSITY = {2: 1.0, 3: 0.7, 4: 0.4}

    def __init__(self, obs_dim=10, difficulty="hard", faults="mild", fault_prob=0.5):
        super().__init__(obs_dim, difficulty)
        setting = FAULT_SETTINGS[faults]
        self.fault_types = setting["types"]
        self.actuator_drain = setting["actuator_drain"]
        self.battery_factor = setting["battery_factor"]
        self.shock_factor = setting["shock_factor"]
        self.fault_prob = fault_prob
        self.nominal_rest_recovery = self.rest_recovery
        self.nominal_shock_probability = self.shock_probability
        self.fault_type = None
        self.fault_onset = None
        self.fault_active = False

    def reset(self):
        self.rest_recovery = self.nominal_rest_recovery
        self.shock_probability = self.nominal_shock_probability
        self.fault_active = False
        self.fault_type, self.fault_onset = None, None
        if random.random() < self.fault_prob:
            self.fault_type = random.choice(self.fault_types)
            self.fault_onset = random.randint(*ONSET)
        return super().reset()

    def step(self, action_idx):
        if self.fault_type and not self.fault_active and self.step_count >= self.fault_onset:
            self.fault_active = True
            if self.fault_type == "battery":
                self.rest_recovery = self.nominal_rest_recovery * self.battery_factor
            elif self.fault_type == "fragile":
                self.shock_probability = self.nominal_shock_probability * self.shock_factor
        if self.fault_active and self.fault_type == "actuator" and action_idx >= 2:
            self.r = max(0.0, self.r - self.actuator_drain * self.INTENSITY[action_idx])
        return super().step(action_idx)


class OracleObserver:
    """(r, d, t) plus the true fault flag."""

    extra_dim = 1

    def reset(self, obs, env):
        return np.append(internal_state(obs), 0.0).astype(np.float32)

    def step(self, obs, env, s, a):
        return np.append(internal_state(obs), float(env.fault_active)).astype(np.float32)


class HistoryObserver:
    """(r, d, t) plus the last K transitions: change in r, change in d, action."""

    K = 4
    extra_dim = K * (2 + N_ACTIONS)
    # Per-step changes are ~0.1 for r and ~0.01 for d; scale them to order one.
    SCALE = np.array([5.0, 20.0], np.float32)

    def reset(self, obs, env):
        self.hist = np.zeros((self.K, 2 + N_ACTIONS), np.float32)
        return np.concatenate([internal_state(obs), self.hist.ravel()])

    def step(self, obs, env, s, a):
        s2 = internal_state(obs)
        row = np.zeros(2 + N_ACTIONS, np.float32)
        row[:2] = (s2[:2] - s[:2]) * self.SCALE
        row[2 + a] = 1.0
        self.hist = np.roll(self.hist, 1, axis=0)
        self.hist[0] = row
        return np.concatenate([s2, self.hist.ravel()])


class SurpriseObserver:
    """(r, d, t) plus surprise-doubt: a running average of the self-model's
    signed surprise on resources and degradation. With drives_beta the size of
    that average also replaces the agent's own doubt in beta."""

    extra_dim = 2

    def __init__(self, agent, drives_beta=False, decay=0.85, clip=5.0):
        self.agent = agent
        self.drives_beta = drives_beta
        self.decay = decay
        self.clip = clip
        self.signal = 0.0
        agent.external_doubt = drives_beta

    def reset(self, obs, env):
        self.e = np.zeros(2, np.float32)
        self.signal = 0.0
        if self.drives_beta:
            self.agent.doubt = 0.0
        return np.concatenate([internal_state(obs), self.e])

    def step(self, obs, env, s, a):
        s2 = internal_state(obs)
        if self.agent.model_ready:
            z = np.clip(self.agent.surprise(s, a, s2), -self.clip, self.clip)
            self.e = (self.decay * self.e + (1 - self.decay) * z).astype(np.float32)
            self.signal = float(np.linalg.norm(self.e))
            if self.drives_beta:
                self.agent.doubt = self.signal
        return np.concatenate([s2, self.e])


VARIANTS = {
    "metacognitive": ("anxiety_real", "surprise", "Metacognitive: + surprise-doubt"),
    "metacognitive_beta": ("anxiety_real", "surprise_beta", "First design: surprise-doubt also drives β"),
    "anxiety_history": ("anxiety_real", "history", "+ memory of last 4 steps"),
    "oracle": ("anxiety_real", "oracle", "+ told the fault (oracle)"),
    "anxiety_real": ("anxiety_real", "plain", "Fault-blind: nothing extra"),
    "dqn": ("dqn", "plain", "DQN (value learning only)"),
}
SETTING_VARIANTS = {"mild": list(VARIANTS), "severe": [v for v in VARIANTS if v != "metacognitive_beta"]}
COLORS = {"metacognitive": "#2a78d6", "anxiety_real": "#eb6834", "anxiety_history": "#1baf7a",
          "oracle": "#eda100", "dqn": "#e87ba4"}
MARKERS = {"metacognitive": "o", "anxiety_real": "s", "anxiety_history": "^",
           "oracle": "D", "dqn": "v"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def build(name, cfg):
    agent_variant, obs_kind, _ = VARIANTS[name]
    cfg.variant = agent_variant
    extra = {"plain": 0, "oracle": OracleObserver.extra_dim, "history": HistoryObserver.extra_dim,
             "surprise": SurpriseObserver.extra_dim, "surprise_beta": SurpriseObserver.extra_dim}[obs_kind]
    agent = LearningMSRA(cfg, state_dim=STATE_DIM + extra)
    if obs_kind.startswith("surprise"):
        return agent, SurpriseObserver(agent, drives_beta=obs_kind == "surprise_beta")
    observer = {"plain": PlainObserver, "oracle": OracleObserver, "history": HistoryObserver}[obs_kind]
    return agent, observer()


def eval_episode(agent, observer, env):
    s = observer.reset(env.reset(), env)
    doubts, betas, actions, rs, ds, total = [], [], [], [], [], 0.0
    while True:
        doubts.append(getattr(observer, "signal", agent.doubt))
        betas.append(float(agent.beta(float(s[0]))))
        rs.append(float(env.r))
        ds.append(float(env.d))
        a = agent.act(s, eps=0.0)
        actions.append(a)
        next_obs, reward, _, done, _, _ = env.step(a)
        s = observer.step(next_obs, env, s, a)
        total += reward
        if done:
            rs.append(float(env.r))
            ds.append(float(env.d))
            return {"fault": env.fault_type, "onset": env.fault_onset, "survived": bool(survived(env)),
                    "reward": total, "doubt": doubts, "beta": betas, "actions": actions, "r": rs, "d": ds}


def examples(episodes, fault_types):
    """The first evaluation episode of each kind, kept whole for replay."""
    out = {}
    for kind in (None,) + tuple(fault_types):
        ep = next((e for e in episodes if e["fault"] == kind), None)
        if ep:
            out[kind or "healthy"] = {k: ep[k] for k in ("fault", "onset", "survived", "reward",
                                                         "doubt", "beta", "actions", "r", "d")}
    return out


def aligned(values, onset):
    out = np.full(BEFORE + AFTER, np.nan)
    for i in range(-BEFORE, AFTER):
        if 0 <= onset + i < len(values):
            out[i + BEFORE] = values[onset + i]
    return out


def analyse(episodes, has_doubt, rng, fault_types):
    fault = [e for e in episodes if e["fault"]]
    healthy = [e for e in episodes if not e["fault"]]
    res = {
        "survival_fault": 100 * np.mean([e["survived"] for e in fault]),
        "survival_healthy": 100 * np.mean([e["survived"] for e in healthy]),
        "reward_fault": float(np.mean([e["reward"] for e in fault])),
        "reward_healthy": float(np.mean([e["reward"] for e in healthy])),
        "n_fault": len(fault), "n_healthy": len(healthy),
    }
    for kind in fault_types:
        eps = [e for e in fault if e["fault"] == kind]
        res[f"survival_{kind}"] = 100 * np.mean([e["survived"] for e in eps])
        res[f"work_trace_{kind}"] = np.nanmean(
            [aligned([float(a >= 2) for a in e["actions"]], e["onset"]) for e in eps], axis=0).tolist()

    if has_doubt:
        # Alarm threshold: exceeded on only 5% of steps in healthy episodes.
        tau = float(np.percentile(np.concatenate([e["doubt"] for e in healthy]), 95))
        res["doubt_threshold"] = tau
        hits = [max(e["doubt"][e["onset"]:e["onset"] + WINDOW + 1], default=0) > tau for e in fault]
        # Same test on healthy episodes at a random pseudo-onset gives the false-alarm rate.
        false = []
        for e in healthy:
            t0 = rng.randint(*ONSET)
            false.append(max(e["doubt"][t0:t0 + WINDOW + 1], default=0) > tau)
        res["detected_within_15"] = 100 * np.mean(hits)
        res["false_alarm_within_15"] = 100 * np.mean(false)
        for kind in fault_types:
            eps = [e for e in fault if e["fault"] == kind]
            res[f"doubt_trace_{kind}"] = np.nanmean(
                [aligned(np.array(e["doubt"]) / tau, e["onset"]) for e in eps], axis=0).tolist()
    return res


def run_job(job):
    import torch
    torch.set_num_threads(1)
    setting, name, seed, episodes, eval_episodes = job
    path = os.path.join(results_dir(setting), f"{name}_seed{seed}.json")
    if os.path.exists(path):
        return path
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = Config(difficulty="hard", episodes=episodes, seed=seed)
    agent, observer = build(name, cfg)
    env = FaultyHomeostasisEnv(faults=setting)
    for ep in range(episodes):
        run_episode(agent, env, epsilon(cfg, ep), learn=True, observer=observer)

    eval_env = FaultyHomeostasisEnv(faults=setting)
    saved = agent.doubt
    episodes_out = [eval_episode(agent, observer, eval_env) for _ in range(eval_episodes)]
    agent.doubt = saved
    types = FAULT_SETTINGS[setting]["types"]
    result = analyse(episodes_out, has_doubt=agent.use_model, rng=random.Random(seed), fault_types=types)
    result.update(variant=name, seed=seed, episodes=episodes, faults=setting,
                  examples=examples(episodes_out, types))
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"done {setting} {name:18s} seed {seed}: survival healthy {result['survival_healthy']:5.1f}%  "
          f"fault {result['survival_fault']:5.1f}%", flush=True)
    return path


def run_baseline(policy, setting, seed, episodes):
    random.seed(seed)
    env = FaultyHomeostasisEnv(faults=setting)
    out = []
    for _ in range(episodes):
        env.reset()
        while True:
            _, _, _, done, _, _ = env.step(policy(env.r, env.d))
            if done:
                break
        out.append((env.fault_type is not None, survived(env)))
    fault = [s for f, s in out if f]
    healthy = [s for f, s in out if not f]
    return 100 * np.mean(healthy), 100 * np.mean(fault)


def load(setting, name, seeds):
    runs = []
    for s in seeds:
        path = os.path.join(results_dir(setting), f"{name}_seed{s}.json")
        if os.path.exists(path):
            with open(path) as f:
                runs.append(json.load(f))
    return runs


T_975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}


def fmt(values, unit=""):
    v = np.asarray(values, float)
    if len(v) < 2:
        return f"{v.mean():.1f}{unit}"
    ci = T_975.get(len(v) - 1, 1.96) * v.std(ddof=1) / np.sqrt(len(v))
    return f"{v.mean():.1f} ± {ci:.1f}{unit}"


DESCRIPTIONS = {
    "mild": "Motor fault: every work action drains 0.25 × intensity extra resources. "
            "Battery fault: resting recovers a quarter as much.",
    "severe": "Motor fault: every work action drains 0.5 × intensity extra resources. "
              "Fragile fault: random shocks become three times as frequent.",
}


def write_summary(setting, seeds, baselines):
    types = FAULT_SETTINGS[setting]["types"]
    head = " | ".join(f"{FAULT_NAMES[t]} fault" for t in types)
    lines = [f"# Hidden-fault experiment: {setting} faults", "",
             "Hard difficulty. In half of all episodes a hidden fault starts at a random step "
             f"between {ONSET[0]} and {ONSET[1]}. {DESCRIPTIONS[setting]} 500 training episodes, "
             f"then 300 greedy evaluation episodes per run; mean ± 95% CI over {len(seeds)} seeds.", "",
             f"| Agent | Survival, healthy | Survival, fault | {head} | Reward, fault | "
             "Fault detected within 15 steps | False alarms |",
             "|---|---|---|" + "---|" * len(types) + "---|---|---|"]
    dashes = " | ".join("–" for _ in types)
    for name, (healthy, fault) in baselines.items():
        lines.append(f"| {name} | {healthy:.1f}% | {fault:.1f}% | {dashes} | – | – | – |")
    for name in SETTING_VARIANTS[setting]:
        runs = load(setting, name, seeds)
        if not runs:
            continue
        get = lambda k: [r[k] for r in runs]
        det = fmt(get("detected_within_15"), "%") if "detected_within_15" in runs[0] else "–"
        fa = fmt(get("false_alarm_within_15"), "%") if "false_alarm_within_15" in runs[0] else "–"
        per_type = " | ".join(fmt(get(f"survival_{t}"), "%") for t in types)
        lines.append(f"| {VARIANTS[name][2]} | {fmt(get('survival_healthy'), '%')} | "
                     f"{fmt(get('survival_fault'), '%')} | {per_type} | {fmt(get('reward_fault'))} | "
                     f"{det} | {fa} |")
    lines += ["", "Detection: the agent's doubt signal rises above the level it exceeds on only 5% of "
              "healthy steps, within 15 steps of the fault. False alarms: the same test on healthy "
              "episodes at a random step. For the metacognitive agents the signal is surprise-doubt; "
              "for the others it is ensemble disagreement."]
    with open(os.path.join(results_dir(setting), "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot(setting, seeds):
    types = FAULT_SETTINGS[setting]["types"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    x = np.arange(-BEFORE, AFTER)

    ax = axes[0]
    names = [n for n in SETTING_VARIANTS[setting] if load(setting, n, seeds)]
    for i, name in enumerate(names):
        runs = load(setting, name, seeds)
        healthy = np.mean([r["survival_healthy"] for r in runs])
        fault = np.mean([r["survival_fault"] for r in runs])
        y = len(names) - 1 - i
        ax.plot([fault, healthy], [y, y], color=MUTED, lw=1.5, zorder=1)
        ax.scatter(healthy, y, s=70, facecolor="white", edgecolor=INK, lw=1.8, zorder=2,
                   label="Healthy episodes" if i == 0 else None)
        ax.scatter(fault, y, s=70, color=INK, zorder=3, label="Episodes with a fault" if i == 0 else None)
        ax.annotate(f"{fault:.0f}%", (fault, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([VARIANTS[n][2] for n in reversed(names)], fontsize=9, color=INK)
    ax.set_xlim(-2, 104)
    ax.set_ylim(-0.6, len(names) - 0.3)
    ax.set_xlabel("Survival (%)", color=INK)
    ax.set_title("Survival with and without a hidden fault", color=INK, fontsize=11)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    style(ax)

    ax = axes[1]
    for name in ("metacognitive", "anxiety_real"):
        runs = load(setting, name, seeds)
        if not runs:
            continue
        for kind, ls in zip(types, ("-", "--")):
            trace = np.nanmean([r[f"doubt_trace_{kind}"] for r in runs], axis=0)
            signal = "surprise-doubt" if name == "metacognitive" else "disagreement"
            ax.plot(x, trace, color=COLORS[name], lw=2, ls=ls, marker=MARKERS[name], markersize=5,
                    markevery=5, label=f"{signal}, {FAULT_NAMES[kind].lower()} fault")
    ax.axhline(1.0, color=MUTED, lw=1.2, ls=":")
    ax.annotate("alarm level (5% of healthy steps)", xy=(0.99, 1.0), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8, color=MUTED)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Steps since the hidden fault began", color=INK)
    ax.set_ylabel("Doubt signal ÷ alarm level", color=INK)
    ax.set_title("Does the doubt signal notice the fault?", color=INK, fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=8)
    style(ax)

    ax = axes[2]
    for name in ("metacognitive", "anxiety_history", "oracle", "anxiety_real"):
        runs = load(setting, name, seeds)
        if not runs:
            continue
        trace = np.nanmean([np.nanmean([r[f"work_trace_{t}"] for t in types], axis=0) for r in runs], axis=0)
        ax.plot(x, 100 * trace, color=COLORS[name], lw=2, marker=MARKERS[name],
                markersize=5, markevery=5, label=VARIANTS[name][2])
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Steps since the hidden fault began", color=INK)
    ax.set_ylabel("Share of agents working (%)", color=INK)
    ax.set_title("Does behaviour change after the fault?", color=INK, fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    style(ax)

    fig.text(0.5, 0.01, f"{setting.capitalize()} faults, hard difficulty, a hidden fault in half of all "
             f"episodes; means over {len(seeds)} seeds × 300 evaluation episodes.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(results_dir(setting), "fault_results.png"), dpi=150)
    plt.close(fig)


def always_rest(r, d):
    return 0


def threshold_rule(r, d):
    if d > 0.7 and r > 0.4:
        return 1
    if r < 0.6:
        return 0
    return 2 if r > 0.8 else 4


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--faults", choices=list(FAULT_SETTINGS), default="severe")
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS))
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--eval-episodes", type=int, default=300)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    os.makedirs(results_dir(args.faults), exist_ok=True)

    variants = args.variants or SETTING_VARIANTS[args.faults]
    jobs = [(args.faults, n, s, args.episodes, args.eval_episodes) for s in args.seeds for n in variants]
    with Pool(args.workers) as pool:
        pool.map(run_job, jobs, chunksize=1)

    baselines = {}
    for label, policy in (("Always rest", always_rest), ("Four-line threshold rule", threshold_rule)):
        runs = [run_baseline(policy, args.faults, s, 300) for s in args.seeds]
        baselines[label] = tuple(np.mean(runs, axis=0))
    write_summary(args.faults, args.seeds, baselines)
    plot(args.faults, args.seeds)
    print(f"wrote {os.path.relpath(results_dir(args.faults), ROOT)}/summary.md and fault_results.png")


if __name__ == "__main__":
    main()
