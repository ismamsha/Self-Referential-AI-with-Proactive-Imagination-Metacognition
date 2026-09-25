"""Build docs/doubting_agent.html: fill page_template.html with recorded episodes
and experiment results. Run after run_experiments.py and fault_experiment.py
(both the mild and severe settings).

Usage: python docs/build_page.py
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
os.chdir(REPO)
import run_experiments as rx
import fault_experiment as fx

SEEDS = [0, 1, 2]
REPLAY = "severe"

meta0 = fx.load(REPLAY, "metacognitive", [0])[0]
episodes = {}
for k, ep in meta0["examples"].items():
    episodes[k] = {"fault": ep["fault"], "onset": ep["onset"], "survived": ep["survived"],
                   "actions": ep["actions"]}
    for key in ("doubt", "beta", "r", "d"):
        episodes[k][key] = [round(v, 4) for v in ep[key]]
order = [k for k in ("fragile", "actuator", "battery", "healthy") if k in episodes]


def mean_final(variant, key):
    return float(np.mean([r["final"][key] for r in rx.load("hard", variant, SEEDS)]))


learning = []
for label, policy in (("Always rest", rx.always_rest), ("Four-line rule", rx.threshold_rule)):
    runs = [rx.run_baseline(policy, "hard", s, 200) for s in SEEDS]
    learning.append({"label": label, "survival": float(np.mean([r["survival"] for r in runs])),
                     "reward": float(np.mean([r["reward"] for r in runs]))})
for variant, label in (("dqn", "Learned values only (DQN)"), ("anxiety_real", "+ doubt + fear of death")):
    learning.append({"label": label, "survival": mean_final(variant, "survival"),
                     "reward": mean_final(variant, "reward"), "best": variant == "anxiety_real"})


def fault_mean(setting, name, key):
    runs = fx.load(setting, name, SEEDS)
    return float(np.mean([r[key] for r in runs])) if runs else None


faults = []
for name, label in (("anxiety_real", "Fault-blind"), ("metacognitive", "Metacognitive (this page)"),
                    ("oracle", "Told the fault (oracle)")):
    faults.append({"label": label,
                   "mild": fault_mean("mild", name, "survival_fault"),
                   "severe": fault_mean("severe", name, "survival_fault"),
                   "detected": fault_mean("severe", name, "detected_within_15") if name == "metacognitive" else None})
false_alarm = fault_mean("severe", "metacognitive", "false_alarm_within_15")
note = ("Doubt does notice faults: under severe faults it crossed its alarm level within 15 steps "
        f"{fault_mean('severe', 'metacognitive', 'detected_within_15'):.0f}% of the time, against "
        f"{false_alarm:.0f}% on healthy stretches. Noticing did not yet help the agent survive. Even the "
        "oracle, told exactly when the fault began, survived less often than the fault-blind agent, "
        "which simply reacts to the resources it can see. The open problem is teaching the agent to "
        "use what it notices.")

data = {
    "threshold": meta0["doubt_threshold"],
    "fault_text": "In half of all episodes a hidden fault starts at a random step: either the motors "
                  "drain extra power on every work action, or the body turns fragile and random shocks "
                  "become three times as frequent.",
    "episode_order": order,
    "episodes": episodes,
    "results": {"learning": learning, "faults": faults, "fault_note": note, "false_alarm": false_alarm},
}
html = open(os.path.join(HERE, "page_template.html")).read().replace("__DATA__", json.dumps(data))
out = os.path.join(HERE, "doubting_agent.html")
open(out, "w").write(html)
print("wrote", os.path.relpath(out, REPO), f"{len(html) / 1024:.0f} KB")
