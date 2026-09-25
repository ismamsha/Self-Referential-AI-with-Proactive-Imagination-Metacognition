"""
Best possible survival once a hidden fault has struck.

Dynamic programming over the known body dynamics gives the highest survival probability any
policy could reach, even one that knows the fault the moment it starts and meets it at full
health (resources 1, damage 0). If the agents already sit near this ceiling, no amount of
metacognition can raise their survival in this environment.

Usage: python survival_ceiling.py [--faults severe]
"""

import argparse

import numpy as np

from main import HomeostasisEnv
from fault_experiment import FAULT_SETTINGS, ONSET

GRID = 201
MAX_STEPS = 100
INTENSITY = {2: 1.0, 3: 0.7, 4: 0.4}


def interpolate(V, r, d):
    """Bilinear lookup of V (on the r x d grid) at arbitrary points; dead points score 0."""
    x = np.clip(r, 0.0, 1.0) * (GRID - 1)
    y = np.clip(d, 0.0, 1.0) * (GRID - 1)
    x0 = np.minimum(np.floor(x).astype(int), GRID - 2)
    y0 = np.minimum(np.floor(y).astype(int), GRID - 2)
    fx, fy = x - x0, y - y0
    v = (V[x0, y0] * (1 - fx) * (1 - fy) + V[x0 + 1, y0] * fx * (1 - fy)
         + V[x0, y0 + 1] * (1 - fx) * fy + V[x0 + 1, y0 + 1] * fx * fy)
    return np.where((r <= 0) | (d >= 1.0), 0.0, v)


def after_action(env, r, d, a, rest_recovery):
    """Resources and damage after the action, starting from the post-shock state."""
    if a == 0:
        return np.minimum(1.0, r + rest_recovery * (1.0 - d)), d
    if a == 1:
        can = r >= env.repair_cost
        return np.where(can, r - env.repair_cost, r), np.where(can, np.maximum(0.0, d - 0.1), d)
    k = INTENSITY[a]
    cost = k * 0.1 * env.work_resource_multiplier
    ok = r >= cost
    good = (r > 0.6) & (d < 0.3)
    cost = np.where(good, k * 0.08 * (1.0 + d), cost)
    d_work = np.minimum(1.0, d + k * 0.01 * env.work_degradation_multiplier)
    return np.where(ok, r - cost, r), np.where(ok, d_work, d)


def survival_table(faults, fault_type):
    """V[t][r, d]: best probability of reaching step 100 alive from state (r, d) at step t,
    with the fault active throughout."""
    env = HomeostasisEnv(difficulty="hard")
    setting = FAULT_SETTINGS[faults]
    shock_p = env.shock_probability * (setting["shock_factor"] if fault_type == "fragile" else 1.0)
    rest_recovery = env.rest_recovery * (setting["battery_factor"] if fault_type == "battery" else 1.0)
    g = np.linspace(0.0, 1.0, GRID)
    R, D = np.meshgrid(g, g, indexing="ij")
    V = [None] * (MAX_STEPS + 1)
    V[MAX_STEPS] = np.where((R > 0) & (D < 1.0), 1.0, 0.0)
    for t in range(MAX_STEPS - 1, -1, -1):
        best = np.zeros_like(R)
        for a in range(5):
            r = R.copy()
            if fault_type == "actuator" and a >= 2:
                r = np.maximum(0.0, r - setting["actuator_drain"] * INTENSITY[a])
            value = 0.0
            for shocked, prob in ((True, shock_p), (False, 1.0 - shock_p)):
                rs = np.maximum(0.0, r - env.shock_resource_loss) if shocked else r
                ds = np.minimum(1.0, D + env.shock_degradation) if shocked else D
                r2, d2 = after_action(env, rs, ds, a, rest_recovery)
                value = value + prob * interpolate(V[t + 1], r2, d2)
            best = np.maximum(best, value)
        V[t] = best
    return V


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--faults", choices=list(FAULT_SETTINGS), default="severe")
    args = p.parse_args()
    onsets = range(ONSET[0], ONSET[1] + 1)
    per_type = {}
    for fault_type in FAULT_SETTINGS[args.faults]["types"]:
        V = survival_table(args.faults, fault_type)
        per_type[fault_type] = float(np.mean([V[t][GRID - 1, 0] for t in onsets]))
        print(f"{fault_type:9s} best possible survival, meeting the fault at full health: "
              f"{100 * per_type[fault_type]:.1f}%")
    print(f"all fault lives (fault types equally likely): {100 * np.mean(list(per_type.values())):.1f}%")


if __name__ == "__main__":
    main()
