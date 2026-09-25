"""
Evolved plastic organism: the metacognitive loop and its learning rule inside one network.

Nothing is trained by backpropagation. Each organism carries a genome, and evolution
(OpenAI-ES) shapes the genome over generations. During a life the network changes itself
using rules that are also written in the genome:

  network        senses (resources, damage, time, doubt) -> 32 tanh units -> 5 actions
  plasticity     every connection has a fixed part and a plastic part:
                   w = W + A * Hebb,   Hebb += eta * m * pre * post   (clipped to [-1, 1])
                 a three-factor rule: presynaptic activity, postsynaptic activity and a
                 neuromodulator m (Miconi et al., 2019, "Backpropamine")
  self-model     a readout from the hidden layer predicts the change in resources and
                 damage for every action; it learns during life by the delta rule
  doubt          a running average of the self-model's signed prediction error, fed
                 back into the network as input
  neuromodulator m = tanh(u . [doubt, |doubt|, 1]); like noradrenaline, it lets surprise
                 switch rewiring on, off or into reverse

W, A, the learning rates, the self-model's starting weights, doubt's gain and memory and
the neuromodulator's weights u are all inherited.

Organisms (each evolved with the same budget):
  innate         fixed wiring, no doubt, no plasticity
  innate_oracle  fixed wiring plus the true fault flag as input
  innate_doubt   fixed wiring; the self-model learns during life and doubt is an input
  plastic        plastic wiring with a constant neuromodulator; no self-model or doubt
  organism       everything: self-model, doubt, and doubt-driven plasticity
  rules_only     the genome holds only learning rules, no weights: each life starts from
                 random wiring and an empty self-model, and every neuron's incoming
                 connections change by its inherited rule,
                   Hebb += eta * m * (A*pre*post + B*pre + C*post + D)

Usage:
  python organism.py --check                   verify the vectorised environment
  python organism.py [--faults severe] [--generations 400] [--seeds 0 1 2]
"""

import os
# Set before numpy loads: every worker process runs on a single thread.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import random
import time
from multiprocessing import Pool

import numpy as np

from main import HomeostasisEnv
from fault_experiment import FAULT_SETTINGS, ONSET, FaultyHomeostasisEnv, BEFORE, AFTER

ROOT = os.path.dirname(os.path.abspath(__file__))
N_ACTIONS = 5
HIDDEN = 32
MAX_STEPS = 100
INTENSITY = np.array([0.0, 0.0, 1.0, 0.7, 0.4])
VARIANTS = {
    "innate": "Innate wiring",
    "innate_oracle": "Innate wiring + told the fault",
    "innate_doubt": "Innate wiring + self-model and doubt",
    "plastic": "Plastic wiring, no doubt",
    "organism": "Full organism: doubt steers plasticity",
    "rules_only": "Rules only: born with random wiring",
}
EXTRA_INPUTS = {"innate": 0, "innate_oracle": 1, "innate_doubt": 2, "plastic": 0, "organism": 2,
                "rules_only": 2}
PLASTIC = ("plastic", "organism", "rules_only")
SELF_MODEL = ("innate_doubt", "organism", "rules_only")


def results_dir(faults):
    return os.path.join(ROOT, "results", f"organism_{faults}")


# ---------------------------------------------------------------------------
# Vectorised environment
# ---------------------------------------------------------------------------

class Scenarios:
    """Everything random about a batch of lives, drawn up front so that every organism
    in a generation meets exactly the same lives."""

    def __init__(self, rng, n, faults, fault_prob=0.5):
        types = FAULT_SETTINGS[faults]["types"]
        self.faults = faults
        self.n = n
        has = rng.random(n) < fault_prob
        self.kind = np.where(has, rng.integers(0, len(types), n), -1)
        self.onset = np.where(has, rng.integers(ONSET[0], ONSET[1] + 1, n), 10 ** 6)
        self.u = rng.random((n, MAX_STEPS))
        self.birth_seed = int(rng.integers(2 ** 31))

    def fault_name(self, i):
        return FAULT_SETTINGS[self.faults]["types"][self.kind[i]] if self.kind[i] >= 0 else None


class VecFaultEnv:
    """FaultyHomeostasisEnv's dynamics for many lives at once (see check_env)."""

    def __init__(self, scen, repeat=1, difficulty="hard"):
        ref = HomeostasisEnv(difficulty=difficulty)
        for k in ("shock_probability", "shock_resource_loss", "shock_degradation", "rest_recovery",
                  "repair_cost", "work_resource_multiplier", "work_degradation_multiplier",
                  "danger_threshold_low", "danger_threshold_high"):
            setattr(self, k, getattr(ref, k))
        setting = FAULT_SETTINGS[scen.faults]
        types = setting["types"]
        self.actuator_drain = setting["actuator_drain"]
        self.battery_factor = setting["battery_factor"]
        self.shock_factor = setting["shock_factor"]
        code = lambda name: types.index(name) if name in types else -99
        self.kind = np.tile(scen.kind, repeat)
        self.onset = np.tile(scen.onset, repeat)
        self.u = np.tile(scen.u, (repeat, 1))
        self.is_actuator = self.kind == code("actuator")
        self.is_battery = self.kind == code("battery")
        self.is_fragile = self.kind == code("fragile")
        n = len(self.kind)
        self.n = n
        self.r = np.ones(n)
        self.d = np.zeros(n)
        self.k = 0
        self.active = np.zeros(n, bool)
        self.alive = np.ones(n, bool)
        self.survived = np.zeros(n, bool)

    def step(self, a):
        live = self.alive.copy()
        k = self.k
        self.active |= live & (self.kind >= 0) & (k >= self.onset)
        rest_recovery = np.where(self.active & self.is_battery,
                                 self.rest_recovery * self.battery_factor, self.rest_recovery)
        shock_p = np.where(self.active & self.is_fragile,
                           self.shock_probability * self.shock_factor, self.shock_probability)
        intensity = INTENSITY[a]
        work = a >= 2
        r, d = self.r.copy(), self.d.copy()

        drain = live & self.active & self.is_actuator & work
        r = np.where(drain, np.maximum(0.0, r - self.actuator_drain * intensity), r)
        shocked = live & (self.u[:, k] < shock_p)
        r = np.where(shocked, np.maximum(0.0, r - self.shock_resource_loss), r)
        d = np.where(shocked, np.minimum(1.0, d + self.shock_degradation), d)
        reward = np.zeros(self.n)

        rest = live & (a == 0)
        r_rest = np.minimum(1.0, r + rest_recovery * (1.0 - d))
        reward = np.where(rest & (r_rest < 0.4), 0.3, reward)
        r = np.where(rest, r_rest, r)

        repair = live & (a == 1)
        can = repair & (r >= self.repair_cost)
        d_rep = np.maximum(0.0, d - 0.1)
        reward = np.where(can, np.where(d_rep > 0.4, 0.4, 0.1), reward)
        reward = np.where(repair & ~can, -0.2, reward)
        r = np.where(can, r - self.repair_cost, r)
        d = np.where(can, d_rep, d)

        wk = live & work
        cost = intensity * 0.1 * self.work_resource_multiplier
        ok = wk & (r >= cost)
        good = (r > 0.6) & (d < 0.3)
        base = intensity * (1.0 + 0.5 * r) + np.where(good, 0.3, 0.0)
        cost = np.where(good, intensity * 0.08 * (1.0 + d), cost)
        r_work = r - cost
        d_work = np.minimum(1.0, d + intensity * 0.01 * self.work_degradation_multiplier)
        work_reward = (base - np.where(r_work < self.danger_threshold_low, 0.3, 0.0)
                       - np.where(d_work > self.danger_threshold_high, 0.4, 0.0))
        reward = np.where(ok, work_reward, reward)
        reward = np.where(wk & ~ok, -0.5, reward)
        r = np.where(ok, r_work, r)
        d = np.where(ok, d_work, d)

        r = np.clip(r, 0.0, 1.0)
        d = np.clip(d, 0.0, 1.0)
        died_r = live & (r <= 0)
        died_d = live & ~died_r & (d >= 1.0)
        finished = live & ~died_r & ~died_d & (k + 1 >= MAX_STEPS)
        reward = reward - 5.0 * died_r - 4.0 * died_d + np.where(finished, 2.0 + 2.0 * r, 0.0)

        self.r = np.where(live, r, self.r)
        self.d = np.where(live, d, self.d)
        self.alive = live & ~(died_r | died_d | finished)
        self.survived |= finished
        self.k += 1
        return np.where(live, reward, 0.0), live


def check_env(episodes=300, faults=("mild", "severe")):
    """Replay random action sequences through both environments with identical
    randomness and compare every step."""
    worst = 0.0
    act_rng = np.random.default_rng(0)
    for setting in faults:
        types = FAULT_SETTINGS[setting]["types"]
        for ep in range(episodes):
            random.seed(ep)
            env = FaultyHomeostasisEnv(faults=setting)
            env.reset()
            kind = ep % (len(types) + 1) - 1
            env.fault_type = types[kind] if kind >= 0 else None
            env.fault_onset = int(act_rng.integers(ONSET[0], ONSET[1] + 1)) if kind >= 0 else None
            state = random.getstate()
            u = [random.random() for _ in range(MAX_STEPS)]
            random.setstate(state)

            scen = Scenarios(np.random.default_rng(0), 1, setting)
            scen.kind[0] = kind
            scen.onset[0] = env.fault_onset if kind >= 0 else 10 ** 6
            scen.u[0] = u
            vec = VecFaultEnv(scen)
            actions = act_rng.integers(0, N_ACTIONS, MAX_STEPS)
            for t in range(MAX_STEPS):
                _, reward, _, done, _, _ = env.step(int(actions[t]))
                v_reward, _ = vec.step(np.array([actions[t]]))
                worst = max(worst, abs(env.r - vec.r[0]), abs(env.d - vec.d[0]), abs(reward - v_reward[0]))
                assert done == (not vec.alive[0]), (setting, ep, t)
                if done:
                    break
    print(f"vectorised environment matches the original: largest difference {worst:.2e}")
    return worst


# ---------------------------------------------------------------------------
# Genome and life
# ---------------------------------------------------------------------------

def genome_spec(variant):
    n_in = 3 + EXTRA_INPUTS[variant] + 1
    if variant == "rules_only":
        spec = [("rule1", (4, HIDDEN)), ("rule2", (4, N_ACTIONS)), ("scale", (2,))]
    else:
        spec = [("W1", (n_in, HIDDEN)), ("W2", (HIDDEN, N_ACTIONS))]
        if variant in PLASTIC:
            spec += [("A1", (n_in, HIDDEN)), ("A2", (HIDDEN, N_ACTIONS))]
    if variant in PLASTIC:
        spec += [("eta", (2,)), ("mod", (5,))]
    if variant in SELF_MODEL:
        if variant != "rules_only":
            spec += [("P", (HIDDEN + n_in, 2 * N_ACTIONS))]
        spec += [("eta_p", (1,)), ("gain", (2,)), ("mem", (1,))]
    return spec, n_in


def init_genome(spec, rng):
    parts = []
    for name, shape in spec:
        if name in ("W1", "W2"):
            parts.append(rng.standard_normal(shape) / np.sqrt(shape[0]))
        elif name in ("rule1", "rule2"):
            parts.append(0.1 * rng.standard_normal(shape))
        elif name == "eta":
            parts.append(np.full(shape, -2.0))
        elif name == "mod":
            parts.append(np.array([0.0, 0.0, 0.0, 0.0, 0.5]))
        elif name == "eta_p":
            parts.append(np.full(shape, -1.0))
        elif name == "gain":
            parts.append(np.log([10.0, 50.0]))
        elif name == "mem":
            parts.append(np.full(shape, 1.7))
        else:
            parts.append(np.zeros(shape))
    return np.concatenate([p.ravel() for p in parts])


def unpack(thetas, spec):
    out, i = {}, 0
    for name, shape in spec:
        size = int(np.prod(shape))
        out[name] = thetas[:, i:i + size].reshape((len(thetas),) + shape)
        i += size
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def hebbian_rule(coef, pre, post):
    """Each postsynaptic neuron's inherited rule A*pre*post + B*pre + C*post + D."""
    A, B, C, D = (coef[:, i, None, :] for i in range(4))
    return A * pre[:, :, None] * post[:, None, :] + B * pre[:, :, None] + C * post[:, None, :] + D


def live(thetas, variant, scen, record=False):
    """Run each genome in `thetas` [pop, D] through every life in `scen`.
    Returns total reward and survival, both shaped [pop, lives]."""
    spec, n_in = genome_spec(variant)
    pop, lives = len(thetas), scen.n
    g = {k: np.repeat(v, lives, axis=0) for k, v in unpack(thetas, spec).items()}
    env = VecFaultEnv(scen, repeat=pop)
    B = env.n
    rows = np.arange(B)
    plastic, self_model, rules = variant in PLASTIC, variant in SELF_MODEL, variant == "rules_only"
    if rules:
        birth = np.random.default_rng(scen.birth_seed)
        w1 = birth.standard_normal((lives, n_in, HIDDEN)) / np.sqrt(n_in)
        w2 = birth.standard_normal((lives, HIDDEN, N_ACTIONS)) / np.sqrt(HIDDEN)
        birth1 = np.tile(w1, (pop, 1, 1)) * np.exp(g["scale"][:, 0])[:, None, None]
        birth2 = np.tile(w2, (pop, 1, 1)) * np.exp(g["scale"][:, 1])[:, None, None]
    if plastic:
        H1 = np.zeros((B, n_in, HIDDEN))
        H2 = np.zeros((B, HIDDEN, N_ACTIONS))
        eta = 0.5 * sigmoid(g["eta"])
    if self_model:
        P = g["P"].copy() if "P" in g else np.zeros((B, HIDDEN + n_in, 2 * N_ACTIONS))
        eta_p = 0.5 * sigmoid(g["eta_p"][:, 0])[:, None]
        gain = np.exp(g["gain"])
        memory = sigmoid(g["mem"][:, 0])[:, None]
    doubt = np.zeros((B, 2))
    total = np.zeros(B)
    trace = {k: [] for k in ("r", "d", "action", "doubt", "mod")} if record else None

    for _ in range(MAX_STEPS):
        if not env.alive.any():
            break
        cols = [env.r, env.d, np.full(B, env.k / MAX_STEPS)]
        if variant == "innate_oracle":
            cols.append((env.active | ((env.kind >= 0) & (env.k >= env.onset))).astype(float))
        if self_model:
            cols += [doubt[:, 0], doubt[:, 1]]
        x = np.stack(cols + [np.ones(B)], axis=1)

        if rules:
            W1, W2 = birth1 + H1, birth2 + H2
        elif plastic:
            W1, W2 = g["W1"] + g["A1"] * H1, g["W2"] + g["A2"] * H2
        else:
            W1, W2 = g["W1"], g["W2"]
        h = np.tanh(np.einsum("bi,bij->bj", x, W1))
        a = np.einsum("bj,bjk->bk", h, W2).argmax(1)
        if self_model:
            feat = np.concatenate([h, x], axis=1)
            pred = np.einsum("bf,bfk->bk", feat, P)
        r0, d0 = env.r.copy(), env.d.copy()
        if record:
            trace["r"].append(r0)
            trace["d"].append(d0)
            trace["action"].append(np.where(env.alive, a, -1))
            trace["doubt"].append(np.linalg.norm(doubt, axis=1))

        reward, lived = env.step(a)
        total += reward
        mask = lived[:, None].astype(float)

        if self_model:
            err = np.stack([env.r - r0 - pred[rows, a], env.d - d0 - pred[rows, N_ACTIONS + a]], axis=1)
            err = np.clip(err, -1.0, 1.0) * mask
            P[rows, :, a] += eta_p * feat * err[:, :1]
            P[rows, :, N_ACTIONS + a] += eta_p * feat * err[:, 1:]
            doubt = memory * doubt + (1.0 - memory) * err * gain
        m = np.zeros(B)
        if plastic:
            z = np.concatenate([doubt, np.abs(doubt), np.ones((B, 1))], axis=1)
            m = np.tanh((z * g["mod"]).sum(1)) * lived
            post = np.eye(N_ACTIONS)[a]
            if rules:
                dH1 = hebbian_rule(g["rule1"], x, h)
                dH2 = hebbian_rule(g["rule2"], h, post)
            else:
                dH1 = x[:, :, None] * h[:, None, :]
                dH2 = h[:, :, None] * post[:, None, :]
            H1 = np.clip(H1 + (eta[:, 0] * m)[:, None, None] * dH1, -1.0, 1.0)
            H2 = np.clip(H2 + (eta[:, 1] * m)[:, None, None] * dH2, -1.0, 1.0)
        if record:
            trace["mod"].append(m)

    shape = (pop, lives)
    out = {"reward": total.reshape(shape), "survived": env.survived.reshape(shape)}
    if record:
        out["trace"] = {k: np.array(v).T for k, v in trace.items()}
        out["trace"]["r_end"], out["trace"]["d_end"] = env.r, env.d
    return out


# ---------------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, dim, lr, b1=0.9, b2=0.999):
        self.lr, self.b1, self.b2 = lr, b1, b2
        self.m = np.zeros(dim)
        self.v = np.zeros(dim)
        self.t = 0

    def step(self, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def centred_ranks(x):
    ranks = np.empty(len(x))
    ranks[np.argsort(x)] = np.arange(len(x))
    return ranks / (len(x) - 1) - 0.5


def split_by_fault(scen, values):
    fault = scen.kind >= 0
    return values[:, ~fault].mean(), values[:, fault].mean()


def evaluate(theta, variant, scen):
    out = live(theta[None], variant, scen)
    healthy_s, fault_s = split_by_fault(scen, out["survived"])
    healthy_r, fault_r = split_by_fault(scen, out["reward"])
    res = {"survival_healthy": 100 * healthy_s, "survival_fault": 100 * fault_s,
           "reward_healthy": healthy_r, "reward_fault": fault_r,
           "survival": 100 * out["survived"].mean(), "reward": out["reward"].mean()}
    for i, name in enumerate(FAULT_SETTINGS[scen.faults]["types"]):
        res[f"survival_{name}"] = 100 * out["survived"][0, scen.kind == i].mean()
    return {k: float(v) for k, v in res.items()}


def aligned(values, onset):
    out = np.full(BEFORE + AFTER, np.nan)
    for i in range(-BEFORE, AFTER):
        if 0 <= onset + i < len(values):
            out[i + BEFORE] = values[onset + i]
    return out


def behaviour(theta, variant, scen):
    """Average traces around fault onset, plus three whole example lives."""
    out = live(theta[None], variant, scen, record=True)
    tr = out["trace"]
    fault_idx = np.where(scen.kind >= 0)[0]
    res = {}
    for key in ("mod", "doubt"):
        res[f"{key}_trace"] = np.nanmean([aligned(tr[key][i], scen.onset[i]) for i in fault_idx], axis=0).tolist()
        healthy = tr[key][scen.kind < 0]
        res[f"{key}_healthy_mean"] = float(np.mean(healthy[tr["action"][scen.kind < 0] >= 0]))
    work = (tr["action"] >= 2).astype(float)
    work[tr["action"] < 0] = np.nan
    res["work_trace"] = np.nanmean([aligned(work[i], scen.onset[i]) for i in fault_idx], axis=0).tolist()
    examples = {}
    for kind in [-1] + list(range(len(FAULT_SETTINGS[scen.faults]["types"]))):
        idx = np.where(scen.kind == kind)[0]
        if len(idx):
            i = idx[0]
            n = int((tr["action"][i] >= 0).sum())
            examples[scen.fault_name(i) or "healthy"] = {
                "fault": scen.fault_name(i), "onset": int(scen.onset[i]) if scen.kind[i] >= 0 else None,
                "survived": bool(out["survived"][0, i]),
                "actions": tr["action"][i, :n].tolist(),
                "r": np.round(np.append(tr["r"][i, :n], tr["r_end"][i]), 4).tolist(),
                "d": np.round(np.append(tr["d"][i, :n], tr["d_end"][i]), 4).tolist(),
                "doubt": np.round(tr["doubt"][i, :n], 4).tolist(),
                "mod": np.round(tr["mod"][i, :n], 4).tolist(),
            }
    res["examples"] = examples
    return res


def evolve(variant, seed, faults, generations, pop, lives, sigma, lr, log_every, verbose=True):
    rng = np.random.default_rng(seed)
    spec, _ = genome_spec(variant)
    theta = init_genome(spec, rng)
    adam = Adam(len(theta), lr)
    validation = Scenarios(np.random.default_rng(10_000 + seed), 256, faults)
    curve = []
    start = time.time()
    for gen in range(generations + 1):
        if gen % log_every == 0:
            ev = evaluate(theta, variant, validation)
            ev["generation"] = gen
            curve.append(ev)
            if verbose:
                print(f"[{variant} {faults} seed={seed}] gen {gen:4d}  survival {ev['survival']:5.1f}% "
                      f"(fault {ev['survival_fault']:5.1f}%)  reward {ev['reward']:6.1f}  "
                      f"{time.time() - start:5.0f}s", flush=True)
        if gen == generations:
            break
        eps = rng.standard_normal((pop // 2, len(theta)))
        thetas = np.concatenate([theta + sigma * eps, theta - sigma * eps])
        fitness = live(thetas, variant, Scenarios(rng, lives, faults))["reward"].mean(1)
        shaped = centred_ranks(fitness)
        grad = (shaped[:pop // 2] - shaped[pop // 2:]) @ eps / (pop * sigma)
        theta = theta + adam.step(grad - 0.005 * theta)

    test = Scenarios(np.random.default_rng(20_000 + seed), 600, faults)
    result = {"variant": variant, "seed": seed, "faults": faults, "generations": generations,
              "population": pop, "lives_per_genome": lives, "parameters": len(theta),
              "final": evaluate(theta, variant, test), "curve": curve,
              "behaviour": behaviour(theta, variant, test), "genome": theta.tolist()}
    if variant in PLASTIC or variant in SELF_MODEL:
        g = unpack(theta[None], spec)
        result["evolved"] = {}
        if variant in PLASTIC:
            result["evolved"]["hebbian_rates"] = (0.5 * sigmoid(g["eta"][0])).tolist()
            result["evolved"]["modulator_weights"] = g["mod"][0].tolist()
        if variant in SELF_MODEL:
            result["evolved"]["self_model_rate"] = float(0.5 * sigmoid(g["eta_p"][0, 0]))
            result["evolved"]["doubt_gain"] = np.exp(g["gain"][0]).tolist()
            result["evolved"]["doubt_memory"] = float(sigmoid(g["mem"][0, 0]))
    return result


def run_job(job):
    variant, seed, args = job
    path = os.path.join(results_dir(args["faults"]), f"{variant}_seed{seed}.json")
    if os.path.exists(path):
        return path
    result = evolve(variant, seed, args["faults"], args["generations"], args["pop"], args["lives"],
                    args["sigma"], args["lr"], args["log_every"], verbose=False)
    with open(path, "w") as f:
        json.dump(result, f)
    fin = result["final"]
    print(f"done {args['faults']} {variant:14s} seed {seed}: survival healthy {fin['survival_healthy']:5.1f}%  "
          f"fault {fin['survival_fault']:5.1f}%  reward {fin['reward']:6.1f}", flush=True)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

COLORS = {"innate": "#2a78d6", "innate_oracle": "#eb6834", "innate_doubt": "#1baf7a",
          "plastic": "#eda100", "organism": "#e87ba4", "rules_only": "#008300"}
MARKERS = {"innate": "o", "innate_oracle": "s", "innate_doubt": "^", "plastic": "D",
           "organism": "v", "rules_only": "P"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
T_975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}


def load(faults, variant, seeds):
    runs = []
    for s in seeds:
        path = os.path.join(results_dir(faults), f"{variant}_seed{s}.json")
        if os.path.exists(path):
            with open(path) as f:
                runs.append(json.load(f))
    return runs


def fmt(values, unit=""):
    v = np.asarray(values, float)
    if len(v) < 2:
        return f"{v.mean():.1f}{unit}"
    ci = T_975.get(len(v) - 1, 1.96) * v.std(ddof=1) / np.sqrt(len(v))
    return f"{v.mean():.1f} ± {ci:.1f}{unit}"


def rl_reference(faults, seeds):
    """The fault-blind reinforcement-learning agent from fault_experiment.py."""
    from fault_experiment import load as load_rl
    runs = load_rl(faults, "anxiety_real", seeds)
    if not runs:
        return None
    return {k: float(np.mean([r[k] for r in runs])) for k in ("survival_healthy", "survival_fault")}


def write_report(faults, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    types = FAULT_SETTINGS[faults]["types"]
    lines = [f"# Evolved organisms: {faults} faults", "",
             "Hard difficulty, a hidden fault in half of all lives. OpenAI-ES, 400 generations, "
             "population 128, 32 lives per genome per generation. Final test: 600 new lives. "
             f"Mean ± 95% CI over {len(seeds)} seeds.", "",
             "| Organism | Genome size | Survival, healthy | Survival, fault | "
             + " | ".join(f"{t.capitalize()} fault" for t in types) + " | Reward |",
             "|---|---|---|---|" + "---|" * len(types) + "---|"]
    for v, label in VARIANTS.items():
        runs = load(faults, v, seeds)
        if not runs:
            continue
        get = lambda k: [r["final"][k] for r in runs]
        per_type = " | ".join(fmt(get(f"survival_{t}"), "%") for t in types)
        lines.append(f"| {label} | {runs[0]['parameters']} | {fmt(get('survival_healthy'), '%')} | "
                     f"{fmt(get('survival_fault'), '%')} | {per_type} | {fmt(get('reward'))} |")
    ref = rl_reference(faults, seeds)
    if ref:
        lines += ["", f"For reference, the fault-blind reinforcement-learning agent from fault_experiment.py "
                  f"survived {ref['survival_healthy']:.1f}% of healthy and {ref['survival_fault']:.1f}% of "
                  "faulty lives in the same setting (different test lives, same distribution)."]
    evolved = [(v, r["evolved"]) for v in PLASTIC for r in load(faults, v, seeds) if "evolved" in r]
    if evolved:
        lines += ["", "## Evolved learning rules", "",
                  "| Organism | Seed | Hebbian rates (layer 1, 2) | Modulator weights "
                  "(doubt r, doubt d, abs r, abs d, bias) | Self-model rate |", "|---|---|---|---|---|"]
        for v in PLASTIC:
            for r in load(faults, v, seeds):
                e = r["evolved"]
                rates = ", ".join(f"{x:.3f}" for x in e["hebbian_rates"])
                mod = ", ".join(f"{x:+.2f}" for x in e["modulator_weights"])
                sm = f"{e['self_model_rate']:.3f}" if "self_model_rate" in e else "–"
                lines.append(f"| {VARIANTS[v]} | {r['seed']} | {rates} | {mod} | {sm} |")
    with open(os.path.join(results_dir(faults), "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"width_ratios": [1, 1.15, 1]})
    ax = axes[0]
    for v in VARIANTS:
        runs = load(faults, v, seeds)
        if not runs:
            continue
        x = [c["generation"] for c in runs[0]["curve"]]
        y = np.array([[c["survival"] for c in r["curve"]] for r in runs])
        ax.plot(x, y.mean(0), color=COLORS[v], lw=2, marker=MARKERS[v], markersize=5, markevery=4,
                label=VARIANTS[v])
    ax.set_ylim(-2, 102)
    ax.set_xlabel("Generation", color=INK)
    ax.set_ylabel("Survival on validation lives (%)", color=INK)
    ax.set_title("Evolution", color=INK, fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=8)

    ax = axes[1]
    names = [v for v in VARIANTS if load(faults, v, seeds)]
    for i, v in enumerate(names):
        runs = load(faults, v, seeds)
        healthy = np.mean([r["final"]["survival_healthy"] for r in runs])
        fault = np.mean([r["final"]["survival_fault"] for r in runs])
        y = len(names) - 1 - i
        ax.plot([fault, healthy], [y, y], color=MUTED, lw=1.5, zorder=1)
        ax.scatter(healthy, y, s=70, facecolor="white", edgecolor=INK, lw=1.8, zorder=2,
                   label="Healthy lives" if i == 0 else None)
        ax.scatter(fault, y, s=70, color=INK, zorder=3, label="Lives with a fault" if i == 0 else None)
        ax.annotate(f"{fault:.0f}%", (fault, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    if ref:
        ax.axvline(ref["survival_fault"], color=MUTED, lw=1.2, ls="--")
        ax.annotate("RL agent, lives with a fault", xy=(ref["survival_fault"], 1.0),
                    xycoords=("data", "axes fraction"), xytext=(4, -12), textcoords="offset points",
                    fontsize=8, color=MUTED)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([VARIANTS[v] for v in reversed(names)], fontsize=9, color=INK)
    ax.set_xlim(-2, 104)
    ax.set_ylim(-0.6, len(names) - 0.2)
    ax.set_xlabel("Survival on 600 test lives (%)", color=INK)
    ax.set_title("Survival with and without a hidden fault", color=INK, fontsize=11)
    ax.legend(loc="lower left", frameon=False, fontsize=9)

    ax = axes[2]
    x = np.arange(-BEFORE, AFTER)
    for v in PLASTIC:
        runs = load(faults, v, seeds)
        if not runs:
            continue
        trace = np.nanmean([r["behaviour"]["mod_trace"] for r in runs], axis=0)
        ax.plot(x, trace, color=COLORS[v], lw=2, marker=MARKERS[v], markersize=5, markevery=5,
                label=VARIANTS[v])
    ax.axvline(0, color=MUTED, lw=1)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Steps since the hidden fault began", color=INK)
    ax.set_ylabel("Neuromodulator m (+ strengthens, − reverses)", color=INK)
    ax.set_title("What the evolved neuromodulator does", color=INK, fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
    fig.text(0.5, 0.01, f"{faults.capitalize()} faults, hard difficulty; means over {len(seeds)} seeds.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(results_dir(faults), "organism_results.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="verify the vectorised environment and exit")
    p.add_argument("--report", action="store_true", help="only rewrite the summary and figure")
    p.add_argument("--faults", choices=list(FAULT_SETTINGS), default="severe")
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--generations", type=int, default=400)
    p.add_argument("--pop", type=int, default=128)
    p.add_argument("--lives", type=int, default=32)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.check:
        check_env()
        return
    os.makedirs(results_dir(args.faults), exist_ok=True)
    if not args.report:
        cfg = {k: getattr(args, k) for k in ("faults", "generations", "pop", "lives", "sigma", "lr", "log_every")}
        jobs = [(v, s, cfg) for s in args.seeds for v in args.variants]
        with Pool(args.workers) as pool:
            pool.map(run_job, jobs, chunksize=1)
    write_report(args.faults, args.seeds)
    print(f"wrote results/organism_{args.faults}/summary.md and organism_results.png")


if __name__ == "__main__":
    main()
