"""
Learning Metacognitive Self-Referential Agent (MSRA-L)

Every decision comes from learned components. Nothing about the environment's
costs, thresholds or death conditions is written into the agent.

  * Value learning  - Double DQN over the agent's internal state (r, d, t).
  * Self-model      - an ensemble of probabilistic networks predicting the next
                      internal state, the reward and termination for each action.
  * Imagination     - Dyna-style: the self-model generates imagined transitions
                      that the Q-network also learns from.
  * Doubt           - disagreement between ensemble members (epistemic
                      uncertainty). Random shocks are absorbed by each member's
                      predicted variance (aleatoric), so they do not raise doubt.
  * Anxiety (beta)  - Eq. 10 of the paper. It scales a penalty on actions whose
                      outcome the self-model is unsure of, and discounts imagined
                      rewards the agent cannot trust.
  * Fear of death   - the self-model also learns P(death | s, a). With it the
                      agent weighs dying beta times more than its probability
                      implies, trading some reward for survival.

Variants:
  dqn            value learning only
  msra_no_doubt  + self-model and imagination, doubt ignored
  msra           + doubt-scaled pessimism
  msra_anxiety   + learned fear of death
  anxiety_real   msra_anxiety without imagination (learns from real steps only)
"""

import argparse
import copy
import json
import os
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from main import HomeostasisEnv

ACTION_NAMES = ["REST", "REPAIR", "WORK_HIGH", "WORK_MED", "WORK_LOW"]
N_ACTIONS = 5
STATE_DIM = 3  # resources, degradation, time progress
VARIANTS = ("dqn", "msra_no_doubt", "msra", "msra_anxiety", "anxiety_real")


@dataclass
class Config:
    variant: str = "msra"
    difficulty: str = "hard"
    episodes: int = 400
    seed: int = 0
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 100_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.5
    target_tau: float = 0.005
    ensemble_size: int = 5
    model_hidden: int = 128
    model_lr: float = 1e-3
    model_train_every: int = 2
    model_warmup: int = 2000
    imagined_per_step: int = 32
    imagined_buffer_size: int = 20_000
    real_ratio: float = 0.5
    beta_base: float = 0.5
    lambda_r: float = 0.6
    tau_r: float = 0.3
    lambda_doubt: float = 3.0
    doubt_decay: float = 0.9
    eval_every: int = 20
    eval_episodes: int = 20
    final_eval_episodes: int = 200


def internal_state(obs):
    """obs[0:3] holds resources, degradation and time progress; the rest is noise."""
    return obs[:STATE_DIM].numpy().astype(np.float32)


def mlp(n_in, n_out, hidden=128):
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, n_out),
    )


def one_hot(actions):
    return F.one_hot(torch.as_tensor(actions, dtype=torch.long), N_ACTIONS).float()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.s = np.zeros((capacity, STATE_DIM), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.s2 = np.zeros((capacity, STATE_DIM), np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, s2, done):
        n = len(a)
        idx = (self.ptr + np.arange(n)) % self.capacity
        self.s[idx], self.a[idx], self.r[idx] = s, a, r
        self.s2[idx], self.done[idx] = s2, done
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, n, idx=None):
        if idx is None:
            idx = np.random.randint(0, self.size, n)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.done[idx]

    def __len__(self):
        return self.size


class EnsembleLinear(nn.Module):
    def __init__(self, k, n_in, n_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(k, n_in, n_out) / (2 * n_in ** 0.5))
        self.bias = nn.Parameter(torch.zeros(k, 1, n_out))

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)


class SelfModelEnsemble(nn.Module):
    """Each member predicts a Gaussian over (dr, dd, dt, reward) plus done and death logits."""

    N_OUT = 4

    def __init__(self, k, hidden):
        super().__init__()
        n_in = STATE_DIM + N_ACTIONS
        self.k = k
        self.l1 = EnsembleLinear(k, n_in, hidden)
        self.l2 = EnsembleLinear(k, hidden, hidden)
        self.l3 = EnsembleLinear(k, hidden, 2 * self.N_OUT + 2)
        self.max_logvar = nn.Parameter(torch.full((1, 1, self.N_OUT), 0.5))
        self.min_logvar = nn.Parameter(torch.full((1, 1, self.N_OUT), -10.0))
        self.register_buffer("y_mean", torch.zeros(self.N_OUT))
        self.register_buffer("y_std", torch.ones(self.N_OUT))

    def forward(self, x):
        h = F.silu(self.l1(x))
        h = F.silu(self.l2(h))
        out = self.l3(h)
        mean = out[..., :self.N_OUT]
        logvar = out[..., self.N_OUT:2 * self.N_OUT]
        done_logit = out[..., -2]
        death_logit = out[..., -1]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar, done_logit, death_logit

    def inputs(self, states, actions):
        x = torch.cat([torch.as_tensor(states), one_hot(actions)], dim=-1)
        return x.unsqueeze(0).expand(self.k, -1, -1)


class LearningMSRA:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.use_model = cfg.variant != "dqn"
        self.use_doubt = cfg.variant in ("msra", "msra_anxiety", "anxiety_real")
        self.fears_death = cfg.variant in ("msra_anxiety", "anxiety_real")
        self.imagines = self.use_model and cfg.variant != "anxiety_real"

        self.q = mlp(STATE_DIM, N_ACTIONS)
        self.q_target = copy.deepcopy(self.q)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.real = ReplayBuffer(cfg.buffer_size)

        if self.use_model:
            self.model = SelfModelEnsemble(cfg.ensemble_size, cfg.model_hidden)
            self.model_opt = torch.optim.Adam(self.model.parameters(), lr=cfg.model_lr,
                                              weight_decay=1e-5)
            self.imagined = ReplayBuffer(cfg.imagined_buffer_size)

        self.doubt = 0.0
        self.total_steps = 0
        self.cautious_choices = 0
        self.decisions = 0

    @property
    def model_ready(self):
        return self.use_model and self.total_steps >= self.cfg.model_warmup

    def beta(self, r):
        c = self.cfg
        return c.beta_base + c.lambda_r * np.maximum(0.0, c.tau_r - r) + c.lambda_doubt * self.doubt

    @torch.no_grad()
    def epistemic(self, states):
        """Ensemble disagreement for every action, in reward units. Shape [n, N_ACTIONS]."""
        n = len(states)
        s = np.repeat(states, N_ACTIONS, axis=0)
        a = np.tile(np.arange(N_ACTIONS), n)
        mean, _, _, _ = self.model(self.model.inputs(s, a))
        spread = mean.std(dim=0).norm(dim=-1) * self.model.y_std[3]
        return spread.view(n, N_ACTIONS).numpy()

    @torch.no_grad()
    def death_prob(self, state):
        """Self-predicted probability of dying on the next step, per action."""
        a = np.arange(N_ACTIONS)
        _, _, _, death_logit = self.model(self.model.inputs(np.repeat(state[None], N_ACTIONS, 0), a))
        return torch.sigmoid(death_logit).mean(0).numpy()

    @torch.no_grad()
    def act(self, state, eps):
        q = self.q(torch.as_tensor(state).unsqueeze(0))[0].numpy()
        score = q
        u = None
        if self.use_doubt and self.model_ready:
            beta = self.beta(state[0])
            u = self.epistemic(state[None])[0]
            score = q - beta * u
            if self.fears_death:
                # Q already prices death at its expected cost; beta makes the
                # agent weigh losing everything it could still earn beyond that.
                score = score - beta * self.death_prob(state) * max(q.max(), 0.0)

        if random.random() < eps:
            a = random.randrange(N_ACTIONS)
        else:
            a = int(np.argmax(score))
            self.decisions += 1
            self.cautious_choices += int(a != int(np.argmax(q)))

        if u is not None:
            d = self.cfg.doubt_decay
            self.doubt = d * self.doubt + (1 - d) * float(u[a])
        return a

    def update_q(self):
        c = self.cfg
        if self.model_ready and len(self.imagined) > 0:
            n_real = int(c.batch_size * c.real_ratio)
            real = self.real.sample(n_real)
            imag = self.imagined.sample(c.batch_size - n_real)
            batch = [np.concatenate(pair) for pair in zip(real, imag)]
        else:
            batch = self.real.sample(c.batch_size)
        s, a, r, s2, done = (torch.as_tensor(x) for x in batch)

        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a2 = self.q(s2).argmax(1, keepdim=True)
            target = r + c.gamma * (1 - done) * self.q_target(s2).gather(1, a2).squeeze(1)
        loss = F.smooth_l1_loss(q, target)
        self.q_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.q_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.mul_(1 - c.target_tau).add_(c.target_tau * p)

    def update_model(self):
        m = self.model
        if self.total_steps % 500 == 0 or self.total_steps <= self.cfg.batch_size:
            s, a, r, s2, _ = self.real.sample(0, idx=np.arange(len(self.real)))
            y = np.concatenate([s2 - s, r[:, None]], axis=1)
            m.y_mean.copy_(torch.as_tensor(y.mean(0)))
            m.y_std.copy_(torch.as_tensor(np.maximum(y.std(0), 1e-3)))

        # Each member trains on its own bootstrap sample.
        idx = np.random.randint(0, len(self.real), (m.k, self.cfg.batch_size))
        s, a, r, s2, done = self.real.sample(0, idx=idx.reshape(-1))
        x = torch.cat([torch.as_tensor(s), one_hot(a)], dim=-1).view(m.k, self.cfg.batch_size, -1)
        y = torch.as_tensor(np.concatenate([s2 - s, r[:, None]], axis=1))
        y = ((y - m.y_mean) / m.y_std).view(m.k, self.cfg.batch_size, -1)
        died = torch.as_tensor(done * (s2[:, 2] < 1.0)).view(m.k, self.cfg.batch_size)
        done = torch.as_tensor(done).view(m.k, self.cfg.batch_size)

        mean, logvar, done_logit, death_logit = m(x)
        nll = ((mean - y) ** 2 * torch.exp(-logvar) + logvar).mean()
        done_loss = F.binary_cross_entropy_with_logits(done_logit, done)
        death_loss = F.binary_cross_entropy_with_logits(death_logit, died)
        loss = nll + done_loss + death_loss + 0.01 * (m.max_logvar.sum() - m.min_logvar.sum())
        self.model_opt.zero_grad()
        loss.backward()
        self.model_opt.step()

    @torch.no_grad()
    def imagine(self, n, eps):
        """Branch one imagined step from real states the agent has visited."""
        m = self.model
        s = self.real.sample(n)[0]
        q = self.q(torch.as_tensor(s)).numpy()
        a = q.argmax(1)
        explore = np.random.rand(n) < eps
        a[explore] = np.random.randint(0, N_ACTIONS, explore.sum())

        mean, logvar, done_logit, _ = m(m.inputs(s, a))
        member = torch.randint(0, m.k, (n,))
        rows = torch.arange(n)
        mean, logvar = mean[member, rows], logvar[member, rows]
        y = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        y = (y * m.y_std + m.y_mean).numpy()
        done = (torch.rand(n) < torch.sigmoid(done_logit[member, rows])).numpy()

        s2 = np.clip(s + y[:, :STATE_DIM], 0.0, 1.0).astype(np.float32)
        r = y[:, 3]
        if self.use_doubt:
            u = self.epistemic(s)[rows.numpy(), a]
            r = r - self.beta(s[:, 0]) * u
        self.imagined.add(s, a, r.astype(np.float32), s2, done.astype(np.float32))


def epsilon(cfg, episode):
    frac = min(1.0, episode / max(1, cfg.eps_decay_frac * cfg.episodes))
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def survived(env):
    return env.step_count >= env.max_steps and env.r > 0 and env.d < 1.0


def run_episode(agent, env, eps, learn):
    c = agent.cfg
    s = internal_state(env.reset())
    total, work = 0.0, 0
    while True:
        a = agent.act(s, eps)
        next_obs, reward, _, done, _, _ = env.step(a)
        s2 = internal_state(next_obs)
        total += reward
        work += a >= 2
        if learn:
            agent.real.add(s[None], np.array([a]), np.array([reward], np.float32),
                           s2[None], np.array([float(done)], np.float32))
            agent.total_steps += 1
            if len(agent.real) >= c.batch_size:
                if agent.use_model and agent.total_steps % c.model_train_every == 0:
                    agent.update_model()
                if agent.model_ready and agent.imagines:
                    agent.imagine(c.imagined_per_step, eps)
                agent.update_q()
        s = s2
        if done:
            return total, survived(env), work / env.step_count, env.step_count


def evaluate(agent, difficulty, episodes):
    env = HomeostasisEnv(obs_dim=10, difficulty=difficulty)
    saved = agent.doubt, agent.cautious_choices, agent.decisions
    agent.cautious_choices = agent.decisions = 0
    results = [run_episode(agent, env, eps=0.0, learn=False) for _ in range(episodes)]
    caution = agent.cautious_choices / max(1, agent.decisions)
    agent.doubt, agent.cautious_choices, agent.decisions = saved
    rewards, surv, work, steps = zip(*results)
    return {
        "survival": float(np.mean(surv) * 100),
        "reward": float(np.mean(rewards)),
        "work_pct": float(np.mean(work) * 100),
        "steps": float(np.mean(steps)),
        "caution_pct": float(caution * 100),
    }


def train(cfg: Config, verbose=True):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = HomeostasisEnv(obs_dim=10, difficulty=cfg.difficulty)
    agent = LearningMSRA(cfg)
    curve = []

    for ep in range(cfg.episodes):
        run_episode(agent, env, epsilon(cfg, ep), learn=True)
        if (ep + 1) % cfg.eval_every == 0:
            ev = evaluate(agent, cfg.difficulty, cfg.eval_episodes)
            ev.update(episode=ep + 1, doubt=agent.doubt)
            curve.append(ev)
            if verbose:
                print(f"[{cfg.variant} {cfg.difficulty} seed={cfg.seed}] ep {ep + 1:4d}  "
                      f"survival {ev['survival']:5.1f}%  reward {ev['reward']:6.2f}  "
                      f"work {ev['work_pct']:4.1f}%  caution {ev['caution_pct']:4.1f}%  "
                      f"doubt {agent.doubt:.3f}", flush=True)

    final = evaluate(agent, cfg.difficulty, cfg.final_eval_episodes)
    return {"config": asdict(cfg), "curve": curve, "final": final}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=VARIANTS, default="msra")
    p.add_argument("--difficulty", choices=["normal", "hard", "expert"], default="hard")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write results JSON here")
    args = p.parse_args()

    torch.set_num_threads(1)
    cfg = Config(variant=args.variant, difficulty=args.difficulty,
                 episodes=args.episodes, seed=args.seed)
    result = train(cfg)
    print(json.dumps(result["final"], indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
