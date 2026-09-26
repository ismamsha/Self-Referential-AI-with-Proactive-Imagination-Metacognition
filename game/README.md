# Amoeba Mind

A small browser game that makes the ideas in this repository visible. A single
cell forages a dish of particles. Each particle has a pattern you can see
(colour and shape) and an effect you can't (food, toxin or inert). The cell has
to learn which is which, and it uses the same mechanisms as the agents in
`main.py` and `dsm_benchmark_b_v4.py`.

Open `index.html` in a browser. No build step and no dependencies.

## What you can do

- Click the dish to drop a particle (pick a pattern below the dish, or leave it on *Any*).
- **Shift the world** (`S`): two foods turn toxic and two toxins become food, as in the V4 `--reversal` test.
- **Metacognition** (`M`): off is the `basic` ablation, where no doubt signal reaches the controller.
- **Numb** (`N`): the cell feels the outcome it imagined instead of the real one, so its doubt falls while its real error grows.
- **Show imagination** (`I`), **Show truth** (`T`), speed (`F`), pause (`Space`).

## How it maps to the research

| In the dish | Mechanism | Source |
|---|---|---|
| Dashed lines and ghost labels | Predicts its own next energy and damage for every nearby particle and for resting, then picks the best composite utility | `imagine_futures`, `compute_utility` |
| Energy, damage | Self-state resources and degradation | `SelfState` |
| Anxiety β | Rises near starvation, heavy damage and high doubt; discounts uncertain bites | `compute_effective_beta` |
| Doubt bar, caution chip | Leaky integrator of imagined-vs-real error, with caution-mode hysteresis | `update_metacognition` |
| New brain slot | Novelty-gated imprint growth, one sparse slot per pattern | V4 imprint growth |
| Red ring on a slot, repair flash | Slot doubt (confidence-weighted errors) opens the plasticity gate and triggers REPAIR | V4.1 slot doubt |

## Files

- `sim.js` holds the whole simulation with no DOM, so it runs in Node too.
- `game.js` draws the dish, the brain slots and the chart, and wires up the controls.
- `tools/headless.js` compares metacognition on and off without a browser:

```
node game/tools/headless.js            # 40 seeds, 600 s, shift every 90 s
```

With the current parameters it prints:

```
metacognition on   deaths 0.95  food 36.40  bad bites 12.75  damage in 45 s after a shift 0.23  repairs 2.63
metacognition off  deaths 0.70  food 45.23  bad bites 16.32  damage in 45 s after a shift 0.31  repairs 0.00
```

Metacognition cuts bad bites by about a fifth and damage after a shift by about
a quarter, but the cautious cell also eats less and starves a little more
often. The prototype is tuned for a readable demo, not for these numbers.
