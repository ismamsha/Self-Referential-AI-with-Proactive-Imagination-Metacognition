# Evolved organisms: severe faults

Hard difficulty, a hidden fault in half of all lives. OpenAI-ES, 400 generations, population 128, 32 lives per genome per generation. Final test: 600 new lives. Mean ± 95% CI over 3 seeds.

| Organism | Genome size | Survival, healthy | Survival, fault | Actuator fault | Fragile fault | Reward |
|---|---|---|---|---|---|---|
| Innate wiring | 288 | 37.7 ± 132.8% | 26.0 ± 99.4% | 37.4 ± 134.3% | 14.3 ± 61.7% | 31.5 ± 68.3 |
| Innate wiring + told the fault | 320 | 68.4 ± 128.3% | 52.0 ± 94.8% | 71.7 ± 117.8% | 32.3 ± 69.6% | 52.5 ± 34.1 |
| Innate wiring + self-model and doubt | 736 | 99.8 ± 0.5% | 71.6 ± 6.9% | 100.0 ± 0.0% | 44.2 ± 7.7% | 55.4 ± 9.2 |
| Plastic wiring, no doubt | 583 | 68.7 ± 129.5% | 46.5 ± 100.5% | 65.4 ± 140.6% | 27.4 ± 61.1% | 51.9 ± 36.6 |
| Full organism: doubt steers plasticity | 1095 | 99.8 ± 1.0% | 69.2 ± 5.5% | 97.6 ± 3.0% | 41.8 ± 8.2% | 59.8 ± 5.6 |
| Rules only: born with random wiring | 161 | 83.4 ± 37.4% | 16.2 ± 68.4% | 24.8 ± 106.8% | 7.1 ± 27.6% | 39.7 ± 12.2 |

For reference, the fault-blind reinforcement-learning agent from fault_experiment.py survived 99.5% of healthy and 71.7% of faulty lives in the same setting (different test lives, same distribution).

## Evolved learning rules

| Organism | Seed | Hebbian rates (layer 1, 2) | Modulator weights (doubt r, doubt d, abs r, abs d, bias) | Self-model rate |
|---|---|---|---|---|
| Plastic wiring, no doubt | 0 | 0.065, 0.053 | -0.53, +0.32, -0.44, +0.01, +0.56 | – |
| Plastic wiring, no doubt | 1 | 0.095, 0.096 | +0.38, +0.33, +0.44, -0.84, +1.19 | – |
| Plastic wiring, no doubt | 2 | 0.058, 0.080 | -0.24, +0.48, +0.68, +0.35, +0.63 | – |
| Full organism: doubt steers plasticity | 0 | 0.061, 0.068 | +0.36, -0.19, +0.06, +0.48, +0.29 | 0.055 |
| Full organism: doubt steers plasticity | 1 | 0.046, 0.036 | +0.65, -0.56, +0.14, +0.12, +0.79 | 0.069 |
| Full organism: doubt steers plasticity | 2 | 0.047, 0.149 | +0.17, +0.58, -0.27, -0.25, +0.50 | 0.096 |
| Rules only: born with random wiring | 0 | 0.051, 0.153 | -0.48, -0.54, +0.68, +0.33, +0.95 | 0.070 |
| Rules only: born with random wiring | 1 | 0.207, 0.128 | +0.35, -0.07, +0.68, +0.66, +1.13 | 0.058 |
| Rules only: born with random wiring | 2 | 0.121, 0.107 | -0.07, -0.35, -0.11, +0.38, +1.22 | 0.060 |
