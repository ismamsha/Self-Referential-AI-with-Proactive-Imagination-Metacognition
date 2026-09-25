# Hidden-fault experiment: severe faults

Hard difficulty. In half of all episodes a hidden fault starts at a random step between 20 and 70. Motor fault: every work action drains 0.5 × intensity extra resources. Fragile fault: random shocks become three times as frequent. 500 training episodes, then 300 greedy evaluation episodes per run; mean ± 95% CI over 3 seeds.

| Agent | Survival, healthy | Survival, fault | Motor fault | Fragile fault | Reward, fault | Fault detected within 15 steps | False alarms |
|---|---|---|---|---|---|---|---|
| Always rest | 60.7% | 27.2% | – | – | – | – | – |
| Four-line threshold rule | 52.1% | 30.8% | – | – | – | – | – |
| Metacognitive: + surprise-doubt | 99.6 ± 0.9% | 43.0 ± 55.8% | 53.0 ± 121.5% | 27.2 ± 23.2% | 39.5 ± 6.4 | 78.4 ± 16.3% | 29.6 ± 9.8% |
| + told the fault (oracle) | 100.0 ± 0.0% | 52.7 ± 36.7% | 78.2 ± 63.7% | 27.8 ± 14.0% | 43.2 ± 8.0 | 25.9 ± 5.1% | 4.0 ± 4.4% |
| Fault-blind: nothing extra | 99.5 ± 2.0% | 71.7 ± 6.7% | 99.6 ± 1.8% | 45.4 ± 19.2% | 46.3 ± 1.4 | 5.2 ± 19.3% | 1.6 ± 5.5% |

Detection: the agent's doubt signal rises above the level it exceeds on only 5% of healthy steps, within 15 steps of the fault. False alarms: the same test on healthy episodes at a random step. For the metacognitive agents the signal is surprise-doubt; for the others it is ensemble disagreement.
