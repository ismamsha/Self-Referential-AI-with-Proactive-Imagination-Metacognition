# Hidden-fault experiment: mild faults

Hard difficulty. In half of all episodes a hidden fault starts at a random step between 20 and 70. Motor fault: every work action drains 0.25 × intensity extra resources. Battery fault: resting recovers a quarter as much. 500 training episodes, then 300 greedy evaluation episodes per run; mean ± 95% CI over 3 seeds.

| Agent | Survival, healthy | Survival, fault | Motor fault | Battery fault | Reward, fault | Fault detected within 15 steps | False alarms |
|---|---|---|---|---|---|---|---|
| Always rest | 59.7% | 59.1% | – | – | – | – | – |
| Four-line threshold rule | 56.0% | 46.1% | – | – | – | – | – |
| Metacognitive: + surprise-doubt | 98.4 ± 3.5% | 94.6 ± 9.8% | 95.2 ± 12.7% | 94.0 ± 6.9% | 50.7 ± 1.5 | 48.7 ± 5.2% | 27.1 ± 3.7% |
| First design: surprise-doubt also drives β | 92.1 ± 17.3% | 86.2 ± 15.2% | 89.7 ± 10.2% | 83.3 ± 31.5% | 37.9 ± 29.7 | 52.7 ± 14.5% | 25.5 ± 8.2% |
| + memory of last 4 steps | 46.8 ± 102.0% | 32.5 ± 55.3% | 20.3 ± 41.5% | 43.3 ± 70.7% | 28.9 ± 20.4 | 2.5 ± 6.5% | 6.3 ± 7.2% |
| + told the fault (oracle) | 99.8 ± 0.9% | 83.5 ± 48.2% | 74.3 ± 89.0% | 89.3 ± 22.6% | 50.3 ± 4.2 | 18.6 ± 72.9% | 1.3 ± 1.4% |
| Fault-blind: nothing extra | 99.3 ± 1.6% | 95.1 ± 3.1% | 97.2 ± 7.0% | 93.3 ± 2.8% | 51.4 ± 1.9 | 7.0 ± 15.0% | 5.8 ± 17.4% |
| DQN (value learning only) | 81.2 ± 16.9% | 75.3 ± 30.9% | 78.2 ± 33.1% | 72.0 ± 28.2% | 50.1 ± 4.3 | – | – |

Detection: the agent's doubt signal rises above the level it exceeds on only 5% of healthy steps, within 15 steps of the fault. False alarms: the same test on healthy episodes at a random step. For the metacognitive agents the signal is surprise-doubt; for the others it is ensemble disagreement.
