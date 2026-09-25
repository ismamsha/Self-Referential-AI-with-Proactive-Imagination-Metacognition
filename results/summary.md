# Results

Final evaluation: 200 greedy episodes (no exploration) per run, mean ± 95% CI over 3 seeds. Survival means reaching step 100 with resources > 0 and degradation < 1.

## Normal

| Policy | Survival | Reward / episode | Time working | Caution changed the action |
|---|---|---|---|---|
| Always rest | 100.0% | 4.0 | 0% | – |
| Four-line threshold rule | 100.0% | 55.9 | 72% | – |
| DQN (value learning only) | 100.0 ± 0.0% | 105.1 ± 2.4 | 57 ± 1% | – |
| MSRA-L without doubt | 100.0 ± 0.0% | 102.2 ± 2.5 | 57 ± 2% | – |
| MSRA-L with doubt | 100.0 ± 0.0% | 86.6 ± 36.8 | 52 ± 10% | 33% |
| MSRA-L with doubt + fear of death | 100.0 ± 0.0% | 94.9 ± 31.3 | 53 ± 11% | 25% |
| Doubt + fear of death, no imagination | 100.0 ± 0.0% | 102.0 ± 11.2 | 58 ± 6% | 19% |

## Hard

| Policy | Survival | Reward / episode | Time working | Caution changed the action |
|---|---|---|---|---|
| Always rest | 57.2% | 1.1 | 0% | – |
| Four-line threshold rule | 53.5% | 33.4 | 41% | – |
| DQN (value learning only) | 78.5 ± 10.2% | 65.6 ± 12.9 | 43 ± 5% | – |
| MSRA-L without doubt | 73.0 ± 37.5% | 60.3 ± 6.7 | 44 ± 3% | – |
| MSRA-L with doubt | 81.0 ± 23.7% | 58.9 ± 4.9 | 44 ± 4% | 7% |
| MSRA-L with doubt + fear of death | 99.5 ± 1.2% | 68.5 ± 1.4 | 38 ± 8% | 14% |
| Doubt + fear of death, no imagination | 100.0 ± 0.0% | 73.0 ± 4.2 | 39 ± 2% | 20% |

## Expert

| Policy | Survival | Reward / episode | Time working | Caution changed the action |
|---|---|---|---|---|
| Always rest | 8.3% | 2.6 | 0% | – |
| Four-line threshold rule | 3.7% | 18.1 | 20% | – |
| DQN (value learning only) | 28.8 ± 44.2% | 19.1 ± 4.7 | 22 ± 13% | – |
| MSRA-L without doubt | 0.5 ± 0.0% | 14.9 ± 5.4 | 39 ± 20% | – |
| MSRA-L with doubt | 1.0 ± 3.3% | 16.8 ± 10.1 | 30 ± 43% | 12% |
| MSRA-L with doubt + fear of death | 1.5 ± 1.2% | 19.3 ± 4.1 | 16 ± 3% | 27% |
| Doubt + fear of death, no imagination | 33.7 ± 60.6% | 24.9 ± 8.3 | 11 ± 4% | 26% |
