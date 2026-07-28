## Quine takeover and detectability vs question-pool size and adversary population

30 seeds per cell, matched adversary-present and adversary-absent runs. Full-mesh
cells sit at their true mean degree (N−1). Alarm rate = share of adaptive-chart
evaluations (3σ trailing window) out of band during steps ≥ 400, pooled over the 30
runs, +1-smoothed; each arm contributes 4,800 evaluations per cell. **Bold** marks
takeover (victim infection > 0.5) that the monitor does not react to (alarm ratio ≤ 1×).

### Q = 50, 1 adversary — infected fraction of victims

| mean degree | 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|
| 99,999 | — | — | — | 1.00 |
| 9,999 | — | — | 1.00 | 1.00 |
| 999 | — | 1.00 | 1.00 | 1.00 |
| 99 | 1.00 | 1.00 | 1.00 | 1.00 |
| 9 | 1.00 | 1.00 | 1.00 | 1.00 |

### Q = 50, 1 adversary — attack alarm rate ÷ baseline alarm rate

| mean degree | 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|
| 99,999 | — | — | — | 1.5× |
| 9,999 | — | — | 1.8× | 1.6× |
| 999 | — | 1.5× | 1.4× | 1.6× |
| 99 | **0.84×** | 1.8× | 1.4× | 1.4× |
| 9 | **0.79×** | 1.4× | 1.5× | 1.6× |

### Q = 500, 10 adversaries — infected fraction of victims

| mean degree | 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|
| 99,999 | — | — | — | 0.11 |
| 9,999 | — | — | 0.70 | 0.11 |
| 999 | — | 1.00 | 0.58 | 0.08 |
| 99 | 1.00 | 0.95 | 0.26 | 0.03 |
| 9 | 1.00 | 0.62 | 0.10 | 0.01 |

### Q = 500, 10 adversaries — attack alarm rate ÷ baseline alarm rate

| mean degree | 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|
| 99,999 | — | — | — | 0.65× |
| 9,999 | — | — | 2.0× | 1.3× |
| 999 | — | 5.0× | **1.0×** | 0.61× |
| 99 | 4.0× | **1.0×** | 1.0× | 0.71× |
| 9 | 8.0× | **1.0×** | 0.50× | 0.59× |

### Q = 500, 100 adversaries — infected fraction of victims

| mean degree | 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|
| 99,999 | — | — | — | 0.70 |
| 9,999 | — | — | 1.00 | 0.69 |
| 999 | — | 1.00 | 1.00 | 0.57 |
| 99 | — | 1.00 | 0.95 | 0.26 |
| 9 | — | 1.00 | 0.62 | 0.10 |

### Q = 500, 100 adversaries — attack alarm rate ÷ baseline alarm rate

| mean degree | 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|
| 99,999 | — | — | — | **0.70×** |
| 9,999 | — | — | **1.0×** | 1.0× |
| 999 | — | 2.5× | 2.0× | **0.65×** |
| 99 | — | 4.0× | 2.0× | 0.63× |
| 9 | — | 3.0× | **1.0×** | 0.63× |

### Raw alarm counts for every cell reporting a ratio above 1×

Ratios above 1 rest on very few absolute alarms; the counts below are attack vs
baseline out of 4,800 chart evaluations per arm. `sig` is a one-sided binomial test
on the split (under H0 each alarm is equally likely to land in either arm).

| grid | N | mean degree | ratio | attack v baseline | p < 0.05 |
|---|---|---|---|---|---|
| Q = 50, 1 adversary | 100,000 | 99,999 | 1.5× | 124 v 84 | yes |
| Q = 50, 1 adversary | 10,000 | 9,999 | 1.8× | 152 v 86 | yes |
| Q = 50, 1 adversary | 100,000 | 9,999 | 1.6× | 133 v 85 | yes |
| Q = 50, 1 adversary | 1,000 | 999 | 1.5× | 146 v 97 | yes |
| Q = 50, 1 adversary | 10,000 | 999 | 1.4× | 148 v 103 | yes |
| Q = 50, 1 adversary | 100,000 | 999 | 1.6× | 146 v 92 | yes |
| Q = 50, 1 adversary | 1,000 | 99 | 1.8× | 149 v 81 | yes |
| Q = 50, 1 adversary | 10,000 | 99 | 1.4× | 137 v 95 | yes |
| Q = 50, 1 adversary | 100,000 | 99 | 1.4× | 129 v 90 | yes |
| Q = 50, 1 adversary | 1,000 | 9 | 1.4× | 143 v 100 | yes |
| Q = 50, 1 adversary | 10,000 | 9 | 1.5× | 135 v 88 | yes |
| Q = 50, 1 adversary | 100,000 | 9 | 1.6× | 133 v 83 | yes |
| Q = 500, 10 adversaries | 10,000 | 9,999 | 2.0× | 1 v 0 | no |
| Q = 500, 10 adversaries | 100,000 | 9,999 | 1.3× | 53 v 40 | no |
| Q = 500, 10 adversaries | 1,000 | 999 | 5.0× | 9 v 1 | yes |
| Q = 500, 10 adversaries | 100 | 99 | 4.0× | 3 v 0 | no |
| Q = 500, 10 adversaries | 100 | 9 | 8.0× | 7 v 0 | yes |
| Q = 500, 100 adversaries | 100,000 | 9,999 | 1.0× | 41 v 40 | no |
| Q = 500, 100 adversaries | 1,000 | 999 | 2.5× | 4 v 1 | no |
| Q = 500, 100 adversaries | 10,000 | 999 | 2.0× | 1 v 0 | no |
| Q = 500, 100 adversaries | 1,000 | 99 | 4.0× | 3 v 0 | no |
| Q = 500, 100 adversaries | 10,000 | 99 | 2.0× | 1 v 0 | no |
| Q = 500, 100 adversaries | 1,000 | 9 | 3.0× | 2 v 0 | no |
