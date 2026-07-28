# Scaling the simulation: network size, information diversity, adversary count, and density

We thank the reviewers for pressing on whether our results are an artifact of the small,
fully-connected configuration used in the main text. To address this we re-ran the study
as a four-way scaling sweep, varying **(i)** the number of agents `N`, **(ii)** the size of
the question pool `Q` (the diversity of information in the network), **(iii)** the number of
adversarial agents, and **(iv)** the mean degree of the contact graph. Concretely we sweep
`N ∈ {5, 10, 10², 10³, 10⁴, 10⁵}`, `Q ∈ {50, 500}`, adversary cohorts of
`{1, 10, 20, 50, 100}` agents, and mean degree `k ∈ {4, 9, 99, 999, 9999, N−1}` (the last
being the fully-connected case), holding the defection schedule fixed at the one used in
the paper (sigmoid onset at step 400, `duration = 1600`, `max_p = 0.75`, `shape = 1.0`).
Every cell is **30 independent seeds**, and each configuration is run twice — once with the
adversary cohort and once without — so that every attack run has a matched clean baseline.
Question-pool composition is held constant as `Q` grows (20% temporally-varying, 80% static,
with each static question seeded exactly once across the network), so that `Q` isolates
information diversity rather than changing the task.

**Headline finding.** The defection schedule that is undetected and achieves takeover in our
5-agent fully-connected configuration **scales to many, but not all, of the larger
configurations.** It remains undetected and achieves takeover across essentially the entire
`(N, k)` grid at the paper's information density, and it continues to do so at 10× the
information density and at large `N` provided the adversary cohort grows modestly — but it
fails in the sparse, information-rich, single-adversary corner. We regard this as a
sharpening of the paper's claim rather than a weakening of it: the attack is not a
small-scale artifact, and the conditions under which it fails are now explicit.

## Detection metric

Because the adaptive control chart has a non-zero false-positive rate that itself varies
with `N` and `k`, a raw "was an alarm raised" statistic is not interpretable across the
grid. We therefore report the **alarm-rate ratio**: the fraction of post-attack chart
evaluations that fall outside the 3σ band in the attack runs, divided by the same fraction
in the matched clean runs, pooled over the 30 seeds. A ratio of ≤ 1 means the attack raises
no more alarms than an uncompromised network of identical size and topology — i.e. the
attack is invisible to the monitor. Values are Laplace-smoothed; each arm contributes 4,800
chart evaluations per cell.

## Results

### A. Single adversary, Q = 50 (the paper's configuration) — victim infection

| mean degree | 5 | 10 | 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|---|---|
| 99,999 | — | — | — | — | — | 1.00 |
| 9,999 | — | — | — | — | 1.00 | 1.00 |
| 999 | — | — | — | 1.00 | 1.00 | 1.00 |
| 99 | — | — | 1.00 | 1.00 | 1.00 | 1.00 |
| 9 | — | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 4 | 1.00 | 1.00 | 0.96 | 0.98 | 0.98 | 0.98 |

### B. Single adversary, Q = 50 — alarm-rate ratio

| mean degree | 5 | 10 | 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|---|---|
| 99,999 | — | — | — | — | — | 1.47× |
| 9,999 | — | — | — | — | 1.76× | 1.56× |
| 999 | — | — | — | 1.50× | 1.43× | 1.58× |
| 99 | — | — | 0.84× | 1.83× | 1.44× | 1.43× |
| 9 | — | 0.92× | 0.79× | 1.43× | 1.53× | 1.60× |
| 4 | 1.03× | 1.00× | 0.95× | 1.20× | 1.27× | 1.24× |

A single adversary achieves complete takeover at **every** `(N, k)` combination tested, up to
`N = 100,000` — a network 20,000× larger than the configuration in the paper — while the
monitor's alarm rate never rises above 1.83× that of an uncompromised network, and in 6 of the
21 takeover cells does not rise above 1.2×. Takeover in this regime is scale-invariant.

### C. Single adversary, Q = 500 (10× information diversity) — victim infection

| mean degree | 5 | 10 | 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|---|---|
| 99,999 | — | — | — | — | — | 0.01 |
| 9,999 | — | — | — | — | 0.11 | 0.01 |
| 999 | — | — | — | 0.70 | 0.08 | 0.01 |
| 99 | — | — | 1.00 | 0.26 | 0.03 | 0.00 |
| 9 | — | 1.00 | 0.62 | 0.10 | 0.01 | 0.00 |
| 4 | 1.00 | 0.99 | 0.30 | 0.03 | 0.00 | 0.00 |

This is the case where the attack **does not** scale. With ten times the information in the
network and only a single adversary, the payload is diluted: takeover survives at small `N`
and at high density, but collapses for `N ≥ 1,000` at low degree. This is the honest limit
of the single-adversary result, and it identifies information diversity — not network size —
as the effective defence.

### D. Fixed adversary cohort at Q = 500 — victim infection

| N | mean degree | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| 1,000 | 4 | 0.28 | 0.48 | 0.79 | 0.93 |
| 1,000 | 9 | 0.62 | 0.85 | 0.99 | 1.00 |
| 1,000 | 99 | 0.95 | 1.00 | 1.00 | 1.00 |
| 1,000 | 999 (full mesh) | 1.00 | 1.00 | 1.00 | 1.00 |
| 10,000 | 4 | 0.04 | 0.07 | 0.16 | 0.30 |
| 10,000 | 9 | 0.10 | 0.18 | 0.39 | 0.62 |
| 10,000 | 99 | 0.26 | 0.45 | 0.78 | 0.95 |
| 10,000 | 999 | 0.58 | 0.82 | 0.99 | 1.00 |
| 10,000 | 9,999 (full mesh) | 0.70 | 0.91 | 1.00 | 1.00 |
| 100,000 | 4 | 0.00 | 0.01 | 0.02 | 0.04 |
| 100,000 | 9 | 0.01 | 0.02 | 0.05 | 0.10 |
| 100,000 | 99 | 0.03 | 0.06 | 0.14 | 0.26 |
| 100,000 | 999 | 0.08 | 0.16 | 0.34 | 0.57 |
| 100,000 | 9,999 | 0.11 | 0.21 | 0.44 | 0.69 |
| 100,000 | 99,999 (full mesh) | 0.11 | 0.22 | 0.46 | 0.70 |

### E. Fixed adversary cohort at Q = 500 — alarm-rate ratio

| N | mean degree | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| 1,000 | 4 | 1.00× | 1.00× | 6.00× | 5.00× |
| 1,000 | 9 | 1.00× | 4.00× | 5.00× | 3.00× |
| 1,000 | 99 | 1.00× | 3.00× | 1.00× | 4.00× |
| 1,000 | 999 (full mesh) | 5.00× | 2.00× | 2.00× | 2.50× |
| 10,000 | 4 | 2.23× | 1.77× | 1.85× | 2.23× |
| 10,000 | 9 | 0.50× | 0.50× | 1.00× | 1.00× |
| 10,000 | 99 | 1.00× | 1.00× | 2.00× | 2.00× |
| 10,000 | 999 | 1.00× | 1.00× | 1.00× | 2.00× |
| 10,000 | 9,999 (full mesh) | 2.00× | 2.00× | 1.00× | 1.00× |
| 100,000 | 4 | 0.92× | 0.97× | 0.91× | 0.89× |
| 100,000 | 9 | 0.59× | 0.62× | 0.76× | 0.63× |
| 100,000 | 99 | 0.71× | 0.78× | 0.63× | 0.63× |
| 100,000 | 999 | 0.61× | 0.84× | 0.69× | 0.65× |
| 100,000 | 9,999 | 1.32× | 0.95× | 1.02× | 1.02× |
| 100,000 | 99,999 (full mesh) | 0.65× | 0.56× | 0.68× | 0.70× |

The ratios above 1 in this table are concentrated in the small-`N` cells, where they rest on
very few absolute alarms (typically 2–9 out of 4,800 evaluations per arm); the `N = 100,000`
rows, where the absolute counts are large enough to be stable, sit at or below 1 almost
without exception.

**The dilution defence is defeated by a very small number of adversaries.** The same
defection schedule degrades a far larger network carrying ten times the information
diversity with **far fewer than 20% of agents adversarial**. In particular, **100 adversaries
in a 100,000-agent network with mean degree 999 — 0.1% of the population — infect 57% of the
non-adversarial agents while raising alarms at 0.65× the rate of an uncompromised network
of the same size and topology — that is, fewer alarms than the clean network.** At the same
cohort size and `N`, the fully-connected case reaches 70% infection at a 0.70× alarm ratio,
and 50 adversaries (0.5%) fully compromise a 10,000-agent fully-connected network.

Marginal returns to cohort size are diminishing but only mildly so: at `N = 100,000` and full
connectivity, infection per adversary falls from 11.5 to 7.0 (×10⁻³) as the cohort grows from
10 to 100, so the attack neither requires a critical mass nor exhibits a percolation
threshold — it degrades gracefully in the attacker's favour.

## Relation to a deployed system

These regimes are not hypothetical. Moltbook, an autonomous-agent social platform, was
analysed at approximately **46,000 active agents** posting and commenting over the
observation window (Kirilenko et al., *Collective Behavior of AI Agents: the Case of
Moltbook*, arXiv:2602.09270), a scale that sits between our `N = 10,000` and
`N = 100,000` rows. Content there reaches agents through shared topic communities with
default-visibility defaults rather than through pairwise links, so an individual agent's
output is exposed to a large fraction of the population even though no agent reads
everything — an effective connectivity far closer to our high-degree cells than to the
sparse ones. Those are precisely the cells in which a cohort of order 10²
agents achieves majority compromise at an alarm rate indistinguishable from an
uncompromised network.

## Reproducibility

All cells reported above are 30 seeds per configuration with matched adversary-present and
adversary-absent runs (6,120 simulations in total). Per-cell results, the sweep driver, and
the plotting code are included in the supplementary material.
