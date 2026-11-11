# Results tables

MinHash estimate vs exact Jaccard: mean absolute error 0.0179 over 400 pairs.

## Task-level reuse at the chosen thresholds (offline replay, all 10 datasets)

| dataset | threshold | reused / tasks | wrong reuses | duplicates caught | lookup (ms/task) |
|---|---|---|---|---|---|
| tickets | 0.60 | 21 / 70 | 0 | 21 / 28 | 1.22 |
| bugs | 0.60 | 16 / 60 | 0 | 16 / 24 | 1.53 |
| docs | 0.50 | 18 / 60 | 0 | 18 / 24 | 2.00 |
| citations | 0.40 | 21 / 70 | 0 | 21 / 28 | 0.59 |
| hosts | 0.30 | 12 / 40 | 0 | 12 / 16 | 0.82 |
| configs | 0.30 | 12 / 40 | 0 | 12 / 16 | 0.86 |
| code | 0.30 | 12 / 40 | 0 | 12 / 16 | 0.83 |
| threads | 0.30 | 13 / 40 | 0 | 13 / 16 | 0.83 |
| revisions | 0.30 | 13 / 40 | 0 | 13 / 16 | 0.95 |
| logs | 0.30 | 12 / 40 | 0 | 12 / 16 | 0.92 |

## Content-level reuse (offline replay: every file of every scenario read once)

| dataset | files | folded | output chars saved | cumulative context saved | ms / scenario |
|---|---|---|---|---|---|
| code | 283 | 139 | 60.9% | 60.0% | 74.6 |
| configs | 212 | 164 | 48.6% | 40.5% | 50.0 |
| hosts | 175 | 151 | 45.7% | 40.5% | 48.3 |
| logs | 181 | 0 | 0.0% | 0.0% | 43.6 |
| revisions | 198 | 163 | 43.7% | 40.9% | 52.8 |
| threads | 710 | 0 | 0.0% | 0.0% | 148.2 |

