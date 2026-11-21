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

## Agent runs (gemini-3.5-flash-lite)

| dataset | config | accuracy | reused | tool calls | model requests | input tokens | time (s, n x median) | MinHash overhead (s) |
|---|---|---|---|---|---|---|---|---|
| tickets | baseline | 80.0% | 0 | 163 | 178 | 1837086 | 819 | 0.00 |
| tickets | task | 86.7% | 3 | 120 | 132 | 904161 | 580 | 0.02 |
| tickets | content | 80.0% | 0 | 146 | 161 | 1071338 | 816 | 1.28 |
| tickets | both | 93.3% | 3 | 106 | 118 | 737353 | 2503 | 0.88 |
| bugs | baseline | 100.0% | 0 | 158 | 173 | 1616129 | 333 | 0.00 |
| bugs | task | 86.7% | 4 | 115 | 126 | 1699653 | 166 | 0.02 |
| bugs | content | 93.3% | 0 | 177 | 192 | 2492865 | 231 | 2.03 |
| bugs | both | 73.3% | 4 | 155 | 166 | 3201896 | 163 | 1.40 |
| docs | baseline | 93.3% | 0 | 147 | 162 | 1180901 | 333 | 0.00 |
| docs | task | 93.3% | 6 | 125 | 134 | 643032 | 198 | 0.03 |
| docs | content | 86.7% | 0 | 196 | 211 | 1686164 | 159 | 1.80 |
| docs | both | 100.0% | 6 | 102 | 111 | 599189 | 110 | 0.71 |
| hosts | baseline | 100.0% | 0 | 127 | 142 | 755329 | 111 | 0.00 |
| hosts | task | 100.0% | 4 | 94 | 105 | 514313 | 104 | 0.01 |
| hosts | content | 100.0% | 0 | 146 | 161 | 797349 | 123 | 1.34 |
| hosts | both | 100.0% | 4 | 99 | 110 | 542899 | 106 | 0.91 |
| configs | baseline | 100.0% | 0 | 150 | 165 | 1005516 | 105 | 0.00 |
| configs | task | 100.0% | 5 | 111 | 121 | 674302 | 89 | 0.01 |
| configs | content | 93.3% | 0 | 143 | 158 | 916280 | 101 | 1.56 |
| configs | both | 100.0% | 5 | 104 | 114 | 481350 | 103 | 0.73 |
| code | baseline | 100.0% | 0 | 152 | 167 | 1006162 | 113 | 0.00 |
| code | task | 100.0% | 3 | 146 | 158 | 1183186 | 100 | 0.01 |
| code | content | 100.0% | 0 | 190 | 205 | 1315954 | 165 | 2.04 |
| code | both | 100.0% | 3 | 140 | 152 | 821477 | 121 | 1.43 |
| revisions | baseline | 93.3% | 0 | 203 | 218 | 1881678 | 191 | 0.00 |
| revisions | task | 100.0% | 3 | 172 | 184 | 1503814 | 199 | 0.01 |
| revisions | content | 80.0% | 0 | 205 | 220 | 1686609 | 208 | 2.53 |
| revisions | both | 73.3% | 2 | 190 | 203 | 1513133 | 195 | 2.38 |
| logs | baseline | 100.0% | 0 | 152 | 167 | 972865 | 143 | 0.00 |
| logs | task | 100.0% | 7 | 77 | 85 | 386954 | 81 | 0.01 |
| logs | content | 100.0% | 0 | 141 | 156 | 876551 | 151 | 0.94 |
| logs | both | 100.0% | 7 | 75 | 83 | 614030 | 89 | 0.63 |

### All 8 datasets combined

| config | accuracy | reused | tool calls | model requests | input tokens | time (s, n x median) | MinHash overhead (s) |
|---|---|---|---|---|---|---|---|
| baseline | 95.8% | 0 | 1252 | 1372 | 10255666 | 2149 | 0.00 |
| task | 95.8% | 35 | 960 | 1045 | 7509415 | 1517 | 0.13 |
| content | 91.7% | 0 | 1344 | 1464 | 10843110 | 1954 | 13.52 |
| both | 92.5% | 34 | 971 | 1057 | 8511327 | 3390 | 9.07 |

