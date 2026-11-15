# Nested subject evaluation

Dataset: `uci_mhealth`. Each subject contributes one held-out score.

| Model or selection policy | Accuracy | Macro-F1 (95% subject bootstrap) | Median fit (s) | Median predict (s) |
|---|---:|---:|---:|---:|
| KNN | 0.9163 | 0.9031 (0.8506–0.9456) | 0.0005 | 0.0466 |
| Logistic Regression | 0.9628 | 0.9577 (0.9302–0.9818) | 0.9318 | 0.0003 |
| Selected pipeline | 0.9628 | 0.9577 (0.9302–0.9818) | 0.9318 | 0.0003 |
| Naive Bayes | 0.9352 | 0.9285 (0.8900–0.9636) | 0.0048 | 0.0029 |
| Neural Network | 0.9522 | 0.9449 (0.9177–0.9703) | 0.4207 | 0.0009 |

Feature construction median: 5.271 s (Q1–Q3 4.996–5.396 s, 3 repetitions).

See [the run manifest](run_manifest.json) for split membership, environment and output hashes.
Intervals are descriptive across the observed subjects with overlapping training folds.
