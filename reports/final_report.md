# AquaPulse Phase 2 final report

## Product decision
Final model: **raw preprocessing + 200-tree random forest**, node-compatible accelerometer + pressure features, 0.25 s windows.
Selection used the worst held-out-network fold performance, not the mean. The fitted artifact is `models\final\model.joblib` (3450.0 KB; 48.218 ms/window on the development CPU).

## Data caveats
The audit found 282 files rather than 280, recordings of roughly 34-61 seconds rather than uniformly 30 seconds, and two standalone hydrophone background files that do not match the signal parser. No position, leak-size, leak-flow, degradation, or RUL labels exist; those tasks remain skipped.

## Coverage and hardware ablation
| configuration   |   signals |   windows |   features |
|:----------------|----------:|----------:|-----------:|
| node_0.25       |        40 |      5708 |         26 |
| all_0.25        |        40 |      5703 |         38 |
| node_0.50       |        40 |      2845 |         26 |
| all_0.50        |        40 |      2842 |         38 |

Hydrophone features were present only when both H1/H2 were available. A matched all-sensor ablation is reported separately; the product result below uses only A1/A2/P1/P2.

### Matched all-sensor ablation
| level   |   accuracy |   balanced_accuracy |   precision |   recall |       f1 |   roc_auc |   pr_auc |   false_alarm_rate_nl |
|:--------|-----------:|--------------------:|------------:|---------:|---------:|----------:|---------:|----------------------:|
| signal  |       0.85 |            0.882937 |       0.925 |  0.90754 | 0.892121 |   0.99359 | 0.998413 |              0.153846 |

### Window-length comparison
|   window_s |   signal_f1 |   signal_balanced_accuracy |   signal_false_alarm_rate_nl |
|-----------:|------------:|---------------------------:|-----------------------------:|
|       0.25 |    0.909899 |                   0.907937 |                    0.0769231 |
|       0.5  |    0.882549 |                   0.869841 |                    0.153846  |

## Primary binary detector: repeated grouped CV
### Signal-level evidence
| level   |   accuracy |   balanced_accuracy |   precision |   recall |       f1 |   roc_auc |   pr_auc |   false_alarm_rate_nl |
|:--------|-----------:|--------------------:|------------:|---------:|---------:|----------:|---------:|----------------------:|
| signal  |      0.875 |            0.907937 |    0.958333 | 0.899206 | 0.909899 |  0.989011 |  0.99881 |             0.0769231 |

Signal-level bootstrap 95% CIs are in `models/final/metadata.json`. The final deployment threshold is fitted from all available training recordings and is stored there.
### Window-level diagnostics
| level   |   accuracy |   balanced_accuracy |   precision |   recall |       f1 |   roc_auc |   pr_auc |   false_alarm_rate_nl |
|:--------|-----------:|--------------------:|------------:|---------:|---------:|----------:|---------:|----------------------:|
| network |    0.87    |            0.887778 |    0.925    | 0.864444 | 0.873081 |  0.986667 | 0.998851 |             0.125     |
| signal  |    0.875   |            0.907937 |    0.958333 | 0.899206 | 0.909899 |  0.989011 | 0.99881  |             0.0769231 |
| window  |    0.86476 |            0.802962 |    0.879556 | 0.971491 | 0.910298 |  0.96415  | 0.991105 |             0.417355  |

Fold-level results are repeated shuffled 5-fold GroupKFold across three fixed seeds. A signal-level decision averages window probabilities within each recording. Group-bootstrap 95% CIs (resampling complete signals) are stored in metadata.json; no random windows from one recording cross a fold.

## Baselines
| baseline           |   accuracy |   balanced_accuracy |   precision |   recall |       f1 |   false_alarm_rate_nl |
|:-------------------|-----------:|--------------------:|------------:|---------:|---------:|----------------------:|
| majority           |   0.801002 |            0.566667 |    0.801002 | 1        | 0.879064 |              1        |
| threshold_logistic |   0.908126 |            0.840717 |    0.913008 | 0.971715 | 0.935699 |              0.325918 |

## Preprocessing ablation
| variant           |   worst_network_f1 |   mean_f1 |   signal_f1 |
|:------------------|-------------------:|----------:|------------:|
| raw               |           0.843939 |  0.89159  |    0.969697 |
| per_recording     |           0.555238 |  0.702979 |    0.888889 |
| baseline_relative |           0.82886  |  0.869432 |    0.888889 |

## Leave-one-condition false alarms
| condition   | value     |   false_alarm_rate_nl |   recall |       f1 |
|:------------|:----------|----------------------:|---------:|---------:|
| flow        | 0.18 LPS  |             0.166667  | 0.769231 | 0.820513 |
| flow        | 0.47 LPS  |             0.2       | 0.75641  | 0.782051 |
| flow        | ND        |             0.166667  | 1        | 0.974359 |
| flow        | Transient |             0.2       | 1        | 0.966667 |
| network     | BR        |             0.111111  | 0.877778 | 0.902222 |
| network     | LO        |             0.142857  | 0.851111 | 0.843939 |
| noise       | N         |             0.0769231 | 0.899206 | 0.909899 |

Transient is shown separately because valve events must not be treated as leaks.

## Five-class bonus
| variant   |   accuracy |   balanced_accuracy |   macro_f1 |   weighted_f1 |
|:----------|-----------:|--------------------:|-----------:|--------------:|
| raw       |   0.511303 |            0.565509 |   0.440673 |      0.522682 |

Per-class precision/recall/F1/support are in `phase2_multiclass_per_class.csv`; every fold confusion matrix is in `phase2_multiclass_confusion.json`. Macro-F1 is computed over all five explicit labels; absent labels receive zero, and folds are checked for class coverage.

## Edge-sized comparison
| model              |   size_kb |   latency_ms |   mean_f1 |
|:-------------------|----------:|-------------:|----------:|
| full_random_forest |  3450.01  |      48.2178 |  0.909899 |
| edge_random_forest |   178.587 |      18.0999 |  0.906309 |

Latency is measured on the development CPU; ESP32-S3 latency still requires target-hardware benchmarking.

## Failure analysis and limitations
The model remains vulnerable to installation/network shift: the network-held-out scores are the selection criterion and are substantially worse than pooled scores. Window-level accuracy overstates practical reliability because adjacent windows are correlated; signal-level metrics are the product evidence. The lab PVC testbed, sensor mounting, background flow conditions, and hydrophone coverage do not represent buried field pipes or the final piezo/pressure electronics. This is training/evaluation only; no firmware, TinyML conversion, dashboard, or fabricated regression/RUL target is included.
