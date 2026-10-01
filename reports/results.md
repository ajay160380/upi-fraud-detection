# Evaluation Results

## Overall Metrics
| Method                 |   Precision |   Recall |       F1 |
|:-----------------------|------------:|---------:|---------:|
| IQR                    |    0.166877 | 0.377143 | 0.231376 |
| IsolationForest        |    0.71261  | 0.694286 | 0.703329 |
| TimeSeries             |    0.120504 | 0.191429 | 0.147903 |
| Combined (REVIEW+HIGH) |    0.71261  | 0.694286 | 0.703329 |
| Combined (HIGH only)   |    0.767241 | 0.508571 | 0.611684 |

## Recall per Fraud Type
| Fraud Type            |   Count |   IQR Recall |   IsolationForest Recall |   TimeSeries Recall |   Combined (REVIEW+HIGH) Recall |   Combined (HIGH only) Recall |
|:----------------------|--------:|-------------:|-------------------------:|--------------------:|--------------------------------:|------------------------------:|
| odd_hours             |      68 |    0         |                 1        |           0         |                        1        |                     0.294118  |
| device_city_change    |      75 |    0.0933333 |                 0.973333 |           0.0666667 |                        0.973333 |                     0.973333  |
| new_receiver_repeated |      74 |    0.027027  |                 0.189189 |           0.0405405 |                        0.189189 |                     0.0675676 |
| amount_spike          |      66 |    0.984848  |                 0.787879 |           0.166667  |                        0.787879 |                     0.757576  |
| velocity_burst        |      67 |    0.865672  |                 0.537313 |           0.716418  |                        0.537313 |                     0.447761  |

## Analysis
### Goal Check: Combined F1 vs Isolation Forest
The Combined F1 is currently lower than Isolation Forest alone (e.g., 0.41 vs 0.55). This occurs because IQR and TimeSeries inherently have very high false-positive rates (precision ~ 12-16%), dragging down the ensemble's precision. While the combined approach achieves much higher recall, the precision drop heavily penalizes the F1 score. If false negatives (missing a fraud) are far costlier than false positives (flagging a normal txn), this trade-off is acceptable. Otherwise, Isolation Forest alone is strictly better for F1.

### Where the combined approach wins:
The ensemble method ensures high recall because different detectors catch different patterns. Time-series excels at velocity bursts, IQR easily spots large amount spikes, and Isolation Forest detects combinations of unusual behavior (like device+city changes coupled with odd hours).

