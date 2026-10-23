| Model | MAE 30 | MAE 60 | MAE 120 | RMSE 60 | Clarke A+B 60 | Clarke D+E 60 | Low sens. 60 | Low prec. 60 | AUROC low 60 |
|---|---|---|---|---|---|---|---|---|---|
| Persistence (no change) | 19.7 | 34.1 | 51.5 | 47.6 | 93% | 7% | 55% | 55% | nan |
| Damped linear trend | 18.6 | 35.9 | 62.7 | 51.8 | 92% | 5% | 72% | 49% | nan |
| Dynamic-only XGBoost | 10.6 | 15.0 | 20.0 | 21.7 | 96% | 4% | 55% | 85% | 0.97 |
| GlucoTwin (EHR + wearable fusion) | 10.4 | 14.5 | 18.9 | 21.2 | 96% | 4% | 56% | 86% | 0.97 |

MAE in mg/dL on 16 unseen synthetic patients (63360 forecasts). Low-glucose event rate at 60 min in the test set: 10.2%.
Share of SHAP importance (60-min model): static EHR 9%, dynamic wearable 91%.

Alert rule P(low in 60 min) >= tau, tau tuned on validation patients for >=80% sensitivity:
- Dynamic-only XGBoost: tau 0.29, sensitivity 73%, precision 69%, false-alert readings per patient-day 9.4
- GlucoTwin (EHR + wearable fusion): tau 0.30, sensitivity 74%, precision 70%, false-alert readings per patient-day 9.5

90% interval coverage (fused): 30 min 90%, 60 min 90%, 120 min 90%.