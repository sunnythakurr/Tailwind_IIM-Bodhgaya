# Digital Twin PoC: 2-hour-ahead hyperglycemia prediction (Type 2 Diabetes)

**Tailwind / IIM Bodhgaya|  **License:** MIT (data is fully synthetic)

## Problem
Predict a *new* glucose excursion (>180 mg/dL, while currently <=180) 2 hours ahead by fusing a static EHR profile with streaming CGM + wearable data.

## Run
`pip install numpy pandas scikit-learn` then `python train.py` (about 1 min). `simulate.py` generates the cohort; `results.json` holds metrics.

## Data (synthetic, DPDP/HIPAA-safe)
300 patients x 14 days at 5-min resolution. Static: age, BMI, HbA1c, metformin, family history. Dynamic: CGM, steps, heart rate, sleep. Glucose comes from a mechanistic simulator (meal absorption, insulin sensitivity, activity uptake, sleep-debt resistance). Meals are hidden from the model.

## Results (patient-level held-out split, event rate 8.3%)
| Model | AUROC | AUPRC | Sens @ 10% FPR |
|---|---|---|---|
| Rule: current glucose | 0.591 | 0.088 | 0.000 |
| Static EHR only | 0.623 | 0.111 | 0.165 |
| CGM only | 0.875 | 0.374 | 0.573 |
| CGM + wearable | 0.874 | 0.372 | 0.572 |
| **Fused (EHR + CGM + wearable)** | **0.887** | **0.410** | **0.612** |

## Limitations
- Data is simulated, so absolute numbers are **not** clinical evidence. EHR-to-glucose coupling exists *by construction* in the simulator, so the fusion gain shows the pipeline works, not that real-world fusion helps by this much.
- Wearable features added nothing here. Real validation needs a real CGM + wearable dataset.
- Next steps: real-data validation, sequence models, per-patient calibration, what-if simulator, clinician dashboard.

