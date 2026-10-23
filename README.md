# AquaPulse pipe leak detector

Training and evaluation code for a hardware-matched leak/no-leak detector using
the Mendeley dataset **Leak Simulations in Experimental Testbed WDS**
(DOI [`10.17632/tbrnp6vrnj.1`](https://doi.org/10.17632/tbrnp6vrnj.1)).

This repository contains offline training/evaluation only. It does not contain
ESP32 firmware, TinyML conversion, a dashboard, or the raw dataset.

## Final published model

The selected product model is:

- Accelerometer + dynamic pressure sensors; no hydrophone dependency
- Accelerometer resampled to 8 kHz
- Pressure resampled to 860 Hz
- Anti-aliasing through `scipy.signal.resample_poly`
- 0.25-second non-overlapping windows
- 26 compact time/frequency features
- 200-tree class-weighted random forest
- Binary labels: `0 = No-leak`, `1 = Leak`

The final artifacts are in [`models/final/`](models/final/):

- [`model.joblib`](models/final/model.joblib): full model
- [`edge_model.joblib`](models/final/edge_model.joblib): reduced model under 200 KB
- [`metadata.json`](models/final/metadata.json): feature order, preprocessing,
  threshold, versions, DOI, and evaluation metadata

Do not load untrusted `.joblib` files: joblib/pickle model files can execute
code during deserialization. Use only artifacts obtained from this repository
or another trusted source.

## Results

The primary evidence is signal-level repeated shuffled 5-fold GroupKFold,
repeated with three fixed seeds. Windows from a recording never cross a fold.

| Metric | Final model |
|---|---:|
| Accuracy | 0.875 |
| Balanced accuracy | 0.908 |
| Precision | 0.958 |
| Recall | 0.899 |
| F1 | 0.910 |
| ROC-AUC | 0.989 |
| PR-AUC | 0.999 |
| No-leak false-alarm rate | 0.077 |

These are laboratory testbed results, not field or ESP32-S3 measurements.
The full report, condition breakdowns, ablations, confidence intervals, and
limitations are in [`reports/final_report.md`](reports/final_report.md).

## Reproduce

Use Python 3.11 or a compatible supported Python 3 environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download and extract the Mendeley dataset into `data/`, preserving its sensor
folders:

```text
data/
  Accelerometer/
  Dynamic Pressure Sensor/
  Hydrophone/
```

Run the complete audit, hardware-matched feature extraction, cross-validation,
ablations, and final artifact generation:

```powershell
python run_pipeline.py
```

The pipeline writes reports under `reports/` and final artifacts under
`models/final/`. Intermediate feature matrices are intentionally ignored by
Git because they are large and reproducible.

## Repository layout

```text
src/
  pipeline.py       Phase 1 audit and baseline pipeline
  phase2.py         Hardened Phase 2 evaluation and finalization
run_pipeline.py     Entry point
models/final/       Published model artifacts
reports/            Audit, final report, figures, and selected metrics
requirements.txt    Python dependencies
```

## Important limitations

The source dataset contains 282 files with inconsistent recording durations
and no leak-position, leak-size, leak-flow, degradation, or RUL labels.
Localization, leak-size regression, and RUL are therefore not trained.
Network-layout shift remains a significant risk, and the lab PVC testbed does
not represent buried field pipes or the final sensor mounting/electronics.
CPU timings in the report are development-machine measurements; target
ESP32-S3 timing still requires hardware validation.
