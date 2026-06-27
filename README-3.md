# Pupil Diameter Processing and Feature Extraction Pipeline

Pupil diameter preprocessing and trial-level feature extraction pipeline for
the visual oddball task described in:

> "Cognitive Vergence and Pupil Response During Oddball Task are Associated
> With Alzheimer's Disease Cerebrospinal Fluid Neurodegenerative Biomarkers"

Eye-tracking data were recorded with a Tobii 5L remote eye tracker (33 Hz)
using the BGaze system (BraingazeSL, Mataro, Spain).

## Scope

This repository processes **only the pupil diameter signal**. The
cognitive-vergence algorithm described in the same manuscript is **not**
included here and will not be released: it is proprietary technology
developed by Braingaze SL, already covered by an existing patent, and its
application to estimate CSF neurodegenerative biomarkers (as presented in
the manuscript) is the subject of a separate, pending patent application.

## What it does

1. Restricts each trial to the post-stimulus ("show_word") window.
2. Discards samples with off-screen gaze coordinates or poor eye-tracker
   validity codes, and excludes trials whose proportion of invalid samples
   exceeds `MAX_INVALID_SAMPLE_RATIO`.
3. Averages left/right pupil diameter and linearly interpolates gaps onto a
   fixed 33 Hz time grid.
4. Baseline-corrects each trial using the mean of the first `BASELINE_N_SAMPLES`
   samples.
5. Applies Gaussian smoothing (`SMOOTH_SIGMA`).
6. Extracts trial-level temporal/magnitude features from the cleaned signal:
   `slope_initial`, `slope_global`, `slope_late`, `auc`, `time_to_peak`,
   `peak_power`.

## Requirements

- Python 3.x
- pandas
- numpy
- scipy
- matplotlib

Install with:

```bash
pip install pandas numpy scipy matplotlib
```

## Usage

```bash
python pupil_processing_pipeline.py --data-path data/eye_tracking_raw_data.csv --output-dir output
```

Or from Python / a notebook:

```python
from pupil_processing_pipeline import run_pupil_analysis

run_pupil_analysis(data_path="data/eye_tracking_raw_data.csv", output_dir="output")
```

### Input data

A CSV export with one row per eye-tracker sample, including (at minimum):
`participant`, `currentTrial`, `timestamp`, `strState`, `leftEyeRawX`,
`leftEyeRawY`, `rightEyeRawX`, `rightEyeRawY`, `leftEyePupilSize`,
`rightEyePupilSize`, `leftValidity`, `rightValidity`, `strColourCode`,
`resultResponse`.

### Output

- `pupil_clean.csv` — sample-level cleaned pupil signal.
- `pupil_trial_features.csv` — trial-level features, including
  `interpolation_ratio`.
- `pupil_summary_results.csv` — participant-level descriptive statistics.
- `pupil_response_all_participants.png` — Target vs. Distractor group-level
  pupil response figure.

## Citation

If you use this code, please cite the manuscript above (submitted to
*Scientific Reports*; citation details will be updated upon publication).

## License

See LICENSE.
