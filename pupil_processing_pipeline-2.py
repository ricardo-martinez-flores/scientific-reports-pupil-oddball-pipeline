"""
Pupil Diameter Processing and Feature Extraction Pipeline
==========================================================

Oculomotor data processing pipeline for the visual oddball task described in:
"Cognitive Vergence and Pupil Response During Oddball Task are Associated With
Alzheimer's Disease Cerebrospinal Fluid Neurodegenerative Biomarkers"

Eye-tracking data were recorded with a Tobii 5L remote eye tracker (33 Hz) using
the BGaze system (BraingazeSL, Mataro, Spain). This script implements the pupil
diameter preprocessing pipeline (artifact rejection, gap interpolation, baseline
correction and smoothing) and the trial-level temporal/magnitude feature
extraction used in the manuscript.

SCOPE OF THIS RELEASE
----------------------
This script processes ONLY the pupil diameter signal. The cognitive-vergence
algorithm described in the same manuscript is NOT included in this release
and will not be released: it is proprietary technology developed by
Braingaze SL, already covered by an existing patent, and its application to
estimate CSF neurodegenerative biomarkers (as presented in the manuscript)
is the subject of a separate, pending patent application (see the Data/Code
Availability statement of the manuscript). No vergence-related computation
(eye-vector geometry, vergence angle, or vergence baseline correction) is
implemented here.

Requirements: Python 3.x, pandas, numpy, scipy, matplotlib

Author note for Code Availability section: deposit this file (and the example
data dictionary, if shared) to a DOI-assigning repository (e.g. Zenodo) and
cite the resulting DOI in the manuscript's Code Availability statement.
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Plot styling
# -----------------------------------------------------------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 24
plt.rcParams["font.weight"] = "bold"

# -----------------------------------------------------------------------------
# Recording apparatus configuration (BGaze system / Tobii 5L, 33 Hz)
# -----------------------------------------------------------------------------
MONITOR_CONFIG = {
    "MonitorModel": "AUO71EC",
    "Width": 1024,
    "Height": 768,
    "WindowWidth": 1366,
    "WindowHeight": 768,
    "PhysicalWidth": 340.0,   # mm
    "PhysicalHeight": 190.0,  # mm
    "TrackerBrand": "tobii",
    "TrackerFrameRate": 33,   # Hz
    "TrackerModel": "IS5_Large_Eyetracker_5L",
    "TrackerProductId": "IS510-100201007814",
    "TrackerFirmwareVersion": "2.11.0-8568daa",
    "BdgVersion": "02.16.04",
    "IsCpt": 0,
}

# -----------------------------------------------------------------------------
# Timing parameters
# -----------------------------------------------------------------------------
TMAX = 2000.0                                   # trial window, ms
FRQ = MONITOR_CONFIG["TrackerFrameRate"]        # 33 Hz
N_SAMPLES = int((TMAX / 1000) * FRQ) + 1        # samples in the fixed time grid (67 @ 33 Hz / 2 s)
DT = 1000.0 / FRQ                               # inter-sample interval, ms (~30.3 ms @ 33 Hz)
TS = np.array(range(0, N_SAMPLES)) * DT         # fixed post-stimulus time grid, ms

BASELINE_N_SAMPLES = 7

# Maximum allowed proportion of invalid (off-screen / poor-validity) samples
# per trial before the trial is excluded.
MAX_INVALID_SAMPLE_RATIO = 0.5

SMOOTH_FLAG = True   # apply Gaussian smoothing to the pupil signal
SMOOTH_SIGMA = 4.0   # standard deviation (in samples) of the Gaussian kernel


# =============================================================================
# Utility functions
# =============================================================================
def myinterp(t, x, t_query):
    """Linear interpolation of x(t) onto t_query, ignoring NaNs in x."""
    valid = np.where(~np.isnan(x))
    if len(valid[0]) > 0:
        return np.interp(t_query, t[valid], x[valid])
    return np.full(len(t_query), np.nan)  # x contains only NaNs


def dfcol_to_array(df, colname):
    """Convert a dataframe column to a numpy array (float)."""
    return np.array(df[colname])


def dfcol_to_array_i(df, colname):
    """Convert a dataframe column to a numpy integer array."""
    return np.array(df[colname], dtype=int)


def nanmean1(arr):
    """Mean of an array, ignoring NaNs."""
    return np.nanmean(arr)


def gaussian_smooth(y, sigma=SMOOTH_SIGMA):
    """Gaussian smoothing filter applied to the pupil signal."""
    if y.size == 0:
        return y
    return gaussian_filter1d(y, sigma=sigma, mode="nearest")


def is_valid_trial_duration(trial_data, min_duration_ms=3000, max_duration_ms=6000):
    """Check whether a trial's total recorded duration falls in a plausible range."""
    if len(trial_data) < 2:
        return False
    timestamps = trial_data["timestamp"].values
    duration_ms = (timestamps[-1] - timestamps[0]) / 1000
    return min_duration_ms <= duration_ms <= max_duration_ms


# =============================================================================
# Trial-level feature extraction (ported from the original R implementation)
# =============================================================================
FEATURE_NAMES = ["slope_initial", "slope_global", "slope_late", "auc", "time_to_peak", "peak_power"]


def slope_fit(t, s):
    """
    Ordinary least-squares slope of signal s as a function of time t,
    ignoring NaNs. Equivalent to the slope coefficient of `lm(s ~ t)` in R.
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    valid = ~np.isnan(s) & ~np.isnan(t)
    if valid.sum() < 2:
        return np.nan
    try:
        slope, _ = np.polyfit(t[valid], s[valid], 1)
        return float(slope)
    except Exception:
        return np.nan


def extract_slope_initial(time, signal, window_ms=500.0):
    """
    Slope of the signal within the initial window (default: 0-500 ms
    post-stimulus). Time is converted to seconds for the regression and the
    resulting coefficient is multiplied by 1000, matching the reference
    implementation: coef(lm(s ~ t_sec))[2] * 1000.
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    mask = time <= window_ms
    if mask.sum() < 2:
        return np.nan
    slope = slope_fit(time[mask] / 1000.0, signal[mask])
    return np.nan if np.isnan(slope) else slope * 1000.0


def extract_slope_global(time, signal):
    """
    Slope of the signal across the full trial window (time in seconds,
    coefficient x1000), matching coef(lm(s ~ t_sec))[2] * 1000.
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    slope = slope_fit(time / 1000.0, signal)
    return np.nan if np.isnan(slope) else slope * 1000.0


def extract_slope_late(time, signal, start_ms=1000.0):
    """
    Slope of the signal within the late window (default: 1000-2000 ms
    post-stimulus). Time is converted to seconds for the regression and the
    resulting coefficient is multiplied by 1000.
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    mask = time >= start_ms
    if mask.sum() < 2:
        return np.nan
    slope = slope_fit(time[mask] / 1000.0, signal[mask])
    return np.nan if np.isnan(slope) else slope * 1000.0


def extract_auc(time, signal):
    """Area under the curve via the trapezoidal rule, ignoring NaNs."""
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    valid = ~np.isnan(signal) & ~np.isnan(time)
    if valid.sum() < 2:
        return np.nan
    t = time[valid]
    s = signal[valid]
    try:
        return float(np.sum(np.diff(t) * (s[:-1] + s[1:]) / 2.0))
    except Exception:
        return np.nan


def extract_time_to_peak(time, signal):
    """Time (ms) at which the signal reaches its peak absolute value."""
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if np.all(np.isnan(signal)):
        return np.nan
    try:
        return float(time[np.nanargmax(np.abs(signal))])
    except Exception:
        return np.nan


def extract_peak_power(signal):
    """Peak absolute amplitude of the signal, ignoring NaNs."""
    signal = np.asarray(signal, dtype=float)
    if np.all(np.isnan(signal)):
        return np.nan
    try:
        return float(np.nanmax(np.abs(signal)))
    except Exception:
        return np.nan


def extract_all_features(time, signal, prefix):
    """
    Extract the full set of temporal/shape and magnitude features for one
    trial's cleaned signal, returned as a dict with keys "{prefix}_{feature}".
    """
    values = [
        extract_slope_initial(time, signal),
        extract_slope_global(time, signal),
        extract_slope_late(time, signal),
        extract_auc(time, signal),
        extract_time_to_peak(time, signal),
        extract_peak_power(signal),
    ]
    return {f"{prefix}_{name}": value for name, value in zip(FEATURE_NAMES, values)}


# =============================================================================
# Main pupil-signal processing pipeline
# =============================================================================


def process_pupil_data(df):
    print("\nProcessing pupil diameter data")
    print(f"  Trial window: {TMAX} ms")
    print(f"  Sampling rate: {FRQ} Hz")
    print(f"  Samples per trial (fixed grid): {N_SAMPLES}")
    print(f"  Inter-sample interval: {DT:.1f} ms")
    print(f"  Baseline window: {BASELINE_N_SAMPLES} samples (~{BASELINE_N_SAMPLES * DT:.0f} ms)")
    print(f"  Max. invalid-sample ratio per trial: {MAX_INVALID_SAMPLE_RATIO}")

    results_detailed = []
    results_features = []
    results_summary = []
    all_interpolation_ratios = []

    for participant in df["participant"].unique():
        print(f"\nParticipant: {participant}")
        df_participant = df[df["participant"] == participant].copy()

        trials = df_participant["currentTrial"].unique()
        pupil_signals = []
        n_processed = 0
        n_excluded = 0

        for k, trial in enumerate(trials):
            if k % 10 == 0:
                print(f"  Trial {trial} ({k + 1}/{len(trials)})")

            dfX = df_participant[df_participant["currentTrial"] == trial].copy()

            if len(dfX) < 10 or not is_valid_trial_duration(dfX):
                n_excluded += 1
                continue

            # Trial/condition codes
            str_colour_code = _first_numeric(dfX, "strColourCode")
            result_response = _first_numeric(dfX, "resultResponse")

            t = dfcol_to_array(dfX, "timestamp") / 1e3
            t = t - t[0]

            LP = dfcol_to_array(dfX, "leftEyePupilSize")
            RP = dfcol_to_array(dfX, "rightEyePupilSize")
            Lx01 = dfcol_to_array(dfX, "leftEyeRawX")
            Ly01 = dfcol_to_array(dfX, "leftEyeRawY")
            Rx01 = dfcol_to_array(dfX, "rightEyeRawX")
            Ry01 = dfcol_to_array(dfX, "rightEyeRawY")
            Lv = dfcol_to_array(dfX, "leftValidity")
            Rv = dfcol_to_array(dfX, "rightValidity")
            St = dfX["strState"].values
            CT = dfcol_to_array_i(dfX, "currentTrial")

            # Restrict to the post-stimulus ("show_word") window
            its = np.where((CT == trial) & (np.array(St) == "show_word"))[0]
            if len(its) < 10:
                n_excluded += 1
                continue

            itso, itf = its[0], its[-1]
            tso = t[itso]
            tw = t[itso:itf + 1] - tso
            if len(tw) < 10:
                n_excluded += 1
                continue

            lx01 = Lx01[itso:itf + 1]
            ly01 = Ly01[itso:itf + 1]
            rx01 = Rx01[itso:itf + 1]
            ry01 = Ry01[itso:itf + 1]
            lp = LP[itso:itf + 1]
            rp = RP[itso:itf + 1]
            lv = Lv[itso:itf + 1]
            rv = Rv[itso:itf + 1]

            window_len = min(len(lx01), len(ly01), len(rx01), len(ry01), len(lv), len(rv), len(lp), len(rp))
            lx01, ly01, rx01, ry01 = lx01[:window_len], ly01[:window_len], rx01[:window_len], ry01[:window_len]
            lv, rv = lv[:window_len], rv[:window_len]
            lp, rp = lp[:window_len], rp[:window_len]

            # Outlier rejection: off-screen gaze coordinates or poor validity
            invalid_idx = np.where(
                (lx01 < 0) | (ly01 < 0) | (rx01 < 0) | (ry01 < 0) | (lv > 0) | (rv > 0)
            )[0]
            n_invalid = len(invalid_idx)
            interpolation_ratio = n_invalid / window_len if window_len > 0 else np.nan

            if n_invalid > MAX_INVALID_SAMPLE_RATIO * window_len:
                n_excluded += 1
                continue

            if n_invalid > 0:
                lp[invalid_idx] = np.nan
                rp[invalid_idx] = np.nan

            # Average across eyes, then interpolate onto the fixed time grid
            try:
                lp_i = myinterp(tw[:window_len], lp, TS)
                rp_i = myinterp(tw[:window_len], rp, TS)
                p = (lp_i + rp_i) / 2.0
            except Exception as e:
                print(f"    Interpolation error: {e}")
                n_excluded += 1
                continue

            # Baseline correction (mean of the first 7 samples, ~212 ms pre-stimulus)
            n_baseline = min(BASELINE_N_SAMPLES, len(p))
            p0 = nanmean1(p[0:n_baseline])

            if not np.isnan(p0):
                p_clean = p - p0
                if SMOOTH_FLAG:
                    p_clean = gaussian_smooth(p_clean)
            else:
                # Baseline could not be estimated (e.g. all-NaN baseline window);
                # fall back to the uncorrected interpolated signal.
                p_clean = p

            if np.all(np.isnan(p_clean)):
                n_excluded += 1
                continue

            # Align all per-sample arrays to a common length
            min_length = min(len(TS), len(p_clean), len(lp_i), len(rp_i))
            ts_aligned = TS[:min_length]
            p_aligned = p_clean[:min_length]
            lp_aligned = lp_i[:min_length]
            rp_aligned = rp_i[:min_length]

            for i in range(min_length):
                if not np.isnan(p_aligned[i]):
                    row = {
                        "participant": participant,
                        "trial": trial,
                        "strColourCode": str_colour_code,
                        "resultResponse": result_response,
                        "timestamp": ts_aligned[i],
                        "pupil_size": p_aligned[i],
                        "left_pupil": lp_aligned[i] if not np.isnan(lp_aligned[i]) else p_aligned[i] / 2,
                        "right_pupil": rp_aligned[i] if not np.isnan(rp_aligned[i]) else p_aligned[i] / 2,
                    }
                    results_detailed.append(row)

            # Trial-level feature extraction on the cleaned signal
            feature_row = {
                "participant": participant,
                "trial": trial,
                "strColourCode": str_colour_code,
                "resultResponse": result_response,
                "interpolation_ratio": interpolation_ratio,
                **extract_all_features(ts_aligned, p_aligned, prefix="pupil"),
            }
            results_features.append(feature_row)
            all_interpolation_ratios.append(interpolation_ratio)

            pupil_signals.append(p_clean)
            n_processed += 1

        print(f"  Trials processed: {n_processed}")
        print(f"  Trials excluded: {n_excluded}")

        if pupil_signals:
            try:
                min_len = min(len(p) for p in pupil_signals)
                pupil_array = np.vstack([p[:min_len] for p in pupil_signals])
                pupil_mean = np.nanmean(pupil_array, axis=0)
                pupil_min = np.nanmin(pupil_array, axis=0)
                pupil_max = np.nanmax(pupil_array, axis=0)

                summary_row = {
                    "participant": participant,
                    "pupil_mean": nanmean1(pupil_mean),
                    "pupil_min": nanmean1(pupil_min),
                    "pupil_max": nanmean1(pupil_max),
                }
                results_summary.append(summary_row)
            except Exception as e:
                print(f"  Error computing participant-level summary statistics: {e}")

    if all_interpolation_ratios:
        ratios = np.array(all_interpolation_ratios)
        pct_le_20 = 100 * np.mean(ratios <= 0.20)
        pct_lt_5 = 100 * np.mean(ratios < 0.05)
        print("\nInterpolation summary across retained trials")
        print(f"  Trials retained: {len(ratios)}")
        print(f"  Mean interpolated samples: {100 * ratios.mean():.1f}%")
        print(f"  Median interpolated samples: {100 * np.median(ratios):.1f}%")
        print(f"  Trials requiring <=20% interpolation: {pct_le_20:.1f}%")
        print(f"  Trials requiring <5% interpolation: {pct_lt_5:.1f}%")

    return results_detailed, results_features, results_summary


def _first_numeric(dfX, colname):
    """Return the first non-null value of `colname` as a float, or NaN."""
    if colname not in dfX.columns:
        return np.nan
    non_null = dfX[colname].dropna()
    if len(non_null) == 0:
        return np.nan
    try:
        return float(non_null.iloc[0])
    except (TypeError, ValueError):
        return np.nan


# =============================================================================
# Plotting
# =============================================================================
def plot_pupil_response(results_detailed):
    """Plot the group-level pupil response curve for Target vs. Distractor trials."""
    print("\nGenerating pupil response plot (all participants)")

    if not results_detailed:
        print("No data available to plot.")
        return

    df = pd.DataFrame(results_detailed)
    df = df.dropna(subset=["pupil_size", "strColourCode"])
    if len(df) == 0:
        print("No valid data after filtering.")
        return

    df["stimulus_type"] = df["strColourCode"].map({1: "Distractor", 2: "Target"})
    df = df.dropna(subset=["stimulus_type"])

    print(f"  Valid samples: {len(df)}")
    print(f"  Time range: {df['timestamp'].min():.1f}-{df['timestamp'].max():.1f} ms")
    print(f"  Participants: {df['participant'].nunique()}")

    hierarchical = _hierarchical_average(df, "pupil_size")
    if len(hierarchical) == 0:
        print("Hierarchical averaging produced no data.")
        return

    plt.figure(figsize=(10, 8))
    colors = {"Target": "#ff4444", "Distractor": "#4444ff"}
    _plot_hierarchical_data(hierarchical, "pupil_size", colors, "Pupil Response - All Participants", "Pupil (mm)")
    plt.tight_layout()

    output_path = os.path.join(os.getcwd(), "pupil_response_all_participants.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")
    plt.show()


def _hierarchical_average(data, value_col, time_col="timestamp", n_bins=50):
    """
    Two-step hierarchical averaging:
      1. Average within each participant x stimulus-type combination, binned
         over time.
      2. (Performed downstream) average across participants.
    """
    results = []
    for participant in data["participant"].unique():
        participant_data = data[data["participant"] == participant]
        for stimulus in participant_data["stimulus_type"].unique():
            stim_data = participant_data[participant_data["stimulus_type"] == stimulus]
            if len(stim_data) < 5:
                continue

            time_bins = np.linspace(stim_data[time_col].min(), stim_data[time_col].max(), n_bins)
            for i in range(len(time_bins) - 1):
                mask = (stim_data[time_col] >= time_bins[i]) & (stim_data[time_col] < time_bins[i + 1])
                if mask.sum() > 0:
                    time_center = (time_bins[i] + time_bins[i + 1]) / 2
                    value_mean = stim_data[mask][value_col].mean()
                    results.append({
                        "participant": participant,
                        "stimulus_type": stimulus,
                        "timestamp": time_center,
                        value_col: value_mean,
                    })
    return pd.DataFrame(results)


def _plot_hierarchical_data(data, value_col, colors, title, ylabel):
    """Plot mean +/- SEM curves for Target and Distractor conditions."""
    final_data = data.groupby(["timestamp", "stimulus_type"])[value_col].agg(["mean", "sem"]).reset_index()

    for stimulus in ["Target", "Distractor"]:
        stim_data = final_data[final_data["stimulus_type"] == stimulus]
        if len(stim_data) == 0:
            continue
        times = stim_data["timestamp"]
        means = stim_data["mean"]
        sems = stim_data["sem"]
        plt.plot(times, means, color=colors[stimulus], label=stimulus, linewidth=2.5, alpha=0.9)
        plt.fill_between(times, means - sems, means + sems, color=colors[stimulus], alpha=0.2)

    if value_col == "pupil_size":
        plt.ylim(-0.08, 0.12)
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    plt.xlabel("Time (ms)", fontsize=24, fontfamily="Times New Roman", fontweight="bold")
    plt.ylabel(ylabel, fontsize=24, fontfamily="Times New Roman", fontweight="bold")
    plt.title(title, fontsize=24, fontweight="bold", fontfamily="Times New Roman")
    plt.legend(loc="upper left", prop={"family": "Times New Roman", "size": 20, "weight": "bold"})
    plt.tick_params(axis="both", which="major", labelsize=24)

    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")
        label.set_fontweight("bold")
    plt.grid(True, alpha=0.3)


# =============================================================================
# Data loading and main pipeline
# =============================================================================
COLUMN_RENAME_MAP = {
    "leftEyeValidity": "leftValidity",
    "rightEyeValidity": "rightValidity",
}

NUMERIC_COLUMNS = [
    "timestamp", "leftEyeRawX", "leftEyeRawY", "rightEyeRawX", "rightEyeRawY",
    "leftEyePupilSize", "rightEyePupilSize", "leftValidity", "rightValidity",
    "validityScore", "currentTrial", "strColourCode", "resultResponse",
]


def load_data(data_path):
    """Load and standardize the raw eye-tracking CSV export."""
    try:
        df = pd.read_csv(data_path)
    except Exception:
        try:
            df = pd.read_csv(data_path, sep=";")
        except Exception:
            df = pd.read_csv(data_path, header=None)
            if df.shape[1] == 1:
                header = df.iloc[0, 0].split(",")
                df = pd.DataFrame([x.split(",") for x in df.iloc[1:, 0]], columns=header)

    if df.shape[1] == 1 and isinstance(df.columns[0], str) and "," in df.columns[0]:
        header = df.columns[0].split(",")
        data = [row[0].split(",") for _, row in df.iterrows()]
        df = pd.DataFrame(data, columns=header)

    if "id" in df.columns and "participant" not in df.columns:
        df["participant"] = df["id"]

    df.columns = [col.strip().replace('"', "") for col in df.columns]
    df = df.rename(columns=COLUMN_RENAME_MAP)

    for col in df.columns:
        if col in NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def run_pupil_analysis(data_path, output_dir="."):
    """Run the full pupil-only processing, feature extraction, and plotting pipeline."""
    print("Pupil diameter processing pipeline")
    print(f"  Sampling rate: {FRQ} Hz | Inter-sample interval: {DT:.1f} ms | Samples/trial: {N_SAMPLES}")
    print("  Vergence is NOT processed in this script (patented method; see manuscript).")

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None

    df = load_data(data_path)
    print(f"Loaded {len(df)} rows for {df['participant'].nunique()} participants")

    results_detailed, results_features, results_summary = process_pupil_data(df)
    if not results_detailed:
        print("No results were generated.")
        return None

    plot_pupil_response(results_detailed)

    os.makedirs(output_dir, exist_ok=True)

    df_detailed = pd.DataFrame(results_detailed)
    detailed_path = os.path.join(output_dir, "pupil_clean.csv")
    df_detailed.to_csv(detailed_path, index=False)
    print(f"Cleaned pupil signal saved to: {detailed_path}")

    df_features = pd.DataFrame(results_features)
    features_path = os.path.join(output_dir, "pupil_trial_features.csv")
    df_features.to_csv(features_path, index=False)
    print(f"Trial-level feature results saved to: {features_path}")

    if results_summary:
        df_summary = pd.DataFrame(results_summary)
        summary_path = os.path.join(output_dir, "pupil_summary_results.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"Participant-level summary saved to: {summary_path}")

    print("Pupil analysis complete.")
    return results_detailed, results_features, results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pupil diameter processing and feature extraction pipeline.")
    parser.add_argument(
        "--data-path",
        default=os.path.join("data", "eye_tracking_raw_data.csv"),
        help="Path to the eye-tracking CSV export.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where result CSVs and the figure will be saved.",
    )
    args, _unknown = parser.parse_known_args()
    run_pupil_analysis(args.data_path, args.output_dir)
