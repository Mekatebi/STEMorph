"""
STEMorph Validation Analysis
============================

Analysis pipeline for the validation and reliability assessment of
the STEMorph stimulus set (an anger-to-happiness morphed face continuum derived
from NimStim).

Author: Mohammad Ebrahim Katebi (mekatebi.2000@gmail.com)
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.regression.mixed_linear_model import MixedLMResults


# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path("..").resolve()
VALIDITY_DIR = PROJECT_ROOT / "Data_Validity"
RELIABILITY_DIR = PROJECT_ROOT / "Data_Reliability"
GENDER_FILE = PROJECT_ROOT / "Subject_Gender.csv"

RESULTS_ROOT = PROJECT_ROOT / "Results"
VALIDITY_OUT = RESULTS_ROOT / "01_Validity"
GENDER_OUT = RESULTS_ROOT / "02_Gender"
RELIABILITY_OUT = RESULTS_ROOT / "03_Reliability"
RT_OUT = RESULTS_ROOT / "04_RT"
MISSING_OUT = RESULTS_ROOT / "05_Missing"

# Experimental constants
N_TRIALS_PER_PARTICIPANT = 396  # 198 stimuli x 2 presentations
N_MORPH_STEPS = 9
VALID_TRIAL_STATE_CODE = 1
TIMEOUT_STATE_CODE = 3
FACE_GENDER_THRESHOLD = 12  # NimStim persons 1--11 female, 12--22 male

# Predictor names used by the model
PREDICTORS = [
    "Morph_Step",
    "Face_Gender",
    "Participant_Gender",
    "MS_FG",
    "MS_PG",
    "FG_PG",
]

PREDICTOR_LABELS = {
    "Morph_Step": "Morph Step (MS)",
    "Face_Gender": "Face Gender (FG)",
    "Participant_Gender": "Participant Gender (PG)",
    "MS_FG": "MS × FG",
    "MS_PG": "MS × PG",
    "FG_PG": "FG × PG",
}

# Plot aesthetics
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)

MORPH_PALETTE = sns.diverging_palette(10, 220, s=70, l=55, n=N_MORPH_STEPS)
PALETTE_PARTICIPANT = {"Female": "#2596be", "Male": "#be4d25"}
PALETTE_FACE = {"Female": "#00cec9", "Male": "#fdcb6e"}


@dataclass
class ReliabilityResults:
    """Container for test-retest reliability outputs."""

    paired_data: pd.DataFrame
    icc_table: pd.DataFrame
    icc_21: float
    icc_21_ci: object
    regression_intercept: float
    regression_slope: float
    regression_r2: float
    spearman_rho: float
    regression_model: LinearRegression


# =============================================================================
# Utilities
# =============================================================================


def ensure_output_dirs() -> None:
    """Create all output directories if they do not already exist."""
    for directory in [RESULTS_ROOT, VALIDITY_OUT, GENDER_OUT, RELIABILITY_OUT, RT_OUT, MISSING_OUT]:
        directory.mkdir(parents=True, exist_ok=True)


def extract_participant_id(file_name: str) -> int:
    """
    Extract a participant ID from a filename.

    The original files appear to follow a Subject_XXXX... convention, but this
    function is intentionally permissive and uses the first integer group found.
    """
    match = re.search(r"(\d+)", file_name)
    if match is None:
        raise ValueError(
            f"Could not extract participant ID from filename: {file_name}")
    return int(match.group(1))


def require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    """Raise a clear error if a dataframe is missing required columns."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")


def save_text(path: Path, text: str) -> None:
    """Write UTF-8 text to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# =============================================================================
# Data loading and preprocessing
# =============================================================================


def load_participant_file(
    file_path: Path,
    *,
    keep_all_states: bool = False,
    keep_rt: bool = False,
    keep_position_id: bool = False,
) -> pd.DataFrame:
    """
    Load and lightly clean a single participant CSV.
    """
    df = pd.read_csv(file_path)
    df["ID"] = extract_participant_id(file_path.name)

    require_columns(
        df, ["ID", "Face Person", "Morph Step", "Answer"], file_path.name)

    if not keep_all_states:
        if "State" in df.columns:
            df = df[df["State"] == VALID_TRIAL_STATE_CODE].copy()
        df = df.dropna(subset=["Morph Step", "Answer", "Face Person"])

    drop_cols = ["ITI", "Trial_Onset", "Stim_Onset", "Stim_Offset"]
    if not keep_position_id:
        drop_cols.append("Position ID")
    if not keep_rt:
        drop_cols.append("RT")
    if not keep_all_states:
        drop_cols.append("State")

    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    return df


def load_validity_dataset(*, keep_all_states: bool = False, keep_rt: bool = False) -> pd.DataFrame:
    """Load and concatenate all validity-task participant CSV files."""
    if not VALIDITY_DIR.is_dir():
        raise FileNotFoundError(
            f"Validity directory not found: {VALIDITY_DIR}")

    csv_files = sorted(VALIDITY_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {VALIDITY_DIR}")

    frames: list[pd.DataFrame] = []
    for file_path in csv_files:
        try:
            frame = load_participant_file(
                file_path,
                keep_all_states=keep_all_states,
                keep_rt=keep_rt,
                keep_position_id=False,
            )
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            print(f"Skipping {file_path.name}: {exc}")

    if not frames:
        raise RuntimeError("No usable validity data could be loaded.")

    return pd.concat(frames, ignore_index=True)


def standardize_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns with spaces to formula-safe names and enforce numeric types.
    """
    out = df.copy()
    out = out.rename(
        columns={"Morph Step": "Morph_Step", "Face Person": "Face_Person"})

    require_columns(out, ["ID", "Face_Person",
                    "Morph_Step", "Answer"], "model dataframe")

    out["ID"] = out["ID"].astype(int)
    out["Face_Person"] = out["Face_Person"].astype(int)
    out["Morph_Step"] = out["Morph_Step"].astype(int)
    out["Answer"] = pd.to_numeric(out["Answer"], errors="coerce")
    out = out.dropna(subset=["Answer"])
    return out


def load_participant_gender() -> pd.DataFrame:
    """Load participant gender lookup and encode it as 0 = female, 1 = male."""
    if not GENDER_FILE.is_file():
        raise FileNotFoundError(
            f"Participant gender file not found: {GENDER_FILE}")

    gender = pd.read_csv(GENDER_FILE)
    require_columns(gender, ["ID", "Gender"], GENDER_FILE.name)
    gender = gender.copy()
    gender["ID"] = gender["ID"].astype(int)
    gender["Gender"] = gender["Gender"].astype(str).str.strip()
    gender["Participant_Gender"] = gender["Gender"].str.lower().eq(
        "male").astype(int)
    return gender[["ID", "Gender", "Participant_Gender"]]


def add_gender_and_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add face gender, participant gender, and all two-way interaction columns.

    Coding:
        Face_Gender:        0 = female face, 1 = male face
        Participant_Gender: 0 = female participant, 1 = male participant
    """
    out = standardize_model_columns(df)
    out["Face_Gender"] = (out["Face_Person"] >=
                          FACE_GENDER_THRESHOLD).astype(int)

    gender = load_participant_gender()
    out = out.merge(gender, on="ID", how="inner")

    out["MS_FG"] = out["Morph_Step"] * out["Face_Gender"]
    out["MS_PG"] = out["Morph_Step"] * out["Participant_Gender"]
    out["FG_PG"] = out["Face_Gender"] * out["Participant_Gender"]
    return out


# =============================================================================
# Missing-trial analysis
# =============================================================================


def analyze_missing_trials(df_all_states: pd.DataFrame) -> dict:
    """
    Analyze timed-out trials (State == 3).
    """
    df = df_all_states.copy()
    require_columns(df, ["State", "Face Person",
                    "Morph Step", "ID"], "all-state validity data")

    df["Missing"] = (df["State"] == TIMEOUT_STATE_CODE).astype(int)
    df["Face_Gender"] = (df["Face Person"] >=
                         FACE_GENDER_THRESHOLD).astype(int)

    gender = load_participant_gender()
    df = df.merge(gender[["ID", "Participant_Gender"]], on="ID", how="left")

    n_total = len(df)
    n_missing = int(df["Missing"].sum())
    pct_missing = 100 * n_missing / n_total if n_total else np.nan

    miss_by_step = df.groupby("Morph Step")["Missing"].sum().reindex(
        range(1, 10), fill_value=0)

    ct_step = pd.crosstab(df["Morph Step"], df["Missing"])
    chi2_step, p_step, _, _ = stats.chi2_contingency(
        ct_step) if ct_step.shape[1] == 2 else (np.nan, np.nan, None, None)

    ct_face = pd.crosstab(df["Face_Gender"], df["Missing"])
    chi2_face, p_face, _, _ = stats.chi2_contingency(
        ct_face) if ct_face.shape == (2, 2) else (np.nan, np.nan, None, None)

    ct_participant = pd.crosstab(df["Participant_Gender"], df["Missing"])
    chi2_participant, p_participant, _, _ = (
        stats.chi2_contingency(ct_participant) if ct_participant.shape == (
            2, 2) else (np.nan, np.nan, None, None)
    )

    summary = {
        "n_total": n_total,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
        "miss_by_step": miss_by_step,
        "chi2_step": chi2_step,
        "p_step": p_step,
        "chi2_face_gender": chi2_face,
        "p_face_gender": p_face,
        "chi2_participant_gender": chi2_participant,
        "p_participant_gender": p_participant,
    }

    save_missing_summary(summary)
    return summary


def save_missing_summary(summary: dict) -> None:
    """Save missing-trial statistics as a text file."""
    text = [
        "Missing-trials analysis (State == 3, timeout)",
        "============================================",
        "",
        f"Total trials                : {summary['n_total']}",
        f"Timed-out trials            : {summary['n_missing']}",
        f"Percentage missing          : {summary['pct_missing']:.2f}%",
        "",
        "Missing trials per morph step:",
        summary["miss_by_step"].to_string(),
        "",
        "Chi-squared test, missingness x morph step:",
        f"    chi2 = {summary['chi2_step']:.3f}, p = {summary['p_step']:.4f}",
        "",
        "Chi-squared test, missingness x face gender:",
        f"    chi2 = {summary['chi2_face_gender']:.3f}, p = {summary['p_face_gender']:.4f}",
        "",
        "Chi-squared test, missingness x participant gender:",
        f"    chi2 = {summary['chi2_participant_gender']:.3f}, p = {summary['p_participant_gender']:.4f}",
        "",
    ]
    save_text(MISSING_OUT / "Missing_Stats.txt", "\n".join(text))


# =============================================================================
# Response-time analysis
# =============================================================================


def analyze_response_times(df_valid_with_rt: pd.DataFrame) -> dict:
    """
    Analyze response times for valid response trials.
    """
    require_columns(df_valid_with_rt, ["RT"], "valid trials with RT")

    df = df_valid_with_rt.copy()
    df = df[df["RT"] > 0].copy()
    df = add_gender_and_interactions(df)
    df["log_RT"] = np.log(df["RT"])
    df["MS_c"] = df["Morph_Step"] - 5
    df["MS_c2"] = df["MS_c"] ** 2
    df["grp"] = 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            "log_RT ~ MS_c + MS_c2 + Face_Gender + Participant_Gender",
            data=df,
            groups=df["grp"],
            vc_formula={
                "ID": "0 + C(ID)", "Face_Person": "0 + C(Face_Person)"},
            re_formula="0",
        ).fit(method="lbfgs")

    rt_by_step = df.groupby("Morph_Step")["RT"].agg(["mean", "std", "count"])
    summary = {
        "model": model,
        "rt_by_step": rt_by_step,
        "n_trials": len(df),
        "mean_rt": float(df["RT"].mean()),
        "median_rt": float(df["RT"].median()),
        "sd_rt": float(df["RT"].std()),
    }

    save_rt_summary(summary)
    return summary


def save_rt_summary(summary: dict) -> None:
    """Save response-time statistics as a text file."""
    text = [
        "Response time analysis (valid trials only, State == 1)",
        "=======================================================",
        "",
        f"N valid trials : {summary['n_trials']}",
        f"Mean RT (s)    : {summary['mean_rt']:.3f}",
        f"Median RT (s)  : {summary['median_rt']:.3f}",
        f"SD RT (s)      : {summary['sd_rt']:.3f}",
        "",
        "RT per morph step:",
        summary["rt_by_step"].to_string(),
        "",
        "Mixed-effects model on log-RT",
        "centered morph step + quadratic term + face/participant gender,",
        "with random intercepts for participant and face identity",
        "",
        summary["model"].summary().as_text(),
    ]
    save_text(RT_OUT / "RT_Stats.txt", "\n".join(text))


# =============================================================================
# Validity: linear mixed-effects model
# =============================================================================


def fit_validity_lmm(df: pd.DataFrame) -> MixedLMResults:
    """
    Fit the primary validity model.

    Model specification:
        Rating ~ MS + FG + PG + MS:FG + MS:PG + FG:PG
                 + (1 | participant)
                 + (1 | face identity)

    In statsmodels this crossed random-intercept structure is implemented using
    a single dummy grouping variable and variance components for participant and
    face identity.
    """
    model_df = df.copy()
    model_df["grp"] = 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            "Answer ~ Morph_Step + Face_Gender + Participant_Gender + MS_FG + MS_PG + FG_PG",
            data=model_df,
            groups=model_df["grp"],
            vc_formula={
                "ID": "0 + C(ID)", "Face_Person": "0 + C(Face_Person)"},
            re_formula="0",
        ).fit(method="lbfgs")

    return model


def lmm_r_squared(model: MixedLMResults) -> tuple[float, float]:
    """
    Compute Nakagawa-style marginal and conditional pseudo-R² for the LMM.

    Marginal R²: fixed effects only.
    Conditional R²: fixed + random effects.
    """
    fixed_pred = model.predict()
    var_fixed = float(np.var(fixed_pred, ddof=1))
    var_random = float(np.sum(model.vcomp))
    var_resid = float(model.scale)
    denom = var_fixed + var_random + var_resid
    return var_fixed / denom, (var_fixed + var_random) / denom


def fixed_effect_partial_residuals(
    df: pd.DataFrame,
    model: MixedLMResults,
    remove_vars: Optional[list[str]] = None,
) -> pd.Series:
    """
    Compute fixed-effect partial residuals for visualization.
    """
    if remove_vars is None:
        remove_vars = ["Face_Gender", "Participant_Gender",
                       "MS_FG", "MS_PG", "FG_PG"]

    contribution = np.zeros(len(df), dtype=float)
    for var in remove_vars:
        contribution += float(model.params[var]) * df[var].to_numpy()
    return df["Answer"] - contribution


def remove_step_outliers(
    df: pd.DataFrame,
    value_col: str,
    *,
    step_col: str = "Morph_Step",
    k_sd: float = 2.0,
) -> pd.DataFrame:
    """Return a dataframe with values > k_sd from the within-step mean removed."""
    out = df.copy()
    group_mean = out.groupby(step_col)[value_col].transform("mean")
    group_sd = out.groupby(step_col)[value_col].transform("std")
    keep = (group_sd == 0) | (
        np.abs(out[value_col] - group_mean) <= k_sd * group_sd)
    return out.loc[keep].copy()


def create_validity_plot(df: pd.DataFrame, model: MixedLMResults, out_path: Path) -> None:
    """Create the main validity figure."""
    plot_df = df.copy()
    plot_df["Partial_Residual"] = fixed_effect_partial_residuals(
        plot_df, model)

    fig, ax = plt.subplots(figsize=(8, 8))

    x_jitter = np.random.normal(0, 0.04, len(plot_df))
    y_jitter = np.random.normal(0, 0.04, len(plot_df))
    ax.scatter(
        plot_df["Morph_Step"] - 1 + x_jitter,
        plot_df["Partial_Residual"] + y_jitter,
        s=2,
        color="#888888",
        alpha=0.35,
        marker=".",
        zorder=1,
        rasterized=True,
    )

    violin_df = remove_step_outliers(
        plot_df[["Morph_Step", "Partial_Residual"]], "Partial_Residual")
    sns.violinplot(
        data=violin_df,
        x="Morph_Step",
        y="Partial_Residual",
        hue="Morph_Step",
        palette=MORPH_PALETTE,
        legend=False,
        inner=None,
        cut=2,
        width=0.72,
        linewidth=0.8,
        alpha=0.5,
        ax=ax,
        zorder=2,
    )

    means = plot_df.groupby("Morph_Step")["Partial_Residual"].mean()
    ax.scatter(means.index - 1, means.values, color="#111111",
               s=70, zorder=5, label="Step mean")

    slope = float(model.params["Morph_Step"])
    intercept = float(plot_df["Partial_Residual"].mean(
    ) - slope * (plot_df["Morph_Step"].mean() - 1))
    x_line = np.array([0.5, 9.5]) - 1
    r2_marg, r2_cond = lmm_r_squared(model)
    ax.plot(
        x_line,
        intercept + slope * x_line,
        color="#111111",
        linewidth=1.8,
        linestyle="--",
        zorder=4,
        label=f"LMM slope = {slope:.3f}\n$R^2_{{marg}}$ = {r2_marg:.3f}, $R^2_{{cond}}$ = {r2_cond:.3f}",
    )

    ax.set_xticks(range(N_MORPH_STEPS))
    ax.set_xticklabels(range(1, N_MORPH_STEPS + 1))
    ax.set_yticks(range(1, 10))
    ax.set_xlim(-1, 9)
    ax.set_ylim(0, 10)
    ax.set_title("Emotional Ratings Across Morphing Steps", pad=10)
    ax.set_xlabel("Morph Step (1 = Angry → 9 = Happy)")
    ax.set_ylabel("Participants' Rating (Partial Residual)")
    ax.legend(loc="upper left", fontsize=12, frameon=True, edgecolor="#cccccc")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)

    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def coefficient_table(model: MixedLMResults) -> pd.DataFrame:
    """Return a tidy fixed-effect coefficient table."""
    conf = model.conf_int().rename(columns={0: "CI_low", 1: "CI_high"})
    table = pd.DataFrame(
        {
            "Predictor": [PREDICTOR_LABELS[p] for p in PREDICTORS],
            "beta": [model.params[p] for p in PREDICTORS],
            "SE": [model.bse[p] for p in PREDICTORS],
            "z": [model.tvalues[p] for p in PREDICTORS],
            "p": [model.pvalues[p] for p in PREDICTORS],
            "CI_low": [conf.loc[p, "CI_low"] for p in PREDICTORS],
            "CI_high": [conf.loc[p, "CI_high"] for p in PREDICTORS],
        }
    )
    return table


def create_coefficient_plot(model: MixedLMResults, out_path: Path) -> None:
    """Create a coefficient plot with 95% CIs and p < .05 asterisks."""
    table = coefficient_table(model)
    abs_max = float(np.abs(table["beta"]).max())
    colors = plt.cm.coolwarm(
        plt.Normalize(-abs_max, abs_max)(table["beta"].to_numpy()))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(table))
    err_low = table["beta"].to_numpy() - table["CI_low"].to_numpy()
    err_high = table["CI_high"].to_numpy() - table["beta"].to_numpy()

    ax.barh(
        y_pos,
        table["beta"],
        xerr=np.vstack([err_low, err_high]),
        color=colors,
        height=0.6,
        capsize=3,
        error_kw={"elinewidth": 1.2, "ecolor": "#333333", "capthick": 1.2},
        zorder=3,
    )

    for i, row in table.iterrows():
        if row["p"] < 0.05:
            if row["beta"] >= 0:
                ax.text(row["CI_high"] + 0.015, i + 0.065, "*",
                        ha="left", va="top", fontsize=18)
            else:
                ax.text(row["CI_low"] - 0.015, i + 0.065, "*",
                        ha="right", va="top", fontsize=18)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(table["Predictor"], fontsize=9)
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xlabel("β Estimate (95% CI)", fontsize=10)
    ax.set_title("Mixed-Effects Regression Coefficients", pad=12)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, color="#cccccc", zorder=0)

    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_validity_outputs(df: pd.DataFrame, model: MixedLMResults) -> None:
    """Save validity model summary, coefficient table, and figures."""
    r2_marg, r2_cond = lmm_r_squared(model)
    table = coefficient_table(model)
    table.to_csv(VALIDITY_OUT / "Validity_Coefficients.csv", index=False)

    text = [
        "Linear mixed-effects model (Equation 1, primary analysis)",
        "=========================================================",
        "",
        "Specification:",
        "  Answer ~ MS + FG + PG + MS:FG + MS:PG + FG:PG",
        "         + (1 | participant ID)",
        "         + (1 | face identity)",
        "",
        f"Marginal R² (fixed effects only)         : {r2_marg:.4f}",
        f"Conditional R² (fixed + random effects)  : {r2_cond:.4f}",
        "",
        model.summary().as_text(),
    ]
    save_text(VALIDITY_OUT / "Validity_Model_Stats.txt", "\n".join(text))

    create_validity_plot(df, model, VALIDITY_OUT / "Validity_Regression.png")
    create_coefficient_plot(model, VALIDITY_OUT / "Validity_Coefficients.png")


# =============================================================================
# Outlier sensitivity analysis
# =============================================================================


def outlier_sensitivity_analysis(df: pd.DataFrame, full_model: Optional[MixedLMResults] = None) -> dict:
    """
    Refit the primary LMM after removing per-morph-step outliers (>2 SD).
    """
    if full_model is None:
        full_model = fit_validity_lmm(df)

    trimmed_df = remove_step_outliers(df, "Answer")
    trimmed_model = fit_validity_lmm(trimmed_df)

    rows = []
    for predictor in PREDICTORS:
        full_beta = float(full_model.params[predictor])
        trimmed_beta = float(trimmed_model.params[predictor])
        delta = trimmed_beta - full_beta
        rows.append(
            {
                "Predictor": PREDICTOR_LABELS[predictor],
                "Full_beta": full_beta,
                "Full_p": float(full_model.pvalues[predictor]),
                "Trimmed_beta": trimmed_beta,
                "Trimmed_p": float(trimmed_model.pvalues[predictor]),
                "Delta_beta": delta,
                "Percent_delta": 100 * delta / (abs(full_beta) + 1e-12),
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(VALIDITY_OUT / "Outlier_Sensitivity.csv", index=False)

    n_dropped = len(df) - len(trimmed_df)
    text = [
        "Outlier sensitivity analysis",
        "============================",
        "",
        "Primary mixed-effects model fit on the full dataset and on the outlier-trimmed dataset",
        "(per morph step, >2 SD from the step mean removed).",
        "",
        f"Full dataset N        : {len(df)}",
        f"Trimmed dataset N     : {len(trimmed_df)}",
        f"Observations excluded : {n_dropped} ({100 * n_dropped / len(df):.2f}%)",
        "",
        comparison.to_string(
            index=False, float_format=lambda value: f"{value:.4f}"),
    ]
    save_text(VALIDITY_OUT / "Outlier_Sensitivity.txt", "\n".join(text))

    return {"trimmed_data": trimmed_df, "trimmed_model": trimmed_model, "comparison": comparison}


# =============================================================================
# Gender subgroup and identity-level analyses
# =============================================================================


def subgroup_partial_residuals(df: pd.DataFrame, model: MixedLMResults, group_type: str) -> pd.Series:
    """
    Compute fixed-effect partial residuals for subgroup visualizations.

    group_type = 'participant': preserve participant-gender terms and remove face-gender terms.
    group_type = 'face': preserve face-gender terms and remove participant-gender terms.
    """
    if group_type == "participant":
        remove_vars = ["Face_Gender", "MS_FG", "FG_PG"]
    elif group_type == "face":
        remove_vars = ["Participant_Gender", "MS_PG", "FG_PG"]
    else:
        raise ValueError("group_type must be either 'participant' or 'face'.")

    contribution = np.zeros(len(df), dtype=float)
    for var in remove_vars:
        contribution += float(model.params[var]) * df[var].to_numpy()
    return df["Answer"] - contribution


def partial_r2(table: pd.DataFrame, label: str) -> float:
    """Squared Pearson correlation between morph step and partial residual for one subgroup."""
    sub = table[table["Gender_Group"] == label]
    if len(sub) < 3:
        return float("nan")
    return float(np.corrcoef(sub["Morph_Step"], sub["Partial_Residual"])[0, 1] ** 2)


def create_subgroup_plot(df: pd.DataFrame, model: MixedLMResults, group_type: str) -> None:
    """Create subgroup validity plots for participant gender or face gender."""
    if group_type == "participant":
        group_col = "Participant_Gender"
        palette = PALETTE_PARTICIPANT
        title = "Participant Gender Subgroups"
        male_slope_extra = float(model.params["MS_PG"])
        male_intercept_extra = float(model.params["Participant_Gender"])
        out_file = GENDER_OUT / "Validity_Participant_Gender_Subgroups.png"
    elif group_type == "face":
        group_col = "Face_Gender"
        palette = PALETTE_FACE
        title = "Face Gender Subgroups"
        male_slope_extra = float(model.params["MS_FG"])
        male_intercept_extra = float(model.params["Face_Gender"])
        out_file = GENDER_OUT / "Validity_Face_Gender_Subgroups.png"
    else:
        raise ValueError("group_type must be either 'participant' or 'face'.")

    plot_df = df.copy()
    plot_df["Partial_Residual"] = subgroup_partial_residuals(
        plot_df, model, group_type)
    plot_df["Gender_Group"] = np.where(
        plot_df[group_col] == 0, "Female", "Male")

    group_mean = plot_df.groupby(["Morph_Step", "Gender_Group"])[
        "Partial_Residual"].transform("mean")
    group_sd = plot_df.groupby(["Morph_Step", "Gender_Group"])[
        "Partial_Residual"].transform("std")
    violin_df = plot_df[np.abs(
        plot_df["Partial_Residual"] - group_mean) <= 2 * group_sd].copy()

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.violinplot(
        data=violin_df,
        x="Morph_Step",
        y="Partial_Residual",
        hue="Gender_Group",
        palette=palette,
        split=True,
        inner=None,
        cut=2,
        gap=0.45,
        linewidth=0.8,
        alpha=0.6,
        width=1.4,
        ax=ax,
        zorder=2,
    )

    base_slope = float(model.params["Morph_Step"])
    base_intercept = float(model.params["Intercept"])
    r2_values = {"Female": partial_r2(
        plot_df, "Female"), "Male": partial_r2(plot_df, "Male")}

    for label, color in palette.items():
        if label == "Female":
            slope = base_slope
            intercept = base_intercept
        else:
            slope = base_slope + male_slope_extra
            intercept = base_intercept + male_intercept_extra

        x_line = np.array([1, 9])
        ax.plot(x_line - 1, intercept + slope * x_line, color=color,
                linestyle="--", linewidth=1.8, zorder=4)

        means = plot_df[plot_df["Gender_Group"] == label].groupby("Morph_Step")[
            "Partial_Residual"].mean()
        ax.scatter(means.index - 1, means.values, color=color,
                   s=80, edgecolor="white", linewidth=1.5, zorder=5)

    handles = [
        plt.Line2D([], [], color=color, linewidth=2, linestyle="--",
                   label=f"{label} $R^2$ = {r2_values[label]:.3f}")
        for label, color in palette.items()
    ]
    ax.legend(handles=handles, loc="upper left",
              fontsize=12, frameon=True, edgecolor="#cccccc")

    ax.set_xticks(range(N_MORPH_STEPS))
    ax.set_xticklabels(range(1, N_MORPH_STEPS + 1))
    ax.set_yticks(range(1, 10))
    ax.set_xlim(-1, 9)
    ax.set_ylim(0, 10)
    ax.set_title(title, pad=10)
    ax.set_xlabel("Morph Step (1 = Angry → 9 = Happy)")
    ax.set_ylabel("Participants' Rating (Fixed-Effect Partial Residual)")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)

    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def analyze_face_identities(df: pd.DataFrame) -> dict:
    """
    Compare face-identity mean ratings by face gender.
    """
    identity_means = (
        df.groupby(["Face_Person", "Face_Gender"], as_index=False)["Answer"]
        .mean()
        .rename(columns={"Answer": "Mean_Rating"})
    )
    identity_means["Gender_Group"] = np.where(
        identity_means["Face_Gender"] == 0, "Female", "Male")

    female = identity_means.loc[identity_means["Gender_Group"]
                                == "Female", "Mean_Rating"]
    male = identity_means.loc[identity_means["Gender_Group"]
                              == "Male", "Mean_Rating"]

    u_stat, p_value = stats.mannwhitneyu(female, male, alternative="greater")
    pooled_sd = np.sqrt((female.var(ddof=1) + male.var(ddof=1)) / 2)
    cohens_d = (female.mean() - male.mean()) / \
        pooled_sd if pooled_sd > 0 else np.nan

    results = {
        "identity_means": identity_means,
        "female_mean": float(female.mean()),
        "female_sd": float(female.std(ddof=1)),
        "male_mean": float(male.mean()),
        "male_sd": float(male.std(ddof=1)),
        "u_stat": float(u_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
    }

    identity_means.to_csv(
        GENDER_OUT / "Identity_Mean_Ratings.csv", index=False)
    save_identity_summary(results)
    create_identity_plot(identity_means, results)
    return results


def save_identity_summary(results: dict) -> None:
    """Save identity-level face-gender comparison statistics."""
    text = [
        "Per-identity face gender comparison",
        "===================================",
        "",
        f"Female faces (n=11): mean = {results['female_mean']:.3f}, SD = {results['female_sd']:.3f}",
        f"Male faces   (n=11): mean = {results['male_mean']:.3f}, SD = {results['male_sd']:.3f}",
        "",
        f"Mann-Whitney U = {results['u_stat']:.2f}",
        f"p (one-tailed, F > M) = {results['p_value']:.4f}",
        f"Cohen's d = {results['cohens_d']:.3f}",
    ]
    save_text(GENDER_OUT / "Identity_Comparison.txt", "\n".join(text))


def create_identity_plot(identity_means: pd.DataFrame, results: dict) -> None:
    """Create a violin/strip plot of identity-level mean ratings by face gender."""
    fig, ax = plt.subplots(figsize=(6, 7))

    sns.violinplot(
        data=identity_means,
        x="Gender_Group",
        y="Mean_Rating",
        hue="Gender_Group",
        palette=PALETTE_FACE,
        inner="quartile",
        cut=1,
        width=0.55,
        linewidth=0.8,
        alpha=0.3,
        legend=False,
        ax=ax,
        zorder=2,
    )
    sns.stripplot(
        data=identity_means,
        x="Gender_Group",
        y="Mean_Rating",
        color="#333333",
        size=5,
        jitter=0.08,
        alpha=0.7,
        ax=ax,
        zorder=3,
    )

    ax.set_xlabel("Face Gender")
    ax.set_ylabel("Mean Rating Across All Steps")
    ax.set_title("Per-Identity Rating Distributions", pad=10)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)
    ax.text(
        0.6,
        0.97,
        f"Mann-Whitney U = {results['u_stat']:.1f}\np (one-tailed, F > M) = {results['p_value']:.3f}\nCohen's d = {results['cohens_d']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(GENDER_OUT / "Identity_Comparison.png")
    plt.close()


# =============================================================================
# Test-retest reliability
# =============================================================================


def add_repetition_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a repetition index to prevent duplicate-stimulus Cartesian merges.
    """
    out = df.copy()
    key_cols = ["Position ID", "Face Person", "Morph Step"]
    require_columns(out, key_cols, "test-retest pairing data")
    out["Repetition"] = out.groupby(key_cols).cumcount() + 1
    return out


def load_raw_validity_file_for_participant(participant_id: int) -> Optional[pd.DataFrame]:
    """Find and load the validity CSV for one participant ID."""
    for file_path in sorted(VALIDITY_DIR.glob("*.csv")):
        try:
            if extract_participant_id(file_path.name) == participant_id:
                return load_participant_file(
                    file_path,
                    keep_all_states=False,
                    keep_rt=False,
                    keep_position_id=True,
                )
        except ValueError:
            continue
    return None


def load_paired_validity_retest() -> pd.DataFrame:
    """
    Build the matched test-retest dataframe.
    """
    if not RELIABILITY_DIR.is_dir():
        raise FileNotFoundError(
            f"Reliability directory not found: {RELIABILITY_DIR}")

    reliability_files = sorted(RELIABILITY_DIR.glob("*.csv"))
    if not reliability_files:
        raise FileNotFoundError(
            f"No reliability CSV files found in {RELIABILITY_DIR}")

    paired_frames: list[pd.DataFrame] = []
    for reliability_path in reliability_files:
        participant_id = extract_participant_id(reliability_path.name)

        retest = load_participant_file(
            reliability_path,
            keep_all_states=False,
            keep_rt=False,
            keep_position_id=True,
        )
        validity = load_raw_validity_file_for_participant(participant_id)
        if validity is None:
            print(
                f"No matching validity file found for participant {participant_id}; skipping.")
            continue

        retest = add_repetition_index(retest)
        validity = add_repetition_index(validity)

        merge_keys = ["Position ID", "Face Person", "Morph Step", "Repetition"]
        merged = pd.merge(
            retest,
            validity,
            on=merge_keys,
            suffixes=("_Retest", "_Validity"),
            how="inner",
        )
        merged["ID"] = participant_id
        paired_frames.append(merged)
        print(
            f"Matched participant {participant_id}: {len(merged)} paired trials")

    if not paired_frames:
        raise RuntimeError("No matched test-retest data could be assembled.")

    paired = pd.concat(paired_frames, ignore_index=True)
    require_columns(
        paired, ["Answer_Retest", "Answer_Validity"], "paired test-retest data")
    return paired


def compute_icc_21(paired_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ICC table for the paired test-retest data.
    """
    rows = []
    for trial_id, row in paired_df.reset_index(drop=True).iterrows():
        rows.append({"trial": trial_id, "session": "validity",
                    "rating": row["Answer_Validity"]})
        rows.append({"trial": trial_id, "session": "retest",
                    "rating": row["Answer_Retest"]})

    long_df = pd.DataFrame(rows)
    return pg.intraclass_corr(data=long_df, targets="trial", raters="session", ratings="rating")


def fit_reliability_regression(paired_df: pd.DataFrame) -> tuple[LinearRegression, float, float, float, float]:
    """Fit companion OLS regression: retest rating ~ validity rating."""
    x = paired_df["Answer_Validity"].to_numpy().reshape(-1, 1)
    y = paired_df["Answer_Retest"].to_numpy()
    model = LinearRegression().fit(x, y)
    r2 = float(model.score(x, y))
    spearman_rho = float(
        paired_df[["Answer_Validity", "Answer_Retest"]].corr("spearman").iloc[0, 1])
    return model, float(model.intercept_), float(model.coef_[0]), r2, spearman_rho


def analyze_reliability() -> ReliabilityResults:
    """Run the complete test-retest reliability analysis."""
    paired = load_paired_validity_retest()
    icc_table = compute_icc_21(paired)
    icc_row = icc_table.loc[icc_table["Type"] == "ICC2"].iloc[0]
    icc_21 = float(icc_row["ICC"])
    icc_21_ci = icc_row["CI95%"]

    regression_model, intercept, slope, r2, spearman_rho = fit_reliability_regression(
        paired)

    results = ReliabilityResults(
        paired_data=paired,
        icc_table=icc_table,
        icc_21=icc_21,
        icc_21_ci=icc_21_ci,
        regression_intercept=intercept,
        regression_slope=slope,
        regression_r2=r2,
        spearman_rho=spearman_rho,
        regression_model=regression_model,
    )

    save_reliability_summary(results)
    create_reliability_plot(results)
    return results


def save_reliability_summary(results: ReliabilityResults) -> None:
    """Save reliability statistics as a text file."""
    text = [
        "Test-retest reliability analysis",
        "================================",
        "",
        "Primary metric: ICC(2,1)",
        "  Two-way random-effects, absolute-agreement, single-measure ICC.",
        "",
        results.icc_table.to_string(index=False),
        "",
        f"Reported ICC(2,1) = {results.icc_21:.3f}, 95% CI {results.icc_21_ci}",
        "",
        "Companion OLS regression (retest ~ validity):",
        f"  Intercept  = {results.regression_intercept:.3f}",
        f"  Slope      = {results.regression_slope:.3f}",
        f"  R²         = {results.regression_r2:.3f}",
        f"  Spearman ρ = {results.spearman_rho:.3f}",
    ]
    save_text(RELIABILITY_OUT / "Reliability_Stats.txt", "\n".join(text))


def create_reliability_plot(results: ReliabilityResults) -> None:
    """Create the reliability figure with ICC and companion regression."""
    df = results.paired_data.copy()
    plot_df = df.rename(
        columns={"Answer_Validity": "Validity_Rating", "Answer_Retest": "Retest_Rating"})

    fig, ax = plt.subplots(figsize=(8, 8))

    violin_df = remove_step_outliers(
        plot_df[["Validity_Rating", "Retest_Rating"]], "Retest_Rating", step_col="Validity_Rating")
    sns.violinplot(
        data=violin_df,
        x="Validity_Rating",
        y="Retest_Rating",
        hue="Validity_Rating",
        palette=MORPH_PALETTE,
        legend=False,
        inner=None,
        cut=2,
        native_scale=True,
        linewidth=0.8,
        width=0.6,
        alpha=0.6,
        ax=ax,
        zorder=2,
    )

    x_jitter = np.random.normal(0, 0.06, len(plot_df))
    y_jitter = np.random.normal(0, 0.06, len(plot_df))
    ax.scatter(
        plot_df["Validity_Rating"] + x_jitter,
        plot_df["Retest_Rating"] + y_jitter,
        s=2,
        color="#888888",
        alpha=0.35,
        marker=".",
        zorder=1,
        rasterized=True,
    )

    means = plot_df.groupby("Validity_Rating")["Retest_Rating"].mean()
    ax.scatter(means.index, means.values, color="#111111",
               s=60, zorder=5, label="Rating mean")

    x_line = np.array([0.5, 9.5])
    y_line = results.regression_model.predict(x_line.reshape(-1, 1))
    ax.plot(
        x_line,
        y_line,
        color="#111111",
        linewidth=1.8,
        linestyle="--",
        zorder=4,
        label=f"ICC(2,1) = {results.icc_21:.3f}\n$R^2$ = {results.regression_r2:.3f}",
    )

    ax.set_xticks(np.arange(1, 10))
    ax.set_yticks(np.arange(1, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Reliability of Participants' Ratings", pad=10)
    ax.set_xlabel("Validity Rating")
    ax.set_ylabel("Matched Retest Rating")
    ax.legend(loc="upper left", fontsize=12, frameon=True, edgecolor="#cccccc")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)

    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(RELIABILITY_OUT / "Reliability_Regression.png")
    plt.close()


# =============================================================================
# Main pipeline
# =============================================================================


def main() -> None:
    """Run the STEMorph analysis pipeline."""
    ensure_output_dirs()

    print("\n[1/7] Loading validity dataset with all trial states...")
    all_trials = load_validity_dataset(keep_all_states=True, keep_rt=True)
    print(f"      Loaded trials: {len(all_trials)}")

    print("\n[2/7] Missing-trials analysis...")
    missing = analyze_missing_trials(all_trials)
    print(
        f"      Timed-out trials: {missing['n_missing']} ({missing['pct_missing']:.2f}%)")

    print("\n[3/7] Response-time analysis...")
    valid_rt = all_trials[all_trials["State"] == VALID_TRIAL_STATE_CODE].copy()
    valid_rt = valid_rt.drop(columns=["State"], errors="ignore")
    rt_summary = analyze_response_times(valid_rt)
    print(
        f"      Mean RT: {rt_summary['mean_rt']:.3f}s; median RT: {rt_summary['median_rt']:.3f}s")

    print("\n[4/7] Validity LMM...")
    model_df = valid_rt.drop(columns=["RT"], errors="ignore")
    model_df = add_gender_and_interactions(model_df)
    validity_model = fit_validity_lmm(model_df)
    r2_marg, r2_cond = lmm_r_squared(validity_model)
    print(f"      Converged: {validity_model.converged}")
    print(
        f"      Morph Step β = {validity_model.params['Morph_Step']:.3f}, p = {validity_model.pvalues['Morph_Step']:.3g}")
    print(f"      R² marginal = {r2_marg:.3f}, conditional = {r2_cond:.3f}")
    save_validity_outputs(model_df, validity_model)

    print("\n[5/7] Outlier sensitivity analysis...")
    outlier_sensitivity_analysis(model_df, validity_model)
    print("      Outlier sensitivity outputs saved.")

    print("\n[6/7] Gender subgroup and identity-level analyses...")
    create_subgroup_plot(model_df, validity_model, "face")
    create_subgroup_plot(model_df, validity_model, "participant")
    identity_results = analyze_face_identities(model_df)
    print(
        f"      Identity-level U = {identity_results['u_stat']:.1f}, "
        f"p = {identity_results['p_value']:.4f}, d = {identity_results['cohens_d']:.3f}"
    )

    print("\n[7/7] Reliability analysis...")
    reliability = analyze_reliability()
    print(
        f"      ICC(2,1) = {reliability.icc_21:.3f}, 95% CI {reliability.icc_21_ci}")
    print(f"      Companion regression R² = {reliability.regression_r2:.3f}")

    print(f"\nDone. All outputs were written under: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
