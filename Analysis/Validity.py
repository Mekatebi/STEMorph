"""
Validity Analysis for STEMorph

Author: Mohammad Ebrahim Katebi (mekatebi.2000@gmail.com)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FILES_ADDRESS = "../Data_Validity/"
RESULTS_DIR = "../Results_Validity/"
GENDER_FILE = "../Subject_Gender.csv"

# Diverging palette: red (angry) -> neutral -> blue (happy)
MORPH_PALETTE = sns.diverging_palette(10, 220, s=70, l=55, n=9, as_cmap=False)


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def validate_data(table: pd.DataFrame) -> bool:
    """Return True only when the table contains the required columns and data."""
    required_columns = ["ID", "Face Person", "Morph Step", "Answer"]
    if not all(col in table.columns for col in required_columns):
        print("Error: Missing required columns.")
        return False
    if table.empty:
        print("Error: Empty data table.")
        return False
    if table["Morph Step"].isnull().any() or table["Answer"].isnull().any():
        print("Error: Null values found in 'Morph Step' or 'Answer'.")
        return False
    return True


def load_and_preprocess_data(file_name: str) -> pd.DataFrame:
    """
    Load a single participant CSV, tag it with the subject ID extracted from
    the filename, filter to valid trials (State == 1), and drop bookkeeping
    columns.
    """
    try:
        data = pd.read_csv(os.path.join(FILES_ADDRESS, file_name), header=0)
        data["ID"] = int(file_name.split("_")[1])

        if "State" in data.columns:
            data = data[data["State"] == 1]

        drop_cols = [
            "State", "Position ID", "ITI",
            "Trial_Onset", "Stim_Onset", "Stim_Offset",
        ]
        data = data.drop(columns=[c for c in drop_cols if c in data.columns])

        return data if validate_data(data) else pd.DataFrame()

    except Exception as exc:
        print(f"Error loading {file_name}: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def create_interaction_terms(table: pd.DataFrame) -> pd.DataFrame:
    """Append all two-way interaction columns required by Equation 1."""
    table = table.copy()
    table["MS_FG"] = table["Morph Step"] * table["Face_Gender"]
    table["MS_SG"] = table["Morph Step"] * table["Subject_Gender"]
    table["FG_SG"] = table["Face_Gender"] * table["Subject_Gender"]
    return table


# ---------------------------------------------------------------------------
# Residuals & outlier removal
# ---------------------------------------------------------------------------

def calculate_partial_residuals(
    table: pd.DataFrame,
    model: sm.regression.linear_model.RegressionResultsWrapper,
) -> pd.Series:
    """
    Remove variance attributable to gender-related predictors from raw answers,
    isolating the morph-step signal for partial regression plots.
    """
    gender_vars = ["Subject_Gender", "Face_Gender", "MS_SG", "MS_FG", "FG_SG"]
    gender_effects = sum(model.params[v] * table[v] for v in gender_vars)
    return table["Answer"] - gender_effects


def remove_outliers(table: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude per-morph-step outliers (> 2 SD from the step mean) for
    visualisation only -- statistical models retain the full dataset.
    """
    cleaned = table.copy()
    for step in table["Morph Step"].unique():
        mask = cleaned["Morph Step"] == step
        step_vals = cleaned.loc[mask, "Partial Residuals"]
        mean, std = step_vals.mean(), step_vals.std()
        cleaned = cleaned[~mask | (np.abs(step_vals - mean) < 2 * std)]
    return cleaned


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _add_jitter(n: int, scale: float, aspect: float) -> tuple:
    """Return (x_jitter, y_jitter) arrays with circular, aspect-corrected noise."""
    angle = np.random.uniform(0, 2 * np.pi, n)
    radius = np.random.uniform(0, scale, n)
    return radius * np.cos(angle), radius * np.sin(angle) / (1 / aspect)


def create_coefficient_plot(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    file_name: str,
) -> None:
    """
    Horizontal bar chart of OLS beta estimates with 95% CIs.
    Bar colours reflect the sign and magnitude of each beta using a
    academic-grade diverging colormap (coolwarm). Asterisks mark
    predictors significant at alpha = 0.05.

    Bug fixed: asterisks for negative estimates are placed at the lower
    CI bound (left tip of bar) rather than near ci_high.
    """
    predictor_labels = {
        "Morph Step":     "Morph Step (MS)",
        "Face_Gender":    "Face Gender (FG)",
        "Subject_Gender": "Participant Gender (PG)",
        "MS_FG":          "MS \u00d7 FG",
        "MS_SG":          "MS \u00d7 PG",
        "FG_SG":          "FG \u00d7 PG",
    }

    cis = model.conf_int()
    cis.columns = ["ci_low", "ci_high"]

    coef_df = (
        pd.DataFrame({"estimate": model.params, "pvalue": model.pvalues})
        .join(cis)
        .drop(index="const")
    )
    coef_df.index = [predictor_labels.get(i, i) for i in coef_df.index]

    abs_max = np.abs(coef_df["estimate"].values).max()
    norm = plt.Normalize(-abs_max, abs_max)
    bar_colors = plt.cm.coolwarm(norm(coef_df["estimate"].values))

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(coef_df))

    err_low = coef_df["estimate"].values - coef_df["ci_low"].values
    err_high = coef_df["ci_high"].values - coef_df["estimate"].values

    ax.barh(
        y_pos,
        coef_df["estimate"],
        xerr=np.array([err_low, err_high]),
        color=bar_colors,
        height=0.6,
        capsize=3,
        error_kw={"elinewidth": 1.2, "ecolor": "#333333", "capthick": 1.2},
        zorder=3,
    )

    # Place asterisks at the CI tip furthest from zero
    pad = 0.015
    for i, row in enumerate(coef_df.itertuples()):
        if row.pvalue < 0.05:
            if row.estimate > 0:
                ax.text(
                    row.ci_high + pad, i - 0.12, "*",
                    ha="left", va="center", fontsize=14, color="#111111",
                )
            else:
                ax.text(
                    row.ci_low - pad, i - 0.12, "*",
                    ha="right", va="center", fontsize=14, color="#111111",
                )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_df.index, fontsize=9)
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xlabel("\u03b2 Estimate (95% CI)", fontsize=10)
    ax.set_title("Regression Coefficients", pad=12)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, color="#cccccc", zorder=0)

    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{file_name} Coefficients.png"))
    plt.close()


def create_validity_plot(
    table: pd.DataFrame,
    file_name: str,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit the full OLS model (Equation 1), compute partial residuals, and produce
    the main validity figure (violin + scatter + regression line) together with
    the coefficient plot.

    Returns the fitted OLS model.

    Bugs fixed:
      - X-tick labels correctly display morph steps 1-9 instead of 0-8.
      - Duplicate aspect-ratio computation removed.
      - Unused `variable` parameter removed from partial residual helper.
    """
    # ------------------------------------------------------------------
    # 1. Encode gender and build interaction terms
    # ------------------------------------------------------------------
    table = table.copy()
    # NimStim convention: persons 1-11 female, 12-22 male
    table["Face_Gender"] = (table["Face Person"] >= 12).astype(int)
    gender_df = pd.read_csv(GENDER_FILE)
    table = table.merge(gender_df, on="ID", how="inner")
    table["Subject_Gender"] = (table["Gender"] == "Male").astype(int)
    table = create_interaction_terms(table)

    # ------------------------------------------------------------------
    # 2. Fit OLS model
    # ------------------------------------------------------------------
    predictors = [
        "Morph Step", "Face_Gender", "Subject_Gender",
        "MS_FG", "MS_SG", "FG_SG",
    ]
    X = sm.add_constant(table[predictors])
    model = sm.OLS(table["Answer"], X).fit()
    partial_resid = calculate_partial_residuals(table, model)

    # ------------------------------------------------------------------
    # 3. Set up figure and compute jitter
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    aspect = bbox.width / bbox.height

    x_jitter, y_jitter = _add_jitter(len(table), scale=0.12, aspect=aspect)

    table["Morph_Step_Jittered"] = table["Morph Step"] + x_jitter - 1
    table["Answer_Resid_Jittered"] = partial_resid + y_jitter

    # ------------------------------------------------------------------
    # 4. Scatter -- raw residuals
    # ------------------------------------------------------------------
    ax.scatter(
        table["Morph_Step_Jittered"],
        table["Answer_Resid_Jittered"],
        s=2,
        color="#888888",
        alpha=0.40,
        marker=".",
        zorder=1,
        rasterized=True
    )

    # ------------------------------------------------------------------
    # 5. Violin plots (outlier-trimmed for visualisation only)
    # ------------------------------------------------------------------
    violin_data = pd.DataFrame(
        {"Morph Step": table["Morph Step"], "Partial Residuals": partial_resid}
    )
    violin_clean = remove_outliers(violin_data)

    sns.violinplot(
        data=violin_clean,
        x="Morph Step",
        y="Partial Residuals",
        palette=MORPH_PALETTE,
        inner=None,
        cut=2,
        width=0.72,
        linewidth=0.8,
        alpha=0.5,
        ax=ax,
        zorder=2,
    )

    # ------------------------------------------------------------------
    # 6. Per-step mean markers
    # ------------------------------------------------------------------
    means = violin_data.groupby("Morph Step")["Partial Residuals"].mean()
    ax.scatter(
        means.index - 1,   # shift to match 0-indexed violin positions
        means.values,
        color="#111111",
        s=70,
        zorder=5,
        label="Step mean",
    )

    # ------------------------------------------------------------------
    # 7. Regression line
    # ------------------------------------------------------------------
    slope = model.params["Morph Step"]
    intercept = partial_resid.mean() - slope * (table["Morph Step"].mean() - 1)
    x_reg = np.array([0.5, 9.5]) - 1
    ax.plot(
        x_reg,
        slope * x_reg + intercept,
        color="#111111",
        linewidth=1.8,
        linestyle="--",
        zorder=4,
        label=f"Model R2 = {model.rsquared:.3f}",
    )

    # ------------------------------------------------------------------
    # 8. Axes, labels, and formatting
    # ------------------------------------------------------------------
    ax.set_xticks(range(9))
    ax.set_xticklabels(range(1, 10))
    ax.set_yticks(range(1, 10))
    ax.set_xlim(-1, 9)
    ax.set_ylim(0, 10)

    ax.set_title("Emotional Ratings Across Morphing Steps", pad=10)
    ax.set_xlabel("Morphing Step  (1 = Angry \u2192 9 = Happy)")
    ax.set_ylabel("Participants\u2019 Rating (Partial Residual)")

    ax.legend(
        loc="upper left",
        fontsize=10,
        frameon=True,
        framealpha=1.0,
        edgecolor="#cccccc",
        borderpad=0.6,
        borderaxespad=1.8,
        markerscale=1,
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)

    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(
        RESULTS_DIR, f"{file_name} Regression.png"), dpi=600)
    plt.close()

    # ------------------------------------------------------------------
    # 9. Coefficient plot and model summary
    # ------------------------------------------------------------------
    create_coefficient_plot(model, file_name)

    with open(os.path.join(RESULTS_DIR, f"{file_name}_model_stats.txt"), "w") as fh:
        fh.write(model.summary().as_text())

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_files = [f for f in os.listdir(FILES_ADDRESS) if f.endswith(".csv")]
    if not csv_files:
        print("Error: No CSV files found in the data directory.")
        return

    frames = []
    for file_name in csv_files:
        table = load_and_preprocess_data(file_name)
        if not table.empty:
            frames.append(table)
            print(f"Loaded:   {file_name}")
        else:
            print(f"Warning:  Skipping {file_name} -- no valid data.")

    if not frames:
        print("Error: No valid data to analyse.")
        return

    table_all = pd.concat(frames, axis=0, ignore_index=True)
    create_validity_plot(table_all, "Validity - Subject Average")
    print(f"\nAnalysis complete. Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
