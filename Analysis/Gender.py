"""
Gender Subgroup Validity Analysis for STEMorph

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
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FILES_ADDRESS = "../Data_Validity/"
RESULTS_DIR = "../Results_gender/"
GENDER_FILE = "../Subject_Gender.csv"

# Distinct hue palettes for each subgroup analysis
PALETTE_SUBJECT = {"Female": "#2596be", "Male": "#be4d25"}
PALETTE_FACE = {"Female": "#00cec9", "Male": "#fdcb6e"}


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
# Partial residuals
# ---------------------------------------------------------------------------

def calculate_partial_residuals(
    table: pd.DataFrame,
    model: sm.regression.linear_model.RegressionResultsWrapper,
    group_type: str,
) -> pd.Series:
    """
    Compute partial residuals by removing variance attributable to the
    cross-gender regressors, isolating the morph-step signal within each
    gender subgroup.

    group_type == 'subject'  ->  Equation 2: remove FG, MS_FG, FG_SG
    group_type == 'face'     ->  Equation 3: remove Subject_Gender, MS_SG, FG_SG

    Bug fixed: removed unused `variable` parameter.
    """
    if group_type == "subject":
        unwanted_vars = ["Face_Gender", "MS_FG", "FG_SG"]
    else:
        unwanted_vars = ["Subject_Gender", "MS_SG", "FG_SG"]

    unwanted_effects = sum(model.params[v] * table[v] for v in unwanted_vars)
    return table["Answer"] - unwanted_effects


# ---------------------------------------------------------------------------
# Subgroup R²
# ---------------------------------------------------------------------------

def calculate_partial_r2(
    table: pd.DataFrame, gender_label: str,
) -> float:
    """
    Return the R² between Morph Step and Partial_Residuals for a single
    gender subgroup (identified by Gender_Group label).
    """
    sub = table[table["Gender_Group"] == gender_label]
    return float(np.corrcoef(sub["Morph Step"], sub["Partial_Residuals"])[0, 1] ** 2)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def create_subgroup_plot(
    table: pd.DataFrame,
    model: sm.regression.linear_model.RegressionResultsWrapper,
    group_type: str,
) -> None:
    """
    Produce a split-violin gender subgroup plot with per-group mean markers
    and OLS trendlines (Equations 2 and 3 from the paper).

    Bugs fixed:
      - SyntaxError in ylabel (unmatched single quote around apostrophe).
      - X-tick labels corrected from 0-8 to 1-9.
      - Typo 'Particiapnt' -> 'Participant' in title and filename.
      - dpi=300 raised to 600.
      - plt.figure() replaced with fig, ax = plt.subplots(); ax= passed to seaborn.
      - Dead xlabel 'Morphing Step | Others' replaced with descriptive label.
    """
    if group_type == "subject":
        group_col = "Subject_Gender"
        palette = PALETTE_SUBJECT
        title_prefix = "Participant Gender"
    else:
        group_col = "Face_Gender"
        palette = PALETTE_FACE
        title_prefix = "Face Gender"

    table = table.copy()
    table["Partial_Residuals"] = calculate_partial_residuals(
        table, model, group_type
    )
    table["Gender_Group"] = np.where(table[group_col] == 0, "Female", "Male")

    # ------------------------------------------------------------------
    # 1. Set up figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))

    # ------------------------------------------------------------------
    # 2. Violin plots (outlier-trimmed per group per step for vis. only)
    # ------------------------------------------------------------------
    violin_data = (
        table
        .groupby(["Morph Step", "Gender_Group"], group_keys=False)
        .apply(
            lambda g: g[
                np.abs(g["Partial_Residuals"] - g["Partial_Residuals"].mean())
                < 2 * g["Partial_Residuals"].std()
            ]
        )
    )

    sns.violinplot(
        data=violin_data,
        x="Morph Step",
        y="Partial_Residuals",
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

    # ------------------------------------------------------------------
    # 3. Per-group mean markers and OLS trendlines
    # ------------------------------------------------------------------
    r2_values = {}

    for gender_label, color in palette.items():
        sub = table[table["Gender_Group"] == gender_label]
        x_steps = sub["Morph Step"].values
        y_resid = sub["Partial_Residuals"].values

        slope, intercept = np.polyfit(x_steps, y_resid, 1)
        r2_values[gender_label] = calculate_partial_r2(table, gender_label)

        formula = (
            f"{gender_label}: Answer = {intercept:.2f} + {slope:.2f} \u00d7 Morph Step"
        )
        print(f"\n{title_prefix} | {formula}")

        # Trendline -- evaluated at morph-step values [1, 9], plotted at
        # categorical positions [0, 8] (shifted by -1 to match violin positions)
        x_reg = np.array([1, 9])
        ax.plot(
            x_reg - 1,
            slope * x_reg + intercept,
            color=color,
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            zorder=4,
        )

        means = sub.groupby("Morph Step")["Partial_Residuals"].mean()
        ax.scatter(
            means.index - 1,
            means.values,
            color=color,
            s=80,
            edgecolor="white",
            linewidth=1.5,
            zorder=5,
        )

    # ------------------------------------------------------------------
    # 4. Legend with R² values
    # ------------------------------------------------------------------
    legend_handles = []
    for gender_label, color in palette.items():
        handle = plt.Line2D(
            [], [],
            color=color,
            linewidth=2,
            linestyle="--",
            label=f"{gender_label} R2 = {r2_values[gender_label]:.3f}",
        )
        legend_handles.append(handle)

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=10,
        frameon=True,
        framealpha=1.0,
        edgecolor="#cccccc",
        borderpad=0.8,
        borderaxespad=1.2,
    )

    # ------------------------------------------------------------------
    # 5. Axes, labels, and formatting
    # ------------------------------------------------------------------
    ax.set_xticks(range(9))
    ax.set_xticklabels(range(1, 10))
    ax.set_yticks(range(1, 10))
    ax.set_xlim(-1, 9)
    ax.set_ylim(0, 10)

    ax.set_title(f"{title_prefix} Subgroups", pad=10)
    ax.set_xlabel("Morphing Step  (1 = Angry \u2192 9 = Happy)")
    ax.set_ylabel("Participants\u2019 Rating (Partial Residual)")

    ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)

    sns.despine(ax=ax)
    plt.tight_layout()

    out_name = f"Validity_{title_prefix.replace(' ', '_')}_Subgroups.png"
    plt.savefig(os.path.join(RESULTS_DIR, out_name), dpi=600)
    plt.close()


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

    all_data = pd.concat(frames, axis=0, ignore_index=True)

    # Encode gender variables (NimStim: persons 1-11 female, 12-22 male)
    all_data["Face_Gender"] = (all_data["Face Person"] >= 12).astype(int)
    gender_df = pd.read_csv(GENDER_FILE)
    all_data = all_data.merge(gender_df, on="ID", how="inner")
    all_data["Subject_Gender"] = (all_data["Gender"] == "Male").astype(int)
    all_data["MS_FG"] = all_data["Morph Step"] * all_data["Face_Gender"]
    all_data["MS_SG"] = all_data["Morph Step"] * all_data["Subject_Gender"]
    all_data["FG_SG"] = all_data["Face_Gender"] * all_data["Subject_Gender"]

    predictors = [
        "Morph Step", "Face_Gender", "Subject_Gender",
        "MS_FG", "MS_SG", "FG_SG",
    ]
    X = sm.add_constant(all_data[predictors])
    model = sm.OLS(all_data["Answer"], X).fit()

    create_subgroup_plot(all_data, model, "subject")
    create_subgroup_plot(all_data, model, "face")

    # with open(os.path.join(RESULTS_DIR, "Model_Summary.txt"), "w") as fh:
    #     fh.write(model.summary().as_text())

    print(f"\nAnalysis complete. Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
