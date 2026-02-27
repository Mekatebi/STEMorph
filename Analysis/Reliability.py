"""
Reliability Analysis for STEMorph

Author: Mohammad Ebrahim Katebi (mekatebi.2000@gmail.com)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model

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
RELIABILITY_FILES_ADDRESS = "../Data_Reliability/"
VALIDITY_FILES_ADDRESS = "../Data_Validity/"
RESULTS_DIR = "../Results_Reliability/"

MORPH_PALETTE = sns.diverging_palette(10, 220, s=70, l=55, n=9, as_cmap=False)

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------


def load_and_preprocess_data(file_name: str, address: str) -> pd.DataFrame:
    """
    Load a participant CSV, filter to valid trials (State == 1), and drop
    bookkeeping columns. Missing columns are skipped gracefully.
    """
    data = pd.read_csv(os.path.join(address, file_name), header=0)
    data = data[data["State"] == 1]

    drop_cols = ["State", "ITI", "Trial_Onset",
                 "Stim_Onset", "Stim_Offset", "RT"]
    data = data.drop(columns=[c for c in drop_cols if c in data.columns])

    return data


# ---------------------------------------------------------------------------
# Outlier removal
# ---------------------------------------------------------------------------

def remove_outliers(table: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude per-validity-rating outliers (> 2 SD from the group mean) for
    visualisation only -- regression uses the full dataset.
    """
    cleaned = table.copy()
    for val_rating in cleaned["Answer_Validity"].unique():
        mask = cleaned["Answer_Validity"] == val_rating
        group = cleaned.loc[mask, "Answer"]
        mean, std = group.mean(), group.std()
        cleaned = cleaned[~mask | (np.abs(group - mean) < 2 * std)]
    return cleaned


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def perform_linear_regression(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Fit a simple OLS regression (retest ~ validity rating).

    Returns (model, intercept, slope, r_squared).
    """
    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    r_squared = round(regr.score(x, y), 3)
    intercept = round(float(regr.intercept_), 3)
    slope = round(float(regr.coef_[0]), 3)

    return regr, intercept, slope, r_squared


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _add_jitter(n: int, scale: float, aspect: float) -> tuple:
    """Return (x_jitter, y_jitter) arrays with circular, aspect-corrected noise."""
    angle = np.random.uniform(0, 2 * np.pi, n)
    radius = np.random.uniform(0, scale, n)
    return radius * np.cos(angle), radius * np.sin(angle) / (1 / aspect)


def create_reliability_plot(table: pd.DataFrame, file_name: str) -> None:
    """
    Produce the reliability figure: violin distributions of retest answers
    at each validity rating level, overlaid with jittered raw points, per-
    level means, and the OLS regression line.
    """
    x = table["Answer_Validity"].values.reshape(-1, 1)
    y = table["Answer"].values

    regr, intercept, slope, r_squared = perform_linear_regression(x, y)

    # ------------------------------------------------------------------
    # 1. Set up figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    aspect = bbox.width / bbox.height

    # ------------------------------------------------------------------
    # 2. Violin plots (outlier-trimmed for visualisation only)
    # ------------------------------------------------------------------
    table_clean = remove_outliers(table)

    sns.violinplot(
        data=table_clean,
        x="Answer_Validity",
        y="Answer",
        hue="Answer_Validity",
        legend=False,
        inner=None,
        cut=2,
        native_scale=True,
        palette=MORPH_PALETTE,
        linewidth=0.8,
        width=0.6,
        alpha=0.6,
        ax=ax,
        zorder=2,
    )

    # ------------------------------------------------------------------
    # 3. Scatter -- raw jittered points
    # ------------------------------------------------------------------
    x_jitter, y_jitter = _add_jitter(len(table), scale=0.1, aspect=aspect)
    table = table.copy()
    table["Answer_Validity_Jittered"] = table["Answer_Validity"] + x_jitter
    table["Answer_Jittered"] = table["Answer"] + y_jitter

    ax.scatter(
        table["Answer_Validity_Jittered"],
        table["Answer_Jittered"],
        s=2,
        color="#888888",
        alpha=0.40,
        marker=".",
        zorder=1,
        rasterized=True,
    )

    # ------------------------------------------------------------------
    # 4. Per-level mean markers
    # ------------------------------------------------------------------
    means = table.groupby("Answer_Validity")["Answer"].mean()
    ax.scatter(
        means.index,
        means.values,
        color="#111111",
        s=60,
        zorder=5,
        label="Step mean",
    )

    # ------------------------------------------------------------------
    # 5. Regression line
    # ------------------------------------------------------------------
    x_line = np.array([0.5, 9.5])
    ax.plot(
        x_line,
        regr.predict(x_line.reshape(-1, 1)),
        color="#111111",
        linewidth=1.8,
        linestyle="--",
        zorder=4,
        label=f"R2 = {r_squared:.3f}",
    )

    # ------------------------------------------------------------------
    # 6. Axes, labels, and formatting
    # ------------------------------------------------------------------
    ax.set_xticks(np.arange(1, 10, 1))
    ax.set_yticks(np.arange(1, 10, 1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.set_title("Reliability of Participants\u2019 Ratings", pad=10)
    ax.set_xlabel("Validity Rating")
    ax.set_ylabel("Matched Retest Rating")

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    reliability_files = os.listdir(RELIABILITY_FILES_ADDRESS)
    validity_files = os.listdir(VALIDITY_FILES_ADDRESS)
    frames = []

    for reliability_file in reliability_files:
        participant_id = reliability_file[8:12]
        reliability_table = load_and_preprocess_data(
            reliability_file, RELIABILITY_FILES_ADDRESS
        )

        validity_file = next(
            (f for f in validity_files if participant_id in f), None
        )

        if validity_file:
            print(f"Matched:  {reliability_file}  <->  {validity_file}")
            validity_table = load_and_preprocess_data(
                validity_file, VALIDITY_FILES_ADDRESS
            )
            merged = pd.merge(
                reliability_table,
                validity_table,
                on=["Position ID", "Face Person", "Morph Step"],
                suffixes=["", "_Validity"],
            )
            frames.append(merged)
        else:
            print(
                f"Warning:  No matching validity file for participant {participant_id}")

    if not frames:
        print("Error: No matched data to analyse.")
        return

    table_all = pd.concat(frames, axis=0, ignore_index=True)
    create_reliability_plot(table_all, "Subject Average")
    print(f"\nAnalysis complete. Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
