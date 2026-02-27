"""
Face Identity Analysis for STEMorph

Author: Mohammad Ebrahim Katebi (mekatebi.2000@gmail.com)
"""

import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

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
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)

FILES_ADDRESS = "../Data_Validity/"
RESULTS_DIR = "../Results_Identity/"

COLOR_FEMALE = "#479C9E"
COLOR_MALE = "#495084"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_prepare_data() -> pd.DataFrame:
    """Load all validity CSVs, filter to valid trials, and assign Face_Gender."""
    frames = []
    for file_name in os.listdir(FILES_ADDRESS):
        if not file_name.endswith(".csv"):
            continue
        data = pd.read_csv(os.path.join(FILES_ADDRESS, file_name), header=0)
        if "State" in data.columns:
            data = data[data["State"] == 1]
        drop_cols = ["State", "Position ID", "ITI",
                     "Trial_Onset", "Stim_Onset", "Stim_Offset"]
        data = data.drop(columns=[c for c in drop_cols if c in data.columns])
        # NimStim convention: persons 1-11 female, 12-22 male
        data["Face_Gender"] = np.where(
            data["Face Person"] < 12, "Female", "Male")
        frames.append(data)

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {FILES_ADDRESS}")
    return pd.concat(frames, axis=0, ignore_index=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calculate_face_identity_means(data: pd.DataFrame) -> pd.DataFrame:
    """Return mean rating, SD, and observation count per face identity."""
    return (
        data.groupby(["Face Person", "Face_Gender"])["Answer"]
        .agg(Mean_Rating="mean", SD_Rating="std", N_Observations="count")
        .reset_index()
        .round(3)
    )


def compute_gender_effect(face_stats: pd.DataFrame) -> dict:
    """
    Test whether female faces are consistently rated higher than male faces.
    Cohen's d uses sample variance (ddof=1).
    Mann-Whitney U is one-tailed (female > male).
    """
    female_means = face_stats[face_stats["Face_Gender"]
                              == "Female"]["Mean_Rating"].values
    male_means = face_stats[face_stats["Face_Gender"]
                            == "Male"]["Mean_Rating"].values

    t_stat, t_pval = stats.ttest_ind(female_means, male_means)
    u_stat, u_pval = stats.mannwhitneyu(
        female_means, male_means, alternative="greater")

    pooled_std = np.sqrt(
        (np.var(female_means, ddof=1) + np.var(male_means, ddof=1)) / 2
    )
    cohens_d = (np.mean(female_means) - np.mean(male_means)) / pooled_std

    return {
        "female_mean_of_means": float(np.mean(female_means)),
        "male_mean_of_means":   float(np.mean(male_means)),
        "female_sd_of_means":   float(np.std(female_means, ddof=1)),
        "male_sd_of_means":     float(np.std(male_means,   ddof=1)),
        "t_statistic":          float(t_stat),
        "t_pvalue":             float(t_pval),
        "mannwhitney_u":        float(u_stat),
        "mannwhitney_pvalue":   float(u_pval),
        "cohens_d":             float(cohens_d),
    }


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def plot_face_by_morph_heatmap(data: pd.DataFrame, face_stats: pd.DataFrame) -> None:
    """
    Heatmap of mean rating for every face x morph-step combination.
    """
    heatmap_data = (
        data.groupby(["Face Person", "Morph Step"])["Answer"]
        .mean()
        .unstack()
    )

    face_order = (
        face_stats
        .sort_values(["Face Person"], ascending=[True])
        ["Face Person"]
        .values
    )
    heatmap_data = heatmap_data.loc[face_order]

    n_rows = len(heatmap_data)
    n_cols = heatmap_data.shape[1]

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(heatmap_data.values, cmap="RdYlGn", aspect="auto",
                   vmin=1, vmax=9, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean Rating", rotation=270, labelpad=15, fontsize=11)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(np.arange(1, n_cols + 1))
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(
        [f"{int(fp)} ({'F' if fp < 12 else 'M'})" for fp in heatmap_data.index],
        fontsize=9,
    )

    # Separator between female and male groups
    female_count = int((heatmap_data.index < 12).sum())
    ax.axhline(y=female_count - 0.5, color="black",
               linewidth=2.5, linestyle="-")

    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j, i, f"{heatmap_data.values[i, j]:.2f}",
                    ha="center", va="center", color="black", fontsize=7)

    ax.set_xlabel("Morphing Step  (1 = Angry \u2192 9 = Happy)")
    ax.set_ylabel("Face Identity  (F = Female, M = Male)")
    ax.set_title("Mean Ratings by Face Identity and Morphing Step", pad=15)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "Face_by_Morph_Heatmap.png"))
    plt.close()
    print("  Saved: Face_by_Morph_Heatmap.png")


# ---------------------------------------------------------------------------
# Identity means bar plot
# ---------------------------------------------------------------------------

def plot_identity_means(face_stats: pd.DataFrame, results: dict) -> None:
    """
    Compact bar chart of per-identity mean ratings sorted ascending,
    colour-coded by gender, with dashed grand-mean reference lines.
    """
    female_stats = face_stats[face_stats["Face_Gender"]
                              == "Female"].sort_values("Mean_Rating")
    male_stats = face_stats[face_stats["Face_Gender"]
                            == "Male"].sort_values("Mean_Rating")

    all_means = np.concatenate([female_stats["Mean_Rating"].values,
                                male_stats["Mean_Rating"].values])
    all_ids = np.concatenate([female_stats["Face Person"].values,
                              male_stats["Face Person"].values])
    all_genders = (["Female"] * len(female_stats)) + \
        (["Male"] * len(male_stats))

    sort_idx = np.argsort(all_means)
    all_means_sorted = all_means[sort_idx]
    all_ids_sorted = all_ids[sort_idx]
    all_gnd_sorted = [all_genders[i] for i in sort_idx]
    colors = [COLOR_FEMALE if g == "Female" else COLOR_MALE
              for g in all_gnd_sorted]

    female_grand = results["female_mean_of_means"]
    male_grand = results["male_mean_of_means"]

    bar_width = 0.25
    x_pos = np.arange(len(all_means_sorted)) * 0.45

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x_pos, all_means_sorted, width=bar_width, color=colors,
           edgecolor="black", linewidth=0.6, alpha=0.85)

    ax.axhline(female_grand, color=COLOR_FEMALE,
               linestyle="--", linewidth=1.8, alpha=0.8)
    ax.axhline(male_grand,   color=COLOR_MALE,
               linestyle="--", linewidth=1.8, alpha=0.8)

    labels = [
        f"{int(id_)}\n{'F' if g == 'Female' else 'M'}"
        for id_, g in zip(all_ids_sorted, all_gnd_sorted)
    ]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(x_pos[0] - 0.5, x_pos[-1] + 0.5)
    ax.set_ylim(4.2, 5.3)
    ax.set_xlabel("Face Identity (sorted by mean rating)")
    ax.set_ylabel("Mean Emotional Rating (across all morph steps)")
    ax.set_title("Individual Face Identity Mean Ratings", pad=10)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)

    legend_handles = [
        mpatches.Patch(color=COLOR_FEMALE, label="Female faces"),
        mpatches.Patch(color=COLOR_MALE,   label="Male faces"),
        plt.Line2D([0], [0], color=COLOR_FEMALE, linestyle="--",
                   linewidth=1.8, label=f"Female mean = {female_grand:.2f}"),
        plt.Line2D([0], [0], color=COLOR_MALE,   linestyle="--",
                   linewidth=1.8, label=f"Male mean = {male_grand:.2f}"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9,
              frameon=True, framealpha=1.0, edgecolor="#cccccc")

    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "Identity_Means_Barplot.png"))
    plt.close()
    print("  Saved: Identity_Means_Barplot.png")


# ---------------------------------------------------------------------------
# Gender distribution
# ---------------------------------------------------------------------------

def plot_gender_distribution(face_stats: pd.DataFrame, results: dict) -> None:
    """
    Violin plot with overlaid individual data points comparing female vs. male
    face identity mean ratings. Median shown as a short horizontal line.
    """
    female_means = (face_stats[face_stats["Face_Gender"] == "Female"]
                    .sort_values("Mean_Rating")["Mean_Rating"].values)
    male_means = (face_stats[face_stats["Face_Gender"] == "Male"]
                  .sort_values("Mean_Rating")["Mean_Rating"].values)

    data_b = [female_means, male_means]
    positions = [1, 2]

    fig, ax = plt.subplots(figsize=(5, 6))

    # Violin
    parts = ax.violinplot(data_b, positions=positions, widths=0.5,
                          showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor([COLOR_FEMALE, COLOR_MALE][i])
        pc.set_alpha(0.55)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.2)

    for pos, vals in zip(positions, data_b):
        median = np.median(vals)
        ax.hlines(median, pos - 0.15, pos + 0.15,
                  colors="darkred", linewidths=2.5, zorder=4)

    # Individual data points (jittered)
    for pos, vals, color in zip(positions, data_b, [COLOR_FEMALE, COLOR_MALE]):
        x_jit = np.random.normal(pos, 0.05, size=len(vals))
        ax.scatter(x_jit, vals, alpha=0.85, s=55, edgecolors="black",
                   linewidths=0.8, c=color, zorder=3)

    # Statistical annotation
    t_p = results["t_pvalue"]
    u_p = results["mannwhitney_pvalue"]
    d_val = results["cohens_d"]
    t_str = "< 0.001" if t_p < 0.001 else f"= {t_p:.3f}"
    u_str = "< 0.001" if u_p < 0.001 else f"= {u_p:.3f}"

    y_top = max(np.max(female_means), np.max(male_means))
    y_ann = y_top + 0.1
    ax.plot([1, 2], [y_ann, y_ann], "k-", linewidth=1.2)
    ax.text(
        1.5, y_ann + 0.04,
        f"U-test p {u_str}\nd = {d_val:.2f}",
        ha="center", va="bottom", fontsize=8, fontstyle="italic",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(["Female\nFaces", "Male\nFaces"], fontsize=10)
    ax.set_ylabel("Mean Emotional Rating")
    ax.set_title("Distribution of Face Identity Means", pad=10)
    ax.set_ylim(3.8, y_top + 0.65)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)

    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "Gender_Distribution.png"))
    plt.close()
    print("  Saved: Gender_Distribution.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data...")
    data = load_and_prepare_data()
    print(f"  {len(data)} observations | "
          f"{data['Face Person'].nunique()} unique faces")

    print("Computing face identity statistics...")
    face_stats = calculate_face_identity_means(data)

    print("Computing gender effect statistics...")
    results = compute_gender_effect(face_stats)

    print("Generating figures...")
    plot_face_by_morph_heatmap(data, face_stats)
    plot_identity_means(face_stats, results)
    plot_gender_distribution(face_stats, results)

    print(f"\nDone. All outputs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
